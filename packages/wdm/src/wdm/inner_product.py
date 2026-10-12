"""Noise-weighted inner products and Gaussian log-likelihoods in the WDM domain.

For two signals $a$ and $b$ in stationary Gaussian noise with PSD
$S_n(f)$, the standard frequency-domain inner product is

$$\\langle a | b \\rangle = 4\\,\\operatorname{Re}\\int_0^\\infty
\\hat{a}^*(f)\\,\\hat{b}(f) / S_n(f)\\,df.$$

This module provides four flavors of WDM-domain inner products and
matching log-likelihoods.

Inner products
~~~~~~~~~~~~~~
Diagonal flavors (cost $O(N)$, all share signature
``(w_a, w_b, var, Nf)``):

* :func:`whittle` — simple sum with band-center variance.
* :func:`band_averaged` — same but with the exact band-averaged variance.
* :func:`evolutionary_whittle` — pixel-dependent variance for
  non-stationary noise.

Exact flavor (cost $O(N \\log N_t)$):

* :func:`gaussian` — inverse-transform back to the frequency domain
  and apply the per-bin PSD weighting.  Equals the standard
  frequency-domain inner product to machine precision.

Log-likelihoods
~~~~~~~~~~~~~~~
Each inner product has a matching log-likelihood that includes the
$-\\tfrac{1}{2}\\ln\\det(2\\pi\\mathbf{C})$ normalization term.  Set
``include_logdet=False`` to drop the normalization (appropriate when
the PSD is fixed during parameter estimation):

* :func:`loglike_whittle`, :func:`loglike_band_averaged`,
  :func:`loglike_evolutionary_whittle` — diagonal-covariance forms.
* :func:`loglike_gaussian` — full-covariance form.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .grid import n_samples, sanitize_psd, validate_finite
from .transform import inverse, inverse_many

type Array = NDArray[np.floating]
type CArray = NDArray[np.complexfloating]
type PsdFunc = Callable[[Array], Array]

__all__ = [
    "band_averaged",
    "evolutionary_whittle",
    "gaussian",
    "gaussian_from_func",
    "loglike_band_averaged",
    "loglike_evolutionary_whittle",
    "loglike_gaussian",
    "loglike_gaussian_many",
    "loglike_whittle",
    "loglike_whittle_many",
    "whittle",
    "whittle_many",
]


def _as_grids(*arrays: Array) -> tuple[Array, ...]:
    """Coerce inputs to float arrays so lists raise our errors, not numpy's.

    Existing float arrays pass through **untouched**, including ``float32``.
    This used to be ``np.asarray(a, dtype=float)``, which silently materialised
    a ``float64`` copy of every grid: measured **8.46 MB of peak allocation for
    a 1.18 MB float32 stack** at $B=2048$, a 7x blowup that undid the whole
    point of the single-precision path at the step a sampler calls most often.

    Precision of the *result* is handled where it belongs, in
    :func:`_reduce_diagonal`, which accumulates in double regardless of input
    dtype -- so single precision saves the memory without the summation losing
    accuracy over $N$ terms.
    """
    return tuple(
        a if isinstance(a, np.ndarray) and a.dtype.kind == "f" else np.asarray(a, float)
        for a in arrays
    )


def _check_psd_grid(Sn_array: Array, N: int, context: str) -> Array:
    """Sanitize ``Sn_array`` and require it to live on the rfft grid.

    Without the length check a broadcastable PSD (a scalar, or length 1)
    silently produces the same number as a correct length-``N//2+1`` array.
    """
    arr = sanitize_psd(Sn_array, context)
    if arr.shape != (N // 2 + 1,):
        raise ValueError(
            f"{context}: Sn_array must have shape ({N // 2 + 1},) -- the "
            f"np.fft.rfftfreq(N, dt) grid for N={N}; got {arr.shape}"
        )
    # The PSD is divided by, and its log is taken.  A zero bin would return
    # nan (or +inf) rather than raising, which is the failure mode the diagonal
    # path already rejects via its variance check -- keep the two consistent.
    interior = arr[1 : N // 2]
    if np.any(interior <= 0.0):
        raise ValueError(
            f"{context}: Sn_array must be strictly positive on the interior "
            f"bins j = 1..N/2-1; got min={float(np.min(interior))!r}. A PSD "
            "with a zero bin gives that bin infinite statistical weight."
        )
    return arr


def _check_grids(w_a: Array, w_b: Array, var: Array, Nf: int, *, batched: bool) -> None:
    """Validate that the three grids agree and match ``Nf``.

    Without this, a mismatched ``Nf`` silently sums the wrong number of bands
    and a transposed or wrongly-shaped grid broadcasts into a plausible-looking
    number instead of raising.

    With ``batched=True`` the coefficient grids carry a leading axis and
    ``var`` may be either one shared ``(Nf+1, Nt)`` grid -- the common case,
    one noise model against many templates -- or a full stack.
    """
    want_ndim = 3 if batched else 2
    shape_hint = "(B, Nf+1, Nt)" if batched else "(Nf+1, Nt)"
    if w_a.shape != w_b.shape:
        raise ValueError(
            f"w_a and w_b must have the same shape; got {w_a.shape} and {w_b.shape}"
        )
    if w_a.ndim != want_ndim or w_a.shape[-2] != Nf + 1:
        raise ValueError(
            f"coefficient grids must have shape {shape_hint} with Nf+1 = "
            f"{Nf + 1} bands for Nf={Nf}; got {w_a.shape}"
        )
    if var.shape != w_a.shape and var.shape != w_a.shape[-2:]:
        raise ValueError(
            f"var must have the same shape as the coefficient grids "
            f"{w_a.shape}, or be one shared grid {w_a.shape[-2:]}; "
            f"got {var.shape}"
        )
    # var is divided by, so a non-positive or non-finite entry turns a
    # likelihood into a silent negative number or a NaN rather than an error.
    interior = var[..., 1:Nf, :]
    if not np.all(np.isfinite(interior)) or np.any(interior <= 0.0):
        raise ValueError(
            "var must be finite and strictly positive on the interior bands "
            "m = 1..Nf-1; got "
            f"min={float(np.min(interior))!r}, "
            f"{int(np.count_nonzero(~np.isfinite(interior)))} non-finite"
        )
    # The coefficient grids get the same treatment as var. A NaN pixel in
    # either one makes the whole sum NaN, and a NaN log-likelihood read by a
    # sampler is worse than an exception: most samplers reject the point
    # rather than crashing, so the run continues and quietly explores nothing.
    validate_finite(w_a, "w_a")
    validate_finite(w_b, "w_b")


def _residual(w_d: Array, w_h: Array) -> Array:
    """``w_d - w_h``, with each operand checked for finiteness *first*.

    Order matters for the error message, not just for hygiene.  Subtracting
    first turns ``inf - inf`` into ``nan``, so the downstream check would
    report a NaN in the residual and leave the caller hunting for it in a
    grid that never contained one.  Checking the operands names the grid the
    caller actually passed.
    """
    validate_finite(w_d, "w_d")
    validate_finite(w_h, "w_h")
    return w_d - w_h


def _diagonal_inner(w_a: Array, w_b: Array, var: Array, Nf: int) -> float:
    """Common implementation of the diagonal inner products.

    Sums $w_a w_b / \\sigma^2$ over the active interior pixels.
    Edge bands are excluded by default; callers that want them
    included should sum manually with the appropriate variance.
    """
    w_a, w_b, var = _as_grids(w_a, w_b, var)
    _check_grids(w_a, w_b, var, Nf, batched=False)
    return float(_reduce_diagonal(w_a, w_b, var, Nf))


def _reduce_diagonal(w_a: Array, w_b: Array, var: Array, Nf: int) -> Array:
    """$\\sum w_a w_b/\\sigma^2$ over the interior pixels of the last two axes.

    Leading axes are preserved, so this serves both the scalar and the batched
    inner products from one expression.  Interior bands $m = 1 \\ldots N_f-1$
    only; the active mask there is all-True, so no masking is needed.
    """
    slab = (..., slice(1, Nf), slice(None))
    # Accumulate in double whatever the inputs are. A float32 stack is worth
    # having for its memory, but summing N terms in single precision is not:
    # the products are formed in the input dtype (no copy) and the reduction
    # promotes.
    return np.sum(w_a[slab] * w_b[slab] / var[slab], axis=(-2, -1), dtype=np.float64)


def _diagonal_inner_many(w_a: Array, w_b: Array, var: Array, Nf: int) -> Array:
    """Batched :func:`_diagonal_inner`: one value per leading-axis entry."""
    w_a, w_b, var = _as_grids(w_a, w_b, var)
    _check_grids(w_a, w_b, var, Nf, batched=True)
    return _reduce_diagonal(w_a, w_b, var, Nf)


def whittle(w_a: Array, w_b: Array, var: Array, Nf: int) -> float:
    """Whittle inner product: $\\sum_{m,n} w_a w_b / \\sigma_{mn}^2$.

    With ``var`` produced by :func:`wdm.whittle_variance`, this equals
    $2\\sum w_a w_b / S_n(f_m)$.  Cost $O(N)$.

    Sums interior bands ($m = 1, \\ldots, N_f-1$) only, matching the
    convention of dropping edge bands from the likelihood.

    Parameters
    ----------
    w_a, w_b : ndarray, shape (Nf+1, Nt)
        WDM coefficients of two signals.
    var : ndarray, shape (Nf+1, Nt)
        Per-pixel variance grid (e.g. from :func:`wdm.whittle_variance`).
    Nf : int

    Returns
    -------
    float
        Inner product.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward_time(x, Nf, Nt, dt)
    >>> var = wdm.whittle_variance(Nf, Nt, dt, lambda f: 1e-3 * np.ones_like(f))
    >>> bool(wdm.inner_product.whittle(w, w, var, Nf) > 0)
    True
    """
    return _diagonal_inner(w_a, w_b, var, Nf)


def band_averaged(w_a: Array, w_b: Array, var: Array, Nf: int) -> float:
    """Same as :func:`whittle` but expects band-averaged variance.

    Use ``var = wdm.band_averaged_variance(...)`` to get the exact
    filter-weighted mean PSD across each band.

    Parameters
    ----------
    w_a, w_b : ndarray, shape (Nf+1, Nt)
    var : ndarray, shape (Nf+1, Nt)
    Nf : int

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward_time(x, Nf, Nt, dt)
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> var = wdm.band_averaged_variance(Nf, Nt, dt, flat)
    >>> bool(wdm.inner_product.band_averaged(w, w, var, Nf) > 0)
    True
    """
    return _diagonal_inner(w_a, w_b, var, Nf)


def evolutionary_whittle(
    w_a: Array,
    w_b: Array,
    var_tf: Array,
    Nf: int,
) -> float:
    """Diagonal inner product with pixel-dependent variance (non-stationary).

    Use ``var_tf = wdm.evolutionary_variance(...)`` for a slowly-varying
    PSD $S_n(f, t)$.

    Parameters
    ----------
    w_a, w_b : ndarray, shape (Nf+1, Nt)
    var_tf : ndarray, shape (Nf+1, Nt)
        Per-pixel variance, varying in both $m$ and $n$.
    Nf : int

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward_time(x, Nf, Nt, dt)
    >>> Sn = lambda f, t: 1e-3 * np.ones((f.size, t.size))
    >>> var = wdm.evolutionary_variance(Nf, Nt, dt, Sn)
    >>> bool(wdm.inner_product.evolutionary_whittle(w, w, var, Nf) > 0)
    True
    """
    return _diagonal_inner(w_a, w_b, var_tf, Nf)


def _frequency_inner(
    a_rfft: CArray,
    b_rfft: CArray,
    Sn_array: Array,
    dt: float,
    N: int,
) -> float:
    """Standard frequency-domain noise-weighted inner product.

    $$\\langle a|b\\rangle
    = \\frac{4\\,\\delta t}{N}\\,
    \\operatorname{Re}\\!\\sum_{j=1}^{N/2-1}
    \\frac{A_j^*\\,B_j}{S_n(f_j)}.$$

    Skips the DC and Nyquist edge bins (real-valued one-sided edge
    terms with halved weight), matching the standard GW convention.

    Parameters
    ----------
    a_rfft, b_rfft : ndarray, complex, shape (N//2+1,)
    Sn_array : ndarray, shape (N//2+1,)
    dt : float
    N : int

    Returns
    -------
    float
    """
    # A zero-power bin (an inf-at-DC LISA curve sanitizes to one) carries no
    # weight rather than producing inf/nan -- same rule as _frequency_quadratic.
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.conj(a_rfft) * b_rfft / Sn_array
    integrand = np.where(Sn_array > 0.0, integrand, 0.0)
    return float(4.0 * dt / N * np.real(np.sum(integrand[1 : N // 2])))


def _gaussian_logdet(Sn_array: Array, N: int) -> float:
    """$\\ln\\det(2\\pi C)$ for the exact Gaussian form.

    Summed over the **full two-sided** grid, which is the fix behind item 1 of
    the change history: summing the one-sided grid instead scaled the term with
    $N/2-1$ pixels rather than $N$, and maximising the likelihood over a PSD
    amplitude returned twice the truth. Zero-PSD bins are skipped rather than
    contributing $-\\infty$.

    Shared by :func:`loglike_gaussian` and :func:`loglike_gaussian_many` so the
    two cannot disagree about it.
    """
    Sn_full = np.empty(N, dtype=float)
    n_rfft = N // 2 + 1
    Sn_full[:n_rfft] = Sn_array
    Sn_full[n_rfft:] = Sn_array[1 : N // 2][::-1]
    contributing = Sn_full > 0.0
    return float(np.sum(np.log(np.pi * Sn_full[contributing])))


def _frequency_quadratic(
    r_rfft: CArray,
    Sn_array: Array,
    dt: float,
    N: int,
) -> float:
    """Full quadratic form $r^T C^{-1} r$ in the frequency domain.

    Identical to :func:`_frequency_inner` except that the DC and Nyquist bins
    are **included at half weight**.  Those bins are real-valued, so they carry
    half the degrees of freedom of an interior bin; including them this way
    reproduces $r^T C_a^{-1} r$ against a dense
    :func:`wdm.matrix.build_C_wdm_active` solve to machine precision, whereas
    dropping them (the standard GW inner-product convention used by
    :func:`gaussian`) understates the quadratic form by $O(2/N)$ -- measured
    7.8% at $N = 32$ and 0.5% at $N = 512$.

    :func:`loglike_gaussian` needs this form rather than the GW one: its
    log-determinant covers all $N$ active pixels, so a quadratic form over
    $N - 2$ of them would not be the log of any normalized density.
    """
    # A zero-power bin has infinite variance: it carries no weight, rather than
    # producing inf/nan.  Only DC and Nyquist can be zero here; the interior is
    # required strictly positive by _check_psd_grid.
    return float(_frequency_quadratic_axis(r_rfft, Sn_array, dt, N))


def _frequency_quadratic_axis(
    r_rfft: CArray,
    Sn_array: Array,
    dt: float,
    N: int,
) -> Array:
    """$r^T C^{-1} r$ reduced over the last axis; leading axes preserved.

    DC and Nyquist carry half weight -- they have no negative-frequency
    partner, so counting them fully double-counts their contribution. Serves
    both the scalar and batched Gaussian forms from one expression.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.real(np.conj(r_rfft) * r_rfft / Sn_array)
    integrand = np.where(Sn_array > 0.0, integrand, 0.0)
    interior = np.sum(integrand[..., 1 : N // 2], axis=-1)
    edges = 0.5 * (integrand[..., 0] + integrand[..., N // 2])
    return 4.0 * dt / N * (interior + edges)


def gaussian(
    w_a: Array,
    w_b: Array,
    Nf: int,
    Nt: int,
    dt: float,
    Sn_array: Array,
) -> float:
    """Exact inner product via inverse-WDM and frequency-domain weighting.

    Inverse-transforms $w_a$ and $w_b$ to the frequency domain and
    computes the standard noise-weighted inner product there.  Equals
    the frequency-domain inner product to machine precision; exact
    regardless of how steeply $S_n(f)$ varies.

    Cost: $O(N \\log N_t)$ from the two inverse FFTs.

    Parameters
    ----------
    w_a, w_b : ndarray, shape (Nf+1, Nt)
    Nf, Nt : int
    dt : float
    Sn_array : ndarray, shape (N//2+1,)
        Precomputed $S_n$ on the rfft frequency grid
        (``np.fft.rfftfreq(N, d=dt)``).

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> N = Nf * Nt
    >>> x = np.random.default_rng(0).standard_normal(N)
    >>> w = wdm.forward_time(x, Nf, Nt, dt)
    >>> Sn = 1e-3 * np.ones(N // 2 + 1)   # on the rfftfreq grid
    >>> bool(wdm.inner_product.gaussian(w, w, Nf, Nt, dt, Sn) > 0)
    True
    """
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "gaussian")
    a_rfft = inverse(w_a, Nf, Nt, dt)
    b_rfft = inverse(w_b, Nf, Nt, dt)
    return _frequency_inner(a_rfft, b_rfft, Sn_array, dt, N)


def gaussian_from_func(
    w_a: Array,
    w_b: Array,
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
) -> float:
    """Convenience wrapper around :func:`gaussian` with a callable PSD.

    Builds ``Sn_array`` on the rfft frequency grid internally.

    Parameters
    ----------
    w_a, w_b : ndarray, shape (Nf+1, Nt)
    Nf, Nt : int
    dt : float
    Sn_func : callable
        ``f -> S_n(f)``.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward_time(x, Nf, Nt, dt)
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> a = wdm.inner_product.gaussian_from_func(w, w, Nf, Nt, dt, flat)
    >>> Sn = flat(np.fft.rfftfreq(Nf * Nt, d=dt))
    >>> b = wdm.inner_product.gaussian(w, w, Nf, Nt, dt, Sn)
    >>> bool(np.isclose(a, b))
    True
    """
    N = n_samples(Nf, Nt)
    freqs = np.fft.rfftfreq(N, d=dt)
    Sn_array = np.asarray(Sn_func(freqs), dtype=float)
    return gaussian(w_a, w_b, Nf, Nt, dt, Sn_array)


def _diagonal_logdet(var: Array, Nf: int) -> float:
    """Sum of $\\ln(2\\pi\\sigma_{mn}^2)$ over the active interior pixels.

    Matches the summation convention of :func:`_diagonal_inner` (only
    interior bands $m = 1, \\ldots, N_f - 1$).
    """
    return float(np.sum(np.log(2.0 * np.pi * var[1:Nf])))


def whittle_many(w_a: Array, w_b: Array, var: Array, Nf: int) -> Array:
    """Batched :func:`whittle`: $(B, N_f+1, N_t) \\to (B,)$.

    One vectorised reduction over the whole stack instead of $B$ Python-level
    calls.  ``var`` is normally the single shared $(N_f+1, N_t)$ noise grid; a
    full stack is accepted too, for per-template noise.

    This is also :func:`band_averaged` and :func:`evolutionary_whittle` in
    batched form -- all three diagonal flavours are the *same* reduction over
    a different variance grid, so there is one batched implementation rather
    than three.

    Motivation, measured: with :func:`wdm.forward_many` batching the transform,
    a Python loop over :func:`whittle` became the dominant cost of a per-source
    step -- **6x the batched transform** at $N_f=8$, $N_t=16$, $B=512$.  The
    batched form is **2.5-31x** faster than that loop (31x at that size, 2.5x
    at $N_f=64$, $N_t=128$, $B=64$ where the per-call overhead is already
    amortised) and **bit-exactly equal** to it.

    Parameters
    ----------
    w_a, w_b : ndarray, float, shape (B, Nf+1, Nt)
    var : ndarray, float, shape (Nf+1, Nt) or (B, Nf+1, Nt)
    Nf : int

    Returns
    -------
    ndarray, float, shape (B,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((5, Nf * Nt))
    >>> W = wdm.forward_time_many(xs, Nf, Nt, dt)
    >>> var = wdm.whittle_variance(Nf, Nt, dt, lambda f: 1e-3 * np.ones_like(f))
    >>> vals = wdm.inner_product.whittle_many(W, W, var, Nf)
    >>> vals.shape
    (5,)
    >>> bool(np.isclose(vals[2], wdm.inner_product.whittle(W[2], W[2], var, Nf)))
    True
    """
    return _diagonal_inner_many(w_a, w_b, var, Nf)


def loglike_whittle_many(
    w_d: Array,
    w_h: Array,
    var: Array,
    Nf: int,
    *,
    include_logdet: bool = True,
) -> Array:
    """Batched :func:`loglike_whittle`: $(B, N_f+1, N_t) \\to (B,)$.

    The form a sampler or global fit wants: one call per proposal *set* rather
    than per proposal.  Covers the ``band_averaged`` and
    ``evolutionary_whittle`` flavours too -- they differ only in ``var``.

    Either operand may carry the leading axis alone: a stack of templates
    against one data grid works by ordinary broadcasting, so ``w_d`` can be
    $(N_f+1, N_t)$ while ``w_h`` is $(B, N_f+1, N_t)$.

    Parameters
    ----------
    w_d, w_h : ndarray, float, broadcasting to (B, Nf+1, Nt)
    var : ndarray, float, shape (Nf+1, Nt) or (B, Nf+1, Nt)
    Nf : int
    include_logdet : bool, optional
        The log-determinant depends only on ``var``, so with one shared noise
        grid it is a constant added to every entry.

    Returns
    -------
    ndarray, float, shape (B,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> d = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w_d = wdm.forward_time(d, Nf, Nt, dt)
    >>> templates = np.zeros((3, Nf + 1, Nt))
    >>> var = wdm.whittle_variance(Nf, Nt, dt, lambda f: 1e-3 * np.ones_like(f))
    >>> ll = wdm.inner_product.loglike_whittle_many(
    ...     w_d, templates, var, Nf, include_logdet=False
    ... )
    >>> ll.shape, bool(np.all(ll < 0))
    ((3,), True)
    """
    w_d, w_h = np.asarray(w_d), np.asarray(w_h)
    try:
        out_shape = np.broadcast_shapes(w_d.shape, w_h.shape)
    except ValueError:
        # Without this, numpy raises "operands could not be broadcast together"
        # from the subtraction inside `_residual` -- a message naming neither
        # this package nor which argument is wrong. The common cause is two
        # stacks with different batch sizes.
        raise ValueError(
            f"w_d and w_h must broadcast together; got {w_d.shape} and "
            f"{w_h.shape}. Either give both the same leading axis, or give "
            "one of them none at all (a single grid against a stack)."
        ) from None
    residual = np.broadcast_to(_residual(w_d, w_h), out_shape)
    quad = _diagonal_inner_many(residual, residual, np.asarray(var), Nf)
    if include_logdet:
        return -0.5 * quad - 0.5 * _diagonal_logdet(np.asarray(var), Nf)
    return -0.5 * quad


def loglike_whittle(
    w_d: Array,
    w_h: Array,
    var: Array,
    Nf: int,
    *,
    include_logdet: bool = True,
) -> float:
    """Whittle Gaussian log-likelihood.

    $$\\ln p(d|h) = -\\tfrac{1}{2}\\sum_{m,n}
    \\left[\\frac{(w^{(d)}_{mn} - w^{(h)}_{mn})^2}{\\sigma_{mn}^2}
    + \\ln(2\\pi\\sigma_{mn}^2)\\right].$$

    Sums interior bands only (matching :func:`whittle`).  The
    log-determinant term should be included whenever the variance grid
    can change between evaluations (e.g. PSD inference); set
    ``include_logdet=False`` to drop it during waveform parameter
    estimation with a fixed PSD.

    Parameters
    ----------
    w_d, w_h : ndarray, shape (Nf+1, Nt)
    var : ndarray, shape (Nf+1, Nt)
        Per-pixel variance grid (e.g. :func:`wdm.whittle_variance`).
    Nf : int
    include_logdet : bool, optional
        If True (default), include $-\\tfrac{1}{2}\\sum\\ln(2\\pi\\sigma^2)$.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> d = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w_d = wdm.forward_time(d, Nf, Nt, dt)
    >>> var = wdm.whittle_variance(Nf, Nt, dt, lambda f: 1e-3 * np.ones_like(f))
    >>> ll = wdm.inner_product.loglike_whittle(
    ...     w_d, np.zeros_like(w_d), var, Nf, include_logdet=False
    ... )
    >>> bool(ll < 0)  # negative for a nonzero residual
    True
    """
    residual = _residual(w_d, w_h)
    quad = whittle(residual, residual, var, Nf)
    if include_logdet:
        return -0.5 * quad - 0.5 * _diagonal_logdet(var, Nf)
    return -0.5 * quad


def loglike_band_averaged(
    w_d: Array,
    w_h: Array,
    var: Array,
    Nf: int,
    *,
    include_logdet: bool = True,
) -> float:
    """Same as :func:`loglike_whittle` but for the band-averaged variance.

    Use ``var = wdm.band_averaged_variance(...)``.

    Parameters
    ----------
    w_d, w_h : ndarray, shape (Nf+1, Nt)
    var : ndarray, shape (Nf+1, Nt)
    Nf : int
    include_logdet : bool, optional

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> d = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w_d = wdm.forward_time(d, Nf, Nt, dt)
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> var = wdm.band_averaged_variance(Nf, Nt, dt, flat)
    >>> ll = wdm.inner_product.loglike_band_averaged(
    ...     w_d, np.zeros_like(w_d), var, Nf, include_logdet=False
    ... )
    >>> bool(ll < 0)
    True
    """
    return loglike_whittle(w_d, w_h, var, Nf, include_logdet=include_logdet)


def loglike_evolutionary_whittle(
    w_d: Array,
    w_h: Array,
    var_tf: Array,
    Nf: int,
    *,
    include_logdet: bool = True,
) -> float:
    """Diagonal log-likelihood with pixel-dependent variance (non-stationary).

    Use ``var_tf = wdm.evolutionary_variance(...)`` for a slowly-varying
    PSD $S_n(f, t)$.

    Parameters
    ----------
    w_d, w_h : ndarray, shape (Nf+1, Nt)
    var_tf : ndarray, shape (Nf+1, Nt)
    Nf : int
    include_logdet : bool, optional

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> d = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w_d = wdm.forward_time(d, Nf, Nt, dt)
    >>> Sn = lambda f, t: 1e-3 * np.ones((f.size, t.size))
    >>> var = wdm.evolutionary_variance(Nf, Nt, dt, Sn)
    >>> ll = wdm.inner_product.loglike_evolutionary_whittle(
    ...     w_d, np.zeros_like(w_d), var, Nf, include_logdet=False
    ... )
    >>> bool(ll < 0)
    True
    """
    return loglike_whittle(w_d, w_h, var_tf, Nf, include_logdet=include_logdet)


def loglike_gaussian_many(
    w_d: Array,
    w_h: Array,
    Nf: int,
    Nt: int,
    dt: float,
    Sn_array: Array,
    *,
    include_logdet: bool = True,
) -> Array:
    """Batched :func:`loglike_gaussian`: $(B, N_f+1, N_t) \\to (B,)$.

    The exact, full-covariance likelihood for a whole proposal set in one
    call.  Like :func:`loglike_whittle_many`, either operand may carry the
    leading axis alone, so one data grid against a stack of templates works by
    broadcasting.

    Shares :func:`_frequency_quadratic_axis` and :func:`_gaussian_logdet` with
    the scalar form, so the two cannot drift apart -- and unlike the diagonal
    flavours this one costs $O(N \\log N_t)$ per entry, so batching buys the
    per-call overhead only, not a change of complexity.

    Parameters
    ----------
    w_d, w_h : ndarray, float, broadcasting to (B, Nf+1, Nt)
    Nf, Nt : int
    dt : float
    Sn_array : ndarray, float, shape (N//2+1,)
        One-sided PSD on the ``rfftfreq(N, dt)`` grid, shared across the stack.
    include_logdet : bool, optional

    Returns
    -------
    ndarray, float, shape (B,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> N = Nf * Nt
    >>> d = np.random.default_rng(0).standard_normal(N)
    >>> w_d = wdm.forward_time(d, Nf, Nt, dt)
    >>> templates = np.zeros((3, Nf + 1, Nt))
    >>> Sn = 1e-3 * np.ones(N // 2 + 1)
    >>> ll = wdm.inner_product.loglike_gaussian_many(
    ...     w_d, templates, Nf, Nt, dt, Sn, include_logdet=False
    ... )
    >>> ll.shape, bool(np.all(ll < 0))
    ((3,), True)
    """
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "loglike_gaussian_many")
    w_d, w_h = np.asarray(w_d), np.asarray(w_h)
    try:
        out_shape = np.broadcast_shapes(w_d.shape, w_h.shape)
    except ValueError:
        raise ValueError(
            f"w_d and w_h must broadcast together; got {w_d.shape} and "
            f"{w_h.shape}. Either give both the same leading axis, or give "
            "one of them none at all (a single grid against a stack)."
        ) from None
    residual = np.broadcast_to(_residual(w_d, w_h), out_shape)
    if residual.ndim != 3:
        raise ValueError(
            f"coefficient grids must broadcast to (B, Nf+1, Nt) with Nf+1 = "
            f"{Nf + 1} bands for Nf={Nf}; got {out_shape}"
        )
    r_rfft = inverse_many(np.ascontiguousarray(residual), Nf, Nt, dt)
    quad = _frequency_quadratic_axis(r_rfft, Sn_array, dt, N)
    if not include_logdet:
        return -0.5 * quad
    return -0.5 * quad - 0.5 * _gaussian_logdet(Sn_array, N)


def loglike_gaussian(
    w_d: Array,
    w_h: Array,
    Nf: int,
    Nt: int,
    dt: float,
    Sn_array: Array,
    *,
    include_logdet: bool = True,
) -> float:
    """Exact Gaussian log-likelihood via the inverse-WDM quadratic form.

    $$\\ln p(d|h) = -\\tfrac{1}{2}\\,r^T \\mathbf{C}^{-1} r
    - \\tfrac{1}{2}\\ln\\det(2\\pi\\,\\mathbf{C}^{(\\mathrm{WDM})}),
    \\qquad r = d - h.$$

    Both terms cover all $N$ active pixels, so this really is the log of a
    normalized density.  Note the quadratic form is therefore **not**
    ``gaussian(w_d - w_h, w_d - w_h, ...)``: :func:`gaussian` follows the
    standard GW convention of dropping the DC and Nyquist bins, which would
    leave the quadratic form over $N-2$ pixels while the determinant covers
    $N$.  Here those two real-valued bins are included at half weight, which
    reproduces $r^T C_a^{-1} r$ against a dense
    :func:`wdm.matrix.build_C_wdm_active` solve to machine precision.

    The log-determinant is computed in the frequency domain.  Because
    $W_a$ is orthogonal up to the sampling scale ($W_a W_a^T =
    \\delta t\\,I$, i.e. $W_a = \\sqrt{\\delta t}\\,Q$ with $Q$
    orthogonal) and :func:`wdm.matrix.build_C_time` is circulant with
    eigenvalues $S_n(|f_j|)/(2\\,\\delta t)$,

    $$\\ln\\det\\bigl(2\\pi\\mathbf{C}^{(\\mathrm{WDM})}\\bigr)
    = \\sum_{j=0}^{N-1} \\ln\\bigl(\\pi\\,S_n(|f_j|)\\bigr),$$

    summed over the **full two-sided** ``fftfreq`` grid -- one term per
    active WDM pixel, of which there are $N$.

    .. note::
       Before 2026-07-27 this summed $\\ln(2\\pi\\sigma_j^2)$ with
       $\\sigma_j^2 = N S_n/(2\\delta t)$ over the $N/2-1$ interior
       rfft bins only.  That mixed the frequency-domain variance into a
       WDM-basis determinant and used roughly half the required number
       of terms, which biased any PSD amplitude inferred by maximising
       this likelihood by a factor of two.  See ``src/wdm/summary.md``.

    Parameters
    ----------
    w_d, w_h : ndarray, shape (Nf+1, Nt)
    Nf, Nt : int
    dt : float
    Sn_array : ndarray, shape (N//2+1,)
    include_logdet : bool, optional
        If True (default), include the log-determinant term.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> N = Nf * Nt
    >>> d = np.random.default_rng(0).standard_normal(N)
    >>> w_d = wdm.forward_time(d, Nf, Nt, dt)
    >>> Sn = 1e-3 * np.ones(N // 2 + 1)
    >>> ll = wdm.inner_product.loglike_gaussian(
    ...     w_d, np.zeros_like(w_d), Nf, Nt, dt, Sn, include_logdet=False
    ... )
    >>> bool(ll < 0)
    True
    """
    # The residual appears twice in <r|r>; inverse-transform it once rather
    # than letting gaussian() run two identical inverse WDM transforms.
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "loglike_gaussian")
    r_rfft = inverse(_residual(w_d, w_h), Nf, Nt, dt)
    quad = _frequency_quadratic(r_rfft, Sn_array, dt, N)
    if not include_logdet:
        return -0.5 * quad
    # ln det(2 pi C_wdm) over all N active pixels.  Sn_array lives on the
    # one-sided rfft grid; mirror it to the two-sided fftfreq grid so that
    # every pixel contributes exactly once.
    Sn_full = np.empty(N, dtype=float)
    n_rfft = N // 2 + 1
    Sn_full[:n_rfft] = Sn_array
    Sn_full[n_rfft:] = Sn_array[1 : N // 2][::-1]
    # A DC or Nyquist bin sanitized to zero power (an `inf`-at-DC LISA curve is
    # the normal case) carries no information, so it contributes no degree of
    # freedom to the determinant.  Including it would give log(0) = -inf and
    # return a silent `nan`; the quadratic form already gives it zero weight.
    contributing = Sn_full > 0.0
    log_det = float(np.sum(np.log(np.pi * Sn_full[contributing])))
    return -0.5 * quad - 0.5 * log_det
