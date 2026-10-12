"""Forward and inverse WDM transforms (RFFT-only, linear-centred slab).

These implementations follow Algorithms 1 and 2 of
``notes/Time_Frequency_Analysis/wdm_transform_algorithms.tex``: the
data lives only in the one-sided ``rfft`` $X[k]$, $k = 0, \\ldots, N/2$;
the two-sided spectrum is never materialised.  Out-of-range slab
indices are served by Hermitian symmetry on read (forward) or
discarded (inverse, where they are redundant by Hermitian symmetry of
the real WDM wavelets).

The slab is linear-centred:

$$\\ell_{m,k} \\equiv k + (m - 1)\\,N_t/2,\\qquad k = 0, \\ldots, N_t - 1.$$

Interior bands ($1 \\le m \\le N_f - 1$) read directly from the rfft;
only the DC band ($m = 0$) and the Nyquist band ($m = N_f$) ever take
the Hermitian-symmetry branch ($N_t$ slab fetches each).

Forward:

$$w_{mn} = \\frac{2\\sqrt{\\pi}}{N_f}\\,(-1)^{(m+1)n}\\,
\\Re\\!\\left[C_{mn}^{*}\\,\\mathrm{IFFT}\\!\\bigl(u_m\\bigr)[n]\\right],$$

with $u_m[k] = X_{\\ell_{m,k}}\\,\\hat\\phi_c[k]$ (centred filter,
peaked at $k = N_t/2$) and $C_{mn} = 1$ if $m+n$ even, $C_{mn} = i$
otherwise.

Inverse:

$$X[\\ell_{m,k}] \\mathrel{+}= \\frac{\\sqrt{\\pi}}{\\delta t}\\,
\\hat\\phi_c[k]\\,\\mathrm{FFT}\\!\\bigl(w_{m\\cdot}\\,C_{mn}\\,
(-1)^{(m+1)n}\\bigr)[k],$$

with writes outside $[0, N/2]$ discarded.

Both algorithms produce $w_{mn}$ identical to the FFT-natural
implementation in earlier versions of this module to machine precision
(the slab/carrier reordering is algebraically equivalent, see the note
``wdm_transform_algorithms.tex`` \\S Differences).

The DC and Nyquist edge-band coefficients are divided by $\\sqrt{2}$ so
that all basis functions have unit norm.  This makes the WDM transform
matrix orthogonal up to the sampling scale ($W_a W_a^T = \\delta t\\,I$,
so invert with $x = W_a^T w/\\delta t$) and the Parseval identity exact:
$\\sum_{m,n} w_{mn}^2 = \\delta t \\sum_k x_k^2$.

This is the package's **only** coefficient convention -- there is no flag.
The raw "code" alternative, in which the edge basis functions have
$\\|g_{0n}\\|^2 = 2$, used to be selectable via ``normalize_edges=False``
throughout the package.  It bought nothing but a way to get the edge bands
silently wrong: the two conventions differ by a factor of two on the edge
bands and nowhere else, no downstream function could detect a mismatch, and
every non-trivial consumer (``matrix``, ``stationary``, ``inner_product``)
either assumed or hard-coded the normalized one anyway.

Precision
---------
Single precision propagates: a ``float32`` series (or ``complex64`` spectrum)
is transformed entirely in single precision and returns ``float32``.  Anything
else, integers included, promotes to double.  It is opt-in by dtype rather
than by flag, so the precision of a result is readable off its own ``.dtype``.

**This is a memory feature, not a speed one**, and the docstrings say so
because the obvious guess is wrong.  Measured end to end it is 0.87-1.19x --
i.e. nothing -- for the same reason ``scipy.fft`` was reverted above: the FFT
is not the bottleneck.  What it buys is half the memory.  Round-trip accuracy
drops to ~2e-7 relative, from ~4e-16.

Where it reaches, and where it stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **The transform** -- ``forward``/``inverse`` and their ``_time`` and
  ``_many`` forms -- follows the input dtype.
* **The variance builders** ``whittle_variance``, ``band_averaged_variance``
  and ``evolutionary_variance`` follow the *PSD's* dtype, so a ``float32`` PSD
  paired with a ``float32`` series keeps the whole coefficient/variance
  working set single-precision.
* **``empirical_variance`` stays double** on purpose. It is a Monte-Carlo
  reference; computing it in single precision would add numerical noise to a
  noise estimate, which defeats the point of having it.
* **``matrix`` and ``stationary`` stay double** on purpose. They are $O(N^2)$
  verification and covariance machinery, where precision *is* the deliverable
  and the arrays are too large to want in the first place.

An earlier revision claimed "half the memory" while only the coefficient grid
was single -- the variance grid it is invariably paired with stayed double, so
the real working set was ~33% smaller, not 50%. Both are single now, and the
claim is measured rather than assumed.

Before any of this, ``float32`` input was silently upcast and returned
``float64`` -- twice the memory the caller asked for, with no indication.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .backend import NUMPY, Backend, get_backend, is_tracer
from .filter import phi_centered
from .grid import active_mask, n_samples, validate_dt, validate_finite, validate_grid

_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type CArray = NDArray[np.complexfloating]

__all__ = [
    "forward",
    "forward_many",
    "forward_time",
    "forward_time_many",
    "inverse",
    "inverse_many",
    "inverse_time",
    "inverse_time_many",
]

# ---------------------------------------------------------------------------
# FFT backend: numpy, deliberately
# ---------------------------------------------------------------------------
# ``scipy.fft`` was tried and reverted. Recorded here so it is not retried on
# the strength of a microbenchmark:
#
#   * Isolated, scipy IS faster: 1.64x single-threaded and 2.78x at
#     ``workers=8`` on a bare (255, 512) complex ifft.
#   * In this transform it is worth **0.93-1.14x** end to end, measured across
#     four grids from Nf=64 to Nf=1024. The FFT is not the bottleneck -- the
#     parity assembly (``np.where`` plus complex construction) and the scatter
#     dominate -- so Amdahl eats the gain.
#   * ``workers=8`` bought 1.00-1.23x for the same reason.
#
# The cost was real: scipy is not bit-identical to numpy (~3.7e-16 relative,
# same algorithm family, different rounding), and ``src/wdm/summary.md`` claims
# byte-for-byte stability of the coefficients across every change. Trading a
# verifiable property for measurement noise is a bad deal.
#
# Worth revisiting only if the parity assembly is ever vectorised further, at
# which point the FFT share of the runtime rises and the arithmetic changes.


def _working_dtypes(a: NDArray[np.generic]) -> tuple[np.dtype, np.dtype]:
    """``(real, complex)`` working dtypes implied by an input array.

    Single precision propagates: a ``float32`` series or ``complex64`` spectrum
    is transformed entirely in single precision and comes back ``float32``.
    Anything else -- including integer input -- promotes to double.

    This is opt-in by dtype rather than by flag, which keeps the default exact
    and makes the precision of a result readable off its own ``.dtype`` instead
    of requiring the caller to remember what they passed.
    """
    if a.dtype in (np.float32, np.complex64):
        return np.dtype(np.float32), np.dtype(np.complex64)
    return np.dtype(np.float64), np.dtype(np.complex128)


def _check_finite(values: NDArray[np.generic], context: str) -> None:
    """:func:`wdm.validate_finite` on concrete input; skipped for JAX tracers.

    Validation is host work, not arithmetic: a tracer has no values to check,
    and a concrete JAX array is checked on the host like a NumPy one.
    """
    if not is_tracer(values):
        validate_finite(values, context)


def _parity(m: int | np.integer, n_arr: NDArray[np.integer]) -> tuple[BoolArray, Array]:
    """``((m+n) even, (-1)^{(m+1)n})`` for one band.

    The Wilson parity/carrier convention is the most error-prone thing in the
    package -- :func:`wdm.stationary._carriers` carries the 2-D form of the
    same rule.  Both live in exactly one place each so that a sign fix cannot
    land in one copy and miss another.
    """
    even = (m + n_arr) % 2 == 0
    sign = 1.0 - 2.0 * (((m + 1) * n_arr) % 2)
    return even, sign


def _build_slab(
    X_rfft: CArray, m: int, Nf: int, Nt: int, cdtype: np.dtype = _C128,
    be: Backend = NUMPY,
) -> CArray:
    """Linear-centred slab $u_m[k] = X[\\ell_{m,k}]$ for $k \\in [0, N_t-1]$.

    Hermitian extension is applied inline: negative indices use the
    conjugate of the positive bin (only fires at $m = 0$, $k < N_t/2$);
    over-Nyquist indices use the conjugate of the mirror bin (only
    fires at $m = N_f$, $k > N_t/2$).  For interior bands every lookup
    is a direct rfft read.

    **Shape-generic in the leading axes.** The slab indices depend only on
    ``(m, Nf, Nt)``, never on the data, so a stack of spectra gathers with the
    same index arrays under ``...`` indexing.  :func:`forward_many` used to
    call this once per row inside a list comprehension and ``np.stack`` the
    results -- which left the two edge bands unbatched and cost **85% of
    ``forward_many``'s runtime** at ``Nf=8, Nt=16, B=512``, the small-grid
    many-series case the batched path exists for.

    Parameters
    ----------
    X_rfft : ndarray, complex, shape (..., N//2+1)
    m : int
    Nf, Nt : int
    cdtype : dtype, optional
        Complex working dtype.

    Returns
    -------
    ndarray, complex, shape (..., Nt)
    """
    N = n_samples(Nf, Nt)
    ell = np.arange(Nt) + (m - 1) * (Nt // 2)
    # Every slot is written below, so zeros (not np.empty) costs nothing and is
    # the only allocation JAX offers.  Integer positions replace the boolean
    # masks: the same assignments, and concrete indices for JAX.
    slab = be.xp.zeros((*X_rfft.shape[:-1], Nt), dtype=cdtype)
    neg = np.flatnonzero(ell < 0)
    over = np.flatnonzero(ell > N // 2)
    direct = np.flatnonzero((ell >= 0) & (ell <= N // 2))
    slab = be.set(slab, (..., direct), X_rfft[..., ell[direct]], copy=False)
    if neg.size:
        slab = be.set(slab, (..., neg), be.xp.conj(X_rfft[..., -ell[neg]]), copy=False)
    if over.size:
        slab = be.set(slab, (..., over), be.xp.conj(X_rfft[..., N - ell[over]]), copy=False)
    return slab


def _prepare_edges_for_inverse(
    wmn: Array, Nf: int, Nt: int, rdtype: np.dtype = _F64, be: Backend = NUMPY
) -> Array:
    """Working copy of ``wmn`` with the edge bands put into ``inverse``'s form.

    Two things, both edge-only:

    1. **Zero the parity-killed entries.** They correspond to zero rows of $W$,
       so the adjoint must ignore them. Without this, junk there leaks into the
       reconstruction -- measured: a unit perturbation of one inactive DC pixel
       moved the *active* pixels of ``forward(inverse(w))`` by 2.5 absolute.
       Anything produced by :func:`forward` already carries ~1e-16 there, so
       this is a no-op on well-formed input.
    2. **Undo the unit-norm $\\sqrt2$** that :func:`forward` applied.

    ``active_mask`` is built once. It used to be called twice, once per edge
    band, allocating the whole $(N_f+1)\\times N_t$ boolean array each time to
    read a single row -- 5.1% of ``inverse`` at $N_f=64$, $N_t=128$.

    Shape-generic in the leading axes, so :func:`inverse` and
    :func:`inverse_many` share it.
    """
    work = wmn.astype(rdtype)          # NumPy: always a fresh copy, so copy=False below is safe
    mask = active_mask(Nf, Nt)
    root2 = np.sqrt(2)
    # ``work[..., 0, :] *= mask[0] * root2`` as a functional update; on NumPy
    # it is the same in-place multiply, so the rounding is unchanged.
    work = be.set(work, (..., 0, slice(None)), work[..., 0, :] * (mask[0] * root2), copy=False)
    work = be.set(work, (..., Nf, slice(None)), work[..., Nf, :] * (mask[Nf] * root2), copy=False)
    return work


def _scatter_interior_bands(
    X: CArray,
    wmn_work: Array,
    Nf: int,
    Nt: int,
    phi_c: Array,
    lam: float,
    be: Backend = NUMPY,
) -> CArray:
    """Accumulate the interior bands into the rfft accumulator ``X``; returns it.

    One batched length-$N_t$ FFT, then a contiguous slice-add per band. Within
    a band the slab indices are consecutive and duplicate-free, so ``X[a:b] +=``
    is exact and much cheaper than ``np.add.at``; consecutive bands overlap by
    $N_t/2$, which the sequential loop handles correctly.

    Shape-generic in the leading axes: ``X`` may be ``(N//2+1,)`` or
    ``(B, N//2+1)`` and ``wmn_work`` correspondingly, so :func:`inverse` and
    :func:`inverse_many` run the same code and cannot drift apart.

    There is no ``Nf < 2`` guard because there is no such grid:
    :func:`wdm.validate_grid` requires ``Nf`` positive **and** even, so the
    smallest legal value is 2 and every caller has at least one interior band.
    """
    n_arr = np.arange(Nt)
    m_int = np.arange(1, Nf)
    m_par = m_int & 1
    n_par = n_arr & 1
    even_int = m_par[:, None] == n_par[None, :]
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    xp = be.xp
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
    # g = w' * C_mn * (-1)^{(m+1)n}: real where (m+n) even, imaginary otherwise.
    G = xp.where(even_int, scaled_int, 0.0) + 1j * xp.where(even_int, 0.0, scaled_int)
    W_int = be.fft.fft(G, axis=-1) * (lam * phi_c)
    # Sequential slice-adds in band order, exactly the oracle's ``+=`` loop.
    # Every rfft bin receives at most two addends (adjacent interior bands
    # overlap by Nt/2; the edge bands only touch bins also covered by band 1 or
    # band Nf-1, plus bin N/2 alone), and a + b == b + a in IEEE arithmetic, so
    # the order is immaterial on any backend.
    for i, m_i in enumerate(m_int):
        start = int(m_i - 1) * (Nt // 2)
        X = be.add_at(X, (..., slice(start, start + Nt)), W_int[..., i, :], copy=False)
    return X


def _scatter_edge_bands(
    X: CArray, wmn_work: Array, Nf: int, Nt: int, N: int, phi_c: Array, lam: float,
    be: Backend = NUMPY,
) -> CArray:
    """Accumulate the DC and Nyquist bands into ``X``; returns it.

    These scatter partly outside $[0, N/2]$; those writes are redundant by
    Hermitian symmetry of the real WDM wavelets and are discarded.

    Shape-generic in the leading axes, as :func:`_scatter_interior_bands`.
    ``ell`` is strictly increasing, so the surviving indices are
    duplicate-free and a plain ``+=`` is exact -- ``np.add.at`` was only ever
    guarding against duplicates that cannot occur here.
    """
    xp = be.xp
    n_arr = np.arange(Nt)
    for m in (0, Nf):
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign
        g = xp.where(even, scaled, 0.0) + 1j * xp.where(even, 0.0, scaled)
        Wm = be.fft.fft(g, axis=-1)
        ell = n_arr + (m - 1) * (Nt // 2)
        valid = np.flatnonzero((ell >= 0) & (ell <= N // 2))
        # ``X[..., ell[valid]] += v``: the indices are strictly increasing, so the
        # fancy-index add is one addition per bin either way.
        X = be.add_at(X, (..., ell[valid]), lam * phi_c[valid] * Wm[..., valid], copy=False)
    return X


def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, be: Backend = NUMPY) -> Array:
    """Coefficient grid(s) from spectra: ``(..., N//2+1) -> (..., Nf+1, Nt)``.

    The whole of the forward algorithm, shape-generic in the leading axes, so
    :func:`forward` and :func:`forward_many` are thin validation wrappers over
    one implementation rather than two copies of it.

    Interior bands $m = 1 \\ldots N_f-1$: every slab index
    $\\ell_{m,k} = k + (m-1)N_t/2$ lies in $[0, N/2]$
    (max $= (N_f-1)N_t/2 - 1 < N/2$), so the slab is a pure gather from the
    rfft with no Hermitian branch, and all of them go through one batched
    length-$N_t$ IFFT.  Unconditional: :func:`wdm.validate_grid` requires $N_f$
    positive **and** even, so $N_f \\ge 2$ and there is always at least one
    interior band.

    Edge bands $m \\in \\{0, N_f\\}$ take the Hermitian-symmetry slab path.
    Two bands, so the Python loop over them is irrelevant to the cost.
    """
    xp = be.xp
    rdtype, cdtype = _working_dtypes(x_rfft)
    phi_c = phi_centered(Nf, Nt, dt).astype(rdtype)          # host constant
    kappa = 2.0 * np.sqrt(np.pi) / Nf
    n_arr = np.arange(Nt)
    out: Array = xp.zeros((*x_rfft.shape[:-1], Nf + 1, Nt), dtype=rdtype)

    m_int = np.arange(1, Nf)
    ell = (m_int - 1)[:, None] * (Nt // 2) + n_arr[None, :]  # (Nf-1, Nt)
    Y = be.fft.ifft(x_rfft[..., ell] * phi_c, axis=-1)

    # The parity patterns depend only on the parities of m and n, so build them
    # from 1D parity vectors instead of forming (m+1)*n over the whole grid.
    m_par = m_int & 1
    n_par = n_arr & 1
    even_int = m_par[:, None] == n_par[None, :]  # (m+n) even <=> parities equal
    # Re(C_mn^* y_m[n]): even -> Re(y_m); odd -> Re(-i y_m) = Im(y_m).
    re_or_im_int = xp.where(even_int, Y.real, Y.imag)
    # (-1)^{(m+1)n}; exponent parity = ((m+1) & 1) * (n & 1).
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    # ``out[..., 1:Nf, :] = (kappa * sign_int) * re_or_im_int``: same grouping,
    # computed in float64 and cast on assignment, exactly as before.
    out = be.set(out, (..., slice(1, Nf), slice(None)), kappa * sign_int * re_or_im_int,
                 copy=False)

    for m in (0, Nf):
        y_m = be.fft.ifft(_build_slab(x_rfft, m, Nf, Nt, cdtype, be) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out = be.set(out, (..., m, slice(None)), kappa * sign * xp.where(even, y_m.real, y_m.imag),
                     copy=False)

    # Unit-norm edge basis functions. Not optional: see the module docstring.
    # ``out[..., 0, :] /= np.sqrt(2)``: the in-place divide computes in float64
    # (np.sqrt(2) is a float64 scalar) and rounds once on store -- as here.
    root2 = np.sqrt(2)
    out = be.set(out, (..., 0, slice(None)), out[..., 0, :] / root2, copy=False)
    out = be.set(out, (..., Nf, slice(None)), out[..., Nf, :] / root2, copy=False)
    return out


def forward(
    x_rfft: CArray,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> Array:
    """Forward WDM transform: $\\text{rfft}(x) \\to w_{mn}$.

    RFFT-only implementation following Algorithm 1 of
    ``wdm_transform_algorithms.tex``: the full-length two-sided spectrum is
    never built, only the one-sided rfft is read.  Batching the interior bands
    trades memory for speed -- the working set is $O(N_f N_t)$, several times
    the input rfft, not the $O(N_t)$ of a per-band loop.  See the Notes below.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (N//2+1,)
        One-sided DFT of the real time series, as returned by
        ``np.fft.rfft(x)`` (no $\\delta t$ scaling).
    Nf : int
        Number of frequency bands; $N = N_f N_t$.
    Nt : int
        Number of time bins.
    dt : float
        Sampling interval.

    Notes
    -----
    Cost: $O(N \\log N_t)$.  The $N_f-1$ interior bands are done in one
    batched length-$N_t$ IFFT; only the two edge bands take the
    Hermitian-symmetry path.  Working set is $O(N_f N_t)$ -- the batched
    slab is materialised -- not the $O(N_t)$ of a per-band loop.
    :func:`inverse` costs modestly more than :func:`forward` (it scatters
    into an accumulator rather than gathering).

    Returns
    -------
    ndarray, float, shape (Nf+1, Nt)
        Real WDM coefficients.  Edge bands have $N_t/2$ active values;
        the inactive entries vanish by Hermitian symmetry of the slab,
        realized to floating-point rounding (~1e-16 relative to the
        peak) rather than bit-exactly.  Compare them with a tolerance.

    Raises
    ------
    ValueError
        If ``Nf`` or ``Nt`` is not a positive even integer, if ``x_rfft``
        does not have length ``N//2 + 1``, or if it holds a ``nan`` or
        ``inf`` (see :func:`wdm.validate_finite` -- one bad bin would
        otherwise silently make every output coefficient ``nan``).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> w.shape
    (9, 16)
    >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(x_rfft)
    x_rfft = be.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape ({N // 2 + 1},) for Nf={Nf}, Nt={Nt}; "
            f"got {x_rfft.shape}"
        )
    _check_finite(x_rfft, "forward: x_rfft")
    return _gather_bands(x_rfft, Nf, Nt, dt, be)


def forward_many(
    x_rfft: CArray,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> Array:
    """Forward-transform a stack of spectra: $(B, N/2+1) \\to (B, N_f+1, N_t)$.

    Same result as looping :func:`forward` over the first axis, but the whole
    stack goes through **one** batched IFFT and one parity assembly instead of
    $B$ of each.  The saving is the per-call Python and NumPy-dispatch
    overhead, which is what dominates for small grids -- exactly the regime of
    a global fit evaluating many sources on the same grid.

    Bit-identical to the loop, not merely close; :func:`forward` and this
    function run the same arithmetic in the same order.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (B, N//2+1)
        Stack of one-sided spectra sharing one grid.
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (B, Nf+1, Nt)

    Raises
    ------
    ValueError
        If ``x_rfft`` is not 2-D with second axis ``N//2+1``, or holds a
        ``nan``/``inf``.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((5, Nf * Nt))
    >>> W = wdm.forward_many(np.fft.rfft(xs, axis=-1), Nf, Nt, dt)
    >>> W.shape
    (5, 9, 16)
    >>> bool(np.allclose(W[2], wdm.forward(np.fft.rfft(xs[2]), Nf, Nt, dt)))
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(x_rfft)
    x_rfft = be.asarray(x_rfft)
    if x_rfft.ndim != 2 or x_rfft.shape[1] != N // 2 + 1:
        raise ValueError(
            f"x_rfft must have shape (B, {N // 2 + 1}) for Nf={Nf}, Nt={Nt}; "
            f"got {x_rfft.shape}"
        )
    _check_finite(x_rfft, "forward_many: x_rfft")
    return _gather_bands(x_rfft, Nf, Nt, dt, be)


def forward_time_many(
    x: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> Array:
    """Convenience wrapper: a stack of time series $(B, N) \\to (B, N_f+1, N_t)$.

    Equivalent to ``forward_many(np.fft.rfft(x, axis=-1), ...)``.

    Parameters
    ----------
    x : ndarray, float, shape (B, N)
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (B, Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((4, Nf * Nt))
    >>> wdm.forward_time_many(xs, Nf, Nt, dt).shape
    (4, 9, 16)
    """
    validate_grid(Nf, Nt)
    be = be or get_backend(x)
    x = be.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(
            f"x must have shape (B, {expected}) = (B, Nf*Nt) for Nf={Nf}, "
            f"Nt={Nt}; got {x.shape}"
        )
    _check_finite(x, "forward_time_many: x")
    return forward_many(be.fft.rfft(x, axis=-1), Nf, Nt, dt, be=be)


def inverse(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> CArray:
    """Inverse WDM transform: $w_{mn} \\to \\text{rfft}(x)$.

    RFFT-only implementation following Algorithm 2 of
    ``wdm_transform_algorithms.tex``.  Each band scatters into the
    one-sided rfft accumulator at its linear-centred slab indices;
    writes outside $[0, N/2]$ (only at the edge bands) are discarded
    as redundant by Hermitian symmetry of the real WDM wavelets.

    Parameters
    ----------
    wmn : ndarray, float, shape (Nf+1, Nt)
        WDM coefficient grid.
    Nf, Nt : int
        Frequency and time bin counts.
    dt : float
        Sampling interval.

    Returns
    -------
    ndarray, complex, shape (N//2+1,)
        Reconstructed one-sided DFT (compatible with ``np.fft.irfft``).

    Raises
    ------
    ValueError
        If the grid is invalid, if ``wmn`` is complex or wrongly shaped, or
        if it holds a ``nan`` or ``inf`` (see :func:`wdm.validate_finite`).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> bool(np.allclose(wdm.inverse(w, Nf, Nt, dt), np.fft.rfft(x)))
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    wmn = be.asarray(wmn)
    if be.xp.iscomplexobj(wmn):
        raise ValueError(
            "wmn must be real; a complex grid would have its imaginary part "
            "silently discarded"
        )
    if wmn.shape != (Nf + 1, Nt):
        raise ValueError(f"wmn must have shape ({Nf + 1}, {Nt}); got {wmn.shape}")
    _check_finite(wmn, "inverse: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = phi_centered(Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype, be)

    lam = np.sqrt(np.pi) / dt
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)

    return X


def inverse_many(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> CArray:
    """Inverse-transform a stack: $(B, N_f+1, N_t) \\to (B, N/2+1)$.

    The batched counterpart of :func:`inverse`, and the return leg of
    :func:`forward_many`.  Shares every helper with the unbatched form -- the
    scatter routines are generic in the leading axes -- so the two cannot
    drift apart.  Bit-identical to looping :func:`inverse`.

    Parameters
    ----------
    wmn : ndarray, float, shape (B, Nf+1, Nt)
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, complex, shape (B, N//2+1)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((4, Nf * Nt))
    >>> W = wdm.forward_time_many(xs, Nf, Nt, dt)
    >>> wdm.inverse_many(W, Nf, Nt, dt).shape
    (4, 65)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    wmn = be.asarray(wmn)
    if be.xp.iscomplexobj(wmn):
        raise ValueError(
            "wmn must be real; a complex grid would have its imaginary part "
            "silently discarded"
        )
    if wmn.ndim != 3 or wmn.shape[1:] != (Nf + 1, Nt):
        raise ValueError(
            f"wmn must have shape (B, {Nf + 1}, {Nt}) for Nf={Nf}, Nt={Nt}; "
            f"got {wmn.shape}"
        )
    _check_finite(wmn, "inverse_many: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = phi_centered(Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype, be)

    lam = np.sqrt(np.pi) / dt
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
    return X


def inverse_time_many(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> Array:
    """Convenience wrapper: $(B, N_f+1, N_t) \\to$ a stack of series $(B, N)$.

    Equivalent to ``np.fft.irfft(inverse_many(wmn, ...), n=N, axis=-1)``.

    Parameters
    ----------
    wmn : ndarray, float, shape (B, Nf+1, Nt)
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (B, N)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((4, Nf * Nt))
    >>> W = wdm.forward_time_many(xs, Nf, Nt, dt)
    >>> bool(np.allclose(wdm.inverse_time_many(W, Nf, Nt, dt), xs, atol=1e-12))
    True
    """
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    return be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be), n=N, axis=-1)


def forward_time(
    x: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> Array:
    """Convenience wrapper: time-domain $x_k \\to w_{mn}$.

    Equivalent to ``forward(np.fft.rfft(x), ...)``.

    Parameters
    ----------
    x : ndarray, float, shape (N,)
        Real time series with $N = N_f N_t$ samples.
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (Nf+1, Nt)

    Raises
    ------
    ValueError
        If the grid is invalid, if ``x`` does not have ``Nf*Nt`` samples, or
        if it holds a ``nan`` or ``inf``.  Gapped detector data often arrives
        ``nan``-filled; see :func:`wdm.validate_finite` for why that has to
        be an error rather than something to propagate.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> wdm.forward_time(x, Nf, Nt, dt).shape
    (9, 16)
    >>> x[5] = np.nan                      # one gap-filled sample
    >>> wdm.forward_time(x, Nf, Nt, dt)    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: forward_time: x: 1 of 128 entries are not finite...
    """
    validate_grid(Nf, Nt)
    be = be or get_backend(x)
    x = be.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.shape != (expected,):
        raise ValueError(
            f"x must have shape ({expected},) = (Nf*Nt,) for Nf={Nf}, Nt={Nt}; "
            f"got {x.shape}"
        )
    # Check here rather than only on the rfft: reporting "index 5 of x" beats
    # reporting that every bin of the transformed spectrum went non-finite.
    _check_finite(x, "forward_time: x")
    return forward(be.fft.rfft(x), Nf, Nt, dt, be=be)


def inverse_time(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
) -> Array:
    """Convenience wrapper: $w_{mn} \\to$ time-domain $x_k$.

    Equivalent to ``np.fft.irfft(inverse(wmn, ...), n=N)``.

    Parameters
    ----------
    wmn : ndarray, float, shape (Nf+1, Nt)
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (N,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wdm.forward_time(x, Nf, Nt, dt)
    >>> bool(np.allclose(wdm.inverse_time(w, Nf, Nt, dt), x, atol=1e-12))
    True
    """
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    x_rfft = inverse(wmn, Nf, Nt, dt, be=be)
    return be.fft.irfft(x_rfft, n=N)
