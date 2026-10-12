"""Per-pixel noise variance grids in the WDM domain.

For stationary Gaussian noise with one-sided PSD $S_n(f)$, the
expected variance of a normalized WDM coefficient $w_{mn}$ is
approximately

$$\\sigma_{mn}^2 \\approx S_n(f_m) / 2,$$

uniformly across all pixels.  This
module exposes three flavors of variance grid:

* :func:`whittle_variance` — point-evaluate the PSD at band centers
  (cheapest, exact when $S_n$ is constant within each band).
* :func:`band_averaged_variance` — filter-weighted mean of $S_n$ over
  each band; exact, and the right choice for steep PSDs.
* :func:`evolutionary_variance` — pixel-dependent variance for
  locally-stationary noise with a time-varying PSD $S_n(f, t)$.

A Monte-Carlo estimator (:func:`empirical_variance`) is provided for
verification.

All variance grids return shape ``(Nf+1, Nt)`` so they slot directly
into the inner-product functions.

Argument order follows the package convention: the grid first
(``Nf, Nt, dt``), then the PSD. It matches :mod:`wdm.matrix` and
:mod:`wdm.inner_product`, so the PSD is always the trailing argument.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .filter import phi_centered
from .grid import (
    band_frequencies,
    delta_F,
    n_samples,
    sanitize_psd,
    time_pixel_centers,
    validate_dt,
    validate_grid,
)
from .transform import forward

type Array = NDArray[np.floating]
type CArray = NDArray[np.complexfloating]
type PsdFunc = Callable[[Array], Array]
type PsdFtFunc = Callable[[Array, Array], Array]

__all__ = [
    "band_averaged_variance",
    "empirical_variance",
    "evaluate_psd_on_band_grid",
    "evolutionary_variance",
    "whittle_variance",
    "whittle_variance_from_array",
]


def evaluate_psd_on_band_grid(Nf: int, dt: float, Sn_func: PsdFunc) -> Array:
    """Evaluate the PSD at all band centers.

    Parameters
    ----------
    Nf : int
    dt : float
    Sn_func : callable
        Function ``f -> S_n(f)`` that accepts an ndarray of frequencies.

    Returns
    -------
    ndarray, shape (Nf+1,)
        $S_n(f_m)$ for $m = 0, \\ldots, N_f$.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Sn = wdm.evaluate_psd_on_band_grid(
    ...     8, 1.0, lambda f: 1e-3 * np.ones_like(f)
    ... )
    >>> Sn.shape, float(Sn[0])
    ((9,), 0.001)
    """
    freqs = band_frequencies(Nf, dt)
    out = sanitize_psd(Sn_func(freqs), "evaluate_psd_on_band_grid")
    if out.shape != freqs.shape:
        raise ValueError(
            f"Sn_func must be elementwise: expected shape {freqs.shape}, "
            f"got {out.shape}"
        )
    return out


def whittle_variance_from_array(Nt: int, Sn_band: Array) -> Array:
    """Whittle variance grid from precomputed band-center PSD values.

    Parameters
    ----------
    Nt : int
        Number of time bins.  Grid arguments come first throughout the
        package; $N_f$ is implied by ``Sn_band``'s length.
    Sn_band : ndarray, shape (Nf+1,)
        $S_n$ evaluated at band centers $f_m = m\\,\\Delta F$.
        Number of time bins.

    Returns
    -------
    ndarray, shape (Nf+1, Nt)
        Variance grid $\\sigma_{mn}^2 = S_n(f_m)/2$, broadcast over $n$.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> var = wdm.whittle_variance_from_array(16, np.full(9, 1e-3))
    >>> var.shape, float(var[0, 0])   # S_n / 2
    ((9, 16), 0.0005)
    """
    sigma2_band = 0.5 * sanitize_psd(Sn_band, "whittle_variance_from_array")
    if sigma2_band.ndim != 1:
        raise ValueError(
            f"Sn_band must be 1D of length Nf+1; got shape {sigma2_band.shape}"
        )
    # Nf is implied by the length; validate the (Nf, Nt) pair so this cannot
    # produce a grid no valid transform could generate (e.g. an empty (9, 0)).
    validate_grid(sigma2_band.size - 1, Nt)
    return np.broadcast_to(sigma2_band[:, None], (sigma2_band.size, Nt)).copy()


def whittle_variance(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
) -> Array:
    """Whittle variance grid: $\\sigma_{mn}^2 = S_n(f_m)/2$.

    Approximates the WDM coefficient variance by point-evaluating the
    PSD at the band center.  Exact when $S_n$ is constant within each
    band.  See :func:`band_averaged_variance` for a better approximation
    when the PSD varies across a band.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable
        Function ``f -> S_n(f)``.

    Returns
    -------
    ndarray, shape (Nf+1, Nt)
        Per-pixel variance.  Independent of $n$ for stationary noise.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> var = wdm.whittle_variance(8, 16, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> var.shape
    (9, 16)
    >>> float(var[1, 0])  # S_n(f_m) / 2
    0.0005
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    Sn_band = evaluate_psd_on_band_grid(Nf, dt, Sn_func)
    return whittle_variance_from_array(Nt, Sn_band)


def band_averaged_variance(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
) -> Array:
    """Filter-weighted mean of $S_n$ over each band, divided by 2.

    For each band $m$, computes the effective PSD

    $$S_n^{\\text{eff}}(m)
    = \\frac{\\sum_j \\hat\\phi^2(f_j - f_m)\\,S_n(f_j)}
            {\\sum_j \\hat\\phi^2(f_j - f_m)}$$

    and returns $\\sigma_{mn}^2 = S_n^{\\text{eff}}(m)/2$ broadcast
    over $n$.  This is exact: it reproduces the diagonal of the WDM
    covariance $W\\,C\\,W^T$ to machine precision, because the pixel
    variance is the $\\hat\\phi^2$-weighted average of the PSD across
    the band.  Strictly more accurate than :func:`whittle_variance`,
    which point-evaluates $S_n$ at the band center.

    .. note::
       Before 2026-07-27 this used the filter-weighted **harmonic**
       mean $\\bigl(\\sum \\hat\\phi^2/S_n\\bigr)^{-1}\\sum\\hat\\phi^2$,
       which is not the pixel variance and was measurably *less*
       accurate than :func:`whittle_variance` in every interior band.
       See ``src/wdm/summary.md``.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable

    Returns
    -------
    ndarray, shape (Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> steep = lambda f: 1e-3 * (1.0 + 1e2 / (f + 1 / (2 * dt * Nf)) ** 2)
    >>> var = wdm.band_averaged_variance(Nf, Nt, dt, steep)
    >>> var.shape
    (9, 16)
    >>> # exact: reproduces the diagonal of the WDM covariance
    >>> truth = np.diag(wdm.matrix.build_C_wdm_active(Nf, Nt, dt, steep))
    >>> bool(abs(var[1, 0] - truth[Nt // 2]) / truth[Nt // 2] < 1e-12)
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    phi = phi_centered(Nf, Nt, dt)
    phi2 = phi**2
    f_offsets = (np.arange(Nt) - Nt // 2) / (n_samples(Nf, Nt) * dt)

    dF = delta_F(Nf, dt)
    f_nyq = 0.5 / dt

    # Fold every band's slab onto the physical band [0, f_Nyq] in one shot.
    # The DC band's slab runs below zero and folds about 0; the Nyquist band's
    # runs above f_Nyq and folds about f_Nyq -- that is where the discrete
    # spectrum actually aliases. Folding the top band about 0 (plain ``abs``)
    # samples the PSD at the wrong frequencies and biases band m=Nf by several
    # percent. One vectorised call also means one user-PSD invocation instead
    # of Nf+1, which matters for an expensive sensitivity curve.
    f_bins = np.arange(Nf + 1)[:, None] * dF + f_offsets[None, :]
    f_folded = np.abs(f_bins)
    f_folded = np.where(f_folded > f_nyq, 2.0 * f_nyq - f_folded, f_folded)
    Sn_vals = sanitize_psd(Sn_func(f_folded), "band_averaged_variance")
    if Sn_vals.shape != f_folded.shape:
        raise ValueError(
            f"Sn_func must be elementwise: expected shape {f_folded.shape}, "
            f"got {Sn_vals.shape}"
        )

    # Weight by every bin the window touches. A bin where S_n vanishes still
    # belongs in the denominator -- it contributes zero power, which is exactly
    # what pulls the band average down. Dropping such bins from both sums (as
    # an earlier ``Sn_vals > 0`` mask did) rescales the average by the surviving
    # weight and overstates a band with a spectral cutoff in it by up to 3.2x.
    # The Meyer window always has positive bins -- measured 3/4 of them at
    # Nf=2, Nt=4 and 1535/2048 at Nf=1024, Nt=2048, across dt from 1e-6 to 1e6 --
    # so ``in_band`` is never empty. An earlier revision carried an
    # ``if not np.any(in_band): Sn_eff = inf`` fallback for that case. It was
    # unreachable, and a silent infinite variance is the last thing this
    # package should produce, so it is gone rather than merely untested.
    in_band = phi2 > 0
    # Match the window weights to the PSD's precision, so a float32 PSD yields
    # a float32 variance grid instead of being promoted back to float64 by the
    # matmul. The window itself is always computed in double; only the
    # reduction is cast.
    weights = phi2[in_band].astype(Sn_vals.dtype, copy=False)
    Sn_eff = Sn_vals[:, in_band] @ weights / np.sum(weights)

    return whittle_variance_from_array(Nt, Sn_eff)


def evolutionary_variance(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_tn_func: PsdFtFunc,
) -> Array:
    """Evolutionary (locally stationary) variance grid.

    For noise with a slowly-varying PSD $S_n(f, t)$, returns the
    per-pixel variance $\\sigma_{mn}^2 = S_n(f_m, t_n)/2$ evaluated at
    the band center frequencies and time-pixel centers.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_tn_func : callable
        Function ``(f, t) -> S_n(f, t)`` accepting 1D arrays of length
        ``Nf+1`` and ``Nt`` respectively, returning a 2D array of shape
        ``(Nf+1, Nt)``.

    Returns
    -------
    ndarray, shape (Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> def Sn(f, t):
    ...     return 1e-3 * (1 + 0.1 * t[None, :] / t[-1]) * np.ones(
    ...         (f.size, t.size)
    ...     )
    >>> var = wdm.evolutionary_variance(Nf, Nt, dt, Sn)
    >>> var.shape, bool(var[4, 0] != var[4, -1])   # varies in time
    ((9, 16), True)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    f = band_frequencies(Nf, dt)
    t = time_pixel_centers(Nf, Nt, dt)
    Sn_grid = sanitize_psd(Sn_tn_func(f, t), "evolutionary_variance")
    if Sn_grid.shape != (Nf + 1, Nt):
        raise ValueError(
            f"Sn_tn_func must return shape ({Nf + 1}, {Nt}); got {Sn_grid.shape}"
        )
    return 0.5 * Sn_grid


def _generate_colored_noise(
    N: int,
    dt: float,
    Sn_func: PsdFunc,
    rng: np.random.Generator,
) -> CArray:
    """Generate one realization of stationary noise with PSD $S_n(f)$.

    Returns the rfft of the time-domain noise, with the convention
    $E[|X_j|^2] = N\\,S_n(f_j)/(2\\,\\delta t)$ at **every** bin.

    The DC and Nyquist bins are real-valued (as they must be for a real
    time series) but carry their full variance, not half: for real
    Gaussian noise $E[|X_j|^2]$ is flat across the rfft grid.

    .. note::
       Before 2026-07-27 the DC and Nyquist draws were divided by
       $\\sqrt 2$, giving them half the correct power.  That biased
       :func:`empirical_variance` low at the two edge bands -- exactly
       where an analytic cross-check is hardest.  See
       ``src/wdm/summary.md``.
    """
    freqs = np.fft.rfftfreq(N, d=dt)
    Sn = sanitize_psd(Sn_func(freqs), "_generate_colored_noise")
    sigma_full = np.sqrt(N * Sn / (2.0 * dt))

    n_rfft = N // 2 + 1
    out = np.zeros(n_rfft, dtype=complex)
    # Interior bins split their variance across the real and imaginary parts;
    # the real-only DC and Nyquist bins carry theirs in one component.
    interior = slice(1, N // 2)
    n_interior = (N // 2) - 1
    out[interior] = (
        sigma_full[interior]
        * (rng.standard_normal(n_interior) + 1j * rng.standard_normal(n_interior))
        / np.sqrt(2.0)
    )
    out[0] = sigma_full[0] * rng.standard_normal()
    out[N // 2] = sigma_full[N // 2] * rng.standard_normal()
    return out


def empirical_variance(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
    n_realizations: int = 1000,
    seed: int | None = None,
) -> Array:
    """Monte-Carlo estimate of $\\langle w_{mn}^2 \\rangle$.

    Generates ``n_realizations`` independent noise realizations with the
    given PSD, transforms each to the WDM domain, and averages the
    squared coefficients.  Useful for verifying the analytical
    :func:`whittle_variance` and :func:`band_averaged_variance`.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable
    n_realizations : int, optional
        Number of realizations to average (default 1000).
    seed : int or None, optional
        Seed for the RNG.

    Returns
    -------
    ndarray, shape (Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> mc = wdm.empirical_variance(8, 16, 1.0, flat, 200, seed=0)
    >>> mc.shape
    (9, 16)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    if int(n_realizations) < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}")
    N = n_samples(Nf, Nt)
    rng = np.random.default_rng(seed)
    accum = np.zeros((Nf + 1, Nt))
    for _ in range(n_realizations):
        x_rfft = _generate_colored_noise(N, dt, Sn_func, rng)
        w = forward(x_rfft, Nf, Nt, dt)
        accum += w * w
    return accum / n_realizations
