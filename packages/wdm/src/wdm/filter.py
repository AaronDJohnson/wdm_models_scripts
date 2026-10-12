"""Meyer scaling function used as the WDM filter.

The Meyer scaling function $\\hat\\phi(\\omega)$ has compact frequency support
on $[-3\\Delta\\Omega/4,\\,+3\\Delta\\Omega/4]$ and a smooth ($C^{2d-1}$)
rolloff controlled by the regularized incomplete beta function $\\nu_d$.

It is the window that, paired with the half-shifted Wilson lattice and the
$C_{mn}$ phase factor, gives an orthonormal time-frequency basis at
critical density (the partition-of-unity property).

Parameters
----------
The shape parameter ``d`` (default ``4``) controls the smoothness of the
rolloff.  The basis functions decay as $|t|^{-(d+1)}$ in time (measured
slopes -3.00/-4.00/-4.99/-6.94/-9.01 at d = 2/3/4/6/8; lisa/wdm b5973b7
corrected the earlier $|t|^{-2d}$ claim).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc

from .grid import validate_dt, validate_grid

type Array = NDArray[np.floating]

__all__ = [
    "nu_d",
    "phi_centered",
    "phi_fft_order",
    "phi_hz",
    "phi_squared_normalization",
]


def nu_d(x: Array, d: int = 4) -> Array:
    """Regularized incomplete beta function $I_x(d, d)$.

    This is the smooth ramp used in the Meyer rolloff:
    $\\nu_d(0) = 0$, $\\nu_d(1) = 1$, and $\\nu_d$ is $C^{d-1}$ smooth.

    Parameters
    ----------
    x : ndarray
        Input values; clipped to ``[0, 1]`` internally.
    d : int, optional
        Smoothness order (default 4).

    Returns
    -------
    ndarray
        $I_x(d, d)$ evaluated element-wise.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.filter.nu_d(np.array([0.0, 0.5, 1.0]))
    array([0. , 0.5, 1. ])
    """
    if not isinstance(d, (int, np.integer)) or d < 1:
        # betainc(0, 0, x) is NaN, so a bad d silently yields a NaN filter.
        raise ValueError(f"d must be an integer >= 1, got {d!r}")
    return betainc(d, d, np.clip(x, 0.0, 1.0))


def phi_centered(Nf: int, Nt: int, dt: float, d: int = 4) -> Array:
    """Meyer filter $\\hat\\phi(\\omega)$ on a length-$N_t$ frequency grid,
    centered at index $N_t/2$.

    The grid spans frequencies $f \\in [-N_t/2,\\,N_t/2-1]/T$ with
    $T = N_f N_t \\delta t$.  In the flat region ($|f| < \\Delta F/4$)
    the value is $1/\\sqrt{\\Delta\\Omega}$; in the rolloff region
    ($\\Delta F/4 \\le |f| < 3\\Delta F/4$) it follows the Meyer
    cosine taper; otherwise it is zero.

    Parameters
    ----------
    Nf : int
        Number of frequency bands (sets $\\Delta F$).
    Nt : int
        Number of time bins (sets the frequency grid length).
    dt : float
        Sampling interval.
    d : int, optional
        Smoothness order for the rolloff (default 4).

    Returns
    -------
    ndarray, shape (Nt,)
        Filter samples in centered ordering (peak at index $N_t/2$).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> phi = wdm.phi_centered(8, 16, 1.0)
    >>> phi.shape, bool(phi[8] == phi.max())   # flat top includes index Nt/2
    ((16,), True)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    T = Nf * Nt * dt
    f = (np.arange(Nt) - Nt // 2) / T
    delta_F = 1.0 / (2.0 * dt * Nf)
    delta_Omega = 2.0 * np.pi * delta_F
    A = delta_F / 4.0
    B = delta_F / 2.0
    amp = 1.0 / np.sqrt(delta_Omega)

    f_abs = np.abs(f)
    out = np.where(f_abs < A, amp, 0.0)
    rolloff = (f_abs >= A) & (f_abs < A + B)
    out = np.where(
        rolloff,
        amp * np.cos(np.pi / 2.0 * nu_d((f_abs - A) / B, d)),
        out,
    )
    return out


def phi_hz(Nf: int, Nt: int, dt: float, d: int = 4) -> Array:
    """The Paper I (eq:Meyer-def, Hz-normalised) window: flat value $1/\\sqrt{\\Delta F}$.

    Equal to $\\sqrt{2\\pi}$ times :func:`phi_centered` (the package's canonical
    $\\omega$-normalised window) to rounding; computed with the June
    ``wdm/filter.py`` arithmetic, so it is bit-identical to the window the
    Paper I literal algorithms (:mod:`wdm.paper`) were written against
    (whose call ``phi_centered(Nt, Nf, dt)`` had the two integers swapped).
    Paired with $\\kappa = \\sqrt2/N_f$ it gives the same coefficients as
    :func:`phi_centered` with $\\kappa = 2\\sqrt\\pi/N_f$.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> h, w = wdm.filter.phi_hz(8, 16, 1.0), wdm.phi_centered(8, 16, 1.0)
    >>> bool(np.allclose(h, np.sqrt(2 * np.pi) * w, rtol=1e-15, atol=0))
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    T = Nf * Nt * dt
    f = (np.arange(Nt) - Nt // 2) / T
    delta_F = 1.0 / (2.0 * dt * Nf)
    A = delta_F / 4.0
    B = delta_F / 2.0
    amp = 1.0 / np.sqrt(delta_F)

    f_abs = np.abs(f)
    out = np.where(f_abs < A, amp, 0.0)
    rolloff = (f_abs >= A) & (f_abs < A + B)
    out = np.where(
        rolloff,
        amp * np.cos(np.pi / 2.0 * nu_d((f_abs - A) / B, d)),
        out,
    )
    return out


def phi_fft_order(Nf: int, Nt: int, dt: float, d: int = 4) -> Array:
    """Meyer filter in FFT-natural ordering (peak at index 0).

    This is the same filter as :func:`phi_centered` but with
    ``np.fft.ifftshift`` applied so it is ready for direct multiplication
    with the output of ``np.fft.ifft`` (which expects FFT-natural order).
    The reindexing absorbs the $(-1)^n$ factor that would otherwise
    appear in the forward transform.

    Parameters
    ----------
    Nf, Nt, dt, d
        See :func:`phi_centered`.

    Returns
    -------
    ndarray, shape (Nt,)
        Filter samples in FFT-natural ordering.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> phi = wdm.phi_fft_order(8, 16, 1.0)
    >>> int(np.argmax(phi))   # peak moved to index 0
    0
    """
    return np.fft.ifftshift(phi_centered(Nf, Nt, dt, d))


def phi_squared_normalization(Nf: int, Nt: int, dt: float, d: int = 4) -> float:
    """Sum $\\sum_k \\hat\\phi_k^2$ for diagnostic checks.

    Returned as-is, for diagnostics only.

    .. warning::
       This does **not** equal $1/\\Delta\\Omega$: on the discrete grid
       $\\sum_k \\hat\\phi_k^2 \\cdot \\Delta\\Omega = N_t/2$, not 1.
       The partition-of-unity property constrains overlapping shifted
       copies of $\\hat\\phi^2$, not the raw sum of one copy.

    Parameters
    ----------
    Nf, Nt, dt, d
        See :func:`phi_centered`.

    Returns
    -------
    float
        Sum of $\\hat\\phi^2$ over the grid.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> total = wdm.filter.phi_squared_normalization(8, 16, 1.0)
    >>> bool(np.isclose(total, np.sum(wdm.phi_centered(8, 16, 1.0) ** 2)))
    True
    """
    return float(np.sum(phi_centered(Nf, Nt, dt, d) ** 2))
