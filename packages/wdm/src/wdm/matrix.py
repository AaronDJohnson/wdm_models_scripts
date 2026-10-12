"""Explicit WDM transform matrix and noise covariance.

The WDM forward transform is a linear map $w = W\\,x$ where $W$ is a
real $(N_f+1)N_t \\times N$ matrix.  Restricted to its $N$ active rows
(skipping the inactive edge-band entries), $W_a$ is square and
orthogonal **up to the sampling scale**:

$$W_a W_a^T = W_a^T W_a = \\delta t\\,I,$$

i.e. $W_a = \\sqrt{\\delta t}\\,Q$ with $Q$ orthogonal.  Invert with
$x = W_a^T w / \\delta t$.  The $\\delta t$ is the same factor that
appears in Parseval ($\\sum w^2 = \\delta t \\sum x^2$); $W_a$ is
exactly orthogonal only when $\\delta t = 1$.

This module exposes:

* :func:`build_W` — full $(N_f+1)N_t \\times N$ matrix built by
  time-domain impulse-response probing.
* :func:`active_W` — the $N \\times N$ restriction ($W_a W_a^T = \\delta t\\,I$).
* :func:`build_C_time` — Toeplitz time-domain noise covariance from
  a PSD (vectorized via ``irfft`` of the power spectrum).
* :func:`build_C_wdm` — WDM-domain noise covariance via the similarity
  transform $C^{(\\text{WDM})} = W\\,C^{(\\text{time})}\\,W^T$.

These functions are $O(N^2)$ in memory and $O(N^3)$ in compute.
They are intended for verification, teaching, and small-$N$ work
(roughly $N \\lesssim 2048$); for production use the FFT-based
transforms in :mod:`wdm.transform`.

Two utilities (:func:`rfft_to_real`, :func:`real_to_rfft`) pack the
complex rfft into a real vector of length $N$ for users who want to
work with a single $N \\times N$ matrix in the real-DFT
representation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .grid import active_mask, n_samples, sanitize_psd, validate_dt, validate_grid
from .transform import inverse_time

type Array = NDArray[np.floating]
type CArray = NDArray[np.complexfloating]
type PsdFunc = Callable[[Array], Array]

__all__ = [
    "active_W",
    "build_C_time",
    "build_C_wdm",
    "build_C_wdm_active",
    "build_W",
    "real_to_rfft",
    "rfft_to_real",
]


def build_W(
    Nf: int,
    Nt: int,
    dt: float,
) -> Array:
    """Build the full $(N_f+1)N_t \\times N$ WDM transform matrix.

    Row $\\alpha = (m, n)$ is the basis function $g_{mn}$ sampled in time, so
    $W$ is *defined* by "column $k_0$ is the WDM transform of a delta at sample
    $k_0$".  It is *built* from the lattice's shift-covariance instead, which
    is several times faster and agrees with the definition to 4e-16 (pinned by
    ``tests/test_vendored_behaviour.py``).

    Parameters
    ----------
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, shape ((Nf+1)*Nt, N)
        Real matrix.  Rows are indexed by the flattened ``(m, n)``
        pixel index $\\alpha = m\\,N_t + n$; columns by the time
        sample $k_0 = 0, \\ldots, N - 1$.

    Notes
    -----
    Cost: $2(N_f+1)$ inverse transforms plus $O(N^2)$ of copying, and
    $O(N^2)$ memory -- several times faster than the $N$-impulse definition,
    the gap widening with $N$.  Run ``python tools/benchmark.py`` for the
    figure on your machine.  Still only practical for small $N$.  Inactive
    edge-band rows are exactly zero.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> W = wdm.matrix.build_W(4, 8, 1.0)
    >>> W.shape
    (40, 32)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    W = np.zeros((Nf + 1, Nt, N))

    # Since W_a^T W_a = dt*I, row (m, n) is exactly dt * inverse(e_mn).  The
    # lattice is shift-covariant in steps of TWO time pixels -- one step flips
    # the carrier parity (-1)^{(m+1)n} -- so rows n and n+2 differ only by a
    # circular shift of 2*Nf samples.  Probing n = 0 and n = 1 per band and
    # rolling the rest turns N transforms into 2(Nf+1).
    #
    # Only ACTIVE pixels are seeded: inactive edge pixels correspond to zero
    # rows of W, and those rows come out exactly zero here rather than carrying
    # the ~1e-17 of cancellation noise that impulse probing leaves.
    #
    # The identity holds for the unit-norm basis, which is the only one there is.
    mask = active_mask(Nf, Nt)
    seed = np.zeros((Nf + 1, Nt))
    for n0 in (0, 1):
        for m in range(Nf + 1):
            if not mask[m, n0]:
                continue
            seed[:] = 0.0
            seed[m, n0] = 1.0
            row = dt * inverse_time(seed, Nf, Nt, dt)
            for n in range(n0, Nt, 2):
                W[m, n] = np.roll(row, (n - n0) * Nf)

    return W.reshape((Nf + 1) * Nt, N)


def active_W(
    Nf: int,
    Nt: int,
    dt: float,
) -> Array:
    """Restrict :func:`build_W` to its $N$ active rows.

    Returns the square $N \\times N$ matrix obtained by keeping only
    the rows corresponding to active pixels (interior bands and the
    parity-selected edge-band entries).  This matrix is orthogonal up to
    the sampling scale:
    $W_a W_a^T = W_a^T W_a = \\delta t\\,I$.  Invert with
    $x = W_a^T w / \\delta t$; the plain transpose is the inverse only
    at $\\delta t = 1$.

    Parameters
    ----------
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, shape (N, N)
        Square real matrix satisfying $W_a W_a^T = \\delta t\\,I$.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Wa = wdm.matrix.active_W(4, 8, 1.0)
    >>> Wa.shape, bool(np.allclose(Wa @ Wa.T, 1.0 * np.eye(32)))
    ((32, 32), True)
    """
    W = build_W(Nf, Nt, dt)
    mask = active_mask(Nf, Nt).ravel()
    return W[mask, :]


def build_C_time(N: int, dt: float, Sn_func: PsdFunc) -> Array:
    """Toeplitz time-domain noise covariance from a one-sided PSD.

    For stationary noise with one-sided PSD $S_n(f)$, the autocovariance
    is $R(\\tau) = E[n_k n_{k+\\tau}]$, and the covariance matrix is
    Toeplitz: $C[k, k'] = R(|k - k'|)$.  Built via the inverse rfft
    of the discrete two-sided power spectrum.

    Parameters
    ----------
    N : int
        Number of time samples.
    dt : float
        Sampling interval.
    Sn_func : callable
        ``f -> S_n(f)``.

    Returns
    -------
    ndarray, shape (N, N)
        Real symmetric Toeplitz covariance matrix.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> C = wdm.matrix.build_C_time(32, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> C.shape, bool(np.allclose(C, C.T))
    ((32, 32), True)
    """
    validate_dt(dt)
    if int(N) <= 0 or int(N) % 2 != 0:
        raise ValueError(f"N must be a positive even integer, got {N}")
    freqs = np.fft.rfftfreq(N, d=dt)
    # Force double. `sanitize_psd` preserves single precision so that a
    # float32 PSD yields a float32 *variance grid* (see wdm.transform's
    # Precision section), but this module is the exact-reference tier: an
    # (N, N) covariance is what everything else is verified against, and it is
    # O(N^2) in memory whatever the dtype, so there is nothing to win and a
    # reference to lose. `stationary` forces double for the same reason.
    Sn = np.asarray(sanitize_psd(Sn_func(freqs), "build_C_time"), dtype=float)
    sigma2 = N * Sn / (2.0 * dt)  # E[|X_j|^2]

    # Autocovariance R[tau] = (1/N^2) * sum_j (full sigma2_j) e^{2 pi i j tau / N}
    # Two-sided: sigma2_j for j=0,...,N/2 then mirror j=N-j' for j' = 1,...,N/2-1.
    # For the rfft-based inverse, build a "spectrum" g[j] s.t. irfft(g, N)
    # gives us R[tau] up to a normalization.
    # irfft(g, N)[tau] = (1/N) [g[0] + g[N/2] (-1)^tau
    #                          + 2 Re sum_{j=1}^{N/2-1} g[j] e^{2pi i j tau/N}]
    # We want R[tau] = (1/N^2) [sigma2[0] + sigma2[N/2] (-1)^tau
    #                            + 2 sum_{j=1}^{N/2-1} sigma2[j] cos(2pi j tau/N)].
    # So g[j] = sigma2[j] / N (real-valued) yields R = irfft(g, N).
    g = sigma2 / N
    R = np.fft.irfft(g, n=N)

    # Build the Toeplitz matrix; use circular distance to match the discrete
    # autocovariance produced by the irfft (periodic).
    idx = np.arange(N)
    tau = np.abs(idx[:, None] - idx[None, :])
    tau = np.minimum(tau, N - tau)
    return R[tau]


def build_C_wdm(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
) -> Array:
    """WDM-domain covariance via the similarity transform $W\\,C^{(\\text{time})}\\,W^T$.

    Returns the full $(N_f+1)N_t \\times (N_f+1)N_t$ covariance matrix
    of the WDM coefficients under stationary noise with PSD $S_n(f)$.
    Block-tridiagonal in the band index $m$: only $|m - m'| \\le 1$
    blocks are nonzero (up to numerical noise).

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable

    Returns
    -------
    ndarray, shape ((Nf+1)*Nt, (Nf+1)*Nt)
        Symmetric real covariance matrix.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> C = wdm.matrix.build_C_wdm(4, 8, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> C.shape
    (40, 40)
    """
    N = n_samples(Nf, Nt)
    W = build_W(Nf, Nt, dt)
    C_time = build_C_time(N, dt, Sn_func)
    return W @ C_time @ W.T


def build_C_wdm_active(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
) -> Array:
    """Active-pixel restriction of :func:`build_C_wdm`.

    Returns the $N \\times N$ covariance of the active WDM coefficients
    only.  Equivalent to ``build_C_wdm`` followed by selecting the
    active rows and columns.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable

    Returns
    -------
    ndarray, shape (N, N)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> C = wdm.matrix.build_C_wdm_active(4, 8, 1.0, flat)
    >>> C.shape, float(np.round(np.diag(C)[0], 6))   # ~ S_n / 2
    ((32, 32), 0.0005)
    """
    N = n_samples(Nf, Nt)
    Wa = active_W(Nf, Nt, dt)
    C_time = build_C_time(N, dt, Sn_func)
    return Wa @ C_time @ Wa.T


def rfft_to_real(x_rfft: CArray, N: int) -> Array:
    """Pack a complex rfft into a real vector of length $N$.

    The packing is

    $$\\bm{x}_R = (\\text{Re}\\,X_0,\\,\\text{Re}\\,X_1,\\,\\text{Im}\\,X_1,
    \\,\\ldots,\\,\\text{Re}\\,X_{N/2-1},\\,\\text{Im}\\,X_{N/2-1},
    \\,\\text{Re}\\,X_{N/2})^T,$$

    yielding $1 + 2(N/2 - 1) + 1 = N$ entries.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (N//2+1,)
    N : int
        Total time-domain sample count.

    Returns
    -------
    ndarray, float, shape (N,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> x = np.random.default_rng(0).standard_normal(32)
    >>> packed = wdm.matrix.rfft_to_real(np.fft.rfft(x), 32)
    >>> packed.shape
    (32,)
    """
    # Interleave Re/Im of bins j = 1..N/2-1 into slots (2j-1, 2j), i.e. the
    # strided ranges 1:N-2:2 and 2:N-1:2.
    x_rfft = np.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape ({N // 2 + 1},) for N={N}; got {x_rfft.shape}"
        )
    xf = np.zeros(N)
    interior = x_rfft[1 : N // 2]
    xf[0] = x_rfft[0].real
    xf[1 : N - 2 : 2] = interior.real
    xf[2 : N - 1 : 2] = interior.imag
    xf[N - 1] = x_rfft[N // 2].real
    return xf


def real_to_rfft(x_real: Array, N: int) -> CArray:
    """Inverse of :func:`rfft_to_real`.

    Parameters
    ----------
    x_real : ndarray, float, shape (N,)
    N : int

    Returns
    -------
    ndarray, complex, shape (N//2+1,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> x = np.random.default_rng(0).standard_normal(32)
    >>> packed = wdm.matrix.rfft_to_real(np.fft.rfft(x), 32)
    >>> back = wdm.matrix.real_to_rfft(packed, 32)
    >>> bool(np.allclose(back, np.fft.rfft(x)))
    True
    """
    x_real = np.asarray(x_real)
    if x_real.shape != (N,):
        raise ValueError(f"x_real must have shape ({N},); got {x_real.shape}")
    n_rfft = N // 2 + 1
    out = np.zeros(n_rfft, dtype=complex)
    out[0] = x_real[0]
    out[1 : N // 2] = x_real[1 : N - 2 : 2] + 1j * x_real[2 : N - 1 : 2]
    out[N // 2] = x_real[N - 1]
    return out
