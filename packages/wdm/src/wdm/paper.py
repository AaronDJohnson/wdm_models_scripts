"""Literal transcriptions of the four Paper I appendix algorithms (NumPy, reference).

Each function mirrors the pseudocode of WDM Paper I (``apssamp.tex``) line by
line -- Python loops, no vectorisation -- so a reader can check the paper
against running code.  They are the *reference*; the fast, dual-backend
implementations are:

==========================  ==============================  =====================================
Paper I                     literal (this module)           fast (production)
==========================  ==============================  =====================================
Alg 1 (alg:fwd)             :func:`alg1_forward`            :func:`wdm.forward_time`
Alg 2 (alg:inv)             :func:`alg2_inverse`            :func:`wdm.inverse_time` (masked)
Alg 3 + tab:K_recipes       :func:`alg3_lambda_from_psd`,   :func:`wdm.stationary.build_lambda_blocks_full_active`
                            :data:`K_RECIPES`
Alg 4 (alg:psd_backward)    :func:`alg4_psd_from_lambda`    :func:`wdm.stationary.psd_from_lambda_blocks`,
                                                            :func:`wdm.stationary.psd_from_K`
==========================  ==============================  =====================================

Transcribed from the June ``wdm/paper_algorithms.py`` with two adaptations:

* every ``phi_centered(Nt, Nf, dt)`` call (June signature, Hz window) is
  :func:`wdm.filter.phi_hz` ``(Nf, Nt, dt)`` -- bit-identical values;
* the manuscript constants are used throughout: kappa = sqrt(2)/Nf,
  lambda = 1/(sqrt(2) dt), mu = 1/N and sigma^2 = S/(2 dt).

Alg 2 is the algorithm **as printed**: it does not mask the parity-killed
edge pixels, so junk there leaks into ``x`` (``tests/test_paper.py::
test_alg2_leak``).  :func:`wdm.inverse` masks them and is the exact adjoint.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .filter import phi_hz
from .grid import n_samples

type Array = NDArray[np.floating]
type CArray = NDArray[np.complexfloating]

__all__ = [
    "K_RECIPES",
    "alg1_forward",
    "alg2_inverse",
    "alg3_lambda_from_psd",
    "alg4_psd_from_lambda",
]


def _C(m: int, n: int | np.integer) -> complex:
    """Wilson phase ``C_{mn} = i^{(m+n) mod 2}`` (1 if (m+n) even, i if odd)."""
    return 1j if (m + int(n)) % 2 else (1.0 + 0j)


def _tphi(phi_c: Array, j: int, Nt: int) -> float:
    """Math window ``tilde_phi[j]`` = ``phi_c[j + Nt/2]`` (0 out of range)."""
    idx = j + Nt // 2
    if 0 <= idx < Nt:
        return phi_c[idx]
    return 0.0


def _active(m: int, Nf: int, Nt: int) -> NDArray[np.integer]:
    """Active-n list: edges keep (m+n) even (Nt/2 pixels), interior is all Nt."""
    if m in (0, Nf):
        return np.array([n for n in range(Nt) if (m + n) % 2 == 0])
    return np.arange(Nt)


# ---------------------------------------------------------------------------
# Algorithm alg:fwd -- Forward WDM: time series x -> WDM grid w.
# ---------------------------------------------------------------------------
def alg1_forward(x: Array, Nf: int, Nt: int, dt: float) -> Array:
    """Forward WDM transform, literal Algorithm alg:fwd.  Inactive pixels are exactly 0."""
    N = n_samples(Nf, Nt)
    X = np.fft.rfft(x)
    kappa = np.sqrt(2.0) / Nf
    phi_c = phi_hz(Nf, Nt, dt)
    w = np.zeros((Nf + 1, Nt))

    for m in range(Nf + 1):
        u = np.zeros(Nt, dtype=complex)
        for k in range(Nt):
            ell = k + (m - 1) * (Nt // 2)
            if ell < 0:
                z = np.conj(X[-ell])
            elif ell > N // 2:
                z = np.conj(X[N - ell])
            else:
                z = X[ell]
            u[k] = z * _tphi(phi_c, k - Nt // 2, Nt)  # tilde_phi[k - Nt/2]
        y = np.fft.ifft(u)
        for n in range(Nt):
            if m in (0, Nf) and (m + n) % 2 != 0:
                w[m, n] = 0.0
                continue
            sign = -1.0 if ((m + 1) * n) % 2 else 1.0
            w[m, n] = kappa * sign * np.real(np.conj(_C(m, n)) * y[n])

    w[0, :] /= np.sqrt(2.0)
    w[Nf, :] /= np.sqrt(2.0)
    return w


# ---------------------------------------------------------------------------
# Algorithm alg:inv -- Inverse WDM: WDM grid w -> real time series x.
# ---------------------------------------------------------------------------
def alg2_inverse(w: Array, Nf: int, Nt: int, dt: float) -> Array:
    """Inverse WDM transform, literal Algorithm alg:inv (as printed: unmasked)."""
    N = n_samples(Nf, Nt)
    wp = np.array(w, dtype=float, copy=True)
    wp[0, :] *= np.sqrt(2.0)
    wp[Nf, :] *= np.sqrt(2.0)
    lam = 1.0 / (np.sqrt(2.0) * dt)
    phi_c = phi_hz(Nf, Nt, dt)
    X = np.zeros(N // 2 + 1, dtype=complex)

    for m in range(Nf + 1):
        g = np.zeros(Nt, dtype=complex)
        for n in range(Nt):
            sign = -1.0 if ((m + 1) * n) % 2 else 1.0
            g[n] = wp[m, n] * _C(m, n) * sign
        Wm = np.fft.fft(g)
        for k in range(Nt):
            ell = k + (m - 1) * (Nt // 2)
            v = lam * _tphi(phi_c, k - Nt // 2, Nt) * Wm[k]
            if ell < 0 or ell > N // 2:
                continue
            X[ell] += v

    return np.fft.irfft(X, n=N)


# ---------------------------------------------------------------------------
# Table tab:K_recipes -- one function per row.
# ---------------------------------------------------------------------------
def _k_interior_diag(m: int, sigma2: Array, phi_c: Array, Nf: int, Nt: int) -> CArray:
    """Row ``(m, m)``, ``1 <= m <= Nf-1``."""
    half = Nt // 2
    K = np.zeros(Nt, dtype=complex)
    for k in range(0, Nt):
        K[k] = _tphi(phi_c, k - half, Nt) ** 2 * sigma2[k + (m - 1) * half]
    return K


def _k_interior_offdiag(m: int, sigma2: Array, phi_c: Array, Nf: int, Nt: int) -> CArray:
    """Row ``(m, m+1)``, ``1 <= m <= Nf-2``."""
    half = Nt // 2
    K = np.zeros(Nt, dtype=complex)
    for k in range(half, Nt):
        K[k] = (_tphi(phi_c, k - half, Nt) * _tphi(phi_c, k - Nt, Nt)
                * sigma2[k + (m - 1) * half])
    return K


def _hermitian_extend(K: CArray, Nt: int) -> CArray:
    """Edge diagonals: ``K[k] = K[Nt - k]`` for ``k > Nt/2``."""
    for k in range(Nt // 2 + 1, Nt):
        K[k] = K[Nt - k]
    return K


def _k_edge_diag_dc(m: int, sigma2: Array, phi_c: Array, Nf: int, Nt: int) -> CArray:
    """Row ``(0, 0)``: half slab below DC, Hermitian-extended."""
    half = Nt // 2
    K = np.zeros(Nt, dtype=complex)
    for k in range(0, half):
        K[k] = _tphi(phi_c, k, Nt) ** 2 * sigma2[k]
    return _hermitian_extend(K, Nt)


def _k_edge_diag_nyquist(m: int, sigma2: Array, phi_c: Array, Nf: int, Nt: int) -> CArray:
    """Row ``(Nf, Nf)``: half slab up to Nyquist inclusive, Hermitian-extended."""
    half = Nt // 2
    N = n_samples(Nf, Nt)
    K = np.zeros(Nt, dtype=complex)
    for k in range(0, half + 1):
        K[k] = _tphi(phi_c, k - half, Nt) ** 2 * sigma2[k + N // 2 - half]
    return _hermitian_extend(K, Nt)


def _k_edge_cross_dc(m: int, sigma2: Array, phi_c: Array, Nf: int, Nt: int) -> CArray:
    """Row ``(0, 1)``: carries the unit-norm edge sqrt(2)."""
    half = Nt // 2
    K = np.zeros(Nt, dtype=complex)
    for k in range(half, Nt):
        K[k] = (np.sqrt(2.0) * _tphi(phi_c, k - half, Nt) * _tphi(phi_c, k - Nt, Nt)
                * sigma2[k - half])
    return K


def _k_edge_cross_nyquist(m: int, sigma2: Array, phi_c: Array, Nf: int, Nt: int) -> CArray:
    """Row ``(Nf-1, Nf)``: carries the unit-norm edge sqrt(2)."""
    half = Nt // 2
    N = n_samples(Nf, Nt)
    K = np.zeros(Nt, dtype=complex)
    for k in range(half, Nt):
        K[k] = (np.sqrt(2.0) * _tphi(phi_c, k - half, Nt) * _tphi(phi_c, k - Nt, Nt)
                * sigma2[k + N // 2 - Nt])
    return K


#: Table tab:K_recipes: row name -> ``f(m, sigma2, phi_c, Nf, Nt) -> K~`` (length Nt).
K_RECIPES: dict[str, Callable[[int, Array, Array, int, int], CArray]] = {
    "interior_diag": _k_interior_diag,
    "interior_offdiag": _k_interior_offdiag,
    "edge_diag_dc": _k_edge_diag_dc,
    "edge_diag_nyquist": _k_edge_diag_nyquist,
    "edge_cross_dc": _k_edge_cross_dc,
    "edge_cross_nyquist": _k_edge_cross_nyquist,
}


def _recipe(m: int, mp: int, Nf: int) -> str:
    """The tab:K_recipes row of the band pair ``(m, mp)``."""
    if mp == m and 1 <= m <= Nf - 1:
        return "interior_diag"
    if mp == m + 1 and 1 <= m <= Nf - 2:
        return "interior_offdiag"
    if (m, mp) == (0, 0):
        return "edge_diag_dc"
    if (m, mp) == (Nf, Nf):
        return "edge_diag_nyquist"
    if (m, mp) == (0, 1):
        return "edge_cross_dc"
    if (m, mp) == (Nf - 1, Nf):
        return "edge_cross_nyquist"
    raise ValueError(f"no tab:K_recipes row for the band pair ({m}, {mp})")


# ---------------------------------------------------------------------------
# Algorithm alg:stat + Table tab:K_recipes -- Stationary forward: S -> Lambda.
# ---------------------------------------------------------------------------
def alg3_lambda_from_psd(S: Array, Nf: int, Nt: int, dt: float
                         ) -> tuple[list[Array], list[Array], list[NDArray[np.integer]]]:
    """Stationary forward ``S -> Lambda``, literal alg:stat + tab:K_recipes.

    ``S`` is the one-sided PSD on the ``rfftfreq(N, dt)`` grid.  Returns
    ``(diag_blocks, offd_blocks, n_active)`` in the layout of
    :func:`wdm.stationary.build_lambda_blocks_full_active` (edge blocks restricted
    to their active ``Nt/2`` sub-block).
    """
    S = np.asarray(S, dtype=float)
    N = n_samples(Nf, Nt)
    sigma2 = S / (2.0 * dt)
    mu = 1.0 / N
    phi_c = phi_hz(Nf, Nt, dt)
    n_active = [_active(m, Nf, Nt) for m in range(Nf + 1)]

    def build_block(m: int, mp: int) -> Array:
        K = K_RECIPES[_recipe(m, mp, Nf)](m, sigma2, phi_c, Nf, Nt)
        Kr = Nt * np.fft.ifft(K)  # band lag kernel K[r]
        full = np.zeros((Nt, Nt))
        for n in range(Nt):
            for npr in range(Nt):
                r = (n - npr) % Nt
                gamma = np.conj(_C(m, n)) * _C(mp, npr)
                s = -1.0 if ((m + 1) * (n + npr)) % 2 else 1.0
                val = mu * s * np.real(gamma * Kr[r])
                if m in (0, Nf) and (m + n) % 2 != 0:
                    val = 0.0
                elif mp in (0, Nf) and (mp + npr) % 2 != 0:
                    val = 0.0
                full[n, npr] = val
        nl, nr = n_active[m], n_active[mp]
        return full[np.ix_(nl, nr)]

    diag_blocks = [build_block(m, m) for m in range(Nf + 1)]
    offd_blocks = [build_block(m, m + 1) for m in range(Nf)]
    return diag_blocks, offd_blocks, n_active


# ---------------------------------------------------------------------------
# Algorithm alg:psd_backward -- Stationary inverse: Lambda -> S.
# ---------------------------------------------------------------------------
def alg4_psd_from_lambda(diag: list[Array], offd: list[Array],
                         n_active: list[NDArray[np.integer]], Nf: int, Nt: int,
                         dt: float) -> Array:
    """Stationary inverse ``Lambda -> S`` (one-sided, rfft grid), literal alg:psd_backward.

    The band-centring modulation ``g[r] = (-1)^{(m+1) r} t[r]`` is applied
    inside the FFT and the result read at the band-local index ``D[k]``; this
    is exact because ``(m-1) Nt/2 == (m+1) Nt/2 (mod Nt)``.
    """
    N = n_samples(Nf, Nt)
    nk = N // 2 + 1
    half = Nt // 2
    T = N * dt
    phi_c = phi_hz(Nf, Nt, dt)

    def Atilde(m: int) -> float:
        return 1.0 if m in (0, Nf) else 1.0 / np.sqrt(2.0)

    def loc(m: int) -> dict[int, int]:  # active-n value -> position in the stored block
        return {int(n): i for i, n in enumerate(n_active[m])}

    S = np.zeros(nk)
    pairs = [(m, m) for m in range(Nf + 1)] + [(m, m + 1) for m in range(Nf)]
    blocks = {(m, m): diag[m] for m in range(Nf + 1)}
    blocks.update({(m, m + 1): offd[m] for m in range(Nf)})

    for (m, mp) in pairs:
        block = blocks[(m, mp)]
        nl, nr = n_active[m], n_active[mp]
        ll, lr = loc(m), loc(mp)
        w = 1.0 if mp == m else 2.0
        both_interior = (m not in (0, Nf)) and (mp not in (0, Nf))
        mu = float(Nt) if both_interior else Nt / 2.0

        # carrier-stripped lag kernel t[r]
        t = np.zeros(Nt, dtype=complex)
        nl_set = {int(x) for x in nl}
        for r in range(Nt):
            # pick an active pair (n, n') with (n - n') mod Nt = r
            found = False
            for npr in nr:
                n = (int(npr) + r) % Nt
                if n in nl_set:
                    t[r] = (mu * _C(m, n) * np.conj(_C(mp, npr))
                            * block[ll[n], lr[int(npr)]])
                    found = True
                    break
            if not found:
                t[r] = 0.0

        g = np.array([(-1.0 if ((m + 1) * r) % 2 else 1.0) * t[r] for r in range(Nt)])
        D = np.fft.fft(g)
        c = (2.0 * w / T) * Atilde(m) * Atilde(mp)

        for k in range(Nt):
            ell = k + (m - 1) * half
            if ell < 0 or ell > N // 2:
                continue
            W = _tphi(phi_c, k - half, Nt) * _tphi(phi_c, k - (mp - m + 1) * half, Nt)
            S[ell] += c * W * np.real(D[k])  # band-local index, per alg L2216
    return S
