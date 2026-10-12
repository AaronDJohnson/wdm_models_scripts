"""Dense oracles for tests and small grids (spec section 4; NumPy only, never on a
likelihood path).  Everything here is ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.

* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a cell set
  (``wdm.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a cell set by the direct sum over rfft bins
  (``wdm.evolutionary.eq45_direct``).
* :func:`blocktri_dense`: a ragged block-tridiagonal matrix assembled densely.
* :func:`hhat`, :func:`eq54_covariance`, :func:`dense_lambda_generic`: the generic
  Eq. (54) covariance of a modulated stationary process on Fourier bins and its WDM
  form (GB ``wdm_full.hhat_at`` / ``dense_lambda_generic``, in the dense form that
  GB ``tests/test_wdm_full.py::generic_M`` pins it to).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import wdm
from wdm.evolutionary import eq45_direct, eq45_plan

from .domains import CellSet

__all__ = [
    "blocktri_dense",
    "dense_W",
    "dense_lambda_generic",
    "eq45_dense",
    "eq54_covariance",
    "hhat",
    "wdm_congruence",
]

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def dense_W(cells: CellSet) -> Array:
    """``(cells.size, N)``: row ``p`` is cell ``p``'s atom sampled in time, so ``W x`` are
    the cells of ``x`` (``wdm.matrix.active_W`` rows, ``W_a W_a^T = dt I``)."""
    g = cells.grid
    return wdm.matrix.active_W(g.Nf, g.Nt, g.dt)[cells.packed_positions()]


def wdm_congruence(Sigma: Array, cells: CellSet) -> Array:
    """``W_K Sigma W_K^T`` for a time-domain covariance ``Sigma (N, N)``."""
    W = dense_W(cells)
    return W @ np.asarray(Sigma, dtype=float) @ W.T


def eq45_dense(cells: CellSet, Sn: Callable[[Any, Any], Any], *, P: int = 24,
               t_max: float | None = None) -> Array:
    """Eq. (45) on ``cells`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n`` inside)."""
    g = cells.grid
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max)
    return eq45_direct(plan, Sn)


def blocktri_dense(diag: Sequence[Array], up: Sequence[Array | None]) -> Array:
    """The symmetric matrix with diagonal blocks ``diag`` and upper blocks ``up``."""
    sizes = [np.asarray(d).shape[0] for d in diag]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    C = np.zeros((off[-1], off[-1]))
    for i, D in enumerate(diag):
        C[off[i]:off[i + 1], off[i]:off[i + 1]] = D
    for i, U in enumerate(up):
        if U is not None:
            C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = U
            C[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = np.asarray(U).T
    return C


def hhat(mu: Array, basis_L: Array, lags: NDArray[np.int64]) -> CArray:
    r""":math:`\hat h[q] = \frac1L \sum_t \sqrt{g(t)}\,e^{-2\pi i q t/L}` at integer ``lags``,
    ``g = exp(mu @ basis_L)`` on the ``L``-point decimation grid (GB ``wdm_full.hhat_at``)."""
    L = basis_L.shape[1]
    R = np.fft.rfft(np.exp(0.5 * (np.asarray(mu) @ basis_L))) / L
    lags = np.asarray(lags)
    out = R[np.abs(lags)]
    return np.where(lags < 0, np.conj(out), out)


def eq54_covariance(s_I: Array, s_G_src: Array, h: CArray, *, margin: int) -> CArray:
    r"""Eq. (54) on ``G`` grid bins: :math:`M = \mathrm{diag}(s_I) + A A^\dagger`,
    :math:`A_{j\kappa} = \sqrt{s_G(\kappa)}\,\hat h[j - \kappa]` over the source columns
    :math:`\kappa = -\mathrm{margin} \ldots G + \mathrm{margin} - 1` (grid units).

    ``s_G_src`` has ``G + 2 margin`` entries; ``h`` is :math:`\hat h` at the lags
    ``-(G + margin) .. G + margin``.  GB ``tests/test_wdm_full.py::generic_M`` verbatim.
    """
    s_I = np.asarray(s_I, dtype=float)
    G = s_I.size
    s_G_src = np.asarray(s_G_src, dtype=float)
    if s_G_src.size != G + 2 * margin:
        raise ValueError(f"s_G_src needs G + 2 margin = {G + 2 * margin} entries; got {s_G_src.size}")
    h = np.asarray(h)
    if h.size != 2 * (G + margin) + 1:
        raise ValueError(f"h needs the {2 * (G + margin) + 1} lags -(G+margin)..G+margin; got {h.size}")
    j = np.arange(G)[:, None]
    k = np.arange(-margin, G + margin)[None, :]
    A = np.sqrt(s_G_src)[None, :] * h[(j - k) + (G + margin)]
    return np.diag(s_I).astype(complex) + A @ A.conj().T


def dense_lambda_generic(U: CArray, s_I: Array, s_G_src: Array, h: CArray, *,
                         margin: int) -> Array:
    r""":math:`\Lambda = \tfrac12\mathrm{Re}(U M U^\dagger)` with ``M`` from
    :func:`eq54_covariance` and ``U`` the unit-norm atoms on the ``G`` grid bins
    (GB ``wdm_full.dense_lambda_generic``, equal to this dense form to 1e-12)."""
    U = np.asarray(U)
    M = eq54_covariance(s_I, s_G_src, h, margin=margin)
    return 0.5 * np.real(U @ M @ U.conj().T)
