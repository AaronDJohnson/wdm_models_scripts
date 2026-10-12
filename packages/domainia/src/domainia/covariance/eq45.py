r"""WDM II Eq. (45): one kernel family, five named builders (spec §4.5).

.. math::  \Lambda_{pq} = \frac1T\,\mathrm{Re}\sum_k \eta_k\,S(f_k, \bar t_{pq})\,
           \hat g_p(k)\,\hat g_q^*(k),\qquad \bar t_{pq} = (t_{n_p} + t_{n_q})/2,

with :math:`\bar t` used literally (pairs straddling the wrap get :math:`\bar t \approx
T/2`: the raw defect), no lag truncation, and only :math:`|m - m'| \le 1`.

=============  ======================  ==============================================
builder        PSD needs               evaluator (oracle)
=============  ======================  ==============================================
``bessel``     capability "bessel"     ``wdm.evolutionary.eq45_coeffs_bessel`` +
                                       ``eq45_blocks_cheb`` (WT ``TriCovariance``; NumPy golden)
``gram``       capability "gram"       ``eq45_blocks_gram`` (WG ``_gram_setup``/``_tri_blocks``;
                                       JAX golden)
``interp``     callable ``psd(f,t,nu)`` ``eq45_coeffs_interp`` + ``eq45_blocks_cheb`` (wdm_dev; NumPy)
``direct``     callable ``psd(f,t,nu)`` ``eq45_direct`` (dense direct sum; tests only)
``separable``  capability "separable"  ``K^I + g(t_bar) K^G`` (GB ``tri_kernels`` / ``pad_static``)
=============  ======================  ==============================================

``assembly="blocktri"`` factorises the band blocks (:class:`BlockTridiagonalCovariance`,
``layout`` ``"ragged"`` | ``"padded"``); ``assembly="dense"`` assembles the matrix and runs
the pinned dense Cholesky (:class:`DenseCovariance`; GB: ``chol_impl="numpy"``,
``0.5 (S + S^T)``).  ``A_n`` is never inside: ``nu = (alpha_c, alpha_d)`` for the
log-linear family, ``nu = mu`` (one channel's envelope coefficients) for ``separable``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from wdm.backend import Backend, get_backend
from wdm.evolutionary import (
    eq45_blocks_cheb,
    eq45_blocks_gram,
    eq45_coeffs_bessel,
    eq45_coeffs_interp,
    eq45_direct,
    eq45_gram_gather,
    eq45_gram_plan,
    eq45_plan,
    eq45_separable_assemble,
    eq45_separable_kernels,
)
from wdm.grid import delta_T, packed_cells
from wdm.transform import band_atoms

from .base import Covariance
from .blocktri import BlockTridiagonalCovariance, blocks_to_dense, to_layout
from .dense import DenseCovariance

__all__ = ["Eq45Covariance", "keep_mask", "window_bins"]

Builder = Literal["bessel", "gram", "interp", "direct", "separable"]
#: The PSD capability each builder needs (``None``: a callable PSD ``psd(f, t, nu)``).
_CAPABILITY: dict[str, str | None] = {"bessel": "bessel", "gram": "gram", "separable": "separable",
                                      "interp": None, "direct": None}


def keep_mask(cells: Any) -> np.ndarray:
    """Boolean mask over ``wdm.grid.packed_cells`` of ``cells.grid`` selecting ``cells``.

    Raises unless ``cells`` lists active cells in band order (increasing packed position),
    the order of every Eq. (45) block and of ``eq45_plan``'s kept cells.
    """
    g = cells.grid
    m_all, n_all = packed_cells(g.Nf, g.Nt)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(m_all, n_all, strict=True))}
    try:
        pos = np.array([index[int(a), int(b)] for a, b in zip(cells.m, cells.n, strict=True)], int)
    except KeyError as err:
        raise ValueError(f"cell {err.args[0]} is not an active cell of {g}") from None
    if pos.size and np.any(np.diff(pos) <= 0):
        raise ValueError("cells must be listed in band order (increasing packed position)")
    keep = np.zeros(m_all.size, bool)
    keep[pos] = True
    return keep


def window_bins(Nt: int, bands: np.ndarray) -> tuple[int, int]:
    """``(g0, G)``: the record bins ``[g0, g0 + G)`` spanned by the ``Nt``-bin windows of the
    contiguous ``bands`` (GB ``make_grid``)."""
    bands = np.asarray(bands, int)
    return (Nt // 2) * int(bands[0]) - Nt // 2, (Nt // 2) * int(bands[-1] - bands[0]) + Nt


class _ChebBlocks:
    """``bessel`` / ``interp``: Chebyshev-in-time blocks on the plan's kept cells."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, P: int, t_max: float | None,
                 kind: str) -> None:
        self.psd, self.grid, self.keep, self.P, self.t_max, self.kind = psd, grid, keep, P, t_max, kind

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        plan = eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=self.P, t_max=self.t_max)
        out: dict[str, Any] = {"plan": plan}
        if self.kind == "bessel":
            out["L"] = np.asarray(self.psd.L(plan.f), float)
        return out

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[list[Any], list[Any]]:
        plan = C["plan"]
        if self.kind == "bessel":
            # WT TriCovariance._cheb_weights with r = t_max / T_obs (1 on the data grid,
            # Nt'/Nt padded); on all K bins at once (elementwise, so bit-identical to
            # WT's per-block L[k] call).
            coeffs = eq45_coeffs_bessel(C["L"], nu[0], nu[1], plan.P, r=plan.t_max / self.psd.T, be=be)
        else:
            nu_h = np.asarray(nu)

            def Sn(f: Any, t: Any) -> Any:
                return self.psd(f, t, nu_h)

            coeffs = eq45_coeffs_interp(Sn, plan.f, plan.t_max, plan.P, k=plan.k_used)
        return eq45_blocks_cheb(plan, coeffs, be=be)


class _GramBlocks:
    """``gram``: one Gram product per band block (log-linear PSDs only)."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, t_max: float | None) -> None:
        self.psd, self.grid, self.keep, self.t_max = psd, grid, keep, t_max

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        plan = eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=2, t_max=self.t_max)
        gplan = eq45_gram_plan(plan)
        return {"gplan": gplan, "gL": eq45_gram_gather(gplan, np.asarray(self.psd.L(plan.f), float))}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        gplan = C["gplan"]
        # WG: al = ac + ad * (gT / T_obs - 0.5), T_obs the data record's
        alpha_cells = nu[0] + nu[1] * (be.asarray(gplan.gT) / self.psd.T - 0.5)
        return eq45_blocks_gram(gplan, alpha_cells, C["gL"], be=be)


class _DirectMatrix:
    """``direct``: the dense direct sum (tests and small grids)."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, t_max: float | None) -> None:
        self.psd, self.grid, self.keep, self.t_max = psd, grid, keep, t_max

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        return {"plan": eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=2, t_max=self.t_max)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        nu_h = np.asarray(nu)
        return be.asarray(eq45_direct(C["plan"], lambda f, t: self.psd(f, t, nu_h)))


def _separable_spectra(psd: Any, grid: Any, bands: np.ndarray, channel: int) -> tuple[np.ndarray, np.ndarray, Any]:
    """``(s_I, s_G, envelope)`` of ``channel`` on the cells' grid window bins."""
    from ..domains import FourierDomain

    (tab_I, env_I), (tab_G, env_G) = psd.components
    if env_I is not None or env_G is None:
        raise ValueError("separable Eq. (45) needs components ((s_I, None), (s_G, envelope))")
    g0, G = window_bins(grid.Nt, bands)
    window = FourierDomain(N=grid.N, dt=grid.dt, k0=g0, n=G, units=tab_I.domain.units)
    sI = np.asarray(tab_I.on(window), float)[channel]                 # (C, G) -> one channel
    sG = np.asarray(tab_G.on(window), float)[channel]
    return sI, sG, env_G


class _SeparableDense:
    """``separable`` + dense assembly: GB ``tri_kernels`` and ``covariance_builder``.

    ``K^X = 1/2 Re(U diag(s_X) U^H)`` over every cell's unit-norm atom ``U`` on the grid
    window (circular lags), ``half = n_p + n_q`` (non-circular) into ``g`` sampled at
    ``t = j dT / 2``, ``j = 0 .. 2 Nt - 2``.
    """

    def __init__(self, psd: Any, cells: Any, channel: int) -> None:
        self.psd, self.cells, self.channel = psd, cells, channel

    def host_constants(self) -> dict[str, Any]:
        g = self.cells.grid
        m = np.asarray(self.cells.m, int)
        n = np.asarray(self.cells.n, int)
        bands = np.unique(m)
        if np.any(np.diff(bands) != 1):
            raise ValueError("separable Eq. (45) needs contiguous bands")
        sI, sG, env = _separable_spectra(self.psd, g, bands, self.channel)
        g0, G = window_bins(g.Nt, bands)
        atoms = band_atoms(bands, g.Nf, g.Nt, g.dt)
        U = np.zeros((m.size, G), complex)                      # GB Grid.dense_atoms()[keep]
        for i, (mm, nn) in enumerate(zip(m, n, strict=True)):
            b = int(mm - bands[0])
            U[i, (g.Nt // 2) * b + np.arange(g.Nt)] = atoms[b, nn]
        Bh = np.asarray(env.basis(np.arange(2 * g.Nt - 1) * delta_T(g.Nf, g.dt) / 2), float)
        return {"KI": eq45_separable_kernels(U, sI), "KG": eq45_separable_kernels(U, sG),
                "half": n[:, None] + n[None, :], "Bh": Bh}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        g_half = be.xp.exp(nu @ C["Bh"])
        return eq45_separable_assemble(C["KI"], C["KG"], g_half, C["half"], be=be)


class _SeparableBlocks:
    """``separable`` + block-tridiagonal assembly on full bands: GB ``pad_static``.

    Per band ``b``: ``1/2 Re(u_b diag(s) u_b^H)`` on its window and, for ``(b, b+1)``, the
    same over the shared half window; ``Lambda = K^I + g(t_bar) K^G`` per block.
    """

    def __init__(self, psd: Any, cells: Any, channel: int) -> None:
        self.psd, self.cells, self.channel = psd, cells, channel

    def host_constants(self) -> dict[str, Any]:
        g = self.cells.grid
        m = np.asarray(self.cells.m, int)
        n = np.asarray(self.cells.n, int)
        bands = np.unique(m)
        Nt, h = g.Nt, g.Nt // 2
        if m.size != bands.size * Nt or not np.array_equal(n, np.tile(np.arange(Nt), bands.size)):
            raise ValueError("block-tridiagonal separable Eq. (45) needs every column of each band")
        sI, sG, env = _separable_spectra(self.psd, g, bands, self.channel)
        atoms = band_atoms(bands, g.Nf, Nt, g.dt)
        idx = (h * (bands - bands[0]))[:, None] + np.arange(Nt)[None, :]
        nB = bands.size

        def blocks(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d = np.empty((nB, Nt, Nt))
            u = np.empty((nB - 1, Nt, Nt))
            for b in range(nB):
                d[b] = eq45_separable_kernels(atoms[b], s[idx[b]])
            for b in range(nB - 1):
                Ua, Ub = atoms[b][:, h:], atoms[b + 1][:, :h]
                sh = s[idx[b, h:]]
                u[b] = 0.5 * ((Ua.real * sh) @ Ub.real.T + (Ua.imag * sh) @ Ub.imag.T)
            return d, u

        KId, KIu = blocks(sI)
        KGd, KGu = blocks(sG)
        nn = np.arange(Nt)
        Bh = np.asarray(env.basis(np.arange(2 * Nt - 1) * delta_T(g.Nf, g.dt) / 2), float)
        return {"KId": KId, "KIu": KIu, "KGd": KGd, "KGu": KGu,
                "half": nn[:, None] + nn[None, :], "Bh": Bh}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        g = be.xp.exp(nu @ C["Bh"])[C["half"]]
        return C["KId"] + g[None] * C["KGd"], C["KIu"] + g[None] * C["KGu"]


class _DenseFromBlocks:
    """Band blocks assembled into the dense matrix (``assembly="dense"``)."""

    def __init__(self, blocks: Any, sizes: tuple[int, ...]) -> None:
        self.blocks, self.sizes = blocks, sizes

    def host_constants(self) -> Any:
        return self.blocks.host_constants()

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.blocks.evaluate(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)


class Eq45Covariance(Covariance):
    """Eq. (45) on ``cells`` with a named ``builder`` and ``assembly`` (module docstring).

    Parameters
    ----------
    psd
        ``LogLinearPSD`` (``bessel``, ``gram``; ``interp``/``direct`` by calling it),
        any callable PSD ``psd(f, t, nu)`` (``GenericPSD``: ``interp``, ``direct``) or
        ``SeparablePSD`` (``separable``).  A builder the PSD cannot serve raises ``ValueError``.
    cells
        A ``CellSet`` in band order.
    assembly, layout
        ``"blocktri"`` (``layout`` ``"ragged"``: default for ``bessel``/``interp``;
        ``"padded"``: default for ``gram``/``separable``) or ``"dense"`` (``chol_impl``,
        ``symmetrize`` as :class:`DenseCovariance`).
    t_max
        End of the Chebyshev / time domain (default: the cells' grid ``T``); a padded grid
        passes its extended ``T'`` and ``alpha(t)`` is continued linearly.
    P
        Chebyshev order of ``bessel`` / ``interp`` (24).
    channel
        ``separable`` only: the channel of the ``SeparablePSD`` spectra.
    """

    def __init__(self, psd: Any, cells: Any, *, builder: Builder, assembly: str = "blocktri",
                 layout: str | None = None, t_max: float | None = None, P: int = 24,
                 channel: int = 0, chol_impl: Literal["numpy", "scipy"] = "numpy",
                 symmetrize: bool = True) -> None:
        if builder not in _CAPABILITY:
            raise ValueError(f"unknown Eq. (45) builder {builder!r}; expected one of {tuple(_CAPABILITY)}")
        need = _CAPABILITY[builder]
        caps = getattr(psd, "capabilities", frozenset())
        if need is None and not callable(psd):
            raise ValueError(f"builder {builder!r} needs a callable PSD psd(f, t, nu); "
                             f"{type(psd).__name__} is not callable")
        if need is not None and need not in caps:
            raise ValueError(f"builder {builder!r} needs a PSD with capability {need!r}; "
                             f"{type(psd).__name__} has {sorted(caps)}")
        if assembly not in ("blocktri", "dense"):
            raise ValueError(f"assembly must be 'blocktri' or 'dense'; got {assembly!r}")
        if builder == "direct" and assembly != "dense":
            raise ValueError("builder 'direct' needs assembly='dense'")
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
        self.n = len(cells.m)
        blocks: Any
        if builder == "separable":
            m = np.asarray(cells.m, int)
            sizes = tuple(int(c) for c in np.bincount(m - m.min()))
            blocks = (_SeparableDense(psd, cells, channel) if assembly == "dense"
                      else _SeparableBlocks(psd, cells, channel))
        else:
            keep = keep_mask(cells)
            m_all, _ = packed_cells(cells.grid.Nf, cells.grid.Nt)
            sizes = tuple(int(np.sum(keep & (m_all == mm))) for mm in range(cells.grid.Nf + 1))
            if builder in ("bessel", "interp"):
                blocks = _ChebBlocks(psd, cells.grid, keep, P, t_max, builder)
            elif builder == "gram":
                blocks = _GramBlocks(psd, cells.grid, keep, t_max)
            else:
                blocks = _DirectMatrix(psd, cells.grid, keep, t_max)
        self.sizes = sizes
        self.block_evaluator = blocks
        if assembly == "blocktri":
            lay = layout or ("padded" if builder in ("gram", "separable") else "ragged")
            self.impl: Covariance = BlockTridiagonalCovariance(blocks, sizes, layout=lay)
        else:
            dense_out = isinstance(blocks, (_DirectMatrix, _SeparableDense))
            matrix = blocks if dense_out else _DenseFromBlocks(blocks, sizes)
            self.impl = DenseCovariance(matrix, self.n, chol_impl=chol_impl, symmetrize=symmetrize)
        self.jittable = self.impl.jittable and builder not in ("interp", "direct")

    @property
    def layout(self) -> str | None:
        """The block layout (``None`` for ``assembly="dense"``)."""
        return getattr(self.impl, "layout", None)

    def host_constants(self) -> Any:
        return self.impl.host_constants()

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._factor(C, nu, be)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._matrix(C, nu, be)

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        """``(diag, up)`` from the constants ``C`` (``assembly="blocktri"`` only)."""
        if not isinstance(self.impl, BlockTridiagonalCovariance):
            raise ValueError("band_blocks needs assembly='blocktri'")
        return self.impl.band_blocks(C, nu, be)

    def blocks(self, nu: Any, be: Backend | None = None) -> Any:
        """The builder's raw output at ``nu``: band blocks (ragged lists for ``bessel``/
        ``interp``, padded arrays for ``gram``/separable-blocktri) or the matrix
        (``direct``, separable-dense).  G6 and the strategy cross-checks read it."""
        be = be or get_backend(nu)
        C = self.constants(be)
        C = C["blocks"] if isinstance(self.impl, BlockTridiagonalCovariance) else C
        return self.block_evaluator.evaluate(C, be.asarray(nu), be)
