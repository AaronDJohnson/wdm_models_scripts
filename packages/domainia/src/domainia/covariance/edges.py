r"""Edge treatments as combinators (spec §4.6): ``cut``, ``SchurEdge``, ``ZeroPadded``.

* :func:`cut` rebuilds the base on the kept cells (``t_bar`` stays literal).
* :class:`SchurEdge` -- exact rows on the edge cells over an approximate interior,
  ``C = [[A, B], [B^T, T]]``, ``A = (R_ee + R_ee^T)/2``, ``B = R_ei``.
  ``assembly="schur"`` (WT ``Schur``): ``S = A - B T^-1 B^T`` and
  ``log|C| = log|T| + log|S|``; ``assembly="dense"`` (GB ``covariance_builder``): the
  assembled matrix, edge rows then columns overwritten, factorised whole.  A non-PD
  ``S`` gives ``ok = False`` (``-inf``): a deliberate prior truncation (band/mid/tri_exact
  are ``-inf`` at 12/13/12 of the 30 sinusoid reference points).
* :class:`ZeroPadded` -- the user's ``M Lambda M`` (ruling 2026-10-04), two explicit
  strategies because the data domains differ: ``"time_marginal"`` (WT ``PaddedMarginal``:
  the extended grid's Gaussian with the padding samples marginalised; the data are the
  residual samples) and ``"cell_congruence"`` (GB ``PadMask``/``pad_covariance``:
  ``A Lambda_R A^T`` with ``A`` the identity except on the rows ``1 - rho > edge_tol``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from wdm.backend import Backend, get_backend, pytree_dataclass
from wdm.grid import active_mask, packed_cells
from wdm.transform import band_atoms, forward_time_many, inverse_time_many

from ..linalg import BandLayout
from .base import (CholFactor, Covariance, DiagFactor, QuadForm, attach_constants, concrete_failure,
                   factor_constants, quad_form)
from .blocktri import blocktri_cov_factor, dummy_slots
from .dense import dense_factor
from .diagonal import DiagonalCovariance
from .eq45 import window_bins
from .pushed import PushedRows, positions

__all__ = ["PadMask", "PaddedFactor", "SchurEdge", "SchurFactor", "ZeroPadded", "assemble_edge_rows", "cut",
           "schur_factor"]


def cut(make_cov: Callable[[Any], Any], cells: Any, lead: int, trail: int) -> Any:
    """The base rebuilt on ``cells`` without the first ``lead`` and last ``trail`` columns
    (WT/GB ``*_cut``: the same literal ``t_bar`` on the kept cells)."""
    return make_cov(cells.drop_columns(lead, trail))


# ---------------------------------------------------------------------------------------
# Schur complement over an approximate interior
# ---------------------------------------------------------------------------------------


@pytree_dataclass(data=("T", "S", "B", "TiBt", "logdet", "ok"), meta=())
@dataclass(frozen=True, eq=False)
class SchurFactor:
    """WT ``Schur``: ``T`` the interior factor, ``S`` the Schur complement's Cholesky,
    ``B = R_ei``, ``TiBt = T^-1 B^T``.  The edge / interior positions are constants: the
    methods read ``C["e"]``, ``C["i"]`` and pass ``C["interior"]`` to ``T``."""

    T: Any
    S: Any
    B: Any
    TiBt: Any
    logdet: Any
    ok: Any

    n_u = 0

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        """WT ``Schur.solve`` line for line."""
        if self.S is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        C = factor_constants(self, C, "solve")
        be = be or get_backend(b, self.B)
        one = b.ndim == 1
        b = b[:, None] if one else b
        e, i = C["e"], C["i"]
        b_e, b_i = b[e], b[i]
        ti = self.T.solve(b_i, C=C["interior"], be=be)
        xe = self.S.solve(b_e - self.B @ ti, be=be)
        xi = ti - self.TiBt @ xe
        x = be.xp.zeros_like(b)
        x = be.set(x, e, xe, copy=False)
        x = be.set(x, i, xi, copy=False)
        return x[:, 0] if one else x

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        """``[L_T^-1 b_i ; L_S^-1 (b_e - B T^-1 b_i)]`` (GB ``EdgeSchurCache``)."""
        if self.S is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        C = factor_constants(self, C, "whiten")
        be = be or get_backend(b, self.B)
        b_e, b_i = b[C["e"]], b[C["i"]]
        z = b_e - self.B @ self.T.solve(b_i, C=C["interior"], be=be)
        return be.xp.concatenate([self.T.whiten(b_i, C=C["interior"], be=be), self.S.whiten(z, be=be)],
                                 axis=0)

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, be or get_backend(b, self.B), C)


def schur_factor(T: Any, A_ee: Any, B_ei: Any, *, chol_impl: Literal["numpy", "scipy"], be: Backend,
                 C_T: Any = None) -> SchurFactor:
    """Factor ``[[A, B], [B^T, T]]`` from the edge rows ``R`` split at the edge / interior
    positions, ``A_ee = R[:, e]``, ``B_ei = R[:, i]`` (WT ``MatrixWdm.solver`` +
    ``Schur.__init__``): ``A = (A_ee + A_ee^T)/2``, ``TiBt = T^-1 B^T``, ``S = A - B TiBt``
    symmetrised; ``ok = T.ok & S.ok``.  ``C_T`` are ``T``'s constants (``C["interior"]``).  A
    concretely failed ``T`` (WT: ``LinAlgError`` before the complement) gives a failed factor
    without ``S``."""
    A = 0.5 * (A_ee + A_ee.T)
    B = B_ei
    if concrete_failure(T.ok):
        return SchurFactor(T, None, B, None, float("nan"), False)
    TiBt = T.solve(B.T, C=C_T, be=be)
    Sc = A - B @ TiBt
    S = CholFactor.build(0.5 * (Sc + Sc.T), impl=chol_impl, be=be)
    return SchurFactor(T, S, B, TiBt, T.logdet + S.logdet, T.ok & S.ok)


def assemble_edge_rows(M_i: Any, R: Any, e: Any, i: Any, be: Backend) -> Any:
    """GB ``covariance_builder`` (``exact``): the interior matrix on ``i``, then the edge rows
    and columns overwritten by ``R`` (rows first, so ``C_ee = R_ee^T``)."""
    n = R.shape[1]
    lam = be.xp.zeros((n, n))
    lam = be.set(lam, (i[:, None], i[None, :]), M_i, copy=False)
    lam = be.set(lam, (e, slice(None)), R, copy=False)
    return be.set(lam, (slice(None), e), R.T, copy=False)


class SchurEdge(Covariance):
    """Exact edge rows over an approximate interior (module docstring).

    ``interior`` covers the destination cells minus ``edge`` (band order; checked against
    ``interior.cells`` when it has them); ``rows`` is a :class:`PushedRows` onto the
    destination cells whose row cells are ``edge``.
    ``chol_impl`` pins the complement's (``"schur"``) or the matrix's (``"dense"``)
    Cholesky: ``"scipy"`` (WT ``Schur``) or ``"numpy"`` (GB).  ``assembly="dense"``
    needs an interior with a dense form (``_matrix``).

    On JAX the rows are evaluated per static Chebyshev order (``rows.order_for``): one
    jitted program per order, or the separately compiled stages when ``rows.staged`` and
    ``rows.phi_cache``; the rest of the factorisation is one more jitted program taking
    the rows.  With ``jit=False`` (or a non-jittable interior) everything runs eagerly in
    the rows' own order -- the WG order when ``phi_cache`` -- and the stages are unused:
    the ``reference`` strategy, never a production JAX path.
    """

    def __init__(self, interior: Covariance, rows: PushedRows, edge: Any, *,
                 assembly: Literal["schur", "dense"] = "schur",
                 chol_impl: Literal["numpy", "scipy"] = "scipy") -> None:
        if assembly not in ("schur", "dense"):
            raise ValueError(f"assembly must be 'schur' or 'dense'; got {assembly!r}")
        if chol_impl not in ("numpy", "scipy"):     # else NumPy raises only after the rows (~3 s)
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        self.interior, self.rows, self.assembly, self.chol_impl = interior, rows, assembly, chol_impl
        self.cells = rows.cells
        self.n = rows.n
        self.e = positions(edge, self.cells)
        if not np.array_equal(self.e, rows.e):
            raise ValueError("the edge cells must be the rows' cells, in the same order")
        self.i = np.setdiff1d(np.arange(self.n), self.e)
        if interior.n != self.i.size:
            raise ValueError(f"the interior has {interior.n} cells; expected {self.i.size}")
        inner = getattr(interior, "cells", None)
        if inner is not None and inner != self.cells.complement(edge):
            raise ValueError(f"the interior's cells ({inner!r}) are not the destination cells "
                             f"minus the edge ({self.cells.complement(edge)!r})")
        self.jittable = interior.jittable

    def host_constants(self) -> dict[str, Any]:
        return {"rows": self.rows.host_constants(), "e": self.e, "i": self.i}

    def children(self) -> dict[str, Covariance]:
        return {"interior": self.interior}

    def state_key(self, nu: Any) -> Any:
        """``(P, interior key)``, ``P = rows.order_for(nu)`` for adaptive rows (``None`` for
        fixed-order and Fourier rows): the static order :meth:`factor` dispatches on."""
        own = self.rows.order_for(nu) if self.rows.order == "adaptive" else None
        inner = self.interior.state_key(nu)
        return None if own is None and inner is None else (own, inner)

    def factor_from_rows(self, C: Any, nu: Any, R: Any, be: Backend) -> Any:
        """Everything after the rows (pure; jitted on JAX with ``R`` as an argument)."""
        e, i = C["e"], C["i"]
        if self.assembly == "schur":
            T = self.interior._factor(C["interior"], nu, be)
            return schur_factor(T, R[:, e], R[:, i], chol_impl=self.chol_impl, be=be, C_T=C["interior"])
        lam = assemble_edge_rows(self.interior._matrix(C["interior"], nu, be), R, e, i, be)
        return dense_factor(lam, chol_impl=self.chol_impl, symmetrize=True, be=be)

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.factor_from_rows(C, nu, self.rows.evaluate(C["rows"], nu, be), be)

    def factor(self, nu: Any, be: Backend | None = None, *, jit: bool | None = None) -> Any:
        be = be or get_backend(nu)
        C = self.constants(be)
        nu = be.asarray(nu)
        P = self.rows.order_for(nu)
        if not (be.name == "jax" and (self.jittable if jit is None else bool(jit))):
            return attach_constants(self.factor_from_rows(C, nu, self.rows.evaluate(C["rows"], nu, be, P=P), be),
                                    C)
        fns = self.__dict__.setdefault("_jit_cache", {})
        if self.rows.staged and self.rows.phi_cache:
            R = self.rows.evaluate_staged(C["rows"], nu, be, P)
        else:
            if ("rows", P) not in fns:
                fns["rows", P] = be.jit(lambda Cr, nu, P=P: self.rows.evaluate(Cr, nu, be, P=P))
            R = fns["rows", P](C["rows"], nu)
        if "rest" not in fns:
            fns["rest"] = be.jit(lambda C, nu, R: self.factor_from_rows(C, nu, R, be))
        return attach_constants(fns["rest"](C, nu, R), C)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        R = self.rows.evaluate(C["rows"], nu, be)
        return assemble_edge_rows(self.interior._matrix(C["interior"], nu, be), R, C["e"], C["i"], be)


# ---------------------------------------------------------------------------------------
# Zero padding: the user's M Lambda M
# ---------------------------------------------------------------------------------------


@pytree_dataclass(data=("Lam", "LiWp", "Q", "logdet", "ok"), meta=("Nf", "Nt", "dt", "n"))
@dataclass(frozen=True, eq=False)
class PaddedFactor:
    """WT ``PaddedMarginal``: ``C_dd^-1 = Q_dd - Q_dp Q_pp^-1 Q_pd`` on the data samples,
    ``Q = W^T Lambda^-1 W``; vectors are length-``n`` time series (``solve`` only).  The
    padding impulses ``W_p`` and the active-cell index ``(m, n_idx)`` are constants
    (``C["Wp"]``, ``C["m"]``, ``C["n_idx"]``); ``Lam`` reads ``C["base"]``."""

    Lam: Any
    LiWp: Any
    Q: Any
    logdet: Any
    ok: Any
    Nf: int
    Nt: int
    dt: float
    n: int

    n_u = 0

    def solve(self, x: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        """WT ``PaddedMarginal.solve`` line for line (``W^T z = dt W^-1 z``)."""
        if self.LiWp is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        C = factor_constants(self, C, "solve")
        Wp, m, n_idx = C["Wp"], C["m"], C["n_idx"]
        be = be or get_backend(x, Wp)
        xp = be.xp
        one = x.ndim == 1
        X = x[None, :] if one else x.T                                      # (k, n)
        ext = be.set(xp.zeros((X.shape[0], self.Nf * self.Nt)), (slice(None), slice(0, self.n)), X, copy=False)
        W = forward_time_many(ext, self.Nf, self.Nt, self.dt, be=be)[:, m, n_idx]
        Liw = self.Lam.solve(W.T, C=C["base"], be=be)                       # (size, k)
        c = self.Q.solve(Wp.T @ Liw, be=be)
        Y = (Liw - self.LiWp @ c).T                                         # (k, size)
        full = be.set(xp.zeros((Y.shape[0], self.Nf + 1, self.Nt)), (slice(None), m, n_idx), Y,
                      copy=False)
        out = self.dt * inverse_time_many(full, self.Nf, self.Nt, self.dt, be=be)[:, :self.n]
        return out[0] if one else out.T

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        raise NotImplementedError("the padded marginal has no whitening; use form='solve'")

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, be or get_backend(b, self.logdet), C)


class ZeroPadded(Covariance):
    """The padded models' ``M Lambda M`` (module docstring).

    ``strategy="time_marginal"`` (sinusoid): ``base_ext`` covers every cell of the extended
    ``grid`` (``Nt' = Nt + pad_columns``, ``alpha`` continued); the data are the
    ``n = Nt Nf`` residual samples and the factor is a :class:`PaddedFactor`
    (``C_dd^-1 = Q_dd - Q_dp Q_pp^-1 Q_pd``, ``log|C_dd| = log|Lambda| - N' log dt + log|Q_pp|``;
    one ``(pad_columns Nf)^2`` Cholesky per state).  The kept-cell route (condition ~1e11)
    is never used.

    ``strategy="cell_congruence"`` (GB): ``base_ext`` covers the store bands x ``Nt'``
    columns of the padded grid (a ``DiagonalCovariance``, or a covariance with
    ``band_blocks`` returning padded ``(nB, Nt', Nt')`` / ``(nB - 1, Nt', Nt')`` blocks, e.g. a
    block-tridiagonal separable ``Eq45Covariance``); ``mask`` the :class:`PadMask` (``E``,
    ``Kpos``, ``AE``, ``k0``, ``NT_ext``); ``n_bands`` analysis bands; ``n_columns`` the data
    columns ``Nt``.  ``factor="dense"`` assembles ``A Lambda_R A^T`` on the kept cells and
    runs the GB dense Cholesky (golden); ``"schur"`` factorises the interior (kept minus
    ``E``) by its structure and the ``E`` rows by a Schur complement (opt-in).
    """

    def __init__(self, base_ext: Covariance, *, strategy: Literal["time_marginal", "cell_congruence"],
                 grid: Any = None, pad_columns: int = 16, mask: Any = None, n_bands: int | None = None,
                 n_columns: int | None = None, factor: Literal["dense", "schur"] = "dense",
                 chol_impl: Literal["numpy", "scipy"] | None = None) -> None:
        if strategy not in ("time_marginal", "cell_congruence"):
            raise ValueError(f"unknown padding strategy {strategy!r}")
        if factor not in ("dense", "schur"):
            raise ValueError(f"factor must be 'dense' or 'schur'; got {factor!r}")
        if chol_impl not in (None, "numpy", "scipy"):   # else it raises only at the first factor
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        self.base, self.strategy, self.factor_kind = base_ext, strategy, factor
        self.pad_columns = int(pad_columns)
        self.jittable = base_ext.jittable
        if strategy == "time_marginal":
            if grid is None:
                raise ValueError("time_marginal needs the extended grid")
            if factor != "dense":                       # never silently ignored
                raise ValueError("factor applies to cell_congruence only; time_marginal has one "
                                 "factorisation (WT PaddedMarginal)")
            if not 0 < self.pad_columns < grid.Nt:
                raise ValueError(f"pad_columns must lie in (0, {grid.Nt}); got {self.pad_columns}")
            if base_ext.n != grid.N:
                raise ValueError(f"the base covers {base_ext.n} cells; the extended grid has {grid.N}")
            self.grid = grid
            self.n = (grid.Nt - self.pad_columns) * grid.Nf
            self.chol_impl = chol_impl or "scipy"
            return
        if mask is None or n_bands is None or n_columns is None:
            raise ValueError("cell_congruence needs mask, n_bands and n_columns")
        if not (isinstance(base_ext, DiagonalCovariance) or hasattr(base_ext, "band_blocks")):
            raise ValueError("cell_congruence needs a DiagonalCovariance or a base with band_blocks "
                             f"(store-band blocks); got {type(base_ext).__name__}")
        n_R = np.asarray(mask.AE).shape[1]
        if base_ext.n != n_R:
            raise ValueError(f"the base covers {base_ext.n} cells; the mask's A_E has {n_R} columns")
        nK, NT = int(n_bands), int(n_columns)
        kpos = (mask.k0 + np.repeat(np.arange(nK), NT)) * mask.NT_ext + np.tile(np.arange(NT), nK)
        if not np.array_equal(np.asarray(mask.Kpos), kpos):     # the layout _congruence_pieces assumes
            raise ValueError(f"mask.Kpos must be the {nK} x {NT} kept cells in band order, "
                             "(k0 + b) NT_ext + n with n < n_columns")
        self.mask, self.n_bands, self.n_columns = mask, nK, NT
        self.n = int(np.asarray(mask.Kpos).size)
        self.chol_impl = chol_impl or "numpy"
        self.e = np.asarray(mask.E, int)
        self.i = np.setdiff1d(np.arange(self.n), self.e)
        band = np.asarray(mask.Kpos) // mask.NT_ext - mask.k0
        col = np.asarray(mask.Kpos) % mask.NT_ext
        self.interior_cols = [col[self.i][band[self.i] == b] for b in range(self.n_bands)]

    # -- constants ----------------------------------------------------------------------
    def host_constants(self) -> dict[str, Any]:
        if self.strategy == "time_marginal":
            g = self.grid
            n_pad = g.N - self.n
            unit = np.zeros((n_pad, g.N))
            unit[np.arange(n_pad), np.arange(self.n, g.N)] = 1.0
            m, n = packed_cells(g.Nf, g.Nt)
            return {"Wp": forward_time_many(unit, g.Nf, g.Nt, g.dt)[:, active_mask(g.Nf, g.Nt)].T,
                    "m": m, "n_idx": n}
        C: dict[str, Any] = {"e": self.e, "i": self.i, "Kpos": np.asarray(self.mask.Kpos, int),
                             "AE": np.asarray(self.mask.AE, float), "interior": {}}
        if self.factor_kind == "schur" and getattr(self.base, "layout", None) is not None:
            sizes = tuple(int(c.size) for c in self.interior_cols)
            padded = self.base.layout == "padded"
            C["interior"] = {"layout": BandLayout.from_sizes(sizes) if padded else None,
                             "dummy": dummy_slots(sizes) if padded else None}
        return C

    def children(self) -> dict[str, Covariance]:
        return {"base": self.base}

    # -- time_marginal ------------------------------------------------------------------
    def _time_factor(self, C: Any, nu: Any, be: Backend) -> PaddedFactor:
        """WT ``PaddedMarginal.__init__`` + ``logdet``."""
        g = self.grid
        Lam = self.base._factor(C["base"], nu, be)
        Wp = C["Wp"]
        if concrete_failure(Lam.ok):        # WT: LinAlgError in Lam -> -inf before Q_pp
            return PaddedFactor(Lam, None, None, float("nan"), False, g.Nf, g.Nt, g.dt, self.n)
        LiWp = Lam.solve(Wp, C=C["base"], be=be)
        Qpp = Wp.T @ LiWp
        Q = CholFactor.build(0.5 * (Qpp + Qpp.T), impl=self.chol_impl, be=be)
        logdet = Lam.logdet - g.N * np.log(g.dt) + Q.logdet
        return PaddedFactor(Lam, LiWp, Q, logdet, Lam.ok & Q.ok, g.Nf, g.Nt, g.dt, self.n)

    # -- cell_congruence ----------------------------------------------------------------
    def _congruence_pieces(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any, Any]:
        """``(lamKK, Y, blocks)``: ``Lambda_R`` on the kept cells, ``Y = A_E Lambda_R`` and the
        base's ``(d, u)`` (``None`` for a diagonal base) -- GB ``pad_covariance``."""
        xp = be.xp
        AE, Kpos = C["AE"], C["Kpos"]
        nE = self.e.size
        if isinstance(self.base, DiagonalCovariance):
            v = self.base.variances(C["base"], nu, be)
            return xp.diag(v[Kpos]), AE * v[None, :], None
        d, u = self.base.band_blocks(C["base"], nu, be)
        if isinstance(d, (list, tuple)):
            raise ValueError("cell_congruence needs the store-band blocks as arrays (nB, NT', NT') "
                             "and (nB - 1, NT', NT'), not ragged lists")
        nB, NTp = d.shape[0], d.shape[1]
        A3 = AE.reshape(nE, nB, NTp)
        Y = xp.einsum("ebn,bnm->ebm", A3, d)
        zero = xp.zeros((nE, 1, NTp))
        Y = Y + xp.concatenate([zero, xp.einsum("ebn,bnm->ebm", A3[:, :-1], u)], axis=1)
        Y = Y + xp.concatenate([xp.einsum("ebn,bmn->ebm", A3[:, 1:], u), zero], axis=1)
        Y = Y.reshape(nE, nB * NTp)
        k0, nK, NT = self.mask.k0, self.n_bands, self.n_columns
        db = [d[k0 + i][:NT, :NT] for i in range(nK)]
        ub = [u[k0 + i][:NT, :NT] for i in range(nK - 1)]
        Z = xp.zeros((NT, NT))
        grid = [[db[i] if i == j else ub[i] if j == i + 1 else ub[j].T if i == j + 1 else Z
                 for j in range(nK)] for i in range(nK)]
        return xp.block(grid), Y, (d, u)

    def _edge_rows(self, C: Any, Y: Any, be: Backend) -> Any:
        """GB ``assemble``: ``rows = Y[:, Kpos]`` with the ``E`` columns replaced by ``Y A_E^T``."""
        rows = Y[:, C["Kpos"]]
        return be.set(rows, (slice(None), C["e"]), Y @ C["AE"].T, copy=False)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.strategy == "time_marginal":
            raise NotImplementedError("time_marginal has no cell-space matrix (its data are samples)")
        lamKK, Y, _ = self._congruence_pieces(C, nu, be)
        rows = self._edge_rows(C, Y, be)
        lam = be.set(lamKK, (C["e"], slice(None)), rows)
        return be.set(lam, (slice(None), C["e"]), rows.T, copy=False)

    def _interior_factor(self, C: Any, lamKK: Any, blocks: Any, be: Backend) -> Any:
        """The interior's own factor: diagonal, or block-tridiagonal over the analysis bands."""
        if blocks is None:
            return DiagFactor.build(be.xp.diagonal(lamKK)[self.i], be)
        d, u = blocks
        k0, cols = self.mask.k0, self.interior_cols
        diag = [d[k0 + b][np.ix_(c, c)] for b, c in enumerate(cols)]
        up = [u[k0 + b][np.ix_(cols[b], cols[b + 1])] for b in range(len(cols) - 1)]
        sizes = tuple(int(c.size) for c in cols)
        Ci = C["interior"]
        return blocktri_cov_factor(diag, up, sizes, self.base.layout, Ci.get("layout"), be, Ci.get("dummy"))

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.strategy == "time_marginal":
            return self._time_factor(C, nu, be)
        if self.factor_kind == "dense":
            return dense_factor(self._matrix(C, nu, be), chol_impl=self.chol_impl, symmetrize=True, be=be)
        lamKK, Y, blocks = self._congruence_pieces(C, nu, be)
        R = self._edge_rows(C, Y, be)
        T = self._interior_factor(C, lamKK, blocks, be)
        return schur_factor(T, R[:, C["e"]], R[:, C["i"]], chol_impl=self.chol_impl, be=be, C_T=C["interior"])


@dataclass(frozen=True, eq=False)
class PadMask:
    """GB ``PadMask``: ``A = W_K Z P Z^T W_R^-1`` in cell units, stored only on the rows ``E``
    (positions among the kept cells, ``1 - rho > edge_tol``); every other row is the
    identity at ``Kpos`` (the kept cells' positions among ``R``: band-major, ``NT_ext``
    columns per band, the first analysis band at band position ``k0``)."""

    E: np.ndarray
    Kpos: np.ndarray
    AE: np.ndarray
    k0: int
    NT_ext: int

    @classmethod
    def build(cls, *, map_rows: np.ndarray, rho: np.ndarray, kept_band: np.ndarray, kept_n: np.ndarray,
              R_grid: Any, R_bands: np.ndarray, record_N: int, store_k0: int, store_n: int, k0: int,
              edge_tol: float = 1e-8, chunk: int = 512) -> PadMask:
        """GB ``pad_mask`` + ``mask_rows`` (host NumPy, once per case).

        ``map_rows`` ``(P, n_store)`` complex: the kept cells are ``Re(map_rows @ z)`` of the
        store bins ``[store_k0, store_k0 + store_n)`` (they carry ``sqrt(N/N')``);
        ``rho`` ``(P,)`` each kept atom's energy on the data; ``kept_band`` (position among
        the analysis bands) and ``kept_n`` (column) of each kept cell; ``R_grid`` the padded
        grid (``wdm.Grid(Nf, Nt', dt)``) and ``R_bands`` its bands (every band overlapping
        the store); ``k0`` the first analysis band's position among ``R_bands``.  A cell
        ``r`` of ``R`` (value 1) has padded bins ``conj(u_r) sqrt(N'/N)``; cropping them to the
        record (``maps.fourier.crop_maps``) gives its store bins.

        Inconsistent inputs are refused up front (each of these otherwise returned a wrong
        mask without an error): ``chunk < 1`` (uninitialised rows), ``rho`` / ``kept_band``
        / ``kept_n`` not one per row of ``map_rows`` (rows silently dropped), non-contiguous
        ``R_bands`` (``window_bins`` and the band windows assume them contiguous), columns
        outside ``[0, Nt')`` or bands outside ``R`` (``Kpos`` wraps into the next band or
        past ``|R|``).  ``edge_tol < 0`` is allowed: every kept row is then computed.
        """
        from ..maps.fourier import crop_maps            # local: maps may import covariance

        map_rows, rho = np.asarray(map_rows), np.asarray(rho, float)
        kept_band, kept_n = np.asarray(kept_band, int), np.asarray(kept_n, int)
        NTp = R_grid.Nt
        R_bands = np.asarray(R_bands, int)
        if int(chunk) < 1:
            raise ValueError(f"chunk must be >= 1; got {chunk}")
        if map_rows.ndim != 2 or map_rows.shape[1] != store_n:
            raise ValueError(f"map_rows must be (P, store_n = {store_n}); got shape {map_rows.shape}")
        n_rows = map_rows.shape[0]
        if not rho.shape == kept_band.shape == kept_n.shape == (n_rows,):
            raise ValueError(f"rho, kept_band and kept_n need one entry per row of map_rows ({n_rows}); got "
                             f"{rho.shape}, {kept_band.shape}, {kept_n.shape}")
        if R_bands.ndim != 1 or R_bands.size == 0 or np.any(np.diff(R_bands) != 1):
            raise ValueError(f"R_bands must be contiguous ascending bands; got {R_bands.tolist()}")
        if n_rows and (kept_n.min() < 0 or kept_n.max() >= NTp):
            raise ValueError(f"kept_n must lie in [0, {NTp}) (the padded grid's columns)")
        if k0 < 0 or (n_rows and (kept_band.min() < 0 or k0 + kept_band.max() >= R_bands.size)):
            raise ValueError(f"k0 + kept_band must lie in [0, {R_bands.size}) (positions among R_bands); "
                             f"got k0 = {k0}")

        E = np.flatnonzero(1.0 - rho > edge_tol)
        Kpos = (k0 + kept_band) * NTp + kept_n
        g0, G = window_bins(NTp, R_bands)
        atoms = band_atoms(R_bands, R_grid.Nf, NTp, R_grid.dt)
        idx = ((NTp // 2) * (R_bands - R_bands[0]))[:, None] + np.arange(NTp)[None, :]
        k = store_k0 + np.arange(store_n)
        CE = map_rows[E]
        X = np.empty((E.size, G), complex)
        Y = np.empty((E.size, G), complex)
        for lo in range(0, G, chunk):
            j = g0 + np.arange(lo, min(lo + chunk, G))
            Ac, Bc = crop_maps(R_grid.N, 0, record_N, k, j)
            X[:, lo:lo + j.size] = CE @ Ac
            Y[:, lo:lo + j.size] = CE @ Bc
        scale = np.sqrt(R_grid.N / record_N)
        out = np.empty((E.size, R_bands.size, NTp))
        for b in range(R_bands.size):
            i = idx[b]
            out[:, b] = scale * np.real(X[:, i] @ np.conj(atoms[b]).T + Y[:, i] @ atoms[b].T)
        return cls(E, Kpos, out.reshape(E.size, -1), int(k0), int(NTp))
