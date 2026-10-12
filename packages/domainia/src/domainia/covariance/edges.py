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
from .base import CholFactor, Covariance, DiagFactor, QuadForm, concrete_failure, quad_form
from .blocktri import blocktri_cov_factor, dummy_slots
from .dense import dense_factor
from .diagonal import DiagonalCovariance
from .eq45 import window_bins
from .pushed import PushedRows, positions

__all__ = ["PaddedFactor", "SchurEdge", "SchurFactor", "ZeroPadded", "assemble_edge_rows", "cut", "schur_factor"]


def cut(make_cov: Callable[[Any], Any], cells: Any, lead: int, trail: int) -> Any:
    """The base rebuilt on ``cells`` without the first ``lead`` and last ``trail`` columns
    (WT/GB ``*_cut``: the same literal ``t_bar`` on the kept cells)."""
    return make_cov(cells.drop_columns(lead, trail))


# ---------------------------------------------------------------------------------------
# Schur complement over an approximate interior
# ---------------------------------------------------------------------------------------


@pytree_dataclass(data=("T", "S", "B", "TiBt", "e", "i", "logdet", "ok"), meta=())
@dataclass(frozen=True, eq=False)
class SchurFactor:
    """WT ``Schur``: ``T`` the interior factor, ``S`` the Schur complement's Cholesky,
    ``B = R_ei``, ``TiBt = T^-1 B^T``; ``e`` / ``i`` the edge / interior positions."""

    T: Any
    S: Any
    B: Any
    TiBt: Any
    e: Any
    i: Any
    logdet: Any
    ok: Any

    n_u = 0

    def solve(self, b: Any, *, be: Backend | None = None) -> Any:
        """WT ``Schur.solve`` line for line."""
        if self.S is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        be = be or get_backend(b, self.B)
        one = b.ndim == 1
        b = b[:, None] if one else b
        b_e, b_i = b[self.e], b[self.i]
        ti = self.T.solve(b_i, be=be)
        xe = self.S.solve(b_e - self.B @ ti, be=be)
        xi = ti - self.TiBt @ xe
        x = be.xp.zeros_like(b)
        x = be.set(x, self.e, xe, copy=False)
        x = be.set(x, self.i, xi, copy=False)
        return x[:, 0] if one else x

    def whiten(self, b: Any, *, be: Backend | None = None) -> Any:
        """``[L_T^-1 b_i ; L_S^-1 (b_e - B T^-1 b_i)]`` (GB ``EdgeSchurCache``)."""
        if self.S is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        be = be or get_backend(b, self.B)
        b_e, b_i = b[self.e], b[self.i]
        z = b_e - self.B @ self.T.solve(b_i, be=be)
        return be.xp.concatenate([self.T.whiten(b_i, be=be), self.S.whiten(z, be=be)], axis=0)

    def quad(self, b: Any, *, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, be or get_backend(b, self.B))


def schur_factor(T: Any, R: Any, e: Any, i: Any, *, chol_impl: Literal["numpy", "scipy"],
                 be: Backend) -> SchurFactor:
    """Factor ``[[A, B], [B^T, T]]`` from the edge rows ``R`` (WT ``MatrixWdm.solver`` +
    ``Schur.__init__``): ``A = (R_ee + R_ee^T)/2``, ``B = R_ei``, ``TiBt = T^-1 B^T``,
    ``S = A - B TiBt`` symmetrised; ``ok = T.ok & S.ok``.  A concretely failed ``T`` (WT:
    ``LinAlgError`` before the complement) gives a failed factor without ``S``."""
    A = R[:, e]
    A = 0.5 * (A + A.T)
    B = R[:, i]
    if concrete_failure(T.ok):
        return SchurFactor(T, None, B, None, e, i, float("nan"), False)
    TiBt = T.solve(B.T, be=be)
    Sc = A - B @ TiBt
    S = CholFactor.build(0.5 * (Sc + Sc.T), impl=chol_impl, be=be)
    return SchurFactor(T, S, B, TiBt, e, i, T.logdet + S.logdet, T.ok & S.ok)


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

    ``interior`` covers the destination cells minus ``edge`` (band order); ``rows`` is a
    :class:`PushedRows` onto the destination cells whose row cells are ``edge``.
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
            return schur_factor(T, R, e, i, chol_impl=self.chol_impl, be=be)
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
            return self.factor_from_rows(C, nu, self.rows.evaluate(C["rows"], nu, be, P=P), be)
        fns = self.__dict__.setdefault("_jit_cache", {})
        if self.rows.staged and self.rows.phi_cache:
            R = self.rows.evaluate_staged(C["rows"], nu, be, P)
        else:
            if ("rows", P) not in fns:
                fns["rows", P] = be.jit(lambda Cr, nu, P=P: self.rows.evaluate(Cr, nu, be, P=P))
            R = fns["rows", P](C["rows"], nu)
        if "rest" not in fns:
            fns["rest"] = be.jit(lambda C, nu, R: self.factor_from_rows(C, nu, R, be))
        return fns["rest"](C, nu, R)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        R = self.rows.evaluate(C["rows"], nu, be)
        return assemble_edge_rows(self.interior._matrix(C["interior"], nu, be), R, C["e"], C["i"], be)


# ---------------------------------------------------------------------------------------
# Zero padding: the user's M Lambda M
# ---------------------------------------------------------------------------------------


@pytree_dataclass(data=("Lam", "LiWp", "Q", "Wp", "m", "n_idx", "logdet", "ok"),
                  meta=("Nf", "Nt", "dt", "n"))
@dataclass(frozen=True, eq=False)
class PaddedFactor:
    """WT ``PaddedMarginal``: ``C_dd^-1 = Q_dd - Q_dp Q_pp^-1 Q_pd`` on the data samples,
    ``Q = W^T Lambda^-1 W``; vectors are length-``n`` time series (``solve`` only)."""

    Lam: Any
    LiWp: Any
    Q: Any
    Wp: Any
    m: Any
    n_idx: Any
    logdet: Any
    ok: Any
    Nf: int
    Nt: int
    dt: float
    n: int

    n_u = 0

    def solve(self, x: Any, *, be: Backend | None = None) -> Any:
        """WT ``PaddedMarginal.solve`` line for line (``W^T z = dt W^-1 z``)."""
        if self.LiWp is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        be = be or get_backend(x, self.Wp)
        xp = be.xp
        one = x.ndim == 1
        X = x[None, :] if one else x.T                                      # (k, n)
        ext = be.set(xp.zeros((X.shape[0], self.Nf * self.Nt)), (slice(None), slice(0, self.n)), X, copy=False)
        W = forward_time_many(ext, self.Nf, self.Nt, self.dt, be=be)[:, self.m, self.n_idx]
        Liw = self.Lam.solve(W.T, be=be)                                    # (size, k)
        c = self.Q.solve(self.Wp.T @ Liw, be=be)
        Y = (Liw - self.LiWp @ c).T                                         # (k, size)
        full = be.set(xp.zeros((Y.shape[0], self.Nf + 1, self.Nt)), (slice(None), self.m, self.n_idx), Y,
                      copy=False)
        out = self.dt * inverse_time_many(full, self.Nf, self.Nt, self.dt, be=be)[:, :self.n]
        return out[0] if one else out.T

    def whiten(self, b: Any, *, be: Backend | None = None) -> Any:
        raise NotImplementedError("the padded marginal has no whitening; use form='solve'")

    def quad(self, b: Any, *, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, be or get_backend(b, self.Wp))


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
        self.base, self.strategy, self.factor_kind = base_ext, strategy, factor
        self.pad_columns = int(pad_columns)
        self.jittable = base_ext.jittable
        if strategy == "time_marginal":
            if grid is None:
                raise ValueError("time_marginal needs the extended grid")
            if base_ext.n != grid.N:
                raise ValueError(f"the base covers {base_ext.n} cells; the extended grid has {grid.N}")
            self.grid = grid
            self.n = (grid.Nt - self.pad_columns) * grid.Nf
            self.chol_impl = chol_impl or "scipy"
            return
        if mask is None or n_bands is None or n_columns is None:
            raise ValueError("cell_congruence needs mask, n_bands and n_columns")
        self.mask, self.n_bands, self.n_columns = mask, int(n_bands), int(n_columns)
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
        C: dict[str, Any] = {"E": self.e, "I": self.i, "Kpos": np.asarray(self.mask.Kpos, int),
                             "AE": np.asarray(self.mask.AE, float)}
        if self.factor_kind == "schur" and getattr(self.base, "layout", None) is not None:
            sizes = tuple(int(c.size) for c in self.interior_cols)
            padded = self.base.layout == "padded"
            C["interior_layout"] = BandLayout.from_sizes(sizes) if padded else None
            C["interior_dummy"] = dummy_slots(sizes) if padded else None
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
            return PaddedFactor(Lam, None, None, Wp, C["m"], C["n_idx"], float("nan"), False,
                                g.Nf, g.Nt, g.dt, self.n)
        LiWp = Lam.solve(Wp, be=be)
        Qpp = Wp.T @ LiWp
        Q = CholFactor.build(0.5 * (Qpp + Qpp.T), impl=self.chol_impl, be=be)
        logdet = Lam.logdet - g.N * np.log(g.dt) + Q.logdet
        return PaddedFactor(Lam, LiWp, Q, Wp, C["m"], C["n_idx"], logdet, Lam.ok & Q.ok,
                            g.Nf, g.Nt, g.dt, self.n)

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
        return be.set(rows, (slice(None), C["E"]), Y @ C["AE"].T, copy=False)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.strategy == "time_marginal":
            raise NotImplementedError("time_marginal has no cell-space matrix (its data are samples)")
        lamKK, Y, _ = self._congruence_pieces(C, nu, be)
        rows = self._edge_rows(C, Y, be)
        lam = be.set(lamKK, (C["E"], slice(None)), rows)
        return be.set(lam, (slice(None), C["E"]), rows.T, copy=False)

    def _interior_factor(self, C: Any, lamKK: Any, blocks: Any, be: Backend) -> Any:
        """The interior's own factor: diagonal, or block-tridiagonal over the analysis bands."""
        if blocks is None:
            return DiagFactor.build(be.xp.diagonal(lamKK)[self.i], be)
        d, u = blocks
        k0, cols = self.mask.k0, self.interior_cols
        diag = [d[k0 + b][np.ix_(c, c)] for b, c in enumerate(cols)]
        up = [u[k0 + b][np.ix_(cols[b], cols[b + 1])] for b in range(len(cols) - 1)]
        sizes = tuple(int(c.size) for c in cols)
        return blocktri_cov_factor(diag, up, sizes, self.base.layout, C.get("interior_layout"), be,
                                   C.get("interior_dummy"))

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.strategy == "time_marginal":
            return self._time_factor(C, nu, be)
        if self.factor_kind == "dense":
            return dense_factor(self._matrix(C, nu, be), chol_impl=self.chol_impl, symmetrize=True, be=be)
        lamKK, Y, blocks = self._congruence_pieces(C, nu, be)
        R = self._edge_rows(C, Y, be)
        T = self._interior_factor(C, lamKK, blocks, be)
        return schur_factor(T, R, C["E"], C["I"], chol_impl=self.chol_impl, be=be)
