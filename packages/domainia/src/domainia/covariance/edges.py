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

from .base import CholFactor, Covariance, QuadForm, concrete_failure, quad_form
from .dense import dense_factor
from .pushed import PushedRows, positions

__all__ = ["SchurEdge", "SchurFactor", "assemble_edge_rows", "cut", "schur_factor"]


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
