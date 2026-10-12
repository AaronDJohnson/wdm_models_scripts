r"""``HermitianCovariance``: a complex Hermitian covariance of Fourier bins (GB ``exact``).

GB ``fullcache.noise_state``: dense complex Cholesky at 3 and 6 months; at 12 months the
envelope is periodic over the record, :math:`\hat g` is at round-off beyond a few tens of
bins and :math:`M'` is block tri-diagonal in blocks of ``NT/2`` bins, factorised by the
block Cholesky ``linalg.cblock_factor`` (exact to round-off, ``O(n b^2)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from wdm.backend import Backend, get_backend, pytree_dataclass

from ..linalg import cblock_backward, cblock_factor, cblock_forward
from .base import Covariance, QuadForm, as_evaluator

__all__ = ["HermitianCovariance", "HermitianFactor"]


@pytree_dataclass(data=("L", "blocks", "logdet", "ok"), meta=())
@dataclass(frozen=True, eq=False)
class HermitianFactor:
    """``M = L L^H`` (``L`` dense) or the block factor ``blocks`` (``linalg.CBlockFactor``)."""

    L: Any
    blocks: Any
    logdet: Any
    ok: Any

    n_u = 0

    def whiten(self, b: Any, *, be: Backend | None = None) -> Any:
        """``L^-1 b`` (GB ``fullcache.forward``)."""
        if self.blocks is not None:
            return cblock_forward(self.blocks, b, be=be)
        be = be or get_backend(b, self.L)
        return be.solve_triangular(self.L, b, lower=True)

    def backward(self, u: Any, *, be: Backend | None = None) -> Any:
        """``L^-H u`` (GB ``fullcache.backward``)."""
        if self.blocks is not None:
            return cblock_backward(self.blocks, u, be=be)
        be = be or get_backend(u, self.L)
        return be.solve_triangular(self.L, u, lower=True, trans=2)

    def solve(self, b: Any, *, be: Backend | None = None) -> Any:
        return self.backward(self.whiten(b, be=be), be=be)

    def quad(self, b: Any, *, form: QuadForm = "whiten", be: Backend | None = None) -> Any:
        """``Re(b^H M^-1 b)``: ``"whiten"`` = ``Re(vdot(w, w))`` (GB ``noise_state`` ``q0``)."""
        be = be or get_backend(b)
        xp = be.xp
        if form == "whiten":
            w = self.whiten(b, be=be)
            return xp.real(xp.vdot(w, w))
        if form == "solve":
            return xp.real(xp.vdot(b, self.solve(b, be=be)))
        raise ValueError(f"HermitianFactor has no quadratic form {form!r}")


class HermitianCovariance(Covariance):
    """``M(nu)`` complex Hermitian ``(n, n)``; ``block=None`` dense Cholesky, ``block=b``
    the block-tridiagonal Cholesky with ``b``-bin blocks (``n % b == 0``).

    ``matrix`` is an evaluator (``ExactFourierCovariance``), a callable ``fn(nu, be)`` or
    a fixed array.  ``logdet = 2 sum log Re diag L`` (GB ``logdet_of``).
    """

    def __init__(self, matrix: Any, n: int | None = None, *, block: int | None = None) -> None:
        self.matrix = as_evaluator(matrix)
        self.n = int(getattr(matrix, "n", n) if n is None else n)
        if block is not None and self.n % int(block):
            raise ValueError(f"{self.n} bins are not whole blocks of {block}")
        self.block = None if block is None else int(block)

    def host_constants(self) -> Any:
        return self.matrix.host_constants()

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.matrix.evaluate(C, nu, be)

    def _factor(self, C: Any, nu: Any, be: Backend) -> HermitianFactor:
        M = self._matrix(C, nu, be)
        xp = be.xp
        if self.block is not None:
            F = cblock_factor(M, self.block, be=be)
            return HermitianFactor(None, F, F.logdet, F.ok)
        L, ok = be.cholesky(M, impl="numpy")
        return HermitianFactor(L, None, 2.0 * xp.sum(xp.log(xp.real(xp.diagonal(L)))), ok)

    def block_is_exact(self, nu: Any, rtol: float = 1e-13) -> bool:
        """Whether ``|ghat|`` beyond one block is below ``rtol |ghat[0]|`` (GB
        ``fullcache.block_is_exact``), so the block factor is exact to round-off."""
        if self.block is None:
            raise ValueError("block_is_exact needs block=b")
        b = self.block
        gh = np.abs(np.asarray(self.matrix.ghat(np.asarray(nu), np.arange(b, 2 * b + 1))))
        return bool(gh.max() < rtol * abs(np.asarray(self.matrix.ghat(np.asarray(nu), np.array([0])))[0]))
