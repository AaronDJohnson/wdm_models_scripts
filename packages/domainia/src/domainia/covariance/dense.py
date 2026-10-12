"""``DenseCovariance``: an assembled matrix factorised by a pinned Cholesky (spec §4.4)."""

from __future__ import annotations

from typing import Any, Literal

from wdm.backend import Backend

from .base import CholFactor, Covariance, as_evaluator

__all__ = ["DenseCovariance", "dense_factor"]


def dense_factor(S: Any, *, chol_impl: Literal["numpy", "scipy"], symmetrize: bool,
                 be: Backend) -> CholFactor:
    """Cholesky of ``S`` (of ``0.5 (S + S^T)`` when ``symmetrize``).

    GB ``_dense_quad_logdet`` / ``MatrixCache``: ``np.linalg.cholesky(0.5 * (S + S.T))``
    (``chol_impl="numpy"``, ``symmetrize=True``).  WT ``Solver(dense=...)``:
    ``cho_factor(S)`` (``"scipy"``, ``symmetrize=False``).
    """
    if symmetrize:
        S = 0.5 * (S + S.T)
    return CholFactor.build(S, impl=chol_impl, be=be)


class DenseCovariance(Covariance):
    """``C = S(nu)`` as a dense matrix.

    ``matrix`` is an :class:`~domainia.covariance.base.Evaluator` returning ``(n, n)``, a
    callable ``fn(nu, be)`` or a fixed array.  ``chol_impl`` pins the Cholesky of the
    oracle (``"scipy"``: WT; ``"numpy"``: GB) and ``symmetrize`` whether ``0.5 (S + S^T)``
    is factorised (GB) or ``S`` itself (WT).
    """

    def __init__(self, matrix: Any, n: int, *, chol_impl: Literal["numpy", "scipy"],
                 symmetrize: bool = True) -> None:
        if chol_impl not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        self.matrix = as_evaluator(matrix)
        self.n = int(n)
        self.chol_impl = chol_impl
        self.symmetrize = bool(symmetrize)

    def host_constants(self) -> Any:
        return self.matrix.host_constants()

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        S = self.matrix.evaluate(C, nu, be)
        if S.shape != (self.n, self.n):
            raise ValueError(f"the matrix has shape {S.shape}; expected ({self.n}, {self.n})")
        return S

    def _factor(self, C: Any, nu: Any, be: Backend) -> CholFactor:
        return dense_factor(self._matrix(C, nu, be), chol_impl=self.chol_impl,
                            symmetrize=self.symmetrize, be=be)
