"""Test-local toys for the section D-D unit tests.

Covariances are section D-C's own classes (``DenseCovariance``, ``HermitianCovariance``,
``DiagonalCovariance``) fed by tiny evaluators (``host_constants()`` + the pure
``evaluate(C, nu, be)``), so the likelihood and sampling tests exercise the real
covariance protocol with the GB ``matrix.py`` arithmetic (``np.linalg.cholesky`` of
``0.5 (S + S^T)``, ``solve_triangular``).  Maps and signals are small JAX pytrees."""

from __future__ import annotations

import numpy as np

from domainia.covariance import DenseCovariance, DiagonalCovariance, HermitianCovariance
from wdm.backend import get_backend, pytree_dataclass


class AffineMatrix:
    """``S(nu) = exp(nu[0]) K0 + nu[1] I`` -- not positive definite once ``nu[1]`` is below
    ``-exp(nu[0]) lambda_min(K0)``."""

    def __init__(self, K0):
        self.K0 = np.asarray(K0, dtype=np.float64)

    def host_constants(self):
        return {"K0": self.K0, "I": np.eye(self.K0.shape[0])}

    def evaluate(self, C, nu, be):
        return be.xp.exp(nu[0]) * C["K0"] + nu[1] * C["I"]


class ScaledArray:
    """``exp(nu[0]) A0`` (a variance vector or a complex Hermitian matrix)."""

    def __init__(self, A0):
        self.A0 = np.asarray(A0)

    def host_constants(self):
        return self.A0

    def evaluate(self, C, nu, be):
        return be.xp.exp(nu[0]) * C


def dense_cov(K0, n):
    """GB ``_dense_quad_logdet`` / ``MatrixCache``: ``np.linalg.cholesky(0.5 (S + S^T))``."""
    return DenseCovariance(AffineMatrix(K0), n, chol_impl="numpy", symmetrize=True)


def hermitian_cov(M0, n):
    """GB ``fullcache``: ``M = exp(nu[0]) M0``, dense complex Cholesky."""
    return HermitianCovariance(ScaledArray(M0), n)


def diag_cov(v0, n):
    """WT ``Solver(diag=v)``: ``v = exp(nu[0]) v0``."""
    return DiagonalCovariance(ScaledArray(v0), n)


@pytree_dataclass(data=("C",), meta=())
class ToyMap:
    """``x -> x @ C.T`` over the last axis (real ``C``)."""

    C: object

    def apply(self, x, be=None):
        return x @ self.C.T


@pytree_dataclass(data=("C",), meta=())
class ToyReMap:
    """``z -> Re(z @ C.T)``: complex bins to real cells (the GB ``Re(C z)`` shape)."""

    C: object

    def apply(self, x, be=None):
        be = be or get_backend(x, self.C)
        return be.xp.real(x @ self.C.T)


@pytree_dataclass(data=(), meta=("start", "stop"))
class ToySelect:
    """``x -> x[..., start:stop]`` (a bin window)."""

    start: int
    stop: int

    def apply(self, x, be=None):
        return x[..., self.start:self.stop]


@pytree_dataclass(data=("B",), meta=("shape",))
class ToySignal:
    """``source -> (source @ B).reshape(shape)`` (``shape=None``: as is)."""

    B: object
    shape: tuple | None = None

    def __call__(self, src):
        h = src @ self.B
        return h if self.shape is None else h.reshape(self.shape)


def spd(n, rng, cond=10.0):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w = np.geomspace(1.0, cond, n)
    K = (Q * w) @ Q.T
    return 0.5 * (K + K.T)


def hpd(n, rng):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = A @ A.conj().T / n + np.eye(n)
    return 0.5 * (M + M.conj().T)
