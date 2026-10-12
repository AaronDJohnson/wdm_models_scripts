"""D-C1: the covariance protocol, DiagFactor / CholFactor, DiagonalCovariance, DenseCovariance.

* Every factor is the dense Gaussian: solve, whiten, the three quadratic-form orders,
  log-determinant -- on both backends.
* ``ok`` is False (the likelihood's -inf) on a covariance that is not positive definite.
* Under JAX the constants reach the compiled factorisation as an argument.
* G6: the sinusoid band/mid raw (quad_C, logdet_C) and the GB band/mid raw
  per-channel (quad, logdet) at the golden points.
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY

import domtest_inputs as T
from domainia.covariance import (CholFactor, Covariance, DenseCovariance, DiagFactor, DiagonalCovariance,
                                 Factor, as_evaluator)

NU0 = np.zeros(2)


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A @ A.T / n + np.eye(n)


def test_diag_factor_is_the_dense_gaussian(backend, rng):
    v = rng.uniform(0.5, 2.0, 12)
    b, B = rng.standard_normal(12), rng.standard_normal((12, 3))
    F = DiagonalCovariance(v, 12).factor(backend.asarray(NU0), backend)
    assert isinstance(F, (DiagFactor, Factor)) and F.n_u == 0 and bool(F.ok)
    C = np.diag(v)
    assert np.allclose(np.asarray(F.solve(backend.asarray(b))), np.linalg.solve(C, b), rtol=1e-14)
    assert np.allclose(np.asarray(F.solve(backend.asarray(B))), np.linalg.solve(C, B), rtol=1e-14)
    want = b @ np.linalg.solve(C, b)
    for form in ("solve", "whiten", "elementwise"):
        assert float(F.quad(backend.asarray(b), form=form)) == pytest.approx(want, rel=1e-14)
    w = np.asarray(F.whiten(backend.asarray(b)))
    assert w @ w == pytest.approx(want, rel=1e-14)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(C)[1], rel=1e-14)


@pytest.mark.parametrize("impl", ["numpy", "scipy"])
def test_chol_factor_is_the_dense_gaussian(backend, rng, impl):
    S = _spd(rng, 10)
    b, B = rng.standard_normal(10), rng.standard_normal((10, 4))
    F = DenseCovariance(S, 10, chol_impl=impl).factor(backend.asarray(NU0), backend)
    assert isinstance(F, CholFactor) and F.impl == impl and bool(F.ok)
    assert np.allclose(np.asarray(F.solve(backend.asarray(B))), np.linalg.solve(S, B), rtol=1e-12)
    want = b @ np.linalg.solve(S, b)
    for form in ("solve", "whiten"):
        assert float(F.quad(backend.asarray(b), form=form)) == pytest.approx(want, rel=1e-12)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(S)[1], rel=1e-13)
    with pytest.raises(ValueError, match="quadratic form"):
        F.quad(backend.asarray(b), form="elementwise")


def test_symmetrize_selects_the_oracle_matrix(rng):
    """``symmetrize=False`` factorises S's lower triangle (WT cho_factor);
    ``symmetrize=True`` factorises 0.5 (S + S^T) (GB)."""
    S = _spd(rng, 8)
    junk = S + 0.1 * np.triu(rng.standard_normal((8, 8)), 1)      # upper triangle perturbed (|.| << 1: PD)
    low = np.tril(junk) + np.tril(junk, -1).T
    F0 = DenseCovariance(junk, 8, chol_impl="scipy", symmetrize=False).factor(NU0)
    assert float(F0.logdet) == pytest.approx(np.linalg.slogdet(low)[1], rel=1e-13)
    F1 = DenseCovariance(junk, 8, chol_impl="numpy", symmetrize=True).factor(NU0)
    assert float(F1.logdet) == pytest.approx(np.linalg.slogdet(0.5 * (junk + junk.T))[1], rel=1e-13)


def test_not_positive_definite_sets_ok_false(backend):
    v = np.array([1.0, -1.0, 2.0])
    with np.errstate(invalid="ignore"):
        F = DiagonalCovariance(v, 3).factor(backend.asarray(NU0), backend)
    assert not bool(F.ok)
    S = np.diag([1.0, -1.0, 2.0])
    for impl in ("numpy", "scipy"):
        F = DenseCovariance(S, 3, chol_impl=impl).factor(backend.asarray(NU0), backend)
        assert not bool(F.ok) and not np.isfinite(float(F.logdet))


def test_variances_from_callables_and_evaluators(backend, rng):
    base = rng.uniform(0.5, 2.0, 6)

    class Scaled:                                                   # an Evaluator
        def host_constants(self):
            return {"base": base}

        def evaluate(self, C, nu, be):
            return C["base"] * be.xp.exp(nu[0])

    nu = np.array([0.3, 0.0])
    for variance in (Scaled(), lambda nu, be: be.asarray(base) * be.xp.exp(nu[0])):
        cov = DiagonalCovariance(variance, 6)
        assert isinstance(cov, Covariance)
        F = cov.factor(backend.asarray(nu), backend)
        assert np.allclose(np.asarray(F.v), base * np.exp(0.3), rtol=1e-15)
        assert np.allclose(cov.dense(nu), np.diag(base * np.exp(0.3)), rtol=1e-15)
    with pytest.raises(ValueError, match="shape"):
        DiagonalCovariance(base, 5).factor(NU0)
    assert as_evaluator(base).evaluate(base, NU0, NUMPY) is base
    # D-B's variances are pytree callables v(nu, *, channel, be), their own constants
    from domainia.domains import CellSet
    from domainia.spectra import LogLinearPSD
    from domainia.variance import BandAveragedVariance

    cells = CellSet.all_active(wdm.Grid(8, 16, 1 / 16))
    var = BandAveragedVariance(LogLinearPSD(f_floor=0.5, f_ref=1.0, T=8.0), cells)
    cov = DiagonalCovariance(var, len(cells.m))
    assert cov.constants() is var
    F = cov.factor(backend.asarray(nu), backend)
    assert np.allclose(np.asarray(F.v), var(nu), rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="pytree callables"):
        DiagonalCovariance(base, 6, channel=1)


@pytest.mark.jax
def test_jit_takes_the_constants_as_an_argument(jax_be):
    """A 2**20-entry constant must not be baked into the compiled program."""
    import jax.numpy as jnp

    big = np.linspace(1.0, 2.0, 1 << 20)

    class Big:
        def host_constants(self):
            return {"big": big}

        def evaluate(self, C, nu, be):
            return C["big"][:16] * be.xp.exp(nu[0])

    cov = DiagonalCovariance(Big(), 16)
    F = cov.factor(jnp.asarray([0.1, 0.0]))
    assert type(F.logdet).__module__.startswith("jax")
    fn = cov._jit_cache["factor"]
    text = fn.lower(cov.constants(jax_be), jnp.asarray([0.1, 0.0])).as_text()
    assert len(text) < 200_000
    assert float(F.logdet) == pytest.approx(float(cov.factor(np.array([0.1, 0.0])).logdet), rel=1e-15)


def test_g6_sinusoid_band_and_mid_raw():
    """quad_C = y^T C^-1 y and logdet_C at the manuscript_mid truth (WT operation order)."""
    s = T.sinusoid_mid()
    y = T.sinusoid_cells_residual()
    for base in ("band", "mid"):
        v = s["arrays"][f"{base}_variance_raw"]
        F = DiagonalCovariance(v, v.size).factor(T.SIN_NU)
        ref = s["g6"]["parts"][f"{base}_raw"]
        assert float(F.quad(y, form="solve")) == pytest.approx(T.num(ref["quad_C"]), rel=1e-13, abs=0)
        assert float(F.logdet) == pytest.approx(T.num(ref["logdet_C"]), rel=1e-13, abs=0)


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo", "gb_12mo"])
def test_g6_gb_band_and_mid_raw(case):
    """GB band/mid raw per channel: (sum r^2 / v, sum log v) at the anchor (elementwise order)."""
    c = T.gb_case(case)
    r = c.residual_cells(c.cells)
    assert np.allclose(r + np.real(c.A["store_template"][:, slice(*_cols(c))] @ c.atoms(c.cells, *c.grid_window()).T),
                       c.A["V_raw_cells"], rtol=0, atol=1e-12)
    from domainia.variance import BandAveragedVariance, MidpointVariance

    for base, Var in (("band", BandAveragedVariance), ("mid", MidpointVariance)):
        v = c.A[f"var_{base}_raw"]
        got = np.array([[float(F.quad(r[ch], form="elementwise")), float(F.logdet)]
                        for ch, F in ((ch, DiagonalCovariance(v[ch], v.shape[1]).factor(NU0)) for ch in range(2))])
        assert np.allclose(got, c.quad_logdet(f"{base}_raw"), rtol=1e-12, atol=0)
        var = Var(c.psd, c.cells)                                  # D-B's separable variance, per channel
        got = np.array([[float(F.quad(r[ch], form="elementwise")), float(F.logdet)]
                        for ch, F in ((ch, DiagonalCovariance(var, v.shape[1], channel=ch).factor(c.mu[ch]))
                                      for ch in range(2))])
        assert np.allclose(got, c.quad_logdet(f"{base}_raw"), rtol=1e-12, atol=0)


def _cols(c: T.GBCase) -> tuple[int, int]:
    g0, G = c.grid_window()
    return g0 - c.store_k0, g0 - c.store_k0 + G
