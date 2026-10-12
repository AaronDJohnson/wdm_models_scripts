"""D-C5: ExactTimeCovariance (Eq. 47, theta = 0) -- the WT fastcov / noise.py port.

* The operator (no endpoint weights: irfft supplies them) is the dense sigma_tilde
  (WT test_fastcov.py); sigma_tilde is the voice factor's G G^T and S1's
  ``time_domain_covariance``; PCG solves it.
* The preconditioner is the Weyl inverse: exact for stationary noise.
* log|Sigma~| = Weyl + c_N alpha_d^2 over the prior box (c_N fitted as WT does), and G6:
  Weyl / full log-det at the manuscript truth with the shipped c_N.
* draw consumes the cosine voices first; JAX parity of the operator and the jitted solve.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor
from wdm.backend import NUMPY
from wdm.evolutionary import time_domain_covariance, time_domain_factor

import domtest_inputs as T
from domainia.covariance import ExactTimeCovariance, OperatorFactor
from domainia.spectra import LogLinearPSD

N, DT = 1024, 16.0 / 1024                                  # WT test_fastcov: Nf 16 x Nt 64
PSD = LogLinearPSD(f_floor=4.0, f_ref=1.0, T=N * DT)


def _cov(**kw) -> ExactTimeCovariance:
    return ExactTimeCovariance(PSD, N, DT, **kw)


@pytest.mark.parametrize("nu", [(-2.0, 1.2), (-4.0, 3.0), (-0.1, -3.0)])
def test_operator_is_the_dense_covariance(nu, rng):
    nu = np.asarray(nu)
    cov = _cov()
    Sig = cov.sigma_tilde(nu)
    A = cov.operator(cov.constants(), nu[0], nu[1], NUMPY)
    x = rng.standard_normal(N)
    assert np.max(np.abs(A(x) - Sig @ x)) <= 1e-12 * np.max(np.abs(Sig @ x))


def test_sigma_tilde_is_the_voice_factor_and_s1(rng):
    nu = np.array([-2.0, 1.2])
    cov = _cov(form="dense")
    Sig = cov.sigma_tilde(nu)
    G = cov.voice_factor(nu)
    assert G.shape == (N, N)
    assert np.max(np.abs(Sig - G @ G.T)) <= 1e-13 * np.max(np.abs(Sig))
    Sn = lambda f, t: PSD(f, t, nu)                            # noqa: E731
    assert np.allclose(G, time_domain_factor(N, DT, Sn), rtol=1e-14, atol=0)
    assert np.max(np.abs(Sig - time_domain_covariance(N, DT, Sn))) <= 1e-13 * np.max(np.abs(Sig))
    z = np.random.default_rng(5).standard_normal(N)
    assert np.array_equal(cov.draw(nu, np.random.default_rng(5)), G @ z)


def test_operator_factor_solves_by_pcg(rng):
    nu = np.array([-1.3, 1.7])
    cov = _cov(c_N=0.0, tol=1e-11)
    F = cov.factor(nu)
    assert isinstance(F, OperatorFactor) and bool(F.ok) and F.n_u == 0
    b = rng.standard_normal(N)
    want = np.linalg.solve(cov.sigma_tilde(nu), b)
    assert np.max(np.abs(F.solve(b) - want)) <= 1e-8 * np.max(np.abs(want))
    assert float(F.quad(b)) == pytest.approx(b @ want, rel=1e-9)
    with pytest.raises(NotImplementedError):
        F.whiten(b)


def test_preconditioner_is_exact_for_stationary_noise(rng):
    cov = _cov()
    C = cov.constants()
    for ac in (-2.0, -0.5):
        A = cov.operator(C, ac, 0.0, NUMPY)
        M = cov.operator(C, -ac, 0.0, NUMPY, scale=4.0 * DT ** 2)
        x = rng.standard_normal(N)
        assert np.max(np.abs(M(A(x)) - x)) <= 1e-10 * np.max(np.abs(x))


def test_logdet_closed_form_over_the_prior_box():
    """c_N from one Cholesky at alpha_d = 3 (WT FastCovariance._load_or_compute_cN), then
    Weyl + c_N alpha_d^2 within 1e-5 of the exact log-det over the box."""
    cov = _cov()
    cf = cho_factor(cov.sigma_tilde(np.array([-2.0, 3.0])), lower=True, check_finite=False)
    exact3 = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    c_N = (exact3 - cov.weyl_logdet(np.array([-2.0, 3.0]))) / 9.0
    fitted = _cov(c_N=c_N)
    for nu in ((-4.0, 3.0), (-0.1, -3.0), (-2.0, 0.0), (-1.3, 1.7)):
        cf = cho_factor(cov.sigma_tilde(np.asarray(nu)), lower=True, check_finite=False)
        exact = 2.0 * np.sum(np.log(np.diag(cf[0])))
        assert abs(fitted.logdet_model(np.asarray(nu)) - exact) < 1e-5
    with pytest.raises(ValueError, match="c_N"):
        cov.factor(np.array([-2.0, 1.2]))


def test_dense_form_is_the_wt_dense_model(rng):
    """form='dense': cho_factor(sigma_tilde) without symmetrisation (WT WdmFullCovariance)."""
    nu = np.array([-2.0, 1.2])
    cov = _cov(form="dense")
    F = cov.factor(nu)
    Sig = cov.sigma_tilde(nu)
    b = rng.standard_normal(N)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(Sig)[1], rel=1e-12)
    assert float(F.quad(b)) == pytest.approx(b @ np.linalg.solve(Sig, b), rel=1e-9)


def test_g6_weyl_and_full_logdet_at_the_truth():
    ref = T.sinusoid_mid()["g6"]["parts"]["exact"]
    c_N = float(np.load(T.DATA / "sinusoid" / "cache" / "cN_N16384_dt0.0009765625_fmin4_fref1.npy"))
    assert c_N == T.num(ref["c_N"])
    cov = ExactTimeCovariance(LogLinearPSD(f_floor=4.0, f_ref=1.0, T=T.SIN_T), T.SIN_N, T.SIN_DT, c_N=c_N)
    assert cov.weyl_logdet(T.SIN_NU) == pytest.approx(T.num(ref["weyl"]), rel=1e-14, abs=0)
    assert cov.logdet_model(T.SIN_NU) == pytest.approx(T.num(ref["full_logdet"]), rel=1e-14, abs=0)


def test_requires_a_log_linear_psd():
    class Other:
        capabilities = frozenset({"separable"})

    with pytest.raises(ValueError, match="log-linear"):
        ExactTimeCovariance(Other(), N, DT)
    # the other constructor and operator guards fail at construction / call, not inside PCG
    with pytest.raises(ValueError, match="on_fail"):
        _cov(on_fail="ignore")
    with pytest.raises(ValueError, match="even"):
        ExactTimeCovariance(PSD, N + 1, DT)
    with pytest.raises(ValueError, match="P >= 1"):
        _cov(P=0)
    cov = _cov(P=8)
    with pytest.raises(ValueError, match=r"P must be in \[1, 8\]"):
        cov.operator(cov.constants(), -2.0, 1.2, NUMPY, P=9)
    dense = _cov(form="dense")
    with pytest.raises(ValueError, match="form='operator'"):
        dense.operator(dense.constants(), -2.0, 1.2, NUMPY)


@pytest.mark.jax
def test_jax_operator_and_jitted_solve(jax_be, rng):
    import jax.numpy as jnp

    nu = np.array([-2.0, 1.2])
    cov = _cov(c_N=0.1, tol=1e-11, on_fail="nan")
    assert cov.jittable
    x = rng.standard_normal(N)
    An = cov.operator(cov.constants(), nu[0], nu[1], NUMPY)
    Aj = cov.operator(cov.constants(jax_be), jnp.asarray(nu[0]), jnp.asarray(nu[1]), jax_be)
    assert np.max(np.abs(np.asarray(Aj(jnp.asarray(x), jax_be)) - An(x))) <= 1e-12 * np.max(np.abs(An(x)))
    Fn, Fj = cov.factor(nu), cov.factor(jnp.asarray(nu))
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-14)
    assert float(Fj.quad(jnp.asarray(x))) == pytest.approx(float(Fn.quad(x)), rel=1e-9)
