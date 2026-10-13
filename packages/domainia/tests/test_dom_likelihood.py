"""D-D2: GaussianLikelihood against transcriptions of its oracles (GB
``matrix.make_lnlike``, ``MatrixCache``, ``fullcache``; WT ``marginal_lnlike``), on the
test-local toys of ``dom_toys`` (NumPy bitwise, JAX <= 1e-12)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from dom_toys import ToyMap, ToyReMap, ToySelect, ToySignal, dense_cov, diag_cov, hermitian_cov, hpd, spd
from domainia.covariance.stack import ChannelStack
from domainia.density import LOG_2PI, ComplexGaussian, RealGaussian
from domainia.likelihood import COUNTERS, GaussianLikelihood, NoiseState, Params, StaticSignal

P, NB, NSRC = 24, 30, 3


def gb_like(rng, strategy="residual", backend=None):
    """Two channels, real cells Re(C z) from complex bins, dense covariance per channel."""
    C = rng.standard_normal((P, NB)) + 1j * rng.standard_normal((P, NB))
    B = rng.standard_normal((NSRC, 2 * NB)) + 1j * rng.standard_normal((NSRC, 2 * NB))
    z = rng.standard_normal((2, NB)) + 1j * rng.standard_normal((2, NB))
    K = [spd(P, rng), spd(P, rng)]
    params = Params(("s0", "s1", "s2", "a0", "b0", "a1", "b1"), (0, 1, 2), (3, 4, 5, 6))
    covs = ChannelStack((dense_cov(K[0], P), dense_cov(K[1], P)), 2)
    lik = GaussianLikelihood(z, ToyReMap(C), covs, RealGaussian(), signal=ToySignal(B, (2, NB)),
                             params=params, strategy=strategy, quad_form="whiten", backend=backend)
    return lik, dict(C=C, B=B, z=z, K=K)


def gb_reference(theta, C, B, z, K):
    """galactic_binary/matrix.py:911-925 + _dense_quad_logdet (:928-938), transcribed."""
    d = np.real(z @ C.T)
    r = d - np.real((theta[:3] @ B).reshape(2, NB) @ C.T)
    total = 0.0
    for c in range(2):
        a, b = theta[3 + 2 * c], theta[4 + 2 * c]
        S = np.exp(a) * K[c] + b * np.eye(P)
        try:
            L = np.linalg.cholesky(0.5 * (S + S.T))
        except np.linalg.LinAlgError:
            return -np.inf
        v = solve_triangular(L, r[c], lower=True, check_finite=False)
        total = total - 0.5 * (np.dot(v, v) + 2.0 * np.sum(np.log(np.diagonal(L))) + P * LOG_2PI)
    return total if np.isfinite(total) else -np.inf


def matrix_cache_reference(theta, C, B, z, K):
    """MatrixCache.state / .signal (galactic_binary/matrix.py:978-994), transcribed."""
    d = np.real(z @ C.T)
    hc = np.real((theta[:3] @ B).reshape(2, NB) @ C.T)
    total = 0.0
    for c in range(2):
        a, b = theta[3 + 2 * c], theta[4 + 2 * c]
        S = np.exp(a) * K[c] + b * np.eye(P)
        L = np.linalg.cholesky(0.5 * (S + S.T))
        u = solve_triangular(L, d[c], lower=True, check_finite=False)
        y = solve_triangular(L, u, lower=True, trans="T", check_finite=False)
        q0, logdet = np.dot(u, u), 2.0 * np.sum(np.log(np.diagonal(L))) + P * LOG_2PI
        t = solve_triangular(L, hc[c], lower=True, check_finite=False)
        total = total - 0.5 * (q0 - 2.0 * np.dot(hc[c], y) + np.dot(t, t) + logdet)
    return total


def rows(rng, n):
    X = np.column_stack([rng.standard_normal((n, 3)), rng.uniform(-0.3, 0.3, n),
                         rng.uniform(0.0, 0.5, n), rng.uniform(-0.3, 0.3, n), rng.uniform(0.0, 0.5, n)])
    return X


def test_params_validation():
    p = Params(("a", "b", "c"), (0,), (2, 1))
    assert (p.ndim, p.index("c")) == (3, 2)
    with pytest.raises(ValueError, match="unique"):
        Params(("a", "a"), (0,), (1,))
    with pytest.raises(ValueError, match="out of range"):
        Params(("a", "b"), (0,), (2,))
    with pytest.raises(ValueError, match="distinct"):
        Params(("a", "b"), (0,), (0,))
    with pytest.raises(KeyError, match="no parameter 'z'"):
        p.index("z")


def test_residual_strategy_is_gb_make_lnlike_bitwise():
    rng = np.random.default_rng(0)
    lik, k = gb_like(rng)
    X = rows(rng, 6)
    got = lik(X)
    want = np.array([gb_reference(x, **k) for x in X])
    assert np.array_equal(got, want)
    assert lik.counters() == {"noise_states": 6, "pd_failed_states": 0, "pd_failed_rows": 0,
                              "evaluations": 6, "cache_hits": 0}
    assert tuple(lik.counters())[:4] == COUNTERS


def test_expanded_strategy_is_matrix_cache_bitwise_and_equals_residual():
    rng = np.random.default_rng(1)
    lik, k = gb_like(rng, strategy="expanded")
    X = rows(rng, 5)
    got = lik(X)
    assert np.array_equal(got, np.array([matrix_cache_reference(x, **k) for x in X]))
    ref = np.array([gb_reference(x, **k) for x in X])
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-12


def test_complex_expanded_is_fullcache_bitwise():
    rng = np.random.default_rng(2)
    n = 20
    M0 = [hpd(n, rng), hpd(n, rng)]
    B = rng.standard_normal((NSRC, 2 * n)) + 1j * rng.standard_normal((NSRC, 2 * n))
    z = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
    params = Params(("s0", "s1", "s2", "m0", "m1"), (0, 1, 2), (3, 4))
    cov = ChannelStack((hermitian_cov(M0[0], n), hermitian_cov(M0[1], n)), 1)
    lik = GaussianLikelihood(z, ToySelect(0, n), cov, ComplexGaussian(), signal=ToySignal(B, (2, n)),
                             params=params, strategy="expanded")
    X = np.column_stack([rng.standard_normal((4, 3)), rng.uniform(-0.5, 0.5, (4, 2))])
    for x in X:
        h = (x[:3] @ B).reshape(2, n)
        total = 0.0
        for c in range(2):                                  # fullcache.py:128-144 + :175-183
            L = np.linalg.cholesky(np.exp(x[3 + c]) * M0[c])
            u = solve_triangular(L, z[c], lower=True, check_finite=False)
            q0 = np.real(np.vdot(u, u))
            logdet = 2.0 * np.sum(np.log(np.real(np.diagonal(L))))
            y = solve_triangular(L, u, lower=True, trans="C", check_finite=False)
            t = solve_triangular(L, h[c], lower=True, check_finite=False)
            total = total + (-(q0 - 2.0 * np.real(np.vdot(h[c], y)) + np.real(np.vdot(t, t))) - logdet)
        assert lik(x[None])[0] == total


def test_amplitude_outside_the_factorisation_is_wt_order():
    """Single channel, A outside, log-Jacobian last: wdm_toys/matrix.py:654-669 without u."""
    rng = np.random.default_rng(4)
    n = 40
    v0 = rng.uniform(0.5, 2.0, n)
    W = rng.standard_normal((n, n))
    data = rng.standard_normal(n)
    B = rng.standard_normal((2, n))
    params = Params(("s0", "s1", "log10_A_n", "a"), (0, 1), (3,))
    log_jac = 0.5 * 16384 * float(np.log(2.0 ** -10))
    lik = GaussianLikelihood(data, ToyMap(W), diag_cov(v0, n), RealGaussian(log_jac, "log10_A_n"),
                             signal=ToySignal(B), params=params)
    x = np.array([0.3, -0.2, 2.6020599913279625, 0.1])
    y = data @ W.T - (x[:2] @ B) @ W.T                       # y_data - map(h)
    v = np.exp(x[3]) * v0
    quad, logdet, A_n = float(y @ (y / v)), float(np.sum(np.log(v))), 10.0 ** x[2]
    want = -0.5 * (quad / A_n + logdet + n * np.log(A_n) + n * LOG_2PI) + log_jac
    got = lik(x[None])[0]
    assert got == want                                       # DiagFactor quad 'solve' = WT y @ (y / v)


def test_flat_marginal_through_nuisance_is_wt_marginal_lnlike_bitwise():
    """``nuisance=U`` is ``FlatMarginal(cov, U)``: wdm_toys/matrix.py:654-669 verbatim."""
    from domainia.nuisance import FlatMarginal

    rng = np.random.default_rng(5)
    n, nu_ = 50, 4
    v0 = rng.uniform(0.5, 2.0, n)
    U = rng.standard_normal((n, nu_))
    data = rng.standard_normal(n)
    B = rng.standard_normal((2, n))
    params = Params(("s0", "s1", "log10_A_n", "a"), (0, 1), (3,))
    lik = GaussianLikelihood(data, ToyMap(np.eye(n)), diag_cov(v0, n), RealGaussian(-3.5, "log10_A_n"),
                             signal=ToySignal(B), params=params, nuisance=U)
    assert isinstance(lik.covs[0], FlatMarginal) and lik.n_eff == (n - nu_,)
    x = np.array([0.3, -0.2, 1.7, 0.1])
    y = data @ np.eye(n).T - (x[:2] @ B) @ np.eye(n).T
    var = np.exp(x[3]) * v0
    Ciy = y / var
    quad, logdet, n_eff = float(y @ Ciy), float(np.sum(np.log(var))), n
    H = U.T @ (U / var[:, None])
    cf = cho_factor(0.5 * (H + H.T), lower=True, check_finite=False)
    b = U.T @ Ciy
    quad -= float(b @ cho_solve(cf, b, check_finite=False))
    logdet += 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    n_eff -= nu_
    A_n = 10.0 ** x[2]
    want = -0.5 * (quad / A_n + logdet + n_eff * np.log(A_n) + n_eff * LOG_2PI) + -3.5
    assert lik(x[None])[0] == want


def test_not_positive_definite_rows_are_minus_inf_and_counted():
    rng = np.random.default_rng(6)
    lik, k = gb_like(rng)
    X = rows(rng, 4)
    X[1, 4] = -50.0                                          # channel 0: C = e^a K0 - 50 I
    X[3, 6] = -50.0                                          # channel 1
    out = lik(X)
    assert np.isneginf(out[[1, 3]]).all() and np.isfinite(out[[0, 2]]).all()
    assert lik.last_failed.tolist() == [False, True, False, True]
    assert lik.counters() == {"noise_states": 4, "pd_failed_states": 2, "pd_failed_rows": 2,
                              "evaluations": 4, "cache_hits": 0}
    lik.reset_counters()
    assert lik.counters() == dict.fromkeys(list(COUNTERS) + ["cache_hits"], 0)


def failed_marginal_like(backend=None):
    """A ``FlatMarginal`` over a dense channel whose factorisation fails at ``bad``
    (``C = e^a K0 - 50 I``): the marginal never forms ``H`` and must not be applied."""
    rng = np.random.default_rng(15)
    n, nu_ = 20, 3
    params = Params(("s0", "s1", "log10_A_n", "a", "b"), (0, 1), (3, 4))
    lik = GaussianLikelihood(rng.standard_normal(n), ToyMap(np.eye(n)), dense_cov(spd(n, rng), n),
                             RealGaussian(-3.5, "log10_A_n"), signal=ToySignal(rng.standard_normal((2, n))),
                             params=params, nuisance=rng.standard_normal((n, nu_)), backend=backend)
    return lik, np.array([0.3, -0.2, 0.4, 0.1, -50.0])


def test_lnl_row_and_evaluate_map_a_failed_combinator_state_to_minus_inf():
    """A concretely failed state (outside ``jit``) is never applied: ``lnl_row`` gives
    ``(-inf, True)`` and ``evaluate`` ``-inf``, as ``__call__`` does -- the NumPy row loop
    of ``BatchedLikelihood`` must not raise the marginal's "check ok first"."""
    lik, bad = failed_marginal_like()
    lnl, failed = lik.lnl_row(lik.parts, bad)
    assert np.isneginf(lnl) and bool(failed) is True
    assert np.isneginf(lik.evaluate(bad, lik.noise_state(lik.noise_slice_host(bad))))
    out = lik(bad[None])
    assert np.isneginf(out).all() and lik.last_failed.tolist() == [True]
    good = bad.copy()
    good[4] = 0.2
    lnl, failed = lik.lnl_row(lik.parts, good)               # the guard leaves a PD row alone
    assert lnl == lik(good[None])[0] and np.isfinite(lnl) and not bool(failed)


def test_row_ok_rejects_without_flagging_a_failure():
    rng = np.random.default_rng(7)
    C = rng.standard_normal((P, NB)) + 1j * rng.standard_normal((P, NB))
    B = rng.standard_normal((NSRC, 2 * NB)) + 1j * rng.standard_normal((NSRC, 2 * NB))
    z = rng.standard_normal((2, NB)) + 1j * rng.standard_normal((2, NB))
    params = Params(("s0", "s1", "s2", "a0", "b0", "a1", "b1"), (0, 1, 2), (3, 4, 5, 6))
    cov = ChannelStack((dense_cov(spd(P, rng), P), dense_cov(spd(P, rng), P)), 2)
    lik = GaussianLikelihood(z, ToyReMap(C), cov, RealGaussian(), signal=ToySignal(B, (2, NB)), params=params,
                             quad_form="whiten", row_ok=lambda th: th[0] < 1.0)
    X = rows(rng, 3)
    X[:, 0] = [0.0, 2.0, 0.5]
    out = lik(X)
    assert np.isneginf(out[1]) and np.isfinite(out[[0, 2]]).all()
    assert not lik.last_failed.any()


def test_noise_clamp_clips_before_the_factorisation():
    rng = np.random.default_rng(8)
    n = 10
    params = Params(("s", "a"), (0,), (1,))
    lik = GaussianLikelihood(rng.standard_normal(n), ToyMap(np.eye(n)), diag_cov(np.ones(n), n),
                             RealGaussian(), signal=ToySignal(rng.standard_normal((1, n))),
                             params=params, noise_clamp={"a": 4.0})
    assert lik.noise_slice_host(np.array([0.0, 9.0])).tolist() == [4.0]
    assert lik.noise_slice_host(np.array([0.0, -9.0])).tolist() == [-4.0]
    assert lik(np.array([[0.1, 9.0]]))[0] == lik(np.array([[0.1, 4.0]]))[0]
    with pytest.raises(ValueError, match="positive"):
        GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), diag_cov(np.ones(n), n), RealGaussian(),
                           signal=ToySignal(np.ones((1, n))), params=params, noise_clamp={"a": 0.0})
    with pytest.raises(ValueError, match="noise parameters only; 's'"):
        GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), diag_cov(np.ones(n), n), RealGaussian(),
                           signal=ToySignal(np.ones((1, n))), params=params, noise_clamp={"s": 1.0})


def test_construction_errors():
    n = 4
    kw = dict(signal=ToySignal(np.ones((1, n))), params=Params(("s", "a"), (0,), (1,)))
    cov = diag_cov(np.ones(n), n)
    with pytest.raises(ValueError, match="strategy"):
        GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), cov, RealGaussian(), strategy="fast", **kw)
    with pytest.raises(ValueError, match="no amplitude form"):
        GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), cov, RealGaussian(0.0, "s"),
                           strategy="expanded", **kw)
    with pytest.raises(TypeError, match="density"):
        GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), cov, object(), **kw)
    with pytest.raises(ValueError, match="expected rows"):
        GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), cov, RealGaussian(), **kw)(np.zeros((2, 3)))


def test_noise_state_is_a_pytree_on_numpy_too():
    rng = np.random.default_rng(9)
    lik, _ = gb_like(rng)
    st = lik.noise_state(np.array([0.1, 0.2, -0.1, 0.3]))
    assert isinstance(st, NoiseState) and len(st.factors) == 2 and st.y is None and bool(st.ok)


# --------------------------------------------------------------------------- JAX


def test_jax_matches_numpy_and_keeps_constants_out_of_the_program():
    jax = pytest.importorskip("jax")
    rng = np.random.default_rng(10)
    lik_np, _ = gb_like(np.random.default_rng(10))
    lik_jx, _ = gb_like(np.random.default_rng(10), backend="jax")
    X = rows(rng, 5)
    a, b = lik_np(X), lik_jx(X)
    assert np.max(np.abs(a - b) / np.abs(a)) <= 1e-12
    st = lik_jx.noise_state(lik_jx.noise_slice_host(X[0]))
    assert isinstance(st.factors[0].L, jax.Array)
    hlo = jax.jit(lik_jx._evaluate).lower(lik_jx.parts, jax.numpy.asarray(X[0]), st).as_text()
    assert len(hlo) < 20_000, len(hlo)   # 5.8 KB measured; closing over parts and state gives 54 KB


def test_jax_expanded_complex_matches_numpy():
    pytest.importorskip("jax")
    out = []
    for backend in (None, "jax"):
        rng = np.random.default_rng(11)
        n = 16
        M0 = [hpd(n, rng), hpd(n, rng)]
        B = rng.standard_normal((NSRC, 2 * n)) + 1j * rng.standard_normal((NSRC, 2 * n))
        z = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
        params = Params(("s0", "s1", "s2", "m0", "m1"), (0, 1, 2), (3, 4))
        cov = ChannelStack((hermitian_cov(M0[0], n), hermitian_cov(M0[1], n)), 1)
        lik = GaussianLikelihood(z, ToySelect(0, n), cov, ComplexGaussian(), signal=ToySignal(B, (2, n)),
                                 params=params, strategy="expanded", backend=backend)
        out.append(lik(np.column_stack([rng.standard_normal((3, 3)), rng.uniform(-0.5, 0.5, (3, 2))])))
    assert np.max(np.abs(out[0] - out[1]) / np.abs(out[0])) <= 1e-12


def test_jax_keyed_covariance_runs_its_own_factor():
    """A covariance with a static key (adaptive rows) is not traced inside the likelihood's
    jit: ``noise_state`` calls ``cov.factor(nu_c, be)`` (eager here) and compiles only the
    rest; ``state_key`` forwards the key; ``BatchedLikelihood`` refuses it (checked once
    ``domainia.sampling`` exists: it lands in D-D4, whose Verify reruns this file)."""
    pytest.importorskip("jax")
    from dom_toys import AffineMatrix
    from domainia.covariance import DenseCovariance

    class Keyed(DenseCovariance):
        def state_key(self, nu):
            return int(np.asarray(nu)[0] > 0)

        def factor(self, nu, be=None, *, jit=None):
            self.factor_calls = getattr(self, "factor_calls", 0) + 1
            return super().factor(nu, be, jit=jit)

    rng = np.random.default_rng(14)
    n = 12
    K0, W, data, B = spd(n, rng), rng.standard_normal((n, n)), rng.standard_normal(n), rng.standard_normal((2, n))
    params = Params(("s0", "s1", "a", "b"), (0, 1), (2, 3))
    X = np.array([[0.3, -0.2, 0.1, 0.2], [-0.5, 0.7, -0.4, 0.05]])
    liks = [GaussianLikelihood(data, ToyMap(W), Keyed(AffineMatrix(K0), n, chol_impl="numpy"), RealGaussian(),
                               signal=ToySignal(B), params=params, quad_form="whiten", backend=b)
            for b in (None, "jax")]
    assert liks[1].jittable and not liks[1].traceable
    assert liks[1].state_key(X[0, 2:]) == (1,) and liks[1].state_key(X[1, 2:]) == (0,)
    a, b = liks[0](X), liks[1](X)
    assert np.max(np.abs(a - b) / np.abs(a)) <= 1e-12
    assert getattr(liks[1].covs[0], "factor_calls", 0) == len(X)    # its own factor, once per row
    if importlib.util.find_spec("domainia.sampling") is None:
        return                          # D-D4 (domainia.sampling) not landed yet
    from domainia.sampling import BatchedLikelihood

    with pytest.raises(TypeError, match="traceable"):
        BatchedLikelihood(liks[1])


def test_jax_failed_combinator_state_eager_and_jitted():
    """Eager JAX ``lnl_row`` takes the concrete guard; under ``jit`` the guard is inert
    (``ok`` is a tracer) and the failed state's NaN maps the row to ``(-inf, True)``."""
    jax = pytest.importorskip("jax")
    lik, bad = failed_marginal_like(backend="jax")
    th = jax.numpy.asarray(bad)
    for fn in (lik.lnl_row, jax.jit(lik.lnl_row)):
        lnl, failed = fn(lik.parts, th)
        assert np.isneginf(float(lnl)) and bool(failed) is True
    assert np.isneginf(float(lik.evaluate(bad, lik.noise_state(lik.noise_slice_host(bad)))))


def test_jax_refuses_a_map_that_is_not_a_pytree():
    pytest.importorskip("jax")

    class NotAPytree:
        def apply(self, x, be=None):
            return x

    n = 4
    with pytest.raises(TypeError, match="not an array"):
        GaussianLikelihood(np.zeros(n), NotAPytree(), diag_cov(np.ones(n), n), RealGaussian(),
                           signal=StaticSignal(lambda s: s), params=Params(("s", "a"), (0,), (1,)),
                           backend="jax")


SRC = Path(__file__).resolve().parents[1] / "src"
WDM_SRC = Path(__file__).resolve().parents[2] / "wdm" / "src"


def _probe(code: str, **env: str) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "JAX_ENABLE_X64")}
    e.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(SRC), str(WDM_SRC),
                                                                        str(Path(__file__).parent)]))
    e.update(env)
    return subprocess.run([sys.executable, "-P", "-c", code], env=e, capture_output=True, text=True)


def test_importing_the_likelihood_modules_does_not_import_jax():
    out = _probe("import sys, domainia.density, domainia.likelihood; print('jax' in sys.modules)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_jax_likelihood_requires_x64():
    pytest.importorskip("jax")
    code = ("import numpy as np\n"
            "from dom_toys import ToyMap, ToySignal, diag_cov\n"
            "from domainia.density import RealGaussian\n"
            "from domainia.likelihood import GaussianLikelihood, Params\n"
            "try:\n"
            "    GaussianLikelihood(np.zeros(4), ToyMap(np.eye(4)), diag_cov(np.ones(4), 4), RealGaussian(),\n"
            "                       signal=ToySignal(np.ones((1, 4))), params=Params(('s', 'a'), (0,), (1,)),\n"
            "                       backend='jax')\n"
            "except RuntimeError as err:\n"
            "    print('refused:', 'float64' in str(err))\n")
    out = _probe(code, JAX_ENABLE_X64="false")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "refused: True"


def test_quad_logdet_reassembles_the_lnl():
    rng = np.random.default_rng(12)
    for strategy in ("residual", "expanded"):
        lik, k = gb_like(np.random.default_rng(12), strategy=strategy)
        x = rows(rng, 1)[0]
        ql = lik.quad_logdet(x)
        assert ql.shape == (2, 2)
        total = 0.0
        for c in range(2):
            total = total + RealGaussian().lnl(ql[c, 0], ql[c, 1], P)
        ref = lik(x[None])[0]
        if strategy == "residual":
            assert total == ref
        else:
            assert abs(total - ref) <= 1e-12 * abs(ref)
    bad = rows(rng, 1)[0]
    bad[4] = -50.0
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        lik.quad_logdet(bad)


def test_dense_oracle_of_wt_test_matrix():
    """Ported from wdm_paper_2/toy_models/tests/test_matrix.py (``oracle``, lines 35-48, and
    ``test_every_wdm_model_matches_its_dense_oracle``, lines 194-204): the likelihood half --
    a dense covariance, A outside the factorisation, a flat marginal over U and the
    log-Jacobian -- equals the dense formula ln N(y; 0, A C) marginalised over U (the
    model half, each model's dense covariance, is D-C's and S3's)."""
    rng = np.random.default_rng(13)
    n, n_u = 30, 3
    K0, U = spd(n, rng), rng.standard_normal((n, n_u))
    W, data, B = rng.standard_normal((n, n)), rng.standard_normal(n), rng.standard_normal((2, n))
    params = Params(("s0", "s1", "log10_A_n", "a", "b"), (0, 1), (3, 4))
    lik = GaussianLikelihood(data, ToyMap(W), dense_cov(K0, n), RealGaussian(-2.5, "log10_A_n"),
                             signal=ToySignal(B), params=params, nuisance=U)
    for x in (np.array([0.3, -0.2, 0.4, 0.1, 0.2]), np.array([-0.5, 0.7, 1.3, -0.4, 0.05])):
        y = data @ W.T - (x[:2] @ B) @ W.T
        C = 10.0 ** x[2] * (np.exp(x[3]) * K0 + x[4] * np.eye(n))
        Ci = np.linalg.inv(C)                                   # test_matrix.py::oracle, verbatim
        quad = y @ Ci @ y
        logdet = np.linalg.slogdet(C)[1]
        m = len(y)
        H = U.T @ Ci @ U
        b = U.T @ Ci @ y
        quad -= b @ np.linalg.solve(H, b)
        logdet += np.linalg.slogdet(H)[1]
        m -= U.shape[1]
        want = -0.5 * (quad + logdet + m * np.log(2 * np.pi)) + -2.5
        assert lik(x[None])[0] == pytest.approx(want, rel=1e-10, abs=1e-7)
