"""D-D1: the densities reproduce their oracles' last lines operand for operand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from domainia.density import LOG_2PI, LOG_PI, ComplexGaussian, RealGaussian


def test_log_constants_are_the_oracles():
    assert LOG_2PI == float(np.log(2.0 * np.pi))              # WT _LOG_2PI, GB _LOG_2PI
    assert LOG_PI == float(np.log(np.pi))


@pytest.mark.parametrize("seed", range(5))
def test_amplitude_form_is_wt_marginal_lnlike(seed):
    rng = np.random.default_rng(seed)
    quad, logdet = float(rng.uniform(1e5, 1e7)), float(rng.uniform(-2e5, -1e5))
    n_eff, log10A = int(rng.integers(1000, 20000)), float(rng.uniform(1, 4))
    log_jac = 0.5 * 16384 * float(np.log(2.0 ** -10))
    A_n = 10.0 ** np.float64(log10A)
    # wdm_toys/matrix.py:669, verbatim
    want = -0.5 * (quad / A_n + logdet + n_eff * np.log(A_n) + n_eff * LOG_2PI) + log_jac
    got = RealGaussian(log_jac, "log10_A_n").lnl(quad, logdet, n_eff, np.float64(log10A))
    assert got == want


@pytest.mark.parametrize("seed", range(5))
def test_plain_form_is_the_gb_matrix_line(seed):
    rng = np.random.default_rng(seed)
    q, logdet, P = float(rng.uniform(100, 3000)), float(rng.uniform(-900, 900)), 720
    total = 0.0
    total = total - 0.5 * (q + logdet + P * LOG_2PI)           # galactic_binary/matrix.py:922
    assert 0.0 + RealGaussian().lnl(q, logdet, P) == total


def test_expanded_real_form_is_matrix_cache_signal():
    rng = np.random.default_rng(7)
    q0, hy, tt = (float(x) for x in rng.uniform(1, 500, 3))
    logdet, P = float(rng.uniform(-900, 900)), 693
    logdet_norm = logdet + P * LOG_2PI                        # matrix.py:983 (state)
    total = 0.0
    total = total - 0.5 * (q0 - 2.0 * hy + tt + logdet_norm)  # matrix.py:991 (signal)
    dens = RealGaussian()
    assert dens.log_norm(P) == P * LOG_2PI
    assert 0.0 + dens.lnl_expanded(q0, hy, tt, logdet + dens.log_norm(P)) == total


def test_complex_forms_are_lnl_complex_and_lnl_from_state():
    rng = np.random.default_rng(3)
    quad, logdet = float(rng.uniform(100, 900)), float(rng.uniform(-90, 90))
    assert ComplexGaussian().lnl(quad, logdet) == -quad - logdet          # likelihoods.py:146-147
    q0, hy, tt = (float(x) for x in rng.uniform(1, 500, 3))
    want = -(q0 - 2.0 * hy + tt) - logdet                                 # fullcache.py:144
    assert ComplexGaussian().lnl_expanded(q0, hy, tt, logdet) == want


def test_offset_to_canonical_is_documented_not_applied():
    g = ComplexGaussian()
    assert g.lnl(10.0, 2.0, 7) == -12.0                  # no n ln pi on the golden path
    assert g.offset_to_canonical(7) == -7 * LOG_PI
    with_pi = ComplexGaussian(include_pi=True)
    assert with_pi.lnl(10.0, 2.0, 7) == -12.0 - 7 * LOG_PI
    assert with_pi.offset_to_canonical(7) == 0.0
    with pytest.raises(ValueError, match="needs n"):
        with_pi.lnl(10.0, 2.0)


def test_misuse_is_refused():
    with pytest.raises(ValueError, match="no amplitude"):
        RealGaussian().lnl(1.0, 2.0, 3, 2.0)
    with pytest.raises(ValueError, match="needs the 'log10_A_n' value"):
        RealGaussian(0.0, "log10_A_n").lnl(1.0, 2.0, 3)
    with pytest.raises(ValueError, match="no amplitude oracle"):
        RealGaussian(0.0, "log10_A_n").lnl_expanded(1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="finite"):
        RealGaussian(math.inf)
    with pytest.raises(ValueError, match="parameter name"):
        RealGaussian(0.0, "")


def test_jax_scalars_give_the_same_numbers():
    jnp = pytest.importorskip("jax.numpy")
    from wdm.backend import jax_backend

    be = jax_backend()
    dens = RealGaussian(-56782.61703147072, "log10_A_n")
    args = (2.1404038378040653e7, -183804.00714711868, 16256, 2.6020599913279625)
    want = dens.lnl(*args)
    got = dens.lnl(*(jnp.asarray(a) for a in args[:2]), 16256, jnp.asarray(args[3]), be=be)
    assert abs(float(got) - want) <= 1e-12 * abs(want)


def test_folded_form_is_wt_fast_full_covariance():
    rng = np.random.default_rng(11)
    quad_unit, logdet_unit = float(rng.uniform(1e6, 1e7)), float(rng.uniform(-7e4, -6e4))
    n, log10A = 16256, np.float64(2.6020599913279625)
    A_n = 10.0 ** log10A
    quad = quad_unit / A_n                                   # wdm_toys/likelihood.py:580-584
    logdet = logdet_unit + n * float(np.log(A_n))
    want = -0.5 * (quad + logdet + n * LOG_2PI)
    assert RealGaussian(0.0, "log10_A_n").lnl_folded(quad_unit, logdet_unit, n, log10A) == want
    with pytest.raises(ValueError, match="needs an amplitude"):
        RealGaussian().lnl_folded(1.0, 2.0, 3, 0.0)
    with pytest.raises(ValueError, match="needs the 'log10_A_n' value"):
        RealGaussian(0.0, "log10_A_n").lnl_folded(1.0, 2.0, 3, None)
