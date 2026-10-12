"""D-C11: ProjectedOperatorMarginal (sinusoid ``exact``) and ChannelStack (GB channels).

* The projected quadratic form is the dense retained-coordinate oracle and ignores the
  excluded components (WT test_bandcov.py); logdet = Weyl + c_N alpha_d^2 + correction.
* G6 at the manuscript truth: quad (PCG, tol 1e-7) and the unit log-determinant with the
  shipped c_N and the shipped bandcov interpolant (evaluated test-locally: the model
  layer's BandCorrection owns it), and lnL at the approved exact gate.
* PCG failure: raises (on_fail="raise"), NaN (on_fail="nan").
* ChannelStack: channels split ``nu``, logdet sums in channel order; GB tri_raw per
  channel through the stack; JAX parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb

import domtest_inputs as T
from domainia.covariance import ChannelStack, Eq45Covariance, ExactTimeCovariance, StackFactor
from domainia.nuisance import FourierBandNuisance, ProjectedFactor, ProjectedOperatorMarginal
from domainia.spectra import LogLinearPSD

LOG_2PI = float(np.log(2.0 * np.pi))
N, DT = 128, 1 / 16                                         # WT test_bandcov: Nf 8 x Nt 16
PSD = LogLinearPSD(f_floor=2.0, f_ref=1.0, T=N * DT)


def _retained(nb: FourierBandNuisance) -> np.ndarray:
    """Q: the retained orthonormal real coordinates (WT test_bandcov ``coordinates``)."""
    t = np.arange(N)
    k = np.flatnonzero(nb.keep_rfft)
    return np.concatenate([np.sqrt(2 / N) * np.cos(2 * np.pi * k[:, None] * t / N),
                           np.sqrt(2 / N) * np.sin(2 * np.pi * k[:, None] * t / N)])


@pytest.mark.parametrize("nu", [(-2.0, 0.0), (-1.7, 1.2), (-3.4, -2.1)])
def test_projected_quadratic_equals_the_retained_oracle(nu):
    nu = np.asarray(nu)
    nb = FourierBandNuisance(N, DT, 3.0)
    op = ExactTimeCovariance(PSD, N, DT, c_N=0.0, tol=1e-11)
    Q = _retained(nb)
    C = Q @ op.sigma_tilde(nu) @ Q.T
    exact_logdet = np.linalg.slogdet(C)[1]
    correction = exact_logdet - op.logdet_model(nu)
    pm = ProjectedOperatorMarginal(op, nb, lambda nu_, be: correction)
    F = pm.factor(nu)
    assert isinstance(F, ProjectedFactor) and F.n_u == nb.n_drop and N - F.n_u == nb.n_keep == Q.shape[0]
    assert pm.n_u == F.n_u and pm.state_key(nu) is None      # static, what a likelihood reads
    r = np.random.default_rng(8).normal(size=N)
    y = Q @ r
    want = y @ np.linalg.solve(C, y)
    assert float(F.quad(r)) == pytest.approx(want, rel=2e-10)
    assert float(F.logdet) == pytest.approx(exact_logdet, rel=1e-14)
    excluded = r - Q.T @ y
    assert float(F.quad(r + 300 * excluded)) == pytest.approx(want, rel=2e-10)


def test_pcg_failure_raises_or_is_nan():
    nb = FourierBandNuisance(N, DT, 3.0)
    r = np.random.default_rng(8).normal(size=N)
    raising = ProjectedOperatorMarginal(ExactTimeCovariance(PSD, N, DT, c_N=0.0, maxit=1), nb, lambda nu, be: 0.0)
    with pytest.raises(RuntimeError):
        raising.factor(np.array([-1.7, 1.2])).quad(r)
    with pytest.raises(ValueError, match="operator"):
        ProjectedOperatorMarginal(ExactTimeCovariance(PSD, N, DT, form="dense"), nb, lambda nu, be: 0.0)
    ok = ProjectedOperatorMarginal(ExactTimeCovariance(PSD, N, DT, c_N=0.0), nb, lambda nu, be: 0.0)
    with pytest.raises(ValueError, match="length 128"):     # WT band.project: never resized by irfft(n=N)
        ok.factor(np.array([-1.7, 1.2])).quad(np.append(r, 0.3))


def _bandcov_correction(nu, be):
    """The shipped bandcov interpolant (WT BandCovariance.logdet, chebval2d on the asinh map)."""
    with np.load(T.DATA / "sinusoid" / "cache" / "bandcov_v1_4bfb466c191c79f9879a.npz", allow_pickle=False) as z:
        coeff = np.array(z["coefficients"], float)
    bounds = np.asarray([[-4.0, 0.0], [-2.0, 2.0]])
    drift_scale = min(0.03, 0.384 * T.SIN_F_MIN / (0.5 / T.SIN_DT))
    ac, ad = float(nu[0]), float(nu[1])
    x = 2 * (ac - bounds[0, 0]) / (bounds[0, 1] - bounds[0, 0]) - 1
    lo, hi = np.arcsinh(bounds[1] / drift_scale)
    y = 2 * (np.arcsinh(ad / drift_scale) - lo) / (hi - lo) - 1
    return cheb.chebval2d(x, y, coeff)


def test_g6_exact_model_at_the_truth():
    s = T.sinusoid_mid()
    ref = s["g6"]["parts"]["exact"]
    c_N = float(np.load(T.DATA / "sinusoid" / "cache" / "cN_N16384_dt0.0009765625_fmin4_fref1.npy"))
    op = ExactTimeCovariance(LogLinearPSD(f_floor=4.0, f_ref=1.0, T=T.SIN_T), T.SIN_N, T.SIN_DT, P=24, c_N=c_N,
                             tol=1e-7)
    nb = FourierBandNuisance(T.SIN_N, T.SIN_DT, T.SIN_FMIN_CUT)
    assert (nb.n_keep, nb.n_drop) == (ref["n_keep"], 128)
    F = ProjectedOperatorMarginal(op, nb, _bandcov_correction).factor(T.SIN_NU)
    q = float(F.quad(s["d"] - s["h"]))
    assert q == pytest.approx(T.num(ref["quad_unit"]), rel=1e-12, abs=0)
    assert float(F.logdet) == pytest.approx(T.num(ref["logdet_unit"]), rel=1e-12, abs=0)
    A_n, n = T.num(ref["A_n"]), nb.n_keep
    lnl = -0.5 * (q / A_n + float(F.logdet) + n * np.log(A_n) + n * LOG_2PI)
    assert abs(lnl - T.num(ref["lnL"])) <= 1e-7                    # the approved exact gate


def test_channel_stack_splits_and_sums(backend):
    from domainia.covariance import DiagonalCovariance

    covs = [DiagonalCovariance(lambda nu, be, s=s: be.xp.exp(nu[0]) * be.asarray(np.full(3, s)), 3) for s in (1.0, 2.0)]
    st = ChannelStack(covs, 2)
    nu = np.array([0.1, 9.0, 0.2, 9.0])
    F = st.factor(backend.asarray(nu), backend, jit=False)
    assert isinstance(F, StackFactor) and len(F.channels) == 2 and st.n == 6 and F.n_u == 0
    assert st.n_u == 0 and st.state_key(nu) is None
    want = 3 * 0.1 + 3 * (np.log(2.0) + 0.2)
    assert float(F.logdet) == pytest.approx(want, rel=1e-14)
    b = np.ones((2, 3))
    assert float(F.quad(backend.asarray(b))) == pytest.approx(3 / np.exp(0.1) + 3 / (2 * np.exp(0.2)), rel=1e-14)
    with pytest.raises(ValueError, match="entries"):
        st.factor(np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        ChannelStack(covs, -1)


def test_g6_gb_tri_raw_through_the_stack():
    c = T.gb_case("gb_3mo")
    st = ChannelStack([Eq45Covariance(c.psd, c.cells, builder="separable", assembly="dense", channel=ch)
                       for ch in range(2)], c.mu.shape[1])
    F = st.factor(c.mu.ravel())
    r = c.residual_cells(c.cells)
    got = np.array([[float(f.quad(r[ch], form="whiten")), float(f.logdet)] for ch, f in enumerate(F.channels)])
    assert np.allclose(got, c.quad_logdet("tri_raw"), rtol=1e-12, atol=0)


@pytest.mark.jax
def test_jax_projected_and_stack(jax_be):
    import jax.numpy as jnp

    nb = FourierBandNuisance(N, DT, 3.0)
    pm = ProjectedOperatorMarginal(ExactTimeCovariance(PSD, N, DT, c_N=0.1, tol=1e-11, on_fail="nan"), nb,
                                   lambda nu, be: 0.0 * nu[0])
    assert pm.jittable
    r = np.random.default_rng(8).normal(size=N)
    nu = np.array([-1.7, 1.2])
    Fn, Fj = pm.factor(nu), pm.factor(jnp.asarray(nu))
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-14)
    assert float(Fj.quad(jnp.asarray(r))) == pytest.approx(float(Fn.quad(r)), rel=1e-9)
    bad = ProjectedOperatorMarginal(ExactTimeCovariance(PSD, N, DT, c_N=0.1, maxit=1, on_fail="nan"), nb,
                                    lambda nu, be: 0.0 * nu[0])
    assert not np.isfinite(float(bad.factor(jnp.asarray(nu)).quad(jnp.asarray(r))))
    c = T.gb_case("gb_3mo")
    st = ChannelStack([Eq45Covariance(c.psd, c.cells, builder="separable", assembly="dense", channel=ch)
                       for ch in range(2)], c.mu.shape[1])
    Gn, Gj = st.factor(c.mu.ravel()), st.factor(jnp.asarray(c.mu.ravel()))
    assert float(Gj.logdet) == pytest.approx(float(Gn.logdet), rel=1e-12)
