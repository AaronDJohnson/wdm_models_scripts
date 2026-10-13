"""D-D3: FourierWhittle in both oracle forms (WT FreqWhittle/MatrixWhittle, GB
lnl_frequency), NumPy bitwise against transcriptions, JAX <= 1e-12."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dom_toys import ToySelect, ToySignal
from domainia.likelihood import FourierWhittle, Params, StaticSignal

N, DT, FMIN = 1024, 2.0 ** -10, 4.0
LOG_2PI = float(np.log(2.0 * np.pi))


class LogF:
    """``L(f) = log(max(f, f_floor) / f_ref)`` (WT ``noise.log_f_clip``); the gate tests use
    ``domainia.spectra.LogLinearPSD`` instead."""

    def L(self, f):
        return np.log(np.maximum(np.asarray(f, dtype=float), 4.0) / 1.0)


def sinusoid(src, t=np.arange(N) * DT):
    return src[0] * np.cos(2.0 * np.pi * src[1] * t + src[2])


def series(seed):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(values=rng.standard_normal(N) * 20.0, domain=SimpleNamespace(N=N, dt=DT))


PARAMS = Params(("A_h", "f_0", "phi_0", "log10_A_n", "alpha"), (0, 1, 2), (3, 4))


def wt_freq_whittle(d, theta):
    """wdm_toys/likelihood.py:240-250 (analysis_fmin > 0), transcribed."""
    r = d - sinusoid(theta[:3])
    f = np.fft.rfftfreq(N, d=DT)
    keep = (f >= FMIN) & (f < 0.5 / DT)
    keep[[0, -1]] = False
    L = LogF().L(f)
    A_n = 10.0 ** theta[3]
    R = np.fft.rfft(r)[keep]
    lam = A_n * np.exp(theta[4] * L[keep]) / (2.0 * DT)
    quad = 2.0 / N * float(np.sum(np.abs(R) ** 2 / lam))
    logdet = 2.0 * float(np.sum(np.log(lam)))
    return -0.5 * (quad + logdet + 2 * int(keep.sum()) * LOG_2PI)


def wt_matrix_whittle_cut(d, theta, lo, hi):
    """wdm_toys/matrix.py:862-898 (whittle_cut, U = 1), transcribed."""
    n = hi - lo
    f = np.fft.rfftfreq(n, d=DT)
    keep = (f >= max(FMIN, 0.0)) & (f < 0.5 / DT)
    keep[[0, -1]] = False
    Lc = LogF().L(f[keep])
    x = (d - sinusoid(theta[:3]))[slice(lo, hi)]
    R = np.fft.rfft(x)[keep]
    lam = 1.0 * 10.0 ** theta[3] * np.exp(theta[4] * Lc) / (2.0 * DT)
    quad = 2.0 / n * float(np.sum(np.abs(R) ** 2 / lam))
    logdet = 2.0 * float(np.sum(np.log(lam)))
    return -0.5 * (quad + logdet + 2 * int(keep.sum()) * LOG_2PI)


THETA = np.array([[0.5, 137.0, 0.6, 2.6020599913279625, -2.0],
                  [0.4, 101.3, -1.1, 2.1, -1.2],
                  [1.1, 300.0, 2.9, 3.4, -3.5]])


def test_real_orthonormal_is_wt_freq_whittle_bitwise():
    ts = series(0)
    lik = FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, signal=StaticSignal(sinusoid),
                         params=PARAMS)
    assert lik.n_keep == 2 * 508                      # bins 4..511 of the 1 Hz grid (no DC, no Nyquist)
    assert np.array_equal(lik(THETA), [wt_freq_whittle(ts.values, th) for th in THETA])


def test_real_orthonormal_crop_is_wt_matrix_whittle_bitwise():
    ts = series(1)
    lo, hi = 6 * 4, (256 - 5) * 4                     # [CUT_START*Nf, (Nt-CUT_END)*Nf) at Nf = 4
    lik = FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, crop=(lo, hi),
                         signal=StaticSignal(sinusoid), params=PARAMS)
    assert lik.Nc == hi - lo
    assert np.array_equal(lik(THETA), [wt_matrix_whittle_cut(ts.values, th, lo, hi) for th in THETA])


def gb_whittle(n=40, nstore=60, k0=7, seed=2):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((2, nstore)) + 1j * rng.standard_normal((2, nstore))
    B = rng.standard_normal((3, 2 * nstore)) + 1j * rng.standard_normal((3, 2 * nstore))
    sI, sG = rng.uniform(0.5, 1.5, (2, n)), rng.uniform(0.1, 0.9, (2, n))
    return z, B, sI, sG, k0, n


def test_complex_form_is_gb_lnl_frequency_bitwise():
    z, B, sI, sG, k0, n = gb_whittle()
    params = Params(("s0", "s1", "s2", "mu_A", "mu_E"), (0, 1, 2), (3, 4))
    lik = FourierWhittle("complex", SimpleNamespace(values=z), map=ToySelect(k0, k0 + n), sI=sI, sG=sG,
                         signal=ToySignal(B, (2, z.shape[1])), params=params)
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.standard_normal((4, 3)), rng.uniform(-1, 1, (4, 2))])
    for x in X:
        r = z - (x[:3] @ B).reshape(2, -1)
        total = 0.0
        for c in range(2):                            # galactic_binary/likelihoods.py:160-162, :331
            m = sI[c] + np.exp(x[3 + c]) * sG[c]
            total = total + (-np.sum(np.abs(r[c, k0:k0 + n]) ** 2 / m) - np.sum(np.log(m)))
        assert lik(x[None])[0] == total
    assert lik.n_keep == n


def test_row_ok_and_non_finite_rows_give_minus_inf():
    ts = series(4)
    lik = FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, signal=StaticSignal(sinusoid),
                         params=PARAMS, row_ok=lambda th: th[0] < 1.0)
    out = lik(THETA)
    assert np.isfinite(out[:2]).all() and np.isneginf(out[2])
    assert not lik.last_failed.any() and lik.counters()["evaluations"] == 3
    assert np.isneginf(lik(np.array([[0.5, 137.0, 0.6, 400.0, -2.0]]))[0])   # 10**400 overflows


def test_form_arguments_are_checked():
    ts = series(5)
    with pytest.raises(ValueError, match="form must be"):
        FourierWhittle("stationary", ts, signal=StaticSignal(sinusoid), params=PARAMS)
    with pytest.raises(ValueError, match="takes psd and fmin"):
        FourierWhittle("real-orthonormal", ts, psd=LogF(), signal=StaticSignal(sinusoid), params=PARAMS)
    with pytest.raises(ValueError, match="positive analysis cut"):
        FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=0.0, signal=StaticSignal(sinusoid),
                       params=PARAMS)
    with pytest.raises(ValueError, match="inside the 1024-sample record"):
        FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, crop=(10, 2000),
                       signal=StaticSignal(sinusoid), params=PARAMS)
    with pytest.raises(ValueError, match="takes map, sI and sG"):
        FourierWhittle("complex", ts, psd=LogF(), signal=StaticSignal(sinusoid), params=PARAMS)
    # the real form reads one record off a time domain: a bare array or a channel axis is refused
    with pytest.raises(TypeError, match="needs a TimeSeries"):
        FourierWhittle("real-orthonormal", ts.values, psd=LogF(), fmin=FMIN,
                       signal=StaticSignal(sinusoid), params=PARAMS)
    two = SimpleNamespace(values=np.stack([ts.values, ts.values]), domain=ts.domain)
    with pytest.raises(ValueError, match=r"one record of shape \(1024,\)"):
        FourierWhittle("real-orthonormal", two, psd=LogF(), fmin=FMIN, signal=StaticSignal(sinusoid),
                       params=PARAMS)


def test_jax_names_a_signal_that_is_not_a_pytree():
    pytest.importorskip("jax")
    with pytest.raises(TypeError, match="leaf of type function is not an array"):
        FourierWhittle("real-orthonormal", series(8), psd=LogF(), fmin=FMIN, signal=sinusoid,
                       params=PARAMS, backend="jax")


def test_jax_matches_numpy():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    ts = series(6)

    def sinusoid_jax(src, t=jnp.asarray(np.arange(N) * DT)):
        return src[0] * jnp.cos(2.0 * jnp.pi * src[1] * t + src[2])

    a = FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, signal=StaticSignal(sinusoid),
                       params=PARAMS)(THETA)
    b = FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, signal=StaticSignal(sinusoid_jax),
                       params=PARAMS, backend="jax")(THETA)
    assert np.max(np.abs(a - b) / np.abs(a)) <= 1e-12
    z, B, sI, sG, k0, n = gb_whittle()
    params = Params(("s0", "s1", "s2", "mu_A", "mu_E"), (0, 1, 2), (3, 4))
    out = []
    for backend in (None, "jax"):
        lik = FourierWhittle("complex", SimpleNamespace(values=z), map=ToySelect(k0, k0 + n), sI=sI,
                             sG=sG, signal=ToySignal(B, (2, z.shape[1])), params=params, backend=backend)
        out.append(lik(np.array([[0.1, -0.2, 0.3, 0.2, -0.4]])))
    assert abs(out[0][0] - out[1][0]) <= 1e-12 * abs(out[0][0])


def test_quad_logdet_reassembles_the_lnl():
    ts = series(7)
    lik = FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=FMIN, signal=StaticSignal(sinusoid),
                         params=PARAMS)
    (quad, logdet), = lik.quad_logdet(THETA[0])
    assert -0.5 * (quad + logdet + lik.n_keep * LOG_2PI) + 0.0 == lik(THETA[:1])[0]
