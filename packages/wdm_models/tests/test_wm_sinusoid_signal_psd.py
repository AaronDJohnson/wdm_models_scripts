"""E-A4: the sinusoid signal (WT/WG operation order) and the noise PSD (WT ``noise``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from domainia.spectra import LogLinearPSD
from wdm_models.sinusoid import config as C
from wdm_models.sinusoid import psd as P
from wdm_models.sinusoid.signal import Sinusoid, sinusoid

CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
SIN_DATA = Path(__file__).resolve().parents[3] / "data" / "sinusoid"     # wdm_models_scripts/data


@pytest.mark.golden
def test_the_signal_is_the_shipped_h_bit_for_bit(golden_io, facts):
    """``A_h cos(2 pi f_0 t + phi_0)`` in WT ``likelihood.signal``'s order reproduces the
    stored ``h`` of every case (and its sha256 in the oracle facts)."""
    for case in CASES:
        cfg = C.config(case)
        with np.load(SIN_DATA / case / "toy_data.npz", allow_pickle=False) as z:
            h = np.array(z["h"])
        got = Sinusoid.for_config(cfg)(cfg.truth("band_raw")[:3])
        assert got.dtype == np.float64 and np.array_equal(got, h), case
        assert golden_io.sha256_array(got) == facts["cases"][case]["signal_sha256"], case
        assert np.array_equal(sinusoid(cfg.times(), cfg.A_h, cfg.f_0, cfg.phi_0), h), case


def test_the_signal_refuses_a_chirp_and_is_a_pytree_on_jax(jax_be):
    s = Sinusoid.for_config(C.config("manuscript_mid"))
    with pytest.raises(ValueError, match="fixed-frequency"):
        s(np.array([0.3, 137.0, 0.6, 0.5]))
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(s)
    assert len(leaves) == 1 and leaves[0].shape == (16384,)
    src = np.array([0.30421576834679953, 137.0, 0.6])
    on_jax = jax.jit(lambda sig, x: sig(x))(jax.device_put(s), jax_be.asarray(src))
    assert np.max(np.abs(np.asarray(on_jax) - s(src))) <= 1e-12


@pytest.mark.golden
def test_the_psd_is_the_oracle_psd(facts):
    p = facts["psd"]
    f, t = np.array(p["f"]), np.array(p["t"])
    assert np.array_equal(P.psd(f, t), np.array(p["values"]))
    assert np.array_equal(P.psd(f), np.array(p["midpoint"]))
    cfg = C.SinusoidConfig()
    assert np.array_equal(P.psd(f, t, cfg=cfg, A_n=p["A_n"], alpha_c=p["alpha_c"], alpha_d=p["alpha_d"]),
                          np.array(p["values"]))


def test_the_domainia_form_is_the_same_psd():
    cfg = C.config("manuscript_mid")
    ll = P.as_loglinear(cfg)
    assert ll == LogLinearPSD(f_floor=4.0, f_ref=1.0, T=16.0)
    f = np.fft.rfftfreq(cfg.N, d=cfg.dt)
    assert np.array_equal(ll.L(f), P.log_f_clip(f, cfg))
    t = cfg.times()[::257]
    nu = (cfg.alpha_c, cfg.alpha_d)
    assert np.array_equal(cfg.A_n * ll(f[None, :], t[:, None], nu), P.psd(f, t, cfg=cfg))
    assert np.array_equal(ll.alpha(t, nu), P.alpha(t, *nu, cfg))


def test_alpha_runs_over_the_record_length_and_the_midpoint_index():
    """``alpha(t) = alpha_c + alpha_d (t/T - 1/2)`` with ``T = N dt`` (not ``(N - 1) dt``); the
    exact covariance's pair midpoint ``s/(2N) - 1/2`` (``s = i + j``, ``sigma_tilde``) is
    ``alpha((t_i + t_j)/2)`` bit for bit on the manuscript grid."""
    cfg = C.config("manuscript_mid")
    assert P.alpha(cfg.T_obs / 2, -2.0, 1.2, cfg) == -2.0
    assert P.alpha(cfg.T_obs, -2.0, 1.2, cfg) == -2.0 + 0.6
    i = np.array([0, 1, 7, 8191, 16383])
    j = np.array([0, 16383, 4096, 8192, 16383])
    t = cfg.times()
    s = i + j
    assert np.array_equal(-2.0 + 1.2 * (s / (2.0 * cfg.N) - 0.5), P.alpha((t[i] + t[j]) / 2.0, -2.0, 1.2, cfg))


def test_the_floor_clips_the_log_frequency():
    cfg = C.config("manuscript_mid")
    assert np.array_equal(P.log_f_clip([0.0, 2.0, 4.0, 8.0], cfg), np.log(np.array([4.0, 4.0, 4.0, 8.0])))
