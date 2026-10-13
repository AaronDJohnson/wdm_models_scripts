"""E-A3: the sinusoid configuration and cases (spec §5.1 config.py) and the model names."""

from __future__ import annotations

import math

import numpy as np
import pytest

import wdm
from wdm.evolutionary import wrap_cut_columns, wrap_fractions
from wdm_models.common import params
from wdm_models.sinusoid import config as C

CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
#: WT ToyConfig fields that SinusoidConfig carries (the rest -- fdot, frequency_study -- are
#: the dropped chirp / prior-table switch, pinned separately).
FIELDS = ("Nf", "Nt", "dt", "A_h", "f_0", "phi_0", "A_n", "alpha_c", "alpha_d", "f_ref", "f_min",
          "N", "T_obs", "fs", "f_nyq", "DF", "DT", "fit_fdot")


def test_the_fifteen_models_in_spec_order():
    assert params.MODELS == ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad",
                             "band_exact", "mid_raw", "mid_cut", "mid_pad", "mid_exact", "tri_raw",
                             "tri_cut", "tri_pad", "tri_exact", "exact")
    assert [params.split_model(m) for m in ("whittle_cut", "tri_pad", "exact")] == [
        ("whittle", "cut"), ("tri", "pad"), ("exact", None)]
    assert [m for m in params.MODELS if params.is_whittle(m)] == ["whittle_raw", "whittle_cut"]
    with pytest.raises(ValueError, match="unknown model 'band_win'"):
        params.check_model("band_win")


@pytest.mark.golden
def test_cases_are_the_oracle_configs(facts):
    assert C.CASE_NAMES == CASES and tuple(C.CASES) == CASES
    for case in CASES:
        cfg, wt = C.CASES[case], facts["cases"][case]["config"]
        for name in FIELDS:
            assert getattr(cfg, name) == wt[name], (case, name)
        assert (wt["fdot"], wt["frequency_study"]) == (0.0, True)


def test_the_amplitudes_are_the_snr_15_rescale():
    """WT config.CONFIGS: A_h = 0.5 * 15 / SNR(A_h = 0.5) per case, as literal floats."""
    want = {"manuscript_low": (33.0, 1.6322595829099906, 4.59485738575295),
            "manuscript_mid": (137.0, 0.30421576834679953, 24.65355441881684),
            "manuscript_high": (385.0, 0.08729868783526586, 85.91194422249085)}
    for case, (f0, A_h, snr) in want.items():
        cfg = C.config(case)
        assert (cfg.f_0, cfg.A_h) == (f0, A_h)
        assert A_h == 0.5 * 15.0 / snr


def test_the_shared_noise_and_grid():
    for cfg in C.CASES.values():
        assert (cfg.Nf, cfg.Nt, cfg.dt, cfg.N) == (64, 256, 2.0 ** -10, 16384)
        assert (cfg.A_n, cfg.alpha_c, cfg.alpha_d, cfg.f_ref, cfg.f_min, cfg.phi_0) == (
            400.0, -2.0, 1.2, 1.0, 4.0, 0.6)
        assert (cfg.T_obs, cfg.f_nyq, cfg.DF, cfg.DT) == (16.0, 512.0, 8.0, 0.0625)
        assert cfg.grid() == wdm.Grid(64, 256, 2.0 ** -10)
        assert np.array_equal(cfg.times(), np.arange(16384) * 2.0 ** -10)
        assert cfg.log10_A_n == float(np.log10(400.0))


@pytest.mark.golden
def test_the_edge_constants(g6):
    assert C.ANALYSIS_FMIN == 4.0 and C.PAD_COLUMNS == 16
    assert C.CUT == (6, 5) == wrap_cut_columns(64, 256, 2.0 ** -10, C.WRAP_TOL)
    assert C.EDGE_EXACT == (8, 7)
    edges = g6["edges"]
    assert (list(C.CUT), list(C.EDGE_EXACT), C.PAD_COLUMNS) == (edges["cut"], edges["edge_exact"],
                                                                edges["pad_columns"])
    w = wrap_fractions(64, 256, 2.0 ** -10)
    # spec §5.1: the worst kept-edge wrap is 0.11200% (column 5) and 0.11708% (column Nt - 5)
    assert (round(100 * np.nanmax(w[:, 5]), 5), round(100 * np.nanmax(w[:, 251]), 5)) == (0.112, 0.11708)
    assert np.nanmax(w[:, 6]) < C.WRAP_TOL and np.nanmax(w[:, 250]) < C.WRAP_TOL
    assert (C.EXACT_P, C.EXACT_TOL) == (24, 1e-7)


@pytest.mark.golden
def test_the_truth_is_the_first_reference_point(g1):
    for case in CASES:
        for model in ("whittle_raw", "band_raw"):
            ref = g1(case, model)["theta"][0]
            truth = C.config(case).truth(model)
            assert truth.dtype == np.float64 and np.array_equal(truth, ref), (case, model)
    assert C.config("manuscript_mid").truth("exact").shape == (6,)


def test_unknown_cases_and_the_chirp_are_refused():
    with pytest.raises(KeyError, match="manuscript_chirp"):
        C.config("manuscript_chirp")
    with pytest.raises(ValueError, match="fixed-frequency"):
        C.SinusoidConfig(fit_fdot=True)
    with pytest.raises(ValueError, match="A_n must be positive"):
        C.SinusoidConfig(A_n=0.0)
    with pytest.raises(ValueError, match="f_min"):
        C.SinusoidConfig(f_min=600.0)
    with pytest.raises(ValueError, match="Nf must be even"):
        C.SinusoidConfig(Nf=63)
    with pytest.raises(ValueError, match="unknown model"):
        C.config("manuscript_mid").truth("band_win")
    assert math.isclose(C.SinusoidConfig(Nf=8, Nt=16, dt=1 / 16, f_min=0.5).T_obs, 8.0)
