"""F-A8: the ``prepare`` port (spec §5.2 prepare.py, §6 ``--from-shipped``): from the shipped
segments and stores every deterministic paper key regenerates ``np.array_equal`` to the shipped
prepared files, ``snr_full`` is the spec's literal bit for bit, ``fit_mu`` refits ``mu_fit`` to
<= 1e-6, and ``calc_Nt`` reproduces the legacy function (lineage C, recorded below)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wdm_models.galactic_binary import config as C
from wdm_models.galactic_binary import prepare as PP
from wdm_models.galactic_binary.prepared import FIT_KEYS, PAPER_KEYS, PreparedCase

DATA = Path(__file__).resolve().parents[3] / "data"
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
SLOW = ("gb_6mo", "gb_12mo")
#: The legacy ``prepare.calc_Nt(case)`` (lineage C, ``gb_legacy calcnt``, 2026-10-13): NT* on
#: config.NT_BAND and the nearest valid tiling.  The shipped files' ``nt_star`` (253.63 /
#: 447.78 / 928.49) is the earlier tiling-dependent estimate that calc_Nt replaced.
LEGACY_CALC_NT = {"gb_3mo": (246.86598221011158, 240), "gb_6mo": (444.3893998481366, 450),
                  "gb_12mo": (925.7765186242784, 900)}
SHIPPED_NT_STAR = {"gb_3mo": 253.63256112042106, "gb_6mo": 447.7834049360398,
                   "gb_12mo": 928.4897855039914}
#: WDM II Table V (tab:LISA-settings): lowest and highest analysed frequency [mHz], 5 digits.
TABLE_V = {"gb_3mo": (0.98174, 1.02727), "gb_6mo": (0.97920, 1.02080), "gb_12mo": (0.97745, 1.02023)}


def _cases(slow=SLOW):
    return [pytest.param(c, marks=pytest.mark.slow) if c in slow else c for c in CASES]


def _window(case):
    d = PP.load_case(case)
    win = PP.analysis_window(d, C.ANALYSIS_NT[case])
    d = PP.restrict(d, win["lo"], win["n"])
    lo = win["nominal"][0] - d["bin_offset"]
    return d, win, slice(lo, lo + win["nominal"][1] - win["nominal"][0])


def test_psd_components_needs_the_fitted_foreground():
    with pytest.raises(ValueError, match="fitted GLASS foreground"):
        PP.psd_components(np.array([1e-3]), None)
    inst, gal = PP.psd_components(np.array([1e-3, 2e-3]), np.array([-84.6, -5.8, 1.7, -5.5, -11.5]))
    assert inst.shape == gal.shape == (2, 2) and np.array_equal(inst[0], inst[1])


def test_tiling_helpers_are_the_legacy_geometry():
    assert PP.even_divisors(48).tolist() == [2, 4, 6, 8, 12, 16, 24]
    for case, (star, valid) in LEGACY_CALC_NT.items():
        N = {"gb_3mo": 1576800, "gb_6mo": 3153600, "gb_12mo": 6307200}[case]
        assert PP.candidates(N, star, count=1) == [valid]
    assert PP.candidates(3153600, 444.3893998481366) == [400, 432, 438, 450, 480]   # 438 is the paper's


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_window_and_restriction_are_the_shipped_ones(case):
    """No JAX: the window, its bins, units and PSDs from the segment, bit for bit."""
    prep = PreparedCase.load(case)
    d, win, freq = _window(case)
    assert (d["bin_offset"], d["n_bins"], win["band"], win["first"]) == (
        prep.bin_offset, prep.n_bins, prep.window_band, prep.first_band)
    assert freq == prep.freq and (win["source"][0], win["source"][1]) == prep.source_band
    for key in ("z", "resid", "injected", "sI", "sG", "sigma0", "S_ref", "basis_L", "inject_glass",
                "fg_params"):
        assert np.array_equal(d[key], prep.arrays[key]), key
    lo_mhz = (prep.bin_offset + freq.start) / prep.T * 1e3
    hi_mhz = (prep.bin_offset + freq.stop - 1) / prep.T * 1e3
    assert (round(lo_mhz, 5), round(hi_mhz, 5)) == TABLE_V[case]


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_snr_full_is_the_spec_literal_bit_for_bit(case, s5):
    prep = PreparedCase.load(case)
    d, _, freq = _window(case)
    got = PP.snr_full(d, prep.mu_fit, freq, prep.exact_block)
    s5(f"G5/snr_full/{case}", abs(got - C.SNR_FULL[case]), 0.0)
    assert got == C.SNR_FULL[case]


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_deterministic_keys_regenerate_bit_for_bit(case, bg, tmp_path, s5):
    """``prepare`` with the shipped fits, then ``add_matrix``: every paper key equals the
    shipped one exactly (``nt_star`` is passed through: see the calc_Nt test)."""
    ship = PreparedCase.load(case)
    report = json.loads((DATA / "galactic_binary" / "prepared" / f"{case}_report.json").read_text())
    nt = {k: report[k] for k in ("nt_star", "nt_star_A", "nt_star_E", "r_f", "r_t")}
    path = PP.prepare(case, out_dir=tmp_path, fits={"mu_fit": ship.mu_fit, "mu_freq": ship.mu_freq, "nt": nt})
    with np.load(path, allow_pickle=False) as z:
        assert tuple(z.files) == PAPER_KEYS[:32]
    assert PP.add_matrix(case, out_dir=tmp_path) == path
    with np.load(path, allow_pickle=False) as z:
        new = {k: np.array(z[k]) for k in z.files}
    assert tuple(new) == PAPER_KEYS
    bad = [k for k in PAPER_KEYS
           if not (new[k].dtype == ship.arrays[k].dtype and np.array_equal(new[k], ship.arrays[k]))]
    s5(f"G5/prepare_port/{case}/keys", len(bad), 0.0)                  # keys not bit for bit
    assert not bad, bad
    rep = json.loads((tmp_path / f"{case}_report.json").read_text())
    assert rep["snr_full"] == C.SNR_FULL[case] and rep["residual_chi2_per_bin"] == report["residual_chi2_per_bin"]
    assert rep["segment"] == PP.segment_rel(case) and rep["segment_sha256"] == PP.SEGMENT_SHA256[case]
    with pytest.raises(RuntimeError, match="already holds"):
        PP.add_matrix(case, out_dir=tmp_path)                           # never overwrites a key


@pytest.mark.golden
@pytest.mark.parametrize("case", _cases())
def test_fit_mu_refits_the_shipped_mu(case, bg, s5):
    ship = PreparedCase.load(case)
    d, _, freq = _window(case)
    mu, mu_freq = PP.fit_mu(d, freq, ship.exact_block)
    worst = max(np.max(np.abs(mu - ship.mu_fit)), np.max(np.abs(mu_freq - ship.mu_freq)))
    s5(f"G5/prepare_port/{case}/mu_fit", worst, 1e-6)
    assert worst <= 1e-6


@pytest.mark.golden
@pytest.mark.parametrize("case", _cases())
def test_calc_nt_is_the_legacy_function(case, bg, s5):
    star, valid = LEGACY_CALC_NT[case]
    got = PP.calc_Nt(case)
    s5(f"G5/prepare_port/{case}/calc_nt", abs(got["nt_star"] / star - 1.0), 1e-6)
    assert abs(got["nt_star"] / star - 1.0) <= 1e-6 and got["nt_valid"] == valid
    assert got["nt_band_hz"] == list(C.NT_BAND)
    shipped = float(PreparedCase.load(case).arrays["nt_star"])
    assert shipped == SHIPPED_NT_STAR[case] and abs(shipped / star - 1.0) > 2e-3   # superseded estimate


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_full_prepare_from_the_shipped_segments(case, bg, tmp_path):
    """The whole preparation (fits included; ~2 / 4 / 20 s): deterministic keys exact, the
    fitted ones at 1e-6 (``nt_star`` against the legacy calc_Nt)."""
    ship = PreparedCase.load(case)
    PP.prepare(case, out_dir=tmp_path)
    path = PP.add_matrix(case, out_dir=tmp_path)
    with np.load(path, allow_pickle=False) as z:
        new = {k: np.array(z[k]) for k in z.files}
    for key in PAPER_KEYS:
        if key in FIT_KEYS:
            continue
        assert np.array_equal(new[key], ship.arrays[key]), key
    assert np.max(np.abs(new["mu_fit"] - ship.mu_fit)) <= 1e-6
    assert np.max(np.abs(new["mu_freq"] - ship.mu_freq)) <= 1e-6
    assert abs(float(new["nt_star"]) / LEGACY_CALC_NT[case][0] - 1.0) <= 1e-6
