"""B5 (G4): GB stress points, fast tier (3- and 6-month targets, points 0-10)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden"
STRESS = GOLDEN / "galactic_binary" / "stress"
WORKERS = REL / "scripts" / "golden" / "workers"
MODELS = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact", "mid_raw",
          "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad", "tri_exact", "exact")
ENVELOPE_NAMES = ["anchor", "noise_hi", "noise_lo", "noise_mixed", "edge_h0", "edge_h1", "edge_h2",
                  "edge_h3", "edge_h4", "source_out_lnA", "source_out_f0"]
WHITTLE_NAMES = ["anchor", "noise_hi", "noise_lo", "noise_mixed", "edge_A_hi", "edge_E_lo",
                 "edge_cross", "level_up", "level_down", "source_out_lnA", "source_out_f0"]
CACHED = {"exact": "FullCache", "band_exact": "EdgeSchurCache", "mid_exact": "EdgeSchurCache",
          "band_pad": "MatrixCache", "mid_pad": "MatrixCache", "tri_raw": "MatrixCache",
          "tri_cut": "MatrixCache", "tri_pad": "MatrixCache", "tri_exact": "MatrixCache"}
#: Points whose covariance is this well conditioned must agree across evaluators.
WELL_CONDITIONED = 1e-6


def _load(case, model):
    return json.loads((STRESS / case / f"{model}.fast.json").read_text())


def _same_mask(a, b):
    return math.isinf(a) == math.isinf(b)


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo"])
@pytest.mark.parametrize("model", MODELS)
def test_stress_fast_tier(case, model):
    rel = f"galactic_binary/stress/{case}/{model}.fast.json"
    assert rel in json.loads((GOLDEN / "MANIFEST.json").read_text())["files"]
    r = _load(case, model)
    names = WHITTLE_NAMES if model.startswith("whittle") else ENVELOPE_NAMES
    assert [p["name"] for p in r["points"]] == names and r["n_points"] == 11
    assert r["cache_class"] == CACHED.get(model)
    pts = r["points"]
    assert math.isfinite(pts[0]["lnl_ref"]) and not pts[0]["marginal"]
    for p in pts:
        # Spec G4: every lnL is finite or exactly -inf, marginal points included (``marginal``
        # lifts only the mask and delta comparisons below, never this check).
        for name in ("lnl_ref", "lnl_cache", "lnl_jax"):
            if name in p:
                v = p[name]
                assert math.isfinite(v) or v == -math.inf, f"{p['name']}: {name} = {v}"
        assert p["mask_consistent"], p["name"]
        if p["source_out"]:
            assert p["lnl_ref"] == -math.inf and p["lnl_jax"] == -math.inf
            assert p.get("lnl_cache", -math.inf) == -math.inf
        if p["marginal"]:
            continue
        if "lnl_cache" in p:
            assert _same_mask(p["lnl_ref"], p["lnl_cache"])
        assert _same_mask(p["lnl_ref"], p["lnl_jax"])
        if math.isfinite(p["lnl_ref"]) and p["pd_margin"] >= WELL_CONDITIONED:
            tol = 1e-10 + 1e-12 * abs(p["lnl_ref"])    # spec §7.3 G4 NumPy cache gate (measured)
            if "lnl_cache" in p:
                assert abs(p["lnl_cache"] - p["lnl_ref"]) <= tol, p["name"]
            assert abs(p["lnl_jax"] - p["lnl_ref"]) <= 1e-6, p["name"]
    if r["counters_cache"] is not None:
        c = r["counters_cache"]
        distinct = len({tuple(p["theta"][8:]) for p in pts})
        failed_rows = sum(1 for p in pts if p["lnl_cache"] == -math.inf and not p["source_out"])
        assert c["evaluations"] == 11 and c["noise_states"] == distinct == 9
        assert c["pd_failed_rows"] == failed_rows == c["pd_failed_states"]
        if not any(p["marginal"] for p in pts):
            assert r["counters_jax"] == c


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo"])
@pytest.mark.parametrize("model", ["whittle_raw", "whittle_cut", "band_raw", "band_cut", "mid_raw",
                                   "mid_cut"])
def test_diagonal_targets_carry_the_pd_margin(case, model):
    """Every target whose covariance is diagonal (``gbw_stress.Covariances.matrix`` kind 'diag':
    Whittle raw/cut and the band/mid raw/cut WDM diagonals) is captured with the diagonal PD margin
    (review I2): every entry is > 0 at all 11 points, so the margin is 1.0 and nothing is exempted
    as marginal. Under the old min/max|.| the band/mid +-8 corners sat at 8e-18..1e-11 (falsely
    marginal); the Whittle files stored a 0.94..0.99 conditioning ratio (never marginal under either
    rule, but not the margin the worker now defines)."""
    pts = _load(case, model)["points"]
    assert all(p["pd_margin_channels"] == [1.0, 1.0] and p["pd_margin"] == 1.0 for p in pts)
    assert not any(p["marginal"] for p in pts)


def test_fast_tier_pins_some_non_pd_states():
    failed = sum(_load(c, m)["counters_cache"]["pd_failed_states"]
                 for c in ("gb_3mo", "gb_6mo") for m in CACHED)
    assert failed >= 20          # tri / edge-exact / pad corners are not positive definite


@pytest.mark.parametrize("name,value", [("lnl_ref", math.inf), ("lnl_cache", math.nan),
                                        ("lnl_jax", math.nan), ("lnl_jax", math.inf)])
def test_marginal_point_must_be_finite_or_neg_inf(monkeypatch, name, value):
    """A NaN or +inf lnL at a marginal point fails the fast-tier check (final review I2)."""
    r = _load("gb_3mo", "exact")
    marginal = [p for p in r["points"] if p["marginal"] and not p["source_out"]]
    assert marginal and name in marginal[0]
    marginal[0][name] = value
    monkeypatch.setitem(globals(), "_load", lambda case, model: r)
    with pytest.raises(AssertionError, match=f"{marginal[0]['name']}: {name} = {value}"):
        test_stress_fast_tier("gb_3mo", "exact")


# --- the worker's PD margin (gbw_stress._margin), loaded in-process ---------------------------------

@pytest.fixture
def gbw_stress(monkeypatch):
    """``workers/gbw_stress.py`` and its two sibling imports, by file path. Neither imports frozen
    code at module level (``gbw_lineage_c.load`` does that lazily, in the lineage-C worker only);
    the ``sys.modules`` entries are removed again after the test."""
    for name in ("gbw_common", "gbw_lineage_c", "gbw_stress"):
        spec = importlib.util.spec_from_file_location(name, WORKERS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["gbw_stress"]


def test_diag_margin_measures_loss_of_pd_not_conditioning(gbw_stress):
    """A diagonal covariance is PD iff every entry is > 0, however ill-conditioned (review I2):
    margin 1.0 then; otherwise a sign-preserving value <= 0 (min / max|.|)."""
    m = gbw_stress._margin
    assert m("diag", np.array([1e-18, 1.0, 3.0])) == 1.0         # was 3.3e-19: falsely "marginal"
    assert m("diag", np.array([2.0, 5.0])) == 1.0
    assert m("diag", np.array([-1.0, 4.0])) == -0.25             # not PD: negative
    assert m("diag", np.array([-1e-14, 4.0])) == -2.5e-15        # not PD, and inside the marginal band
    assert m("diag", np.array([0.0, 4.0])) == 0.0                # singular: marginal, not "not PD"


def test_dense_margin_unchanged(gbw_stress):
    m = gbw_stress._margin
    assert m("dense", np.array([[2.0, 1.0], [1.0, 2.0]])) == pytest.approx(1.0 / 3.0, rel=1e-15)
    assert m("dense", np.array([[1.0, 2.0], [2.0, 1.0]])) == pytest.approx(-1.0 / 3.0, rel=1e-15)
    assert m("dense", np.diag([1e-18, 1.0])) == 1e-18
