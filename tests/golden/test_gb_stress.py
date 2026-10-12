"""B5 (G4): GB stress points, fast tier (3- and 6-month targets, points 0-10)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden"
STRESS = GOLDEN / "galactic_binary" / "stress"
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


def test_fast_tier_pins_some_non_pd_states():
    failed = sum(_load(c, m)["counters_cache"]["pd_failed_states"]
                 for c in ("gb_3mo", "gb_6mo") for m in CACHED)
    assert failed >= 20          # tri / edge-exact / pad corners are not positive definite
