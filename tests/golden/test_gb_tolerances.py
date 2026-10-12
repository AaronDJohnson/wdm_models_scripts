"""B8: measured galactic-binary tolerances (G3 rows, G4 stress) and their MANIFEST summary."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from scripts.golden import measure_tolerances_gb as mt

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden"


@pytest.fixture(scope="module")
def measured():
    return json.loads((GOLDEN / "galactic_binary" / "tolerances_measured.json").read_text())


def test_manifest_summary(measured):
    m = json.loads((GOLDEN / "MANIFEST.json").read_text())
    assert "galactic_binary/tolerances_measured.json" in m["files"]
    summary = m["tolerances"]["galactic_binary"]
    assert summary["status"] == measured["status"] == "measured; pending user approval"
    assert summary["G3_targets"] == 38 and summary["G3_runs"] == 58


def test_g3_gates_cover_observations(measured):
    for target, t in measured["G3"].items():
        assert t["rows"] >= 3 * t["runs"], target
        assert t["gate_numpy"] >= max(10 * t["observed_numpy"], 1e-9)
        assert t["gate_jax"] >= max(10 * t["observed_jax"], 1e-8)
        assert t["observed_numpy"] <= 1e-6 and t["observed_jax"] <= 1e-6


def test_g3_observations_recomputed_from_rows(measured):
    worst = 0.0
    for path in (GOLDEN / "galactic_binary" / "chain_rows").glob("*/*.json"):
        if path.name != "index.json":
            worst = max([worst] + [r["abs_delta_numpy"] for r in json.loads(path.read_text())["rows"]])
    assert measured["summary"]["G3_max_observed_numpy"] == worst


def test_g4_masks_and_gates(measured):
    g4 = measured["G4"]
    assert g4["mask_mismatches"] == [] and measured["summary"]["G4_mask_mismatches"] == 0
    assert g4["cache_within_numpy_gate"]
    assert g4["gate_jax_well_conditioned"] <= 1e-6                     # spec 7.3 ceiling
    assert g4["gate_jax_well_conditioned"] >= max(10 * g4["observed_jax_well_conditioned"], 1e-10)
    assert measured["summary"]["G4_gate_jax_well_conditioned"] == g4["gate_jax_well_conditioned"]
    assert len(g4["per_target"]) >= 30


# --- the measurement script itself (B8 quality review) -------------------------------------------

OUT = "galactic_binary/tolerances_measured.json"


def test_measure_reproduces_committed_bytes():
    """Every G3/G4 number (incl. the G4 gates) is recomputed from the committed row/stress goldens."""
    assert mt.OUT == OUT
    assert mt.render(mt.measure(GOLDEN)) == (GOLDEN / OUT).read_bytes()


def _scratch_golden(tmp_path) -> Path:
    golden = tmp_path / "golden"
    for sub in ("chain_rows", "stress"):
        shutil.copytree(GOLDEN / "galactic_binary" / sub, golden / "galactic_binary" / sub)
    for rel in ("MANIFEST.json", OUT):
        (golden / rel).write_bytes((GOLDEN / rel).read_bytes())
    return golden


def test_refusal_comes_before_any_write(tmp_path, monkeypatch):
    """Without --replace a changed measurement is refused before golden/ is touched (scratch copy)."""
    golden = _scratch_golden(tmp_path)
    monkeypatch.setattr(mt, "GOLDEN", golden)
    snapshot = lambda: {p.relative_to(golden).as_posix(): p.read_bytes()   # noqa: E731
                        for p in golden.rglob("*") if p.is_file()}
    row_file = sorted((golden / "galactic_binary" / "chain_rows").glob("gb_3mo/*.fast.json"))[0]
    original = row_file.read_bytes()
    r = json.loads(original)
    r["rows"][0]["abs_delta_numpy"] = 3e-9                 # a re-capture that measures differently
    row_file.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    before = snapshot()
    with pytest.raises(SystemExit, match="--replace"):
        mt.cli([])
    assert snapshot() == before
    row_file.write_bytes(original)                         # same inputs: written, manifest merged
    assert mt.cli([]) == 0
    after = snapshot()
    assert after[OUT] == (GOLDEN / OUT).read_bytes()
    assert json.loads(after["MANIFEST.json"])["files"] == json.loads(before["MANIFEST.json"])["files"]


# --- synthetic inputs: failures must surface, never read as "zero observed" ----------------------

def _row_file(golden, *, backends=("numpy", "jax"), **row):
    rec = {"row": 0, "abs_delta_numpy": 1e-12, "abs_delta_jax": 0.0}
    rec.update(row)
    rec = {k: v for k, v in rec.items() if v is not None}
    path = golden / "galactic_binary" / "chain_rows" / "gb_3mo" / "exact_seed1.fast.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case": "gb_3mo", "model": "exact", "run": "exact_seed1", "tier": "fast",
                                "backends": list(backends), "rows": [rec]}))


def _stress_file(golden, *, backends=("numpy", "jax"), cache_class="FullCache", **point):
    p = {"name": "anchor", "marginal": False, "pd_margin": 1e-3, "source_out": False, "lnl_ref": -100.0,
         "lnl_cache": -100.0, "lnl_jax": -100.0}
    p.update(point)
    p = {k: v for k, v in p.items() if v is not None}
    path = golden / "galactic_binary" / "stress" / "gb_3mo" / "exact.fast.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case": "gb_3mo", "model": "exact", "tier": "fast", "n_points": 1,
                                "points": [p], "backends": list(backends), "cache_class": cache_class,
                                "counters_cache": None, "counters_jax": None}))


def test_synthetic_clean_inputs_measure(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, lnl_jax=-100.0 + 1e-9)
    m = mt.measure(tmp_path)
    assert m["G3"]["gb_3mo/exact"]["observed_numpy"] == 1e-12
    assert m["G4"]["mask_mismatches"] == [] and m["G4"]["observed_jax_well_conditioned"] == pytest.approx(1e-9)
    assert m["G4"]["gate_jax_well_conditioned"] == 10 * m["G4"]["observed_jax_well_conditioned"]


@pytest.mark.parametrize("delta", [1e-6, 5e-7, 3e-6])
def test_synthetic_jax_gate_above_ceiling_raises(tmp_path, delta):
    """10 x observed > 1e-6 at a well-conditioned point: S0 fails, the gate is never capped below it."""
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=1e-3, lnl_jax=-100.0 + delta)
    with pytest.raises(ValueError, match=r"fails its own data.*gb_3mo/exact/fast: observed"):
        mt.measure(tmp_path)


def test_synthetic_jax_gate_at_ceiling_and_ill_conditioned(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, lnl_ref=0.0, lnl_cache=0.0, lnl_jax=-1e-7)    # 10 x observed == 1e-6
    g4 = mt.measure(tmp_path)["G4"]
    assert g4["observed_jax_well_conditioned"] == 1e-7 and g4["gate_jax_well_conditioned"] == 1e-6
    _stress_file(tmp_path, pd_margin=1e-8, lnl_jax=-100.0 + 1e-6)      # condition-scaled, not gated
    g4 = mt.measure(tmp_path)["G4"]
    assert g4["observed_jax_well_conditioned"] == 0.0 and g4["condition_scaled_jax_max"] > 0.0


@pytest.mark.parametrize("ref,jax,margin", [(-100.0, -math.inf, 1e-3), (-math.inf, -100.0, -1e-3)])
def test_synthetic_mask_mismatch_is_recorded(tmp_path, ref, jax, margin):
    """A backend that disagrees with a correct reference (finite at PD, -inf at non-PD)."""
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=margin, lnl_ref=ref, lnl_cache=ref, lnl_jax=jax)
    m = mt.measure(tmp_path)
    assert m["G4"]["mask_mismatches"] == ["gb_3mo/exact/fast:anchor:lnl_jax"]
    assert m["summary"]["G4_mask_mismatches"] == 1


# The expected mask is the worker's ``expected_neg_inf = source_out or pd_margin < 0`` (gbw_stress):
# every evaluator, the NumPy reference included, is checked against it at non-marginal points.

@pytest.mark.parametrize("margin", [-1e-3, -1e-10])        # -1e-10 is non-marginal (|m| < 1e-10 is not)
def test_synthetic_negative_margin_finite_is_mismatch(tmp_path, margin):
    """Not PD (margin <= -1e-10): finite lnL anywhere fails S0, never a condition-scaled zero."""
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=margin, lnl_ref=-100.0, lnl_jax=-50.0, lnl_cache=-99.0)
    m = mt.measure(tmp_path)
    g4 = m["G4"]
    assert g4["mask_mismatches"] == [
        f"gb_3mo/exact/fast:anchor:{name} is finite, expected -inf (pd_margin {margin:.3e}, source_out False)"
        for name in ("lnl_ref", "lnl_cache", "lnl_jax")]
    assert m["summary"]["G4_mask_mismatches"] == 3
    assert g4["observed_jax_well_conditioned"] == 0.0 and g4["observed_cache_well_conditioned"] == 0.0
    assert g4["condition_scaled_jax_max"] == 0.0 and g4["condition_scaled_cache_max"] == 0.0


@pytest.mark.parametrize("name", ["lnl_jax", "lnl_cache"])
def test_synthetic_negative_margin_finite_backend_only(tmp_path, name):
    """Reference correctly -inf, one backend finite: exactly that backend is flagged."""
    _row_file(tmp_path)
    vals = {"lnl_ref": -math.inf, "lnl_cache": -math.inf, "lnl_jax": -math.inf, name: -100.0}
    _stress_file(tmp_path, pd_margin=-1e-3, **vals)
    assert mt.measure(tmp_path)["G4"]["mask_mismatches"] == [f"gb_3mo/exact/fast:anchor:{name}"]


def test_synthetic_source_out_finite_is_mismatch(tmp_path):
    """Source outside the GLASS box: -inf expected even when the covariance is PD."""
    _row_file(tmp_path)
    _stress_file(tmp_path, source_out=True)
    assert mt.measure(tmp_path)["G4"]["mask_mismatches"] == [
        f"gb_3mo/exact/fast:anchor:{name} is finite, expected -inf (pd_margin 1.000e-03, source_out True)"
        for name in ("lnl_ref", "lnl_cache", "lnl_jax")]


def test_synthetic_neg_inf_reference_at_pd_point_is_mismatch(tmp_path):
    """PD and in the box: a -inf lnL is a mismatch even when reference and cache agree on it; the
    backend that is right (finite) is not flagged for disagreeing with the wrong reference."""
    _row_file(tmp_path)
    _stress_file(tmp_path, lnl_ref=-math.inf, lnl_cache=-math.inf, lnl_jax=-100.0)
    m = mt.measure(tmp_path)
    assert m["G4"]["mask_mismatches"] == [
        "gb_3mo/exact/fast:anchor:lnl_ref is -inf, expected finite (pd_margin 1.000e-03, source_out False)",
        "gb_3mo/exact/fast:anchor:lnl_cache is -inf, expected finite (pd_margin 1.000e-03, source_out False)"]
    assert m["G4"]["observed_jax_well_conditioned"] == 0.0


@pytest.mark.parametrize("point,match", [
    ({"pd_margin": 0.0}, r"marginal=False but \|pd_margin\| = 0.000e\+00 < 1e-10"),
    ({"pd_margin": -1e-12}, r"marginal=False but \|pd_margin\| = 1.000e-12 < 1e-10"),
    ({"source_out": None}, r"has no \['source_out'\]")])
def test_synthetic_stress_inconsistent_point_raises(tmp_path, point, match):
    _row_file(tmp_path)
    _stress_file(tmp_path, **point)
    with pytest.raises(ValueError, match=match):
        mt.measure(tmp_path)


@pytest.mark.parametrize("margin", [1e-3, 1e-8])          # well-conditioned and condition-scaled
@pytest.mark.parametrize("name", ["lnl_jax", "lnl_cache"])
def test_synthetic_stress_nan_is_recorded_not_dropped(tmp_path, name, margin):
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=margin, **{name: math.nan})
    m = mt.measure(tmp_path)
    assert m["G4"]["mask_mismatches"] == [f"gb_3mo/exact/fast:anchor:{name} is nan (ref finite)"]
    assert m["summary"]["G4_mask_mismatches"] == 1


def test_synthetic_stress_nan_margin_raises(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=math.nan)
    with pytest.raises(ValueError, match="pd_margin = nan"):
        mt.measure(tmp_path)


@pytest.mark.parametrize("delta", [math.nan, math.inf])
@pytest.mark.parametrize("key", ["abs_delta_numpy", "abs_delta_jax"])
def test_synthetic_row_non_finite_delta_raises(tmp_path, key, delta):
    _row_file(tmp_path, **{key: delta})
    _stress_file(tmp_path)
    with pytest.raises(ValueError, match=f"{key} = {delta}"):
        mt.measure(tmp_path)


def test_synthetic_row_missing_backend_raises(tmp_path):
    _row_file(tmp_path, backends=("numpy",), abs_delta_jax=None)
    _stress_file(tmp_path)
    with pytest.raises(ValueError, match=r"backend\(s\) \['jax'\] not captured.*no abs_delta_jax"):
        mt.measure(tmp_path)


def test_synthetic_stress_missing_backend_raises(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, backends=("numpy",), lnl_jax=None)
    with pytest.raises(ValueError, match=r"backend\(s\) \['jax'\] not captured.*has no \['lnl_jax'\]"):
        mt.measure(tmp_path)


def test_synthetic_stress_missing_cache_value_raises(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, lnl_cache=None)                 # cache_class set but no lnl_cache
    with pytest.raises(ValueError, match=r"has no \['lnl_cache'\]"):
        mt.measure(tmp_path)
    _stress_file(tmp_path, cache_class=None, lnl_cache=None)   # Whittle: no cache class, fine
    assert mt.measure(tmp_path)["G4"]["mask_mismatches"] == []
