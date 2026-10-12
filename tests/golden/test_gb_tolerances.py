"""B8: measured galactic-binary tolerances (G3 rows, G4 stress) and their MANIFEST summary;
F4: the gates the user approved on 2026-10-12 (spec 7.3), rendered by the measurement script."""

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
    assert summary["status"] == measured["status"] == "APPROVED by the user 2026-10-12"
    assert summary["G3_targets"] == 38 and summary["G3_runs"] == 58
    assert summary == dict(measured["summary"], status=measured["status"], file=OUT)
    assert summary["approved_on"] == "2026-10-12"
    assert summary["decisions"] == {"D2": "Uniform 1e-8", "D3": "Pooled 8.3e-8, report the rest"}
    assert summary["approved_gates"] == {"G3_numpy": 1e-8, "G3_jax": 1e-8, "G4_jax_well_conditioned": 8.32e-8,
                                         "G4_numpy_cache_abs": 1e-10, "G4_numpy_cache_rel": 1e-12}
    assert summary["G4_proposed_gate_jax_well_conditioned"] == measured["G4"]["proposed_gate_jax_well_conditioned"]
    assert "G4_gate_jax_well_conditioned" not in summary          # the proposal is never read as the gate


def test_g3_proposals_cover_observations(measured):
    """The original B8 proposals (10 x observed, floors 1e-9 / 1e-8) stay recorded, informational."""
    for target, t in measured["G3"].items():
        assert t["rows"] >= 3 * t["runs"], target
        assert t["proposed_gate_numpy"] >= max(10 * t["observed_numpy"], 1e-9)
        assert t["proposed_gate_jax"] >= max(10 * t["observed_jax"], 1e-8)
        assert t["observed_numpy"] <= 1e-6 and t["observed_jax"] <= 1e-6
        assert "gate_numpy" not in t and "gate_jax" not in t      # never mistaken for the approved gate


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
    assert g4["proposed_gate_jax_well_conditioned"] <= 1e-6            # spec 7.3 ceiling
    assert g4["proposed_gate_jax_well_conditioned"] >= max(10 * g4["observed_jax_well_conditioned"], 1e-10)
    assert (measured["summary"]["G4_proposed_gate_jax_well_conditioned"]
            == g4["proposed_gate_jax_well_conditioned"])
    assert len(g4["per_target"]) >= 30


def test_g4_per_target_jax_proposals(measured):
    """Per-target G4 JAX proposals max(10 x observed, 1e-10) next to the pooled one (review harness
    F9), kept as informational (D3); the pooled proposal is the largest of them."""
    g4 = measured["G4"]
    per = g4["proposed_gate_jax_well_conditioned_per_target"]
    assert sorted(per) == sorted(g4["per_target"])
    for target, gate in per.items():
        assert gate == max(10 * g4["per_target"][target]["jax_well"], 1e-10), target
    assert max(per.values()) == g4["proposed_gate_jax_well_conditioned"] <= 1e-6


def test_g4_marginal_threshold_from_each_file(measured):
    for target, t in measured["G4"]["per_target"].items():
        stress = GOLDEN / "galactic_binary" / "stress" / "{}/{}.{}.json".format(*target.split("/"))
        assert t["marginal_threshold"] == json.loads(stress.read_text())["marginal_threshold"] == 1e-10


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


def _stress_file(golden, *, backends=("numpy", "jax"), cache_class="FullCache", threshold=1e-10,
                 **point):
    """One stress file with one point. ``expected_neg_inf`` defaults to the worker's rule
    (``source_out or pd_margin < 0``); ``None`` drops a key (``threshold=None``: no
    ``marginal_threshold``)."""
    p = {"name": "anchor", "marginal": False, "pd_margin": 1e-3, "source_out": False, "lnl_ref": -100.0,
         "lnl_cache": -100.0, "lnl_jax": -100.0}
    p.update(point)
    if "expected_neg_inf" not in point:
        p["expected_neg_inf"] = bool(p["source_out"] or p["pd_margin"] < 0)
    p = {k: v for k, v in p.items() if v is not None}
    path = golden / "galactic_binary" / "stress" / "gb_3mo" / "exact.fast.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    r = {"case": "gb_3mo", "model": "exact", "tier": "fast", "n_points": 1, "points": [p],
         "backends": list(backends), "cache_class": cache_class, "counters_cache": None,
         "counters_jax": None}
    if threshold is not None:
        r["marginal_threshold"] = threshold
    path.write_text(json.dumps(r))


def test_synthetic_clean_inputs_measure(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, lnl_jax=-100.0 + 1e-9)
    m = mt.measure(tmp_path)
    assert m["G3"]["gb_3mo/exact"]["observed_numpy"] == 1e-12
    assert m["G4"]["mask_mismatches"] == [] and m["G4"]["observed_jax_well_conditioned"] == pytest.approx(1e-9)
    assert m["G4"]["proposed_gate_jax_well_conditioned"] == 10 * m["G4"]["observed_jax_well_conditioned"]


@pytest.mark.parametrize("delta", [1e-6, 5e-7, 3e-6])
def test_synthetic_jax_gate_above_ceiling_raises(tmp_path, delta):
    """10 x observed > 1e-6 at a well-conditioned point: S0 fails, the gate is never capped below it."""
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=1e-3, lnl_jax=-100.0 + delta)
    with pytest.raises(ValueError, match=r"fails its own data.*gb_3mo/exact/fast: observed"):
        mt.measure(tmp_path)


def test_synthetic_jax_gate_at_ceiling_and_ill_conditioned(tmp_path):
    """The proposal rule alone (``measure_stress``): 1e-7 observed would fail the approved 8.32e-8."""
    _row_file(tmp_path)
    _stress_file(tmp_path, lnl_ref=0.0, lnl_cache=0.0, lnl_jax=-1e-7)    # 10 x observed == 1e-6
    g4 = mt.measure_stress(tmp_path / "galactic_binary")
    assert g4["observed_jax_well_conditioned"] == 1e-7 and g4["proposed_gate_jax_well_conditioned"] == 1e-6
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


# --- marginal points (review I2): validated, never just skipped ----------------------------------

@pytest.mark.parametrize("value", [math.nan, math.inf])
@pytest.mark.parametrize("name", ["lnl_ref", "lnl_cache", "lnl_jax"])
def test_synthetic_marginal_point_nan_or_pos_inf_raises(tmp_path, name, value):
    """Exempt from the mask/delta checks, but still finite or exactly -inf (spec G4)."""
    _row_file(tmp_path)
    _stress_file(tmp_path, marginal=True, pd_margin=1e-12, **{name: value})
    with pytest.raises(ValueError, match=f"point anchor is marginal but {name} = {value} "):
        mt.measure(tmp_path)


@pytest.mark.parametrize("values", [(-math.inf, -math.inf, -math.inf), (-1e3, -5.0, -7.0e3)])
def test_synthetic_marginal_point_finite_or_neg_inf_is_counted(tmp_path, values):
    """Identical -inf masks, any deltas (D3: marginal points are not gated)."""
    _row_file(tmp_path)
    ref, cache, jax = values
    _stress_file(tmp_path, marginal=True, pd_margin=-1e-12, lnl_ref=ref, lnl_cache=cache, lnl_jax=jax)
    g4 = mt.measure(tmp_path)["G4"]
    assert g4["per_target"]["gb_3mo/exact/fast"]["marginal"] == 1 and g4["mask_mismatches"] == []
    assert g4["observed_jax_well_conditioned"] == 0.0 and g4["condition_scaled_jax_max"] == 0.0


@pytest.mark.parametrize("ref,cache,jax,bad", [(-1e3, -math.inf, -5.0, ["lnl_cache"]),
                                               (-math.inf, -math.inf, -5.0, ["lnl_jax"]),
                                               (-1e3, -math.inf, -math.inf, ["lnl_cache", "lnl_jax"])])
def test_synthetic_marginal_point_mask_must_be_identical(tmp_path, ref, cache, jax, bad):
    """D3 (approved 2026-10-12): marginal points are finite or -inf WITH identical -inf masks; an
    evaluator whose mask differs from the reference's is a mismatch (S0 fails)."""
    _row_file(tmp_path)
    _stress_file(tmp_path, marginal=True, pd_margin=1e-12, lnl_ref=ref, lnl_cache=cache, lnl_jax=jax)
    m = mt.measure(tmp_path)
    assert m["G4"]["mask_mismatches"] == [
        f"gb_3mo/exact/fast:anchor:{name} -inf mask differs from lnl_ref at a marginal point" for name in bad]
    assert m["summary"]["G4_mask_mismatches"] == len(bad)


def test_synthetic_marginal_threshold_is_read_from_the_file(tmp_path):
    """The file's own ``marginal_threshold`` classifies its points, not a constant."""
    _row_file(tmp_path)
    _stress_file(tmp_path, threshold=1e-12, pd_margin=1e-11, lnl_jax=-100.0 + 1e-12)
    t = mt.measure(tmp_path)["G4"]["per_target"]["gb_3mo/exact/fast"]
    assert t["marginal"] == 0 and t["jax_cond"] > 0.0 and t["marginal_threshold"] == 1e-12
    _stress_file(tmp_path, threshold=1e-8, pd_margin=1e-9)        # marginal under 1e-8, not flagged
    with pytest.raises(ValueError, match=r"marginal=False but \|pd_margin\| = 1.000e-09 < 1e-08"):
        mt.measure(tmp_path)


@pytest.mark.parametrize("threshold", [None, 0.0, -1e-10, math.nan, "1e-10"])
def test_synthetic_marginal_threshold_missing_or_invalid_raises(tmp_path, threshold):
    _row_file(tmp_path)
    _stress_file(tmp_path)
    path = tmp_path / "galactic_binary" / "stress" / "gb_3mo" / "exact.fast.json"
    r = json.loads(path.read_text())
    if threshold is None:
        del r["marginal_threshold"]
    else:
        r["marginal_threshold"] = threshold
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="marginal_threshold"):
        mt.measure(tmp_path)


@pytest.mark.parametrize("point,match", [
    ({"marginal": True, "pd_margin": 1e-3}, r"marginal=True but \|pd_margin\| = 1.000e-03 >= 1e-10"),
    ({"marginal": True, "pd_margin": 1e-10}, r"marginal=True but \|pd_margin\| = 1.000e-10 >= 1e-10"),
    ({"marginal": True, "pd_margin": math.nan}, r"pd_margin = nan"),
    ({"expected_neg_inf": True}, r"expected_neg_inf=True but source_out False, pd_margin 1.000e-03"),
    ({"expected_neg_inf": False, "pd_margin": -1e-3, "lnl_ref": -math.inf, "lnl_cache": -math.inf,
      "lnl_jax": -math.inf}, r"expected_neg_inf=False but source_out False, pd_margin -1.000e-03"),
    ({"expected_neg_inf": False, "marginal": True, "pd_margin": -1e-12},
     r"expected_neg_inf=False but source_out False, pd_margin -1.000e-12"),
    ({"expected_neg_inf": None}, r"has no \['expected_neg_inf'\]")])
def test_synthetic_stored_flags_disagree_with_their_rule_raises(tmp_path, point, match):
    """``marginal`` == |pd_margin| < marginal_threshold and ``expected_neg_inf`` == source_out or
    pd_margin < 0 at every point, marginal ones included; a disagreement is never trusted."""
    _row_file(tmp_path)
    _stress_file(tmp_path, **point)
    with pytest.raises(ValueError, match=match):
        mt.measure(tmp_path)


# --- the user's approval of 2026-10-12 (spec 7.3; Task F4) ---------------------------------------

def test_decision_table_is_explicit_and_dated(measured):
    assert mt.APPROVED_ON == "2026-10-12" and mt.STATUS == "APPROVED by the user 2026-10-12"
    assert {k: d["label"] for k, d in mt.DECISIONS.items()} == {"D2": "Uniform 1e-8",
                                                                "D3": "Pooled 8.3e-8, report the rest"}
    assert mt.APPROVED == {"G3_numpy": 1e-8, "G3_jax": 1e-8, "G4_jax_well_conditioned": 8.32e-8,
                           "G4_numpy_cache_abs": 1e-10, "G4_numpy_cache_rel": 1e-12}
    assert measured["approval"] == {"date": "2026-10-12", "decisions": mt.DECISIONS}
    assert measured["status"] == mt.STATUS
    # D3 keeps the NumPy-cache gate as proposed; D2 keeps the G3 JAX gate at its 1e-8 floor
    assert (mt.FLOOR["G4_numpy_abs"], mt.FLOOR["G4_numpy_rel"]) == (mt.APPROVED["G4_numpy_cache_abs"],
                                                                  mt.APPROVED["G4_numpy_cache_rel"])
    assert mt.FLOOR["G3_jax"] == mt.APPROVED["G3_jax"]


def test_d2_uniform_g3_gate_for_all_38_targets(measured):
    """One NumPy G3 gate of 1e-8 for every target (the flat 1e-9 proposal and the per-target
    10 x values are withdrawn); JAX stays 1e-8. Every stored row passes it."""
    g3 = measured["approved"]["G3"]
    assert sorted(g3["numpy"]) == sorted(g3["jax"]) == sorted(measured["G3"]) and len(g3["numpy"]) == 38
    assert set(g3["numpy"].values()) == {1e-8} and set(g3["jax"].values()) == {1e-8}
    n = 0
    for path in (GOLDEN / "galactic_binary" / "chain_rows").glob("*/*.json"):
        if path.name == "index.json":
            continue
        r = json.loads(path.read_text())
        target = f"{r['case']}/{r['model']}"
        for row in r["rows"]:
            assert row["abs_delta_numpy"] <= g3["numpy"][target] and row["abs_delta_jax"] <= g3["jax"][target]
            n += 1
    assert n == sum(t["rows"] for t in measured["G3"].values()) >= 3 * 58


def test_d3_pooled_g4_gate_applies_to_every_target(measured):
    """One JAX gate of 8.32e-8 at well-conditioned points (PD margin >= 1e-6) for every stress
    target; the per-target proposals stay in the file as informational values."""
    g4 = measured["approved"]["G4"]
    targets = sorted({k.rsplit("/", 1)[0] for k in measured["G4"]["per_target"]})
    assert len(targets) == 30
    assert g4["jax_well_conditioned"] == 8.32e-8 and g4["well_conditioned_margin"] == 1e-6
    assert g4["jax_well_conditioned_per_target"] == {t: 8.32e-8 for t in targets}
    assert (g4["numpy_cache_abs"], g4["numpy_cache_rel"]) == (1e-10, 1e-12)
    per = measured["G4"]["proposed_gate_jax_well_conditioned_per_target"]
    assert min(per.values()) < 8.32e-8                   # informational only: not the operative gate
    n = 0
    for path in (GOLDEN / "galactic_binary" / "stress").glob("*/*.json"):
        r = json.loads(path.read_text())
        gate = g4["jax_well_conditioned_per_target"][f"{r['case']}/{r['model']}"]
        for p in r["points"]:
            if not p["marginal"] and p["pd_margin"] >= 1e-6 and math.isfinite(p["lnl_ref"]):
                assert abs(p["lnl_jax"] - p["lnl_ref"]) <= gate, (path.name, p["name"])
                n += 1
    assert n > 0


def test_d3_ill_conditioned_points_are_reported_not_gated(measured):
    """1e-10 <= margin < 1e-6: condition-scaled and reported (3-mo exact edge_h3 differs by ~5e-6,
    far above 8.32e-8), never gated."""
    assert "not gated" in measured["approved"]["G4"]["ill_conditioned"]
    assert measured["G4"]["condition_scaled_jax_max"] > 0.0
    worst = 0.0
    for path in (GOLDEN / "galactic_binary" / "stress").glob("*/*.json"):
        for p in json.loads(path.read_text())["points"]:
            if not p["marginal"] and 1e-10 <= p["pd_margin"] < 1e-6 and math.isfinite(p["lnl_ref"]):
                worst = max(worst, abs(p["lnl_jax"] - p["lnl_ref"]))
    assert worst > 8.32e-8


def test_d3_marginal_rule_is_recorded(measured):
    assert measured["approved"]["G4"]["marginal"] == (
        "finite or -inf with identical -inf masks across evaluators; deltas not gated")


def test_synthetic_ill_conditioned_point_is_never_gated(tmp_path):
    _row_file(tmp_path)
    _stress_file(tmp_path, pd_margin=1e-8, lnl_jax=-100.0 + 1e-3, lnl_cache=-100.0 + 1e-3)
    g4 = mt.measure(tmp_path)["G4"]                     # no raise: reported, condition-scaled
    assert g4["observed_jax_well_conditioned"] == 0.0 and g4["condition_scaled_jax_max"] > 0.0


@pytest.mark.parametrize("row,point,match", [
    ({"abs_delta_numpy": 1.5e-8}, {}, r"G3 gb_3mo/exact: observed numpy 1.500e-08 > approved 1e-08"),
    ({"abs_delta_jax": 1.5e-8}, {}, r"G3 gb_3mo/exact: observed jax 1.500e-08 > approved 1e-08"),
    ({}, {"lnl_jax": -100.0 + 9e-8},
     r"G4 gb_3mo/exact/fast: observed JAX 9.\d+e-08 > approved 8.32e-08"),
    ({}, {"lnl_cache": -100.0 + 3e-10}, r"G4 NumPy cache .* 1e-10 \+ 1e-12\|lnL\|")])
def test_synthetic_data_failing_an_approved_gate_raise(tmp_path, row, point, match):
    """Inside the 10 x proposal rule (and the 1e-6 ceiling) but above the approved gate: S0 fails."""
    _row_file(tmp_path, **row)
    _stress_file(tmp_path, **point)
    with pytest.raises(ValueError, match="an approved gate fails its own data.*" + match):
        mt.measure(tmp_path)
