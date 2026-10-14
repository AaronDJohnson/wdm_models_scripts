"""F-C6: scripts/galactic_binary/validate.py -- the G3/G4/G5/G6 rules, fail-closed, and small
real runs (the 38-target gate itself is tests/golden/test_s6_*.py, F-C8).

Synthetic inputs pin the rules (masks, expected masks, NaN, the well/ill/marginal classes, the
counter expectations); one cheap target per golden kind runs for real; a builder that returns
the wrong model, or NaN, must fail; the CLI writes nothing under the release tree.  P4P (I-1):
a non-finite stress reference or G4 gate, and a golden whose sha256 is not the one
golden/MANIFEST.json pins, fail closed (``_golden_copy`` re-signs a tampered copy so the rule
behind the sha256 check is reached).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil

import numpy as np
import pytest

from scripts.galactic_binary import validate as V
from scripts.golden.paths import legacy_available
from wdm_models.sampling.validate import ValidationError

pytestmark = pytest.mark.golden
WELL = 1e-6
STRESS = "galactic_binary/stress/gb_3mo/band_raw.fast.json"
TOL = "galactic_binary/tolerances_measured.json"


def _golden_copy(tmp_path, edits=None, *, resign=True):
    """A copy of golden/galactic_binary and golden/MANIFEST.json; ``edits`` maps a relpath to a
    function that edits its parsed JSON in place; the edited files are re-signed in the copy's
    MANIFEST unless ``resign`` is False (a tampered golden)."""
    root = tmp_path / "golden"
    shutil.copytree(V.GOLDEN / "galactic_binary", root / "galactic_binary")
    man = json.loads((V.GOLDEN / "MANIFEST.json").read_text())
    for rel, edit in (edits or {}).items():
        rec = json.loads((root / rel).read_text())
        edit(rec)
        (root / rel).write_text(json.dumps(rec, indent=1))
        if resign:
            man["files"][rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    (root / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return root


def _point(name, key, value):
    def edit(rec):
        next(p for p in rec["points"] if p["name"] == name)[key] = value
    return edit


def _shift_anchor(rec):                       # the control: a finite reference 1.0 away fails
    p = next(p for p in rec["points"] if p["name"] == "anchor")
    p["lnl_ref"] = float(p["lnl_ref"]) + 1.0


def _g4_rule(key, value):
    def edit(rec):
        rec["approved"]["G4"][key] = value
    return edit


def _pts(margin, ref, *, source_out=None, marginal_thr=1e-10):
    margin, ref = np.asarray(margin, float), np.asarray(ref, float)
    so = np.zeros(ref.size, bool) if source_out is None else np.asarray(source_out, bool)
    return {"margin": margin, "marginal": np.abs(margin) < marginal_thr, "source_out": so,
            "expected_neg_inf": so | (margin < 0)}


def test_stress_rules_on_synthetic_points():
    #           well     well      ill       marginal  not PD   source out
    margin = [0.5, 0.2, 1e-8, 1e-12, -1e-3, 0.5]
    ref = np.array([-100.0, -200.0, -300.0, -400.0, -np.inf, -np.inf])
    pts = _pts(margin, ref, source_out=[0, 0, 0, 0, 0, 1])
    ok = V.stress_compare(ref + [1e-11, 0, 5e-9, 3.0, 0, 0], pts, ref, abs_gate=1e-10, rel_gate=1e-12, well=WELL)
    assert ok["ok"] and ok["well"].tolist() == [True, True, False, False, False, False]
    assert ok["ill"].tolist() == [False, False, True, False, False, False]
    assert ok["well_max_abs"] == pytest.approx(1e-11) and ok["condition_scaled_max"] > 0
    over = V.stress_compare(ref + [1e-9, 0, 0, 0, 0, 0], pts, ref, abs_gate=1e-10, rel_gate=1e-12, well=WELL)
    assert not over["ok"] and over["over"] == 1
    mask = V.stress_compare(np.where(np.isinf(ref), -50.0, ref), pts, ref, abs_gate=1e-10, rel_gate=1e-12, well=WELL)
    assert not mask["ok"] and mask["inf_mismatch"] == 2 and mask["expected_mismatch"] == 2
    marg = ref.copy()
    marg[3] = -np.inf                       # a marginal point may go either way, but like the reference
    m = V.stress_compare(marg, pts, ref, abs_gate=1e-10, rel_gate=1e-12, well=WELL)
    assert not m["ok"] and m["inf_mismatch"] == 1 and m["expected_mismatch"] == 0
    nan = V.stress_compare(np.where(np.arange(6) == 0, np.nan, ref), pts, ref, abs_gate=1e-10, rel_gate=1e-12,
                           well=WELL)
    assert not nan["ok"] and nan["invalid"] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("rel_gate", [1e-12, 0.0])                 # NumPy, JAX
def test_stress_rules_fail_closed_on_a_non_finite_reference(bad, rel_gate):
    """I-1: a NaN or +inf reference at a well-conditioned point is over every gate (NaN compares
    False both ways; inf <= inf on NumPy), never a pass."""
    margin = [0.5, 0.2, 1e-8]
    ref = np.array([-100.0, -200.0, -300.0])
    pts = _pts(margin, ref)
    bad_ref = ref.copy()
    bad_ref[0] = bad
    out = V.stress_compare(ref, pts, bad_ref, abs_gate=1e-10, rel_gate=rel_gate, well=WELL)
    assert not out["ok"]
    assert V.stress_compare(ref, pts, ref, abs_gate=1e-10, rel_gate=rel_gate, well=WELL)["ok"]


@pytest.mark.parametrize("abs_gate,rel_gate", [(math.nan, 1e-12), (1e-10, math.nan), (math.nan, 0.0)])
def test_a_nan_gate_is_over_at_every_well_conditioned_point(abs_gate, rel_gate):
    """I-1 step 1: ``~(delta <= allowed)``, so a NaN gate (``delta > NaN`` is False) never passes."""
    ref = np.array([-100.0, -200.0, -300.0])
    out = V.stress_compare(ref, _pts([0.5, 0.2, 1e-8], ref), ref, abs_gate=abs_gate, rel_gate=rel_gate, well=WELL)
    assert not out["ok"] and out["over"] == 2


def test_counter_expectations():
    class FourierWhittle:
        pass

    failed = np.array([False, True, False])
    assert V.bare_counters(object(), 3, failed) == {"evaluations": 3, "noise_states": 3, "pd_failed_rows": 1,
                                                     "pd_failed_states": 1}
    assert V.bare_counters(FourierWhittle(), 3, np.zeros(3, bool))["noise_states"] == 0
    res = {"counters": {"evaluations": 3, "noise_states": 2}, "last_failed": failed}
    probs = V.counter_problems(res, {"evaluations": 3, "noise_states": 3}, failed_mask=~failed)
    assert len(probs) == 2 and "noise_states" in probs[0] and "last_failed" in probs[1]


def test_fail_closed_inputs(tmp_path):
    with pytest.raises(ValidationError, match="fourier"):
        V.run(waveform="onfly", goldens=["g3"], cases=["gb_3mo"], models=["band_raw"])
    with pytest.raises(ValidationError, match="gb_3mo and gb_6mo"):
        V.run(goldens=["g4"], cases=["gb_12mo"])
    with pytest.raises(ValidationError, match="empty selection"):
        V.select(["gb_12mo"], ["band_cut"])
    with pytest.raises(ValidationError, match="unknown"):
        V.select(["gb_24mo"], None)
    with pytest.raises(ValidationError, match="approval"):
        V.chain_runs("gb_3mo", "band_raw", "full")
    with pytest.raises(ValidationError, match="NumPy test-only adapter"):
        V.run(backend="jax", templates="gbonfly", cases=["gb_3mo"], models=["band_raw"], goldens=["g3"])
    bad = _golden_copy(tmp_path, {TOL: lambda t: t.update(status="PROPOSED")})
    with pytest.raises(ValidationError, match="not approved"):
        V.approved(bad)


@pytest.mark.parametrize("key", ["well_conditioned_margin", "numpy_cache_abs", "numpy_cache_rel",
                                 "jax_well_conditioned"])
@pytest.mark.parametrize("value", [math.nan, math.inf, "nan", "inf", 0.0, -1e-6, "x"])
def test_approved_refuses_a_g4_rule_that_is_not_finite_and_positive(tmp_path, key, value):
    bad = _golden_copy(tmp_path, {TOL: _g4_rule(key, value)})
    with pytest.raises(ValidationError, match="G4"):
        V.approved(bad)


def test_approved_needs_every_per_target_jax_gate_equal_to_the_pooled_one(tmp_path):
    assert V.approved()["G4"] == {"well": 1e-6, "numpy_abs": 1e-10, "numpy_rel": 1e-12, "jax_well": 8.32e-8}

    def other(t):
        t["approved"]["G4"]["jax_well_conditioned_per_target"]["gb_6mo/exact"] = 1e-7

    def missing(t):
        del t["approved"]["G4"]["jax_well_conditioned_per_target"]["gb_3mo/band_raw"]

    def nan(t):
        t["approved"]["G4"]["jax_well_conditioned_per_target"]["gb_3mo/tri_raw"] = math.nan

    for i, edit in enumerate((other, missing, nan)):
        with pytest.raises(ValidationError, match="per-target"):
            V.approved(_golden_copy(tmp_path / str(i), {TOL: edit}))


@pytest.mark.parametrize("key,value", [("lnl_ref", math.nan), ("lnl_ref", math.inf), ("lnl_ref", "nan"),
                                       ("lnl_ref", "inf")])
def test_stress_points_refuse_a_reference_that_is_nan_or_plus_inf(tmp_path, key, value):
    """``check_reference`` semantics: finite or exactly -inf (the source_out points stay -inf)."""
    bad = _golden_copy(tmp_path, {STRESS: _point("edge_h1", key, value)})
    with pytest.raises(ValidationError, match="NaN or \\+inf"):
        V.stress_points("gb_3mo", "band_raw", "fast", bad)
    cached = "galactic_binary/stress/gb_3mo/tri_exact.fast.json"
    bad = _golden_copy(tmp_path / "c", {cached: _point("anchor", "lnl_cache", value)})
    with pytest.raises(ValidationError, match="lnl_cache"):
        V.stress_points("gb_3mo", "tri_exact", "fast", bad)


@pytest.mark.parametrize("edits,match", [
    ({STRESS: _point("anchor", "lnl_ref", math.nan)}, "NaN or"),
    ({STRESS: _point("anchor", "lnl_ref", math.inf)}, "NaN or"),
    ({STRESS: _shift_anchor, TOL: _g4_rule("numpy_cache_abs", math.nan)}, "G4"),
    ({STRESS: _shift_anchor, TOL: _g4_rule("numpy_cache_rel", math.inf)}, "G4"),
    ({STRESS: _shift_anchor, TOL: _g4_rule("well_conditioned_margin", math.nan)}, "G4"),
], ids=["nan_ref", "inf_ref", "nan_abs_gate", "inf_rel_gate", "nan_margin"])
def test_the_review_probes_fail_closed_on_numpy(tmp_path, capsys, monkeypatch, edits, match):
    """I-1's end-to-end probes (each PASSed at v0.10.0): the CLI exits 1, ``run`` raises."""
    monkeypatch.setenv("XLA_FLAGS", "")                    # main() sets it; restored after the test
    bad = _golden_copy(tmp_path, edits)
    with pytest.raises(ValidationError, match=match):
        V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["g4"], golden=bad)
    assert V.main(["--golden-root", str(bad), "--case", "gb_3mo", "--model", "band_raw", "--golden", "g4"]) == 1
    assert "[validate] cannot validate" in capsys.readouterr().err


def test_the_shifted_reference_alone_fails_and_a_resigned_clean_copy_passes(tmp_path):
    rep = V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["g4"], golden=_golden_copy(tmp_path / "s",
                                                                                         {STRESS: _shift_anchor}))
    assert not rep["ok"] and rep["targets"]["gb_3mo/band_raw"]["g4"]["reference"]["over"] == 1
    rep = V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["g4"], golden=_golden_copy(tmp_path / "c"))
    assert rep["ok"], rep["problems"]


@pytest.mark.parametrize("rel,kind", [
    (STRESS, "g4"), (TOL, "g4"), ("galactic_binary/products.json", "g3"),
    ("galactic_binary/chain_rows/gb_3mo/band_raw_flow_seed3008.fast.json", "g3"),
    ("galactic_binary/chain_rows/gb_3mo/index.json", "g3"), ("galactic_binary/intermediates_gb_3mo.npz", "g6"),
    ("galactic_binary/intermediates_gb_3mo.json", "g6")])
def test_a_golden_whose_sha256_is_not_the_manifests_is_refused(tmp_path, capsys, monkeypatch, rel, kind):
    """At every run (the CLI on Colab has no root-lane MANIFEST test): one appended byte fails."""
    monkeypatch.setenv("XLA_FLAGS", "")
    bad = _golden_copy(tmp_path)
    path = bad / rel
    path.write_bytes(path.read_bytes() + b"\n")       # parses as before: only the sha256 can tell
    with pytest.raises(ValidationError, match="sha256"):
        V.run(cases=["gb_3mo"], models=["band_raw"], goldens=[kind], golden=bad)
    assert V.main(["--golden-root", str(bad), "--case", "gb_3mo", "--model", "band_raw", "--golden", kind]) == 1
    assert "sha256" in capsys.readouterr().err


def test_a_golden_missing_from_the_manifest_or_no_manifest_is_refused(tmp_path):
    bad = _golden_copy(tmp_path)
    man = json.loads((bad / "MANIFEST.json").read_text())
    del man["files"][STRESS]
    (bad / "MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(ValidationError, match="not pinned"):
        V.stress_points("gb_3mo", "band_raw", "fast", bad)
    (bad / "MANIFEST.json").unlink()
    with pytest.raises(ValidationError, match="MANIFEST"):
        V.approved(bad)


def test_one_target_each_kind():
    """3-month band_raw (G3, G5, G6) and tri_exact (G4: reference + MatrixCache counters)."""
    rep = V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["g3", "g5", "g6"])
    assert rep["ok"], rep["problems"]
    t = rep["targets"]["gb_3mo/band_raw"]
    run = t["g3"]["runs"]["band_raw_flow_seed3008"]
    assert len(run["rows"]) == 3 and t["g3"]["max_abs"] <= 1e-8 and t["g3"]["lineage_c_max_abs"] <= 1e-8
    assert run["counters"] == {"noise_states": 3, "pd_failed_states": 0, "pd_failed_rows": 0, "evaluations": 3,
                               "cache_hits": 0}
    assert t["g6"]["bitwise"] is rep["recording_build"] or t["g6"]["ok"]
    assert set(rep["g5"]["gb_3mo"]["items"]) == {
        "prepared_sha256", "describe", "template_store_sum_h2", "template_store_re_zh", "template_window_sum_h2",
        "template_window_re_zh", "prior_box"}
    g4 = V.run(cases=["gb_3mo"], models=["tri_exact"], goldens=["g4"])
    assert g4["ok"], g4["problems"]
    e = g4["targets"]["gb_3mo/tri_exact"]["g4"]
    assert {k: e["cache"]["counters"][k] for k in V.COUNTER_KEYS} == e["cache"]["counters_golden"]
    assert e["reference"]["counters"]["pd_failed_rows"] == 6 and e["reference"]["inf_mismatch"] == 0
    assert e["reference"]["well_max_abs"] <= 1e-10


def test_a_wrong_or_nan_likelihood_fails():
    from wdm_models.registry import build

    def wrong(toy, case, model, **kw):                  # mid_raw's likelihood validated as band_raw
        return build(toy, case, "mid_raw", **kw)

    rep = V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["g3"], builder=wrong)
    assert not rep["ok"] and rep["failed"] == ["gb_3mo/band_raw"]

    class NaNLik:
        def __init__(self, lik):
            self.lik, self.ndim = lik, lik.ndim

        def __call__(self, X):
            return np.full(len(np.atleast_2d(X)), np.nan)

    rep = V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["g3"],
                builder=lambda *a, **k: NaNLik(build(*a, **k)))
    assert not rep["ok"] and any("NaN" in p for p in rep["problems"])


def test_onfly_audit_is_reported_not_gated():
    rep = V.run(cases=["gb_3mo"], models=["band_raw"], goldens=["audit"])
    a = rep["targets"]["gb_3mo/band_raw"]["audit"]
    assert rep["ok"] and a["one_minus_ff"].shape == (3,) and 0 < a["max_one_minus_ff"] < 1e-5
    assert 0 < a["max_abs_dlnl"] < 1.0


def test_cpu_determinism_flag(monkeypatch):
    """Spec section 3.4: appended unless set; an unrelated flag is kept, an explicit choice wins."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")    # restored afterwards
    assert V.cpu_determinism() == "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
    assert V.cpu_determinism() == "--xla_cpu_multi_thread_eigen=true"
    monkeypatch.delenv("XLA_FLAGS")
    assert V.cpu_determinism() == V.CPU_DETERMINISM


def test_cli_report_and_refusals(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "")                    # restored after the test
    out = tmp_path / "gb.json"
    assert V.main(["--case", "gb_3mo", "--model", "whittle_raw", "--golden", "g3", "g6", "--report", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["n_targets"] == 1 and rep["waveform"] == "fourier"
    assert rep["fingerprint"]["threads"]["XLA_FLAGS"] == V.CPU_DETERMINISM          # set by main()
    assert "[validate] PASS" in capsys.readouterr().out
    assert V.main(["--case", "gb_3mo", "--model", "band_raw", "--golden", "g3", "--waveform", "onfly"]) == 1
    with pytest.raises(SystemExit) as err:
        V.main(["--report", str(V.REL / "gb.json")])
    assert err.value.code == 2


@pytest.mark.legacy
@pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
def test_the_gbonfly_adapter_is_bitwise_against_lineage_c():
    rep = V.run(cases=["gb_3mo"], models=["tri_raw", "whittle_cut"], goldens=["g3"], templates="gbonfly")
    assert rep["ok"], rep["problems"]
    for key in ("gb_3mo/tri_raw", "gb_3mo/whittle_cut"):
        for run in rep["targets"][key]["g3"]["runs"].values():
            assert run["lineage_c"]["exact"] is rep["recording_build"]
            if rep["recording_build"]:
                assert run["lineage_c"]["bitwise_rows"] == 3, (key, run["lineage_c"]["abs_delta"])
