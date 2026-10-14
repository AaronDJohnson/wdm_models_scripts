"""F-D11: the S7 report summarises fail-closed (synthetic junit and validate reports; the report
itself is run by the S7 gate command, not from here -- it runs these tests)."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.golden.goldens_io import read_json
from scripts.golden.paths import GOLDEN, REL
from scripts.golden.s7_report import (CPU_FLAG, REQUIRED, STRESS_CASES, TESTS, collect, main, required_problems,
                                      summarise_validate, targets)

pytestmark = pytest.mark.golden
GB = GOLDEN / "galactic_binary"
_XML = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.golden.test_s7_gb_jax" name="test_g3_fast_tier_on_cpu_jax[gb_3mo-tri_raw]"/>
<testcase classname="tests.golden.test_s7_build_jax" name="test_b"><failure message="boom"/></testcase>
<testcase classname="tests.golden.test_s7_prior_sanitize" name="test_c"><skipped message="no jax"/></testcase>
</testsuite></testsuites>"""


def _report() -> dict:
    """A ``scripts/galactic_binary/validate.py --backend jax``-shaped report whose values ARE lineage
    C's own JAX values (``lnlike_jax``, ``lnl_jax``): every oracle delta is exactly 0."""
    out = {}
    for case, models in targets().items():
        for model in models:
            runs = {}
            for run in read_json(GB / "chain_rows" / case / "index.json")["targets"][model]:
                rows = read_json(GB / "chain_rows" / case / f"{run}.fast.json")["rows"]
                runs[run] = {"value": [r["lnlike_jax"] for r in rows], "ms_per_eval": 1.0, "repeat_max_abs": 0.0}
            g4 = {"applicable": False, "ok": True, "problems": []}
            if case in STRESS_CASES:
                pts = read_json(GB / "stress" / case / f"{model}.fast.json")["points"]
                g4 = {"reference": {"value": [p["lnl_jax"] for p in pts], "condition_scaled_max": 0.0}, "ok": True,
                      "problems": [], "ms_per_eval": 1.0, "repeat_max_abs": 0.0}
            out[f"{case}/{model}"] = {"case": case, "model": model, "ok": True, "problems": [],
                                      "g3": {"runs": runs, "ok": True, "problems": []}, "g4": g4}
    return {"schema": 1, "family": "galactic_binary", "backend": "jax", "tier": "fast", "goldens": ["g3", "g4"],
            "waveform": "fourier", "ok": True, "failed": [], "problems": [], "n_targets": len(out), "targets": out,
            "run_meta": {"backend": "jax", "device_kind": "cpu", "x64": True, "deterministic_ops": False,
                         "XLA_FLAGS": CPU_FLAG, "cpu_single_thread_eigen": True}}


def test_tests_cover_every_s7_file_and_not_the_report_itself():
    assert "tests/golden/test_s7_report.py" not in TESTS and all((REL / t).is_file() for t in TESTS)
    assert sorted(p.name for p in (REL / "tests" / "golden").glob("test_s7_*.py")) == sorted(
        [t.rsplit("/", 1)[-1] for t in TESTS if t.startswith("tests/golden/")] + ["test_s7_report.py"])
    assert sorted(p.name for p in (REL / "tests" / "parity").glob("test_s7_*.py")) == ["test_s7_gb_jax_states.py"]
    assert sum(len(m) for m in targets().values()) == 38


def test_collect_counts_and_lists_passed_nodes(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text(_XML)
    counts, passed = collect(xml)
    assert counts == {"tests": 3, "failures": 1, "errors": 0, "skipped": 1}
    assert passed == {"test_s7_gb_jax::test_g3_fast_tier_on_cpu_jax[gb_3mo-tri_raw]"}


def test_required_problems_name_every_short_pattern():
    assert len(required_problems(set(), full=True)) == len([r for r in REQUIRED if r[2] > 0])
    assert len(required_problems(set(), full=False)) == len([r for r in REQUIRED if r[1] > 0])
    one = {"test_s7_gb_jax::test_g3_fast_tier_on_cpu_jax[gb_3mo-tri_raw]"}
    assert any("1 passed node(s), need 30" in p for p in required_problems(one, full=False))


def test_a_report_of_lineage_c_jax_values_is_clean():
    models, problems = summarise_validate(_report())
    assert problems == [] and len(models) == 15
    assert summarise_validate(_report(), expect_device="cpu")[1] == []
    for m, s in models.items():
        assert s["g3_oracle_max_abs"] == 0.0, m
        assert s["g3_max_abs"] <= 1e-8 and s["g4_well_max_abs"] <= 8.32e-8, m
    assert models["tri_raw"]["ms_per_eval"] == {"gb_3mo": 1.0, "gb_6mo": 1.0, "gb_12mo": 1.0}
    # spec section 3.4: a CPU-JAX run needs single-threaded Eigen (validate.main sets it), both in
    # XLA_FLAGS and as run_meta's cpu_single_thread_eigen
    for flags, single in ((None, None), ("", False), ("--xla_cpu_multi_thread_eigen=true", False),
                          (CPU_FLAG, False), (CPU_FLAG, None), (None, True)):
        rep = _report()
        rep["run_meta"].update(XLA_FLAGS=flags, cpu_single_thread_eigen=single)
        for expect in (None, "cpu"):
            assert any(f"a CPU run without XLA_FLAGS={CPU_FLAG}" in p
                       for p in summarise_validate(rep, expect_device=expect)[1]), (flags, single, expect)


def _set(path, value):
    def apply(r):
        d = r
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return apply


def _shift(tag, kind, delta):
    """Move the first value of ``tag``'s first G3 run (``"g3"``) or its G4 anchor (``"g4"``)."""
    def apply(r):
        t = r["targets"][tag]
        values = t["g3"]["runs"][sorted(t["g3"]["runs"])[0]]["value"] if kind == "g3" else t["g4"]["reference"]["value"]
        values[0] += delta
    return apply


@pytest.mark.parametrize("break_it, expect", [
    (lambda r: r["targets"].pop("gb_12mo/tri_pad"), "validate targets: 37"),
    (_set(("targets", "gb_3mo/exact", "g3", "runs", "exact_flow_seed3009", "value", 0), "nan"), "finite"),
    (_set(("targets", "gb_6mo/band_raw", "g4", "reference", "value", 0), "-inf"), "-inf mask"),
    (_set(("targets", "gb_6mo/band_exact", "g4", "reference", "value", 1), -2755.0), "-inf mask"),
    (_set(("targets", "gb_3mo/tri_raw", "g4", "reference", "value", 0), "inf"), "NaN or +inf"),
    (lambda r: r["targets"]["gb_3mo/tri_cut"]["g3"]["runs"].pop("tri_cut_flow_seed3010"), "are not the golden's"),
    (_set(("targets", "gb_12mo/exact", "g4"), {"reference": {"value": []}}), "G4 entry at 12 months"),
    (_set(("ok",), False), "ok is not true"),
    (_set(("backend",), "numpy"), "backend 'numpy'"),
    (_set(("waveform",), "onfly"), "need 'fourier'"),
    (_set(("failed",), ["gb_3mo/exact"]), "failed targets"),
    (_set(("problems",), ["gb_3mo/exact: g3: over the G3 gate"]), "problem(s)"),
    (_set(("targets", "gb_6mo/exact", "g3", "runs", "exact_flow_seed3007", "ms_per_eval"), 0.0), "ms_per_eval"),
    (lambda r: r.pop("run_meta"), "no run_meta"),
    (_set(("run_meta", "x64"), False), "x64"),
    (_set(("run_meta", "XLA_FLAGS"), None), CPU_FLAG),
    (_set(("run_meta", "device_kind"), "gpu"), "deterministic_ops"),
    (lambda r: r["targets"]["gb_3mo/whittle_raw"].pop("g3"), "no g3 runs"),
    (_shift("gb_6mo/tri_raw", "g3", 2e-8), "over the G3 gate"),
    (_shift("gb_3mo/exact", "g4", 1e-7), "over the G4 gate"),
])
def test_a_validate_summary_fails_closed(break_it, expect):
    rep = copy.deepcopy(_report())
    break_it(rep)
    _, problems = summarise_validate(json.loads(json.dumps(rep)))
    assert any(expect in p for p in problems), problems


def test_a_gpu_report_with_the_deterministic_flag_is_clean(tmp_path):
    rep = _report()
    rep["run_meta"].update(device_kind="gpu", deterministic_ops=True, XLA_FLAGS="--xla_gpu_deterministic_ops=true",
                           cpu_single_thread_eigen=False)
    assert summarise_validate(rep)[1] == []
    assert summarise_validate(rep, expect_device="gpu")[1] == []
    # the device each path expects: the user's GPU sign-off (--validate) refuses any other
    # device -- a CPU fallback included -- and the local gate refuses a GPU report
    assert any("device_kind 'gpu'; need 'cpu'" in p for p in summarise_validate(rep, expect_device="cpu")[1])
    cpu = _report()
    assert any("device_kind 'cpu'; need 'gpu'" in p for p in summarise_validate(cpu, expect_device="gpu")[1])
    no_flag = copy.deepcopy(rep)
    no_flag["run_meta"]["deterministic_ops"] = False
    assert any("deterministic_ops" in p for p in summarise_validate(no_flag, expect_device="gpu")[1])
    bare = _report()                                  # a CPU-JAX report without any flag
    bare["run_meta"] = {"backend": "jax", "device_kind": "cpu", "x64": True, "deterministic_ops": False,
                        "XLA_FLAGS": None}
    probs = summarise_validate(bare, expect_device="gpu")[1]
    assert any("need 'gpu'" in p for p in probs) and any("a CPU run without" in p for p in probs)
    with pytest.raises(ValueError, match="expect_device"):
        summarise_validate(rep, expect_device="tpu")
    # main(): --validate is the GPU sign-off; the problems name the device
    xml = tmp_path / "j.xml"
    xml.write_text(_XML)
    for report, refused in ((cpu, True), (rep, False)):
        path, out = tmp_path / "validate.json", tmp_path / "s7.json"
        path.write_text(json.dumps(report))
        assert main(["--validate", str(path), "--junit", str(xml), "--out", str(out)]) == 1   # the junit fails
        got = json.loads(out.read_text())
        assert got["expect_device"] == "gpu" and got["ok"] is False
        assert any("need 'gpu'" in p for p in got["problems"]) is refused, got["problems"]
        assert any("1 test(s) failed" in p for p in got["problems"]), got["problems"]


def test_out_inside_the_release_tree_is_refused():
    with pytest.raises(SystemExit, match="outside the release tree"):
        main(["--out", str(REL / "s7.json")])
    assert not (REL / "s7.json").exists()
