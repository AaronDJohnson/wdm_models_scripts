"""E-D10: the S4 report summarises fail-closed (synthetic junit and validate reports; the
report itself is run by the S4 gate command, not from here -- it runs these tests)."""

from __future__ import annotations

import copy
import json
import math

import pytest

from scripts.golden.goldens_io import read_json
from scripts.golden.paths import GOLDEN, REL
from scripts.golden.s4_report import (CASES, EC_TESTS, ED_TESTS, MODELS, REQUIRED, TESTS, collect, main,
                                      required_problems, summarise_validate)

pytestmark = pytest.mark.golden
_XML = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.golden.test_s4_sinusoid_jax" name="test_g1_and_g2_fast_tier_on_cpu_jax[a]"/>
<testcase classname="tests.golden.test_s4_prior_clamp" name="test_b"><failure message="boom"/></testcase>
<testcase classname="tests.golden.test_s4_prior_clamp" name="test_c"><skipped message="no jax"/></testcase>
</testsuite></testsuites>"""


def _report_from(lineage: str, backend: str) -> dict:
    """A ``scripts/sinusoid/validate.py``-shaped report whose values ARE a lineage's own
    (lineage B for JAX, A for NumPy): every oracle delta is exactly 0."""
    eval_dir = GOLDEN / "sinusoid" / "legacy_eval"
    refs = read_json(eval_dir / f"{lineage}_references_fast.json")["points"]
    rows = read_json(eval_dir / f"{lineage}_chain_rows_fast.json")["points"]
    targets = {}
    for tag, p in refs.items():
        case, model = tag.split("__")
        g1 = {k: p[k] for k in ("index", "near", "reference", "value", "abs_delta")}
        g2 = {k: rows[tag][k] for k in ("reference", "value", "abs_delta")}
        targets[f"{case}/{model}"] = {"case": case, "model": model, "ok": True, "problems": [],
                                      "g1": {**g1, "ms_per_eval": 1.0}, "g2": {**g2, "ms_per_eval": 1.0}}
    return {"schema": 1, "family": "sinusoid", "backend": backend, "tier": "fast", "ok": True,
            "failed_targets": [], "problems": [], "n_targets": len(targets), "targets": targets,
            "run_meta": {"backend": backend, "device_kind": "cpu", "x64": True, "deterministic_ops": False}}


def test_tests_cover_every_s4_file_and_not_the_report_itself():
    assert "tests/golden/test_s4_report.py" not in TESTS          # no recursion
    assert TESTS == ED_TESTS + EC_TESTS and all((REL / t).is_file() for t in TESTS)
    assert sorted(p.name for p in (REL / "tests" / "golden").glob("test_s4_*.py")) == sorted(
        [t.rsplit("/", 1)[-1] for t in ED_TESTS if t.startswith("tests/golden/")] + ["test_s4_report.py"])
    assert sorted(p.name for p in (REL / "tests" / "parity").glob("test_s4_*.py")) == sorted(
        t.rsplit("/", 1)[-1] for t in EC_TESTS if t.startswith("tests/parity/"))
    assert len(CASES) * len(MODELS) == 45


def test_collect_counts_and_lists_passed_nodes(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text(_XML)
    counts, passed = collect(xml)
    assert counts == {"tests": 3, "failures": 1, "errors": 0, "skipped": 1}
    assert passed == {"test_s4_sinusoid_jax::test_g1_and_g2_fast_tier_on_cpu_jax[a]"}


def test_required_problems_name_every_short_pattern():
    assert len(required_problems(set(), full=True)) == len([r for r in REQUIRED if r[2] > 0])
    assert len(required_problems(set(), full=False)) == len([r for r in REQUIRED if r[1] > 0])
    one = {"test_s4_sinusoid_jax::test_g1_and_g2_fast_tier_on_cpu_jax[a]"}
    assert any("1 passed node(s), need 12" in p for p in required_problems(one, full=False))


def test_a_report_of_the_wg_values_is_clean():
    models, problems = summarise_validate(_report_from("B", "jax"))
    assert problems == [] and sorted(models) == sorted(MODELS)
    for m, s in models.items():
        assert s["oracle_max_abs_near"] == 0.0 and s["oracle_max_rel_stress"] == 0.0, m
        assert s["inf_points"] == (3 if m in ("band_exact", "mid_exact", "tri_exact") else 0), m
        assert s["ms_per_eval"] == 1.0


def test_a_numpy_report_of_the_wt_values_is_clean():
    models, problems = summarise_validate(_report_from("A", "numpy"), backend="numpy")
    assert problems == [] and all(s["oracle_max_abs_near"] == 0.0 for s in models.values())


def _set(path, value):
    def apply(r):
        d = r
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return apply


@pytest.mark.parametrize("break_it, expect", [
    (lambda r: r["targets"].pop("manuscript_low/tri_pad"), "validate targets: 44"),
    (_set(("targets", "manuscript_mid/exact", "g1", "value", 0), "nan"), "NaN"),
    (_set(("targets", "manuscript_mid/band_raw", "g1", "value", 1), "-inf"), "-inf G1 point"),
    (_set(("targets", "manuscript_mid/band_exact", "g1", "value", 3), 1.0), "-inf G1 point"),
    (_set(("ok",), False), "ok is not true"),
    (_set(("backend",), "numpy"), "backend 'numpy'"),
    (_set(("failed_targets",), ["manuscript_mid/exact"]), "failed targets"),
    (_set(("problems",), ["manuscript_mid/exact: 1 point(s) over the gate"]), "problem(s)"),
    (_set(("targets", "manuscript_mid/exact", "g2", "ms_per_eval"), 0.0), "ms_per_eval"),
    (lambda r: r.pop("run_meta"), "no run_meta"),
    (_set(("run_meta", "x64"), False), "x64"),
    (_set(("run_meta", "device_kind"), "gpu"), "deterministic_ops"),
    (lambda r: r["targets"]["manuscript_mid/tri_raw"].pop("g2"), "no g1/g2 values"),
])
def test_a_validate_summary_fails_closed(break_it, expect):
    rep = copy.deepcopy(_report_from("B", "jax"))
    break_it(rep)
    _, problems = summarise_validate(json.loads(json.dumps(rep)))
    assert any(expect in p for p in problems), problems


def test_a_gpu_report_with_the_deterministic_flag_is_clean():
    rep = _report_from("B", "jax")
    rep["run_meta"].update(device_kind="gpu", deterministic_ops=True)
    assert summarise_validate(rep)[1] == []


def test_out_inside_the_release_tree_is_refused():
    with pytest.raises(SystemExit, match="outside the release tree"):
        main(["--out", str(REL / "s4.json")])
    assert not (REL / "s4.json").exists()
