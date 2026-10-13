"""D-D9: the S2 completion report (scripts/golden/s2_report.py) -- inventory, deferral
table and junit summary on synthetic input; the real run is the ``slow`` test."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.golden import s2_report as R
from scripts.golden.paths import REL

pytestmark = pytest.mark.golden


def test_inventory_lists_every_g6_item():
    items = R.inventory()
    assert len(items) == len(set(items)) == 313
    for probe in ("sinusoid/manuscript_mid/variances/band_raw", "sinusoid/manuscript_mid/tri_blocks/tri_pad",
                  "sinusoid/manuscript_mid/exact_rows", "sinusoid/manuscript_mid/maps/Wp",
                  "sinusoid/manuscript_mid/maps/mid_cut/u", "sinusoid/manuscript_mid/parts/exact/density",
                  "galactic_binary/gb_3mo/fingerprints/KI", "galactic_binary/gb_6mo/arrays/pad_E",
                  "galactic_binary/gb_12mo/scalars/Mprime_c1",
                  "galactic_binary/gb_12mo/quad_logdet/exact/composition"):
        assert probe in items, probe


def test_deferred_items_are_inventory_items_with_owners():
    later, items = R.deferred(), set(R.inventory())
    assert len(later) == 37 and set(later) <= items
    assert "galactic_binary/gb_12mo/arrays/V_exact_rows" not in later      # asserted in S2 (empty)
    assert not any(i.startswith("sinusoid/") for i in later)              # S3 covers them (E-B8)
    assert later["galactic_binary/gb_6mo/quad_logdet/tri_raw/composition"].startswith("S6")
    assert later["galactic_binary/gb_3mo/arrays/store_template"].startswith("S5")
    assert "galactic_binary/gb_3mo/quad_logdet/exact/composition" not in later   # an S2 preview


def _junit(path, records, failed=(), skipped=()):
    cases = []
    for i, (item, observed, gate) in enumerate(records):
        prop = json.dumps({"item": item, "observed": observed, "gate": gate}).replace('"', "&quot;")
        bad = "<failure message='x'/>" if i in failed else "<skipped message='x'/>" if i in skipped else ""
        cases.append(f'<testcase classname="t" name="c{i}"><properties><property name="g6" '
                     f'value="{prop}"/></properties>{bad}</testcase>')
    path.write_text(f'<testsuites><testsuite name="s">{"".join(cases)}</testsuite></testsuites>')
    return path


def test_summary_marks_covered_deferred_uncovered_and_failures(tmp_path):
    xml = _junit(tmp_path / "a.xml", [("sinusoid/manuscript_mid/parts/band_raw/composition/quad", 0.0, 1e-9),
                                      ("sinusoid/manuscript_mid/parts/band_raw/composition/lnL", 2e-9, 1e-9),
                                      ("sinusoid/manuscript_mid/exact_rows", 1e-13, 1e-12)])
    records, counts, passed = R.collect(xml)
    rep = R.summarise(records, counts, R.inventory(), R.deferred(), passed)
    assert rep["inventory"]["covered"] == 2
    assert rep["failed_items"] == ["sinusoid/manuscript_mid/parts/band_raw/composition/lnL"]
    assert not rep["ok"]
    assert "sinusoid/manuscript_mid/variances/band_raw" in rep["inventory"]["uncovered"]
    assert "sinusoid/manuscript_mid/parts/tri_cut/composition" in rep["inventory"]["uncovered"]
    assert "galactic_binary/gb_6mo/quad_logdet/tri_raw/composition" in rep["inventory"]["deferred"]


def test_asserted_items_need_their_passing_tests(tmp_path):
    """An item asserted in a per-module test body counts only when every mapped test node
    passed; every map entry names inventory items."""
    items = R.inventory()
    for pattern, _ in R.ASSERTED:
        assert any(__import__("re").fullmatch(pattern, i) for i in items), pattern
    assert R.asserted_by("galactic_binary/gb_6mo/arrays/cov_mid_pad_diag") == (
        "test_dom_pad_mask::test_g6_pad_covariance_and_quad_logdet[mid-gb_6mo]",)
    assert R.asserted_by("sinusoid/manuscript_mid/parts/band_raw/density") == ()
    # the 10 picked rows always run; all 959 rows are the ``slow`` node (only --full asserts the item)
    assert R.asserted_by("sinusoid/manuscript_mid/exact_rows") == (
        "test_dom_pushed_rows::test_g6_picked_exact_rows", "test_dom_pushed_rows::test_g6_all_exact_rows")
    # basis_half: domainia's Eq. (45) half-step basis C["Bh"], not the envelope stand-in fixture
    assert R.asserted_by("galactic_binary/gb_6mo/arrays/basis_half") == (
        "test_dom_eq45_separable::test_g6_kernels_half_and_tri_raw[gb_6mo]",)
    assert R.asserted_by("galactic_binary/gb_3mo/arrays/V_exact_rows") == (
        "test_dom_pushed_rows::test_g6_gb_exact_rows[gb_3mo]",)
    assert R.asserted_by("galactic_binary/gb_12mo/arrays/V_exact_rows") == (
        "test_dom_pushed_rows::test_g6_gb_12mo_has_no_exact_rows",)
    node = ('<testcase classname="packages.domainia.tests.test_dom_variance" '
            'name="test_g6_sinusoid_raw_variances_are_bitwise">{}</testcase>')
    for bad, want in (("", True), ("<failure message='x'/>", False)):
        xml = tmp_path / f"n{len(bad)}.xml"
        xml.write_text(f'<testsuites><testsuite name="s">{node.format(bad)}</testsuite></testsuites>')
        rep = R.summarise(*R.collect(xml)[:2], items, R.deferred(), R.collect(xml)[2])
        got = "sinusoid/manuscript_mid/variances/mid_raw" in rep["inventory"]["asserted"]
        assert got is want
        assert ("sinusoid/manuscript_mid/variances/mid_raw" in rep["inventory"]["uncovered"]) is not want
    # a run without --full: the picked exact rows passed, the slow all-rows node did not run
    picked = tmp_path / "picked.xml"
    picked.write_text('<testsuites><testsuite name="s"><testcase classname="packages.domainia.tests.'
                      'test_dom_pushed_rows" name="test_g6_picked_exact_rows"/></testsuite></testsuites>')
    rep = R.summarise(*R.collect(picked)[:2], items, R.deferred(), R.collect(picked)[2])
    assert "sinusoid/manuscript_mid/exact_rows" in rep["inventory"]["uncovered"]


def test_asserted_nodes_are_test_functions():
    """Every node an inventory item maps to names a test function of a domainia or
    golden-test module (a misspelt node would leave its items uncovered only in a real run)."""
    import ast

    defined: dict[str, set[str]] = {}
    for d in ("packages/domainia/tests", "tests/golden"):
        for f in (REL / d).glob("test_*.py"):
            tree = ast.parse(f.read_text())
            defined[f.stem] = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    nodes = {n for i in R.inventory() for n in R.asserted_by(i)}
    assert nodes
    for node in nodes:
        module, name = node.split("::")
        assert name.split("[")[0] in defined.get(module, set()), node


def test_mid_and_tri_maps_share_the_band_arrays():
    """The map entries of ASSERTED rely on it: per edge treatment the golden mid/tri maps are
    the band maps (equal sha256)."""
    from scripts.golden.goldens_io import read_json
    from scripts.golden.paths import GOLDEN

    maps = read_json(GOLDEN / "sinusoid" / "intermediates" / "manuscript_mid_truth.json")["maps"]
    for edge in ("raw", "cut", "exact", "pad"):
        for base in ("mid", "tri"):
            for key in ("y_data", "u"):
                assert maps[f"{base}_{edge}"][key]["sha256"] == maps[f"band_{edge}"][key]["sha256"]
    assert maps["band_exact"]["u"]["sha256"] == maps["band_raw"]["u"]["sha256"]


def test_main_exit_codes(tmp_path):
    ok = _junit(tmp_path / "ok.xml", [("sinusoid/manuscript_mid/exact_rows", 0.0, 1e-12)])
    assert R.main(["--junit", str(ok), "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True
    assert R.main(["--junit", str(ok), "--strict"]) == 1                 # uncovered items remain
    bad = _junit(tmp_path / "bad.xml", [("sinusoid/manuscript_mid/exact_rows", 0.0, 1e-12)], failed=(0,))
    assert R.main(["--junit", str(bad)]) == 1                            # a failed test fails the report
    with pytest.raises(SystemExit, match="outside the release tree"):
        R.main(["--junit", str(ok), "--out", str(REL / "s2_report.json")])


def test_the_report_fails_closed(tmp_path, monkeypatch):
    """An empty report, an all-skipped report, a report with no G6 comparison, a non-finite
    delta and an aborted pytest run never pass (with or without ``--strict``); ``--out``
    inside the release tree is refused before any test runs."""
    empty = _junit(tmp_path / "empty.xml", [])
    assert not R.summarise(*R.collect(empty)[:2], R.inventory(), R.deferred())["ok"]
    assert R.main(["--junit", str(empty)]) == 1                          # nothing ran
    # every test skipped, or nothing G6 checked: not ok, even without --strict.  A skipped
    # test checked nothing, so a record it wrote before skipping does not count.
    skipped = _junit(tmp_path / "skipped.xml", [("sinusoid/manuscript_mid/exact_rows", 0.0, 1e-12)], skipped=(0,))
    records, counts, passed = R.collect(skipped)
    assert records == [] and counts == {"tests": 1, "failures": 0, "errors": 0, "skipped": 1} and not passed
    rep = R.summarise(records, counts, R.inventory(), R.deferred(), passed)
    assert not rep["ok"] and "every test was skipped" in rep["problems"]
    assert R.main(["--junit", str(skipped)]) == 1
    # a G6 test skipped after recording, an unrelated test passed: no G6 comparison
    prop = json.dumps({"item": "sinusoid/manuscript_mid/exact_rows", "observed": 0.0, "gate": 1e-12})
    prop = prop.replace('"', "&quot;")
    g6_skipped = tmp_path / "g6_skipped.xml"
    g6_skipped.write_text('<testsuites><testsuite name="s"><testcase classname="t" name="c0"><properties>'
                          f'<property name="g6" value="{prop}"/></properties>'
                          '<skipped message="x"/></testcase><testcase classname="t" name="c1"/>'
                          '</testsuite></testsuites>')
    rep = R.summarise(*R.collect(g6_skipped)[:2], R.inventory(), R.deferred(), R.collect(g6_skipped)[2])
    assert rep["problems"] == ["no G6 comparison was recorded or asserted"] and not rep["ok"]
    assert R.main(["--junit", str(g6_skipped)]) == 1
    nan = _junit(tmp_path / "nan.xml", [("sinusoid/manuscript_mid/exact_rows", float("nan"), 1e-12)])
    rep = R.summarise(*R.collect(nan)[:2], R.inventory(), R.deferred())
    assert rep["failed_items"] == ["sinusoid/manuscript_mid/exact_rows"] and not rep["ok"]

    def no_run(*a, **k):
        raise AssertionError("ran the tests")
    monkeypatch.setattr(R, "run_tests", no_run)
    with pytest.raises(SystemExit, match="outside the release tree"):
        R.main(["--out", str(REL / "s2_report.json")])
    monkeypatch.undo()

    ok = _junit(tmp_path / "rc.xml", [("sinusoid/manuscript_mid/exact_rows", 0.0, 1e-12)])

    def aborted(cmd, **k):                     # pytest interrupted after writing a partial report
        xml = next(c for c in cmd if c.startswith("--junitxml="))[len("--junitxml="):]
        R.Path(xml).write_text(ok.read_text())
        return subprocess.CompletedProcess(cmd, 2)
    monkeypatch.setattr(R.subprocess, "run", aborted)
    with pytest.raises(SystemExit, match="pytest exited with code 2"):
        R.run_tests(["tests/golden/test_s2_report.py"], full=False)


@pytest.mark.slow
def test_the_report_runs_the_s2_gate_tests(tmp_path):
    out = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s2_report", "--full",
                           "--tests", "tests/golden/test_s2_g6_*.py", "--out", str(out)],
                          cwd=str(REL), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["tests"]["failures"] == 0
    assert len(rep["records"]) >= 132 and rep["inventory"]["covered"] == 86   # + E-B8's 8 x 3
    # only the gate tests ran: the one ASSERTED node among them is D-D6's band_pad map check
    assert set(rep["inventory"]["asserted"]) == {f"sinusoid/manuscript_mid/maps/{b}_pad/y_data" for b in ("mid", "tri")}
