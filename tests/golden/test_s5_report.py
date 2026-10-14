"""F-B8: the S5 completion report (scripts/golden/s5_report.py) -- inventory, the two junit
properties (``s5``, ``s5_audit``) and the fail-closed summary on synthetic junit input; the real
run (the S5 gate, ``--full --strict``) is the ``slow`` test."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts.golden import s5_report as R
from scripts.golden.paths import REL

pytestmark = pytest.mark.golden


def test_inventory_lists_every_s5_item():
    items = R.inventory()
    assert len(items) == len(set(items)) == 53 + 46 == 99           # sections F-A and F-B
    assert len([i for i in items if R._family(i) in R.FAMILIES[:8]]) == 53
    for probe in ("G5/probe/gb_3mo/onfly", "G5/template/gb_12mo/store_vs_gbonfly",
                  "G5/template/gb_6mo/window/re_zh", "G5/prepare_port/gb_6mo/calc_nt_legacy",
                  "G5/table_iii/gb_3mo/E", "G5/segments/gb_12mo/injection_store",
                  "G5/foreground/gb_6mo/fit_S_inst", "G5/prepared/gb_12mo/fits",
                  "G5/prepared/gb_3mo/legacy_engine", "proposals/gb_3mo/initial_covariance_band_raw"):
        assert probe in items, probe
    assert {R._family(i) for i in items} == set(R.FAMILIES)
    assert not set(R.DEFERRED) & set(items)                    # the impulse MAP waits for S9


def test_the_default_tests_are_the_s5_tests():
    files = {str(p.relative_to(REL)) for pat in R.DEFAULT_TESTS for p in REL.glob(pat)}
    for f in (*R.PACKAGE_TESTS, "tests/golden/test_s5_gb_prepare_script.py", "tests/golden/test_s5_calc_nt.py",
              "tests/golden/test_s2_g6_gb_waveform.py", "tests/identity/test_imports.py",
              "tests/golden/test_s5_report.py"):
        assert f in files, f
    # only the named package tests: S6/S7's GB package tests are not S5's
    assert {f for f in files if f.startswith("packages/")} == set(R.PACKAGE_TESTS)
    assert R.EXCLUDE == ("tests/golden/test_s5_report.py",)


def test_the_report_imports_without_the_release_packages():
    """s5_report runs as ``$PY -m scripts.golden.s5_report`` with only the release root on
    PYTHONPATH (env/activate.sh): it must not import wdm, domainia or wdm_models."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(REL), PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys, scripts.golden.s5_report as R; R.inventory(); "
            "print(sorted(m for m in ('wdm', 'domainia', 'wdm_models') if m in sys.modules))")
    p = subprocess.run([sys.executable, "-P", "-c", code], cwd=str(REL), env=env, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == "[]", p.stderr[-2000:]


def _junit(path, records, failed=(), skipped=(), extra=""):
    cases = []
    for i, (item, observed, gate) in enumerate(records):
        prop = json.dumps({"item": item, "observed": observed, "gate": gate}).replace('"', "&quot;")
        bad = "<failure message='x'/>" if i in failed else "<skipped message='x'/>" if i in skipped else ""
        cases.append(f'<testcase classname="t" name="c{i}"><properties><property name="s5" '
                     f'value="{prop}"/></properties>{bad}</testcase>')
    path.write_text(f'<testsuites><testsuite name="s">{"".join(cases)}{extra}</testsuite></testsuites>')
    return path


AUDIT = {"item": "audit/onfly/gb_3mo/store", "one_minus_ff": 1.85e-06, "scale_re": 1.0000114, "scale_im": 2e-07}
AUDIT_CASE = ('<testcase classname="w" name="t0"><properties><property name="s5_audit" value="'
              + json.dumps(AUDIT).replace('"', "&quot;") + '"/></properties></testcase>')


def test_summary_reads_both_properties(tmp_path):
    xml = _junit(tmp_path / "a.xml", [("G5/segments/gb_3mo/window", 0.0, 0.0),
                                      ("G5/template/gb_3mo/store_vs_gbonfly", 9.96e-17, 1e-12),
                                      ("G5/template/gb_6mo/store_vs_gbonfly", 2e-12, 1e-12),
                                      ("G5/prepared/gb_3mo/fits", 2e-6, 1e-6),
                                      ("G5/foreground/gb_3mo/map", 0.0, 0.0)], extra=AUDIT_CASE)
    records, audit, counts = R.collect(xml)
    rep = R.summarise(records, audit, counts, R.inventory())
    assert rep["records"]["G5/template/gb_3mo/store_vs_gbonfly"] == {
        "observed": 9.96e-17, "gate": 1e-12, "tests": ["t::c1"], "ok": True}
    assert rep["failed_items"] == ["G5/prepared/gb_3mo/fits", "G5/template/gb_6mo/store_vs_gbonfly"]
    assert rep["onfly_audit"] == [{"test": "w::t0", **AUDIT}]                 # reported, never gated
    assert rep["inventory"]["covered"] == 4 and rep["deferred"] == ["G5/foreground/gb_6mo/map",
                                                                     "G5/foreground/gb_12mo/map"]
    assert rep["families"]["template"]["items"] == 2 and rep["families"]["probe"]["items"] == 0
    assert rep["tests"]["tests"] == 6 and not rep["ok"]


def test_the_report_fails_closed(tmp_path, monkeypatch):
    empty = _junit(tmp_path / "empty.xml", [])
    assert R.main(["--junit", str(empty)]) == 1                               # nothing ran
    skipped = _junit(tmp_path / "skipped.xml", [("G5/segments/gb_3mo/window", 0.0, 0.0)], skipped=(0,))
    records, audit, counts = R.collect(skipped)
    assert records == [] and counts["skipped"] == 1
    assert "every test was skipped" in R.summarise(records, audit, counts, R.inventory())["problems"]
    plain = tmp_path / "plain.xml"                                           # a passing test, no record
    plain.write_text('<testsuites><testsuite name="s"><testcase classname="t" name="c"/></testsuite></testsuites>')
    assert R.summarise(*R.collect(plain), R.inventory())["problems"] == ["no S5 comparison was recorded"]
    nan = _junit(tmp_path / "nan.xml", [("G5/segments/gb_3mo/window", float("nan"), 0.0)])
    assert R.summarise(*R.collect(nan), R.inventory())["failed_items"] == ["G5/segments/gb_3mo/window"]
    bad = _junit(tmp_path / "bad.xml", [("G5/segments/gb_3mo/window", 0.0, 0.0)], failed=(0,))
    assert R.main(["--junit", str(bad)]) == 1
    ok = _junit(tmp_path / "ok.xml", [("G5/segments/gb_3mo/window", 0.0, 0.0)])
    assert R.main(["--junit", str(ok), "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True
    assert R.main(["--junit", str(ok), "--strict"]) == 1                      # 98 items uncovered
    with pytest.raises(SystemExit, match="outside the release tree"):
        R.main(["--junit", str(ok), "--out", str(REL / "s5_report.json")])
    with pytest.raises(SystemExit, match="no test file"):                   # a named file that is gone
        R.run_tests([*R.DEFAULT_TESTS, "tests/identity/test_gone.py"], full=False)

    seen = {}

    def aborted(cmd, **k):                     # pytest interrupted after writing a partial report
        seen["cmd"] = cmd
        xml = next(c for c in cmd if c.startswith("--junitxml="))[len("--junitxml="):]
        R.Path(xml).write_text(ok.read_text())
        return subprocess.CompletedProcess(cmd, 2)
    monkeypatch.setattr(R.subprocess, "run", aborted)
    with pytest.raises(SystemExit, match="pytest exited with code 2"):
        R.run_tests(list(R.DEFAULT_TESTS), full=False)
    assert "tests/golden/test_s5_report.py" not in seen["cmd"]               # never itself


@pytest.mark.slow
def test_the_report_is_the_s5_gate(tmp_path):
    """The S5 gate: every inventory item recorded and inside its gate, every S5 test passing
    (the pinned PyPI release binary-galactica 0.5.0 installed, the frozen trees present)."""
    out = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s5_report", "--full", "--strict",
                           "--out", str(out)], cwd=str(REL), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["tests"]["failures"] == rep["tests"]["errors"] == 0
    assert rep["inventory"]["covered"] == rep["inventory"]["total"] == 99
    assert rep["records"]["G5/prepared/gb_3mo/legacy_engine"]["observed"] == 0.0
    assert sorted(a["item"] for a in rep["onfly_audit"]) == [f"audit/onfly/{c}/store" for c in
                                                             ("gb_12mo", "gb_3mo", "gb_6mo")]
