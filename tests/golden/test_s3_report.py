"""E-B6: the S3 completion report (scripts/golden/s3_report.py) -- inventory and the
fail-closed summary on synthetic junit input; the real run is the ``slow`` test."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts.golden import s3_report as R
from scripts.golden.paths import REL
from scripts.golden.s3_cases import G5_ITEMS, recording_build

pytestmark = pytest.mark.golden


def test_inventory_lists_every_s3_item():
    items = R.inventory()
    assert len(items) == len(set(items)) == 45 * (4 + 2 + 3 + 2) + len(G5_ITEMS) == 515
    for probe in ("sinusoid/manuscript_mid/band_exact/g1/point4", "sinusoid/manuscript_low/whittle_raw/g1/point25",
                  "sinusoid/manuscript_high/exact/g1/counters", "sinusoid/manuscript_mid/band_exact/g2/row1619",
                  "sinusoid/manuscript_mid/tri_pad/g2/mask", "sinusoid/g5/c_N", "sinusoid/manuscript_low/g5/w_d"):
        assert probe in items, probe
    assert not any(i.endswith("/lineage_A") for i in items)        # extra records, not inventory


def test_s3_cases_agree_with_validate_and_the_registry():
    from scripts.golden import s3_cases as C
    from scripts.sinusoid import validate as V
    from wdm_models import registry

    assert C.CASES == V.CASES == tuple(registry.TARGETS["sinusoid"])
    assert C.MODELS == V.MODELS == registry.MODELS == tuple(registry.TARGETS["sinusoid"]["manuscript_mid"])


def test_the_report_imports_without_the_release_packages():
    """s3_report runs as ``$PY -m scripts.golden.s3_report`` with only the release root on
    PYTHONPATH (env/activate.sh): it must not import wdm, domainia or wdm_models."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(REL), PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys, scripts.golden.s3_report as R; R.inventory(); "
            "print(sorted(m for m in ('wdm', 'domainia', 'wdm_models') if m in sys.modules))")
    p = subprocess.run([sys.executable, "-P", "-c", code], cwd=str(REL), env=env, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == "[]", p.stderr[-2000:]


def _junit(path, records, failed=(), skipped=()):
    cases = []
    for i, (item, observed, gate) in enumerate(records):
        prop = json.dumps({"item": item, "observed": observed, "gate": gate}).replace('"', "&quot;")
        bad = "<failure message='x'/>" if i in failed else "<skipped message='x'/>" if i in skipped else ""
        cases.append(f'<testcase classname="t" name="c{i}"><properties><property name="s3" '
                     f'value="{prop}"/></properties>{bad}</testcase>')
    path.write_text(f'<testsuites><testsuite name="s">{"".join(cases)}</testsuite></testsuites>')
    return path


def test_summary_marks_covered_uncovered_and_failures(tmp_path):
    xml = _junit(tmp_path / "a.xml", [("sinusoid/manuscript_mid/band_raw/g1/point0", 0.0, 0.0),
                                      ("sinusoid/manuscript_mid/band_raw/g2/row0", 2e-5, 1e-5),
                                      ("sinusoid/g5/c_N", 0.0, 0.0)])
    rep = R.summarise(*R.collect(xml), R.inventory())
    assert rep["inventory"]["covered"] == 3 and len(rep["inventory"]["uncovered"]) == 512
    assert rep["failed_items"] == ["sinusoid/manuscript_mid/band_raw/g2/row0"] and not rep["ok"]
    assert rep["families"]["g1"] == {"items": 1, "max_observed": 0.0, "zero": 1}


def test_the_report_fails_closed(tmp_path, monkeypatch):
    empty = _junit(tmp_path / "empty.xml", [])
    assert R.main(["--junit", str(empty)]) == 1                               # nothing ran
    skipped = _junit(tmp_path / "skipped.xml", [("sinusoid/g5/c_N", 0.0, 0.0)], skipped=(0,))
    records, counts = R.collect(skipped)
    assert records == [] and counts["skipped"] == 1
    assert "every test was skipped" in R.summarise(records, counts, R.inventory())["problems"]
    assert R.main(["--junit", str(skipped)]) == 1
    plain = tmp_path / "plain.xml"                                           # a passing test, no record
    plain.write_text('<testsuites><testsuite name="s"><testcase classname="t" name="c"/></testsuite></testsuites>')
    assert R.summarise(*R.collect(plain), R.inventory())["problems"] == ["no S3 comparison was recorded"]
    nan = _junit(tmp_path / "nan.xml", [("sinusoid/g5/c_N", float("nan"), 0.0)])
    rep = R.summarise(*R.collect(nan), R.inventory())
    assert rep["failed_items"] == ["sinusoid/g5/c_N"] and not rep["ok"]
    bad = _junit(tmp_path / "bad.xml", [("sinusoid/g5/c_N", 0.0, 0.0)], failed=(0,))
    assert R.main(["--junit", str(bad)]) == 1
    ok = _junit(tmp_path / "ok.xml", [("sinusoid/g5/c_N", 0.0, 0.0)])
    assert R.main(["--junit", str(ok), "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True
    assert R.main(["--junit", str(ok), "--strict"]) == 1                      # 514 items uncovered
    with pytest.raises(SystemExit, match="outside the release tree"):
        R.main(["--junit", str(ok), "--out", str(REL / "s3_report.json")])

    def aborted(cmd, **k):                     # pytest interrupted after writing a partial report
        xml = next(c for c in cmd if c.startswith("--junitxml="))[len("--junitxml="):]
        R.Path(xml).write_text(ok.read_text())
        return subprocess.CompletedProcess(cmd, 2)
    monkeypatch.setattr(R.subprocess, "run", aborted)
    with pytest.raises(SystemExit, match="pytest exited with code 2"):
        R.run_tests(["tests/golden/test_s3_report.py"], full=False)


@pytest.mark.slow
def test_the_report_is_the_s3_gate(tmp_path):
    """The S3 gate: every G1/G2/G5 item recorded, inside its gate; on the recording build
    every G1 point exactly 0.0."""
    out = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s3_report", "--full", "--strict",
                           "--out", str(out)], cwd=str(REL), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["tests"]["failures"] == rep["tests"]["errors"] == 0
    assert rep["inventory"]["covered"] == rep["inventory"]["total"] == 515
    if recording_build():           # 270 inventory items + 180 lineage-A echoes + 3 D1 points
        assert rep["families"]["g1"]["zero"] == rep["families"]["g1"]["items"] == 45 * 6 + 45 * 4 + 3
