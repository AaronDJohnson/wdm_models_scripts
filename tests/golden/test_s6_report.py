"""F-C9: the S6 completion report (scripts/golden/s6_report.py) -- inventory, families, audit and
junit summary on synthetic input; the real run is the S6 gate command (``--full --strict``)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.golden import s6_cases as S6
from scripts.golden import s6_report as R
from scripts.golden.paths import REL

pytestmark = pytest.mark.golden


def test_inventory_is_every_fast_tier_item():
    items = S6.inventory()
    assert len(items) == len(set(items)) == 1843
    assert (len(S6.g3_items("g3")), len(S6.g4_items()), len(S6.g5_items()), len(S6.g6_items())) == (424, 784, 21, 190)
    for probe in ("galactic_binary/gb_3mo/exact/g3/exact_flow_seed3009/row0",
                  "galactic_binary/gb_12mo/tri_pad/g3_gbonfly/tri_pad_flow_seed3007/row0/lineage_C",
                  "galactic_binary/gb_3mo/tri_exact/g4/cache/counters", "galactic_binary/gb_6mo/exact/g4/anchor",
                  "galactic_binary/gb_6mo/band_pad/g4/noise_hi/mask", "galactic_binary/gb_12mo/g5/template_store_re_zh",
                  "galactic_binary/gb_12mo/exact/g6/c1_logdet"):
        assert probe in items, probe
    assert "galactic_binary/gb_6mo/band_pad/g4/noise_hi" not in items       # marginal: mask only
    assert not any("/gb_12mo/" in i and "/g4/" in i for i in items)          # no 12-month stress points


def test_targets_are_the_registrys():
    from wdm_models import registry

    assert S6.TARGETS == registry.TARGETS["galactic_binary"] and S6.MODELS == registry.MODELS
    assert S6.CACHED == __import__("wdm_models.options", fromlist=["GB_CACHED"]).GB_CACHED


def _junit(path, records, audit=(), failed=(), skipped=()):
    cases = []
    for i, (item, observed, gate) in enumerate(records):
        prop = json.dumps({"item": item, "observed": observed, "gate": gate}).replace('"', "&quot;")
        bad = "<failure message='x'/>" if i in failed else "<skipped message='x'/>" if i in skipped else ""
        cases.append(f'<testcase classname="t" name="c{i}"><properties><property name="s6" '
                     f'value="{prop}"/></properties>{bad}</testcase>')
    for j, a in enumerate(audit):
        prop = json.dumps(a).replace('"', "&quot;")
        cases.append(f'<testcase classname="t" name="a{j}"><properties><property name="s6_audit" '
                     f'value="{prop}"/></properties></testcase>')
    path.write_text(f'<testsuites><testsuite name="s">{"".join(cases)}</testsuite></testsuites>')
    return path


def test_summary_marks_covered_failed_uncovered_and_the_audit(tmp_path):
    good = {"item": "galactic_binary/gb_3mo/band_raw/audit", "max_one_minus_ff": 1e-6, "max_abs_dlnl": 0.01,
            "rows": 3}
    xml = _junit(tmp_path / "a.xml", [("galactic_binary/gb_3mo/band_raw/g3/band_raw_flow_seed3008/row0", 5e-11, 1e-8),
                                      ("galactic_binary/gb_3mo/band_raw/g6/lnL", 2e-8, 1e-8),
                                      ("galactic_binary/gb_3mo/g5/describe", 0.0, 0.0)], audit=[good])
    rep = R.summarise(*R.collect(xml), S6.inventory())
    assert rep["inventory"]["covered"] == 3 and rep["failed_items"] == ["galactic_binary/gb_3mo/band_raw/g6/lnL"]
    assert not rep["ok"] and rep["families"]["g3"]["items"] == 1 and rep["families"]["g5"]["zero"] == 1
    assert rep["audit"]["targets"] == 1 and rep["audit"]["max_abs_dlnl"] == 0.01
    nan = _junit(tmp_path / "b.xml", [("galactic_binary/gb_3mo/g5/describe", float("nan"), 0.0)],
                 audit=[{**good, "max_abs_dlnl": float("nan")}])
    rep = R.summarise(*R.collect(nan), S6.inventory())
    assert not rep["ok"] and rep["failed_items"] and any("audit" in p for p in rep["problems"])
    skip = _junit(tmp_path / "c.xml", [("galactic_binary/gb_3mo/g5/describe", 0.0, 0.0)], skipped=(0,))
    rep = R.summarise(*R.collect(skip), S6.inventory())
    assert not rep["ok"] and "every test was skipped" in rep["problems"]
    assert "galactic_binary/gb_3mo/g5/describe" in rep["inventory"]["uncovered"]


def test_out_inside_the_release_is_refused():
    p = subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s6_report", "--junit", "x.xml",
                        "--out", str(REL / "s6.json")], cwd=str(REL), capture_output=True, text=True)
    assert p.returncode != 0 and "outside the release tree" in p.stderr
