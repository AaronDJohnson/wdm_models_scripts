"""S6 completion report: every G3/G4/G5/G6 comparison of the galactic binary on NumPy with its
observed max delta and its gate, plus the onfly audit (spec section 7.6, S6).

    $PY -m scripts.golden.s6_report [--full] [--strict] [--out PATH] [--tests GLOB ...] [--junit XML]

The S6 tests run in ONE pytest subprocess (root config, cwd = the release root) with a junit
XML report; each comparison is a junit property ``s6`` (``scripts.golden.s6_cases.record``) and
each audit value an ``s6_audit`` property.  The report lists every recorded item (max observed,
smallest gate, pass) against the inventory (:func:`scripts.golden.s6_cases.inventory`: the
fast-tier G3 rows of the 58 run directories with binary_galactica ``fourier`` and with the
gbonfly adapter, the G4 stress points of the 30 three- and six-month targets with their masks
and counters, the G5 products of the 3 cases and the 38 G6 anchor compositions), the inventory
items left uncovered, and the audit (``1 - FF``, ``lnL_onfly - lnL_fourier``: reported, never
gated).  Exit 1 if a recorded item exceeds its gate (a non-finite delta always does), a test
failed or errored, no test ran, every test was skipped, nothing was recorded, or (``--full``)
the S2 gate (``s2_report --full --strict``) failed -- with or without ``--strict``;
``--strict`` also exits 1 on any uncovered item (a skipped test, e.g. the gbonfly adapter
without the frozen trees, leaves its items uncovered).  ``--full`` includes the ``slow`` tests
(the 12-month tri_raw and pads, the 6-month tri_pad) and the S2 gate; the S6 gate is
``--full --strict``.  Writes nothing under the release tree: the XML goes to
``$TMPDIR/golden_run``.  The full tier (32 rows per run directory) is approval-gated and not
part of this report (``tests/golden/test_s6_full_tier.py``, marker ``golden_full``).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.golden.paths import REL, tmp_root
from scripts.golden.s6_cases import AUDIT_PROPERTY, PROPERTY, inventory

#: The S6 gate tests: the G3/G4/G5/G6 records and the audit, plus the model layer (F-C1..F-C4),
#: the gbonfly adapter (F-C5) and validate.py's own rules (F-C6).  Listed file by file: the
#: ``test_wm_gb_*`` glob would also take S5's tests (F-A), which are S5's gate.
DEFAULT_TESTS = ("tests/golden/test_s6_*.py", "tests/golden/test_s2_g6_gb_registry.py",
                 "tests/golden/test_gb_validate.py", "tests/golden/test_gb_templates.py",
                 "packages/wdm_models/tests/test_wm_gb_options.py", "packages/wdm_models/tests/test_wm_gb_models.py",
                 "packages/wdm_models/tests/test_wm_gb_registry.py",
                 "packages/domainia/tests/test_dom_whittle_apply_data.py")
FAMILIES = ("g3", "g3_gbonfly", "g4", "g5", "g6")


def collect(xml_path: Path) -> tuple[list[dict], list[dict], dict[str, int]]:
    """``s6`` records, ``s6_audit`` records and the outcome counts of a junit XML report; a
    skipped test checked nothing, so a record it wrote before skipping is dropped."""
    tree = ET.parse(xml_path)
    records: list[dict] = []
    audit: list[dict] = []
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for case in tree.iter("testcase"):
        counts["tests"] += 1
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                counts[key] += 1
        if case.find("skipped") is not None:
            continue
        test = f"{case.get('classname')}::{case.get('name')}"
        for prop in case.iter("property"):
            if prop.get("name") == PROPERTY:
                rec = json.loads(prop.get("value"))
                records.append({"test": test, "item": rec["item"], "observed": float(rec["observed"]),
                                "gate": float(rec["gate"])})
            elif prop.get("name") == AUDIT_PROPERTY:
                audit.append({"test": test, **json.loads(prop.get("value"))})
    return records, audit, counts


def _family(item: str) -> str | None:
    parts = item.split("/")
    if len(parts) >= 4 and parts[2] == "g5":
        return "g5"
    return parts[3] if len(parts) >= 4 and parts[3] in FAMILIES else None


def summarise(records: list[dict], audit: list[dict], counts: dict[str, int], items: list[str]) -> dict:
    by_item: dict[str, dict] = {}
    for r in records:
        e = by_item.setdefault(r["item"], {"observed": 0.0, "gate": r["gate"], "tests": []})
        # fail closed: a non-finite delta (max() would drop a NaN) is over every gate
        e["observed"] = max(e["observed"], r["observed"] if math.isfinite(r["observed"]) else math.inf)
        e["gate"] = min(e["gate"], r["gate"])
        e["tests"].append(r["test"])
    for e in by_item.values():
        e["ok"] = e["observed"] <= e["gate"]
        e["tests"] = sorted(set(e["tests"]))
    uncovered = [i for i in items if i not in by_item]
    failed = sorted(k for k, e in by_item.items() if not e["ok"])
    families = {}
    for fam in FAMILIES:
        keys = [k for k in by_item if _family(k) == fam]
        families[fam] = {"items": len(keys),
                         "max_observed": max((by_item[k]["observed"] for k in keys), default=None),
                         "zero": sum(by_item[k]["observed"] == 0.0 for k in keys)}
    bad_audit = [a["item"] for a in audit
                 if not (math.isfinite(a.get("max_one_minus_ff", math.nan))
                         and math.isfinite(a.get("max_abs_dlnl", math.nan)))]
    problems = [p for p, bad in (
        ("no test ran (empty junit report)", counts["tests"] == 0),
        ("every test was skipped", 0 < counts["tests"] == counts["skipped"]),
        ("no S6 comparison was recorded", not by_item),
        (f"{counts['failures']} test(s) failed", counts["failures"] > 0),
        (f"{counts['errors']} test error(s)", counts["errors"] > 0),
        (f"{len(failed)} item(s) over their gate", bool(failed)),
        (f"non-finite audit values at {bad_audit}", bool(bad_audit))) if bad]
    return {"records": dict(sorted(by_item.items())), "tests": counts, "families": families,
            "audit": {"targets": len(audit),
                      "max_one_minus_ff": max((a["max_one_minus_ff"] for a in audit), default=None),
                      "max_abs_dlnl": max((a["max_abs_dlnl"] for a in audit), default=None),
                      "per_target": {a["item"]: {k: a[k] for k in ("max_one_minus_ff", "max_abs_dlnl", "rows")}
                                     for a in audit}},
            "inventory": {"total": len(items), "covered": len(items) - len(uncovered), "uncovered": uncovered},
            "failed_items": failed, "problems": problems, "ok": not problems}


def run_tests(patterns: list[str], *, full: bool) -> Path:
    files = sorted({str(p.relative_to(REL)) for pat in patterns for p in REL.glob(pat)})
    if not files:
        raise SystemExit(f"no test files match {patterns}")
    xml = tmp_root() / f"s6_report_{uuid.uuid4().hex[:12]}.xml"
    cmd = [sys.executable, "-P", "-m", "pytest", "-c", "pytest.ini", "-p", "no:cacheprovider", "-q",
           "-o", "junit_family=xunit1", f"--junitxml={xml}", *(["-m", "not golden_full"] if full else []), *files]
    rc = subprocess.run(cmd, cwd=str(REL), check=False).returncode
    if not xml.is_file():
        raise SystemExit("pytest wrote no junit report")
    if rc not in (0, 1):
        raise SystemExit(f"pytest exited with code {rc} (interrupted, internal or usage error, or no tests)")
    return xml


def run_s2_gate() -> int:
    """``s2_report --full --strict`` (S6 closed the S2 gate's GB compositions): its exit status."""
    return subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s2_report", "--full", "--strict"],
                          cwd=str(REL), check=False, stdout=subprocess.DEVNULL).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.golden.s6_report")
    ap.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS), help="test globs, relative to the release root")
    ap.add_argument("--full", action="store_true", help="include the slow tests and the S2 gate")
    ap.add_argument("--strict", action="store_true", help="fail on uncovered items (the S6 gate)")
    ap.add_argument("--junit", type=Path, help="summarise an existing junit XML instead of running pytest")
    ap.add_argument("--out", type=Path, help="also write the JSON report here (outside the release tree)")
    a = ap.parse_args(argv)
    out = a.out.resolve() if a.out is not None else None
    if out is not None and out.is_relative_to(REL.resolve()):     # refuse before the test run
        raise SystemExit("--out must lie outside the release tree")
    xml = a.junit if a.junit is not None else run_tests(a.tests, full=a.full)
    rep = summarise(*collect(xml), inventory())
    if a.full and a.junit is None:
        rep["s2_gate_exit"] = run_s2_gate()
        if rep["s2_gate_exit"] != 0:
            rep["problems"].append(f"s2_report --full --strict exited {rep['s2_gate_exit']}")
            rep["ok"] = False
    text = json.dumps(rep, indent=1)
    if out is not None:
        out.write_text(text + "\n")
    print(text)
    fams = ", ".join(f"{k} {v['items']}" for k, v in rep["families"].items())
    au = rep["audit"]
    print(f"[s6_report] {len(rep['records'])} items recorded ({fams}), {len(rep['failed_items'])} over gate; "
          f"inventory {rep['inventory']['covered']}/{rep['inventory']['total']} covered, "
          f"{len(rep['inventory']['uncovered'])} uncovered; audit {au['targets']} targets "
          f"(max 1-FF {au['max_one_minus_ff']}, max |dlnL| {au['max_abs_dlnl']}, reported); tests "
          f"{rep['tests']['tests']} ({rep['tests']['failures']} failed, {rep['tests']['errors']} errors, "
          f"{rep['tests']['skipped']} skipped)", file=sys.stderr)
    if not rep["ok"]:
        print(f"[s6_report] FAIL: {'; '.join(rep['problems'])}", file=sys.stderr)
        return 1
    return 1 if a.strict and rep["inventory"]["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
