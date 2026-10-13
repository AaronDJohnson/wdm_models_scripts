"""S3 completion report: every G1/G2/G5 comparison of the sinusoid NumPy gate with its
observed max delta and its gate (spec section 7.6, S3: "G1 fast tier at section 7.3").

    $PY -m scripts.golden.s3_report [--full] [--strict] [--out PATH] [--tests GLOB ...] [--junit XML]

The S3 gate tests run in ONE pytest subprocess (root config, cwd = the release root) with a
junit XML report; each comparison is a junit property ``s3``
(``scripts.golden.s3_cases.record``).  The report lists every recorded item (max observed,
smallest gate, pass) against the inventory (:func:`inventory`: the fast-tier G1 points,
masks and counters of the 45 targets, the G2 rows, masks and counters, and the G5 items),
and the inventory items left uncovered.  Exit 1 if a recorded item exceeds its gate (a
non-finite delta always does), a test failed or errored, no test ran, every test was
skipped, or nothing was recorded -- with or without ``--strict`` (pytest itself exiting other
than 0 or 1 aborts the report); ``--strict`` also exits 1 on any uncovered item.  ``--full``
includes the ``slow`` tests (tri_*, *_pad and *_exact targets: ~6 min); without it those
items stay uncovered.  The S3 gate is ``--full --strict``.  Writes nothing under the release
tree: the XML goes to ``$TMPDIR/golden_run``.
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
from scripts.golden.s3_cases import G5_ITEMS, PROPERTY, g1g2_items

#: The S3 gate tests: the G1/G2/G5 records, plus the model layer against its dense and WT
#: oracles (E-A9's ports of the WT likelihood-matrix tests and E-B5's WT-class transcriptions).
DEFAULT_TESTS = ("tests/golden/test_s3_g*.py", "packages/wdm_models/tests/test_wm_sinusoid_models.py",
                 "packages/wdm_models/tests/test_wm_model_level.py")


def inventory() -> list[str]:
    """Every S3 gate item: 45 x (4 G1 points + mask + counters + 3 G2 rows + mask + counters)
    + the G5 items."""
    return g1g2_items("fast") + list(G5_ITEMS)


def collect(xml_path: Path) -> tuple[list[dict], dict[str, int]]:
    """``s3`` records and the outcome counts of a junit XML report; a skipped test checked
    nothing, so a record it wrote before skipping is dropped."""
    tree = ET.parse(xml_path)
    records: list[dict] = []
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
    return records, counts


def summarise(records: list[dict], counts: dict[str, int], items: list[str]) -> dict:
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
    for fam in ("g1", "g2", "g5"):
        keys = [k for k in by_item if f"/{fam}/" in k]
        families[fam] = {"items": len(keys),
                         "max_observed": max((by_item[k]["observed"] for k in keys), default=None),
                         "zero": sum(by_item[k]["observed"] == 0.0 for k in keys)}
    problems = [p for p, bad in (
        ("no test ran (empty junit report)", counts["tests"] == 0),
        ("every test was skipped", 0 < counts["tests"] == counts["skipped"]),
        ("no S3 comparison was recorded", not by_item),
        (f"{counts['failures']} test(s) failed", counts["failures"] > 0),
        (f"{counts['errors']} test error(s)", counts["errors"] > 0),
        (f"{len(failed)} item(s) over their gate", bool(failed))) if bad]
    return {"records": dict(sorted(by_item.items())), "tests": counts, "families": families,
            "inventory": {"total": len(items), "covered": len(items) - len(uncovered), "uncovered": uncovered},
            "failed_items": failed, "problems": problems, "ok": not problems}


def run_tests(patterns: list[str], *, full: bool) -> Path:
    files = sorted({str(p.relative_to(REL)) for pat in patterns for p in REL.glob(pat)})
    if not files:
        raise SystemExit(f"no test files match {patterns}")
    xml = tmp_root() / f"s3_report_{uuid.uuid4().hex[:12]}.xml"
    cmd = [sys.executable, "-P", "-m", "pytest", "-c", "pytest.ini", "-p", "no:cacheprovider", "-q",
           "-o", "junit_family=xunit1", f"--junitxml={xml}", *(["-m", "not golden_full"] if full else []), *files]
    rc = subprocess.run(cmd, cwd=str(REL), check=False).returncode
    if not xml.is_file():
        raise SystemExit("pytest wrote no junit report")
    if rc not in (0, 1):
        raise SystemExit(f"pytest exited with code {rc} (interrupted, internal or usage error, or no tests)")
    return xml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.golden.s3_report")
    ap.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS), help="test globs, relative to the release root")
    ap.add_argument("--full", action="store_true", help="include the slow tests")
    ap.add_argument("--strict", action="store_true", help="fail on uncovered items (the S3 gate)")
    ap.add_argument("--junit", type=Path, help="summarise an existing junit XML instead of running pytest")
    ap.add_argument("--out", type=Path, help="also write the JSON report here (outside the release tree)")
    a = ap.parse_args(argv)
    out = a.out.resolve() if a.out is not None else None
    if out is not None and out.is_relative_to(REL.resolve()):     # refuse before the test run
        raise SystemExit("--out must lie outside the release tree")
    xml = a.junit if a.junit is not None else run_tests(a.tests, full=a.full)
    rep = summarise(*collect(xml), inventory())
    text = json.dumps(rep, indent=1)
    if out is not None:
        out.write_text(text + "\n")
    print(text)
    fams = ", ".join(f"{k} {v['items']}" for k, v in rep["families"].items())
    print(f"[s3_report] {len(rep['records'])} items recorded ({fams}), "
          f"{len(rep['failed_items'])} over gate; inventory {rep['inventory']['covered']}/{rep['inventory']['total']} "
          f"covered, {len(rep['inventory']['uncovered'])} uncovered; tests {rep['tests']['tests']} "
          f"({rep['tests']['failures']} failed, {rep['tests']['errors']} errors, {rep['tests']['skipped']} skipped)",
          file=sys.stderr)
    if not rep["ok"]:
        print(f"[s3_report] FAIL: {'; '.join(rep['problems'])}", file=sys.stderr)
        return 1
    return 1 if a.strict and rep["inventory"]["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
