"""S5 completion report: the galactic-binary waveform, model layer and data products (spec section
7.6, S5: "two-tier probe; G5 templates; prior boxes; prepared regeneration"), every recorded
comparison with its observed delta and its gate.

    $PY -m scripts.golden.s5_report [--full] [--strict] [--out PATH] [--tests FILE_OR_GLOB ...] [--junit XML]

The S5 tests (sections F-A and F-B of plan part 4, :data:`DEFAULT_TESTS`) run in ONE pytest
subprocess (root config, cwd = the release root) with a junit XML report.  Two junit properties
are read:

* ``s5`` -- ``{"item", "observed", "gate"}`` (the ``s3``/``s4`` convention): F-A's probe, PSDs,
  envelope, Table III, prior boxes, ``snr_full``, ``prepare`` port and template items, F-B's
  segments, foreground fits, prepared regeneration and proposals;
* ``s5_audit`` -- ``{"item", "one_minus_ff", "scale_re", "scale_im"}`` (F-A9's ``onfly`` audit
  against ``fourier``), reported, never gated.

The report lists every recorded item (max observed, smallest gate, pass) against the
inventory (:func:`inventory`, 99 items) and the inventory items left uncovered; the checks the
S5 tests assert in-test (the construction guard, the binary_galactica pin, the policies of
``prepare.py --from-shipped``, ...) count through the test outcomes.  Exit 1 if a recorded item
exceeds its gate (a non-finite delta always does), a test failed or errored, no test ran, every
test was skipped, or nothing was recorded; ``--strict`` also exits 1 on any uncovered item.
``--full`` includes the ``slow`` tests (the 6- and 12-month ``mu_fit``/``calc_Nt`` items and
prepared regeneration); without it those items stay uncovered.  The S5 gate is
``--full --strict``; it needs the pinned PyPI release binary-galactica 0.5.0 (F-A11:
``test_imports.py`` checks the version and that pip installed it from the index) and the frozen
trees (the lineage-C tests; a skipped test records nothing, so their items stay uncovered).  A
named (non-glob) test file that does not exist aborts the run.  The foreground MAP items
(:data:`DEFERRED`) need impulse (S9) and are listed as deferred unless recorded.  Writes nothing
under the release tree: the XML goes to ``$TMPDIR/golden_run``.
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

PROPERTY = "s5"
AUDIT_PROPERTY = "s5_audit"
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
#: The S5 package tests, by name (section F-A, then F-B): S6/S7's later ``test_wm_gb_*`` files
#: are not S5's and stay out of this run.
PACKAGE_TESTS = tuple(f"packages/wdm_models/tests/test_wm_gb_{name}.py" for name in (
    "config", "instrument", "foreground", "envelope", "prepared", "priors", "prepare", "waveform",
    "synthetic", "foreground_fit", "segments", "proposals"))
#: Every S5 test: the package tests, the root ``tests/golden/test_s5_*.py`` (the glob is S5's),
#: F-A9's store-template G6 items and the binary_galactica identity probes; never the report's
#: own test.
DEFAULT_TESTS = (*PACKAGE_TESTS, "tests/golden/test_s5_*.py", "tests/golden/test_s2_g6_gb_waveform.py",
                 "tests/identity/test_imports.py")
EXCLUDE = ("tests/golden/test_s5_report.py",)
SEGMENT_ITEMS = ("window", "store_window", "instrument_segment", "instrument_store", "selection",
                 "injection_segment", "injection_store")
FOREGROUND_ITEMS = ("fit_blocks", "fit_S_model", "fit_S_inst")
PREPARED_ITEMS = ("deterministic_keys", "fits", "snr_full")
PROPOSAL_ITEMS = ("fisher_inject", "fisher_point", "noise_fisher_whittle_raw", "noise_fisher_band_raw",
                  "initial_covariance_whittle_raw", "initial_covariance_band_raw")
#: The impulse MAP of the shipped binned periodograms (``slow``, needs the [sampling] extra: S9).
DEFERRED = tuple(f"G5/foreground/{case}/map" for case in CASES)
FAMILIES = ("probe", "psd", "envelope", "table_iii", "prior", "snr_full", "prepare_port", "template",
            "segments", "foreground", "prepared", "proposals")


def inventory() -> list[str]:
    """Every S5 gate item: section F-A's 53 (2 probe + 3 x (2 PSD + 1 C + 2 Table III + 2 prior
    + 1 snr_full + 4 prepare port + 5 template)) and section F-B's 46 (3 x (7 segment +
    3 foreground + 3 prepared) + the legacy prepare engine + 6 proposal comparisons) = 99."""
    items = [f"G5/probe/gb_3mo/{b}" for b in ("fourier", "onfly")]
    for c in CASES:
        items += [f"G5/psd/{c}/S_inst", f"G5/psd/{c}/S_fg", f"G5/envelope/{c}/C",
                  f"G5/table_iii/{c}/A", f"G5/table_iii/{c}/E", f"G5/prior/{c}/box",
                  f"G5/prior/{c}/inject_glass", f"G5/snr_full/{c}"]
        items += [f"G5/prepare_port/{c}/{k}" for k in ("keys", "mu_fit", "calc_nt", "calc_nt_legacy")]
        items += [f"G5/template/{c}/{w}/{s}" for w in ("window", "store") for s in ("sum_h2", "re_zh")]
        items += [f"G5/template/{c}/store_vs_gbonfly"]
    items += [f"G5/segments/{c}/{i}" for c in CASES for i in SEGMENT_ITEMS]
    items += [f"G5/foreground/{c}/{i}" for c in CASES for i in FOREGROUND_ITEMS]
    items += [f"G5/prepared/{c}/{i}" for c in CASES for i in PREPARED_ITEMS]
    items += ["G5/prepared/gb_3mo/legacy_engine"]
    items += [f"proposals/gb_3mo/{i}" for i in PROPOSAL_ITEMS]
    return items


def _family(item: str) -> str:
    parts = item.split("/")
    return parts[1] if parts[0] == "G5" else parts[0]


def collect(xml_path: Path) -> tuple[list[dict], list[dict], dict[str, int]]:
    """``s5`` records, ``s5_audit`` records and the outcome counts of a junit XML report; a
    skipped test checked nothing, so its records are dropped."""
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
            name, value = prop.get("name"), prop.get("value")
            if name == PROPERTY:
                rec = json.loads(value)
                records.append({"test": test, "item": rec["item"], "observed": float(rec["observed"]),
                                "gate": float(rec["gate"])})
            elif name == AUDIT_PROPERTY:
                audit.append({"test": test, **json.loads(value)})
    return records, audit, counts


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
                         "max_observed": max((by_item[k]["observed"] for k in keys), default=None)}
    problems = [p for p, bad in (
        ("no test ran (empty junit report)", counts["tests"] == 0),
        ("every test was skipped", 0 < counts["tests"] == counts["skipped"]),
        ("no S5 comparison was recorded", not by_item),
        (f"{counts['failures']} test(s) failed", counts["failures"] > 0),
        (f"{counts['errors']} test error(s)", counts["errors"] > 0),
        (f"{len(failed)} item(s) over their gate", bool(failed))) if bad]
    return {"records": dict(sorted(by_item.items())), "onfly_audit": audit, "tests": counts,
            "families": families,
            "inventory": {"total": len(items), "covered": len(items) - len(uncovered), "uncovered": uncovered},
            "deferred": [i for i in DEFERRED if i not in by_item],
            "failed_items": failed, "problems": problems, "ok": not problems}


def run_tests(patterns: list[str], *, full: bool) -> Path:
    # fail closed: a named file that is gone (renamed, typo) would otherwise drop out silently;
    # test_s2_g6_gb_waveform.py and test_imports.py record no s5 item, so --strict misses them
    missing = [p for p in patterns if not any(c in p for c in "*?[") and not (REL / p).is_file()]
    if missing:
        raise SystemExit(f"no test file {missing}")
    files = sorted({str(p.relative_to(REL)) for pat in patterns for p in REL.glob(pat)} - set(EXCLUDE))
    if not files:
        raise SystemExit(f"no test files match {patterns}")
    xml = tmp_root() / f"s5_report_{uuid.uuid4().hex[:12]}.xml"
    cmd = [sys.executable, "-P", "-m", "pytest", "-c", "pytest.ini", "-p", "no:cacheprovider", "-q",
           "-o", "junit_family=xunit1", f"--junitxml={xml}", *(["-m", "not golden_full"] if full else []), *files]
    rc = subprocess.run(cmd, cwd=str(REL), check=False).returncode
    if not xml.is_file():
        raise SystemExit("pytest wrote no junit report")
    if rc not in (0, 1):
        raise SystemExit(f"pytest exited with code {rc} (interrupted, internal or usage error, or no tests)")
    return xml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.golden.s5_report")
    ap.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS),
                    help="test files or globs, relative to the release root")
    ap.add_argument("--full", action="store_true", help="include the slow tests")
    ap.add_argument("--strict", action="store_true", help="fail on uncovered items (the S5 gate)")
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
    print(f"[s5_report] {len(rep['records'])} items recorded ({fams}), "
          f"{len(rep['failed_items'])} over gate; inventory {rep['inventory']['covered']}/{rep['inventory']['total']} "
          f"covered, {len(rep['inventory']['uncovered'])} uncovered, {len(rep['deferred'])} deferred to S9; "
          f"onfly audit {len(rep['onfly_audit'])} record(s); tests {rep['tests']['tests']} "
          f"({rep['tests']['failures']} failed, {rep['tests']['errors']} errors, {rep['tests']['skipped']} skipped)",
          file=sys.stderr)
    if not rep["ok"]:
        print(f"[s5_report] FAIL: {'; '.join(rep['problems'])}", file=sys.stderr)
        return 1
    return 1 if a.strict and rep["inventory"]["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
