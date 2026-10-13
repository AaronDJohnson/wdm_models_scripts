"""S2 completion report: every G6 comparison with its observed max delta and its gate.

    $PY -m scripts.golden.s2_report [--full] [--strict] [--out PATH] [--tests PATH ...]

The G6 tests run in ONE pytest subprocess (root config, cwd = the release root) with a
junit XML report; each comparison is a junit property ``g6`` whose value is
``{"item": ..., "observed": ..., "gate": ...}`` (``scripts.golden.s2_cases.record``; tests
under ``packages/`` call ``record_property("g6", json.dumps({...}))`` themselves).  The
report then lists, against the inventory of G6 items in ``golden/``: every recorded item
(max observed, smallest gate, pass), the inventory items covered by a record, those
asserted inside a passing per-module test of sections D-A..D-C (:data:`ASSERTED`: those
tests compare the golden values in their bodies and record no property), those deferred
to a later sub-project (owner named), and those uncovered.  Exit 1 if a recorded item
exceeds its gate (a non-finite delta always does), a test failed or no test ran (pytest
itself exiting other than 0 or 1 aborts the report); ``--strict`` also exits 1 on any
uncovered item (the S2 gate).
``--full`` includes the ``slow`` tests (deselected by the root ``addopts``).  Writes
nothing under the release tree: the XML goes to ``$TMPDIR/golden_run``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.golden.goldens_io import read_json
from scripts.golden.paths import GOLDEN, REL, tmp_root

SIN_PREVIEW = ("whittle_raw", "whittle_cut", "band_raw", "mid_cut", "tri_raw", "band_pad", "band_exact")
GB_PREVIEW = {"gb_3mo": ("whittle_raw", "band_raw", "tri_exact", "exact")}
DEFAULT_TESTS = ("tests/golden/test_s2_g6_*.py", "packages/domainia/tests/test_dom_*.py")
_GB = r"galactic_binary/(?P<case>gb_\w+)"
#: Inventory items the per-module G6 tests of sections D-A..D-C assert in their bodies:
#: (item regex, test nodes ``<module>::<name>`` formatted with the regex's named groups).
#: An item counts as asserted when every listed node passed in the junit report.  The
#: mid/tri map entries share the band entries' arrays (equal sha256 in the golden file,
#: checked by tests/golden/test_s2_report.py).
ASSERTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"sinusoid/manuscript_mid/variances/(band|mid)_raw",
     ("test_dom_variance::test_g6_sinusoid_raw_variances_are_bitwise",)),
    (r"sinusoid/manuscript_mid/variances/(band|mid)_pad",
     ("test_dom_variance::test_g6_sinusoid_pad_variances_are_bitwise",)),
    (r"sinusoid/manuscript_mid/tri_blocks/(?P<model>tri_\w+)", ("test_dom_eq45::test_g6_tri_blocks[{model}]",)),
    (r"sinusoid/manuscript_mid/exact_rows", ("test_dom_pushed_rows::test_g6_picked_exact_rows",)),
    (r"sinusoid/manuscript_mid/maps/Wp", ("test_dom_maps_wdm::test_g6_sinusoid_padding_impulses_Wp_are_bitwise",
                                          "test_dom_zero_padded::test_g6_padding_impulses")),
    (r"sinusoid/manuscript_mid/maps/(band|mid|tri)_(raw|cut|exact)/y_data",
     ("test_dom_maps_wdm::test_g6_sinusoid_cells_are_bitwise",)),
    (r"sinusoid/manuscript_mid/maps/(band|mid|tri)_pad/y_data",
     ("test_s2_g6_sinusoid::test_mapped_data_and_nuisance_directions[band_pad]",)),
    (r"sinusoid/manuscript_mid/maps/(band|mid|tri)_(raw|cut|exact|pad)/u", ("test_dom_nuisance::test_g6_pushed_nuisance",)),
    (_GB + r"/fingerprints/(V_(raw|cut|exact)_[CUbn]|V_cut_cells)",
     ("test_dom_maps_cells::test_g6_cell_matrices_raw_cut_exact[{case}]",)),
    (_GB + r"/(fingerprints/V_pad_[CUbn]|arrays/V_pad_cells)",
     ("test_dom_maps_cells::test_g6_pad_cell_matrix_and_its_cells[{case}]",)),
    (_GB + r"/arrays/V_raw_cells", ("test_dom_maps_cells::test_g6_cells_of_the_store_data[{case}]",)),
    (_GB + r"/(fingerprints/(KI|KG|half|cov_tri_raw_c[01])|arrays/(KI_diag|KG_diag|cov_tri_raw_diag))",
     ("test_dom_eq45_separable::test_g6_kernels_half_and_tri_raw[{case}]",)),
    (_GB + r"/arrays/basis_half", ("test_dom_eq45_separable::test_envelope_stand_in_is_the_golden_basis[{case}]",)),
    (_GB + r"/(fingerprints/Mprime_c[01]|scalars/Mprime_c[01]|arrays/ghat_c[01])",
     ("test_dom_hermitian::test_g6_mprime_q0_logdet_and_ghat[{case}]",)),
    (_GB + r"/(arrays/V_exact_rows|arrays/exact_rows_c[01]_sel|fingerprints/exact_rows_c[01])",
     ("test_dom_pushed_rows::test_g6_gb_exact_rows[{case}]",)),
    (_GB + r"/(fingerprints/pad_AE|arrays/(pad_E|pad_Kpos|pad_AE_row0))",
     ("test_dom_pad_mask::test_g6_pad_map_and_mask[{case}]",)),
    (_GB + r"/(fingerprints/cov_(?P<base>band|mid|tri)_pad_c[01])",
     ("test_dom_pad_mask::test_g6_pad_covariance_and_quad_logdet[{base}-{case}]",)),
    (_GB + r"/arrays/cov_(?P<base>band|mid|tri)_pad_diag",
     ("test_dom_pad_mask::test_g6_pad_covariance_and_quad_logdet[{base}-{case}]",)),
    (_GB + r"/arrays/bandpsd_(band|mid)_s[IG]", ("test_dom_variance::test_g6_gb_band_means[{case}]",)),
    (_GB + r"/arrays/var_(band|mid)_raw", ("test_dom_variance::test_g6_gb_variances_at_the_anchor[{case}]",)),
)


def asserted_by(item: str) -> tuple[str, ...]:
    """The per-module test nodes that assert ``item`` in their bodies (``()``: none)."""
    for pattern, nodes in ASSERTED:
        m = re.fullmatch(pattern, item)
        if m:
            return tuple(n.format(**m.groupdict()) for n in nodes)
    return ()


def inventory() -> list[str]:
    """Every G6 item in golden/ (spec section 7.2, G6), as report item names."""
    items: list[str] = []
    s = read_json(GOLDEN / "sinusoid" / "intermediates" / "manuscript_mid_truth.json")
    root = "sinusoid/manuscript_mid"
    items += [f"{root}/variances/{k}" for k in sorted(s["variances"])]
    items += [f"{root}/tri_blocks/{k}" for k in sorted(s["tri_blocks"])]
    items.append(f"{root}/exact_rows")
    for k in sorted(s["maps"]):
        items += [f"{root}/maps/Wp"] if k == "Wp" else [f"{root}/maps/{k}/y_data", f"{root}/maps/{k}/u"]
    for m in sorted(s["parts"]):
        items += [f"{root}/parts/{m}/density", f"{root}/parts/{m}/composition"]
    for case in ("gb_3mo", "gb_6mo", "gb_12mo"):
        g = read_json(GOLDEN / "galactic_binary" / f"intermediates_{case}.json")
        root = f"galactic_binary/{case}"
        items += [f"{root}/fingerprints/{k}" for k in sorted(g["fingerprints"])]
        items += [f"{root}/arrays/{k}" for k in g["arrays"] if k != "quad_logdet"]
        items += [f"{root}/scalars/{k}" for k in sorted(g["scalars"]) if k.startswith("Mprime_")]
        for m in g["quad_logdet_models"]:
            items += [f"{root}/quad_logdet/{m}/density", f"{root}/quad_logdet/{m}/composition"]
    return items


def deferred() -> dict[str, str]:
    """Inventory items S2 does not own, with the sub-project that will cover them."""
    out: dict[str, str] = {}
    for item in inventory():
        parts = item.split("/")
        if item.endswith("/composition") and item.startswith("sinusoid/") and parts[3] not in SIN_PREVIEW:
            out[item] = "S3: the full sinusoid registry (spec section 7.6)"
        elif item.endswith("/composition") and item.startswith("galactic_binary/") \
                and parts[3] not in GB_PREVIEW.get(parts[1], ()):
            out[item] = "S6: the full GB registry (spec section 7.6)"
        elif item.endswith("/arrays/store_template"):
            out[item] = "S5: the waveform (binary_galactica) and its template goldens (G5)"
        elif item == "galactic_binary/gb_12mo/arrays/V_exact_rows":
            out[item] = "S6: the periodic 12-month record has no exact-edge rows (an empty array; the registry's rule)"
    return out


def collect(xml_path: Path) -> tuple[list[dict], dict[str, int], set[str]]:
    """``g6`` records, the test outcome counts and the passed test nodes
    (``<module>::<name>``) of a junit XML report."""
    tree = ET.parse(xml_path)
    records: list[dict] = []
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    passed: set[str] = set()
    for case in tree.iter("testcase"):
        counts["tests"] += 1
        bad = False
        for tag in ("failure", "error", "skipped"):
            if case.find(tag) is not None:
                bad = True
                counts["failures" if tag == "failure" else "errors" if tag == "error" else "skipped"] += 1
        if not bad:
            passed.add(f"{(case.get('classname') or '').rsplit('.', 1)[-1]}::{case.get('name')}")
        test = f"{case.get('classname')}::{case.get('name')}"
        for prop in case.iter("property"):
            if prop.get("name") == "g6":
                rec = json.loads(prop.get("value"))
                records.append({"test": test, "item": rec["item"], "observed": float(rec["observed"]),
                                "gate": float(rec["gate"])})
    return records, counts, passed


def summarise(records: list[dict], counts: dict[str, int], items: list[str],
              later: dict[str, str], passed: set[str] | frozenset[str] = frozenset()) -> dict:
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
    covered = [i for i in items if i in by_item or any(k.startswith(i + "/") for k in by_item)]
    asserted = {i: list(asserted_by(i)) for i in items
                if i not in covered and asserted_by(i) and all(n in passed for n in asserted_by(i))}
    uncovered = [i for i in items if i not in covered and i not in asserted and i not in later]
    failed = sorted(k for k, e in by_item.items() if not e["ok"])
    return {"records": dict(sorted(by_item.items())), "tests": counts,
            "inventory": {"total": len(items), "covered": len(covered), "asserted": asserted,
                          "deferred": {i: later[i] for i in items if i in later and i not in covered},
                          "uncovered": uncovered},
            "failed_items": failed,
            "ok": (not failed and counts["tests"] > 0          # an empty report ran nothing
                   and counts["failures"] == 0 and counts["errors"] == 0)}


def run_tests(patterns: list[str], *, full: bool) -> Path:
    files = sorted({str(p.relative_to(REL)) for pat in patterns for p in REL.glob(pat)})
    if not files:
        raise SystemExit(f"no test files match {patterns}")
    xml = tmp_root() / f"s2_report_{uuid.uuid4().hex[:12]}.xml"
    cmd = [sys.executable, "-P", "-m", "pytest", "-c", "pytest.ini", "-p", "no:cacheprovider", "-q",
           "-o", "junit_family=xunit1",            # xunit2 (the default) warns on record_property
           f"--junitxml={xml}", *(["-m", "not golden_full"] if full else []), *files]
    rc = subprocess.run(cmd, cwd=str(REL), check=False).returncode
    if not xml.is_file():
        raise SystemExit("pytest wrote no junit report")
    if rc not in (0, 1):                       # 1: test failures, summarised from the report
        raise SystemExit(f"pytest exited with code {rc} (interrupted, internal or usage error, or no tests)")
    return xml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.golden.s2_report")
    ap.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS), help="test globs, relative to the release root")
    ap.add_argument("--full", action="store_true", help="include the slow tests")
    ap.add_argument("--strict", action="store_true", help="fail on uncovered G6 items (the S2 gate)")
    ap.add_argument("--junit", type=Path, help="summarise an existing junit XML instead of running pytest")
    ap.add_argument("--out", type=Path, help="also write the JSON report here (outside the release tree)")
    a = ap.parse_args(argv)
    out = a.out.resolve() if a.out is not None else None
    if out is not None and out.is_relative_to(REL.resolve()):     # refuse before the test run
        raise SystemExit("--out must lie outside the release tree")
    xml = a.junit if a.junit is not None else run_tests(a.tests, full=a.full)
    records, counts, passed = collect(xml)
    rep = summarise(records, counts, inventory(), deferred(), passed)
    text = json.dumps(rep, indent=1, sort_keys=False)
    if out is not None:
        out.write_text(text + "\n")
    print(text)
    print(f"[s2_report] {len(rep['records'])} items recorded, {len(rep['failed_items'])} over gate; "
          f"inventory {rep['inventory']['covered']}/{rep['inventory']['total']} covered, "
          f"{len(rep['inventory']['asserted'])} asserted in-test, "
          f"{len(rep['inventory']['deferred'])} deferred, {len(rep['inventory']['uncovered'])} uncovered; "
          f"tests {counts['tests']} ({counts['failures']} failed, {counts['errors']} errors, "
          f"{counts['skipped']} skipped)", file=sys.stderr)
    if not rep["ok"]:
        return 1
    return 1 if a.strict and rep["inventory"]["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
