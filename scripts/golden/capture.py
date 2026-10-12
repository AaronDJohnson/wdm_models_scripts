"""Capture goldens from the frozen trees and write golden/MANIFEST.json (spec §7.1; Task A10).

    $PY -m scripts.golden.capture --family sinusoid --tier fast    # S0 fast tier, ~10-15 min CPU
    $PY -m scripts.golden.capture --family sinusoid --tier full    # REQUIRES USER APPROVAL (Task A11)
    $PY -m scripts.golden.capture --manifest-only                  # re-hash golden/ into MANIFEST.json

Writes only below golden/. Refuses to start when the frozen trees differ from
golden/frozen_trees.json; fails when a frozen tree or a lineage __pycache__ changed during
the capture. Other sections register their family in FAMILIES.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from scripts.golden import measure_tolerances, sinusoid_goldens
from scripts.golden.goldens_io import golden_files, read_json, write_json
from scripts.golden.lineage import SOURCE_DIRS
from scripts.golden.paths import GOLDEN, ROOT
from scripts.golden.workers._common import runtime_fingerprint
from tools import frozen_tree_hashes as fth

MANIFEST = GOLDEN / "MANIFEST.json"
#: family -> (capture(tier) -> MANIFEST "sources" block, measure() -> tolerance report with "file"/"status")
FAMILIES: dict[str, tuple[Callable[[str], dict], Callable[[], dict]]] = {
    "sinusoid": (sinusoid_goldens.capture_all, measure_tolerances.measure_sinusoid),
}


class FrozenTreeChanged(RuntimeError):
    """A frozen legacy tree (or a lineage __pycache__) differs from what was recorded."""


def guard_state() -> tuple:
    dirs = sorted({d for paths in SOURCE_DIRS.values() for d in paths})
    return fth.scan(ROOT, hash_files=False), fth.pycache_fingerprint(dirs)


def write_manifest(*, family: str | None = None, sources: dict | None = None,
                   tolerances: dict | None = None) -> dict:
    current = read_json(MANIFEST) if MANIFEST.is_file() else {"schema": 1}
    current["schema"] = 1
    current.setdefault("sources", {})          # a manifest first written by Section B's merge
    current.setdefault("tolerances", {})       # has neither key yet
    if family is not None:
        if sources is not None:
            current["sources"][family] = sources
        if tolerances is not None:
            current["tolerances"][family] = tolerances
    current["sources"]["frozen_trees"] = {"summary": fth.SUMMARY, "listing": fth.LISTING}
    current["build"] = runtime_fingerprint()
    current["files"] = golden_files()
    write_json(MANIFEST, current)
    return current


def capture(family: str, tier: str) -> dict:
    problems = fth.verify(ROOT, GOLDEN, full=False)
    if problems:
        raise FrozenTreeChanged("frozen trees differ from golden/frozen_trees.json:\n" + "\n".join(problems[:20]))
    run, measure = FAMILIES[family]
    before = guard_state()
    sources = run(tier)
    if guard_state() != before:
        raise FrozenTreeChanged("a frozen tree or a lineage __pycache__ changed during the capture")
    report = measure()
    return write_manifest(family=family, sources=sources,
                          tolerances={"file": report["file"], "status": report["status"]})


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args(argv)
    if a.manifest_only:
        m = write_manifest()
    elif a.family:
        m = capture(a.family, a.tier)
    else:
        ap.error("give --family or --manifest-only")
    print(f"golden/MANIFEST.json: {len(m['files'])} files; tolerance families {sorted(m['tolerances'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
