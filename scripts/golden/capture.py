"""Capture goldens from the frozen trees and write golden/MANIFEST.json (spec §7.1; Task A10).

    $PY -m scripts.golden.capture --family sinusoid --tier fast    # S0 fast tier, ~10-15 min CPU
    $PY -m scripts.golden.capture --family sinusoid --tier full    # REQUIRES USER APPROVAL (Task A11)
    $PY -m scripts.golden.capture --manifest-only                  # add new golden/ files to MANIFEST.json

Writes only below golden/. Refuses to start when the frozen trees differ from
golden/frozen_trees.json; fails when a frozen tree or a lineage __pycache__ changed during
the capture. Other sections register their family in FAMILIES.

Never re-approves a golden silently: ``--manifest-only`` only ADDS files that are not in the
manifest yet, and a ``--family F`` capture re-pins only files under ``golden/F/``. Any other
entry whose bytes changed or whose file disappeared is refused (``SystemExit``, nothing
written; a family capture checks before it runs), and ``build`` is kept, unless ``--replace``.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from scripts.golden import manifest_merge, measure_tolerances, sinusoid_goldens
from scripts.golden.goldens_io import golden_files, read_json, write_json
from scripts.golden.lineage import SOURCE_DIRS
from scripts.golden.paths import GOLDEN, ROOT
from scripts.golden.workers._common import runtime_fingerprint
from tools import frozen_tree_hashes as fth

#: family -> (capture(tier) -> MANIFEST "sources" block, measure() -> tolerance report with "file"/"status")
FAMILIES: dict[str, tuple[Callable[[str], dict], Callable[[], dict]]] = {
    "sinusoid": (sinusoid_goldens.capture_all, measure_tolerances.measure_sinusoid),
}


class FrozenTreeChanged(RuntimeError):
    """A frozen legacy tree (or a lineage __pycache__) differs from what was recorded."""


def guard_state() -> tuple:
    dirs = sorted({d for paths in SOURCE_DIRS.values() for d in paths})
    return fth.scan(ROOT, hash_files=False), fth.pycache_fingerprint(dirs)


def refused_entries(current: dict, on_disk: dict[str, str], family: str | None) -> list[str]:
    """Manifest entries that ``on_disk`` would re-pin or drop, through ``manifest_merge.conflicts``
    (a removed file is the entry ``None``), except the files under ``<family>/`` that a family
    capture may re-pin. New files are never refused."""
    removed = {rel: None for rel in current.get("files", {}) if rel not in on_disk}
    bad = manifest_merge.conflicts(current, {**on_disk, **removed})
    if family is not None:
        bad = [rel for rel in bad if not (rel.startswith(f"{family}/") and rel in on_disk)]
    return bad


def refuse_repins(current: dict, on_disk: dict[str, str], family: str | None) -> None:
    bad = refused_entries(current, on_disk, family)
    if bad:
        changed = [rel for rel in bad if rel in on_disk]
        removed = [rel for rel in bad if rel not in on_disk]
        scope = (f"a --family {family} capture re-pins only files under golden/{family}/" if family
                 else "--manifest-only only adds new files")
        raise SystemExit(
            f"golden/MANIFEST.json: {len(changed)} recorded file(s) changed {changed[:5]}"
            f"{' ...' if len(changed) > 5 else ''} and {len(removed)} disappeared {removed[:5]}"
            f"{' ...' if len(removed) > 5 else ''}; {scope}; nothing written; rerun with --replace "
            "only if the change is intended")


def write_manifest(*, family: str | None = None, sources: dict | None = None,
                   tolerances: dict | None = None, replace: bool = False) -> dict:
    manifest = GOLDEN / "MANIFEST.json"
    current = read_json(manifest) if manifest.is_file() else {"schema": 1}
    on_disk = golden_files(GOLDEN)
    if not replace:
        refuse_repins(current, on_disk, family)
    current["schema"] = 1
    current.setdefault("sources", {})          # a manifest first written by Section B's merge
    current.setdefault("tolerances", {})       # has neither key yet
    if family is not None:
        if sources is not None:
            current["sources"][family] = sources
        if tolerances is not None:
            current["tolerances"][family] = tolerances
    current["sources"]["frozen_trees"] = {"summary": fth.SUMMARY, "listing": fth.LISTING}
    build = runtime_fingerprint()
    if replace or "build" not in current:
        current["build"] = build
    elif current["build"] != build:
        print("note: this runtime differs from MANIFEST build; build kept (--replace re-stamps it)",
              file=sys.stderr)
    current["files"] = on_disk
    write_json(manifest, current)
    return current


def capture(family: str, tier: str, *, replace: bool = False) -> dict:
    problems = fth.verify(ROOT, GOLDEN, full=False)
    if problems:
        raise FrozenTreeChanged("frozen trees differ from golden/frozen_trees.json:\n" + "\n".join(problems[:20]))
    if not replace:                      # refused before the capture writes anything
        manifest = GOLDEN / "MANIFEST.json"
        refuse_repins(read_json(manifest) if manifest.is_file() else {}, golden_files(GOLDEN), family)
    run, measure = FAMILIES[family]
    before = guard_state()
    sources = run(tier)
    if guard_state() != before:
        raise FrozenTreeChanged("a frozen tree or a lineage __pycache__ changed during the capture")
    report = measure()
    return write_manifest(family=family, sources=sources,
                          tolerances={"file": report["file"], "status": report["status"]}, replace=replace)


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--replace", action="store_true",
                    help="re-approve changed or removed goldens outside the run's scope and re-stamp build")
    a = ap.parse_args(argv)
    if a.manifest_only:
        m = write_manifest(replace=a.replace)
    elif a.family:
        m = capture(a.family, a.tier, replace=a.replace)
    else:
        ap.error("give --family or --manifest-only")
    print(f"golden/MANIFEST.json: {len(m['files'])} files; tolerance families {sorted(m['tolerances'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
