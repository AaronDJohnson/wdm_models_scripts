"""Pin external, never-shipped inputs by sha256 in golden/MANIFEST.json (spec §1 'Not shipped'; Task B13).

    $PY -m scripts.golden.pin_external [--h5 PATH] [--replace]

Reads the private LDC2 Sangria training file once (3.1 GB, about 7 s, read-only) and records
``sources.external.sangria_h5 = {path, bytes, sha256, used_by}``. Every other key of the
manifest is left alone, and A10's ``capture.write_manifest`` keeps ``sources.external``
(it rewrites only ``sources.<family>``, ``sources.frozen_trees``, ``build`` and ``files``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.golden.goldens_io import read_json, sha256_file, write_json
from scripts.golden.paths import GOLDEN

MANIFEST = GOLDEN / "MANIFEST.json"
#: toy_models/galactic_binary/build_segments.py::RAW_H5 (the file every GB segment came from).
SANGRIA_H5 = Path("/Users/aaron/Documents/lisa/project-catalog/LDC2_sangria_training_v2.h5")
USED_BY = "toy_models/galactic_binary/build_segments.py::RAW_H5"


def pin(h5: Path, *, replace: bool = False) -> dict:
    entry = {"path": str(h5), "bytes": h5.stat().st_size, "sha256": sha256_file(h5), "used_by": USED_BY}
    manifest = read_json(MANIFEST)
    if manifest.get("schema") != 1:
        raise ValueError(f"{MANIFEST}: schema {manifest.get('schema')!r}, expected 1")
    external = manifest.setdefault("sources", {}).setdefault("external", {})
    old = external.get("sangria_h5")
    if old is not None and old["sha256"] != entry["sha256"] and not replace:
        raise SystemExit(f"sangria_h5 sha256 would change from {old['sha256']} to {entry['sha256']}; "
                         "rerun with --replace only if the change is intended")
    external["sangria_h5"] = entry
    write_json(MANIFEST, manifest)
    return entry


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", type=Path, default=SANGRIA_H5)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    e = pin(a.h5, replace=a.replace)
    print(f"sources.external.sangria_h5: {e['bytes']} bytes, sha256 {e['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
