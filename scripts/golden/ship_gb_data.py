"""Copy the shipped galactic-binary inputs from the frozen trees into ``data/``.

    $PY -P scripts/golden/ship_gb_data.py [--workspace DIR] [--replace]

Read-only toward the sources: each file is copied byte for byte, its sha256 is checked
before and after the copy, and ``data/MANIFEST.json`` gains, per file, A4's schema:
``files[rel]`` = sha256 and ``provenance[rel]`` = ``{"sha256", "bytes", "from"}`` (+ ``identical_to``
for the foreground twins), merged and never rewritten (see ``manifest_merge``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if not sys.flags.safe_path:
    raise SystemExit("run with python -P (PYTHONSAFEPATH)")
HERE = Path(__file__).resolve().parent
REL = HERE.parents[1]
for _p in (str(HERE), str(REL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import manifest_merge  # noqa: E402
from scripts.golden.paths import DATA, ROOT  # noqa: E402  (Section A)

DEFAULT_WORKSPACE = ROOT
STUDY = "wdm_paper_2/final_chains/gb_duration_study"

#: data/ relpath -> workspace relpath of the frozen original.
SOURCES: dict[str, str] = {}
for _m in (3, 6, 12):
    SOURCES[f"galactic_binary/prepared/gb_{_m}mo.npz"] = f"toy_models/galactic_binary/prepared/gb_{_m}mo.npz"
    SOURCES[f"galactic_binary/prepared/gb_{_m}mo_report.json"] = (
        f"toy_models/galactic_binary/prepared/gb_{_m}mo_report.json")
    SOURCES[f"galactic_binary/segments/segment_snr7_{_m}mo_3bands.npz"] = (
        f"{STUDY}/data_3bands_ntopt/segment_snr7_{_m}mo_3bands.npz")
    SOURCES[f"galactic_binary/segments/store_snr7_{_m}mo_3bands_pm3.npz"] = (
        f"{STUDY}/data_3bands_ntopt/store_snr7_{_m}mo_3bands_pm3.npz")
    SOURCES[f"galactic_binary/foreground/foreground_{_m}mo_snr7.json"] = (
        f"{STUDY}/foreground_fit/foreground_{_m}mo_snr7.json")
    SOURCES[f"galactic_binary/foreground/foreground_{_m}mo_snr7.npz"] = (
        f"{STUDY}/foreground_fit/foreground_{_m}mo_snr7.npz")

#: Byte-identical twins of shipped files (build_segments.FOREGROUND reads the twin's name).
TWINS = {f"galactic_binary/foreground/foreground_{m}mo_snr7.npz":
         f"{STUDY}/foreground_fit/foreground_{m}mo.npz" for m in (3, 6, 12)}


def ship(workspace: Path, data_dir: Path, *, replace: bool = False) -> dict[str, dict]:
    files, provenance = {}, {}
    for rel, src_rel in SOURCES.items():
        src = workspace / src_rel
        sha = manifest_merge.sha256_file(src)
        files[rel] = sha
        provenance[rel] = {"sha256": sha, "bytes": src.stat().st_size, "from": src_rel}
        if rel in TWINS:
            if manifest_merge.sha256_file(workspace / TWINS[rel]) != sha:
                raise RuntimeError(f"{TWINS[rel]} is no longer identical to {src_rel}")
            provenance[rel]["identical_to"] = TWINS[rel]
    bad = manifest_merge.conflicts(manifest_merge.load(data_dir / "MANIFEST.json"), files)
    if bad and not replace:
        raise SystemExit(f"data/MANIFEST.json already holds other bytes for {bad}; use --replace")
    for rel, src_rel in SOURCES.items():
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / src_rel, dst)
        if manifest_merge.sha256_file(dst) != files[rel]:
            raise RuntimeError(f"copy of {src_rel} does not match its source")
    manifest_merge.merge_entries(data_dir / "MANIFEST.json", files, provenance=provenance, replace=replace)
    return provenance


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    files = ship(a.workspace.resolve(), a.data_dir.resolve(), replace=a.replace)
    total = sum((a.data_dir / rel).stat().st_size for rel in files)
    print(f"shipped {len(files)} galactic-binary files ({total / 2**20:.2f} MiB) into {a.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
