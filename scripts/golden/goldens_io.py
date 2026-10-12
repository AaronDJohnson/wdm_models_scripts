"""File helpers shared by every golden family (Tasks A4-A10; other sections reuse them).

Re-exports the hashing/JSON helpers of scripts.golden.workers._common so the harness and
the workers serialise identically.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from scripts.golden.paths import DATA, GOLDEN
from scripts.golden.workers._common import (array_summary, jsonable, read_json, sha256_array,
                                            sha256_bytes, sha256_file, to_float, write_json)

__all__ = ["array_summary", "golden_files", "jsonable", "read_json", "read_npz", "sha256_array",
           "sha256_bytes", "sha256_file", "to_float", "update_data_manifest", "write_json", "write_npz"]

_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_IGNORED = {"MANIFEST.json", ".DS_Store"}


def write_npz(path: Path | str, arrays: dict[str, np.ndarray]) -> str:
    """Deterministic compressed .npz (sorted members, fixed timestamps); returns its sha256."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, member.getvalue(), compresslevel=6)
    body = buf.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return sha256_bytes(body)


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def update_data_manifest(files: dict[str, str], provenance: dict[str, dict] | None = None,
                         manifest: Path | None = None, *, replace: bool = False) -> dict:
    """Merge ``files`` (relpath under data/ -> sha256) and ``provenance`` (relpath ->
    {sha256, bytes, from, ...}) into data/MANIFEST.json, the schema shared with Section B's
    manifest_merge: ``{"schema": 1, "files": {relpath: sha256}, "provenance": {...}}``.
    Every other key is left alone; an existing entry with other bytes raises unless ``replace``."""
    manifest = DATA / "MANIFEST.json" if manifest is None else Path(manifest)
    current = read_json(manifest) if manifest.is_file() else {"schema": 1, "files": {}}
    if current.get("schema") != 1:
        raise ValueError(f"{manifest}: schema {current.get('schema')!r}, expected 1")
    have = current.setdefault("files", {})
    clash = sorted(rel for rel, sha in files.items() if have.get(rel) not in (None, sha))
    if clash and not replace:
        raise ValueError(f"{manifest}: sha256 would change for {clash}; pass replace=True only if intended")
    have.update(files)
    if provenance:
        current.setdefault("provenance", {}).update(provenance)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, current)
    return current


def golden_files(golden: Path | None = None) -> dict[str, str]:
    """relpath -> sha256 of every file under golden/ except MANIFEST.json and .DS_Store."""
    golden = GOLDEN if golden is None else Path(golden)
    return {p.relative_to(golden).as_posix(): sha256_file(p)
            for p in sorted(golden.rglob("*"))
            if p.is_file() and p.name not in _IGNORED and "__pycache__" not in p.parts}
