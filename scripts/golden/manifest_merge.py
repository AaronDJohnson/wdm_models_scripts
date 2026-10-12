"""Read-merge-write for ``data/MANIFEST.json`` and ``golden/MANIFEST.json``.

Both manifests have ``{"schema": 1, "files": {relpath: sha256}, ...}`` with ``relpath`` relative
to the manifest's own directory (Section A: A4 for ``data/``, A10 for ``golden/``). Per-file details
go in a separate ``provenance`` section, ``{relpath: {"sha256", "bytes", "from", ...}}``; ``files``
values are always plain sha256 strings.
Section B never rewrites a manifest from scratch: it adds entries, refuses to change
an existing entry's sha256 unless ``replace=True``, and leaves every other key alone.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA = 1


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def entry_sha(entry) -> str | None:
    """The sha256 of a manifest entry: the string itself (or the ``sha256`` of a provenance dict)."""
    return entry.get("sha256") if isinstance(entry, dict) else entry


def load(manifest_path) -> dict:
    path = Path(manifest_path)
    if not path.exists():
        return {"schema": SCHEMA, "files": {}}
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {manifest.get('schema')!r}, expected {SCHEMA}")
    manifest.setdefault("files", {})
    return manifest


def conflicts(manifest: dict, files: dict) -> list[str]:
    """Relpaths already in ``manifest`` whose recorded sha256 differs from the new entry's."""
    have = manifest.get("files", {})
    return sorted(rel for rel, entry in files.items()
                  if rel in have and entry_sha(have[rel]) != entry_sha(entry))


def merge_entries(manifest_path, files: dict, *, provenance: dict | None = None,
                  tolerances: dict | None = None, replace: bool = False) -> dict:
    """Add ``files`` (and optional ``provenance`` / ``tolerances`` sections) to a manifest."""
    path = Path(manifest_path)
    manifest = load(path)
    bad = conflicts(manifest, files)
    if bad and not replace:
        raise ValueError(f"{path}: sha256 would change for {bad[:5]} (+{max(0, len(bad) - 5)} more); "
                         "rerun with replace=True only if the change is intended")
    manifest["files"].update(files)
    if provenance:
        manifest.setdefault("provenance", {}).update(provenance)
    if tolerances:
        manifest.setdefault("tolerances", {}).update(tolerances)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return manifest
