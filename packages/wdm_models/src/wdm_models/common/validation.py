"""Shipped data: where they live and how a shipped file is checked (spec §1 "Shipped data").

The release data sit in ``wdm_models_scripts/data`` with ``data/MANIFEST.json`` (relpath ->
sha256).  Every loader passes the sha256 **pinned in code**; :func:`shipped` additionally
requires the manifest to record the same digest, then returns a ``domainia.assets.Artifact``
that re-hashes the file on load.  Nothing is ever written.

``data_root`` resolution: an explicit argument wins, then ``$WDM_MODELS_DATA``, then the
repository's ``data/`` (when this package runs from ``packages/wdm_models/src``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from domainia.assets import Artifact, ArtifactError, ArtifactMissing

__all__ = ["DATA_ENV", "default_data_root", "manifest_sha256", "resolve_data_root", "shipped"]

DATA_ENV = "WDM_MODELS_DATA"
#: packages/wdm_models/src/wdm_models/common/validation.py -> wdm_models_scripts/data
_REPO_DATA = Path(__file__).resolve().parents[5] / "data"


def default_data_root() -> Path:
    env = os.environ.get(DATA_ENV)
    if env:
        return Path(env)
    if (_REPO_DATA / "MANIFEST.json").is_file():
        return _REPO_DATA
    raise FileNotFoundError(f"shipped data not found: pass data_root= or set ${DATA_ENV} "
                            "(the wdm_models_scripts data/ directory)")


def resolve_data_root(data_root: str | os.PathLike | None) -> Path:
    return default_data_root() if data_root is None else Path(data_root)


def manifest_sha256(data_root: str | os.PathLike, rel: str) -> str:
    """The digest ``data_root/MANIFEST.json`` records for ``rel`` (fail-closed)."""
    manifest = Path(data_root) / "MANIFEST.json"
    if not manifest.is_file():
        raise ArtifactMissing(f"{manifest} is missing")
    files = json.loads(manifest.read_text()).get("files", {})
    if rel not in files:
        raise ArtifactError(f"{rel} is not in {manifest}")
    return files[rel]


def shipped(data_root: str | os.PathLike, rel: str, sha256: str) -> Artifact:
    """``Artifact(data_root/rel, sha256=sha256)`` after checking that the manifest records the
    same pinned digest."""
    recorded = manifest_sha256(data_root, rel)
    if recorded != sha256:
        raise ArtifactError(f"MANIFEST.json records {recorded} for {rel}; the release pins {sha256}")
    return Artifact(Path(data_root) / rel, sha256=sha256)
