"""Paths shared by the golden harness (spec §1 layout, §7.1)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from tools.frozen_tree_hashes import DEFAULT_ROOT, FROZEN_TREES

#: The release repository (wdm_models_scripts/).
REL = Path(__file__).resolve().parents[2]
#: The legacy workspace holding the frozen trees (wdm_test/; override with WDM_LEGACY_ROOT).
ROOT = DEFAULT_ROOT
GOLDEN = REL / "golden"
DATA = REL / "data"
WORKERS = REL / "scripts" / "golden" / "workers"


def legacy_available() -> bool:
    """True when every frozen tree is present (this machine); False on CI or Colab."""
    return all((ROOT / tree).is_dir() for tree in FROZEN_TREES)


def tmp_base() -> Path:
    return Path(tempfile.gettempdir()).resolve()


def tmp_root() -> Path:
    """$TMPDIR/golden_run: cwd of every lineage subprocess and parent of every out_dir."""
    path = tmp_base() / "golden_run"
    path.mkdir(parents=True, exist_ok=True)
    return path
