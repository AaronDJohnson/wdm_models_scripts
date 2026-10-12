"""B13: the private Sangria HDF5 is pinned by sha256 in golden/MANIFEST.json (spec §1 'Not shipped')."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.golden.goldens_io import read_json, sha256_file
from scripts.golden.paths import GOLDEN, ROOT, legacy_available

pytestmark = pytest.mark.golden
H5 = "/Users/aaron/Documents/lisa/project-catalog/LDC2_sangria_training_v2.h5"
SHA = "f54ca61355d0c6208fbea5fc1b2dcce7332025de264fef33ebe87dea657c7ee6"
BYTES = 3082321484
USED_BY = "toy_models/galactic_binary/build_segments.py::RAW_H5"


def _entry() -> dict:
    return read_json(GOLDEN / "MANIFEST.json")["sources"]["external"]["sangria_h5"]


def test_sangria_h5_is_pinned():
    e = _entry()
    assert (e["path"], e["bytes"], e["sha256"], e["used_by"]) == (H5, BYTES, SHA, USED_BY)


@pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
def test_path_is_the_frozen_build_segments_literal():
    text = (ROOT / "toy_models" / "galactic_binary" / "build_segments.py").read_text()
    match = re.search(r'^RAW_H5 = Path\("([^"]+)"\)$', text, flags=re.M)
    assert match is not None and match.group(1) == _entry()["path"]


@pytest.mark.slow
def test_sangria_h5_bytes_match():
    path = Path(_entry()["path"])
    if not path.is_file():
        pytest.skip(f"private Sangria file not present: {path}")
    assert path.stat().st_size == BYTES
    assert sha256_file(path) == SHA
