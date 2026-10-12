"""golden/MANIFEST.json (spec §7.1 'Output'; Task A10)."""
from __future__ import annotations

import pytest

from scripts.golden.goldens_io import golden_files, read_json, sha256_file
from scripts.golden.paths import GOLDEN, ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")


def _manifest() -> dict:
    return read_json(GOLDEN / "MANIFEST.json")


def test_manifest_lists_every_golden_file_with_its_sha256():
    m = _manifest()
    assert m["schema"] == 1
    assert m["files"] == golden_files()


def test_build_fingerprint():
    b = _manifest()["build"]
    assert {"python", "numpy", "scipy", "jax", "jaxlib", "machine", "platform", "blas", "threads"} <= set(b)
    assert (b["python"], b["numpy"], b["scipy"], b["jax"], b["jaxlib"]) == (
        "3.12.4", "2.3.5", "1.16.3", "0.9.0.1", "0.9.0.1")


def test_sinusoid_sources_and_tolerances():
    m = _manifest()
    s = m["sources"]["sinusoid"]
    assert len(s["chains"]) == 45 and len(s["data"]) == 5
    assert set(s["lineages"]) == {"A", "B"}
    assert m["tolerances"]["sinusoid"]["file"] == "sinusoid/tolerances_measured.json"
    assert m["tolerances"]["sinusoid"]["status"] == "APPROVED by the user 2026-10-12"     # Task F4
    measured = read_json(GOLDEN / "sinusoid" / "tolerances_measured.json")
    assert m["tolerances"]["sinusoid"]["approved"] == measured["approved"]


@requires_legacy
def test_sources_match_the_frozen_trees():
    s = _manifest()["sources"]["sinusoid"]
    assert sha256_file(ROOT / s["bundle"]["path"]) == s["bundle"]["sha256"]
    assert sha256_file(ROOT / s["sinusoid_final_manifest"]["path"]) == s["sinusoid_final_manifest"]["sha256"]
    for key, v in s["chains"].items():
        path = ROOT / "wdm_paper_2" / "sinusoid_final" / v["run"] / "chain_0.bin"
        assert sha256_file(path) == v["chain_0_sha256"], key
