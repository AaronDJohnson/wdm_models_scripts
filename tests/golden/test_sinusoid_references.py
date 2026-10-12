"""G1: the 45 sinusoid CPU reference files (spec §7.2 G1; Task A5)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.golden import sinusoid_goldens as SG
from scripts.golden.goldens_io import read_json, read_npz, sha256_file
from scripts.golden.paths import legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
EDGE_INF = [4, 8, 12, 16, 18, 20, 22, 25, 26, 27, 28, 29]
NEG_INF = {"band_exact": EDGE_INF, "mid_exact": sorted(EDGE_INF + [10]), "tri_exact": EDGE_INF}
NEAR = [0] + list(range(1, 25, 2))
#: spec §7.2 G1 truth table plus the three models it omits (read from the files); checked to 1e-6.
TRUTH = {"whittle_raw": -60478.525184258615, "whittle_cut": -55246.34617419086,
         "band_raw": -55272.55006513103, "mid_raw": -55298.8849777882, "band_cut": -37497.08979094244,
         "tri_raw": -55257.970597, "band_pad": -36817.455302, "tri_pad": -36782.851333,
         "band_exact": -36661.864656, "mid_exact": -36656.587913, "tri_exact": -36631.425316,
         "exact": -36628.959581, "mid_pad": -36810.35689860746, "mid_cut": -37491.71663964758,
         "tri_cut": -37464.81354909294}


def _index() -> dict:
    return read_json(SG.REFS / "INDEX.json")


def test_exactly_the_45_paper_references_are_present():
    names = {f"{c}_{m}_reference.npz" for c in SG.CASES for m in SG.MODELS}
    assert {p.name for p in SG.REFS.glob("*.npz")} == names
    assert set(_index()["files"]) == names


def test_files_match_their_index_sha256():
    for name, entry in _index()["files"].items():
        assert sha256_file(SG.REFS / name) == entry["sha256"], name


@pytest.mark.parametrize("case", SG.CASES)
@pytest.mark.parametrize("model", SG.MODELS)
def test_reference_layout(case, model):
    ref = read_npz(SG.REFS / f"{case}_{model}_reference.npz")
    n, ndim = (27, 5) if model.startswith("whittle") else (30, 6)
    assert ref["theta"].shape == (n, ndim) and ref["loglike"].shape == (n,) and ref["near"].shape == (n,)
    assert np.flatnonzero(ref["near"]).tolist() == NEAR
    assert np.flatnonzero(ref["loglike"] == -np.inf).tolist() == NEG_INF.get(model, [])
    assert np.all(np.isfinite(ref["theta"]))
    assert abs(float(ref["loglike"][0]) - TRUTH[model]) <= 1e-6
    entry = _index()["files"][f"{case}_{model}_reference.npz"]
    assert entry["fast_index"] == [0, 1, 2, 4 if model in NEG_INF else 25]
    assert entry["truth_lnlike"] == float(ref["loglike"][0])


def test_fast_indices():
    ll = np.zeros(30)
    assert SG.fast_indices(ll) == [0, 1, 2, 25]
    ll[[4, 8]] = -np.inf
    assert SG.fast_indices(ll) == [0, 1, 2, 4]
    assert SG.fast_indices(np.zeros(27)) == [0, 1, 2, 25]


@requires_legacy
def test_reference_sha_equals_every_run_meta():
    manifest = read_json(SG.SINUSOID_FINAL / "manifest.json")
    for case in SG.CASES:
        for model in SG.MODELS:
            meta = read_json(SG.SINUSOID_FINAL / manifest["runs"][case][model] / "run_meta.json")
            assert meta["target"]["asset_sha256"]["reference"] == sha256_file(
                SG.REFS / f"{case}_{model}_reference.npz"), (case, model)


@requires_legacy
def test_bundle_hash_is_the_recorded_one():
    assert sha256_file(SG.BUNDLE) == _index()["bundle"]["sha256"]
