"""Fast-tier re-evaluation of G1 by lineages A and B (spec §7.1 gate S0; Tasks A6, A7)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.golden import sinusoid_goldens as SG
from scripts.golden.goldens_io import read_json, read_npz, to_float
from scripts.golden.paths import ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
#: committed reports record wdm_file relative to ROOT (portable); live worker meta is absolute.
VENDORED_WDM_REL = "toy_models/vendor/wdm/src/wdm/__init__.py"
VENDORED_WDM = str(ROOT / VENDORED_WDM_REL)
#: spec §3.6: bitwise reproduction verified (max|delta| = 0.0) for these models.
VERIFIED_BITWISE = ("whittle_raw", "whittle_cut", "band_raw", "mid_raw", "band_cut", "tri_raw")
CACHE_SHA = {
    "bandcov_v1_4bfb466c191c79f9879a.npz": "752b3ea46cf14feaf5b8845ad39999a0b596a956d6af9e58072f691a8cd6c2c2",
    "cN_N16384_dt0.0009765625_fmin4_fref1.npy": "d263bf25960ab6f4e371fd597da0e28123cba569ca085e2af5e6c3599257213f",
}
TAGS = {f"{c}__{m}" for c in SG.CASES for m in SG.MODELS}


def _report(lineage: str) -> dict:
    return read_json(SG.EVAL / f"{lineage}_references_fast.json")


def test_compare_splits_near_and_stress_and_counts_inf_mismatch():
    cmp = SG.compare([1.0, -np.inf, 2.5, -np.inf], [1.0 + 1e-12, -np.inf, 2.0, 5.0], [True, False, False, False])
    assert cmp["inf_mismatch"] == 1
    assert cmp["near_max_abs"] == pytest.approx(1e-12, rel=1e-3)
    assert (cmp["stress_max_abs"], cmp["stress_max_rel"]) == (0.5, 0.25)
    with pytest.raises(FloatingPointError):
        SG.compare([np.nan], [1.0], [True])


@requires_legacy
@pytest.mark.legacy
def test_lineage_A_worker_reproduces_two_reference_points_bitwise():
    from scripts.golden.legacy_eval import evaluate
    ref = read_npz(SG.REFS / "manuscript_mid_band_raw_reference.npz")
    got = evaluate("A", "sinusoid", "manuscript_mid", "band_raw", ref["theta"][:2])
    assert np.array_equal(got, ref["loglike"][:2])


@requires_legacy
@pytest.mark.legacy
def test_lineage_A_exact_uses_a_private_cache_copy():
    from scripts.golden.legacy_eval import Job, evaluate_jobs
    ref = read_npz(SG.REFS / "manuscript_mid_exact_reference.npz")
    results, meta = evaluate_jobs("A", "sinusoid", [Job("t", "manuscript_mid", "exact", ref["theta"][:1])])
    # exact is not claimed bitwise (spec §3.6: PCG oracle self-noise 4e-9); 1e-7 is the spec floor.
    assert abs(results["t"]["lnlike"][0] - ref["loglike"][0]) <= 1e-7
    assert meta["cache_sha256"] == CACHE_SHA
    assert meta["wdm_file"] == VENDORED_WDM


def test_A_fast_report_covers_every_target():
    rep = _report("A")
    assert (rep["lineage"], rep["tier"], rep["kind"]) == ("A", "fast", "references")
    assert set(rep["points"]) == TAGS
    assert set(rep["models"]) == set(SG.MODELS)
    for m in rep["models"].values():
        assert (m["n_points"], m["inf_mismatch"]) == (12, 0)
    assert rep["worker"]["wdm_file"] == VENDORED_WDM_REL
    assert rep["worker"]["cache_sha256"] == CACHE_SHA


def test_A_reproduces_the_verified_models_bitwise():
    models = _report("A")["models"]
    for model in VERIFIED_BITWISE:
        assert (models[model]["near_max_abs"], models[model]["stress_max_abs"]) == (0.0, 0.0), model
