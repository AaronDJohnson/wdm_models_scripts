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
#: Lineage A re-evaluates its own G1 references: measured bitwise (max|delta| = 0.0) for all 15
#: models x 12 fast-tier points in A_references_fast.json (spec §3.6 claimed only 6).
VERIFIED_BITWISE = SG.MODELS
CACHE_SHA = {
    "bandcov_v1_4bfb466c191c79f9879a.npz": "752b3ea46cf14feaf5b8845ad39999a0b596a956d6af9e58072f691a8cd6c2c2",
    "cN_N16384_dt0.0009765625_fmin4_fref1.npy": "d263bf25960ab6f4e371fd597da0e28123cba569ca085e2af5e6c3599257213f",
}
TAGS = {f"{c}__{m}" for c in SG.CASES for m in SG.MODELS}


def _report(lineage: str) -> dict:
    return read_json(SG.EVAL / f"{lineage}_references_fast.json")


def test_compare_splits_near_and_stress_and_counts_inf_mismatch():
    cmp = SG.compare([1.0, -np.inf, 2.5], [1.0 + 1e-12, -np.inf, 2.0], [True, False, False])
    assert cmp["inf_mismatch"] == 0
    assert cmp["near_max_abs"] == pytest.approx(1e-12, rel=1e-3)
    assert (cmp["stress_max_abs"], cmp["stress_max_rel"]) == (0.5, 0.25)
    with pytest.raises(FloatingPointError):
        SG.compare([np.nan], [1.0], [True])


@pytest.mark.parametrize("values, reference", [([-np.inf], [5.0]), ([5.0], [-np.inf])])
def test_compare_records_a_minus_inf_mismatch_as_an_infinite_delta(values, reference):
    """Never abs_delta = 0: the mismatch shows in the per-point delta and in the maxima."""
    cmp = SG.compare([1.0, *values], [1.0, *reference], [True, False])
    assert cmp["inf_mismatch"] == 1
    assert cmp["abs_delta"][1] == np.inf and cmp["rel_delta"][1] == np.inf
    assert (cmp["stress_max_abs"], cmp["stress_max_rel"]) == (np.inf, np.inf)
    with pytest.raises(FloatingPointError):            # a NaN or +inf reference is refused too
        SG.compare([1.0], [np.nan], [True])


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
    rep = _report("A")
    assert len(VERIFIED_BITWISE) == 15
    for model in VERIFIED_BITWISE:
        m = rep["models"][model]
        assert (m["near_max_abs"], m["stress_max_abs"], m["stress_max_rel"]) == (0.0, 0.0, 0.0), model
    for tag, p in rep["points"].items():                # value == reference, -inf included
        assert p["value"] == p["reference"], tag


#: toy_models/sinusoid/gpu/run.py::main: validate_reference(atol=1e-4 exact / 1e-5 matrix, stress_rtol=1e-8)
RUNNER_ATOL = {"exact": 1e-4}


def test_B_fast_report_covers_every_target_on_cpu_x64():
    rep = _report("B")
    assert (rep["lineage"], rep["tier"], rep["kind"]) == ("B", "fast", "references")
    assert set(rep["points"]) == TAGS
    for m in rep["models"].values():
        assert (m["n_points"], m["inf_mismatch"]) == (12, 0)
    assert (rep["worker"]["jax"]["backend"], rep["worker"]["jax"]["x64"], rep["worker"]["jax"]["version"]) == (
        "cpu", True, "0.9.0.1")
    assert rep["worker"]["wdm_file"] == VENDORED_WDM_REL   # committed reports are ROOT-relative (A6 review)
    assert rep["worker"]["cache_sha256"] == CACHE_SHA


def test_B_meets_the_runner_validation_gate():
    for tag, p in _report("B")["points"].items():
        atol = RUNNER_ATOL.get(p["model"], 1e-5)
        for d, ref, near in zip(p["abs_delta"], p["reference"], p["near"]):
            r = to_float(ref)
            if math.isfinite(r):
                assert d <= (atol if near else atol + 1e-8 * abs(r)), tag


def test_B_flags_exactly_the_minus_inf_rows():
    for tag, p in _report("B")["points"].items():
        assert p["failed"] == [to_float(v) == -math.inf for v in p["value"]], tag


@requires_legacy
@pytest.mark.legacy
def test_lineage_B_worker_reproduces_an_edge_exact_mask():
    from scripts.golden.legacy_eval import evaluate
    ref = read_npz(SG.REFS / "manuscript_mid_band_exact_reference.npz")
    got = evaluate("B", "sinusoid", "manuscript_mid", "band_exact", ref["theta"][[0, 4]])
    # pins the -inf mask; the near-point level is the runner's gate (S0 measures the real one)
    assert got[1] == -np.inf and abs(got[0] - ref["loglike"][0]) <= 1e-5
