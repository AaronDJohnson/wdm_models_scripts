"""P0 tolerance measurement (spec §7.3; Task A10)."""
from __future__ import annotations

import pytest

from scripts.golden import measure_tolerances as MT
from scripts.golden.goldens_io import read_json
from scripts.golden.sinusoid_goldens import EVAL, MODELS

pytestmark = pytest.mark.golden


def test_propose_takes_ten_times_observed_or_the_floor():
    floor = {"atol": 1e-9, "rtol": 0.0}
    assert MT.propose({"near_max_abs": 0.0, "stress_max_rel": 0.0}, floor) == {"atol": 1e-9, "rtol": 0.0}
    got = MT.propose({"near_max_abs": 3e-9, "stress_max_rel": 2e-15}, floor)
    assert got["atol"] == pytest.approx(3e-8) and got["rtol"] == pytest.approx(2e-14)


def test_violations_use_the_runner_split():
    points = {"t": {"model": "band_raw", "abs_delta": [1e-6, 2e-3], "reference": [-5.0e4, -5.0e5],
                    "near": [True, False]}}
    assert MT.violations(points, {"band_raw": {"atol": 1e-5, "rtol": 1e-8}}) == []
    assert len(MT.violations(points, {"band_raw": {"atol": 1e-7, "rtol": 1e-9}})) == 2
    inf_point = {"t": {"model": "band_raw", "abs_delta": [0.0], "reference": ["-inf"], "near": [False]}}
    assert MT.violations(inf_point, {"band_raw": {"atol": 0.0, "rtol": 0.0}}) == []


def test_floors_cover_every_model():
    for backend in ("numpy", "jax"):
        assert set(MT.FLOORS[backend]) == set(MODELS)


def test_measured_file_proposes_a_gate_for_every_model_and_backend():
    t = read_json(MT.OUT)
    assert t["status"] == MT.STATUS
    for backend in ("numpy", "jax"):
        p = t["proposed"][backend]
        assert p["inf_mask"] == "identical"
        for m in MODELS:
            assert p["references"][m]["atol"] >= MT.FLOORS[backend][m]["atol"]
            assert p["references"][m]["rtol"] >= MT.FLOORS[backend][m]["rtol"]
            assert p["chain_rows"][m]["atol"] >= MT.FLOORS["jax"][m]["atol"]
            assert t["observed"][backend]["references"][m]["inf_mismatch"] == 0


def test_every_recorded_point_passes_its_proposed_gate():
    t = read_json(MT.OUT)
    for lineage, backend in MT.BACKEND.items():
        for path in EVAL.glob(f"{lineage}_references_*.json"):
            assert MT.violations(read_json(path)["points"], t["proposed"][backend]["references"]) == []
        for path in EVAL.glob(f"{lineage}_chain_rows_*.json"):
            assert MT.violations(read_json(path)["points"], t["proposed"][backend]["chain_rows"]) == []
