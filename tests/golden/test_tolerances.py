"""P0 tolerance measurement (spec §7.3; Task A10)."""
from __future__ import annotations

import math

import pytest

from scripts.golden import measure_tolerances as MT
from scripts.golden.goldens_io import read_json, write_json
from scripts.golden.sinusoid_goldens import EVAL, GS, MODELS

pytestmark = pytest.mark.golden


def test_propose_takes_ten_times_observed_or_the_floor():
    floor = {"atol": 1e-9, "rtol": 0.0}
    assert MT.propose({"near_max_abs": 0.0, "stress_max_rel": 0.0}, floor) == {"atol": 1e-9, "rtol": 0.0}
    got = MT.propose({"near_max_abs": 3e-9, "stress_max_rel": 2e-15}, floor)
    assert got["atol"] == pytest.approx(3e-8) and got["rtol"] == pytest.approx(2e-14)


def test_violations_use_the_runner_split():
    points = {"t": {"model": "band_raw", "abs_delta": [1e-6, 2e-3], "reference": [-5.0e4, -5.0e5],
                    "value": [-5.0e4 + 1e-6, -5.0e5 + 2e-3], "near": [True, False]}}
    assert MT.violations(points, {"band_raw": {"atol": 1e-5, "rtol": 1e-8}}) == []
    assert len(MT.violations(points, {"band_raw": {"atol": 1e-7, "rtol": 1e-9}})) == 2
    inf_point = {"t": {"model": "band_raw", "abs_delta": [0.0], "reference": ["-inf"], "value": ["-inf"],
                       "near": [False]}}
    assert MT.violations(inf_point, {"band_raw": {"atol": 0.0, "rtol": 0.0}}) == []


LOOSE = {"band_raw": {"atol": 1e300, "rtol": 1e300}}


@pytest.mark.parametrize("value, reference, abs_delta", [
    ("-inf", -5.0e4, 0.0),           # the old compare() stored abs_delta 0 at a mask mismatch
    (-5.0e4, "-inf", 0.0),
    ("-inf", -5.0e4, "inf"),         # compare() now records the mismatch as an infinite delta
])
def test_violations_flag_a_minus_inf_mask_mismatch_at_any_gate(value, reference, abs_delta):
    points = {"t": {"model": "band_raw", "abs_delta": [abs_delta], "reference": [reference], "value": [value],
                    "near": [False]}}
    for gates in (LOOSE, {"band_raw": {"atol": 0.0, "rtol": 0.0}}):
        got = MT.violations(points, gates)
        assert len(got) == 1 and "mask" in got[0], got


@pytest.mark.parametrize("value, reference", [("nan", -5.0e4), (-5.0e4, "nan"), ("inf", -5.0e4), (-5.0e4, "inf")])
def test_violations_flag_nan_or_plus_inf(value, reference):
    points = {"t": {"model": "band_raw", "abs_delta": ["nan"], "reference": [reference], "value": [value]}}
    assert len(MT.violations(points, LOOSE)) == 1


def test_violations_flag_a_nan_delta_and_ragged_points():
    nan_delta = {"t": {"model": "band_raw", "abs_delta": ["nan"], "reference": [-5.0e4], "value": [-5.0e4]}}
    assert len(MT.violations(nan_delta, LOOSE)) == 1
    ragged = {"t": {"model": "band_raw", "abs_delta": [0.0], "reference": [-5.0e4, -6.0e4], "value": [-5.0e4]}}
    assert len(MT.violations(ragged, LOOSE)) == 1


def _synthetic_reports(eval_dir, *, drop=None):
    """Four minimal lineage reports; ``drop`` = (lineage, kind, model) leaves that model out."""
    for lineage in ("A", "B"):
        for kind, count in (("references", "n_points"), ("chain_rows", "n_rows")):
            stats = ({"near_max_abs": 0.0, "stress_max_abs": 0.0, "stress_max_rel": 0.0} if kind == "references"
                     else {"max_abs": 0.0, "max_rel": 0.0})
            models = {m: dict(stats, inf_mismatch=0, **{count: 12}) for m in MODELS
                      if (lineage, kind, m) != drop}
            write_json(eval_dir / f"{lineage}_{kind}_fast.json",
                       {"tier": "fast", "kind": kind, "lineage": lineage, "models": models, "points": {}})


@pytest.fixture
def scratch_gs(tmp_path, monkeypatch):
    gs = tmp_path / "sinusoid"
    (gs / "legacy_eval").mkdir(parents=True)
    monkeypatch.setattr(MT, "GS", gs)
    monkeypatch.setattr(MT, "EVAL", gs / "legacy_eval")
    monkeypatch.setattr(MT, "OUT", gs / "tolerances_measured.json")
    return gs


def test_measure_sinusoid_runs_on_complete_synthetic_reports(scratch_gs):
    _synthetic_reports(scratch_gs / "legacy_eval")
    MT.measure_sinusoid()
    assert (scratch_gs / "tolerances_measured.json").is_file()


@pytest.mark.parametrize("drop", [("A", "references", "band_raw"), ("B", "chain_rows", "exact")])
def test_measure_sinusoid_refuses_a_model_with_no_points(scratch_gs, drop):
    _synthetic_reports(scratch_gs / "legacy_eval", drop=drop)
    with pytest.raises(ValueError, match=drop[2]):
        MT.measure_sinusoid()
    assert not (scratch_gs / "tolerances_measured.json").exists()


def test_measure_sinusoid_refuses_an_inf_mismatch_recorded_as_inf(scratch_gs):
    """A mismatching report carries "inf" maxima (compare()); the -inf mask check still fires."""
    _synthetic_reports(scratch_gs / "legacy_eval")
    path = scratch_gs / "legacy_eval" / "A_references_fast.json"
    rep = read_json(path)
    rep["models"]["tri_pad"].update(inf_mismatch=1, stress_max_abs="inf", stress_max_rel="inf")
    write_json(path, rep)
    with pytest.raises(ValueError, match="tri_pad"):
        MT.measure_sinusoid()
    assert not (scratch_gs / "tolerances_measured.json").exists()


def test_measure_sinusoid_reproduces_the_committed_bytes(tmp_path, monkeypatch):
    """Every proposed gate is recomputed from the committed lineage reports (written to a scratch dir)."""
    gs = tmp_path / "sinusoid"
    gs.mkdir()
    monkeypatch.setattr(MT, "GS", gs)
    monkeypatch.setattr(MT, "OUT", gs / "tolerances_measured.json")
    MT.measure_sinusoid()
    assert (gs / "tolerances_measured.json").read_bytes() == (GS / "tolerances_measured.json").read_bytes()


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
        for kind in ("references", "chain_rows"):
            paths = sorted(EVAL.glob(f"{lineage}_{kind}_*.json"))
            assert paths, f"no {lineage} {kind} report under {EVAL}"
            for path in paths:
                points = read_json(path)["points"]
                assert points, path
                assert MT.violations(points, t["proposed"][backend][kind]) == [], path
                if kind == "references":        # the real data exercise the -inf mask branch too
                    assert any(MT.to_float(r) == -math.inf for p in points.values() for r in p["reference"]), path
