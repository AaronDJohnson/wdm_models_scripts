"""B8: measured galactic-binary tolerances (G3 rows, G4 stress) and their MANIFEST summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden"


@pytest.fixture(scope="module")
def measured():
    return json.loads((GOLDEN / "galactic_binary" / "tolerances_measured.json").read_text())


def test_manifest_summary(measured):
    m = json.loads((GOLDEN / "MANIFEST.json").read_text())
    assert "galactic_binary/tolerances_measured.json" in m["files"]
    summary = m["tolerances"]["galactic_binary"]
    assert summary["status"] == measured["status"] == "measured; pending user approval"
    assert summary["G3_targets"] == 38 and summary["G3_runs"] == 58


def test_g3_gates_cover_observations(measured):
    for target, t in measured["G3"].items():
        assert t["rows"] >= 3 * t["runs"], target
        assert t["gate_numpy"] >= max(10 * t["observed_numpy"], 1e-9)
        assert t["gate_jax"] >= max(10 * t["observed_jax"], 1e-8)
        assert t["observed_numpy"] <= 1e-6 and t["observed_jax"] <= 1e-6


def test_g3_observations_recomputed_from_rows(measured):
    worst = 0.0
    for path in (GOLDEN / "galactic_binary" / "chain_rows").glob("*/*.json"):
        if path.name != "index.json":
            worst = max([worst] + [r["abs_delta_numpy"] for r in json.loads(path.read_text())["rows"]])
    assert measured["summary"]["G3_max_observed_numpy"] == worst


def test_g4_masks_and_gates(measured):
    g4 = measured["G4"]
    assert g4["mask_mismatches"] == [] and measured["summary"]["G4_mask_mismatches"] == 0
    assert g4["cache_within_numpy_gate"]
    assert g4["gate_jax_well_conditioned"] <= 1e-6
    assert len(g4["per_target"]) >= 30
