"""B2 (G3): stored GB chain rows re-evaluated by lineage C, fast tier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.golden.paths import ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden"
ROWS = GOLDEN / "galactic_binary" / "chain_rows"
CHAINS = ROOT / "wdm_paper_2" / "final_chains" / "gb_likelihood_matrix"
#: Spec section 7.2 G3: row 0 of these runs, verbatim from the chain files.
SPOTS = {("gb_3mo", "exact_flow_seed3009"): -600.5155231449987,
         ("gb_3mo", "tri_exact_flow_seed3009"): -1427.7735926699656,
         ("gb_12mo", "exact_flow_seed3001"): -2765.8831636952596,
         ("gb_12mo", "tri_raw_flow_seed3001"): -5873.84696710096,
         ("gb_6mo", "band_exact_flow_seed3001"): -2755.4883541571508}
#: Sanity ceiling (lineage C's own tests/test_jax.py); the gates come from measure_tolerances_gb.
CEILING = 1e-6


def _manifest_runs() -> dict[tuple[str, str], list[str]]:
    m = json.loads((CHAINS / "manifest.json").read_text())
    return {(case, model): (e["runs"] if "runs" in e else [e["run"]])
            for case, d in m.items() for model, e in d.items()}


@requires_legacy
def test_manifest_has_38_targets_in_58_run_dirs():
    runs = _manifest_runs()
    assert len(runs) == 38
    assert sum(len(r) for r in runs.values()) == 58          # the spec's "59" counted chains/logs
    assert {c for c, _ in runs} == {"gb_3mo", "gb_6mo", "gb_12mo"}


@requires_legacy
def test_index_and_one_fast_file_per_run_dir():
    golden_manifest = json.loads((GOLDEN / "MANIFEST.json").read_text())["files"]
    for (case, model), runs in _manifest_runs().items():
        index = json.loads((ROWS / case / "index.json").read_text())
        assert index["targets"][model] == runs
        for run in runs:
            rel = f"galactic_binary/chain_rows/{case}/{run}.fast.json"
            assert rel in golden_manifest, rel
            rec = json.loads((GOLDEN / rel).read_text())
            assert (rec["case"], rec["model"], rec["run"], rec["tier"]) == (case, model, run, "fast")
            n = rec["rows_total"]
            assert rec["row_indices"] == sorted({0, n // 2, n - 1})
            assert rec["width"] == rec["ndim"] + 4 and rec["lnlike_column"] == rec["ndim"]
            assert rec["chain_file"]["bytes"] == 8 * rec["width"] * n
            assert set(rec["backends"]) == {"numpy", "jax"}
            for row in rec["rows"]:
                assert len(row["theta"]) == rec["ndim"]
                assert row["numpy_raised"] is False
                assert row["abs_delta_numpy"] <= CEILING and row["abs_delta_jax"] <= CEILING


@pytest.mark.parametrize("case,run", sorted(SPOTS))
def test_spec_spot_values(case, run):
    rec = json.loads((ROWS / case / f"{run}.fast.json").read_text())
    row0 = rec["rows"][0]
    assert row0["row"] == 0
    assert row0["lnlike_stored"] == SPOTS[(case, run)]
    assert row0["abs_delta_numpy"] <= 5e-10 and row0["abs_delta_jax"] <= 5e-10
