"""B12: lineage C behind Section A's legacy_eval.evaluate (spec 7.1 harness interface)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.golden import legacy_eval
from scripts.golden.lineage import WorkerError
from scripts.golden.paths import DATA, GOLDEN, ROOT, legacy_available

pytestmark = [pytest.mark.golden, pytest.mark.legacy,
              pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")]
ROWS = GOLDEN / "galactic_binary" / "chain_rows"
#: (case, model, run directory): one Fourier-exact, one Schur edge-exact, one cropped Whittle.
RUNS = (("gb_3mo", "exact", "exact_flow_seed3009"), ("gb_3mo", "band_exact", "band_exact_flow_seed3008"),
        ("gb_3mo", "whittle_cut", "whittle_cut_flow_seed3007"))


def _g3(case: str, run: str) -> tuple[np.ndarray, np.ndarray]:
    rec = json.loads((ROWS / case / f"{run}.fast.json").read_text())
    return (np.array([r["theta"] for r in rec["rows"]], dtype=np.float64),
            np.array([r["lnlike_numpy"] for r in rec["rows"]], dtype=np.float64))


def test_lineage_c_is_registered():
    assert legacy_eval.WORKERS[("C", "galactic_binary")] == "gb_legacy"
    assert legacy_eval.WORKER_ARGS["galactic_binary"]() == ["--workspace", str(ROOT), "--data-dir", str(DATA)]


def test_evaluate_reproduces_the_g3_numpy_values():
    jobs, stored = [], {}
    for case, model, run in RUNS:
        theta, values = _g3(case, run)
        jobs.append(legacy_eval.Job(run, case, model, theta))
        stored[run] = values
    results, meta = legacy_eval.evaluate_jobs("C", "galactic_binary", jobs)
    assert meta["lineage"] == "C" and meta["wdm_file"] is None
    for _, model, run in RUNS:
        r = results[run]
        assert r["info"]["model"] == model and r["info"]["backend"] == "numpy"
        assert not r["failed"].any()
        assert np.array_equal(r["lnlike"], stored[run]), run
    assert abs(results["exact_flow_seed3009"]["lnlike"][0] - (-600.5155231449987)) <= 5e-10


def test_wrap_phase_is_refused():
    theta, _ = _g3("gb_3mo", "exact_flow_seed3009")
    with pytest.raises(WorkerError, match="wrap_phase"):
        legacy_eval.evaluate("C", "galactic_binary", "gb_3mo", "exact", theta[:1], wrap_phase=True)
