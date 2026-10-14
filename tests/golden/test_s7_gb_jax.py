"""F-D5 (S7: G3 and G4 on CPU-JAX): ``registry.build("galactic_binary", ..., backend="jax")`` with
the legacy JAX evaluators reproduces

* G3: the stored lnlike of the fast-tier chain rows of every run directory of all 38 targets (58
  directories, 3 rows each) at the approved JAX gate (``approved.G3.jax``: 1e-8), every row finite
  and unflagged, and -- one fresh call per run directory, as lineage C did -- the counters lineage C
  recorded (``jax_counters``; the uncached ``BatchedLikelihood`` counts rows only);
* G4: the 11 stress points of the 30 targets at 3 and 6 months in ONE call (as the capture):
  ``-inf`` exactly where ``expected_neg_inf`` (PD failure or source outside the box) at non-marginal
  points; finite-or-``-inf`` with ``lnl_ref``'s mask at marginal points (|margin| < 1e-10); at
  well-conditioned points (margin >= 1e-6) ``|JAX - lnl_ref| <= 8.32e-8`` (``approved.G4``); at
  ill-conditioned points the condition-scaled delta is recorded, not gated; ``last_failed`` flags
  exactly the ``-inf`` points whose source is inside the box; the counters equal ``counters_jax``.

The values go to the bare JAX evaluator, never through ``sanitize`` (lineage C evaluated the bare
``jax_models.build``).  Gate arithmetic is this file's own (the rules of
``scripts/golden/measure_tolerances_gb.py``), independent of the code under test; NaN, ``+inf`` or
an empty comparison fail.  Every target records its ``s7`` junit property before asserting.
Default lane: gb_3mo and gb_6mo; gb_12mo is ``slow``.
"""

from __future__ import annotations

import gc
import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from scripts.golden.gb_jax_legacy import CACHE_CLASSES, LEGACY_JAX  # noqa: E402
from scripts.golden.paths import GOLDEN  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import build  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
GB = GOLDEN / "galactic_binary"
TOL = json.loads((GB / "tolerances_measured.json").read_text())
APPROVED = TOL["approved"]
EPS = 2.220446049250313e-16
GOLDEN_OPTIONS = ModelOptions(waveform_backend="fourier")
COUNTERS = ("evaluations", "noise_states", "pd_failed_rows", "pd_failed_states")


def _targets(cases=("gb_3mo", "gb_6mo", "gb_12mo")):
    return [pytest.param(c, m, id=f"{c}-{m}", marks=[pytest.mark.slow] if c == "gb_12mo" else [])
            for c in cases for m in LEGACY_JAX[c]]


def _counters(lik) -> dict:
    return {k: v for k, v in lik.counters().items() if k in COUNTERS}


def _uncached(n: int, failed: int = 0) -> dict:
    """``BatchedLikelihood(count_states=False)``: rows and PD-failed rows only."""
    return {"evaluations": n, "noise_states": 0, "pd_failed_rows": failed, "pd_failed_states": 0}


@pytest.mark.parametrize("case, model", _targets())
def test_g3_fast_tier_on_cpu_jax(case, model, record_property):
    root = GB / "chain_rows" / case
    runs = json.loads((root / "index.json").read_text())["targets"][model]
    gate = APPROVED["G3"]["jax"][f"{case}/{model}"]
    cached = LEGACY_JAX[case][model]["evaluator"] in CACHE_CLASSES
    lik = build("galactic_binary", case, model, backend="jax", options=GOLDEN_OPTIONS)
    results, worst, worst_legacy, n_rows = [], 0.0, 0.0, 0
    for run in runs:
        r = json.loads((root / f"{run}.fast.json").read_text())
        theta = np.array([row["theta"] for row in r["rows"]], dtype=float)
        stored = np.array([row["lnlike_stored"] for row in r["rows"]], dtype=float)
        legacy = np.array([row["lnlike_jax"] for row in r["rows"]], dtype=float)
        lik.reset_counters()                                   # lineage C: reset per run directory
        values = np.asarray(lik(theta), dtype=float)
        d = np.abs(values - stored)
        results.append({"run": run, "shape": theta.shape, "stored_finite": bool(np.all(np.isfinite(stored))),
                        "values": values, "d": d, "failed": lik.last_failed.tolist(),
                        "counters": _counters(lik), "want": r["jax_counters"] if cached else _uncached(3)})
        worst, worst_legacy = max(worst, float(d.max())), max(worst_legacy, float(np.abs(values - legacy).max()))
        n_rows += len(values)
    record_property("s7", json.dumps({"item": f"galactic_binary/{case}/{model}/jax/g3", "rows": n_rows,
                                      "max_abs": worst, "legacy_jax_max_abs": worst_legacy, "gate": gate}))
    assert results, (case, model)
    for x in results:
        assert x["shape"] == (3, lik.ndim) and x["stored_finite"], (x["run"], x["shape"])
        assert np.all(np.isfinite(x["values"])), (x["run"], x["values"])
        assert np.all(x["d"] <= gate), (x["run"], x["d"].max(), gate)
        assert x["failed"] == [False] * 3, (x["run"], x["failed"])
        assert x["want"] is not None and x["counters"] == x["want"], (x["run"], x["counters"], x["want"])
    del lik
    gc.collect()


def classify(points: list[dict], values: np.ndarray, threshold: float) -> dict:
    """``measure_tolerances_gb``'s G4 rules for one evaluator's values (module docstring)."""
    out = {"mismatches": [], "well_max_abs": 0.0, "n_well": 0, "cond_scaled_max": 0.0, "marginal": 0}
    for p, v in zip(points, values, strict=True):
        kind = "-inf" if v == -math.inf else ("finite" if math.isfinite(v) else "invalid")
        if kind == "invalid":
            out["mismatches"].append(f"{p['name']}: {v}")
            continue
        if p["marginal"]:
            assert abs(p["pd_margin"]) < threshold
            out["marginal"] += 1
            if (kind == "-inf") != (p["lnl_ref"] == -math.inf):
                out["mismatches"].append(f"{p['name']}: marginal -inf mask")
            continue
        expected = "-inf" if (p["source_out"] or p["pd_margin"] < 0) else "finite"
        assert expected == ("-inf" if p["expected_neg_inf"] else "finite")
        if kind != expected:
            out["mismatches"].append(f"{p['name']}: {kind}, expected {expected}")
            continue
        if kind == "-inf":
            continue
        d = abs(v - p["lnl_ref"])
        if p["pd_margin"] >= TOL["well_conditioned_margin"]:
            out["n_well"] += 1
            out["well_max_abs"] = max(out["well_max_abs"], d)
        else:
            out["cond_scaled_max"] = max(out["cond_scaled_max"], d / (EPS * abs(p["lnl_ref"]) / p["pd_margin"]))
    return out


@pytest.mark.parametrize("case, model", _targets(("gb_3mo", "gb_6mo")))
def test_g4_stress_fast_tier_on_cpu_jax(case, model, record_property):
    r = json.loads((GB / "stress" / case / f"{model}.fast.json").read_text())
    pts = r["points"]
    assert len(pts) == r["n_points"] == 11
    lik = build("galactic_binary", case, model, backend="jax", options=GOLDEN_OPTIONS)
    lik.reset_counters()
    values = np.asarray(lik(np.array([p["theta"] for p in pts], dtype=float)), dtype=float)
    c = classify(pts, values, float(r["marginal_threshold"]))
    gate = APPROVED["G4"]["jax_well_conditioned"]
    record_property("s7", json.dumps({"item": f"galactic_binary/{case}/{model}/jax/g4", "n_well": c["n_well"],
                                      "well_max_abs": c["well_max_abs"], "cond_scaled_max": c["cond_scaled_max"],
                                      "marginal": c["marginal"], "gate": gate}))
    assert c["mismatches"] == [] and c["n_well"] >= 1
    assert c["well_max_abs"] <= gate, (c["well_max_abs"], gate)
    src = np.array([p["source_out"] for p in pts])
    assert lik.last_failed.tolist() == ((values == -np.inf) & ~src).tolist()
    want = r["counters_jax"] if r["counters_jax"] is not None else _uncached(11)
    assert _counters(lik) == want, (_counters(lik), want)
    del lik
    gc.collect()


def test_gate_catches_a_wrong_target():
    """Negative control: gb_3mo's tri_raw at gb_6mo's tri_raw rows violates the G3 gate at every
    row, and gb_6mo's stress points through gb_3mo's model break the well-conditioned gate."""
    lik = build("galactic_binary", "gb_3mo", "tri_raw", backend="jax", options=GOLDEN_OPTIONS)
    root = GB / "chain_rows" / "gb_6mo"
    run = json.loads((root / "index.json").read_text())["targets"]["tri_raw"][0]
    r = json.loads((root / f"{run}.fast.json").read_text())
    values = np.asarray(lik(np.array([row["theta"] for row in r["rows"]])), dtype=float)
    stored = np.array([row["lnlike_stored"] for row in r["rows"]])
    assert np.all(np.abs(values - stored) > APPROVED["G3"]["jax"]["gb_6mo/tri_raw"])
    s = json.loads((GB / "stress" / "gb_6mo" / "tri_raw.fast.json").read_text())
    v = np.asarray(lik(np.array([p["theta"] for p in s["points"]])), dtype=float)
    c = classify(s["points"], v, float(s["marginal_threshold"]))
    assert c["mismatches"] or c["well_max_abs"] > APPROVED["G4"]["jax_well_conditioned"]
