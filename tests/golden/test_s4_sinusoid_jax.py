"""E-D5 (S4: G1 and G2 on CPU-JAX): ``registry.build(..., backend="jax")`` with the
historical JAX strategies (gram + padded Eq. 45, whitened tri marginal, the reduce pad
marginal, the scaled Schur complement over a diagonal interior, adaptive staged exact rows,
batched PCG with the table bandcov correction, the alpha_d clamp) reproduces

* G1: the fast-tier reference points of all 45 targets at the approved JAX gates
  (``approved.jax.references``: near atol 1e-5 / exact 1e-4; stress + 1e-8 |lnL|), with
  -inf exactly where the reference is -inf and those rows flagged in ``last_failed``;
* G2: the fast-tier stored chain rows (``approved.jax.chain_rows``: atol 1e-5 / exact 1e-4),
  phase-wrapped through the runner's prior sanitising.

G1 points go to the bare likelihood, G2 rows through ``sanitize`` -- exactly how lineages A
and B evaluated them (``PriorSanitized.wrap`` rounds phi_0 even inside [-pi, pi)).

Default lane: manuscript_mid, every model but the edge-exact ones; ``slow``: the other two
cases and the nine edge-exact targets (an edge-exact target holds its Phi, ~2.8 GB).  Gate
arithmetic is the harness's own (``scripts.golden.sinusoid_goldens.compare``,
``measure_tolerances.violations``), independent of the code under test; every check fails
loudly on NaN, +inf or an empty set.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S4 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from scripts.golden.goldens_io import read_json, read_npz  # noqa: E402
from scripts.golden.measure_tolerances import violations  # noqa: E402
from scripts.golden.paths import GOLDEN  # noqa: E402
from scripts.golden.sinusoid_goldens import CASES, CHAIN_ROWS, MODELS, REFS, compare  # noqa: E402
from wdm_models.registry import build  # noqa: E402
from wdm_models.sampling.prior_sanitize import sanitize  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
APPROVED = read_json(GOLDEN / "sinusoid" / "tolerances_measured.json")["approved"]["jax"]
G1 = read_json(REFS / "INDEX.json")["files"]
G2_FAST = read_json(CHAIN_ROWS / "INDEX.json")["fast_positions"]
EDGE = ("band_exact", "mid_exact", "tri_exact")
#: G1 fast tier: one -inf point (index 4) per edge-exact target, none elsewhere.
FAST_INF = {m: 1 if m in EDGE else 0 for m in MODELS}


def _targets():
    out = []
    for case in CASES:
        for model in MODELS:
            slow = case != "manuscript_mid" or model in EDGE
            out.append(pytest.param(case, model, id=f"{case}-{model}",
                                    marks=[pytest.mark.slow] if slow else []))
    return out


def evaluate(case: str, model: str) -> dict:
    """G1 fast points and G2 fast rows of one target through sanitize(build(..., "jax"))."""
    lik = build("sinusoid", case, model, backend="jax")
    s = sanitize(lik, "sinusoid", case, model)
    name = f"{case}_{model}_reference.npz"
    ref = read_npz(REFS / name)
    idx = np.asarray(G1[name]["fast_index"], dtype=int)
    values, failed = [], []
    for row in ref["theta"][idx]:          # bare, as the lineage workers (the wrap rounds phi_0)
        values.append(float(lik(row[None, :])[0]))
        failed.append(bool(np.asarray(lik.last_failed)[0]))
    rows = read_npz(CHAIN_ROWS / f"{case}_{model}.npz")
    pos = np.asarray(G2_FAST, dtype=int)
    row_values = np.asarray(s(rows["theta"][pos]), dtype=float)
    out = {"g1": (np.asarray(values), ref["loglike"][idx], ref["near"][idx], np.asarray(failed)),
           "g2": (row_values, rows["lnlike"][pos]), "counters": lik.counters()}
    del s, lik
    gc.collect()
    return out


@pytest.mark.parametrize("case, model", _targets())
def test_g1_and_g2_fast_tier_on_cpu_jax(case, model, record_property):
    out = evaluate(case, model)
    values, reference, near, failed = out["g1"]
    assert values.size == reference.size == 4                       # never an empty comparison
    cmp = compare(values, reference, near)                           # raises on NaN / +inf
    tag = f"{case}__{model}"
    v1 = violations({tag: {"model": model, "value": values.tolist(), "reference": reference.tolist(),
                           "abs_delta": cmp["abs_delta"].tolist(), "near": near.tolist()}},
                    APPROVED["references"])
    row_values, row_ref = out["g2"]
    assert row_values.size == row_ref.size == len(G2_FAST) == 3
    rcmp = compare(row_values, row_ref, np.ones(row_ref.size, bool))
    v2 = violations({tag: {"model": model, "value": row_values.tolist(), "reference": row_ref.tolist(),
                           "abs_delta": rcmp["abs_delta"].tolist()}}, APPROVED["chain_rows"])
    finite = np.isfinite(reference)
    rel = np.where(finite, cmp["abs_delta"] / np.maximum(np.abs(np.where(finite, reference, 1.0)), 1e-300), 0.0)
    record_property("s4", json.dumps({"item": f"sinusoid/{case}/{model}/jax/g1",
                                      "near_max_abs": cmp["near_max_abs"],
                                      "stress_max_rel": float(rel[~near].max(initial=0.0)),
                                      "g2_max_abs": float(rcmp["abs_delta"].max()),
                                      "inf_points": int(np.sum(np.isneginf(reference))),
                                      "violations": len(v1) + len(v2)}))
    assert not v1 and not v2, v1 + v2
    assert int(np.sum(np.isneginf(values))) == FAST_INF[model]
    assert failed.tolist() == np.isneginf(values).tolist()          # every -inf is a flagged PD failure
    assert cmp["inf_mismatch"] == 0 and rcmp["inf_mismatch"] == 0


def test_gate_catches_a_wrong_target():
    """Negative control: manuscript_mid's band_raw evaluated at manuscript_low's G1 points
    violates every gate (the harness fails loudly on a wrong likelihood)."""
    lik = build("sinusoid", "manuscript_mid", "band_raw", backend="jax")
    name = "manuscript_low_band_raw_reference.npz"
    ref = read_npz(REFS / name)
    idx = np.asarray(G1[name]["fast_index"], dtype=int)
    values = np.array([float(lik(row[None, :])[0]) for row in ref["theta"][idx]])
    cmp = compare(values, ref["loglike"][idx], ref["near"][idx])
    v = violations({"wrong": {"model": "band_raw", "value": values.tolist(),
                              "reference": ref["loglike"][idx].tolist(), "abs_delta": cmp["abs_delta"].tolist(),
                              "near": ref["near"][idx].tolist()}}, APPROVED["references"])
    assert len(v) == 4, v
