"""E-D6: the noise-state counters of a scripted call sequence, release vs WG (lineage B).

The batches exercise every counter path of WG ``MatrixGPU._cached_rows``
(``gpu_sinusoid/wdmtoysgpu/matrix.py:726-789``) on one likelihood object: misses, hits
inside one call, a failed (not positive definite) state and its cached repeats, LRU
eviction past 16 states, re-proposal of evicted states, and the alpha_d clamp folding
4.5 and 5.0 onto the key of 4.0 (and -7 onto -4).  Built deterministically from the G1
reference file of the target (its truth, points 1 and 2, and point 4, the first -inf
point of the edge-exact models).

    batches = scripted_batches("manuscript_mid", "band_exact")
    wg = wg_counters("manuscript_mid", "band_exact", batches)   # one lineage-B subprocess
"""

from __future__ import annotations

import uuid

import numpy as np

from scripts.golden.goldens_io import read_json, read_npz, to_float
from scripts.golden.lineage import run_worker
from scripts.golden.paths import DATA, GOLDEN, tmp_root

#: WG ``MatrixGPU``'s default LRU size (``matrix.py:196``): the comparison runs the release at it.
WG_CACHE_SIZE = 16
COUNTER_KEYS = ("noise_states", "pd_failed_states", "pd_failed_rows", "evaluations", "cache_hits")


def scripted_batches(case: str, model: str) -> dict[str, np.ndarray]:
    """``{"b0": (8, 6), "b1": (20, 6), "b2": (8, 6)}`` (drift parameter layout)."""
    if model.startswith("whittle") or model == "exact":
        raise ValueError(f"{model}: WG kept no noise-state counters for this model")
    ref = read_npz(GOLDEN / "sinusoid" / "references" / f"{case}_{model}_reference.npz")
    th = np.asarray(ref["theta"], dtype=float)
    truth = th[0]

    def row(alpha_c: float, alpha_d: float, src: np.ndarray | None = None) -> np.ndarray:
        r = truth.copy()
        if src is not None:
            r[:3] = src
        r[4], r[5] = alpha_c, alpha_d
        return r

    fresh = [row(truth[4] + 0.01 * k, truth[5] - 0.01 * k) for k in range(20)]
    return {
        # 4 misses (truth, point 1, point 4, point 2), 4 hits; point 4 is -inf for edge-exact
        "b0": np.array([truth, row(truth[4], truth[5], th[1][:3]), th[1], th[1], th[4], th[2], truth, th[4]]),
        # 19 new states + 1 hit (fresh[0] is the truth's noise): evicts past 16
        "b1": np.array(fresh),
        # evicted states again, one surviving hit, the clamp (4.5 and 5.0 -> 4.0, -7 -> -4)
        "b2": np.array([truth, th[1], th[2], fresh[19], fresh[0], row(truth[4], 4.5), row(truth[4], 5.0),
                        row(truth[4], -7.0)]),
    }


def wg_counters(case: str, model: str, batches: dict[str, np.ndarray]) -> dict:
    """WG's ``lnlike`` on ``batches`` (in order, one object): ``{"calls": [{"values",
    "failed", "counters"}], "class", "cache_size"}`` with the values as floats."""
    stage = tmp_root() / "jobs" / uuid.uuid4().hex[:12]
    stage.mkdir(parents=True)
    path = stage / "batches.npz"
    np.savez(path, **batches)
    out = run_worker("B", "sinusoid_wg", "counters", "--case", case, "--model", model, "--batches", str(path),
                     "--data-root", str(DATA / "sinusoid"), "--cache-src", str(DATA / "sinusoid" / "cache"),
                     timeout=3600.0)
    rep = read_json(out / "counters.json")
    for call in rep["calls"]:
        call["values"] = [to_float(v) for v in call["values"]]
    return rep
