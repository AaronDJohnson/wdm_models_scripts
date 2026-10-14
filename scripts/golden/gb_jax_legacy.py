"""S7: the legacy GB JAX evaluators (lineage C) -- what ``gpu/jax_models.build`` ran the chains with.

``LEGACY_JAX[case][model]`` records, per target, the evaluator class, whether the rows of one call
go through one vmapped graph, the LRU capacity and the bytes of one noise state at the anchor
(measured on CPU-JAX 2026-10-13 by ``legacy_states``: lineage C, ``workers/gbw_states.py``).
The release's historical JAX column (``options.resolve``) and its per-state memory are gated
against this table; ``legacy_states(case)`` re-measures it (a ``legacy`` test pins the two equal).

    from scripts.golden.gb_jax_legacy import LEGACY_JAX, legacy_states
    legacy_states("gb_3mo")          # one lineage-C subprocess, ~7 s

``scripted_batches`` / ``legacy_counters`` (F-D6) run a fixed call sequence through lineage C's
cache class (``workers/gbw_counters.py``) so the release's counters can be compared call by call.
"""

from __future__ import annotations

from scripts.golden.goldens_io import read_json
from scripts.golden.lineage import run_worker
from scripts.golden.paths import DATA, ROOT

__all__ = ["CACHE_CLASSES", "COUNTER_KEYS", "LEGACY_JAX", "legacy_counters", "legacy_states", "scripted_batches"]

#: The legacy cache classes (``matrix.MatrixCache``, ``matrix.EdgeSchurCache``, ``fullcache.FullCache``).
CACHE_CLASSES = ("MatrixCache", "EdgeSchurCache", "FullCache")


def _row(evaluator: str, state_bytes: int | None, *, batched: bool = True) -> dict:
    cached = evaluator in CACHE_CLASSES
    return {"evaluator": evaluator, "batched": batched, "capacity": 24 if cached else None,
            "state_bytes": state_bytes}


def _case(matrix: dict[str, int], schur: int, full: int, uncached: tuple[str, ...]) -> dict[str, dict]:
    out = {m: _row("MatrixCache", b) for m, b in matrix.items()}
    out.update({m: _row("EdgeSchurCache", schur) for m in ("band_exact", "mid_exact") if schur})
    out["exact"] = _row("FullCache", full, batched=False)
    out.update({m: _row("BatchedLikelihood", None) for m in uncached})
    return dict(sorted(out.items()))


_UNCACHED = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "mid_raw", "mid_cut")
#: case -> model -> {"evaluator", "batched", "capacity", "state_bytes"} (lineage C, CPU-JAX).
LEGACY_JAX: dict[str, dict[str, dict]] = {
    "gb_3mo": _case({"tri_raw": 8305952, "tri_cut": 7695104, "tri_pad": 8305952, "tri_exact": 8305952,
                     "band_pad": 8305952, "mid_pad": 8305952}, 333680, 4158752, _UNCACHED),
    "gb_6mo": _case({"tri_raw": 27646592, "tri_cut": 26522528, "tri_pad": 27646592, "tri_exact": 27646592,
                     "band_pad": 27646592, "mid_pad": 27646592}, 609296, 13833824, _UNCACHED),
    "gb_12mo": _case({"tri_raw": 116683232, "tri_pad": 116683232, "band_pad": 116683232,
                      "mid_pad": 116683232}, 0, 38923232, ("whittle_raw", "band_raw", "mid_raw")),
}


def legacy_states(case: str, models: tuple[str, ...] = ()) -> dict[str, dict]:
    """Lineage C's ``gpu.jax_models.build`` per model of ``case`` (every target by default):
    ``{model: {"evaluator", "batched", "capacity", "state_bytes"}}`` (one subprocess)."""
    args = ["--workspace", str(ROOT), "--data-dir", str(DATA), "--case", case]
    if models:
        args += ["--models", ",".join(models)]
    out = run_worker("C", "gb_legacy", "states", *args, timeout=1800.0)
    return read_json(out / "states.json")["models"]


# --- a scripted call sequence (F-D6) --------------------------------------------------------------

#: Counters both the legacy caches and the release keep (``fullcache.COUNTERS``).
COUNTER_KEYS = ("noise_states", "pd_failed_states", "pd_failed_rows", "evaluations")


def scripted_batches(case: str, model: str) -> dict[str, "np.ndarray"]:
    """Three batches exercising every counter path of the legacy LRU (``fullcache.NoiseStateCache``):

    * ``b0`` (8 rows): the anchor, a source move (same noise state: a hit), a new noise state, the
      anchor again, the first stress point whose covariance failed in lineage C (``lnl_ref = -inf``
      with the source inside the box; when the target has none at this case -- ``exact`` at 3
      months -- another new state) and its repeat (a cached failure), the anchor's
      noise with ``lnA`` above the box (``source_ok``: -inf, no PD failure), a second source move;
    * ``b1`` (26 rows): 26 fresh noise states -- past the capacity of 24, so the first states are
      evicted;
    * ``b2`` (6 rows): the anchor and the failing point again (evicted: factorised and counted
      again), the newest fresh state (a hit), the oldest fresh state (evicted), the anchor.

    Built from the anchor (``inject_glass`` + ``mu_fit``) and ``golden/galactic_binary/stress``."""
    import numpy as np

    from scripts.golden.paths import GOLDEN

    stress = read_json(GOLDEN / "galactic_binary" / "stress" / case / f"{model}.fast.json")
    pts = {p["name"]: p for p in stress["points"]}
    x = np.asarray(pts["anchor"]["theta"], dtype=float)
    src1, src2 = x.copy(), x.copy()
    src1[0] += 0.05
    src2[3] += 0.01
    new = x.copy()
    new[8] += 0.01
    fails = [p for p in stress["points"] if p["lnl_ref"] == -np.inf and not p["source_out"]]   # PD failures
    if fails:
        bad = np.asarray(fails[0]["theta"], dtype=float)
    else:
        bad = x.copy()
        bad[13] -= 0.02
    out_lnA = np.asarray(pts["source_out_lnA"]["theta"], dtype=float)
    fresh = []
    for k in range(26):
        f = x.copy()
        f[9] += 0.002 * (k + 1)
        fresh.append(f)
    return {"b0": np.stack([x, src1, new, x, bad, bad, out_lnA, src2]),
            "b1": np.stack(fresh),
            "b2": np.stack([x, bad, fresh[25], fresh[0], src1, x])}


def legacy_counters(case: str, model: str, batches: dict) -> dict:
    """Lineage C's cache class on ``batches`` (in name order, one object): ``{"class", "capacity",
    "calls": [{"name", "values", "counters"}]}`` with the values as floats (one subprocess)."""
    import uuid

    import numpy as np

    from scripts.golden.goldens_io import to_float
    from scripts.golden.paths import tmp_root

    stage = tmp_root() / "jobs" / uuid.uuid4().hex[:12]
    stage.mkdir(parents=True)
    path = stage / "batches.npz"
    np.savez(path, **batches)
    out = run_worker("C", "gb_legacy", "counters", "--workspace", str(ROOT), "--data-dir", str(DATA),
                     "--case", case, "--model", model, "--batches", str(path), timeout=1800.0)
    rep = read_json(out / "counters.json")
    for call in rep["calls"]:
        call["values"] = [to_float(v) for v in call["values"]]
    return rep
