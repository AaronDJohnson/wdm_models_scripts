"""S7: the legacy GB JAX evaluators (lineage C) -- what ``gpu/jax_models.build`` ran the chains with.

``LEGACY_JAX[case][model]`` records, per target, the evaluator class, whether the rows of one call
go through one vmapped graph, the LRU capacity and the bytes of one noise state at the anchor
(measured on CPU-JAX 2026-10-13 by ``legacy_states``: lineage C, ``workers/gbw_states.py``).
The release's historical JAX column (``options.resolve``) and its per-state memory are gated
against this table; ``legacy_states(case)`` re-measures it (a ``legacy`` test pins the two equal).

    from scripts.golden.gb_jax_legacy import LEGACY_JAX, legacy_states
    legacy_states("gb_3mo")          # one lineage-C subprocess, ~7 s
"""

from __future__ import annotations

from scripts.golden.goldens_io import read_json
from scripts.golden.lineage import run_worker
from scripts.golden.paths import DATA, ROOT

__all__ = ["CACHE_CLASSES", "LEGACY_JAX", "legacy_states"]

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
