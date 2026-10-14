"""S7: the legacy GB JAX evaluators (lineage C, CPU-JAX) -- class, row batching, state memory.

    gb_legacy.py OUT_DIR states --workspace DIR --data-dir DIR --case CASE [--models M1,M2,...]

For each model of the case (default: every target in the final-chains manifest),
``gpu.jax_models.build(Prepared(path), model)`` -- the evaluator that sampled the chains --
is built and recorded:

* ``evaluator``: its class (``MatrixCache`` / ``EdgeSchurCache`` / ``FullCache`` /
  ``BatchedLikelihood``);
* ``batched``: whether the rows of one call go through one vmapped graph
  (``jax_models.BatchedLikelihood``, or a cache class with ``_batch``: ``fullcache.batch_signal``);
* ``capacity``: the LRU size of a cache class (``None`` otherwise);
* ``state_bytes``: for a cache class, ONE noise state (both channels) at the anchor
  (``Prepared.anchor(model)[8:]``) through its own ``state(mu_flat)``: the sum of ``nbytes`` over
  the state's array leaves -- what one LRU slot holds (``None`` for ``BatchedLikelihood``).

Writes ``OUT_DIR/states.json`` (``{"schema", "case", "models": {model: {...}}, "build"}``);
nothing else is written (``gb_legacy.py``'s write guard).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR states")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", choices=gbw_lineage_c.CASES, required=True)
    ap.add_argument("--models", default="", help="comma list; default every target of the case")
    return ap.parse_args(argv)


def run(out_dir, argv) -> None:
    a = _parse(argv)
    L = gbw_lineage_c.load(a.workspace, jax=True)
    import jax

    manifest = json.loads((L.workspace / gbw_lineage_c.CHAINS / "manifest.json").read_text())
    models = sorted(gbw_lineage_c.targets(manifest, a.case))
    models = [m for m in models if not a.models or m in a.models.split(",")]
    if not models:
        raise SystemExit(f"{a.case}: no model selected")
    prep = L.likelihoods.Prepared(a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{a.case}.npz")
    out = {}
    for model in models:
        J = L.jax_models.build(prep, model)
        rec = {"evaluator": type(J).__name__, "capacity": getattr(J, "capacity", None),
               "batched": isinstance(J, L.jax_models.BatchedLikelihood) or getattr(J, "_batch", None) is not None,
               "state_bytes": None}
        if hasattr(J, "state"):
            st = J.state(np.asarray(prep.anchor(model)[8:], dtype=np.float64))
            if st is None:
                raise SystemExit(f"{a.case}/{model}: the anchor's noise state failed")
            rec["state_bytes"] = int(sum(int(x.nbytes) for x in jax.tree_util.tree_leaves(st)
                                         if hasattr(x, "nbytes")))
        out[model] = rec
        print(f"[states] {a.case}/{model}: {rec}", flush=True)
        del J
        gc.collect()
    (Path(out_dir) / "states.json").write_text(json.dumps(
        {"schema": 1, "case": a.case, "models": out, "build": gbw_common.build_info()}, indent=1,
        sort_keys=True) + "\n")
