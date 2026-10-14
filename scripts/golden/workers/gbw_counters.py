"""S7: a scripted call sequence through a legacy GB JAX cache class (lineage C, CPU-JAX).

    gb_legacy.py OUT_DIR counters --workspace DIR --data-dir DIR --case CASE --model MODEL --batches B.npz

``gpu.jax_models.build(Prepared(path), model)`` -- ``MatrixCache`` / ``EdgeSchurCache`` /
``FullCache`` -- is built ONCE and called on the arrays of ``B.npz`` in the order of their names
(``b0``, ``b1``, ...), after a ``reset_counters()``.  After every call the values and the four
counters (``fullcache.COUNTERS``) are recorded: ``OUT_DIR/counters.json`` =
``{"schema", "case", "model", "class", "capacity", "calls": [{"name", "values", "counters"}], "build"}``
(values as strict-JSON floats, ``"-inf"`` for -inf).  Nothing else is written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR counters")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", choices=gbw_lineage_c.CASES, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", type=Path, required=True)
    return ap.parse_args(argv)


def _f(v: float):
    return v if math.isfinite(v) else ("-inf" if v == -math.inf else ("inf" if v > 0 else "nan"))


def run(out_dir, argv) -> None:
    a = _parse(argv)
    L = gbw_lineage_c.load(a.workspace, jax=True)
    prep = L.likelihoods.Prepared(a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{a.case}.npz")
    J = L.jax_models.build(prep, a.model)
    if not hasattr(J, "counters"):
        raise SystemExit(f"{a.case}/{a.model}: {type(J).__name__} keeps no counters")
    J.reset_counters()
    with np.load(a.batches, allow_pickle=False) as z:
        names = sorted(z.files, key=lambda k: int(k[1:]))
        batches = {k: np.asarray(z[k], dtype=np.float64) for k in names}
    calls = []
    for name in names:
        values = np.asarray(J(batches[name]), dtype=np.float64)
        calls.append({"name": name, "values": [_f(float(v)) for v in values],
                      "counters": {k: int(v) for k, v in J.counters().items()}})
        print(f"[counters] {a.case}/{a.model} {name}: {calls[-1]['counters']}", flush=True)
    (Path(out_dir) / "counters.json").write_text(json.dumps(
        {"schema": 1, "case": a.case, "model": a.model, "class": type(J).__name__,
         "capacity": int(J.capacity), "calls": calls, "build": gbw_common.build_info()}, indent=1) + "\n")
