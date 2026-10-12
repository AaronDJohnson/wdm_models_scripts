"""G3: re-evaluate stored GB chain rows (``lnlike`` column) with both lineage-C evaluators.

* ``numpy``: ``likelihoods.make_lnlike(Prepared, model, numpy_backend())``, the reference;
* ``jax``: ``gpu.jax_models.build(Prepared, model)`` on CPU, the class that sampled the chains.

Rows: ``gbw_common.select_rows`` (fast: first, middle, last; full: + 29 seeded rows).
One JSON per run directory: ``galactic_binary/chain_rows/<case>/<run>.<tier>.json``, plus
``galactic_binary/chain_rows/<case>/index.json`` (target -> run directories).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c

EVALUATORS = {"numpy": "likelihoods.make_lnlike(Prepared(path), model, numpy_backend())",
              "jax": "gpu.jax_models.build(Prepared(path), model)  (CPU, x64)"}


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR rows")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", choices=gbw_lineage_c.CASES, required=True)
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    ap.add_argument("--backends", default="numpy,jax")
    ap.add_argument("--models", default="", help="comma list; default every target of the case")
    return ap.parse_args(argv)


def _delta(a: float, b: float) -> float:
    if np.isneginf(a) and np.isneginf(b):
        return 0.0
    return float(abs(a - b))


def run(out_dir, argv) -> None:
    a = _parse(argv)
    backends = [b for b in a.backends.split(",") if b]
    if not backends or not set(backends) <= {"numpy", "jax"}:
        raise SystemExit(f"--backends must be a subset of numpy,jax; got {a.backends!r}")
    L = gbw_lineage_c.load(a.workspace, jax="jax" in backends)
    pay = gbw_common.Payload(out_dir, lineage="C", worker="gb_legacy", subcommand="rows", args=argv,
                             base=L.workspace)
    root = L.workspace / gbw_lineage_c.CHAINS
    pay.add_input(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    prep_path = a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{a.case}.npz"
    prep_sha = pay.add_input(prep_path)
    prep = L.likelihoods.Prepared(prep_path)
    targets = gbw_lineage_c.targets(manifest, a.case)
    pay.write_json(f"galactic_binary/chain_rows/{a.case}/index.json",
                   {"schema": 1, "case": a.case, "targets": targets,
                    "n_runs": sum(len(r) for r in targets.values())})
    wanted = [m for m in targets if not a.models or m in a.models.split(",")]
    for model in wanted:
        names = list(L.config.parameter_names(model))
        nd = len(names)
        if nd != prep.ndim(model):
            raise RuntimeError(f"{model}: {nd} parameter names but Prepared.ndim = {prep.ndim(model)}")
        f_np = (L.likelihoods.make_lnlike(prep, model, L.likelihoods.numpy_backend())
                if "numpy" in backends else None)
        f_jax = L.jax_models.build(prep, model) if "jax" in backends else None
        for run_name in targets[model]:
            d = root / "chains" / a.case / run_name
            meta = json.loads((d / "run_meta.json").read_text())
            if (meta["model"], meta["case"], list(meta["parameter_names"])) != (model, a.case, names):
                raise RuntimeError(f"{d}: run_meta does not describe {a.case}/{model}")
            width = nd + 4
            chain = d / "chain_0.bin"
            n = gbw_common.chain_length(chain, width)
            idx = gbw_common.select_rows(n, a.tier)
            R = gbw_common.read_rows(chain, width, idx)
            theta, stored = R[:, :nd], R[:, nd]
            vn = np.full(len(idx), np.nan)
            raised = [False] * len(idx)
            if f_np is not None:
                for i, th in enumerate(theta):
                    try:
                        vn[i] = float(f_np(th))
                    except np.linalg.LinAlgError:
                        vn[i], raised[i] = -np.inf, True
            vj, counters = None, None
            if f_jax is not None:
                if hasattr(f_jax, "reset_counters"):
                    f_jax.reset_counters()
                vj = np.asarray(f_jax(theta), dtype=np.float64)
                counters = f_jax.counters() if hasattr(f_jax, "counters") else None
            rows = []
            for i, r in enumerate(idx):
                rec = {"row": r, "theta": theta[i].tolist(), "lnlike_stored": float(stored[i])}
                if f_np is not None:
                    rec.update(lnlike_numpy=float(vn[i]), numpy_raised=raised[i],
                               abs_delta_numpy=_delta(vn[i], stored[i]))
                if vj is not None:
                    rec.update(lnlike_jax=float(vj[i]), abs_delta_jax=_delta(vj[i], stored[i]))
                rows.append(rec)
            out = {"schema": 1, "case": a.case, "model": model, "run": run_name, "tier": a.tier,
                   "parameter_names": names, "ndim": nd, "width": width, "rows_total": n,
                   "row_indices": idx, "lnlike_column": nd,
                   "chain_file": {"path": str(chain.relative_to(L.workspace)),
                                  "bytes": chain.stat().st_size},
                   "rows": rows, "backends": backends,
                   "evaluators": {b: EVALUATORS[b] for b in backends},
                   "prepared_sha256": prep_sha, "jax_counters": counters,
                   "build": gbw_common.build_info()}
            for b in backends:
                out[f"max_abs_delta_{b}"] = max(rec[f"abs_delta_{b}"] for rec in rows)
            pay.write_json(f"galactic_binary/chain_rows/{a.case}/{run_name}.{a.tier}.json", out)
            print(f"[rows] {a.case}/{run_name}: n={n} rows={idx} "
                  + " ".join(f"{b}={out[f'max_abs_delta_{b}']:.2e}" for b in backends), flush=True)
        del f_np, f_jax
        gc.collect()
    pay.close()
