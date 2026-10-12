"""B12: lineage C behind Section A's ``legacy_eval.evaluate(lineage, toy, case, model, theta)``.

    gb_legacy.py OUT_DIR evaluate --jobs JOBS.json --workspace DIR --data-dir DIR [--backend numpy|jax]

``JOBS.json`` is A6's job list (``legacy_eval.evaluate_jobs``): ``[{"tag", "case", "model",
"points", "wrap_phase"}]``, ``points`` an ``.npz`` holding ``theta`` (n, ndim). The worker writes
``<tag>.npz`` (``lnlike``, ``seconds``, ``theta``, ``failed``) per job and one ``evaluate.json``
(``{"lineage", "wdm_file", "runtime", "jobs": {tag: info}}``), the files ``evaluate_jobs`` reads.

* ``numpy`` (default): ``likelihoods.make_lnlike(Prepared(path), model, numpy_backend())``, one
  point at a time; ``LinAlgError`` gives ``-inf`` with ``failed = True`` (as G3/G4 record it);
* ``jax``: ``jax_models.build(Prepared(path), model)`` on CPU, all points of a job in one call.

The GB likelihoods fold the periodic source angles themselves (``likelihoods.wrap_source``), so
the sinusoid runner's ``wrap_phase`` is refused rather than silently ignored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c

EVALUATORS = {"numpy": "likelihoods.make_lnlike(Prepared(path), model, numpy_backend())",
              "jax": "gpu.jax_models.build(Prepared(path), model)  (CPU, x64)"}


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR evaluate")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=tuple(EVALUATORS), default="numpy")
    return ap.parse_args(argv)


def run(out_dir, argv) -> None:
    a = _parse(argv)
    jobs = json.loads(a.jobs.read_text())
    for job in jobs:
        if job.get("wrap_phase", False):
            raise SystemExit(f"job {job['tag']}: wrap_phase is the sinusoid runner's convention; the GB "
                             "likelihoods wrap their own periodic angles (likelihoods.wrap_source)")
    L = gbw_lineage_c.load(a.workspace, jax=a.backend == "jax")
    out_dir = Path(out_dir)
    preps, current, f, build_seconds = {}, None, None, 0.0
    info = {}
    for job in jobs:
        tag, case, model = job["tag"], job["case"], job["model"]
        if case not in preps:
            preps[case] = L.likelihoods.Prepared(
                a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{case}.npz")
        prep = preps[case]
        if (case, model) != current:
            f = None                                       # release the previous model first
            t0 = time.perf_counter()
            f = (L.likelihoods.make_lnlike(prep, model, L.likelihoods.numpy_backend())
                 if a.backend == "numpy" else L.jax_models.build(prep, model))
            build_seconds, current = time.perf_counter() - t0, (case, model)
        with np.load(job["points"], allow_pickle=False) as z:
            theta = np.atleast_2d(np.asarray(z["theta"], dtype=np.float64))
        if theta.shape[1] != prep.ndim(model):
            raise SystemExit(f"job {tag}: theta has {theta.shape[1]} columns; {model} needs {prep.ndim(model)}")
        values = np.empty(len(theta))
        seconds = np.empty(len(theta))
        failed = np.zeros(len(theta), dtype=bool)
        if a.backend == "numpy":
            for i, point in enumerate(theta):
                t0 = time.perf_counter()
                try:
                    values[i] = float(f(point))
                except np.linalg.LinAlgError:
                    values[i], failed[i] = -np.inf, True
                seconds[i] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            values[:] = np.asarray(f(theta), dtype=np.float64)
            seconds[:] = (time.perf_counter() - t0) / len(theta)
        np.savez(out_dir / f"{tag}.npz", lnlike=values, seconds=seconds, theta=theta, failed=failed)
        info[tag] = {"case": case, "model": model, "n": int(len(theta)), "backend": a.backend,
                     "evaluator": EVALUATORS[a.backend], "build_seconds": build_seconds,
                     "n_failed": int(failed.sum())}
        print(f"[evaluate] {tag}: {case}/{model} n={len(theta)} backend={a.backend}", flush=True)
    (out_dir / "evaluate.json").write_text(json.dumps(
        {"lineage": "C", "wdm_file": None, "runtime": gbw_common.build_info(), "jobs": info},
        indent=1, sort_keys=True) + "\n")
