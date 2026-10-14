"""Lineage-B worker: the JAX backends (wdmtoysgpu) on CPU-JAX, the code every final
sinusoid chain evaluated (spec §7.1; Task A7).

    $PY -P sinusoid_wg.py <out_dir> evaluate --jobs <jobs.json> --data-root <dir> --cache-src <dir>
    $PY -P sinusoid_wg.py <out_dir> counters --case C --model M --batches <b.npz> --data-root <dir> --cache-src <dir>

Construction is toy_models/sinusoid/gpu/run.py::build_backend for the matrix models:
``import wdmtoysgpu`` (enables x64) then build_matrix(model, cfg, d, analysis_fmin=4.0,
**({P: 24, tol: 1e-7, cache_dir} if exact)). ``evaluate``: each point is one batch of one
row; ``last_failed`` is recorded per row. ``counters`` (E-D6): the batches ``b0, b1, ...``
of an .npz go through ONE likelihood object in order (its LRU carries over); the counters,
the values and ``last_failed`` are recorded after every call. PYTHONPATH = lineage B only.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


def _load_common():
    path = Path(__file__).resolve().with_name("_common.py")
    spec = importlib.util.spec_from_file_location("golden_worker_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load_common()
ANALYSIS_FMIN = 4.0                    # gpu/run.py::_ANALYSIS_FMIN
FULL_ORDER, FULL_TOL = 24, 1e-7        # gpu/run.py::_FULL_ORDER, _FULL_TOL
COUNTERS = ("noise_states", "pd_failed_states", "pd_failed_rows", "evaluations", "cache_hits")
COMMANDS: dict[str, Callable[[Path, list[str]], None]] = {}


def command(name: str):
    def register(fn):
        COMMANDS[name] = fn
        return fn
    return register


@command("evaluate")
def cmd_evaluate(out_dir: Path, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="sinusoid_wg evaluate")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache-src", type=Path, required=True)
    a = ap.parse_args(argv)
    import wdmtoysgpu  # noqa: F401  (enables x64 before anything touches jax)
    import jax
    from wdmtoysgpu.matrix import build_matrix
    from wdm_toys.config import CONFIGS
    from wdm_toys.data import load
    C.check_wdm(required=True)
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit(f"lineage B runs on CPU-JAX with x64; got {jax.default_backend()}, "
                         f"x64={jax.config.jax_enable_x64}")
    cache = out_dir / "cache"
    cache_sha = C.copy_cache(a.cache_src, cache)
    current, lik, build_seconds = None, None, 0.0
    jobs_out = {}
    for job in C.read_json(a.jobs):
        case, model, tag = job["case"], job["model"], job["tag"]
        if (case, model) != current:
            lik = None
            cfg = CONFIGS[case]()
            data = np.asarray(load(str(a.data_root / case / "toy_data.npz"), cfg)["d"], dtype=float)
            kw = dict(P=FULL_ORDER, tol=FULL_TOL, cache_dir=str(cache)) if model == "exact" else {}
            t0 = time.perf_counter()
            lik = build_matrix(model, cfg, data, analysis_fmin=ANALYSIS_FMIN, **kw)
            build_seconds, current = time.perf_counter() - t0, (case, model)
        with np.load(job["points"], allow_pickle=False) as z:
            theta = np.asarray(z["theta"], dtype=float)
        if job.get("wrap_phase", False):
            theta = C.wrap_phase_runner(theta)
        values = np.empty(len(theta))
        failed = np.zeros(len(theta), dtype=bool)
        seconds = np.empty(len(theta))
        for i, point in enumerate(theta):
            t0 = time.perf_counter()
            values[i] = np.asarray(lik.lnlike(point[None, :]), dtype=float)[0]
            seconds[i] = time.perf_counter() - t0
            last = getattr(lik, "last_failed", None)
            failed[i] = bool(np.asarray(last)[0]) if last is not None else False
        np.savez(out_dir / f"{tag}.npz", lnlike=values, failed=failed, seconds=seconds, theta=theta)
        jobs_out[tag] = {"case": case, "model": model, "n": int(len(theta)), "build_seconds": build_seconds,
                         "class": type(lik).__name__,
                         "counters": {k: int(getattr(lik, k)) for k in COUNTERS if hasattr(lik, k)}}
    C.assert_cache_unchanged(cache, cache_sha)
    C.write_json(out_dir / "evaluate.json", {
        "wdm_file": C.check_wdm(required=True), "runtime": C.runtime_fingerprint(),
        "jax": {"backend": jax.default_backend(), "x64": bool(jax.config.jax_enable_x64),
                "version": jax.__version__},
        "cache_sha256": cache_sha, "jobs": jobs_out})


@command("counters")
def cmd_counters(out_dir: Path, argv: list[str]) -> None:
    """Scripted batches through ONE WG likelihood (E-D6): after every ``lnlike`` call the
    counters, the values and ``last_failed``.  Construction as ``evaluate``; the batches are
    ``b0, b1, ...`` of an .npz, evaluated in that order on one object (the LRU carries over)."""
    ap = argparse.ArgumentParser(prog="sinusoid_wg counters")
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache-src", type=Path, required=True)
    a = ap.parse_args(argv)
    import wdmtoysgpu  # noqa: F401  (enables x64 before anything touches jax)
    import jax
    from wdmtoysgpu.matrix import build_matrix
    from wdm_toys.config import CONFIGS
    from wdm_toys.data import load
    C.check_wdm(required=True)
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit(f"lineage B runs on CPU-JAX with x64; got {jax.default_backend()}, "
                         f"x64={jax.config.jax_enable_x64}")
    cache = out_dir / "cache"
    cache_sha = C.copy_cache(a.cache_src, cache)
    cfg = CONFIGS[a.case]()
    data = np.asarray(load(str(a.data_root / a.case / "toy_data.npz"), cfg)["d"], dtype=float)
    kw = dict(P=FULL_ORDER, tol=FULL_TOL, cache_dir=str(cache)) if a.model == "exact" else {}
    lik = build_matrix(a.model, cfg, data, analysis_fmin=ANALYSIS_FMIN, **kw)
    with np.load(a.batches, allow_pickle=False) as z:
        names = sorted(z.files, key=lambda k: int(k[1:]))
        batches = [np.asarray(z[k], dtype=float) for k in names]
    calls = []
    for X in batches:
        values = np.asarray(lik.lnlike(X), dtype=float)
        last = getattr(lik, "last_failed", None)
        calls.append({"values": values, "failed": None if last is None else np.asarray(last, dtype=bool),
                      "counters": {k: int(getattr(lik, k)) for k in COUNTERS if hasattr(lik, k)}})
    C.assert_cache_unchanged(cache, cache_sha)
    C.write_json(out_dir / "counters.json", {
        "case": a.case, "model": a.model, "class": type(lik).__name__, "batches": names,
        "cache_size": getattr(lik, "cache_size", None), "calls": calls,
        "wdm_file": C.check_wdm(required=True), "runtime": C.runtime_fingerprint()})


def main() -> None:
    out_dir, argv = C.start(("B",))
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"command must be one of {sorted(COMMANDS)}")
    COMMANDS[argv[0]](out_dir, argv[1:])


if __name__ == "__main__":
    main()
