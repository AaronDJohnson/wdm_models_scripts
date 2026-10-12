"""Lineage-A worker: the NumPy oracle (wdm_toys) that wrote the sinusoid references (spec §7.1).

    $PY -P sinusoid_wt.py <out_dir> wd-check --data-root <dir>
    $PY -P sinusoid_wt.py <out_dir> evaluate --jobs <jobs.json> --data-root <dir> --cache-src <dir>
    $PY -P sinusoid_wt.py <out_dir> intermediates --data-root <dir> --cache-src <dir>

PYTHONPATH = lineage A only (toy_models/vendor/wdm/src : wdm_paper_2/toy_models/src).
Writes only into <out_dir>. The exact model's cache_dir is a private copy inside
<out_dir>, re-hashed after use (WT FastCovariance np.save's a missing c_N into it).
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
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
NF, NT, DT = 64, 256, 2.0 ** -10
COMMANDS: dict[str, Callable[[Path, list[str]], None]] = {}


def command(name: str):
    def register(fn):
        COMMANDS[name] = fn
        return fn
    return register


@command("wd-check")
def cmd_wd_check(out_dir: Path, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="sinusoid_wt wd-check")
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args(argv)
    import wdm
    report = {"wdm_file": C.check_wdm(required=True), "cases": {}}
    for case in CASES:
        with np.load(a.data_root / case / "toy_data.npz", allow_pickle=False) as z:
            d, w_d = z["d"], z["w_d"]
        w = wdm.forward_time(d, NF, NT, DT)
        report["cases"][case] = {"array_equal": bool(np.array_equal(w, w_d)),
                                 "max_abs": float(np.max(np.abs(w - w_d))),
                                 "shape": list(w.shape), "sha256_w": C.sha256_array(w)}
    C.write_json(out_dir / "wd_check.json", report)


# --- evaluate (Task A6) --------------------------------------------------------------------
ANALYSIS_FMIN = 4.0     # gpu/run.py::_ANALYSIS_FMIN == make_bundle.ANALYSIS_FMIN


@command("evaluate")
def cmd_evaluate(out_dir: Path, argv: list[str]) -> None:
    """lnlike at the given points, built as make_matrix_bundle.cpu_references does:
    wdm_toys.matrix.build(model, CONFIGS[case](), load(path, cfg)["d"], analysis_fmin=4.0,
    cache_dir=<private copy> for exact only), then lik.lnlike(point) one point at a time."""
    ap = argparse.ArgumentParser(prog="sinusoid_wt evaluate")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache-src", type=Path, required=True)
    a = ap.parse_args(argv)
    from wdm_toys import matrix as M
    from wdm_toys.config import CONFIGS
    from wdm_toys.data import load
    C.check_wdm(required=True)
    cache = out_dir / "cache"
    cache_sha = C.copy_cache(a.cache_src, cache)
    current, lik, build_seconds = None, None, 0.0
    jobs_out = {}
    for job in C.read_json(a.jobs):
        case, model, tag = job["case"], job["model"], job["tag"]
        if (case, model) != current:
            lik = None                                   # release the previous model first
            cfg = CONFIGS[case]()
            data = np.asarray(load(str(a.data_root / case / "toy_data.npz"), cfg)["d"], dtype=float)
            kw = {"cache_dir": str(cache)} if model == "exact" else {}
            t0 = time.perf_counter()
            lik = M.build(model, cfg, data, analysis_fmin=ANALYSIS_FMIN, **kw)
            build_seconds, current = time.perf_counter() - t0, (case, model)
        with np.load(job["points"], allow_pickle=False) as z:
            theta = np.asarray(z["theta"], dtype=float)
        if job.get("wrap_phase", False):
            theta = C.wrap_phase_runner(theta)
        values = np.empty(len(theta))
        seconds = np.empty(len(theta))
        for i, point in enumerate(theta):
            t0 = time.perf_counter()
            values[i] = lik.lnlike(point)
            seconds[i] = time.perf_counter() - t0
        np.savez(out_dir / f"{tag}.npz", lnlike=values, seconds=seconds, theta=theta)
        jobs_out[tag] = {"case": case, "model": model, "n": int(len(theta)), "build_seconds": build_seconds,
                         "class": type(lik).__name__, "pd_failed_rows": int(getattr(lik, "pd_failed_rows", 0))}
    C.assert_cache_unchanged(cache, cache_sha)
    C.write_json(out_dir / "evaluate.json", {"wdm_file": C.check_wdm(required=True),
                                             "runtime": C.runtime_fingerprint(),
                                             "cache_sha256": cache_sha, "jobs": jobs_out})


def main() -> None:
    out_dir, argv = C.start(("A",))
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"command must be one of {sorted(COMMANDS)}")
    COMMANDS[argv[0]](out_dir, argv[1:])


if __name__ == "__main__":
    main()
