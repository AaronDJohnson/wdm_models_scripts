"""evaluate(lineage, toy, case, model, theta) -> lnL through a frozen lineage (spec §7.1).

Each call stages the points under $TMPDIR/golden_run/jobs/, runs ONE worker subprocess
(scripts.golden.lineage.run_worker) and reads back <tag>.npz and evaluate.json. Other
sections register their workers in WORKERS and their per-toy arguments in WORKER_ARGS
(galactic binary: ("C", "galactic_binary") -> "gb_legacy").
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from scripts.golden.goldens_io import read_json, read_npz, write_json
from scripts.golden.lineage import run_worker
from scripts.golden.paths import DATA, tmp_root

WORKERS: dict[tuple[str, str], str] = {("A", "sinusoid"): "sinusoid_wt", ("B", "sinusoid"): "sinusoid_wg"}
WORKER_ARGS: dict[str, Callable[[], list[str]]] = {
    "sinusoid": lambda: ["--data-root", str(DATA / "sinusoid"), "--cache-src", str(DATA / "sinusoid" / "cache")],
}


@dataclass(frozen=True, eq=False)
class Job:
    tag: str                 # unique file stem
    case: str
    model: str
    theta: np.ndarray        # (n, ndim)
    wrap_phase: bool = False  # apply gpu/run.py::_wrap_phase first (chain rows)


def evaluate_jobs(lineage: str, toy: str, jobs: list[Job], *, timeout: float = 1800.0) -> tuple[dict, dict]:
    """``({tag: {"lnlike", "seconds", "theta", ["failed"], "info"}}, worker meta)``."""
    if (lineage, toy) not in WORKERS:
        raise KeyError(f"no worker registered for lineage {lineage!r} and toy {toy!r}")
    tags = [job.tag for job in jobs]
    if len(set(tags)) != len(tags) or any("/" in t or not t for t in tags):
        raise ValueError("job tags must be unique, non-empty file stems")
    stage = tmp_root() / "jobs" / uuid.uuid4().hex[:12]
    stage.mkdir(parents=True)
    spec = []
    for job in jobs:
        points = stage / f"{job.tag}.npz"
        np.savez(points, theta=np.atleast_2d(np.asarray(job.theta, dtype=float)))
        spec.append({"tag": job.tag, "case": job.case, "model": job.model, "points": str(points),
                     "wrap_phase": bool(job.wrap_phase)})
    write_json(stage / "jobs.json", spec)
    out = run_worker(lineage, WORKERS[(lineage, toy)], "evaluate", "--jobs", str(stage / "jobs.json"),
                     *WORKER_ARGS[toy](), timeout=timeout)
    summary = read_json(out / "evaluate.json")
    results = {}
    for job in jobs:
        r = read_npz(out / f"{job.tag}.npz")
        r["info"] = summary["jobs"][job.tag]
        results[job.tag] = r
    return results, {k: v for k, v in summary.items() if k != "jobs"}


def evaluate(lineage: str, toy: str, case: str, model: str, theta, *, wrap_phase: bool = False) -> np.ndarray:
    """lnL of ``model`` at the rows of ``theta`` in ``lineage`` (one subprocess)."""
    results, _ = evaluate_jobs(lineage, toy, [Job("query", case, model, np.atleast_2d(theta), wrap_phase)])
    return results["query"]["lnlike"]
