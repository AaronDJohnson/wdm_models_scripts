"""Run a worker in one frozen code lineage, in its own subprocess (spec §7.1).

The vendored, June and release packages all import as ``wdm``, so each lineage gets a
fresh interpreter whose PYTHONPATH is exactly that lineage's roots:

    out_dir = run_worker("A", "sinusoid_wt", "evaluate", "--jobs", jobs_json, ...)

Workers live in scripts/golden/workers/, write only into ``out_dir`` (under $TMPDIR)
and never into the frozen trees; harness code copies validated results into golden/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from scripts.golden.paths import REL, ROOT, WORKERS, tmp_base, tmp_root

_A = ["toy_models/vendor/wdm/src", "wdm_paper_2/toy_models/src"]
_C = ["toy_models/galactic_binary", "toy_models/galactic_binary/gpu", "wdm_paper_2/gbonfly/src",
      "wdm_paper_2/toy_models/package/src", "wdm_paper_2/toy_models/src"]
_R = ["packages/wdm/src", "packages/domainia/src", "packages/wdm_models/src"]

#: lineage -> PYTHONPATH entries (absolute), spec §7.1.
LINEAGES: dict[str, list[str]] = {
    "A": [str(ROOT / p) for p in _A],
    "B": [str(ROOT / p) for p in _A + ["wdm_paper_2/toy_models/gpu_sinusoid"]],
    "C": [str(ROOT / p) for p in _C],
    "D": [str(ROOT), str(ROOT / "wdm_paper_1" / "notes")],
    "R": [str(REL / p) for p in _R],
}
#: lineage -> the directory ``wdm.__file__`` must lie under (None: must not import wdm).
EXPECTED_WDM: dict[str, str | None] = {
    "A": str(ROOT / "toy_models" / "vendor" / "wdm" / "src" / "wdm"),
    "B": str(ROOT / "toy_models" / "vendor" / "wdm" / "src" / "wdm"),
    "C": None,
    "D": str(ROOT / "wdm"),
    "R": str(REL / "packages" / "wdm" / "src" / "wdm"),
}
#: Frozen source directories whose __pycache__ must never change (D: the packages it imports,
#: not the whole workspace root).
SOURCE_DIRS: dict[str, list[str]] = {
    "A": LINEAGES["A"],
    "B": LINEAGES["B"],
    "C": LINEAGES["C"],
    "D": [str(ROOT / "wdm"), str(ROOT / "wdm_paper_1" / "notes")],
}
#: Interpreter for every worker (the wdmrel python running the harness).
PY = Path(os.environ.get("WDM_GOLDEN_PY", sys.executable))
_STRIP = ("PYTHON", "JAX_", "XLA_", "WDM_GOLDEN_")


class WorkerError(RuntimeError):
    """A worker exited non-zero or timed out; the message carries its stderr tail."""


def worker_env(lineage: str) -> dict[str, str]:
    if lineage not in LINEAGES:
        raise KeyError(f"unknown lineage {lineage!r}; choose from {sorted(LINEAGES)}")
    env = {k: v for k, v in os.environ.items() if not k.startswith(_STRIP)}
    env.update({
        "PYTHONPATH": os.pathsep.join(LINEAGES[lineage]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(tmp_base() / "wdmrel_pyc"),
        "JAX_ENABLE_X64": "true",
        "JAX_PLATFORMS": "cpu",
        "WDM_GOLDEN_LINEAGE": lineage,
        "WDM_GOLDEN_EXPECT_WDM": EXPECTED_WDM[lineage] or "",
        "WDM_LEGACY_ROOT": str(ROOT),
        "WDM_RELEASE_ROOT": str(REL),
    })
    return env


def run_worker(lineage: str, worker: str, *args: str, timeout: float = 1800.0) -> Path:
    env = worker_env(lineage)
    script = WORKERS / f"{worker}.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cwd = tmp_root()
    out_dir = cwd / "out" / f"{lineage}-{worker}-{uuid.uuid4().hex[:12]}"
    out_dir.mkdir(parents=True)
    cmd = [str(PY), "-P", str(script), str(out_dir), *map(str, args)]
    (out_dir / "command.json").write_text(json.dumps(
        {"cmd": cmd, "cwd": str(cwd), "lineage": lineage, "PYTHONPATH": env["PYTHONPATH"]}, indent=1) + "\n")
    with (out_dir / "stdout.log").open("w") as so, (out_dir / "stderr.log").open("w") as se:
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=so, stderr=se, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise WorkerError(f"{worker} (lineage {lineage}) timed out after {timeout:.0f} s; logs in {out_dir}") from exc
    if proc.returncode != 0:
        tail = (out_dir / "stderr.log").read_text(errors="replace")[-4000:]
        raise WorkerError(f"{worker} (lineage {lineage}) exited {proc.returncode}; logs in {out_dir}\n{tail}")
    return out_dir
