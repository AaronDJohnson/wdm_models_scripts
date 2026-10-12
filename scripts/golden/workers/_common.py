"""Helpers shared by the golden workers and the harness (spec §7.1; Task A3).

Workers run as ``$PY -P <worker>.py <out_dir> <command> ...`` with PYTHONPATH set to ONE
frozen lineage, so this module is not importable by name inside a worker: each worker
loads it by file path (importlib.util.spec_from_file_location), which changes neither
sys.path nor PYTHONPATH. The harness imports it as ``scripts.golden.workers._common``.
Standard library + numpy only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

ENV_KEYS = ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONSAFEPATH", "PYTHONNOUSERSITE",
            "PYTHONPYCACHEPREFIX", "JAX_ENABLE_X64", "JAX_PLATFORMS", "WDM_GOLDEN_LINEAGE",
            "WDM_GOLDEN_EXPECT_WDM", "WDM_LEGACY_ROOT", "WDM_RELEASE_ROOT")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "XLA_FLAGS")
SAMPLE_SEED = 20261011
_CHUNK = 1 << 22


def start(allowed: tuple[str, ...]) -> tuple[Path, list[str]]:
    """Check the worker's environment; return ``(out_dir, argv after out_dir)``."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: <worker>.py <out_dir> <command> [options]")
    out_dir = Path(sys.argv[1]).resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    if not out_dir.is_dir() or not out_dir.is_relative_to(tmp):
        raise SystemExit(f"out_dir {out_dir} must be an existing directory under {tmp}")
    if not (sys.flags.safe_path and sys.flags.dont_write_bytecode):
        raise SystemExit("workers run only as `python -P` with PYTHONDONTWRITEBYTECODE=1")
    lineage = os.environ.get("WDM_GOLDEN_LINEAGE", "")
    if lineage not in allowed:
        raise SystemExit(f"this worker serves lineages {allowed}; WDM_GOLDEN_LINEAGE={lineage!r}")
    return out_dir, sys.argv[2:]


def check_wdm(*, required: bool = False) -> str | None:
    """``wdm.__file__`` if imported, after asserting it lies under this lineage's root."""
    expect = os.environ.get("WDM_GOLDEN_EXPECT_WDM", "")
    module = sys.modules.get("wdm")
    if module is None:
        if required:
            raise AssertionError("this worker must have imported wdm by now")
        return None
    path = Path(module.__file__).resolve()
    if not expect:
        raise AssertionError(f"lineage {os.environ.get('WDM_GOLDEN_LINEAGE')} must not import wdm; got {path}")
    if not path.is_relative_to(Path(expect).resolve()):
        raise AssertionError(f"wdm resolved to {path}; expected a file under {expect}")
    return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(a) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def jsonable(x):
    """Strict-JSON form: arrays -> lists, numpy scalars -> Python, non-finite -> "inf"/"-inf"/"nan"."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if math.isfinite(v):
            return v
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, str):
        return x
    raise TypeError(f"not JSON-serialisable: {type(x).__name__}")


def to_float(v) -> float:
    """Inverse of :func:`jsonable` for one number."""
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v]
    return float(v)


def write_json(path: Path | str, obj) -> None:
    Path(path).write_text(json.dumps(jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def array_summary(a, *, k: int = 50, seed: int = SAMPLE_SEED) -> dict:
    """Shape, dtype, sha256 of the C-order bytes, L2 norm, sum and ``k`` entries at fixed
    flat indices (default_rng(seed)): bitwise and tolerance checks without storing ``a``."""
    a = np.asarray(a)
    flat = a.reshape(-1)
    n = min(k, flat.size)
    idx = (np.sort(np.random.default_rng(seed).choice(flat.size, size=n, replace=False))
           if n else np.zeros(0, dtype=int))
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_array(a),
            "l2": float(np.linalg.norm(flat)), "sum": float(flat.sum()),
            "sample_index": idx.tolist(), "sample_values": flat[idx].tolist()}


def wrap_phase_runner(theta, index: int = 2) -> np.ndarray:
    """toy_models/sinusoid/gpu/run.py::_wrap_phase, the arithmetic the chains were evaluated with."""
    out = np.array(theta, dtype=float, copy=True)
    lo, hi = -np.pi, np.pi
    out[..., index] = lo + np.mod(out[..., index] - lo, hi - lo)
    return out


def copy_cache(src: Path | str, dst: Path | str) -> dict[str, str]:
    """Private copy of a cache directory (WT FastCovariance np.save's a missing c_N into it)."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    out = {}
    for path in sorted(p for p in src.iterdir() if p.is_file()):
        body = path.read_bytes()
        (dst / path.name).write_bytes(body)
        out[path.name] = sha256_bytes(body)
    return out


def assert_cache_unchanged(dst: Path | str, before: dict[str, str]) -> None:
    after = {p.name: sha256_file(p) for p in sorted(Path(dst).iterdir())}
    if after != before:
        raise AssertionError(f"legacy code wrote into its private cache dir {dst}: "
                             f"before {sorted(before)}, after {sorted(after)}")


def runtime_fingerprint() -> dict:
    """Interpreter, wheel versions, BLAS and thread environment of this process."""
    fp = {"python": platform.python_version(), "executable": sys.executable,
          "machine": platform.machine(), "platform": platform.platform()}
    for dist in ("numpy", "scipy", "jax", "jaxlib"):
        try:
            fp[dist] = md.version(dist)
        except md.PackageNotFoundError:
            fp[dist] = None
    try:
        fp["blas"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except (KeyError, TypeError):
        fp["blas"] = None
    fp["threads"] = {k: os.environ.get(k) for k in THREAD_KEYS}
    return fp
