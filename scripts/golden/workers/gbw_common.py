"""Shared helpers for the section-B golden workers (lineages C and D).

Workers append this directory to the END of ``sys.path`` (lineage modules win) and
import this module. NumPy only, so it imports in every lineage.

* :func:`install_write_guard`: an audit hook that raises ``PermissionError`` on any file
  write, rename, removal or mkdir outside the allowed roots (``out_dir`` and ``$TMPDIR``).
  This is the backstop for the cache-writing trap: the legacy code cannot write into the
  frozen trees even through a path nobody found by grepping.
* :class:`Payload`: everything a worker produces goes to ``<out_dir>/payload/<relpath>``
  (``relpath`` relative to ``golden/``), listed with its sha256 in
  ``<out_dir>/payload_index.json``; ``capture_payload.py`` copies it into ``golden/``.
* :func:`fingerprint`: sha256 of the bytes, Frobenius norm, sums and 50 entries at fixed
  indices, for arrays too large to store.
* :func:`select_rows`: the G3 chain-row tiers.

Why not ``scripts.golden.goldens_io``: a worker's PYTHONPATH is exactly its lineage's, so it cannot
import the ``scripts`` package. ``write_npz`` here and A4's ``goldens_io.write_npz`` are both
byte-deterministic but are different writers (streamed vs buffered members), so the same arrays
give different bytes; only this one writes Section-B goldens. Likewise ``Payload.write_json``
keeps non-finite floats as JSON ``-Infinity``/``NaN`` (Python's ``json`` round-trips them), whereas
A's ``_common.jsonable`` writes the strings ``"-inf"``/``"nan"``; readers of Section-B files use
``json.loads`` directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = 1
FP_SAMPLES = 50
FP_SEED = 20261011
ROW_SEED = 20261011
FULL_EXTRA_ROWS = 29

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_PATH_EVENTS = frozenset({"os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.truncate",
                          "os.chmod", "os.chown", "os.utime", "os.symlink", "os.link",
                          "shutil.rmtree", "shutil.copyfile", "shutil.move"})


def _real(p):
    if isinstance(p, (str, bytes, os.PathLike)):
        return os.path.realpath(os.fsdecode(p))
    return None


def install_write_guard(*allowed) -> tuple[str, ...]:
    """Block every write outside ``allowed`` (and ``/dev/null``) for the rest of the process."""
    roots = tuple(sorted({os.path.realpath(str(a)) for a in allowed})) + ("/dev/null",)

    def inside(p: str) -> bool:
        return any(p == r or p.startswith(r.rstrip(os.sep) + os.sep) for r in roots)

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            p = _real(path)
            if p is None:
                return
            writing = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                       or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS)))
            if writing and not inside(p):
                raise PermissionError(f"golden worker write guard: blocked write to {p}")
        elif event in _PATH_EVENTS:
            for a in args:
                p = _real(a)
                if p is not None and not inside(p):
                    raise PermissionError(f"golden worker write guard: blocked {event} on {p}")

    sys.addaudithook(hook)
    return roots


def require_safe_interpreter() -> None:
    if not sys.flags.safe_path:
        raise SystemExit("golden workers run with python -P")
    if not sys.flags.dont_write_bytecode:
        raise SystemExit("golden workers run with PYTHONDONTWRITEBYTECODE=1")


def check_origin(module, expected_dir) -> str:
    """``module.__file__`` must live directly in ``expected_dir`` (lineage check)."""
    here = Path(module.__file__).resolve()
    exp = Path(expected_dir).resolve()
    if here.parent != exp and exp not in here.parents:
        raise RuntimeError(f"{module.__name__} resolved to {here}, expected under {exp}")
    return str(here)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(a, n: int = FP_SAMPLES, seed: int = FP_SEED) -> dict:
    """Comparable summary of an array: exact bytes hash plus values a tolerance can test."""
    arr = np.ascontiguousarray(np.asarray(a))
    flat = arr.reshape(-1)
    k = min(n, flat.size)
    idx = (np.sort(np.random.default_rng(seed).choice(flat.size, size=k, replace=False))
           if k else np.zeros(0, np.int64))
    out = {"shape": list(arr.shape), "dtype": arr.dtype.str, "sha256": sha256_bytes(arr.tobytes()),
           "fro": float(np.linalg.norm(flat.astype(np.complex128 if np.iscomplexobj(arr)
                                                   else np.float64))) if flat.size else 0.0,
           "idx": [int(i) for i in idx]}
    if np.iscomplexobj(arr):
        out["sum_re"] = float(np.sum(flat.real))
        out["sum_im"] = float(np.sum(flat.imag))
        out["values_re"] = flat[idx].real.tolist()
        out["values_im"] = flat[idx].imag.tolist()
    else:
        out["sum"] = float(np.sum(flat, dtype=np.float64))
        out["values"] = flat[idx].tolist()
    return out


def chain_length(path, width: int) -> int:
    size = os.path.getsize(path)
    if size % (8 * width):
        raise ValueError(f"{path}: {size} bytes is not a whole number of {width}-column rows")
    return size // (8 * width)


def select_rows(n: int, tier: str) -> list[int]:
    """G3 rows: ``fast`` = first, middle, last; ``full`` = those + 29 from default_rng(20261011)."""
    if n < 1:
        raise ValueError("empty chain")
    fast = sorted({0, n // 2, n - 1})
    if tier == "fast":
        return fast
    if tier != "full":
        raise ValueError(f"unknown tier {tier!r}")
    pool = np.setdiff1d(np.arange(n), fast)
    extra = np.random.default_rng(ROW_SEED).choice(pool, size=min(FULL_EXTRA_ROWS, pool.size),
                                                   replace=False)
    return sorted(set(fast) | {int(i) for i in extra})


def read_rows(path, width: int, rows) -> np.ndarray:
    """``(len(rows), width)`` float64 rows of a raw chain file, opened read-only."""
    mm = np.memmap(path, dtype="<f8", mode="r")
    try:
        return np.stack([np.array(mm[i * width:(i + 1) * width], dtype=np.float64) for i in rows])
    finally:
        del mm


def build_info() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "platform": platform.platform(), "numpy": np.__version__}
    for name in ("scipy", "jax", "jaxlib", "gbonfly"):
        mod = sys.modules.get(name)
        if mod is not None:
            info[name] = str(getattr(mod, "__version__", "unknown"))
    info["threads"] = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "XLA_FLAGS", "JAX_PLATFORMS", "JAX_ENABLE_X64")}
    return info


def write_npz(path, arrays: dict) -> None:
    """Byte-deterministic ``.npz`` (fixed zip timestamps, sorted keys), loadable by np.load."""
    from numpy.lib import format as npformat
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                npformat.write_array(fh, np.asanyarray(arrays[key]), allow_pickle=False)


def jsonable(x):
    """NumPy scalars/arrays to plain Python (floats keep repr round-trip precision)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


class Payload:
    """Files a worker hands to ``capture_payload.py`` (paths relative to ``golden/``)."""

    def __init__(self, out_dir, *, lineage: str, worker: str, subcommand: str, args, base=None):
        self.out = Path(out_dir).resolve()
        self.base = None if base is None else Path(base).resolve()
        self.root = self.out / "payload"
        self.meta = {"schema": SCHEMA, "lineage": lineage, "worker": worker,
                     "subcommand": subcommand, "args": list(args)}
        self.files: dict[str, str] = {}
        self.inputs: dict[str, str] = {}

    def _target(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"payload path {rel!r} escapes the payload directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def add_input(self, path) -> str:
        """Record an input file's sha256 (keyed relative to ``base`` when inside it)."""
        p = Path(path).resolve()
        sha = sha256_file(p)
        key = str(p.relative_to(self.base)) if self.base is not None and self.base in p.parents else str(p)
        self.inputs[key] = sha
        return sha

    def write_json(self, rel: str, obj) -> str:
        p = self._target(rel)
        p.write_text(json.dumps(jsonable(obj), indent=1, sort_keys=True, allow_nan=True) + "\n")
        self.files[rel] = sha256_file(p)
        return self.files[rel]

    def write_npz(self, rel: str, arrays: dict) -> str:
        p = self._target(rel)
        write_npz(p, arrays)
        self.files[rel] = sha256_file(p)
        return self.files[rel]

    def close(self) -> Path:
        index = dict(self.meta, files=dict(sorted(self.files.items())),
                     inputs=dict(sorted(self.inputs.items())), build=build_info())
        path = self.out / "payload_index.json"
        path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
        return path
