"""Golden-reference validation of a likelihood: the comparison rules only (spec section 6:
every runner validates its target before sampling; section 7.3: the approved gates).

No golden-file layout lives here (``scripts/sinusoid/validate.py`` reads the goldens) and
nothing samples.  The rules are those of ``toy_models/sinusoid/gpu/run.py::validate_reference``
and ``scripts/golden/sinusoid_goldens.compare``, made fail-closed:

* near points pass when ``|value - reference| <= atol``; stress points when
  ``<= atol + rtol |reference|``; with ``exact=True`` only ``== reference`` passes;
* ``-inf`` must appear exactly where the reference has ``-inf`` (a mismatch is an infinite
  delta); NaN and ``+inf`` never pass;
* a likelihood's ``last_failed`` must flag exactly its ``-inf`` rows, and its counters
  (``domainia.likelihood.COUNTERS``) must count the rows evaluated and the reference's
  ``-inf`` rows;
* anything that makes the comparison meaningless (no points, a malformed reference, a gate
  that is not a finite non-negative number) raises :class:`ValidationError`, never passes.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["COUNTER_KEYS", "RECORDING_KEYS", "Gate", "ValidationError", "build_fingerprint",
           "check_counters", "check_reference", "compare", "evaluate", "jsonable", "same_build",
           "wrap_periodic", "write_report"]

#: ``domainia.likelihood.COUNTERS`` (``chain_stats.COUNTERS``): what every likelihood counts.
COUNTER_KEYS = ("noise_states", "pd_failed_states", "pd_failed_rows", "evaluations")
#: Fingerprint keys that decide whether this is the build a golden was recorded on.
RECORDING_KEYS = ("python", "machine", "numpy", "scipy", "blas")
_THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "XLA_FLAGS")


class ValidationError(RuntimeError):
    """The validation cannot be carried out (nothing selected, a reference missing or
    malformed, a gate not approved): never a pass."""


@dataclass(frozen=True)
class Gate:
    """``|delta| <= atol`` at near points, ``<= atol + rtol |reference|`` at stress points."""

    atol: float
    rtol: float = 0.0

    def __post_init__(self) -> None:
        for name in ("atol", "rtol"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValidationError(f"gate {name} must be a finite non-negative number; got {v!r}")

    def allowed(self, reference: Any, near: Any) -> np.ndarray:
        ref = np.asarray(reference, dtype=float)
        mag = np.abs(np.where(np.isfinite(ref), ref, 0.0))
        return np.where(np.asarray(near, dtype=bool), float(self.atol), self.atol + self.rtol * mag)


def wrap_periodic(theta: Any, periodic: Mapping[int, tuple[float, float]]) -> np.ndarray:
    """``lo + mod(x - lo, hi - lo)`` per periodic column (gpu/run.py ``_wrap_phase``, the
    arithmetic of ``domainia.sampling.PriorSanitized.wrap``); returns a copy."""
    out = np.array(theta, dtype=np.float64, copy=True)
    for i, (lo, hi) in periodic.items():
        out[..., i] = lo + np.mod(out[..., i] - lo, hi - lo)
    return out


def check_reference(theta: Any, loglike: Any, near: Any, ndim: int, *, what: str) -> None:
    """Raise :class:`ValidationError` unless ``theta (n, ndim)``, ``loglike (n,)`` and
    ``near (n,)`` describe at least one point, every theta is finite, every loglike is
    finite or ``-inf`` and at least one is finite."""
    theta, loglike, near = np.asarray(theta), np.asarray(loglike), np.asarray(near)
    n = len(loglike)
    if n == 0:
        raise ValidationError(f"{what}: no points selected")
    if theta.shape != (n, ndim) or loglike.shape != (n,) or near.shape != (n,):
        raise ValidationError(f"{what}: theta/loglike/near shapes {theta.shape}/{loglike.shape}/{near.shape}, "
                              f"expected ({n}, {ndim})/({n},)/({n},)")
    if not np.all(np.isfinite(theta)):
        raise ValidationError(f"{what}: non-finite reference theta")
    loglike = loglike.astype(float)
    if not np.all(np.isfinite(loglike) | (loglike == -np.inf)):
        raise ValidationError(f"{what}: a reference loglike is NaN or +inf")
    if not np.any(np.isfinite(loglike)):
        raise ValidationError(f"{what}: every reference loglike is -inf")


def compare(values: Any, reference: Any, near: Any, gate: Gate, *, exact: bool = False) -> dict:
    """Point-by-point comparison; ``ok`` only when every point passes."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    near = np.asarray(near, dtype=bool)
    if values.shape != reference.shape or near.shape != reference.shape or values.ndim != 1:
        raise ValidationError(f"values {values.shape}, reference {reference.shape}, near {near.shape} disagree")
    if values.size == 0:
        raise ValidationError("nothing to compare")
    v_inf, r_inf = values == -np.inf, reference == -np.inf
    invalid = ~(np.isfinite(values) | v_inf)                    # NaN or +inf
    mismatch = (v_inf != r_inf) & ~invalid
    both = ~v_inf & ~r_inf & ~invalid
    delta = np.zeros(values.shape)
    delta[both] = np.abs(values[both] - reference[both])
    delta[mismatch | invalid] = np.inf
    rel = np.zeros(values.shape)
    rel[both] = delta[both] / np.maximum(np.abs(reference[both]), np.finfo(float).tiny)
    rel[mismatch | invalid] = np.inf
    allowed = np.zeros(values.shape) if exact else gate.allowed(reference, near)
    passed = delta <= allowed
    return {"abs_delta": delta, "rel_delta": rel, "allowed": allowed, "passed": passed, "exact": bool(exact),
            "inf_mismatch": int(mismatch.sum()), "invalid": int(invalid.sum()),
            "max_abs": float(delta.max()), "near_max_abs": float(delta[near].max(initial=0.0)),
            "stress_max_abs": float(delta[~near].max(initial=0.0)),
            "stress_max_rel": float(rel[~near].max(initial=0.0)), "ok": bool(passed.all())}


def evaluate(lik: Any, theta: Any) -> dict:
    """One call ``lik(theta)`` on fresh counters: values, ``last_failed``, counters, seconds."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    if hasattr(lik, "reset_counters"):
        lik.reset_counters()
    t0 = time.perf_counter()
    values = np.asarray(lik(theta), dtype=np.float64)
    seconds = time.perf_counter() - t0
    if values.shape != (len(theta),):
        raise ValidationError(f"the likelihood returned shape {values.shape} for {len(theta)} rows")
    failed = getattr(lik, "last_failed", None)
    failed = np.zeros(len(theta), dtype=bool) if failed is None else np.asarray(failed, dtype=bool)
    counters = {k: int(v) for k, v in lik.counters().items()} if hasattr(lik, "counters") else {}
    return {"values": values, "last_failed": failed, "counters": counters, "seconds": float(seconds)}


def check_counters(result: Mapping[str, Any], reference: Any, *, noise_states: int | None) -> list[str]:
    """Problems with the ``-inf`` bookkeeping of one :func:`evaluate` (empty: consistent).

    ``last_failed`` must flag exactly the ``-inf`` values; ``evaluations`` must equal the
    rows; ``pd_failed_rows`` and ``pd_failed_states`` the reference's ``-inf`` rows (one
    state per row: the uncached path); ``noise_states`` the given count (``None``: not
    checked, e.g. behind a cache)."""
    reference = np.asarray(reference, dtype=float)
    n, want = len(reference), int(np.sum(reference == -np.inf))
    values, failed, c = result["values"], np.asarray(result["last_failed"], bool), result["counters"]
    problems = []
    if failed.shape != (n,):
        problems.append(f"last_failed has shape {failed.shape}, expected ({n},)")
    elif not np.array_equal(failed, values == -np.inf):
        problems.append(f"last_failed flags rows {np.flatnonzero(failed).tolist()}, "
                        f"-inf values at {np.flatnonzero(values == -np.inf).tolist()}")
    missing = [k for k in COUNTER_KEYS if k not in c]
    if missing:
        return problems + [f"counters missing {missing}"]
    expected = {"evaluations": n, "pd_failed_rows": want, "pd_failed_states": want}
    if noise_states is not None:
        expected["noise_states"] = int(noise_states)
    for key, value in expected.items():
        if c[key] != value:
            problems.append(f"counter {key} = {c[key]}, expected {value}")
    return problems


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def build_fingerprint(backend: str) -> dict:
    """Interpreter, wheels, BLAS, threads, the release packages and (JAX) the devices."""
    fp: dict[str, Any] = {"python": platform.python_version(), "executable": sys.executable,
                          "machine": platform.machine(), "platform": platform.platform(),
                          "backend": backend}
    for dist in ("numpy", "scipy", "jax", "jaxlib"):
        fp[dist] = _version(dist)
    try:
        fp["blas"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except (KeyError, TypeError):
        fp["blas"] = None
    fp["threads"] = {k: os.environ.get(k) for k in _THREAD_KEYS}
    fp["packages"] = {}
    for name in ("wdm", "domainia", "wdm_models"):
        mod = sys.modules.get(name)
        fp["packages"][name] = None if mod is None else {"version": getattr(mod, "__version__", None),
                                                         "file": getattr(mod, "__file__", None)}
    if backend == "jax" and "jax" in sys.modules:
        jax = sys.modules["jax"]
        fp["jax_backend"] = jax.default_backend()
        fp["jax_devices"] = [str(d) for d in jax.devices()]
        fp["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
    return fp


def same_build(a: Mapping[str, Any], b: Mapping[str, Any], keys: tuple[str, ...] = RECORDING_KEYS) -> bool:
    """True when two fingerprints agree on every key (and none is missing)."""
    return all(a.get(k) is not None and a.get(k) == b.get(k) for k in keys)


def jsonable(x: Any) -> Any:
    """Strict JSON: arrays -> lists, numpy scalars -> Python, non-finite -> "inf"/"-inf"/"nan"
    (``scripts/golden/workers/_common.jsonable``, the golden reports' convention)."""
    if isinstance(x, Mapping):
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


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(report), indent=1, sort_keys=True, allow_nan=False) + "\n")
