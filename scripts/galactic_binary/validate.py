"""Validate the galactic-binary likelihoods against their goldens; evaluates, never samples
(spec sections 6, 7.2 G3-G6 and 7.3).

From the release root, after ``source env/activate.sh``:

    $PY -P -m scripts.galactic_binary.validate --backend numpy --tier fast [--case C ...] [--model M ...] \\
        [--golden g3 g4 g5 g6 audit] [--waveform fourier] [--templates bg|gbonfly] \\
        [--report /outside/the/release/out.json] [--golden-root DIR] [--data-root DIR]

For each selected target ``(case, model)`` (``wdm_models.registry.TARGETS["galactic_binary"]``,
38) the release likelihood is evaluated at

* **G3** -- the stored cold-chain rows of every run directory of the target
  (``golden/galactic_binary/chain_rows/<case>/<run>.<tier>.json``, the runs listed by
  ``index.json``; fast tier: rows 0, n/2, n-1 of each of the 58 run directories), against the
  chain's ``lnlike`` column at the approved uniform gate (``approved.G3.<backend>``, 1e-8);
  also against lineage C's NumPy value (``lnlike_numpy``): **exactly** with the gbonfly
  adapter on the build that recorded the goldens, else at the same G3 gate;
* **G4** -- the 11 stress points of ``stress/<case>/<model>.<tier>.json`` (3 and 6 months):
  identical ``-inf`` masks against lineage C's reference at every point (marginal included)
  and the worker's expected mask (``source_out or margin < 0``) at every non-marginal point;
  ``|delta| <= 1e-10 + 1e-12 |lnL|`` at well-conditioned points (PD margin >= 1e-6) on NumPy
  (8.32e-8 on JAX; a NaN delta or gate is over it); ill-conditioned points reported
  condition-scaled, marginal points only finite-or-``-inf``; the golden ``lnl_ref`` and
  ``lnl_cache`` must be finite or exactly ``-inf`` (``check_reference``), and each target needs a
  gated well-conditioned point; the counters of the bare evaluation (one state per row; failed states =
  the non-``source_out`` ``-inf`` points); for the cached models also the cache evaluator
  (``ModelOptions(evaluator="cache")``, all points in one call): ``lnl_cache`` at the same
  rules and the counters equal to ``counters_cache``;
* **G5** -- per case: the prepared file's sha256 (the release pin, ``data/MANIFEST.json`` and
  every G3/G4 golden's ``prepared_sha256``), ``registry.describe`` == lineage C's
  ``describe`` (products.json), the template sums at the injection (``fourier``: sum |h|^2 and
  Re<z, h> on the store and the window; 1e-10 relative on the recording build, else 1e-8)
  and the GLASS prior box (bitwise);
* **G6** -- the anchor (``inject_glass`` + ``mu_fit``/``mu_freq``) with the golden gbonfly
  store template: each channel's (quad, logdet) at 1e-12 relative and lnL (``make_lnlike``)
  at the G3 gate; bitwise points are counted;
* **audit** (reported, never gated) -- ``onfly`` against ``fourier`` at the G3 rows: the
  template mismatch ``1 - FF`` on the store and ``lnL_onfly - lnL_fourier``.

``--waveform`` picks the binary_galactica backend of G3/G4 (the release default ``onfly`` is
refused for the golden checks: the chains were evaluated with gbonfly = ``fourier``).
``--templates gbonfly`` (test-only adapter, needs the frozen trees): the G3/G4 templates are
gbonfly's, computed by lineage C (``scripts/golden/workers/gbw_templates.py``) and streamed in,
so the comparison measures the likelihood numerics alone.

Every golden file read is checked against its sha256 in ``golden/MANIFEST.json`` (of
``--golden-root``) at every run, as the sinusoid CLI checks ``references/INDEX.json``: a missing
manifest, a file it does not pin, or other bytes stop the run.

Exit status 0: every check passed.  1: a check failed, or the validation could not be carried
out: an empty selection; a missing golden or one whose sha256 is not the manifest's; a golden
reference that is NaN or ``+inf``; approved gates that are not APPROVED, not finite and positive,
or (G4 JAX) a per-target gate other than the pooled one; a build error or a NaN likelihood value.
2: usage.  The CLI runs XLA-CPU with single-threaded Eigen: :func:`main` appends
``--xla_cpu_multi_thread_eigen=false`` to ``XLA_FLAGS`` before JAX starts unless an
``--xla_cpu_multi_thread_eigen`` setting is already there (spec section 3.4, the user's
validation rule: the binary_galactica templates run on CPU-JAX even on the NumPy backend), and
the report's fingerprint records ``XLA_FLAGS``.  The full tier (32 rows per run directory, 12
stress points) has no captured goldens
yet: capturing them (``scripts.golden.capture_gb --tier full``, lineage C, ~1 h) and running it
need the user's approval (spec section 6).

JAX (S7, section F-D): ``--backend jax`` evaluates the registry's JAX evaluator -- lineage C's
``gpu/jax_models.build`` classes: ``NoiseStateCache`` for the cached models, ``BatchedLikelihood``
otherwise -- at ``approved.G3.jax`` and the pooled G4 JAX gate (8.32e-8 at well-conditioned
points).  Its counters are lineage C's own JAX counters on the same call (``jax_counters`` per
run directory, ``counters_jax`` per stress file; the uncached models count rows only: lineage C's
``BatchedLikelihood`` kept none), not the NumPy reference's one state per row, and G4 has no
separate cache pass (on JAX the default evaluator IS the cache).  The CPU flag above is what
makes CPU-JAX deterministic (on a GPU validation sets ``--xla_gpu_deterministic_ops=true``); the
report records the flags in force as ``run_meta`` (``wdm_models.sampling.device``) and, per run
directory and stress file, ``ms_per_eval`` and
``repeat_max_abs`` of a warm repeat (fresh counters and cache, the programs compiled) and the
largest difference from lineage C's own JAX values -- reported, never gated.  Colab GPU: ``COLAB``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REL = Path(__file__).resolve().parents[2]
#: The release sources, in front of sys.path unless already there (explicit and
#: ``__file__``-relative, so ``python -P`` stays cwd-free: spec section 1, guard 5).
SOURCES = tuple(REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models"))
for _src in reversed(SOURCES):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from wdm_models.galactic_binary import models as gb_models  # noqa: E402  (after the path bootstrap)
from wdm_models.options import GB_CACHED, ModelOptions  # noqa: E402
from wdm_models.registry import MODELS, TARGETS, describe  # noqa: E402
from wdm_models.sampling.device import run_meta  # noqa: E402
from wdm_models.sampling.validate import (Gate, ValidationError, build_fingerprint, check_reference,  # noqa: E402
                                          compare, evaluate, write_report)

GOLDEN = REL / "golden"
TOY = "galactic_binary"
CASES: tuple[str, ...] = tuple(TARGETS[TOY])
STRESS_CASES = ("gb_3mo", "gb_6mo")
BACKENDS = ("numpy", "jax")
TIERS = ("fast", "full")
GOLDENS = ("g3", "g4", "g5", "g6", "audit")
WAVEFORMS = ("fourier", "onfly")
TEMPLATES = ("bg", "gbonfly")
#: Build keys that decide whether this process is the build the GB goldens were recorded on
#: (``gbw_common.build_info``; no BLAS key was recorded).
GB_RECORDING_KEYS = ("python", "machine", "numpy", "scipy", "jax", "jaxlib")
COUNTER_KEYS = ("evaluations", "noise_states", "pd_failed_rows", "pd_failed_states")
#: The G4 rules of ``tolerances_measured.json`` ``approved.G4``, each finite and positive.
G4_RULES = {"well": "well_conditioned_margin", "numpy_abs": "numpy_cache_abs", "numpy_rel": "numpy_cache_rel",
            "jax_well": "jax_well_conditioned"}
#: XLA-CPU determinism for every validation run (spec section 3.4, the user's rule): :func:`main`
#: appends it to ``XLA_FLAGS`` before JAX starts (:func:`cpu_determinism`).
CPU_DETERMINISM = "--xla_cpu_multi_thread_eigen=false"
EPS = float(np.finfo(float).eps)
#: The user-run Colab GPU validation (S7; README "GPU validation of the galactic binary").
COLAB = """\
On a Colab GPU runtime (T4/L4/A100), the release tree unzipped at /content/wdm_models_scripts
(README.md, "GPU validation of the galactic binary"):
  pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt
  pip install -q -r /content/wdm_models_scripts/env/requirements-colab-cuda12.txt
  pip install -q --no-deps -r /content/wdm_models_scripts/env/requirements-bg.txt
  cd /content/wdm_models_scripts
  export PYTHONPATH=$PWD:$PWD/packages/wdm/src:$PWD/packages/domainia/src:$PWD/packages/wdm_models/src
  export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_FLAGS=--xla_gpu_deterministic_ops=true
  export JAX_PLATFORMS=cuda
  python -P -m scripts.galactic_binary.validate --backend jax --tier fast --golden g3 g4 \\
      --report /content/drive/MyDrive/wdm_release/validate_gb_jax_gpu_fast.json
"""

Builder = Callable[..., Any]


# --- golden readers -------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValidationError(f"missing golden file {path}")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _golden_bytes(path: Path, golden: Path) -> bytes:
    """The bytes of the golden file ``path`` (under ``golden``), refused unless their sha256 is the
    one ``golden/MANIFEST.json`` pins: read at every run, since the root lane's MANIFEST test does
    not run where the CLI does (Colab)."""
    if not path.is_file():
        raise ValidationError(f"missing golden file {path}")
    manifest = golden / "MANIFEST.json"
    if not manifest.is_file():
        raise ValidationError(f"missing {manifest}: the goldens cannot be checked against their sha256")
    rel = path.relative_to(golden).as_posix()
    want = json.loads(manifest.read_text()).get("files", {}).get(rel)
    if not isinstance(want, str):
        raise ValidationError(f"golden {rel} is not pinned in {manifest}")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != want:
        raise ValidationError(f"golden {rel}: sha256 differs from {manifest} (an altered golden)")
    return body


def _golden_json(path: Path, golden: Path) -> Any:
    return json.loads(_golden_bytes(path, golden))


def _f(v: Any) -> float:
    return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v] if isinstance(v, str) else float(v)


def _positive(v: Any, what: str) -> float:
    """A gate value: finite and > 0 (``_f`` spellings accepted), else :class:`ValidationError`."""
    try:
        x = _f(v)
    except (KeyError, TypeError, ValueError) as err:
        raise ValidationError(f"approved GB gate {what} = {v!r} is not a number") from err
    if not (math.isfinite(x) and x > 0):
        raise ValidationError(f"approved GB gate {what} = {v!r} is not finite and positive")
    return x


def approved(golden: Path = GOLDEN) -> dict:
    """The approved GB gates (``tolerances_measured.json``, sha256-checked): refuses a file that
    is not APPROVED, a G3 table missing a target, a G3 gate or G4 rule (the well-conditioned PD
    margin, the NumPy abs/rel gates, the pooled JAX gate) that is not finite and positive, a
    per-target JAX G4 table that does not cover the 30 three- and six-month targets with the
    pooled gate, and a -inf rule other than the approved one."""
    tol = _golden_json(golden / TOY / "tolerances_measured.json", golden)
    if not str(tol.get("status", "")).startswith("APPROVED"):
        raise ValidationError(f"GB tolerances are not approved (status {tol.get('status')!r})")
    a = tol.get("approved", {})
    try:
        g3 = {b: {k: _positive(v, f"G3.{b}.{k}") for k, v in a["G3"][b].items()} for b in BACKENDS}
        g4 = a["G4"]
        rules = {k: _positive(g4[name], f"G4.{name}") for k, name in G4_RULES.items()}
        per_target = dict(g4["jax_well_conditioned_per_target"])
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        raise ValidationError(f"malformed approved GB gates (G3 or G4): {err!r}") from err
    want = {f"{c}/{m}" for c in CASES for m in TARGETS[TOY][c]}
    for b in BACKENDS:
        if set(g3[b]) != want:
            raise ValidationError(f"approved G3 {b} gates cover {len(g3[b])} targets, expected the 38")
    stress = {f"{c}/{m}" for c in STRESS_CASES for m in TARGETS[TOY][c]}
    if set(per_target) != stress:
        raise ValidationError(f"approved G4 per-target JAX gates cover {len(per_target)} targets, expected "
                              f"the {len(stress)} of {STRESS_CASES}")
    odd = sorted(k for k, v in per_target.items() if _positive(v, f"G4.jax per-target {k}") != rules["jax_well"])
    if odd:
        raise ValidationError(f"approved G4 per-target JAX gates differ from the pooled {rules['jax_well']:g} at {odd}")
    if not str(g4.get("inf_mask", "")).startswith("identical"):
        raise ValidationError(f"unexpected G4 -inf rule {g4.get('inf_mask')!r}")
    return {"G3": g3, "G4": rules}


def select(cases: Sequence[str] | None, models: Sequence[str] | None) -> list[tuple[str, str]]:
    """The targets in (case, model) order; refuses unknown names and an empty selection."""
    cases = list(CASES if cases is None else cases)
    models = list(MODELS if models is None else models)
    bad = [c for c in cases if c not in CASES] + [m for m in models if m not in MODELS]
    if bad:
        raise ValidationError(f"unknown case/model {bad}")
    out = [(c, m) for c in CASES if c in cases for m in TARGETS[TOY][c] if m in models]
    if not out:
        raise ValidationError("empty selection: no (case, model) target")
    return out


def recording_build(golden: Path = GOLDEN) -> bool:
    """True on the build that recorded the GB goldens (``products.json`` ``build``)."""
    rec = _golden_json(golden / TOY / "products.json", golden)["build"]
    now = build_fingerprint("numpy")                   # python, machine, numpy, scipy, jax, jaxlib
    return all(rec.get(k) is not None and rec.get(k) == now.get(k) for k in GB_RECORDING_KEYS)


def chain_runs(case: str, model: str, tier: str, golden: Path = GOLDEN) -> list[dict]:
    """G3 rows of one target, one entry per run directory: ``run``, ``theta (n, ndim)``,
    ``reference`` (the chain's ``lnlike``), ``lineage_c`` (lineage C NumPy), ``rows``,
    ``prepared_sha256``."""
    root = golden / TOY / "chain_rows" / case
    runs = _golden_json(root / "index.json", golden)["targets"].get(model)
    if not runs:
        raise ValidationError(f"chain_rows/{case}/index.json lists no run for {model}")
    out = []
    for run in runs:
        path = root / f"{run}.{tier}.json"
        if not path.is_file():
            raise ValidationError(f"missing G3 golden {path.relative_to(golden)}" + (
                " (the full tier's 32 rows per run are not captured: capture needs the user's approval)"
                if tier == "full" else ""))
        rec = _golden_json(path, golden)
        if (rec.get("case"), rec.get("model"), rec.get("tier")) != (case, model, tier):
            raise ValidationError(f"{path.name} holds {rec.get('case')}/{rec.get('model')}/{rec.get('tier')}")
        rows = rec["rows"]
        if not rows:
            raise ValidationError(f"{path.name}: no rows")
        out.append({"run": run, "theta": np.array([r["theta"] for r in rows], dtype=np.float64),
                    "reference": np.array([_f(r["lnlike_stored"]) for r in rows]),
                    "lineage_c": np.array([_f(r["lnlike_numpy"]) for r in rows]),
                    "rows": np.array([int(r["row"]) for r in rows]), "prepared_sha256": rec["prepared_sha256"],
                    "lineage_jax": np.array([_f(r["lnlike_jax"]) for r in rows]),
                    "jax_counters": rec.get("jax_counters")})
    return out


def stress_points(case: str, model: str, tier: str, golden: Path = GOLDEN) -> dict:
    """G4 points of one target (3 and 6 months) with their tags and the cache counters; refuses
    an ``lnl_ref`` or ``lnl_cache`` that is NaN or ``+inf`` (``check_reference``: finite or
    exactly ``-inf``, at least one finite).  ``lnl_jax`` feeds a reported value only."""
    path = golden / TOY / "stress" / case / f"{model}.{tier}.json"
    rec = _golden_json(path, golden)
    if (rec.get("case"), rec.get("model"), rec.get("tier")) != (case, model, tier):
        raise ValidationError(f"{path.name} holds {rec.get('case')}/{rec.get('model')}/{rec.get('tier')}")
    pts = rec["points"]
    thr = _f(rec.get("marginal_threshold", math.nan))
    if not pts or not (math.isfinite(thr) and thr > 0):
        raise ValidationError(f"{path.name}: no points or no marginal threshold")
    margin = np.array([_f(p["pd_margin"]) for p in pts])
    if not np.all(np.isfinite(margin)):
        raise ValidationError(f"{path.name}: a non-finite PD margin")
    cache = [p.get("lnl_cache") for p in pts]
    theta = np.array([p["theta"] for p in pts], dtype=np.float64)
    lnl_ref = np.array([_f(p["lnl_ref"]) for p in pts])
    lnl_cache = None if any(v is None for v in cache) else np.array([_f(v) for v in cache])
    ndim = theta.shape[1] if theta.ndim == 2 else -1
    for what, values in (("lnl_ref", lnl_ref), ("lnl_cache", lnl_cache)):
        if values is not None:
            check_reference(theta, values, np.ones(len(pts), bool), ndim, what=f"stress/{case}/{model} {what}")
    return {"theta": theta, "names": [p["name"] for p in pts], "lnl_ref": lnl_ref, "lnl_cache": lnl_cache,
            "margin": margin, "marginal": np.array([bool(p["marginal"]) for p in pts]),
            "source_out": np.array([bool(p["source_out"]) for p in pts]),
            "expected_neg_inf": np.array([bool(p["expected_neg_inf"]) for p in pts]),
            "counters_cache": rec.get("counters_cache"), "counters_jax": rec.get("counters_jax"),
            "lnl_jax": np.array([_f(p["lnl_jax"]) for p in pts]), "threshold": thr,
            "prepared_sha256": rec["prepared_sha256"]}


# --- templates -------------------------------------------------------------------------------

class StoredTemplate:
    """The golden gbonfly store template at the injection (G6): any other source raises."""

    def __init__(self, p8: np.ndarray, h: np.ndarray) -> None:
        self.key, self.h = np.asarray(p8, dtype=np.float64).tobytes(), np.asarray(h)

    def __call__(self, p8: Any) -> np.ndarray:
        if np.asarray(p8, dtype=np.float64).tobytes() != self.key:
            raise KeyError("the stored template is the injection's only")
        return self.h


def stream_gbonfly(rows: list[tuple[str, np.ndarray]]) -> dict:
    """``{case: TemplateTable}``: gbonfly store templates from lineage C (the test-only adapter)."""
    from scripts.golden.gb_templates import streamed_templates

    try:
        tables, _ = streamed_templates(rows)
    except RuntimeError as err:
        raise ValidationError(f"--templates gbonfly: {err}") from err
    return tables


# --- comparisons -----------------------------------------------------------------------------

def stress_compare(values: np.ndarray, pts: dict, reference: np.ndarray, *, abs_gate: float,
                   rel_gate: float, well: float) -> dict:
    """G4 rules for one evaluator (module docstring): masks, gated well-conditioned deltas,
    condition-scaled ill-conditioned deltas, marginal points finite-or-``-inf``.  Fail closed: a
    NaN or ``+inf`` value or reference is invalid, and a NaN delta or gate is over the gate."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    v_inf, r_inf = values == -np.inf, reference == -np.inf
    invalid = ~(np.isfinite(values) | v_inf) | ~(np.isfinite(reference) | r_inf)
    mismatch = (v_inf != r_inf) & ~invalid
    expected = (v_inf != pts["expected_neg_inf"]) & ~pts["marginal"] & ~invalid
    both = ~v_inf & ~r_inf & ~invalid
    delta = np.zeros(values.size)
    delta[both] = np.abs(values[both] - reference[both])
    delta[mismatch | invalid] = np.inf
    well_pts = both & (pts["margin"] >= well)
    ill_pts = both & (pts["margin"] < well) & ~pts["marginal"]
    allowed = np.full(values.size, np.inf)
    allowed[well_pts] = abs_gate + rel_gate * np.abs(reference[well_pts])
    scaled = np.zeros(values.size)
    scaled[ill_pts] = delta[ill_pts] / (EPS * np.abs(reference[ill_pts]) / pts["margin"][ill_pts])
    over = well_pts & ~(delta <= allowed)
    return {"abs_delta": delta, "allowed": allowed, "well": well_pts, "ill": ill_pts,
            "mask_bad": mismatch | expected | invalid,
            "condition_scaled": scaled, "inf_mismatch": int(mismatch.sum()), "expected_mismatch": int(expected.sum()),
            "invalid": int(invalid.sum()), "over": int(over.sum()),
            "well_max_abs": float(delta[well_pts].max(initial=0.0)),
            "condition_scaled_max": float(scaled.max(initial=0.0)),
            "ok": not (mismatch.any() or expected.any() or invalid.any() or over.any())}


def counter_problems(res: dict, expected: dict, *, failed_mask: np.ndarray | None) -> list[str]:
    """``res`` (``sampling.validate.evaluate``) against expected counters (and failed rows)."""
    c = res["counters"]
    out = [f"counter {k} = {c.get(k)}, expected {v}" for k, v in expected.items() if c.get(k) != v]
    if failed_mask is not None and not np.array_equal(np.asarray(res["last_failed"], bool), failed_mask):
        out.append(f"last_failed flags rows {np.flatnonzero(res['last_failed']).tolist()}, "
                   f"expected {np.flatnonzero(failed_mask).tolist()}")
    return out


def bare_counters(lik: Any, n: int, failed: np.ndarray) -> dict:
    """The bare (reference) evaluator's counters: one noise state per row of a
    ``GaussianLikelihood`` (none for the Whittle forms); failed = rows with a failed state."""
    states = 0 if type(lik).__name__ == "FourierWhittle" else n
    k = int(failed.sum())
    return {"evaluations": n, "noise_states": states, "pd_failed_rows": k, "pd_failed_states": k}


def jax_counters(lik: Any, n: int, golden: dict | None) -> dict:
    """The JAX evaluator's expected counters: lineage C's own on the same call (``jax_counters`` /
    ``counters_jax``) for a cached model; rows only for an uncached ``BatchedLikelihood`` (lineage
    C's had no counters; the release counts no noise state for it, F-D3)."""
    if golden is not None:
        return {k: int(golden[k]) for k in COUNTER_KEYS}
    if getattr(lik, "count_states", True):
        raise ValidationError(f"no lineage-C JAX counters to compare a {type(lik).__name__} with")
    return {"evaluations": n, "noise_states": 0, "pd_failed_rows": 0, "pd_failed_states": 0}


def jax_repeat(lik: Any, theta: np.ndarray, first: dict, legacy: np.ndarray) -> dict:
    """A warm repeat (``evaluate``: fresh counters and cache, the programs compiled): its
    ``ms_per_eval``, ``repeat_max_abs`` (largest difference from the gated call) and
    ``lineage_c_jax_max_abs`` (from lineage C's own JAX values) -- reported, never gated."""
    again = evaluate(lik, theta)
    n = len(theta)
    v = np.asarray(first["values"], dtype=float)
    both = np.isfinite(v) & np.isfinite(legacy)
    return {"ms_per_eval": 1e3 * again["seconds"] / n,
            "repeat_max_abs": compare(again["values"], v, np.ones(n, bool), Gate(0.0), exact=True)["max_abs"],
            "lineage_c_jax_max_abs": float(np.max(np.abs(v[both] - legacy[both]), initial=0.0))}


# --- one target ------------------------------------------------------------------------------

def _build(builder: Builder, case: str, model: str, backend: str, options: ModelOptions, template: Any,
           data_root: Path | None):
    """``registry.build`` (the release template, ``options.waveform_backend``) or, with a
    ``template`` (the gbonfly adapter's table), ``galactic_binary.models.build``."""
    if template is None:
        return builder(TOY, case, model, backend=backend, options=options, data_root=data_root)
    return gb_models.build(case, model, backend=backend, options=options, template=template, data_root=data_root)


def validate_g3(lik: Any, case: str, model: str, tier: str, gate: float, *, exact_lineage: bool,
                golden: Path, backend: str = "numpy") -> dict:
    """Every run directory of the target: stored lnlike at ``gate``; lineage C NumPy exactly when
    ``exact_lineage`` (gbonfly adapter, recording build), else reported."""
    runs = chain_runs(case, model, tier, golden)
    entry: dict[str, Any] = {"runs": {}, "problems": []}
    for run in runs:
        if run["theta"].shape[1] != int(lik.ndim):
            raise ValidationError(f"{case}/{model}/{run['run']}: rows have {run['theta'].shape[1]} columns, "
                                  f"the likelihood {lik.ndim}")
        res = evaluate(lik, run["theta"])
        n = len(run["reference"])
        cmp = compare(res["values"], run["reference"], np.ones(n, bool), Gate(gate))
        cl = compare(res["values"], run["lineage_c"], np.ones(n, bool), Gate(gate), exact=exact_lineage)
        expected = (jax_counters(lik, n, run["jax_counters"]) if backend == "jax"
                    else bare_counters(getattr(lik, "lik", lik), n, np.zeros(n, bool)))
        probs = counter_problems(res, expected, failed_mask=np.zeros(n, bool))
        if cmp["invalid"] or cmp["inf_mismatch"]:
            probs.append(f"{cmp['invalid']} NaN/+inf and {cmp['inf_mismatch']} -inf mismatch(es)")
        if not cmp["ok"]:
            probs.append(f"over the G3 gate {gate:g} (max |d| {cmp['max_abs']:.3e})")
        if not cl["ok"]:
            probs.append(("not bitwise against" if exact_lineage else "over the G3 gate against")
                         + f" lineage C NumPy (max |d| {cl['max_abs']:.3e})")
        entry["runs"][run["run"]] = {"rows": run["rows"], "reference": run["reference"], "value": res["values"],
                                     "abs_delta": cmp["abs_delta"], "allowed": cmp["allowed"],
                                     "lineage_c": {"value": run["lineage_c"], "abs_delta": cl["abs_delta"],
                                                   "exact": exact_lineage,
                                                   "bitwise_rows": int(np.sum(cl["abs_delta"] == 0.0))},
                                     "counters": res["counters"], "inf_mismatch": cmp["inf_mismatch"],
                                     "invalid": cmp["invalid"], "seconds": res["seconds"], "problems": probs,
                                     "prepared_sha256": run["prepared_sha256"], "counters_golden": expected,
                                     "ms_per_eval": 1e3 * res["seconds"] / n}
        if backend == "jax":
            entry["runs"][run["run"]].update(jax_repeat(lik, run["theta"], res, run["lineage_jax"]))
        entry["problems"] += [f"{run['run']}: {p}" for p in probs]
    deltas = [d for r in entry["runs"].values() for d in r["abs_delta"]]
    entry.update(max_abs=float(max(deltas)), n_rows=len(deltas), gate=gate,
                 lineage_c_max_abs=float(max(d for r in entry["runs"].values() for d in r["lineage_c"]["abs_delta"])),
                 ok=not entry["problems"])
    return entry


def validate_g4(case: str, model: str, tier: str, *, builder: Builder, backend: str, options: ModelOptions,
                template: Any, rules: dict, golden: Path, data_root: Path | None) -> dict:
    """The stress points: the bare evaluator, and the cache evaluator for the cached models."""
    pts = stress_points(case, model, tier, golden)
    abs_gate, rel_gate = ((rules["numpy_abs"], rules["numpy_rel"]) if backend == "numpy"
                          else (rules["jax_well"], 0.0))
    out: dict[str, Any] = {"names": pts["names"], "margin": pts["margin"], "marginal": pts["marginal"],
                           "lnl_ref": pts["lnl_ref"], "problems": []}
    n = len(pts["names"])
    lik = _build(builder, case, model, backend, options, template, data_root)
    res = evaluate(lik, pts["theta"])
    cmp = stress_compare(res["values"], pts, pts["lnl_ref"], abs_gate=abs_gate, rel_gate=rel_gate, well=rules["well"])
    failed = (pts["lnl_ref"] == -np.inf) & ~pts["source_out"]
    expected = jax_counters(lik, n, pts["counters_jax"]) if backend == "jax" else bare_counters(lik, n, failed)
    probs = counter_problems(res, expected,
                             failed_mask=None if type(lik).__name__ == "FourierWhittle" else failed)
    out["reference"] = {"value": res["values"], "counters": res["counters"], **cmp}
    out["counters_golden"] = expected
    out["ms_per_eval"] = 1e3 * res["seconds"] / n
    if backend == "jax":                      # the default JAX evaluator IS the cache: no second pass
        out.update(jax_repeat(lik, pts["theta"], res, pts["lnl_jax"]))
    del lik
    if backend == "numpy" and model in GB_CACHED:
        if pts["lnl_cache"] is None or pts["counters_cache"] is None:
            raise ValidationError(f"stress/{case}/{model}: a cached model without lnl_cache / counters_cache")
        cache = _build(builder, case, model, backend, dataclasses.replace(options, evaluator="cache"), template,
                       data_root)
        rc = evaluate(cache, pts["theta"])
        cc = stress_compare(rc["values"], pts, pts["lnl_cache"], abs_gate=abs_gate, rel_gate=rel_gate,
                            well=rules["well"])
        probs += [f"cache {p}" for p in counter_problems(
            rc, {k: int(pts["counters_cache"][k]) for k in COUNTER_KEYS}, failed_mask=None)]
        out["cache"] = {"value": rc["values"], "counters": rc["counters"], "counters_golden": pts["counters_cache"],
                        **cc}
        if not cc["ok"]:
            probs.append(f"cache: {cc['inf_mismatch']} mask / {cc['expected_mismatch']} expected-mask / "
                         f"{cc['invalid']} NaN / {cc['over']} over the gate")
    if not cmp["ok"]:
        probs.append(f"reference: {cmp['inf_mismatch']} mask / {cmp['expected_mismatch']} expected-mask / "
                     f"{cmp['invalid']} NaN / {cmp['over']} over the gate")
    if not cmp["well"].any():
        probs.append(f"reference: no well-conditioned point (PD margin >= {rules['well']:g}) was gated")
    out["problems"] = probs
    out["gate"] = {"abs": abs_gate, "rel": rel_gate, "well_conditioned_margin": rules["well"]}
    out["prepared_sha256"] = pts["prepared_sha256"]
    out["ok"] = not probs
    return out


def validate_g6(case: str, model: str, gate: float, *, golden: Path, data_root: Path | None) -> dict:
    """The anchor composition with the golden gbonfly store template (G6 ``quad_logdet``, lnL)."""
    from wdm_models.galactic_binary.prepared import PreparedCase

    g = _golden_json(golden / TOY / f"intermediates_{case}.json", golden)
    with np.load(io.BytesIO(_golden_bytes(golden / TOY / f"intermediates_{case}.npz", golden)),
                 allow_pickle=False) as z:
        h, qls = np.array(z["store_template"]), np.array(z["quad_logdet"])
    a = PreparedCase.load(case, data_root=data_root).arrays
    p8 = np.asarray(a["inject_glass"], dtype=np.float64)
    lik = gb_models.build(case, model, template=StoredTemplate(p8, h), data_root=data_root)
    mu = a["mu_freq"] if model.startswith("whittle") else np.asarray(a["mu_fit"]).ravel()
    theta = np.concatenate([p8, mu])
    ql = np.asarray(lik.quad_logdet(theta))
    lnl = float(lik(theta[None, :])[0])
    ref = qls[g["quad_logdet_models"].index(model)]
    want = _f(g["lnl_at_anchor"][model]["lnl_make_lnlike"])
    items = {}
    for c in range(2):
        for j, key in enumerate(("quad", "logdet")):
            d = abs(float(ql[c, j]) - float(ref[c, j]))
            items[f"c{c}_{key}"] = (d, 1e-12 * abs(float(ref[c, j])))
    items["lnL"] = (abs(lnl - want), gate)
    probs = [f"{k}: |d| {d:.3e} > {g_:.3e}" for k, (d, g_) in items.items() if not (d <= g_)]
    return {"items": {k: {"abs_delta": d, "gate": g_} for k, (d, g_) in items.items()}, "lnL": lnl,
            "reference": want, "bitwise": all(d == 0.0 for d, _ in items.values()), "problems": probs,
            "ok": not probs}


# --- G5 and the onfly audit (per case) --------------------------------------------------------

def golden_prepared_sha256(case: str, golden: Path = GOLDEN) -> set[str]:
    """The prepared file's sha256 as every G3/G4 golden of ``case`` recorded it (the G6
    intermediates record none)."""
    root = golden / TOY
    files = sorted((root / "chain_rows" / case).glob("*.fast.json")) + sorted((root / "stress" / case).glob("*.json"))
    if not files:
        raise ValidationError(f"no G3/G4 golden of {case} records a prepared sha256")
    return {_golden_json(f, golden)["prepared_sha256"] for f in files}


def validate_g5(case: str, models: Sequence[str], *, golden: Path, data_root: Path | None,
                recording: bool) -> dict:
    """Shipped prepared file, describe counts, fourier template sums, prior box (module docstring)."""
    from wdm_models.galactic_binary.prepared import PreparedCase
    from wdm_models.galactic_binary.waveform import GBTemplate

    prod = _golden_json(golden / TOY / "products.json", golden)["cases"][case]
    prep = PreparedCase.load(case, data_root=data_root)
    a = prep.arrays
    manifest = _read_json((data_root or REL / "data") / "MANIFEST.json")["files"][f"galactic_binary/prepared/{case}.npz"]
    items: dict[str, tuple[float, float]] = {}
    sha = _sha256(Path(prep.path))
    items["prepared_sha256"] = (float(len({sha, manifest} | golden_prepared_sha256(case, golden)) - 1), 0.0)
    bad = 0
    for m in models:
        d = describe(TOY, case, m)
        core = {k: v for k, v in d.items() if k not in ("toy", "case", "model", "params")}
        bad += core != prod["describe"][m]
    items["describe"] = (float(bad), 0.0)
    h = np.asarray(GBTemplate(prep, backend="fourier", window="store")(np.asarray(a["inject_glass"])))
    z = np.asarray(a["store_z"])
    lo = int(a["bin_offset"]) - int(a["store_offset"])
    tol = 1e-10 if recording else 1e-8
    for win, sl in (("store", slice(None)), ("window", slice(lo, lo + int(a["n_bins"])))):
        ref = prod["template"][win]
        s2 = float(np.sum(np.abs(h[:, sl]) ** 2))
        zh = float(np.real(np.vdot(h[:, sl], z[:, sl])))
        items[f"template_{win}_sum_h2"] = (abs(s2 - ref["sum_h2"]) / abs(ref["sum_h2"]), tol)
        items[f"template_{win}_re_zh"] = (abs(zh - ref["re_zh"]) / abs(ref["re_zh"]), tol)
    box = np.array_equal(a["prior_lo"], np.asarray(prod["prior"]["lo"])) and \
        np.array_equal(a["prior_hi"], np.asarray(prod["prior"]["hi"]))
    items["prior_box"] = (0.0 if box else 1.0, 0.0)
    probs = [f"{k}: {d:.3e} > {g:.3e}" for k, (d, g) in items.items() if not (d <= g)]
    return {"items": {k: {"observed": d, "gate": g} for k, (d, g) in items.items()}, "sha256": sha,
            "problems": probs, "ok": not probs}


def onfly_audit(case: str, model: str, tier: str, *, golden: Path, data_root: Path | None) -> dict:
    """``onfly`` vs ``fourier`` at the target's G3 rows (reported, never gated): per row the store
    template mismatch ``1 - FF`` and ``lnL_onfly - lnL_fourier``."""
    from wdm_models.galactic_binary.prepared import PreparedCase
    from wdm_models.galactic_binary.waveform import GBTemplate

    prep = PreparedCase.load(case, data_root=data_root)
    tf = GBTemplate(prep, backend="fourier", window="store")
    to = GBTemplate(prep, backend="onfly", window="store")
    lf = gb_models.build(case, model, template=tf, data_root=data_root)
    lo = gb_models.build(case, model, template=to, data_root=data_root)
    one_minus_ff, dlnl = [], []
    for run in chain_runs(case, model, tier, golden):
        for th in run["theta"]:
            hf, ho = np.asarray(tf(th[:8])), np.asarray(to(th[:8]))
            ff = np.real(np.vdot(ho, hf)) / math.sqrt(np.real(np.vdot(ho, ho)) * np.real(np.vdot(hf, hf)))
            one_minus_ff.append(1.0 - ff)
        dlnl += list(lo(run["theta"]) - lf(run["theta"]))
    one_minus_ff, dlnl = np.array(one_minus_ff), np.array(dlnl)
    probs = [] if np.all(np.isfinite(one_minus_ff)) and np.all(np.isfinite(dlnl)) else ["a NaN/inf in the audit"]
    return {"one_minus_ff": one_minus_ff, "dlnl": dlnl, "max_one_minus_ff": float(np.max(one_minus_ff)),
            "max_abs_dlnl": float(np.max(np.abs(dlnl))), "problems": probs, "ok": not probs}


# --- the whole run ---------------------------------------------------------------------------

def run(*, backend: str = "numpy", tier: str = "fast", cases: Sequence[str] | None = None,
        models: Sequence[str] | None = None, goldens: Sequence[str] = ("g3", "g4", "g5", "g6"),
        waveform: str = "fourier", templates: str = "bg", golden: Path = GOLDEN, data_root: Path | None = None,
        builder: Builder | None = None, tables: dict | None = None,
        progress: Callable[[str], None] | None = None) -> dict:
    """Validate every selected target; the report's ``ok`` is the gate.  ``tables``
    (``templates="gbonfly"`` only): streamed gbonfly tables already at hand
    (``scripts.golden.gb_templates.streamed_templates``), e.g. one stream for a whole test session."""
    if backend not in BACKENDS or tier not in TIERS or waveform not in WAVEFORMS or templates not in TEMPLATES:
        raise ValidationError(f"backend/tier/waveform/templates {backend!r}/{tier!r}/{waveform!r}/{templates!r} "
                              f"not in {BACKENDS}/{TIERS}/{WAVEFORMS}/{TEMPLATES}")
    goldens = tuple(dict.fromkeys(goldens))
    if not goldens or any(g not in GOLDENS for g in goldens):
        raise ValidationError(f"goldens {goldens!r} must be a non-empty subset of {GOLDENS}")
    if waveform != "fourier" and any(g in goldens for g in ("g3", "g4")):
        raise ValidationError("the G3/G4 golden checks need --waveform fourier (the chains' templates: gbonfly); "
                              "onfly is the audit's (--golden audit)")
    if templates == "gbonfly" and backend != "numpy":
        raise ValidationError("--templates gbonfly is the NumPy test-only adapter")
    targets = select(cases, models)
    if "g4" in goldens and goldens == ("g4",) and not any(c in STRESS_CASES for c, _ in targets):
        raise ValidationError("empty selection: G4 stress points exist for gb_3mo and gb_6mo only")
    if builder is None:
        from wdm_models.registry import build as builder
    gates = approved(golden)
    if backend == "jax":
        from wdm.backend import enable_x64

        enable_x64()
    fp = build_fingerprint(backend)
    meta = run_meta(backend)
    recording = recording_build(golden)
    options = ModelOptions(waveform_backend=waveform)
    if tables is not None and templates != "gbonfly":
        raise ValidationError("tables= are the gbonfly adapter's (templates='gbonfly')")
    tables = dict(tables or {})
    if templates == "gbonfly" and any(g in goldens for g in ("g3", "g4")) \
            and not all(c in tables for c, _ in targets):
        rows = []
        for c, m in targets:
            if "g3" in goldens:
                rows += [(c, r["theta"][:, :8]) for r in chain_runs(c, m, tier, golden)]
            if "g4" in goldens and c in STRESS_CASES:
                rows.append((c, stress_points(c, m, tier, golden)["theta"][:, :8]))
        tables = stream_gbonfly(rows)
    out_targets: dict[str, dict] = {}
    for case, model in targets:
        t0 = time.perf_counter()
        key = f"{case}/{model}"
        res: dict[str, Any] = {"case": case, "model": model, "problems": []}
        template = tables.get(case)
        for kind in goldens:
            try:
                if kind == "g3":
                    lik = _build(builder, case, model, backend, options, template, data_root)
                    res["g3"] = validate_g3(lik, case, model, tier, gates["G3"][backend][key],
                                            exact_lineage=templates == "gbonfly" and recording, golden=golden,
                                            backend=backend)
                    del lik
                elif kind == "g4":
                    if case not in STRESS_CASES:
                        res["g4"] = {"applicable": False, "ok": True, "problems": []}
                        continue
                    res["g4"] = validate_g4(case, model, tier, builder=builder, backend=backend, options=options,
                                            template=template, rules=gates["G4"], golden=golden,
                                            data_root=data_root)
                elif kind == "g6":
                    res["g6"] = validate_g6(case, model, gates["G3"]["numpy"][key], golden=golden, data_root=data_root)
                elif kind == "audit":
                    res["audit"] = onfly_audit(case, model, tier, golden=golden, data_root=data_root)
            except ValidationError:
                raise
            except Exception as err:                 # a failed target, never a pass
                res[kind] = {"ok": False, "problems": [f"{type(err).__name__}: {err}"],
                             "traceback": traceback.format_exc(limit=6)}
            gc.collect()
        res["seconds"] = time.perf_counter() - t0
        res["ok"] = all(res[k]["ok"] for k in goldens if k in res and k != "g5")
        out_targets[key] = res
        if progress is not None:
            progress(f"{key}: {'ok' if res['ok'] else 'FAIL'} ({res['seconds']:.1f} s)")
    g5: dict[str, dict] = {}
    if "g5" in goldens:
        for case in dict.fromkeys(c for c, _ in targets):
            try:
                g5[case] = validate_g5(case, [m for c, m in targets if c == case], golden=golden,
                                       data_root=data_root, recording=recording)
            except Exception as err:
                g5[case] = {"ok": False, "problems": [f"{type(err).__name__}: {err}"]}
    problems = [f"{k}: {kind}: {p}" for k, v in out_targets.items() for kind in goldens
                if kind in v for p in v[kind].get("problems", [])]
    problems += [f"{c}: g5: {p}" for c, v in g5.items() for p in v["problems"]]
    n_checked = sum(1 for v in out_targets.values() for k in goldens
                    if k in v and v[k].get("applicable", True) and "traceback" not in v[k]) + len(g5)
    if n_checked == 0:
        problems.append("nothing was evaluated")
    failed = sorted([k for k, v in out_targets.items() if not v["ok"]] + [c for c, v in g5.items() if not v["ok"]])
    return {"schema": 1, "family": TOY, "backend": backend, "tier": tier, "goldens": list(goldens),
            "waveform": waveform, "templates": templates, "recording_build": recording,
            "selection": {"cases": sorted({c for c, _ in targets}, key=CASES.index),
                          "models": sorted({m for _, m in targets}, key=MODELS.index)},
            "fingerprint": fp, "run_meta": meta, "gates": {"G3": gates["G3"][backend], "G4": gates["G4"]},
            "targets": out_targets, "g5": g5, "n_targets": len(out_targets), "failed": failed,
            "problems": problems, "ok": not problems and not failed}


def summary_lines(rep: dict) -> list[str]:
    """One line per golden kind: what was checked and the largest delta."""
    lines = []
    t = rep["targets"].values()
    if "g3" in rep["goldens"]:
        rows = [r for v in t if "runs" in v.get("g3", {}) for r in v["g3"]["runs"].values()]
        d = max((float(np.max(r["abs_delta"])) for r in rows), default=math.nan)
        dl = max((float(np.max(r["lineage_c"]["abs_delta"])) for r in rows), default=math.nan)
        bit = sum(r["lineage_c"]["bitwise_rows"] for r in rows)
        lines.append(f"g3 runs={len(rows)} rows={sum(len(r['rows']) for r in rows)} max|d| stored={d:.3e} "
                     f"lineage_C={dl:.3e} bitwise_vs_lineage_C={bit}")
    if "g4" in rep["goldens"]:
        g4 = [v["g4"] for v in t if v.get("g4", {}).get("applicable", True) and "reference" in v.get("g4", {})]
        wm = max((e["reference"]["well_max_abs"] for e in g4), default=math.nan)
        cm = max((e["cache"]["well_max_abs"] for e in g4 if "cache" in e), default=math.nan)
        mm = sum(e["reference"]["inf_mismatch"] + e.get("cache", {}).get("inf_mismatch", 0) for e in g4)
        lines.append(f"g4 targets={len(g4)} well_max|d| reference={wm:.3e} cache={cm:.3e} mask_mismatches={mm}")
    if "g6" in rep["goldens"]:
        g6 = [v["g6"] for v in t if "items" in v.get("g6", {})]
        lines.append(f"g6 targets={len(g6)} bitwise={sum(e['bitwise'] for e in g6)} max|d| lnL="
                     f"{max((e['items']['lnL']['abs_delta'] for e in g6), default=math.nan):.3e}")
    if "g5" in rep["goldens"]:
        lines.append(f"g5 cases={len(rep['g5'])} ok={sum(v['ok'] for v in rep['g5'].values())}")
    if "audit" in rep["goldens"]:
        au = [v["audit"] for v in t if "dlnl" in v.get("audit", {})]
        lines.append(f"audit targets={len(au)} max(1-FF)={max((a['max_one_minus_ff'] for a in au), default=math.nan):.3e} "
                     f"max|dlnL| onfly-fourier={max((a['max_abs_dlnl'] for a in au), default=math.nan):.3e} (reported)")
    return lines


def cpu_determinism() -> str:
    """Append :data:`CPU_DETERMINISM` to ``XLA_FLAGS`` unless an ``--xla_cpu_multi_thread_eigen``
    setting is already there (an explicit choice wins); returns the flags in force.  It acts only
    before JAX starts (``main`` calls it first; importing this module does not import JAX)."""
    flags = os.environ.get("XLA_FLAGS", "").strip()
    if "--xla_cpu_multi_thread_eigen" not in flags:
        flags = f"{flags} {CPU_DETERMINISM}".strip()
        os.environ["XLA_FLAGS"] = flags
    return flags


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:                      # spec section 1, guard 5
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(prog="python -P -m scripts.galactic_binary.validate",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=BACKENDS, default="numpy")
    ap.add_argument("--tier", choices=TIERS, default="fast")
    ap.add_argument("--case", action="append", choices=CASES, help="repeatable; default: all three")
    ap.add_argument("--model", action="append", choices=MODELS, help="repeatable; default: every target")
    ap.add_argument("--golden", nargs="+", choices=GOLDENS, default=["g3", "g4", "g5", "g6"], dest="goldens")
    ap.add_argument("--waveform", choices=WAVEFORMS, default="fourier")
    ap.add_argument("--templates", choices=TEMPLATES, default="bg")
    ap.add_argument("--report", type=Path, help="write the JSON report here (outside the release tree)")
    ap.add_argument("--golden-root", type=Path, default=GOLDEN)
    ap.add_argument("--data-root", type=Path, default=None)
    a = ap.parse_args(argv)
    report = a.report.resolve() if a.report is not None else None
    if report is not None and report.is_relative_to(REL.resolve()):
        ap.error("--report must lie outside the release tree")
    cpu_determinism()                                # spec section 3.4: before JAX starts
    try:
        rep = run(backend=a.backend, tier=a.tier, cases=a.case, models=a.model, goldens=a.goldens,
                  waveform=a.waveform, templates=a.templates, golden=a.golden_root, data_root=a.data_root,
                  progress=lambda s: print(f"[validate] {s}", file=sys.stderr, flush=True))
    except ValidationError as err:
        print(f"[validate] cannot validate: {err}", file=sys.stderr)
        return 1
    if report is not None:
        write_report(report, rep)
    for line in summary_lines(rep):
        print(line)
    verdict = "PASS" if rep["ok"] else "FAIL"
    print(f"[validate] {verdict}: {rep['n_targets']} targets, backend {rep['backend']}, tier {rep['tier']}, "
          f"templates {rep['templates']}, recording build {rep['recording_build']}; failed {rep['failed']}")
    for p in rep["problems"][:20]:
        print(f"[validate]   {p}", file=sys.stderr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
