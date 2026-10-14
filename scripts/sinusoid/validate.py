"""Validate the sinusoid likelihoods against their goldens; evaluates, never samples
(spec sections 6, 7.2 G1/G2 and 7.3).

From the release root, after ``source env/activate.sh`` (which puts only the release root on
PYTHONPATH; this module adds ``packages/{wdm,domainia,wdm_models}/src`` itself, from its own
location, never from the cwd):

    $PY -P -m scripts.sinusoid.validate --backend numpy --tier fast [--case C ...] [--model M ...] \\
        [--golden g1 g2] [--report /outside/the/release/out.json] [--golden-root DIR] [--data-root DIR]

For each selected target ``(case, model)`` the release likelihood
(``wdm_models.registry.build("sinusoid", case, model, backend=...)``) is evaluated at

* **G1**: the reference points of ``golden/sinusoid/references/<case>_<model>_reference.npz``
  (sha256 checked against ``INDEX.json``); fast tier: ``INDEX.json`` ``fast_index`` (truth,
  first tight, first wide, then the first ``-inf`` point or the first stress corner), full
  tier: all 30 (Whittle: 27);
* **G2**: the stored cold-chain rows of ``golden/sinusoid/chain_rows/<case>_<model>.npz``;
  fast tier: positions 0, 1, 19 of the 20 (``INDEX.json`` ``fast_positions``), full tier:
  all 20; phi_0 wrapped first (``priors.periodic(model)``, as ``gpu/run.py::_wrap_phase`` did),

and compared with ``wdm_models.sampling.validate`` at the approved gates of
``golden/sinusoid/tolerances_measured.json`` (``approved.<backend>.references`` /
``.chain_rows``): identical ``-inf`` masks, ``last_failed`` and the counters consistent with
them.  On NumPy, on the build that recorded lineage A (``legacy_eval/A_references_fast.json``
``worker.runtime``), the release is also compared with lineage A's own value at every point
of lineage A's report for the tier (``legacy_eval/A_<kind>_<tier>.json``; captured for the
fast tier only):
G1 **exactly** (the approved rule: bitwise wherever lineage A is bitwise -- all 15 models at
the fast tier), G2 at the approved NumPy *reference* gate of the model (the chain rows'
approved gate is against the chain; the NumPy-vs-NumPy contract of spec section 3.6 holds
there too, and the report shows how many rows are bitwise).

Exit status 0: every check passed.  1: a check failed, or the validation could not be
carried out (empty selection, a missing or altered reference, an unapproved gate, a build
error).  2: usage.  The full tier on NumPy (2232 evaluations; estimated 45-50 min, never
run here: edge-exact ~3-5 s per point) needs the user's approval locally (spec section 6).

JAX (S4): ``--backend jax`` turns on float64 first (``wdm.backend.enable_x64``), gates at
``approved.jax`` (no lineage comparison; lineage B is ``scripts.golden.s4_report``'s), and
records ``run_meta`` (``wdm_models.sampling.device``: device kind, platform, jax/jaxlib, x64,
``XLA_FLAGS``, whether ``--xla_gpu_deterministic_ops=true`` is set (``deterministic_ops``)
and whether ``--xla_cpu_multi_thread_eigen=false`` is (``cpu_single_thread_eigen``)) plus,
per target and golden kind, ``ms_per_eval`` from a second, warm call (fresh counters and
cache, the programs compiled) and ``repeat_max_abs``, its largest difference from the gated
call -- both reported, never gated.  On NumPy ``ms_per_eval`` is the gated call's.  Each
target is released before the next (an edge-exact target holds ~2.8 GB of Phi).

CPU determinism (spec section 3.4): before JAX starts, ``main`` appends
``--xla_cpu_multi_thread_eigen=false`` to ``XLA_FLAGS`` unless an
``--xla_cpu_multi_thread_eigen`` setting is already there (:func:`cpu_determinism`; importing
this module imports no JAX).  On a GPU the flag is a no-op.  The in-process :func:`run` (the
tests) uses its process's flags.

Colab GPU (user-run, S4): see ``COLAB`` below and the README.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

COLAB = """\
On a Colab GPU runtime (T4/L4/A100), the release tree unzipped at /content/wdm_models_scripts
(README.md, "GPU validation of the sinusoid"):
  cd /content/wdm_models_scripts
  pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt
  pip install -q -r env/requirements-colab-cuda12.txt
  export PYTHONPATH=$PWD:$PWD/packages/wdm/src:$PWD/packages/domainia/src:$PWD/packages/wdm_models/src
  export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_FLAGS=--xla_gpu_deterministic_ops=true
  export JAX_PLATFORMS=cuda
  python -P -m scripts.sinusoid.validate --backend jax --tier fast \\
      --report /content/drive/MyDrive/wdm_release/validate_sinusoid_jax_gpu_fast.json
"""

REL = Path(__file__).resolve().parents[2]
#: The release sources, in front of sys.path unless already there (explicit and
#: ``__file__``-relative, so ``python -P`` stays cwd-free: spec section 1, guard 5).
SOURCES = tuple(REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models"))
for _src in reversed(SOURCES):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from wdm_models.registry import MODELS, TARGETS  # noqa: E402  (after the path bootstrap)
from wdm_models.sampling.device import CPU_DETERMINISM, run_meta  # noqa: E402
from wdm_models.sampling.validate import (Gate, ValidationError, build_fingerprint, check_counters,  # noqa: E402
                                          check_reference, compare, evaluate, same_build, wrap_periodic,
                                          write_report)
from wdm_models.sinusoid import priors  # noqa: E402

GOLDEN = REL / "golden"
TOY = "sinusoid"
CASES: tuple[str, ...] = tuple(TARGETS[TOY])
BACKENDS = ("numpy", "jax")
TIERS = ("fast", "full")
GOLDENS = ("g1", "g2")
#: golden kind -> (tolerances section, lineage-A report stem)
KIND = {"g1": ("references", "references"), "g2": ("chain_rows", "chain_rows")}

Builder = Callable[..., Any]


# --- golden readers ------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValidationError(f"missing golden file {path}")
    return json.loads(path.read_text())


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValidationError(f"missing golden file {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_float(v: Any) -> float:
    return {"inf": np.inf, "-inf": -np.inf, "nan": np.nan}[v] if isinstance(v, str) else float(v)


def approved_gates(backend: str, kind: str, golden: Path = GOLDEN) -> dict[str, Gate]:
    """``model -> Gate`` from ``tolerances_measured.json`` ``approved.<backend>.<kind>``;
    refuses a file whose status is not APPROVED or whose -inf rule is not "identical"."""
    tol = _read_json(golden / TOY / "tolerances_measured.json")
    if not str(tol.get("status", "")).startswith("APPROVED"):
        raise ValidationError(f"sinusoid tolerances are not approved (status {tol.get('status')!r})")
    try:
        block = tol["approved"][backend]
        if block["inf_mask"] != "identical":
            raise ValidationError(f"unexpected -inf rule {block['inf_mask']!r}")
        table = block[kind]
    except KeyError as err:
        raise ValidationError(f"no approved {backend} {kind} gates") from err
    missing = [m for m in MODELS if m not in table]
    if missing:
        raise ValidationError(f"approved {backend} {kind} gates miss {missing}")
    return {m: Gate(_to_float(table[m]["atol"]), _to_float(table[m]["rtol"])) for m in MODELS}


def select(cases: Sequence[str] | None, models: Sequence[str] | None) -> list[tuple[str, str]]:
    """The targets in (case, model) order; refuses an unknown name and an empty selection."""
    cases = list(CASES if cases is None else cases)
    models = list(MODELS if models is None else models)
    bad = [c for c in cases if c not in CASES] + [m for m in models if m not in MODELS]
    if bad:
        raise ValidationError(f"unknown case/model {bad}")
    targets = [(c, m) for c in CASES if c in cases for m in TARGETS[TOY][c] if m in models]
    if not targets:
        raise ValidationError("empty selection: no (case, model) target")
    return targets


def reference_points(case: str, model: str, tier: str, golden: Path = GOLDEN) -> dict:
    """G1 points of one target: ``theta, reference, near, index``."""
    refs = golden / TOY / "references"
    name = f"{case}_{model}_reference.npz"
    entry = _read_json(refs / "INDEX.json")["files"].get(name)
    if entry is None:
        raise ValidationError(f"{name} is not in references/INDEX.json")
    path = refs / name
    if not path.is_file():
        raise ValidationError(f"missing reference {path}")
    if _sha256(path) != entry["sha256"]:
        raise ValidationError(f"{path}: sha256 differs from references/INDEX.json")
    z = _read_npz(path)
    idx = np.asarray(entry["fast_index"] if tier == "fast" else np.arange(len(z["loglike"])), dtype=int)
    return {"theta": z["theta"][idx], "reference": z["loglike"][idx], "near": z["near"][idx], "index": idx}


def chain_rows(case: str, model: str, tier: str, golden: Path = GOLDEN) -> dict:
    """G2 rows of one target: ``theta`` (phi_0 wrapped), ``reference`` (the chain's lnlike),
    ``near`` (all True: chain rows are gated by atol alone), ``index`` (row numbers)."""
    root = golden / TOY / "chain_rows"
    index = _read_json(root / "INDEX.json")
    key = f"{case}_{model}"
    if key not in index["runs"]:
        raise ValidationError(f"{key} is not in chain_rows/INDEX.json")
    z = _read_npz(root / f"{key}.npz")
    pos = np.asarray(index["fast_positions"] if tier == "fast" else np.arange(len(z["lnlike"])), dtype=int)
    return {"theta": wrap_periodic(z["theta"][pos], priors.periodic(model)), "reference": z["lnlike"][pos],
            "near": np.ones(len(pos), dtype=bool), "index": z["row_index"][pos]}


def lineage_a(kind: str, tier: str, golden: Path = GOLDEN) -> dict | None:
    """Lineage A's report for ``kind`` at ``tier`` (``None`` when it was not captured)."""
    path = golden / TOY / "legacy_eval" / f"A_{KIND[kind][1]}_{tier}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def recording_runtime(golden: Path = GOLDEN) -> dict:
    """The build that recorded lineage A (``A_references_fast.json`` ``worker.runtime``)."""
    return _read_json(golden / TOY / "legacy_eval" / "A_references_fast.json")["worker"]["runtime"]


# --- one target --------------------------------------------------------------------------

def _points(kind: str, case: str, model: str, tier: str, golden: Path) -> dict:
    return (reference_points if kind == "g1" else chain_rows)(case, model, tier, golden)


def _lineage_values(report: dict | None, case: str, model: str, kind: str, index: np.ndarray) -> np.ndarray | None:
    if report is None:
        return None
    entry = report["points"].get(f"{case}__{model}")
    if entry is None:
        raise ValidationError(f"lineage A's {kind} report holds no {case}/{model} entry")
    stored = np.asarray(entry["index"] if kind == "g1" else entry["row_index"], dtype=int)
    if not np.array_equal(stored, np.asarray(index, dtype=int)):
        raise ValidationError(f"lineage A {kind} {case}/{model} holds points {stored.tolist()}, "
                              f"validating {np.asarray(index).tolist()}")
    return np.array([_to_float(v) for v in entry["value"]])


def validate_target(case: str, model: str, *, backend: str, tier: str, goldens: Sequence[str],
                    gates: dict[str, dict[str, Gate]], lineage: dict[str, dict | None],
                    lineage_gates: dict[str, Gate] | None = None, golden: Path = GOLDEN,
                    data_root: Path | None = None, builder: Builder | None = None, options: Any = None) -> dict:
    """Every comparison of one target; ``out[kind]["ok"]`` and ``out["ok"]`` summarise them.
    ``lineage[kind]``: lineage A's report (or None); G1 is compared with it exactly, G2 at
    ``lineage_gates[model]`` (the NumPy reference gates; None: exactly)."""
    if builder is None:
        from wdm_models.registry import build as builder
    out: dict[str, Any] = {"case": case, "model": model, "problems": []}
    kw = {"backend": backend, "data_root": data_root}
    if options is not None:
        kw["options"] = options
    try:
        t0 = time.perf_counter()
        lik = builder(TOY, case, model, **kw)
        out["build_seconds"] = time.perf_counter() - t0
    except Exception as err:                       # report it as a failed target, never a pass
        out["problems"].append(f"build failed: {type(err).__name__}: {err}")
        out["traceback"] = traceback.format_exc(limit=4)
        out["ok"] = False
        return out
    whittle = model.startswith("whittle")
    for kind in goldens:
        pts = _points(kind, case, model, tier, golden)
        check_reference(pts["theta"], pts["reference"], pts["near"], int(lik.ndim), what=f"{case}/{model} {kind}")
        try:
            res = evaluate(lik, pts["theta"])
        except Exception as err:
            out[kind] = {"ok": False, "problems": [f"evaluation failed: {type(err).__name__}: {err}"],
                         "index": pts["index"]}
            continue
        cmp = compare(res["values"], pts["reference"], pts["near"], gates[kind][model])
        problems = check_counters(res, pts["reference"], noise_states=0 if whittle else len(pts["reference"])) \
            if backend == "numpy" else check_counters(res, pts["reference"], noise_states=None)
        entry = {"index": pts["index"], "near": pts["near"], "reference": pts["reference"],
                 "value": res["values"], "last_failed": res["last_failed"], "counters": res["counters"],
                 "seconds": res["seconds"], "gate": {"atol": gates[kind][model].atol, "rtol": gates[kind][model].rtol},
                 **{k: cmp[k] for k in ("abs_delta", "rel_delta", "allowed", "passed", "inf_mismatch", "invalid",
                                        "max_abs", "near_max_abs", "stress_max_abs", "stress_max_rel")},
                 "problems": problems, "ms_per_eval": 1e3 * res["seconds"] / len(pts["reference"])}
        if backend == "jax":
            # the gated call compiled; a warm repeat (fresh counters and cache) times the
            # steady state and shows run-to-run agreement -- reported, never gated
            try:
                again = evaluate(lik, pts["theta"])
            except Exception as err:
                problems.append(f"repeat evaluation failed: {type(err).__name__}: {err}")
            else:
                entry["ms_per_eval"] = 1e3 * again["seconds"] / len(pts["reference"])
                entry["repeat_max_abs"] = compare(again["values"], res["values"], pts["near"],
                                                  Gate(0.0), exact=True)["max_abs"]
        if cmp["invalid"]:
            problems.append(f"{cmp['invalid']} NaN/+inf value(s)")
        if cmp["inf_mismatch"]:
            problems.append(f"-inf mask differs at {cmp['inf_mismatch']} point(s)")
        if not cmp["ok"]:
            problems.append(f"{int((~cmp['passed']).sum())} point(s) over the gate (max |d| {cmp['max_abs']:.3e})")
        ref_a = _lineage_values(lineage.get(kind), case, model, kind, pts["index"])
        if ref_a is not None:
            exact = kind == "g1" or lineage_gates is None
            cmp_a = compare(res["values"], ref_a, pts["near"],
                            gates[kind][model] if exact else lineage_gates[model], exact=exact)
            entry["lineage_A"] = {"value": ref_a, "exact": exact, "abs_delta": cmp_a["abs_delta"],
                                  "allowed": cmp_a["allowed"], "passed": cmp_a["passed"], "max_abs": cmp_a["max_abs"],
                                  "bitwise_points": int(np.sum(cmp_a["abs_delta"] == 0.0))}
            if not cmp_a["ok"]:
                what = "not bitwise against" if exact else "over the NumPy reference gate against"
                problems.append(f"{what} lineage A at {int((~cmp_a['passed']).sum())} point(s) "
                                f"(max |d| {cmp_a['max_abs']:.3e})")
        entry["ok"] = not problems
        out[kind] = entry
    out["ok"] = not out["problems"] and all(out[k]["ok"] for k in goldens)
    return out


# --- the whole run ------------------------------------------------------------------------

def _summary(targets: dict, kind: str) -> dict:
    models: dict[str, dict] = {}
    for t in targets.values():
        e = t.get(kind)
        if e is None or "abs_delta" not in e:
            continue
        m = models.setdefault(t["model"], {"near_max_abs": 0.0, "stress_max_abs": 0.0, "stress_max_rel": 0.0,
                                           "inf_mismatch": 0, "n_points": 0, "bitwise_points": 0})
        for key in ("near_max_abs", "stress_max_abs", "stress_max_rel"):
            m[key] = max(m[key], e[key])
        m["inf_mismatch"] += e["inf_mismatch"]
        m["n_points"] += len(e["abs_delta"])
        m["bitwise_points"] += int(np.sum(np.asarray(e["abs_delta"]) == 0.0))
    return models


def run(*, backend: str = "numpy", tier: str = "fast", cases: Sequence[str] | None = None,
        models: Sequence[str] | None = None, goldens: Sequence[str] = GOLDENS, golden: Path = GOLDEN,
        data_root: Path | None = None, builder: Builder | None = None, options: Any = None,
        progress: Callable[[str], None] | None = None) -> dict:
    """Validate every selected target; the report's ``ok`` is the gate."""
    if backend not in BACKENDS or tier not in TIERS:
        raise ValidationError(f"backend {backend!r} / tier {tier!r} not in {BACKENDS} / {TIERS}")
    goldens = tuple(dict.fromkeys(goldens))
    if not goldens or any(g not in GOLDENS for g in goldens):
        raise ValidationError(f"goldens {goldens!r} must be a non-empty subset of {GOLDENS}")
    targets = select(cases, models)
    gates = {k: approved_gates(backend, KIND[k][0], golden) for k in goldens}
    if backend == "jax":
        from wdm.backend import enable_x64

        enable_x64()                 # spec section 3.4: scripts turn float64 on first
    fp = build_fingerprint(backend)
    meta = run_meta(backend)
    lineage: dict[str, dict | None] = {k: None for k in goldens}
    lineage_gates = None
    recording = False
    if backend == "numpy":
        recording = same_build(fp, recording_runtime(golden))
        if recording:
            lineage = {k: lineage_a(k, tier, golden) for k in goldens}
            lineage_gates = approved_gates("numpy", "references", golden)
    out_targets: dict[str, dict] = {}
    for case, model in targets:
        t0 = time.perf_counter()
        res = validate_target(case, model, backend=backend, tier=tier, goldens=goldens, gates=gates,
                              lineage=lineage, lineage_gates=lineage_gates, golden=golden,
                              data_root=data_root, builder=builder, options=options)
        res["seconds"] = time.perf_counter() - t0
        out_targets[f"{case}/{model}"] = res
        gc.collect()                     # release the target (its likelihood is out of scope)
        if progress is not None:
            progress(f"{case}/{model}: {'ok' if res['ok'] else 'FAIL'} ({res['seconds']:.1f} s)")
    failed = [k for k, v in out_targets.items() if not v["ok"]]
    n_points = sum(len(v[k]["abs_delta"]) for v in out_targets.values() for k in goldens
                   if k in v and "abs_delta" in v[k])
    problems = [f"{k}: {p}" for k, v in out_targets.items()
                for p in v["problems"] + [q for g in goldens if g in v for q in v[g].get("problems", [])]]
    if n_points == 0:
        problems.append("no point was evaluated")
    return {"schema": 1, "family": TOY, "backend": backend, "tier": tier, "goldens": list(goldens),
            "selection": {"cases": sorted({c for c, _ in targets}, key=CASES.index),
                          "models": sorted({m for _, m in targets}, key=MODELS.index)},
            "fingerprint": fp, "run_meta": meta, "recording_build": recording,
            "bitwise_against": {k: lineage[k] is not None for k in goldens},
            "gates": {k: {m: {"atol": g.atol, "rtol": g.rtol} for m, g in gates[k].items()} for k in goldens},
            "targets": out_targets, "models": {k: _summary(out_targets, k) for k in goldens},
            "n_targets": len(out_targets), "n_points": n_points, "failed_targets": failed,
            "problems": problems, "ok": not failed and n_points > 0 and not problems}


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
    ap = argparse.ArgumentParser(prog="python -P -m scripts.sinusoid.validate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=BACKENDS, default="numpy")
    ap.add_argument("--tier", choices=TIERS, default="fast")
    ap.add_argument("--case", action="append", choices=CASES, help="repeatable; default: all three")
    ap.add_argument("--model", action="append", choices=MODELS, help="repeatable; default: all fifteen")
    ap.add_argument("--golden", nargs="+", choices=GOLDENS, default=list(GOLDENS), dest="goldens")
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
                  golden=a.golden_root, data_root=a.data_root,
                  progress=lambda s: print(f"[validate] {s}", file=sys.stderr, flush=True))
    except ValidationError as err:
        print(f"[validate] cannot validate: {err}", file=sys.stderr)
        return 1
    if report is not None:
        write_report(report, rep)
    for kind in rep["goldens"]:
        for model, m in rep["models"][kind].items():
            print(f"{kind} {model:12s} points={m['n_points']:3d} bitwise={m['bitwise_points']:3d} "
                  f"near_max_abs={m['near_max_abs']:.3e} stress_max_abs={m['stress_max_abs']:.3e} "
                  f"inf_mismatch={m['inf_mismatch']}")
    verdict = "PASS" if rep["ok"] else "FAIL"
    print(f"[validate] {verdict}: {rep['n_targets']} targets, {rep['n_points']} points, backend {rep['backend']}, "
          f"tier {rep['tier']}, recording build {rep['recording_build']}; failed {rep['failed_targets']}; "
          f"device {rep['run_meta']['device_kind']}")
    for p in rep["problems"][:20]:
        print(f"[validate]   {p}", file=sys.stderr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
