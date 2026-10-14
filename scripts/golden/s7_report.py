"""S7 completion report: the galactic binary on JAX, release level, fail-closed (spec section 7.6, S7).

    $PY -m scripts.golden.s7_report [--full] [--validate JAX_REPORT] [--junit XML] [--out PATH]

Three sources, each required:

1. **pytest** -- one subprocess (root config, cwd = the release root, junit XML) over :data:`TESTS`,
   the F-D tests: the legacy JAX evaluators and the JAX column (F-D1), the lag tables as constants
   (F-D2), the JAX build (F-D3), prior sanitising (F-D4), G3/G4 on CPU-JAX (F-D5), the counters
   against lineage C (F-D6), batched = loop and determinism (F-D7), per-state memory and the HLO
   budget (F-D8), validate.py on JAX (F-D9) and the Colab doc (F-D10).  Every test must pass; none
   may skip; every :data:`REQUIRED` node pattern must match enough passed nodes (default /
   ``--full`` counts).  The pytest subprocess does NOT get the CPU determinism flag (validate.py's
   ``main`` sets it for its own runs; the tests choose their XLA flags themselves).
2. **validate.py --backend jax --tier fast --golden g3 g4** (run here, or ``--validate`` a report
   from elsewhere, e.g. the user's Colab GPU run): its own verdict (``ok``, nothing failed, no
   problem) and, independently, the 38 targets with every G3 run directory of the golden index
   (3 finite rows each within the approved G3 JAX gate of the stored lnlike, ``ms_per_eval`` finite
   and positive), G4 entries for the 30 targets at 3 and 6 months (none applicable at 12) with the
   golden ``-inf`` masks and within the approved G4 JAX gate (8.32e-8) at the well-conditioned
   points, ``waveform == "fourier"``, ``run_meta`` with x64 on and the determinism flag of the
   device (spec section 3.4; GPU: ``deterministic_ops`` true, i.e.
   ``--xla_gpu_deterministic_ops=true``; CPU: ``--xla_cpu_multi_thread_eigen=false`` in
   ``XLA_FLAGS`` and ``cpu_single_thread_eigen`` true -- validate.py's ``main`` sets it).  The
   device is required too (``expect_device``): ``--validate`` is the user's GPU sign-off and
   refuses any report whose ``run_meta.device_kind`` is not ``"gpu"`` (a CPU fallback included);
   the run made here is the local CPU-JAX gate and must say ``"cpu"``.
3. with ``--full``: the ``slow`` tests, **validate.py --backend numpy --tier fast --golden g3 g4**
   (the S6 gate after S7's changes) and **s2_report --full --strict** (domainia changed in F-D2).

The deltas, recomputed here from the report against the goldens (never trusting its own
summary): per model, max |JAX - stored lnlike| (G3), max |JAX - lineage C's own JAX value| (G3,
``lnlike_jax``), the G4 well-conditioned max |JAX - lnl_ref| (the approved margin, 1e-6), the
reported condition-scaled max, the ``-inf`` count and the median warm ``ms_per_eval`` per case
(the CPU timing smoke: reported, never gated).  Exit 1 on any problem; the JSON report goes to
``--out`` (outside the release tree) and stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.golden.goldens_io import read_json, to_float
from scripts.golden.paths import GOLDEN, REL, tmp_root

GB = GOLDEN / "galactic_binary"
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
STRESS_CASES = ("gb_3mo", "gb_6mo")
SRC = tuple(REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models"))
#: The CPU-JAX determinism flag (spec section 3.4; set by ``scripts/galactic_binary/validate.py main``).
CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"
DEVICES = ("cpu", "gpu")
#: The F-D test files.
TESTS = (*(f"tests/golden/test_s7_{n}.py" for n in ("legacy_jax", "build_jax", "prior_sanitize", "gb_jax",
                                                   "counters_legacy", "batched_determinism", "validate_jax",
                                                   "colab_doc")),
         "tests/parity/test_s7_gb_jax_states.py", "packages/domainia/tests/test_dom_exact_fourier_constants.py")
#: (node regex over ``<module>::<name>``, minimum passed nodes without / with ``--full``).
REQUIRED: tuple[tuple[str, int, int], ...] = (
    (r"test_s7_legacy_jax::test_the_table_covers_the_38_targets", 1, 1),
    (r"test_s7_legacy_jax::test_the_table_is_lineage_c\[.*\]", 1, 3),
    (r"test_s7_legacy_jax::test_the_jax_column_is_the_legacy_evaluator\[.*\]", 2, 2),
    (r"test_s7_legacy_jax::test_reference_on_jax_is_the_numpy_column", 1, 1),
    (r"test_dom_exact_fourier_constants::.*", 4, 4),
    (r"test_s7_build_jax::test_every_target_builds_inside_its_legacy_evaluator\[.*\]", 15, 38),
    (r"test_s7_build_jax::test_a_template_that_is_not_the_pytree_is_refused_on_jax", 1, 1),
    (r"test_s7_build_jax::test_the_release_default_onfly_builds_on_jax", 1, 1),
    (r"test_s7_build_jax::test_the_padded_store_layout_is_bitwise_the_ragged_one_on_numpy", 1, 1),
    (r"test_s7_prior_sanitize::.*", 7, 7),
    (r"test_s7_gb_jax::test_g3_fast_tier_on_cpu_jax\[.*\]", 30, 38),
    (r"test_s7_gb_jax::test_g4_stress_fast_tier_on_cpu_jax\[.*\]", 30, 30),
    (r"test_s7_gb_jax::test_gate_catches_a_wrong_target", 1, 1),
    (r"test_s7_counters_legacy::test_counters_masks_and_values_equal_lineage_c\[.*\]", 3, 9),
    (r"test_s7_batched_determinism::test_one_call_with_r_rows_equals_the_row_loop\[.*\]", 4, 8),
    (r"test_s7_batched_determinism::test_the_row_loop_option_is_bitwise_the_loop\[.*\]", 1, 2),
    (r"test_s7_batched_determinism::test_repeats_and_fresh_builds_are_bitwise_identical\[.*\]", 3, 3),
    (r"test_s7_batched_determinism::test_two_fresh_processes_are_bitwise_identical", 0, 1),
    (r"test_s7_gb_jax_states::test_noise_state_memory_is_within_1p5x_of_legacy\[.*\]", 9, 23),
    (r"test_s7_gb_jax_states::test_the_memory_gate_bites", 1, 1),
    (r"test_s7_gb_jax_states::test_gb_programs_carry_no_large_literal\[.*\]", 9, 12),
    (r"test_s7_validate_jax::.*", 3, 4),
    (r"test_s7_colab_doc::.*", 2, 2),
)


def targets() -> dict[str, tuple[str, ...]]:
    """case -> models, from the G3 golden indices (the 38 targets)."""
    return {c: tuple(sorted(read_json(GB / "chain_rows" / c / "index.json")["targets"])) for c in CASES}


def collect(xml_path: Path) -> tuple[dict[str, int], set[str]]:
    """Outcome counts and the passed nodes (``<module>::<name>``) of a junit report."""
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    passed: set[str] = set()
    for case in ET.parse(xml_path).iter("testcase"):
        counts["tests"] += 1
        bad = False
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                counts[key] += 1
                bad = True
        if not bad:
            passed.add(f"{(case.get('classname') or '').rsplit('.', 1)[-1]}::{case.get('name')}")
    return counts, passed


def required_problems(passed: set[str], *, full: bool) -> list[str]:
    out = []
    for pattern, lo, hi in REQUIRED:
        need = hi if full else lo
        n = sum(1 for node in passed if re.fullmatch(pattern, node))
        if n < need:
            out.append(f"{pattern}: {n} passed node(s), need {need}")
    return out


def _f(x) -> float:
    """A report number (``write_report`` spells the non-finite ones as strings); anything else
    is NaN, which every check below refuses."""
    try:
        return to_float(x) if isinstance(x, str) else float(x)
    except (KeyError, TypeError, ValueError):
        return math.nan


def _meta_problems(rep: dict, backend: str, expect_device: str | None) -> list[str]:
    meta = rep.get("run_meta")
    if not isinstance(meta, dict):
        return ["validate: no run_meta"]
    if backend != "jax":
        return []
    out = []
    if meta.get("x64") is not True:
        out.append(f"validate: run_meta x64 is {meta.get('x64')!r}")
    if expect_device is not None and meta.get("device_kind") != expect_device:
        out.append(f"validate: run_meta device_kind {meta.get('device_kind')!r}; need {expect_device!r}")
    if meta.get("device_kind") == "gpu":
        if meta.get("deterministic_ops") is not True:
            out.append("validate: a GPU run without XLA_FLAGS=--xla_gpu_deterministic_ops=true")
    elif meta.get("cpu_single_thread_eigen") is not True or CPU_FLAG not in (meta.get("XLA_FLAGS") or ""):
        out.append(f"validate: a CPU run without XLA_FLAGS={CPU_FLAG} (XLA_FLAGS {meta.get('XLA_FLAGS')!r})")
    return out


def summarise_validate(rep: dict, *, backend: str = "jax",
                       expect_device: str | None = None) -> tuple[dict, list[str]]:
    """Per-model deltas of a ``scripts/galactic_binary/validate.py`` report, recomputed against the
    goldens, and its problems (fail closed).  On JAX the report's device must carry its
    determinism flag (GPU: ``deterministic_ops``; CPU: :data:`CPU_FLAG`) and, with
    ``expect_device`` (``"gpu"``: the user's ``--validate`` sign-off; ``"cpu"``: the local gate),
    be that device."""
    if expect_device is not None and expect_device not in DEVICES:
        raise ValueError(f"expect_device must be one of {DEVICES} or None; got {expect_device!r}")
    problems = []
    if rep.get("backend") != backend or rep.get("tier") != "fast":
        problems.append(f"validate report is backend {rep.get('backend')!r} tier {rep.get('tier')!r}; "
                        f"need {backend!r} 'fast'")
    if rep.get("waveform") != "fourier" or not {"g3", "g4"} <= set(rep.get("goldens") or ()):
        problems.append(f"validate: waveform {rep.get('waveform')!r} goldens {rep.get('goldens')!r}; need "
                        "'fourier' with g3 and g4")
    if rep.get("ok") is not True:
        problems.append("validate: ok is not true")
    if rep.get("failed"):
        problems.append(f"validate: failed targets {rep['failed'][:5]}")
    if rep.get("problems"):
        problems.append(f"validate: {len(rep['problems'])} problem(s), e.g. {rep['problems'][0]}")
    problems += _meta_problems(rep, backend, expect_device)
    approved = read_json(GB / "tolerances_measured.json")["approved"]
    g3_gates, g4_gates = approved["G3"][backend], approved["G4"]["jax_well_conditioned_per_target"]
    well = float(approved["G4"]["well_conditioned_margin"])
    want = {f"{c}/{m}" for c, ms in targets().items() for m in ms}
    got = rep.get("targets") or {}
    if set(got) != want:
        problems.append(f"validate targets: {len(got)}; missing {sorted(want - set(got))[:5]}, "
                        f"extra {sorted(set(got) - want)[:5]}")
    models: dict[str, dict] = {}
    for tag, t in sorted(got.items()):
        if tag not in want:
            continue
        case, model = tag.split("/")
        m = models.setdefault(model, {"g3_max_abs": 0.0, "g3_oracle_max_abs": 0.0, "g4_well_max_abs": 0.0,
                                      "g4_condition_scaled_max": 0.0, "g4_inf_points": 0, "ms_per_eval": {}})
        g3 = t.get("g3") if isinstance(t, dict) else None
        if not isinstance(g3, dict) or not isinstance(g3.get("runs"), dict) or not g3["runs"]:
            problems.append(f"{tag}: no g3 runs")
            continue
        index = read_json(GB / "chain_rows" / case / "index.json")["targets"][model]
        if sorted(g3["runs"]) != sorted(index):
            problems.append(f"{tag}: g3 runs {sorted(g3['runs'])} are not the golden's {sorted(index)}")
        ms = []
        for run, r in sorted(g3["runs"].items()):
            if run not in index:
                continue
            gold = read_json(GB / "chain_rows" / case / f"{run}.fast.json")["rows"]
            r = r if isinstance(r, dict) else {}
            v = [_f(x) for x in r["value"]] if isinstance(r.get("value"), list) else []
            if len(v) != len(gold) or not all(math.isfinite(x) for x in v):
                problems.append(f"{tag}/{run}: {len(v)} values, not {len(gold)} finite ones")
                continue
            d = max(abs(a - g["lnlike_stored"]) for a, g in zip(v, gold))
            if not d <= float(g3_gates[tag]):                     # the approved gate, independently
                problems.append(f"{tag}/{run}: max |JAX - stored lnlike| {d:.3e} over the G3 gate {g3_gates[tag]:.1e}")
            m["g3_max_abs"] = max(m["g3_max_abs"], d)
            m["g3_oracle_max_abs"] = max(m["g3_oracle_max_abs"], *(abs(a - g["lnlike_jax"]) for a, g in zip(v, gold)))
            x = _f(r.get("ms_per_eval", math.nan))
            if not (math.isfinite(x) and x > 0):
                problems.append(f"{tag}/{run}: ms_per_eval {x}")
            else:
                ms.append(x)
        m["ms_per_eval"][case] = statistics.median(ms) if ms else math.nan
        g4 = t.get("g4")
        if case not in STRESS_CASES:
            if not (isinstance(g4, dict) and g4.get("applicable") is False):
                problems.append(f"{tag}: a G4 entry at 12 months")
            continue
        ref = g4.get("reference") if isinstance(g4, dict) else None
        if not isinstance(ref, dict) or not isinstance(ref.get("value"), list):
            problems.append(f"{tag}: no g4 values")
            continue
        pts = read_json(GB / "stress" / case / f"{model}.fast.json")["points"]
        v = [_f(x) for x in ref["value"]]
        if len(v) != len(pts) or any(math.isnan(x) or x == math.inf for x in v):
            problems.append(f"{tag}: g4 values {len(v)} (need {len(pts)}) / NaN or +inf")
            continue
        for a, p in zip(v, pts):
            neg = a == -math.inf
            # validate's rule: the reference's -inf mask everywhere, expected_neg_inf off the marginal points
            if neg != (p["lnl_ref"] == -math.inf) or (not p["marginal"] and neg != bool(p["expected_neg_inf"])):
                problems.append(f"{tag}: g4 {p['name']} -inf mask")
            elif not neg and not p["marginal"] and p["pd_margin"] >= well:
                d = abs(a - p["lnl_ref"])
                if backend == "jax" and not d <= float(g4_gates[tag]):
                    problems.append(f"{tag}: g4 {p['name']} |JAX - lnl_ref| {d:.3e} over the G4 gate "
                                    f"{g4_gates[tag]:.2e}")
                m["g4_well_max_abs"] = max(m["g4_well_max_abs"], d)
            m["g4_inf_points"] += int(neg)
        m["g4_condition_scaled_max"] = max(m["g4_condition_scaled_max"], _f(ref.get("condition_scaled_max", 0.0)))
    return dict(sorted(models.items())), problems


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.pathsep.join(map(str, (REL, *SRC))), PYTHONDONTWRITEBYTECODE="1",
               JAX_ENABLE_X64="true")
    return env


def run_pytest(patterns: tuple[str, ...], *, full: bool) -> Path:
    missing = [p for p in patterns if not (REL / p).is_file()]
    if missing:
        raise SystemExit(f"no test file {missing} (section F-D)")
    xml = tmp_root() / f"s7_report_{uuid.uuid4().hex[:12]}.xml"
    cmd = [sys.executable, "-P", "-m", "pytest", "-c", "pytest.ini", "-p", "no:cacheprovider", "-q",
           "-o", "junit_family=xunit1", f"--junitxml={xml}", *(["-m", "not golden_full"] if full else []),
           *patterns]
    rc = subprocess.run(cmd, cwd=str(REL), env=_env(), check=False).returncode
    if not xml.is_file():
        raise SystemExit("pytest wrote no junit report")
    if rc not in (0, 1):
        raise SystemExit(f"pytest exited with code {rc} (interrupted, internal or usage error, or no tests)")
    return xml


def run_validate(backend: str) -> tuple[dict, int]:
    out = tmp_root() / f"s7_validate_{backend}_{uuid.uuid4().hex[:12]}.json"
    cmd = [sys.executable, "-P", "-m", "scripts.galactic_binary.validate", "--backend", backend, "--tier", "fast",
           "--golden", "g3", "g4", "--report", str(out)]
    rc = subprocess.run(cmd, cwd=str(tmp_root()), env=_env(), check=False).returncode
    if not out.is_file():
        raise SystemExit(f"validate --backend {backend} wrote no report (exit {rc})")
    return json.loads(out.read_text()), rc


def run_s2_gate() -> int:
    out = tmp_root() / f"s7_s2_{uuid.uuid4().hex[:12]}.json"
    return subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s2_report", "--full", "--strict",
                           "--out", str(out)], cwd=str(REL), env=_env(), check=False).returncode


def _numpy_problems(rep: dict, rc: int) -> list[str]:
    out = [] if rc == 0 else [f"validate --backend numpy exited {rc}"]
    if rep.get("ok") is not True or rep.get("backend") != "numpy" or rep.get("n_targets") != 38:
        out.append(f"numpy: ok {rep.get('ok')!r}, backend {rep.get('backend')!r}, {rep.get('n_targets')} targets")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.golden.s7_report")
    ap.add_argument("--full", action="store_true", help="slow tests, the NumPy GB tier and the S2 gate")
    ap.add_argument("--validate", type=Path, help="summarise this validate --backend jax report (the user's "
                                                  "GPU sign-off: run_meta.device_kind must be 'gpu') instead")
    ap.add_argument("--junit", type=Path, help="summarise an existing junit XML instead of running pytest")
    ap.add_argument("--out", type=Path, help="also write the JSON report here (outside the release tree)")
    a = ap.parse_args(argv)
    out = a.out.resolve() if a.out is not None else None
    if out is not None and out.is_relative_to(REL.resolve()):
        raise SystemExit("--out must lie outside the release tree")
    problems: list[str] = []
    xml = a.junit if a.junit is not None else run_pytest(TESTS, full=a.full)
    counts, passed = collect(xml)
    problems += [p for p, bad in (
        ("no test ran (empty junit report)", counts["tests"] == 0),
        (f"{counts['failures']} test(s) failed", counts["failures"] > 0),
        (f"{counts['errors']} test error(s)", counts["errors"] > 0),
        (f"{counts['skipped']} test(s) skipped (the S7 gate runs every test)", counts["skipped"] > 0)) if bad]
    problems += required_problems(passed, full=a.full)
    if a.validate is not None:                     # the user's GPU sign-off: a GPU report only
        jax_rep, rc, expect = json.loads(a.validate.read_text()), 0, "gpu"
    else:                                          # the local gate: CPU-JAX, validate.main's CPU flag
        (jax_rep, rc), expect = run_validate("jax"), "cpu"
    if rc != 0:
        problems.append(f"validate --backend jax exited {rc}")
    models, vp = summarise_validate(jax_rep, expect_device=expect)
    problems += vp
    extra: dict = {}
    if a.full:
        np_rep, np_rc = run_validate("numpy")
        np_problems = _numpy_problems(np_rep, np_rc)
        extra["numpy_fast_tier"] = {"exit": np_rc, "ok": np_rep.get("ok"), "n_targets": np_rep.get("n_targets"),
                                    "problems": np_problems}
        problems += np_problems
        extra["s2_gate_exit"] = run_s2_gate()
        if extra["s2_gate_exit"] != 0:
            problems.append(f"s2_report --full --strict exited {extra['s2_gate_exit']}")
    rep = {"schema": 1, "kind": "s7_report", "full": bool(a.full), "tests": counts, "expect_device": expect,
           "run_meta": jax_rep.get("run_meta"), "models": models, **extra, "problems": problems,
           "ok": not problems}
    text = json.dumps(rep, indent=1, allow_nan=True)
    if out is not None:
        out.write_text(text + "\n")
    print(text)
    device = (jax_rep.get("run_meta") or {}).get("device_kind")
    print(f"[s7_report] tests {counts['tests']} ({counts['failures']} failed, {counts['errors']} errors, "
          f"{counts['skipped']} skipped); {len(models)} models; device {device} (need {expect}); "
          f"{len(problems)} problem(s)", file=sys.stderr)
    if problems:
        print("[s7_report] FAIL: " + "; ".join(problems[:10]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
