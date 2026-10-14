"""S4 completion report: the sinusoid on JAX, release level, fail-closed (spec section 7.6, S4).

    $PY -m scripts.golden.s4_report [--full] [--validate JAX_REPORT] [--junit XML] [--out PATH]

Four sources, each required:

1. **pytest** -- one subprocess (root config, cwd = the release root, junit XML) over
   :data:`TESTS`: the E-D tests (G1/G2 on CPU-JAX, counters vs WG, batched = loop,
   determinism, the JAX build and its prior sanitising/clamp/validity mask, validate.py on
   JAX, the Colab doc, the strategy columns, the correction orders, ``count_states``) and
   the E-C tests (one factor family and its ok rule, factor constants, the WG-shaped pad and
   Schur forms, device-side Phi, keyed jit, per-state memory within 1.5x WG and one program
   per order, the production HLO budget).  Every test
   must pass; none may skip; every :data:`REQUIRED` node pattern must match enough passed
   nodes (default / ``--full`` counts).
2. **validate.py --backend jax --tier fast** (run here, or ``--validate`` a report from
   elsewhere, e.g. the user's Colab GPU run): its own verdict (``ok``, no failed target, no
   problem) and, independently, the 45 targets with G1 and G2 entries, no NaN/+inf, exactly
   one -inf G1 point per edge-exact target and none elsewhere, a finite positive
   ``ms_per_eval`` everywhere, ``run_meta`` with x64 on (and, on a GPU, the
   deterministic-ops flag set: spec section 3.4).
3. with ``--full``: **validate.py --backend numpy --tier fast** (the S3 gate after the S4
   domainia changes, lineage A bitwise on the recording build) and **s2_report --full
   --strict** (the S2 gate).
4. the deltas: per model, max |JAX - G1| at near points, max relative at stress points, max
   |JAX - G2|, max |JAX - WG| (``oracle_*``: lineage B, golden/sinusoid/legacy_eval/
   B_references_fast.json; lineage A for the NumPy tier) and the median ms/eval --
   reported; the gates are (1)-(3).

``--full`` adds the ``slow`` tests and (3); the S4 gate is ``--full``.  Exit 1 on any
problem; the JSON report goes to ``--out`` (outside the release tree) and stdout.
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

MODELS = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact", "mid_raw", "mid_cut",
          "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad", "tri_exact", "exact")
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
EDGE = ("band_exact", "mid_exact", "tri_exact")
SRC = tuple(REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models"))
#: The E-D test files.
ED_TESTS = (*(f"tests/golden/test_s4_{n}.py" for n in ("sinusoid_jax", "counters_wg", "batched_determinism",
                                                      "prior_clamp", "validate_jax", "colab_doc")),
            "packages/wdm_models/tests/test_wm_evaluation.py", "packages/wdm_models/tests/test_wm_exact_jax.py",
            "packages/domainia/tests/test_dom_sampling.py")
#: The E-C test files (the S4 domainia deferrals, section E-C): (a) one factor family
#: (linalg_factors, linalg_blocktri), (b) constants out of the states, the WG pad/Schur forms
#: and the per-state memory, (c) device-side Phi (pushed_rows), keyed jit and the HLO budget.
EC_TESTS = (*(f"packages/domainia/tests/test_dom_{n}.py" for n in ("factor_constants", "pad_reduce",
                                                                   "schur_scaled_batching", "keyed_jit",
                                                                   "linalg_factors", "linalg_blocktri",
                                                                   "pushed_rows")),
            "tests/parity/test_s4_jax_states.py", "tests/parity/test_s4_hlo_budget.py")
TESTS = ED_TESTS + EC_TESTS
#: (node regex over ``<module>::<name>``, minimum passed nodes without / with ``--full``).
ED_REQUIRED: tuple[tuple[str, int, int], ...] = (
    (r"test_s4_sinusoid_jax::test_g1_and_g2_fast_tier_on_cpu_jax\[.*\]", 12, 45),
    (r"test_s4_sinusoid_jax::test_gate_catches_a_wrong_target", 1, 1),
    (r"test_s4_counters_wg::test_counters_last_failed_and_values_equal_wg\[.*\]", 2, 12),
    (r"test_s4_batched_determinism::test_one_call_with_r_rows_equals_the_row_loop\[.*\]", 4, 6),
    (r"test_s4_batched_determinism::test_the_gpu_row_batching_policy_matches_the_loop_on_cpu\[.*\]", 1, 2),
    (r"test_s4_batched_determinism::test_two_fresh_builds_are_bitwise_identical\[.*\]", 2, 2),
    (r"test_s4_batched_determinism::test_two_fresh_processes_are_bitwise_identical", 0, 1),
    (r"test_s4_batched_determinism::test_whittle_repeats_agree_to_an_ulp_with_threaded_eigen\[.*\]", 2, 2),
    (r"test_s4_prior_clamp::test_the_jax_build_is_e_c_s_wg_composition\[.*\]", 2, 4),
    (r"test_s4_prior_clamp::.*", 15, 17),
    (r"test_s4_validate_jax::.*", 3, 4),
    (r"test_s4_colab_doc::.*", 2, 2),
    (r"test_wm_evaluation::.*", 7, 7),
    (r"test_wm_exact_jax::.*", 5, 5),
    (r"test_dom_sampling::test_batched_count_states_false_counts_rows_but_no_states", 1, 1),
)
#: E-C's gates (its test names as written in section E-C).
EC_REQUIRED: tuple[tuple[str, int, int], ...] = (
    (r"test_dom_factor_constants::.*", 20, 20),
    (r"test_dom_pad_reduce::.*", 7, 7),
    (r"test_dom_schur_scaled_batching::.*", 5, 5),
    (r"test_dom_keyed_jit::.*", 5, 5),
    (r"test_dom_linalg_factors::test_the_duplicate_family_is_gone", 1, 1),
    (r"test_dom_linalg_blocktri::test_one_ok_rule_finite_and_positive", 1, 1),
    (r"test_dom_pushed_rows::test_phi_is_built_per_chunk_on_the_backend", 1, 1),
    (r"test_dom_pushed_rows::test_phi_chunks_live_on_the_device", 1, 1),
    (r"test_s4_jax_states::test_noise_state_memory_is_within_1p5x_of_wg\[.*\]", 0, 3),
    (r"test_s4_jax_states::test_band_exact_compiles_one_program_per_order_and_beats_eager", 0, 1),
    (r"test_s4_jax_states::test_the_memory_gate_bites\[.*\]", 0, 2),
    (r"test_s4_hlo_budget::.*", 3, 3),
)
REQUIRED = ED_REQUIRED + EC_REQUIRED
#: The lineage each backend's values are compared with (spec section 7.1): WT for NumPy, WG for JAX.
ORACLE = {"numpy": "A", "jax": "B"}


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


def _vals(xs: list) -> list[float]:
    return [to_float(x) if isinstance(x, str) else float(x) for x in xs]


def summarise_validate(rep: dict, *, backend: str = "jax") -> tuple[dict, list[str]]:
    """Per-model deltas of a ``scripts/sinusoid/validate.py`` report (E-B's schema: targets
    ``"<case>/<model>"`` with ``g1``/``g2`` entries) against G1, G2 and the backend's oracle
    lineage, and its problems (fail closed)."""
    problems = []
    if rep.get("backend") != backend or rep.get("tier") != "fast":
        problems.append(f"validate report is backend {rep.get('backend')!r} tier {rep.get('tier')!r}; "
                        f"need {backend!r} 'fast'")
    if rep.get("ok") is not True:
        problems.append("validate: ok is not true")
    if rep.get("failed_targets"):
        problems.append(f"validate: failed targets {rep['failed_targets'][:5]}")
    if rep.get("problems"):
        problems.append(f"validate: {len(rep['problems'])} problem(s), e.g. {rep['problems'][0]}")
    meta = rep.get("run_meta")
    if not isinstance(meta, dict):
        problems.append("validate: no run_meta")
    elif backend == "jax":
        if meta.get("x64") is not True:
            problems.append(f"validate: run_meta x64 is {meta.get('x64')!r}")
        if meta.get("device_kind") == "gpu" and meta.get("deterministic_ops") is not True:
            problems.append("validate: a GPU run without XLA_FLAGS=--xla_gpu_deterministic_ops=true")
    targets = rep.get("targets", {})
    want = {f"{c}/{m}" for c in CASES for m in MODELS}
    if set(targets) != want:
        problems.append(f"validate targets: {len(targets)}; missing {sorted(want - set(targets))[:5]}, "
                        f"extra {sorted(set(targets) - want)[:5]}")
    oracle = read_json(GOLDEN / "sinusoid" / "legacy_eval" / f"{ORACLE[backend]}_references_fast.json")["points"]
    models: dict[str, dict] = {}
    for tag, t in sorted(targets.items()):
        g1, g2, model = t.get("g1"), t.get("g2"), t.get("model")
        if model not in MODELS or not isinstance(g1, dict) or not isinstance(g2, dict) \
                or "value" not in g1 or "value" not in g2:
            problems.append(f"{tag}: no g1/g2 values")
            continue
        v, r, near = _vals(g1["value"]), _vals(g1["reference"]), [bool(x) for x in g1["near"]]
        rows = _vals(g2["value"]) + _vals(g2["abs_delta"])
        if any(math.isnan(x) or x == math.inf for x in v + rows):
            problems.append(f"{tag}: NaN or +inf in the values")
            continue
        n_inf = sum(1 for x in v if x == -math.inf)
        if n_inf != (1 if model in EDGE else 0):
            problems.append(f"{tag}: {n_inf} -inf G1 point(s) at the fast tier")
        o = oracle.get(tag.replace("/", "__"))
        w = _vals(o["value"]) if o is not None and list(o["index"]) == list(g1["index"]) else None
        if w is None:
            problems.append(f"{tag}: no lineage-{ORACLE[backend]} values at the same indices")
        m = models.setdefault(model, {"near_max_abs": 0.0, "stress_max_rel": 0.0, "g2_max_abs": 0.0,
                                      "oracle_max_abs_near": 0.0, "oracle_max_rel_stress": 0.0,
                                      "inf_points": 0, "ms_per_eval": []})
        m["inf_points"] += n_inf
        for i, (a, b, is_near) in enumerate(zip(v, r, near, strict=True)):
            if b == -math.inf:
                continue
            if is_near:
                m["near_max_abs"] = max(m["near_max_abs"], abs(a - b))
            else:
                m["stress_max_rel"] = max(m["stress_max_rel"], abs(a - b) / abs(b))
            if w is not None and w[i] != -math.inf:
                if is_near:
                    m["oracle_max_abs_near"] = max(m["oracle_max_abs_near"], abs(a - w[i]))
                else:
                    m["oracle_max_rel_stress"] = max(m["oracle_max_rel_stress"], abs(a - w[i]) / abs(w[i]))
        m["g2_max_abs"] = max(m["g2_max_abs"], max(_vals(g2["abs_delta"]), default=math.inf))
        for kind, e in (("g1", g1), ("g2", g2)):
            ms = float(e.get("ms_per_eval", math.nan))
            if not (math.isfinite(ms) and ms > 0):
                problems.append(f"{tag}: {kind} ms_per_eval {ms}")
        m["ms_per_eval"].append(float(g1.get("ms_per_eval", math.nan)))
    for m in models.values():
        finite = [x for x in m["ms_per_eval"] if math.isfinite(x)]
        m["ms_per_eval"] = statistics.median(finite) if finite else math.nan
    return dict(sorted(models.items())), problems


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.pathsep.join(map(str, (REL, *SRC))), PYTHONDONTWRITEBYTECODE="1",
               JAX_ENABLE_X64="true")
    return env


def run_pytest(patterns: tuple[str, ...], *, full: bool) -> Path:
    missing = [p for p in patterns if not (REL / p).is_file()]
    if missing:
        owner = "E-C (S4 domainia deferrals)" if set(missing) & set(EC_TESTS) else "E-D"
        raise SystemExit(f"no test file {missing} ({owner})")
    xml = tmp_root() / f"s4_report_{uuid.uuid4().hex[:12]}.xml"
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
    out = tmp_root() / f"s4_validate_{backend}_{uuid.uuid4().hex[:12]}.json"
    cmd = [sys.executable, "-P", "-m", "scripts.sinusoid.validate", "--backend", backend, "--tier", "fast",
           "--report", str(out)]
    rc = subprocess.run(cmd, cwd=str(tmp_root()), env=_env(), check=False).returncode
    if not out.is_file():
        raise SystemExit(f"validate --backend {backend} wrote no report (exit {rc})")
    return json.loads(out.read_text()), rc


def run_s2_gate() -> int:
    out = tmp_root() / f"s4_s2_{uuid.uuid4().hex[:12]}.json"
    return subprocess.run([sys.executable, "-P", "-m", "scripts.golden.s2_report", "--full", "--strict",
                           "--out", str(out)], cwd=str(REL), env=_env(), check=False).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.golden.s4_report")
    ap.add_argument("--full", action="store_true", help="slow tests, the NumPy fast tier and the S2 gate")
    ap.add_argument("--validate", type=Path, help="summarise this validate --backend jax report instead")
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
        (f"{counts['skipped']} test(s) skipped (the S4 gate runs every test)", counts["skipped"] > 0)) if bad]
    problems += required_problems(passed, full=a.full)
    if a.validate is not None:
        jax_rep, rc = json.loads(a.validate.read_text()), 0
    else:
        jax_rep, rc = run_validate("jax")
    if rc != 0:
        problems.append(f"validate --backend jax exited {rc}")
    models, vp = summarise_validate(jax_rep)
    problems += vp
    extra: dict = {}
    if a.full:
        np_rep, np_rc = run_validate("numpy")
        np_models, np_problems = summarise_validate(np_rep, backend="numpy")
        extra["numpy_fast_tier"] = {"exit": np_rc, "ok": np_rep.get("ok"),
                                    "recording_build": np_rep.get("recording_build"), "models": np_models,
                                    "problems": np_problems}
        problems += [f"numpy: {p}" for p in np_problems]
        if np_rc != 0:
            problems.append(f"validate --backend numpy exited {np_rc}")
        extra["s2_gate_exit"] = run_s2_gate()
        if extra["s2_gate_exit"] != 0:
            problems.append(f"s2_report --full --strict exited {extra['s2_gate_exit']}")
    rep = {"schema": 1, "kind": "s4_report", "full": bool(a.full), "tests": counts,
           "run_meta": jax_rep.get("run_meta"), "models": models, **extra,
           "problems": problems, "ok": not problems}
    text = json.dumps(rep, indent=1, allow_nan=True)
    if out is not None:
        out.write_text(text + "\n")
    print(text)
    print(f"[s4_report] tests {counts['tests']} ({counts['failures']} failed, {counts['errors']} errors, "
          f"{counts['skipped']} skipped); {len(models)} models; {len(problems)} problem(s)", file=sys.stderr)
    if problems:
        print("[s4_report] FAIL: " + "; ".join(problems[:10]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
