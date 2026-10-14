"""F-D9 (S7): ``scripts/galactic_binary/validate.py --backend jax`` on CPU-JAX.

The CLI evaluates the G3 fast rows and the G4 stress points through the release registry's JAX
evaluator (F-D3) and gates them at ``approved.G3.jax`` / ``approved.G4`` (8.32e-8 at
well-conditioned points); the counters are lineage C's own JAX counters (``jax_counters``,
``counters_jax``; rows only for the uncached ``BatchedLikelihood``), not the NumPy reference's one
state per row.  On JAX the report carries ``run_meta`` (device kind, x64, ``XLA_FLAGS``) and, per
run directory / stress file, ``ms_per_eval`` and ``repeat_max_abs`` from a warm repeat (reported,
never gated).  Before JAX starts the CLI adds ``--xla_cpu_multi_thread_eigen=false`` to
``XLA_FLAGS`` unless the flag is already set (CPU determinism, spec section 3.4; a no-op on a
GPU).  It fails closed: an exception or a gate violation fails the target (exit 1).  The 38-target
JAX fast tier is ``slow``.  P4P (I-1): on JAX too, a NaN or +inf stress reference, a NaN or inf G4
gate (with a reference shifted by 1.0) and a golden whose sha256 is not golden/MANIFEST.json's
fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import os
import subprocess
import sys

import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from scripts.galactic_binary import validate as V  # noqa: E402
from scripts.golden.gb_jax_legacy import CACHE_CLASSES, LEGACY_JAX  # noqa: E402
from scripts.golden.paths import GOLDEN, REL, tmp_root  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
SRC = [REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models")]
SUBSET = ("whittle_raw", "band_raw", "tri_raw", "band_exact", "exact")
CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"


def _cli(*args: str, xla_flags: str | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "XLA_FLAGS")}
    env.update(PYTHONPATH=os.pathsep.join(map(str, [REL, *SRC])), PYTHONDONTWRITEBYTECODE="1",
               JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if xla_flags is not None:
        env["XLA_FLAGS"] = xla_flags
    return subprocess.run([sys.executable, "-P", "-m", "scripts.galactic_binary.validate", "--backend", "jax",
                           "--tier", "fast", "--golden", "g3", "g4", *args], cwd=str(tmp_root()), env=env,
                          capture_output=True, text=True)


def _golden_copy(root, edits=None, *, resign=True):
    """golden/galactic_binary + golden/MANIFEST.json copied under ``root``; ``edits`` (relpath ->
    in-place edit of the parsed JSON) re-signed in the copy's MANIFEST unless ``resign`` is False."""
    shutil.copytree(GOLDEN / "galactic_binary", root / "galactic_binary")
    man = json.loads((GOLDEN / "MANIFEST.json").read_text())
    for rel, edit in (edits or {}).items():
        rec = json.loads((root / rel).read_text())
        edit(rec)
        (root / rel).write_text(json.dumps(rec, indent=1))
        if resign:
            man["files"][rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    (root / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return root


def _check_target(tag: str, t: dict) -> None:
    case, model = tag.split("/")
    cached = LEGACY_JAX[case][model]["evaluator"] in CACHE_CLASSES
    assert t["ok"] is True and t["g3"]["ok"] is True and t["g3"]["problems"] == [], tag
    runs = t["g3"]["runs"]
    assert len(runs) == len(json.loads((GOLDEN / "galactic_binary" / "chain_rows" / case / "index.json").read_text())
                            ["targets"][model]), tag
    for run, r in runs.items():
        assert len(r["value"]) == 3 and max(r["abs_delta"]) <= 1e-8, (tag, run)
        assert math.isfinite(r["ms_per_eval"]) and r["ms_per_eval"] > 0, (tag, run)
        assert r["repeat_max_abs"] == 0.0, (tag, run)                  # GB repeats are bitwise (F-D7)
        assert {k: r["counters"][k] for k in V.COUNTER_KEYS} == r["counters_golden"], (tag, run)
        if not cached:                                                 # rows only (lineage C kept none)
            assert r["counters_golden"] == {"evaluations": 3, "noise_states": 0, "pd_failed_rows": 0,
                                            "pd_failed_states": 0}, (tag, run)
    g4 = t["g4"]
    if case == "gb_12mo":
        assert g4["applicable"] is False
        return
    assert g4["ok"] is True and "cache" not in g4, tag                  # on JAX the default evaluator IS the cache
    ref = g4["reference"]
    assert ref["inf_mismatch"] == ref["expected_mismatch"] == ref["invalid"] == ref["over"] == 0, tag
    assert ref["well_max_abs"] <= 8.32e-8 and g4["repeat_max_abs"] == 0.0 and g4["ms_per_eval"] > 0, tag
    if cached:
        assert {k: ref["counters"][k] for k in V.COUNTER_KEYS} == g4["counters_golden"], tag


def test_cli_subset_on_cpu_jax(tmp_path):
    out = tmp_path / "report.json"
    p = _cli("--case", "gb_3mo", *(a for m in SUBSET for a in ("--model", m)), "--report", str(out))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    rep = json.loads(out.read_text())
    assert rep["backend"] == "jax" and rep["tier"] == "fast" and rep["ok"] is True and rep["problems"] == []
    assert rep["n_targets"] == len(SUBSET) and rep["waveform"] == "fourier" and rep["templates"] == "bg"
    meta = rep["run_meta"]
    assert meta["device_kind"] == "cpu" and meta["platform"] == "cpu" and meta["x64"] is True
    assert CPU_FLAG in meta["XLA_FLAGS"]                                 # set by the CLI before JAX started
    for tag, t in rep["targets"].items():
        _check_target(tag, t)
    assert "g3 runs=" in p.stdout and "g4 targets=" in p.stdout and "PASS" in p.stdout


def test_an_explicit_xla_flag_is_kept_and_the_cpu_flag_appended(tmp_path):
    out = tmp_path / "report.json"
    p = _cli("--case", "gb_3mo", "--model", "band_raw", "--report", str(out),
             xla_flags="--xla_force_host_platform_device_count=1")
    assert p.returncode == 0, p.stderr[-2000:]
    flags = json.loads(out.read_text())["run_meta"]["XLA_FLAGS"]
    assert flags == f"--xla_force_host_platform_device_count=1 {CPU_FLAG}"


def test_a_wrong_likelihood_fails_the_jax_gates():
    """In-process: gb_6mo's model at gb_3mo's goldens (same ndim) fails both kinds, never passes."""
    from wdm_models.registry import build

    def wrong(toy, case, model, **kw):
        return build(toy, "gb_6mo", model, **kw)

    rep = V.run(backend="jax", tier="fast", cases=["gb_3mo"], models=["tri_raw"], goldens=("g3", "g4"),
                builder=wrong)
    assert rep["ok"] is False and rep["failed"] == ["gb_3mo/tri_raw"]
    t = rep["targets"]["gb_3mo/tri_raw"]
    assert t["g3"]["ok"] is False and t["g4"]["ok"] is False


def test_the_review_probes_fail_closed_on_jax(tmp_path):
    """I-1 on JAX (each probe PASSed at v0.10.0): a NaN / +inf stress reference, a NaN / inf G4
    gate with the anchor's reference shifted by 1.0, a tampered golden; the shift alone fails."""
    stress, tol = "galactic_binary/stress/gb_3mo/band_raw.fast.json", "galactic_binary/tolerances_measured.json"

    def ref(value):
        def edit(rec):
            next(p for p in rec["points"] if p["name"] == "anchor")["lnl_ref"] = value
        return edit

    def shift(rec):
        p = next(p for p in rec["points"] if p["name"] == "anchor")
        p["lnl_ref"] = float(p["lnl_ref"]) + 1.0

    def gate(value):
        return lambda t: t["approved"]["G4"].update(jax_well_conditioned=value)

    def per_target(t):
        t["approved"]["G4"]["jax_well_conditioned_per_target"]["gb_3mo/band_raw"] = math.inf

    probes = {"nan_ref": ({stress: ref(math.nan)}, "NaN or"), "inf_ref": ({stress: ref(math.inf)}, "NaN or"),
              "nan_gate": ({stress: shift, tol: gate(math.nan)}, "G4"),
              "inf_gate": ({stress: shift, tol: gate(math.inf)}, "G4"),
              "inf_per_target": ({stress: shift, tol: per_target}, "per-target")}
    for name, (edits, match) in probes.items():
        bad = _golden_copy(tmp_path / name, edits)
        with pytest.raises(V.ValidationError, match=match):
            V.run(backend="jax", cases=["gb_3mo"], models=["band_raw"], goldens=("g4",), golden=bad)
    bad = _golden_copy(tmp_path / "tampered", {stress: shift}, resign=False)
    with pytest.raises(V.ValidationError, match="sha256"):
        V.run(backend="jax", cases=["gb_3mo"], models=["band_raw"], goldens=("g3", "g4"), golden=bad)
    p = _cli("--case", "gb_3mo", "--model", "band_raw", "--golden-root", str(bad))
    assert p.returncode == 1 and "sha256" in p.stderr and "PASS" not in p.stdout, p.stderr[-2000:]
    rep = V.run(backend="jax", cases=["gb_3mo"], models=["band_raw"], goldens=("g4",),
                golden=_golden_copy(tmp_path / "shifted", {stress: shift}))
    g4 = rep["targets"]["gb_3mo/band_raw"]["g4"]
    assert rep["ok"] is False and g4["reference"]["over"] == 1 and g4["ok"] is False


@pytest.mark.slow
def test_the_jax_fast_tier_of_all_38_targets(tmp_path):
    out = tmp_path / "report.json"
    p = _cli("--report", str(out))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] is True and rep["n_targets"] == 38 and rep["failed"] == []
    for tag, t in rep["targets"].items():
        _check_target(tag, t)
