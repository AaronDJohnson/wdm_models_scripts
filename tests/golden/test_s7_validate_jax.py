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
JAX fast tier is ``slow``.
"""

from __future__ import annotations

import json
import math
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


@pytest.mark.slow
def test_the_jax_fast_tier_of_all_38_targets(tmp_path):
    out = tmp_path / "report.json"
    p = _cli("--report", str(out))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] is True and rep["n_targets"] == 38 and rep["failed"] == []
    for tag, t in rep["targets"].items():
        _check_target(tag, t)
