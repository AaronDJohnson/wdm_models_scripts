"""E-D8 (S4): ``scripts/sinusoid/validate.py --backend jax`` on CPU-JAX.

The CLI evaluates the G1 fast points and G2 fast rows through the release registry (the
JAX evaluator of E-D4) and gates them at ``approved.jax``; the report carries ``run_meta``
(device kind, x64, ``XLA_FLAGS``) and per target and kind ``ms_per_eval`` and
``repeat_max_abs`` from a warm repeat (reported, never gated).  It fails closed on JAX as on
NumPy: an exception while building or evaluating a target fails that target (exit 1),
never a skip, and a gate violation fails.  The full 45-target JAX fast tier is ``slow``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

jax = pytest.importorskip("jax", reason="the S4 JAX gates need jax")

from scripts.golden.paths import REL, tmp_root  # noqa: E402
from scripts.sinusoid import validate as V  # noqa: E402
from wdm_models.sampling.validate import Gate  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
SRC = [REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models")]
SUBSET = ("whittle_raw", "band_raw", "tri_raw", "exact")


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "XLA_FLAGS")}
    env.update(PYTHONPATH=os.pathsep.join(map(str, [REL, *SRC])), PYTHONDONTWRITEBYTECODE="1",
               JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    return subprocess.run([sys.executable, "-P", "-m", "scripts.sinusoid.validate", "--backend", "jax",
                           "--tier", "fast", *args], cwd=str(tmp_root()), env=env, capture_output=True, text=True)


def _check_target(tag: str, t: dict) -> None:
    assert t["ok"] is True and t["problems"] == [], tag
    assert len(t["g1"]["value"]) == len(t["g1"]["reference"]) == 4, tag
    assert len(t["g2"]["value"]) == len(t["g2"]["reference"]) == 3, tag
    for kind in ("g1", "g2"):
        e = t[kind]
        assert math.isfinite(e["ms_per_eval"]) and e["ms_per_eval"] > 0, (tag, kind)
        scale = max(abs(float(v)) for v in e["value"] if math.isfinite(float(v)))
        # an ulp: E-D7's threaded-Eigen allowance (the CLI sets the CPU flag; 0 of 90 differed in P3C's gate)
        assert e["repeat_max_abs"] <= 1e-15 * scale, (tag, kind)
        assert e["inf_mismatch"] == 0 and "lineage_A" not in e, (tag, kind)


def test_cli_subset_on_cpu_jax(tmp_path, monkeypatch):
    # spec section 3.4: main() appends the CPU flag before JAX starts unless a setting is there
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")     # restored afterwards
    assert V.cpu_determinism() == "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
    assert V.cpu_determinism() == "--xla_cpu_multi_thread_eigen=true"               # an explicit choice wins
    monkeypatch.delenv("XLA_FLAGS")
    assert V.cpu_determinism() == V.CPU_DETERMINISM == "--xla_cpu_multi_thread_eigen=false"
    out = tmp_path / "report.json"
    p = _cli("--case", "manuscript_mid", *(a for m in SUBSET for a in ("--model", m)), "--report", str(out))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    rep = json.loads(out.read_text())
    assert rep["backend"] == "jax" and rep["tier"] == "fast" and rep["ok"] is True and rep["problems"] == []
    assert rep["recording_build"] is False and rep["bitwise_against"] == {"g1": False, "g2": False}
    meta = rep["run_meta"]
    assert meta["device_kind"] == "cpu" and meta["platform"] == "cpu" and meta["x64"] is True
    assert meta["jax"] == "0.9.0.1" and meta["deterministic_ops"] is False
    assert meta["XLA_FLAGS"] == V.CPU_DETERMINISM and meta["cpu_single_thread_eigen"] is True   # _cli drops XLA_FLAGS
    assert rep["gates"]["g1"]["exact"] == {"atol": 1e-4, "rtol": 1e-8}
    assert rep["gates"]["g2"]["band_raw"] == {"atol": 1e-5, "rtol": 0.0}
    assert sorted(rep["targets"]) == sorted(f"manuscript_mid/{m}" for m in SUBSET)
    for tag, t in rep["targets"].items():
        _check_target(tag, t)


def test_a_target_that_raises_on_jax_fails_the_run():
    from wdm_models.registry import build

    def broken(toy, case, model, **kw):
        lik = build(toy, case, model, **kw)
        if model != "band_raw":
            return lik

        class Exploding:
            ndim = lik.ndim

            def __call__(self, Z):
                raise RuntimeError("simulated device failure")

        return Exploding()

    rep = V.run(backend="jax", tier="fast", cases=["manuscript_mid"], models=["whittle_raw", "band_raw"],
                builder=broken)
    assert rep["ok"] is False and rep["failed_targets"] == ["manuscript_mid/band_raw"]
    assert any("simulated device failure" in p for p in rep["problems"])
    assert rep["targets"]["manuscript_mid/whittle_raw"]["ok"] is True


def test_a_jax_gate_violation_fails(monkeypatch):
    real = V.approved_gates

    def tight(backend, kind, golden=V.GOLDEN):
        gates = real(backend, kind, golden)
        if backend == "jax" and kind == "references":
            gates["band_raw"] = Gate(0.0, 0.0)
        return gates

    monkeypatch.setattr(V, "approved_gates", tight)
    rep = V.run(backend="jax", tier="fast", cases=["manuscript_mid"], models=["whittle_raw", "band_raw"])
    assert rep["ok"] is False and rep["failed_targets"] == ["manuscript_mid/band_raw"]
    assert rep["problems"] and all("band_raw" in p for p in rep["problems"])


@pytest.mark.slow
def test_cli_the_whole_jax_fast_tier(tmp_path):
    out = tmp_path / "report.json"
    p = _cli("--report", str(out))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] is True and rep["failed_targets"] == [] and rep["n_targets"] == 45
    assert rep["n_points"] == 45 * 4 + 45 * 3
    assert rep["run_meta"]["XLA_FLAGS"] == V.CPU_DETERMINISM and rep["run_meta"]["cpu_single_thread_eigen"] is True
    for tag, t in rep["targets"].items():
        _check_target(tag, t)
        n_inf = sum(1 for v in t["g1"]["value"] if v == "-inf")
        assert n_inf == (1 if t["model"].endswith("_exact") and t["model"] != "exact" else 0), tag
