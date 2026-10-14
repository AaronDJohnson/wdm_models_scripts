"""E-D7 (S4): on CPU-JAX one call with R rows equals the row loop, and the values are
reproducible bit for bit -- across fresh builds in one process and across processes.

* Batched vs loop (the 8 first G1 points of manuscript_mid; point 4 is -inf for the
  edge-exact models): an LRU model with the CPU policy (``batch_rows=False``, WG on a CPU)
  is bitwise the loop; one ``jit(vmap(...))`` (``BatchedLikelihood``: Whittle, diagonals,
  the exact model's batched PCG) and the GPU policy forced on the CPU
  (``NoiseStateCache(batch_rows=True)``) agree to 1e-11 |lnL| (measured <= 8.5e-13
  relative, the exact model's PCG); -inf masks and ``last_failed`` identical.
* Determinism: two fresh builds give identical bytes; two fresh interpreters give
  identical bytes (``float.hex``) for six models (``slow``) with single-threaded Eigen
  (``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false``).  With XLA-CPU's default threaded
  Eigen the Whittle programs (one 16384-point rfft per row) are NOT bitwise run to run:
  repeated calls differ by up to 1.6e-16 relative (measured: 3-13 of 20 calls at
  manuscript_low/high), which the last test pins as <= 1e-15; every other model repeated
  bitwise in the same measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S4 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from domainia.sampling import NoiseStateCache  # noqa: E402
from scripts.golden.goldens_io import read_npz  # noqa: E402
from scripts.golden.paths import REL, tmp_root  # noqa: E402
from scripts.golden.sinusoid_goldens import REFS  # noqa: E402
from wdm_models.registry import build  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
CASE = "manuscript_mid"
RTOL = 1e-11


def _rows(model: str, n: int = 8) -> np.ndarray:
    return np.asarray(read_npz(REFS / f"{CASE}_{model}_reference.npz")["theta"][:n], dtype=float)


def _loop(model: str, X: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    lik = build("sinusoid", CASE, model, backend="jax")
    vals, failed = [], []
    for row in X:
        vals.append(float(lik(row[None, :])[0]))
        failed.append(bool(lik.last_failed[0]))
    return np.asarray(vals), failed


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    fin = np.isfinite(b)
    return (np.array_equal(np.isneginf(a), np.isneginf(b)) and not np.isnan(a).any()
            and bool(np.all(np.abs(a[fin] - b[fin]) <= RTOL * np.abs(b[fin]))))


@pytest.mark.parametrize("model", ["whittle_raw", "band_raw", "tri_raw", "exact",
                                   pytest.param("band_pad", marks=pytest.mark.slow),
                                   pytest.param("band_exact", marks=pytest.mark.slow)])
def test_one_call_with_r_rows_equals_the_row_loop(model):
    X = _rows(model)
    want, want_failed = _loop(model, X)
    lik = build("sinusoid", CASE, model, backend="jax")
    got = np.asarray(lik(X), dtype=float)
    assert got.shape == (8,) and np.isfinite(got).sum() >= 7
    assert lik.last_failed.tolist() == want_failed
    if isinstance(lik, NoiseStateCache):
        assert lik.batch_rows is False                  # the CPU policy (WG batch_rungs off on a CPU)
        assert np.array_equal(got, want)               # same compiled programs, same states: bitwise
    else:
        assert _close(got, want), np.max(np.abs(got - want))


@pytest.mark.parametrize("model", ["tri_raw", pytest.param("band_pad", marks=pytest.mark.slow)])
def test_the_gpu_row_batching_policy_matches_the_loop_on_cpu(model):
    X = _rows(model)
    want, _ = _loop(model, X)
    lik = build("sinusoid", CASE, model, backend="jax")
    forced = NoiseStateCache(lik.lik, lik.capacity, batch_rows=True)
    assert _close(np.asarray(forced(X), dtype=float), want)


@pytest.mark.parametrize("model", ["tri_raw", "exact"])
def test_two_fresh_builds_are_bitwise_identical(model):
    X = _rows(model, 6)
    a = build("sinusoid", CASE, model, backend="jax")(X)
    b = build("sinusoid", CASE, model, backend="jax")(X)
    assert [float(v).hex() for v in a] == [float(v).hex() for v in b]


_PROBE = """
import json, sys
import numpy as np
from wdm.backend import enable_x64
enable_x64()
from wdm_models.registry import build
out = {}
for model in sys.argv[1].split(","):
    with np.load(sys.argv[2] + f"/manuscript_mid_{model}_reference.npz") as z:
        X = np.asarray(z["theta"][:6], dtype=float)
    out[model] = [float(v).hex() for v in build("sinusoid", "manuscript_mid", model, backend="jax")(X)]
print(json.dumps(out))
"""


@pytest.mark.slow
def test_two_fresh_processes_are_bitwise_identical():
    models = "whittle_raw,band_raw,tri_raw,band_pad,band_exact,exact"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "XLA_FLAGS"))}
    env.update(PYTHONPATH=os.pathsep.join(str(REL / p) for p in (".", "packages/wdm/src", "packages/domainia/src",
                                                                "packages/wdm_models/src")),
               PYTHONDONTWRITEBYTECODE="1", JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    runs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, "-P", "-c", _PROBE, models, str(REFS)], env=env, cwd=str(tmp_root()),
                           capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr[-2000:]
        runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    assert set(runs[0]) == set(models.split(",")) and all(len(v) == 6 for v in runs[0].values())
    assert runs[0] == runs[1]


@pytest.mark.parametrize("model", ["whittle_raw", "whittle_cut"])
def test_whittle_repeats_agree_to_an_ulp_with_threaded_eigen(model):
    """The default XLA-CPU threading makes the Whittle rfft programs non-bitwise run to run
    (module docstring); the differences stay at the last bit."""
    X = np.asarray(read_npz(REFS / f"manuscript_high_{model}_reference.npz")["theta"], dtype=float)
    lik = build("sinusoid", "manuscript_high", model, backend="jax")
    first = np.asarray(lik(X), dtype=float)
    assert np.isfinite(first).all()
    for _ in range(10):
        again = np.asarray(lik(X), dtype=float)
        assert np.all(np.abs(again - first) <= 1e-15 * np.abs(first)), np.max(np.abs(again - first) / np.abs(first))
