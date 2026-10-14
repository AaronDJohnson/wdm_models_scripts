"""F-D7 (S7): on CPU-JAX one call with R rows equals the row loop, and the GB values are
reproducible bit for bit -- across calls, fresh builds and processes.

* Batched vs loop (six gb_3mo rows: the anchor, a source move, a new noise state, the ``noise_hi``
  stress point (not positive definite for the matrix models), an edge stress point and a source
  outside the box): the vmapped evaluators -- the
  ``MatrixCache``/``EdgeSchurCache`` models (``NoiseStateCache(batch_rows=True)``, stacked states)
  and ``BatchedLikelihood`` -- agree with one-row calls to 1e-12 relative (measured <= 2.2e-14:
  the vmapped template and contractions round differently), with identical -inf masks and
  ``last_failed``; ``exact`` (``FullCache``: a row loop) and ``ModelOptions(batch_rows=False)``
  are bitwise the loop.
* Determinism: repeated calls are bitwise (also with XLA-CPU's default threaded Eigen: measured on
  whittle_raw/cut, band_raw, tri_raw, band_exact and exact, unlike the sinusoid's 16384-point
  Whittle rfft); two fresh builds give identical bytes; two fresh interpreters give identical
  bytes (``float.hex``) with ``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`` (``slow``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from domainia.sampling import BatchedLikelihood, NoiseStateCache  # noqa: E402
from scripts.golden.paths import GOLDEN, REL, tmp_root  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import build  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
CASE = "gb_3mo"
RTOL = 1e-12
GOLDEN_OPTIONS = ModelOptions(waveform_backend="fourier")


def _rows(model: str) -> np.ndarray:
    pts = [np.asarray(p["theta"], dtype=float) for p in json.loads(
        (GOLDEN / "galactic_binary" / "stress" / CASE / f"{model}.fast.json").read_text())["points"]]
    x = pts[0]
    move, noise = x.copy(), x.copy()
    move[0] += 0.05
    noise[8] += 0.01
    return np.stack([x, move, noise, pts[1], pts[5], pts[9]])     # anchor ... source_out_lnA


def _loop(model: str, X: np.ndarray, options: ModelOptions = GOLDEN_OPTIONS) -> tuple[np.ndarray, list[bool]]:
    lik = build("galactic_binary", CASE, model, backend="jax", options=options)
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
                                   pytest.param("band_exact", marks=pytest.mark.slow),
                                   pytest.param("tri_pad", marks=pytest.mark.slow),
                                   pytest.param("band_pad", marks=pytest.mark.slow),
                                   pytest.param("tri_exact", marks=pytest.mark.slow)])
def test_one_call_with_r_rows_equals_the_row_loop(model):
    X = _rows(model)
    want, want_failed = _loop(model, X)
    lik = build("galactic_binary", CASE, model, backend="jax", options=GOLDEN_OPTIONS)
    got = np.asarray(lik(X), dtype=float)
    assert got.shape == (6,) and np.isfinite(got[:3]).all() and got[5] == -np.inf
    assert lik.last_failed.tolist() == want_failed
    if isinstance(lik, NoiseStateCache) and not lik.batch_rows:
        assert model == "exact" and np.array_equal(got, want)          # FullCache: the row loop, bitwise
    else:
        assert isinstance(lik, BatchedLikelihood) or lik.batch_rows
        assert _close(got, want), np.max(np.abs(got - want))


@pytest.mark.parametrize("model", ["tri_raw", pytest.param("band_exact", marks=pytest.mark.slow)])
def test_the_row_loop_option_is_bitwise_the_loop(model):
    X = _rows(model)
    loop_options = ModelOptions(waveform_backend="fourier", batch_rows=False)
    lik = build("galactic_binary", CASE, model, backend="jax", options=loop_options)
    assert isinstance(lik, NoiseStateCache) and lik.batch_rows is False
    assert np.array_equal(np.asarray(lik(X), dtype=float), _loop(model, X, loop_options)[0])


@pytest.mark.parametrize("model", ["whittle_raw", "tri_raw", "exact"])
def test_repeats_and_fresh_builds_are_bitwise_identical(model):
    X = _rows(model)
    lik = build("galactic_binary", CASE, model, backend="jax", options=GOLDEN_OPTIONS)
    first = [float(v).hex() for v in lik(X)]
    for _ in range(3):
        lik.reset_counters()
        assert [float(v).hex() for v in lik(X)] == first
    again = build("galactic_binary", CASE, model, backend="jax", options=GOLDEN_OPTIONS)(X)
    assert [float(v).hex() for v in again] == first


_PROBE = """
import json, sys
import numpy as np
from wdm.backend import enable_x64
enable_x64()
from wdm_models.options import ModelOptions
from wdm_models.registry import build
out = {}
for model in sys.argv[1].split(","):
    pts = json.loads(open(sys.argv[2] + f"/{model}.fast.json").read())["points"]
    X = np.array([p["theta"] for p in pts[:6]], dtype=float)
    lik = build("galactic_binary", "gb_3mo", model, backend="jax", options=ModelOptions(waveform_backend="fourier"))
    out[model] = [float(v).hex() for v in lik(X)]
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
        p = subprocess.run([sys.executable, "-P", "-c", _PROBE, models, str(GOLDEN / "galactic_binary" / "stress" / CASE)],
                           env=env, cwd=str(tmp_root()), capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr[-2000:]
        runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    assert set(runs[0]) == set(models.split(",")) and all(len(v) == 6 for v in runs[0].values())
    assert runs[0] == runs[1]
