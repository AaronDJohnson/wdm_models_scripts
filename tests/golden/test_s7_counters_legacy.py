"""F-D6 (S7): the GB JAX caches count exactly as lineage C's ``MatrixCache`` / ``EdgeSchurCache`` /
``FullCache`` (``toy_models/galactic_binary/fullcache.py`` ``NoiseStateCache``: keyed by the float64
bytes of the flat noise slice, failed states cached as ``None``, LRU of 24).

For every cached gb_3mo target, ``scripts.golden.gb_jax_legacy.scripted_batches`` -- hits inside a
call, a failed (not positive definite) state and its cached repeat, a source outside the box
(``source_ok``: -inf but no PD failure), 26 fresh states evicting past the capacity, evicted states
proposed again -- goes through the legacy class (lineage C, one subprocess) and through
``registry.build(..., backend="jax")``; after every call the four counters must be identical, the
-inf masks identical, ``last_failed`` exactly the -inf rows inside the box, and the finite values
inside the approved G3 JAX gate (1e-8).  Default lane: tri_raw, band_exact, exact; the rest ``slow``.
Skips only when the frozen trees are absent (CI/Colab).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from scripts.golden.gb_jax_legacy import COUNTER_KEYS, LEGACY_JAX, legacy_counters, scripted_batches  # noqa: E402
from scripts.golden.paths import GOLDEN, legacy_available  # noqa: E402
from wdm_models.galactic_binary.prepared import PreparedCase  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import build  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax, pytest.mark.legacy,
              pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")]
CASE = "gb_3mo"
DEFAULT = ("tri_raw", "band_exact", "exact")
GATE = json.loads((GOLDEN / "galactic_binary" / "tolerances_measured.json").read_text())["approved"]["G3"]["jax"]
CACHED = tuple(m for m, r in LEGACY_JAX[CASE].items() if r["state_bytes"] is not None)


def _models():
    return [pytest.param(m, id=m, marks=[] if m in DEFAULT else [pytest.mark.slow]) for m in CACHED]


@pytest.mark.parametrize("model", _models())
def test_counters_masks_and_values_equal_lineage_c(model):
    batches = scripted_batches(CASE, model)
    legacy = legacy_counters(CASE, model, batches)
    assert legacy["class"] == LEGACY_JAX[CASE][model]["evaluator"] and legacy["capacity"] == 24
    assert [c["name"] for c in legacy["calls"]] == ["b0", "b1", "b2"]
    prep = PreparedCase.load(CASE)
    lo, hi = np.asarray(prep.prior_lo), np.asarray(prep.prior_hi)
    lik = build("galactic_binary", CASE, model, backend="jax", options=ModelOptions(waveform_backend="fourier"))
    for call in legacy["calls"]:
        X = batches[call["name"]]
        values = np.asarray(lik(X), dtype=float)
        want = np.asarray(call["values"], dtype=float)
        assert values.shape == want.shape and values.size > 0
        assert {k: lik.counters()[k] for k in COUNTER_KEYS} == call["counters"], (call["name"], lik.counters(),
                                                                                  call["counters"])
        assert not np.isnan(values).any() and not np.isposinf(values).any()
        assert np.array_equal(np.isneginf(values), np.isneginf(want)), call["name"]
        inside = np.all((X[:, :8] >= lo) & (X[:, :8] <= hi), axis=1)
        assert lik.last_failed.tolist() == (np.isneginf(values) & inside).tolist(), call["name"]
        fin = np.isfinite(want)
        assert np.all(np.abs(values[fin] - want[fin]) <= GATE[f"{CASE}/{model}"]), call["name"]
    # the script exercised what it claims: 3 misses, 26 more (evicting past 24), 3 evicted states again
    assert [c["counters"]["noise_states"] for c in legacy["calls"]] == [3, 29, 32]
    assert [c["counters"]["evaluations"] for c in legacy["calls"]] == [8, 34, 40]
    fails = [c["counters"]["pd_failed_states"] for c in legacy["calls"]]
    assert fails == ([1, 1, 2] if model != "exact" else [0, 0, 0]), fails      # exact: PD at every gb_3mo point
