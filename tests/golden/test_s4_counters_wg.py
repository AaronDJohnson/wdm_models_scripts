"""E-D6 (S4): the JAX registry's noise-state counters are exactly WG's.

For each WDM model WG counted (``wdmtoysgpu/matrix.py:726-789``), the scripted batches of
``scripts.golden.sinusoid_counters`` go through the WG likelihood (lineage B, one
subprocess) and through ``registry.build(..., backend="jax")`` at WG's LRU size (16); after
every call the five counters (``chain_stats.COUNTERS`` + ``cache_hits``) and
``last_failed`` must be identical, the -inf masks identical and the values inside the
approved JAX stress gate (1e-5 + 1e-8 |lnL|).  The uncached diagonal models count no noise
state (``BatchedLikelihood(count_states=False)``), as WG.  Default lane: tri_raw,
band_raw; the rest ``slow``.  Skips only when the frozen trees are absent (CI/Colab).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S4 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from scripts.golden.paths import legacy_available  # noqa: E402
from scripts.golden.sinusoid_counters import COUNTER_KEYS, WG_CACHE_SIZE, scripted_batches, wg_counters  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import build  # noqa: E402
from wdm_models.sinusoid.evaluation import CACHED, UNCACHED_DIAGONAL  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax, pytest.mark.legacy,
              pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")]
CASE = "manuscript_mid"
DEFAULT = ("tri_raw", "band_raw")


def _models():
    return [pytest.param(m, id=m, marks=[] if m in DEFAULT else [pytest.mark.slow])
            for m in CACHED + UNCACHED_DIAGONAL]


@pytest.mark.parametrize("model", _models())
def test_counters_last_failed_and_values_equal_wg(model):
    batches = scripted_batches(CASE, model)
    wg = wg_counters(CASE, model, batches)
    assert len(wg["calls"]) == len(batches) == 3
    if model in CACHED:
        assert wg["cache_size"] == WG_CACHE_SIZE
    lik = build("sinusoid", CASE, model, backend="jax", options=ModelOptions(cache_size=WG_CACHE_SIZE))
    for name, call in zip(sorted(batches, key=lambda k: int(k[1:])), wg["calls"], strict=True):
        values = np.asarray(lik(batches[name]), dtype=float)
        want = np.asarray(call["values"], dtype=float)
        assert values.shape == want.shape and values.size > 0
        assert set(call["counters"]) == set(COUNTER_KEYS)
        assert lik.counters() == call["counters"], (name, lik.counters(), call["counters"])
        assert lik.last_failed.tolist() == call["failed"], name
        assert not np.isnan(values).any() and not np.isposinf(values).any()
        assert np.array_equal(np.isneginf(values), np.isneginf(want)), name
        fin = np.isfinite(want)
        assert np.all(np.abs(values[fin] - want[fin]) <= 1e-5 + 1e-8 * np.abs(want[fin])), name
    if model in UNCACHED_DIAGONAL:
        assert lik.counters()["noise_states"] == 0 and lik.counters()["cache_hits"] == 0
