"""F-D1 (S7): the GB historical JAX column is the legacy JAX evaluator, target by target.

``scripts.golden.gb_jax_legacy.LEGACY_JAX`` records what lineage C's ``gpu/jax_models.build`` ran
the chains with (class, rows of one call batched, LRU capacity, one state's bytes).  The table is
pinned against a fresh lineage-C measurement (``legacy``; gb_3mo by default, the other two ``slow``)
and ``options.resolve("galactic_binary", model, "jax")`` must reproduce it on both device kinds:
``evaluator="cache"`` exactly for the cache classes (``MatrixCache``, ``EdgeSchurCache``,
``FullCache``), ``"batched"`` for ``BatchedLikelihood``; ``schur="schur"`` exactly for
``EdgeSchurCache``; the expanded strategy for the cache classes; ``cache_size`` the capacity (24);
``batch_rows`` the legacy batching -- on for ``MatrixCache``/``EdgeSchurCache`` on every device
(``fullcache.batch_signal``), OFF for ``FullCache`` (it looped over the rows).
"""

from __future__ import annotations

import pytest

from scripts.golden.gb_jax_legacy import CACHE_CLASSES, LEGACY_JAX, legacy_states
from scripts.golden.paths import legacy_available
from wdm_models.options import ModelOptions, resolve
from wdm_models.registry import TARGETS

pytestmark = pytest.mark.golden
GOLDEN_OPTIONS = ModelOptions(waveform_backend="fourier")


def test_the_table_covers_the_38_targets():
    assert {c: set(m) for c, m in LEGACY_JAX.items()} == {c: set(m) for c, m in TARGETS["galactic_binary"].items()}
    assert sum(len(m) for m in LEGACY_JAX.values()) == 38
    for case, models in LEGACY_JAX.items():
        for model, rec in models.items():
            cached = rec["evaluator"] in CACHE_CLASSES
            assert (rec["capacity"] == 24) == cached and (rec["state_bytes"] is not None) == cached, (case, model)


@pytest.mark.legacy
@pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
@pytest.mark.parametrize("case", ["gb_3mo", pytest.param("gb_6mo", marks=pytest.mark.slow),
                                  pytest.param("gb_12mo", marks=pytest.mark.slow)])
def test_the_table_is_lineage_c(case):
    assert legacy_states(case) == LEGACY_JAX[case]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_the_jax_column_is_the_legacy_evaluator(device):
    for case, models in LEGACY_JAX.items():
        for model, rec in models.items():
            r = resolve("galactic_binary", model, "jax", GOLDEN_OPTIONS, device=device)
            cached = rec["evaluator"] in CACHE_CLASSES
            assert (r.backend, r.device, r.cells, r.pad_factor) == ("jax", device, "dense", "dense"), (case, model)
            assert r.evaluator == ("cache" if cached else "batched"), (case, model)
            assert r.schur == ("schur" if rec["evaluator"] == "EdgeSchurCache" else "dense"), (case, model)
            assert r.strategy == ("expanded" if cached else "residual"), (case, model)
            assert r.batch_rows is (rec["batched"] if cached else False), (case, model)
            assert r.cache_size == 24 and r.waveform_backend == "fourier", (case, model)


def test_reference_on_jax_is_the_numpy_column():
    for model in LEGACY_JAX["gb_3mo"]:
        j = resolve("galactic_binary", model, "jax", ModelOptions(strategy="reference", waveform_backend="fourier"),
                    device="cpu")
        n = resolve("galactic_binary", model, "numpy", GOLDEN_OPTIONS)
        assert (j.evaluator, j.schur, j.strategy, j.batch_rows) == (n.evaluator, n.schur, n.strategy, False), model
        assert n.evaluator == "reference" and n.batch_rows is False, model
