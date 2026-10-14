"""F-C2: the galactic binary's strategy table (spec section 3.2; ``options.GBResolved``)."""

from __future__ import annotations

import pytest

from wdm_models.common.params import MODELS
from wdm_models.options import GB_CACHED, GBResolved, ModelOptions, resolve

GB_SCHUR = ("band_exact", "mid_exact")


def test_numpy_historical_is_lineage_c_numpy_reference():
    for model in MODELS:
        r = resolve("galactic_binary", model, "numpy")
        assert isinstance(r, GBResolved)
        assert (r.backend, r.device, r.cells, r.pad_factor, r.schur, r.evaluator, r.cache_size, r.batch_rows) == (
            "numpy", "cpu", "dense", "dense", "dense", "reference", 24, False), model
        assert r.strategy == ("expanded" if model == "exact" else "residual"), model
        assert r.waveform_backend == "onfly"


def test_the_numpy_cache_evaluator_is_lineage_c_cache_classes():
    assert GB_CACHED == ("tri_raw", "tri_cut", "tri_pad", "tri_exact", "band_exact", "mid_exact", "band_pad",
                         "mid_pad", "exact")
    for model in GB_CACHED:
        r = resolve("galactic_binary", model, "numpy", ModelOptions(evaluator="cache"))
        assert (r.evaluator, r.strategy, r.cache_size, r.batch_rows) == ("cache", "expanded", 24, False), model
        assert r.schur == ("schur" if model in GB_SCHUR else "dense"), model
    for model in sorted(set(MODELS) - set(GB_CACHED)):
        with pytest.raises(ValueError, match="cached GB models"):
            resolve("galactic_binary", model, "numpy", ModelOptions(evaluator="cache"))


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_jax_historical_is_gpu_jax_models_build(device):
    for model in MODELS:
        r = resolve("galactic_binary", model, "jax", device=device)
        cached = model in GB_CACHED
        assert (r.backend, r.device, r.cells, r.pad_factor, r.cache_size) == ("jax", device, "dense", "dense", 24)
        assert r.evaluator == ("cache" if cached else "batched"), model
        assert r.strategy == ("expanded" if cached else "residual"), model
        assert r.schur == ("schur" if model in GB_SCHUR else "dense"), model
        assert r.batch_rows is cached, model
        ref = resolve("galactic_binary", model, "jax", ModelOptions(strategy="reference"), device=device)
        assert (ref.evaluator, ref.schur, ref.batch_rows) == ("reference", "dense", False), model


def test_overrides_and_refusals():
    r = resolve("galactic_binary", "tri_pad", "numpy",
                ModelOptions(cells="fft", gb_pad_factor="schur", cache_size=3, waveform_backend="fourier"))
    assert (r.cells, r.pad_factor, r.cache_size, r.waveform_backend) == ("fft", "schur", 3, "fourier")
    assert resolve("galactic_binary", "band_raw", "numpy", ModelOptions(strategy="fast")) == \
        resolve("galactic_binary", "band_raw", "numpy")
    with pytest.raises(ValueError, match="evaluator"):
        ModelOptions(evaluator="lazy")
    with pytest.raises(ValueError, match="batched"):
        resolve("galactic_binary", "band_raw", "numpy", ModelOptions(evaluator="batched"))
    with pytest.raises(KeyError, match="chirp"):
        resolve("chirp", "band_raw", "numpy")
