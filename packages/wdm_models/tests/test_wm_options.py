"""E-A8: ModelOptions and the per-backend strategy table (spec §3.2 "The one knob", §5.3)."""

from __future__ import annotations

import dataclasses

import pytest

from wdm_models.common.params import MODELS
from wdm_models.options import ModelOptions, Resolved, resolve

TRI_UNMARGINALISED = ("tri_raw", "tri_cut")
EDGE_EXACT = ("band_exact", "mid_exact", "tri_exact")


def test_the_defaults():
    o = ModelOptions()
    assert (o.strategy, o.tri_build, o.rows_order, o.cells, o.gb_pad_factor, o.waveform_backend,
            o.cache_size, o.exact_chunk, o.whitened) == ("historical", None, None, None, None,
                                                         "fourier", None, None, False)
    assert (o.layout, o.phi_cache, o.staged, o.marginal_form, o.batch_rows) == (None,) * 5
    assert hash(o) == hash(ModelOptions())


def test_numpy_historical_is_the_wt_column():
    for model in MODELS:
        r = resolve("sinusoid", model, "numpy")
        assert isinstance(r, Resolved)
        assert (r.backend, r.device, r.tri_build, r.layout, r.rows_order, r.phi_cache, r.staged) == (
            "numpy", "cpu", "bessel", "ragged", "fixed", False, False), model
        assert (r.exact_on_fail, r.marginal_form, r.noise_clamp, r.cache_size, r.batch_rows) == (
            "raise", "solve", None, 32, False), model
        assert r.exact_chunk is None
        assert r.density_order == ("folded" if model == "exact" else "default"), model


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_jax_historical_is_the_wg_column(device):
    for model in MODELS:
        r = resolve("sinusoid", model, "jax", device=device)
        assert (r.backend, r.device, r.tri_build, r.layout, r.rows_order, r.phi_cache) == (
            "jax", device, "gram", "padded", "adaptive", True), model
        assert r.staged is (device == "cpu"), model
        assert r.exact_on_fail == "nan" and r.noise_clamp == (("alpha_d", 4.0),), model
        assert r.marginal_form == ("whiten" if model in TRI_UNMARGINALISED else "solve"), model
        assert r.cache_size == (32 if device == "cpu" else 16), model
        assert r.exact_chunk == (64 if device == "cpu" else 192), model
        assert r.batch_rows is (device != "cpu" and model not in EDGE_EXACT), model
        assert r.density_order == ("folded" if model == "exact" else "default"), model


def test_reference_forces_the_numpy_column_on_jax():
    for model in MODELS:
        r = resolve("sinusoid", model, "jax", ModelOptions(strategy="reference"), device="gpu")
        n = resolve("sinusoid", model, "numpy")
        assert (r.tri_build, r.layout, r.rows_order, r.phi_cache, r.staged, r.exact_on_fail,
                r.marginal_form, r.noise_clamp, r.batch_rows) == (
            n.tri_build, n.layout, n.rows_order, n.phi_cache, n.staged, n.exact_on_fail,
            n.marginal_form, n.noise_clamp, n.batch_rows), model
        assert (r.backend, r.device, r.cache_size) == ("jax", "gpu", 16)


def test_fast_is_historical_until_an_opt_in_passes_its_gate():
    for backend, device in (("numpy", None), ("jax", "cpu"), ("jax", "gpu")):
        for model in MODELS:
            assert resolve("sinusoid", model, backend, ModelOptions(strategy="fast"), device=device) == \
                resolve("sinusoid", model, backend, device=device)


def test_per_job_overrides_win():
    r = resolve("sinusoid", "tri_exact", "numpy",
                ModelOptions(tri_build="gram", rows_order="adaptive", layout="padded", phi_cache=True,
                             staged=True, exact_chunk=8, cache_size=4, marginal_form="whiten",
                             batch_rows=True))
    assert (r.tri_build, r.rows_order, r.layout, r.phi_cache, r.staged, r.exact_chunk, r.cache_size,
            r.marginal_form, r.batch_rows) == ("gram", "adaptive", "padded", True, True, 8, 4,
                                               "whiten", True)


def test_bad_options_are_refused():
    for bad in (dict(strategy="quick"), dict(tri_build="direct"), dict(rows_order="random"),
                dict(layout="dense"), dict(marginal_form="cholesky"), dict(cells="sparse"),
                dict(gb_pad_factor="lu"), dict(waveform_backend="gbonfly"), dict(cache_size=0),
                dict(exact_chunk=0), dict(staged=True, phi_cache=False)):
        with pytest.raises(ValueError):
            ModelOptions(**bad)
    with pytest.raises(ValueError, match="unknown model"):
        resolve("sinusoid", "band_win", "numpy")
    with pytest.raises(ValueError, match="backend"):
        resolve("sinusoid", "band_raw", "torch")
    with pytest.raises(KeyError, match="galactic_binary"):
        resolve("galactic_binary", "band_raw", "numpy")
    with pytest.raises(ValueError, match="device"):
        resolve("sinusoid", "band_raw", "jax", device="tpu")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ModelOptions().strategy = "fast"
