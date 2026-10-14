"""E-D2: the S4 completion of the strategy columns (the exact model's correction order and
E-C's WG-shaped forms),
WG's evaluation policy per model (spec sections 3.2 and 3.5) and the run metadata."""

from __future__ import annotations

import pytest

from wdm_models.common.params import MODELS, split_model
from wdm_models.options import ModelOptions, resolve
from wdm_models.sampling.device import DETERMINISTIC_FLAG, device_kind, run_meta
from wdm_models.sinusoid.evaluation import CACHED, EDGE_EXACT_MODELS, UNCACHED_DIAGONAL, Wrapper, wrap, wrapper_for


def test_the_s4_fields_complete_both_columns():
    """The exact model's correction order and E-C's WG-shaped forms: WT's everywhere on
    NumPy and under ``reference``; WG's on JAX (``reduce`` pads, ``scaled`` Schur over a
    diagonal interior, the ``table`` correction)."""
    for model in MODELS:
        base, edge = split_model(model)
        n = resolve("sinusoid", model, "numpy")
        assert (n.correction, n.pad_form, n.schur_form) == ("chebval2d", "solve", "solve"), model
        for device in ("cpu", "gpu"):
            j = resolve("sinusoid", model, "jax", device=device)
            assert j.correction == "table", model
            assert j.pad_form == ("reduce" if edge == "pad" else "solve"), model
            assert j.schur_form == ("scaled" if edge == "exact" and base in ("band", "mid") else "solve"), model
            ref = resolve("sinusoid", model, "jax", ModelOptions(strategy="reference"), device=device)
            assert (ref.correction, ref.pad_form, ref.schur_form) == ("chebval2d", "solve", "solve"), model


def test_the_model_sets_partition_the_fifteen_models():
    assert set(CACHED) | set(UNCACHED_DIAGONAL) | {"whittle_raw", "whittle_cut", "exact"} == set(MODELS)
    assert not set(CACHED) & set(UNCACHED_DIAGONAL) and set(EDGE_EXACT_MODELS) <= set(CACHED)


def test_numpy_builds_are_the_bare_likelihood():
    for model in MODELS:
        assert wrapper_for(model, resolve("sinusoid", model, "numpy")) == Wrapper("none"), model
    lik = object()
    assert wrap(lik, "tri_raw", resolve("sinusoid", "tri_raw", "numpy")) is lik


@pytest.mark.parametrize("device, capacity", [("cpu", 32), ("gpu", 16)])
def test_the_jax_policy_is_wg_per_model(device, capacity):
    for model in MODELS:
        w = wrapper_for(model, resolve("sinusoid", model, "jax", device=device))
        if model in CACHED:
            rows = device == "gpu" and model not in EDGE_EXACT_MODELS
            assert w == Wrapper("cache", capacity=capacity, batch_rows=rows), model
        else:
            assert w == Wrapper("batched", count_states=model not in UNCACHED_DIAGONAL), model


def test_options_reach_the_policy():
    r = resolve("sinusoid", "tri_raw", "jax", ModelOptions(cache_size=5, batch_rows=True), device="cpu")
    assert wrapper_for("tri_raw", r) == Wrapper("cache", capacity=5, batch_rows=True)


def test_device_kind_and_run_meta(monkeypatch):
    assert device_kind("numpy") == "cpu"
    with pytest.raises(ValueError, match="backend"):
        device_kind("torch")
    meta = run_meta("numpy")
    assert meta["backend"] == "numpy" and meta["device_kind"] == "cpu" and "jax" not in meta
    pytest.importorskip("jax")
    monkeypatch.setenv("XLA_FLAGS", DETERMINISTIC_FLAG)
    meta = run_meta("jax")
    assert meta["device_kind"] == device_kind("jax") and isinstance(meta["x64"], bool)
    assert meta["deterministic_ops"] is True and meta["jax"] == meta["jaxlib"] == "0.9.0.1"
    monkeypatch.delenv("XLA_FLAGS")
    assert run_meta("jax")["deterministic_ops"] is False
