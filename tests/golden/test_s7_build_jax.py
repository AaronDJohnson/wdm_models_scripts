"""F-D3 (S7): ``registry.build("galactic_binary", case, model, backend="jax")``.

Every one of the 38 targets builds on JAX inside the legacy evaluator of
``scripts.golden.gb_jax_legacy.LEGACY_JAX`` (``NoiseStateCache`` of 24 states for the cache classes,
the rows of one call batched exactly where the legacy class batched them; ``BatchedLikelihood``
without noise-state counting for the rest -- the legacy class had no counters), its template is
F-A's ``GBTemplate`` pytree (orbit and scale are jit arguments: no closed-over array), and at the
anchor it agrees with the NumPy build within the approved G3 JAX gate (1e-8).  The padded store
grid's block layout (``PAD_LAYOUT = "padded"``: jittable) is bitwise the ragged one on NumPy.
Default lane: gb_3mo; gb_6mo/gb_12mo ``slow`` (12-month dense factors are 2700 x 2700).
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from domainia.sampling import BatchedLikelihood, NoiseStateCache  # noqa: E402
from scripts.golden.gb_jax_legacy import CACHE_CLASSES, LEGACY_JAX  # noqa: E402
from scripts.golden.paths import GOLDEN  # noqa: E402
from wdm_models.galactic_binary import models as gb_models  # noqa: E402
from wdm_models.galactic_binary.prepared import PreparedCase  # noqa: E402
from wdm_models.galactic_binary.waveform import GBTemplate  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import build  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
GOLDEN_OPTIONS = ModelOptions(waveform_backend="fourier")
G3_JAX = json.loads((GOLDEN / "galactic_binary" / "tolerances_measured.json").read_text())["approved"]["G3"]["jax"]


def _targets():
    return [pytest.param(case, model, id=f"{case}-{model}", marks=[] if case == "gb_3mo" else [pytest.mark.slow])
            for case, models in LEGACY_JAX.items() for model in models]


@pytest.mark.parametrize("case, model", _targets())
def test_every_target_builds_inside_its_legacy_evaluator(case, model):
    rec = LEGACY_JAX[case][model]
    lik = build("galactic_binary", case, model, backend="jax", options=GOLDEN_OPTIONS)
    if rec["evaluator"] in CACHE_CLASSES:
        assert type(lik) is NoiseStateCache and lik.capacity == rec["capacity"] == 24
        assert lik.batch_rows is rec["batched"] and lik.misses == "serial"
    else:
        assert type(lik) is BatchedLikelihood and lik.count_states is False
    inner = lik.lik
    assert inner.be.name == "jax" and getattr(inner, "jittable", True)
    assert isinstance(inner.parts.signal, GBTemplate)            # the pytree itself, not a closure
    prep = PreparedCase.load(case)
    x = prep.anchor(model)
    got = float(lik(x[None, :])[0])
    want = float(build("galactic_binary", case, model, backend="numpy", options=GOLDEN_OPTIONS)(x[None, :])[0])
    assert np.isfinite(got) and abs(got - want) <= G3_JAX[f"{case}/{model}"], (got, want)
    assert lik.last_failed.tolist() == [False]
    del lik
    gc.collect()


def test_a_template_that_is_not_the_pytree_is_refused_on_jax():
    prep = PreparedCase.load("gb_3mo")
    t = GBTemplate(prep, backend="fourier", window="store", check=False)
    with pytest.raises(TypeError, match="GBTemplate"):
        gb_models.build("gb_3mo", "band_raw", backend="jax", options=GOLDEN_OPTIONS, template=lambda p8: t(p8))
    with pytest.raises(ValueError, match="store"):
        gb_models.build("gb_3mo", "band_raw", backend="jax", options=GOLDEN_OPTIONS, template=t.on("window", prep))


def test_the_release_default_onfly_builds_on_jax():
    """The release default waveform ``onfly`` (spec section 5.2) builds on JAX too (no options): the
    signal is the onfly ``GBTemplate`` and at the anchor the JAX value equals the NumPy build's within
    the approved G3 JAX gate (measured <= 1.1e-13).  Its distance from ``fourier`` (~1e-2 in lnL) is
    F-C's onfly audit: reported, never gated."""
    prep = PreparedCase.load("gb_3mo")
    for model in ("band_raw", "tri_raw"):                      # BatchedLikelihood, NoiseStateCache
        x = prep.anchor(model)
        lik = build("galactic_binary", "gb_3mo", model, backend="jax")
        assert lik.lik.parts.signal.backend == "onfly", model
        got = float(lik(x[None, :])[0])
        want = float(build("galactic_binary", "gb_3mo", model)(x[None, :])[0])
        assert np.isfinite(got) and abs(got - want) <= G3_JAX[f"gb_3mo/{model}"], (model, got, want)


def test_the_padded_store_layout_is_bitwise_the_ragged_one_on_numpy(monkeypatch):
    prep = PreparedCase.load("gb_3mo")
    X = np.stack([prep.anchor("tri_pad"), prep.anchor("tri_pad") + np.r_[np.zeros(8), 0.05 * np.ones(10)]])
    assert gb_models.PAD_LAYOUT == "padded"
    padded = build("galactic_binary", "gb_3mo", "tri_pad", options=GOLDEN_OPTIONS)(X)
    monkeypatch.setattr(gb_models, "PAD_LAYOUT", "ragged")
    ragged = build("galactic_binary", "gb_3mo", "tri_pad", options=GOLDEN_OPTIONS)(X)
    assert np.all(np.isfinite(padded)) and np.array_equal(padded, ragged)
