"""F-C4: the galactic binary in the registry (spec section 5.3): 38 targets, ``build`` and
``describe`` == lineage C's ``matrix.describe`` (golden/galactic_binary/products.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domainia.likelihood import FourierWhittle, GaussianLikelihood
from domainia.sampling import BatchedLikelihood, NoiseStateCache
from wdm_models import registry as R
from wdm_models.options import ModelOptions

PRODUCTS = Path(__file__).resolve().parents[3] / "golden" / "galactic_binary" / "products.json"


def test_targets():
    gb = R.TARGETS["galactic_binary"]
    assert tuple(gb) == ("gb_3mo", "gb_6mo", "gb_12mo")
    assert gb["gb_3mo"] == gb["gb_6mo"] == R.MODELS
    assert gb["gb_12mo"] == ("whittle_raw", "band_raw", "mid_raw", "tri_raw", "band_pad", "mid_pad", "tri_pad",
                             "exact")
    t = R.targets("galactic_binary")
    assert len(t) == len(set(t)) == 38 and R.targets()[45:] == t


@pytest.mark.golden
def test_describe_is_lineage_c_describe():
    """Key for key (720 / 693 / 27 / 99 at 3 months ...), plus toy/case/model/params."""
    prod = json.loads(PRODUCTS.read_text())["cases"]
    for _, case, model in R.targets("galactic_binary"):
        d = R.describe("galactic_binary", case, model)
        assert (d.pop("toy"), d.pop("case"), d.pop("model")) == ("galactic_binary", case, model)
        params = d.pop("params")
        assert len(params) == (10 if model.startswith("whittle") else 18) and params[:2] == ("f0T", "cos_colat")
        assert d == prod[case]["describe"][model], (case, model)
    by = {m: R.describe("galactic_binary", "gb_3mo", m) for m in R.MODELS}
    assert (by["band_raw"]["P"], by["band_cut"]["P"], by["tri_exact"]["exact_rows"], by["mid_pad"]["mask_rows"],
            by["whittle_raw"]["P"], by["whittle_cut"]["P"]) == (720, 693, 27, 99, 360, 347)


@pytest.mark.golden
def test_build_returns_the_models():
    pytest.importorskip("binary_galactica", reason="binary_galactica (JAX) not installed")
    lik = R.build("galactic_binary", "gb_3mo", "band_raw", options=ModelOptions(waveform_backend="fourier"))
    assert isinstance(lik, GaussianLikelihood) and lik.ndim == 18 and lik.n_eff == (720, 720)
    w = R.build("galactic_binary", "gb_6mo", "whittle_cut", options=ModelOptions(waveform_backend="fourier"))
    assert isinstance(w, FourierWhittle) and w.ndim == 10 and w.n_keep == 644
    c = R.build("galactic_binary", "gb_3mo", "tri_exact", options=ModelOptions(evaluator="cache"))
    assert isinstance(c, NoiseStateCache) and c.lik.strategy == "expanded"
    j = R.build("galactic_binary", "gb_3mo", "band_raw", backend="jax", options=ModelOptions(waveform_backend="fourier"))
    assert isinstance(j, BatchedLikelihood) and j.count_states is False and j.lik.be.name == "jax"   # S7 (F-D3)
