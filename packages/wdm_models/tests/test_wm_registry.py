"""E-A10: the registry (spec §5.3): MODELS, TARGETS, build, describe."""

from __future__ import annotations

import numpy as np
import pytest

from domainia.likelihood import FourierWhittle, GaussianLikelihood
from wdm_models import registry as R
from wdm_models.options import ModelOptions
from wdm_models.sinusoid import config as C

CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")


def test_models_and_targets():
    assert R.MODELS == ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact",
                        "mid_raw", "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad",
                        "tri_exact", "exact")
    assert R.TOYS == ("sinusoid",)
    assert R.TARGETS == {"sinusoid": {case: R.MODELS for case in CASES}}
    t = R.targets()
    assert len(t) == 45 and len(set(t)) == 45 and t[0] == ("sinusoid", "manuscript_low", "whittle_raw")
    assert R.targets("sinusoid") == t


@pytest.mark.golden
def test_describe_is_the_legacy_counts(g6):
    """The counts the WT oracle reports (G6 ``maps``/``exact_rows``/``parts``): cells 16384 raw,
    15681 cut, 959 exact rows over 15425 interior cells, 17408 padded cells; Whittle 16256 / 15556
    retained coordinates; 128 discarded Fourier modes everywhere."""
    for model in R.MODELS:
        d = R.describe("sinusoid", "manuscript_mid", model)
        assert (d["toy"], d["case"], d["model"]) == ("sinusoid", "manuscript_mid", model)
        assert d["params"] == ("A_h", "f_0", "phi_0", "log10_A_n") + (
            ("alpha",) if model.startswith("whittle") else ("alpha_c", "alpha_d"))
        if model.startswith("whittle"):
            part = g6["parts"][model]
            assert d["P"] == part["n_keep"], model
            if model == "whittle_cut":
                assert (list(d["crop"]), d["samples"]) == (part["crop"], part["Nc"])
            continue
        if model == "exact":
            assert (d["P"], d["n_nuisance"], d["n_eff"]) == (16384, 128, g6["parts"]["exact"]["n_keep"])
            continue
        m = g6["maps"][model]
        assert d["P"] == m["n_cells"], model
        assert d["n_nuisance"] == m["u"]["shape"][1] == 128, model
        assert d["n_eff"] == m["n_cells"] - 128, model
        assert d["log_jacobian"] == m["log_jac"], model
    by = {m: R.describe("sinusoid", "manuscript_mid", m) for m in R.MODELS}
    assert (by["band_raw"]["P"], by["band_cut"]["P"], by["tri_exact"]["exact_rows"],
            by["mid_pad"]["lambda_cells"]) == (16384, 15681, 959, 17408)
    assert by["tri_exact"]["interior"] == 15425 == g6["exact_rows"]["n_interior"]
    assert by["band_exact"]["exact_rows"] == g6["exact_rows"]["n_edge"]
    assert by["tri_pad"]["lambda_cells"] == g6["maps"]["Wp"]["shape"][0]
    assert by["band_pad"]["NT"] == 272 and by["band_raw"]["NT"] == 256
    assert by["tri_cut"]["cut"] == (6, 5) and by["mid_exact"]["exact_columns"] == (8, 7)


@pytest.mark.golden
def test_build_returns_the_models_on_the_shipped_data():
    lik = R.build("sinusoid", "manuscript_mid", "band_raw")
    assert isinstance(lik, GaussianLikelihood)
    assert lik.param_names == ("A_h", "f_0", "phi_0", "log10_A_n", "alpha_c", "alpha_d")
    assert lik.n_eff == (16256,)
    w = R.build("sinusoid", "manuscript_low", "whittle_cut")
    assert isinstance(w, FourierWhittle) and w.crop == (384, 16064) and w.n_keep == 15556
    ex = R.build("sinusoid", "manuscript_high", "exact")
    assert ex.density_order == "folded" and ex.n_eff == (16256,)
    assert R.build("sinusoid", "manuscript_mid", "tri_raw", options=ModelOptions(marginal_form="whiten")) \
        .covs[0].form == "whiten"
    truth = C.config("manuscript_mid").truth("band_raw")
    assert np.isfinite(lik(truth[None, :])[0])


def test_what_the_registry_does_not_build_is_refused():
    with pytest.raises(ValueError, match="backend"):
        R.build("sinusoid", "manuscript_mid", "band_raw", backend="torch")
    with pytest.raises(KeyError, match="S5"):
        R.build("galactic_binary", "gb_3mo", "band_raw")
    with pytest.raises(KeyError, match="S5"):
        R.describe("galactic_binary", "gb_3mo", "band_raw")
    with pytest.raises(KeyError, match="manuscript_chirp"):
        R.build("sinusoid", "manuscript_chirp", "band_raw")
    with pytest.raises(ValueError, match="unknown model"):
        R.describe("sinusoid", "manuscript_mid", "band_win")
