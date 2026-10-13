"""E-A5: the single sinusoid prior table (spec §5.1 priors.py) against the oracle's
``priors.for_model``, the runner's recorded bounds of all 45 final runs and the bandcov key."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from domainia.likelihood import Params
from wdm_models.common.params import MODELS
from wdm_models.sinusoid import priors

BANDCOV = (Path(__file__).resolve().parents[3] / "data" / "sinusoid" / "cache"
           / "bandcov_v1_4bfb466c191c79f9879a.npz")


@pytest.mark.golden
def test_the_table_is_the_oracle_prior_for_every_case(facts):
    assert sorted(facts["cases"]) == ["manuscript_high", "manuscript_low", "manuscript_mid"]
    for case, f in facts["cases"].items():
        for wt_model, model in (("freq", "whittle_raw"), ("wdm_full", "band_raw")):
            wt = f["priors"][wt_model]
            table = priors.table(model)
            assert [b.name for b in table] == wt["names"], (case, model)
            assert [b.lo for b in table] == wt["lo"] and [b.hi for b in table] == wt["hi"], (case, model)
            assert [b.periodic for b in table] == wt["periodic"], (case, model)


@pytest.mark.golden
def test_every_final_run_recorded_this_table(facts):
    """The runner's ``_DRIFT_BOUNDS`` / ``_FREQUENCY_BOUNDS`` as each of the 45 runs stored them."""
    runs = facts["runner"]["runs"]
    assert len(runs) == 45
    for key, r in runs.items():
        model = key.split("/")[1]
        assert list(priors.names(model)) == r["parameter_names"], key
        assert priors.bounds(model).tolist() == r["prior_bounds"], key
        assert {str(k): list(v) for k, v in priors.periodic(model).items()} == r["periodic"], key


@pytest.mark.golden
def test_noise_bounds_are_pinned_because_they_key_the_bandcov_artifact(g6):
    nb = priors.noise_bounds()
    assert nb.dtype == np.float64 and nb.shape == (2, 2)
    assert np.array_equal(nb, np.asarray([[-4.0, 0.0], [-2.0, 2.0]]))
    assert nb.tolist() == g6["parts"]["exact"]["bandcov_identity"]["bounds"]
    with np.load(BANDCOV, allow_pickle=False) as z:
        assert nb.tolist() == json.loads(str(z["metadata"]))["identity"]["bounds"]
    nb[0, 0] = 99.0                                   # a copy: the table cannot be edited through it
    assert priors.noise_bounds()[0, 0] == -4.0


def test_layouts():
    for model in MODELS:
        p = priors.params(model)
        assert isinstance(p, Params) and p.names == priors.names(model)
        assert p.source == (0, 1, 2)
        assert p.noise == ((3, 4) if model.startswith("whittle") else (4, 5))
        assert p.index("log10_A_n") == 3
        assert priors.bounds(model).shape == (p.ndim, 2)
        assert priors.periodic(model) == {2: (-math.pi, math.pi)} == priors.PERIODIC


def test_the_box_wraps_the_phase_and_rejects_the_rest():
    truth = np.array([0.30421576834679953, 137.0, 0.6, np.log10(400.0), -2.0, 1.2])
    X = np.stack([truth, truth, truth, truth, truth])
    X[1, 2] = 0.6 + 2 * np.pi            # wrapped back into [-pi, pi)
    X[2, 5] = 2.5                        # alpha_d outside [-2, 2]
    X[3, 1] = np.nan
    X[4, 0] = -1e-9                      # A_h below 0
    assert priors.in_prior("band_raw", X).tolist() == [True, True, False, False, False]
    assert priors.wrap("band_raw", X)[1, 2] == pytest.approx(0.6, abs=1e-14)
    assert priors.in_prior("whittle_raw", truth[:5][None, :]).tolist() == [True]
    with pytest.raises(ValueError, match=r"\(R, 6\)"):
        priors.in_prior("band_raw", truth[:5][None, :])
