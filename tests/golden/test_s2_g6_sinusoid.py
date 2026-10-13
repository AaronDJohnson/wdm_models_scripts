"""D-D6 (S2 gate, G6): the sinusoid models assembled from domainia blocks reproduce
golden/sinusoid/intermediates/manuscript_mid_truth.json at the truth.

A preview of the S3 wiring on a representative set (spec S2 gate): both Whittle forms,
band_raw, mid_cut (default lane); tri_raw, band_pad, band_exact (``slow``: Eq. 45
blocks, the 1024^2 pad marginal and 959 exact rows take ~20 s).  Gates: (quad, logdet)
relative 1e-12 and lnL at the approved NumPy reference gate (``atol + rtol |lnL|``,
golden/sinusoid/tolerances_measured.json); on the recording build the bitwise-class
models must be exact.  The mapped data and nuisance directions are checked as G6 maps.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.s2_cases import (SIN_BITWISE, SIN_DEFAULT, SIN_SLOW, record, recording_build,
                                     sinusoid_compare, sinusoid_gate, sinusoid_golden, sinusoid_likelihood)
from scripts.golden.workers._common import array_summary

pytestmark = pytest.mark.golden
EXACT_HERE = recording_build()
ROOT = "sinusoid/manuscript_mid"


def _model_params():
    return ([pytest.param(m, id=m) for m in SIN_DEFAULT]
            + [pytest.param(m, id=m, marks=pytest.mark.slow) for m in SIN_SLOW])


@pytest.mark.parametrize("model", _model_params())
def test_quad_logdet_lnl_at_the_truth(model, record_property):
    out = sinusoid_compare(model)
    atol, rtol = sinusoid_gate(model)
    exact = EXACT_HERE and model in SIN_BITWISE
    for key, (val, ref, d) in out.items():
        gate = 0.0 if exact else (atol + rtol * abs(ref) if key == "lnL" else 1e-12 * abs(ref))
        record(record_property, f"{ROOT}/parts/{model}/composition/{key}", d, gate)
    for key, (val, ref, d) in out.items():
        gate = 0.0 if exact else (atol + rtol * abs(ref) if key == "lnL" else 1e-12 * abs(ref))
        assert d <= gate, (model, key, val, ref)


@pytest.mark.parametrize("model", [m for m in SIN_DEFAULT if not m.startswith("whittle")]
                         + [pytest.param(m, marks=pytest.mark.slow) for m in SIN_SLOW])
def test_mapped_data_and_nuisance_directions(model, record_property):
    """``maps.<model>.y_data`` and ``.u`` (the WT ``v.map(d)`` and ``nuisance_basis(...)``)."""
    golden = sinusoid_golden().json["maps"][model]
    lik = sinusoid_likelihood(model)
    arrays = {"y_data": np.asarray(lik.parts.y_data), "u": np.asarray(lik.covs[0].U)}
    assert lik.n_eff == (int(golden["n_cells"]) - arrays["u"].shape[1],)
    for key, arr in arrays.items():
        g, s = golden[key], array_summary(arr)
        assert s["shape"] == g["shape"]
        d = abs(s["l2"] - float(g["l2"])) / float(g["l2"])
        record(record_property, f"{ROOT}/maps/{model}/{key}", d, 0.0 if EXACT_HERE else 1e-12)
        if EXACT_HERE:
            assert s["sha256"] == g["sha256"], (model, key)
        assert d <= 1e-12, (model, key, d)
