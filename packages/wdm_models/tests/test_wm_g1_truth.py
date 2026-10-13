"""E-A11 (section gate): ``registry.build`` reproduces the G1 references on NumPy.

Every one of the 15 models at the manuscript_mid truth (G1 point 0) equals its CPU reference
bit for bit on the recording build (lineage A is bitwise there; the ``exact`` model through
``density_order="folded"``), else within the approved NumPy near-point gate ``atol``
(golden/sinusoid/tolerances_measured.json, approved 2026-10-12; ``rtol`` is the stress
term).  G1 point 4, where the edge-exact Schur complement is not positive definite, is
``-inf`` with ``last_failed`` set for all three edge-exact models, as in G1.  ~25 s: the
three edge-exact models compute 959 exact rows per point.
"""

from __future__ import annotations

import numpy as np
import pytest

from wdm_models import registry as R
from wdm_models.sinusoid import config as C

pytestmark = pytest.mark.golden
CASE = "manuscript_mid"
EDGE_EXACT = ("band_exact", "mid_exact", "tri_exact")


@pytest.mark.parametrize("model", R.MODELS)
def test_the_truth_is_the_g1_reference(model, g1, numpy_gate, recording_build):
    ref = g1(CASE, model)
    whittle = model.startswith("whittle")
    assert ref["theta"].shape == ((27, 5) if whittle else (30, 6))
    truth = C.config(CASE).truth(model)
    assert np.array_equal(ref["theta"][0], truth)
    assert ref["near"].shape == ref["loglike"].shape and bool(ref["near"][0])   # the truth is a near point
    want = float(ref["loglike"][0])
    assert np.isfinite(want)
    lik = R.build("sinusoid", CASE, model)
    got = float(lik(truth[None, :])[0])
    assert np.isfinite(got), model
    if recording_build:
        assert got == want, (model, got - want)
    else:
        atol, _ = numpy_gate(model)                   # near point: atol only (rtol is the stress term)
        assert abs(got - want) <= atol, (model, got - want)


@pytest.mark.parametrize("model", EDGE_EXACT)
def test_the_non_positive_definite_point_is_minus_inf(model, g1):
    ref = g1(CASE, model)
    assert ref["loglike"][4] == -np.inf
    lik = R.build("sinusoid", CASE, model)
    got = lik(ref["theta"][4][None, :])
    assert got.tolist() == [-np.inf]
    assert lik.last_failed.tolist() == [True]
    assert lik.counters()["pd_failed_states"] == 1 and lik.counters()["pd_failed_rows"] == 1


def test_the_bitwise_gate_sees_an_algorithm_change(g1, recording_build):
    """Negative control: the ``gram`` Eq. (45) build (the JAX default, exact Bessel sums) moves
    tri_raw's truth lnL off the ``bessel`` reference by a few 1e-9 (spec §3.2: <= 6.7e-9) --
    nonzero, so the bitwise comparison above is not vacuous, and inside the JAX gate."""
    from wdm_models.options import ModelOptions

    want = float(g1(CASE, "tri_raw")["loglike"][0])
    truth = C.config(CASE).truth("tri_raw")[None, :]
    got = float(R.build("sinusoid", CASE, "tri_raw", options=ModelOptions(tri_build="gram"))(truth)[0])
    assert 0.0 < abs(got - want) <= 1e-5
    if recording_build:
        assert abs(got - want) <= 6.7e-9
