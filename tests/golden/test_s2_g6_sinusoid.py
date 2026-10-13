"""D-D6 (S2 gate, G6): the sinusoid models assembled from domainia blocks reproduce
golden/sinusoid/intermediates/manuscript_mid_truth.json at the truth.

A preview of the S3 wiring on a representative set (spec S2 gate): both Whittle forms,
band_raw, mid_cut (default lane); tri_raw, band_pad, band_exact (``slow``: Eq. 45
blocks, the 1024^2 pad marginal and 959 exact rows take ~20 s).  Gates: (quad, logdet)
relative 1e-12 and lnL at the approved NumPy reference gate (``atol + rtol |lnL|``,
golden/sinusoid/tolerances_measured.json); on the recording build the bitwise-class
models must be exact.  The mapped data and nuisance directions are checked as G6 maps.
The ``exact`` model (an S3 composition, not a preview) is checked here only for its density
order against its G1 references (G-F1, D1); it records no G6 item.
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


#: The truth and the two G1 points where the default density order is not bitwise (G-F1, D1).
EXACT_G1_POINTS = (0, 8, 18)


def test_exact_folded_density_order_reproduces_g1():
    """D1 (G-F1): the sinusoid ``exact`` composition (``ProjectedOperatorMarginal`` over the
    shipped ``c_N`` and bandcov interpolant) with ``density_order="folded"`` -- WT
    ``FastFullCovariance.lnlike``'s association, the registry default for ``exact`` --
    reproduces the G1 reference loglike (golden/sinusoid/references/
    manuscript_mid_exact_reference.npz) bitwise at the truth and at points 8 and 18 on the
    recording build.  The ``"default"`` order (``RealGaussian.lnl``) is unchanged: inside
    the approved gate everywhere, bitwise at the truth, off in the last bits at 8 and 18."""
    from scripts.golden.goldens_io import read_npz
    from scripts.golden.paths import GOLDEN

    ref = read_npz(GOLDEN / "sinusoid" / "references" / "manuscript_mid_exact_reference.npz")
    idx = list(EXACT_G1_POINTS)
    theta, want = ref["theta"][idx], ref["loglike"][idx]
    folded = sinusoid_likelihood("exact")
    default = sinusoid_likelihood("exact", density_order="default")
    assert (folded.density_order, default.density_order) == ("folded", "default")
    assert folded.n_eff == (16384 - 128,)
    got, got_default = folded(theta), default(theta)
    atol, rtol = sinusoid_gate("exact")
    gate = atol + rtol * np.abs(want)
    assert np.all(np.abs(got - want) <= gate) and np.all(np.abs(got_default - want) <= gate)
    if EXACT_HERE:
        assert np.array_equal(got, want), (got - want)
        assert got_default[0] == want[0] and np.all(got_default[1:] != want[1:]), (got_default - want)


@pytest.mark.slow
def test_exact_folded_density_order_at_every_g1_point():
    """All 30 G1 points (``slow``, ~7 s): the folded order is bitwise at every point on the
    recording build; the default order differs at exactly points 8 and 18 (the review's
    measurement), and both stay inside the approved gate."""
    from scripts.golden.goldens_io import read_npz
    from scripts.golden.paths import GOLDEN

    ref = read_npz(GOLDEN / "sinusoid" / "references" / "manuscript_mid_exact_reference.npz")
    theta, want = ref["theta"], ref["loglike"]
    assert np.all(np.isfinite(want))
    got = sinusoid_likelihood("exact")(theta)
    got_default = sinusoid_likelihood("exact", density_order="default")(theta)
    atol, rtol = sinusoid_gate("exact")
    gate = atol + rtol * np.abs(want)
    assert np.all(np.abs(got - want) <= gate) and np.all(np.abs(got_default - want) <= gate)
    if EXACT_HERE:
        assert np.array_equal(got, want), np.flatnonzero(got != want)
        assert np.flatnonzero(got_default != want).tolist() == [8, 18]
