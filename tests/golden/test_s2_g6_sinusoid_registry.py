"""E-B8 (S3 closes its part of the S2 gate, spec section 7.2 G6): the eight sinusoid
compositions the S2 preview left to the full registry -- mid_raw, band_cut, tri_cut,
mid_pad, tri_pad, mid_exact, tri_exact and exact -- built by
``wdm_models.registry.build("sinusoid", "manuscript_mid", model)`` reproduce
golden/sinusoid/intermediates/manuscript_mid_truth.json ``parts.<model>`` (quad, logdet,
lnL) at the truth (the exact model's unit-amplitude ``quad_unit``/``logdet_unit``).

The S2 gate (``scripts/golden/s2_report.py --full --strict``, test glob
``tests/golden/test_s2_g6_*.py``) runs this file: its records cover the eight
``sinusoid/manuscript_mid/parts/<model>/composition`` items, which
``s2_report.deferred()`` no longer lists -- a run without them leaves them uncovered
(``--strict``: exit 1).  Gates as tests/golden/test_s2_g6_sinusoid.py: on the recording
build the bitwise-class models exactly; otherwise (quad, logdet) relative 1e-12 and lnL at
the approved NumPy reference gate (``atol + rtol |lnL|``).  Default lane: mid_raw, band_cut,
exact (~1 s); ``slow``: the tri, pad and edge-exact models (Eq. 45 blocks, the pad marginal,
959 exact rows; ~15 s).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.goldens_io import to_float
from scripts.golden.s2_cases import (SIN_BITWISE, SIN_DEFAULT, SIN_SLOW, record, recording_build,
                                     sinusoid_gate, sinusoid_golden)
from wdm_models import registry
from wdm_models.sinusoid.config import config

pytestmark = pytest.mark.golden
EXACT_HERE = recording_build()
ROOT = "sinusoid/manuscript_mid"
S3_DEFAULT = ("mid_raw", "band_cut", "exact")
S3_SLOW = ("tri_cut", "mid_pad", "tri_pad", "mid_exact", "tri_exact")
#: ``quad_logdet`` is at unit amplitude (A = 1); the exact model's G6 entry stores ``quad`` and
#: ``logdet`` at A_n (WT ``FastFullCovariance``) next to the unit values.
UNIT = {("exact", "quad"): "quad_unit", ("exact", "logdet"): "logdet_unit"}


def test_the_registry_set_is_exactly_what_s2_deferred():
    """The S2 preview (``s2_cases``) and this file partition the 15 models."""
    s2, s3 = set(SIN_DEFAULT + SIN_SLOW), set(S3_DEFAULT + S3_SLOW)
    assert not s2 & s3 and s2 | s3 == set(registry.MODELS) and len(s3) == 8


@pytest.mark.parametrize("model", [pytest.param(m, id=m) for m in S3_DEFAULT]
                         + [pytest.param(m, id=m, marks=pytest.mark.slow) for m in S3_SLOW])
def test_registry_composition_at_the_truth(model, record_property):
    golden = sinusoid_golden().json["parts"][model]
    lik = registry.build("sinusoid", "manuscript_mid", model)
    theta = config("manuscript_mid").truth(model)
    (quad, logdet), = lik.quad_logdet(theta)
    lnl = float(lik(theta[None, :])[0])
    atol, rtol = sinusoid_gate(model)
    exact = EXACT_HERE and model in SIN_BITWISE
    out = {}
    for key, val in (("quad", quad), ("logdet", logdet), ("lnL", lnl)):
        ref = to_float(golden[UNIT.get((model, key), key)])
        assert np.isfinite(ref) and np.isfinite(val), (model, key, val, ref)    # never a NaN pass
        gate = 0.0 if exact else (atol + rtol * abs(ref) if key == "lnL" else 1e-12 * abs(ref))
        out[key] = (float(val), ref, abs(float(val) - ref), gate)
        record(record_property, f"{ROOT}/parts/{model}/composition/{key}", out[key][2], gate)
    for key, (val, ref, d, gate) in out.items():
        assert d <= gate, (model, key, val, ref, d, gate)
