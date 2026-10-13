"""D-D5 (S2 gate, G6): the densities turn every golden (quad, logdet) into its golden lnL.

Pure scalar arithmetic on golden inputs: exact on the recording build, <= 1e-12 relative
elsewhere (``np.log`` may differ by an ulp across libms).  Covers all 15 sinusoid models
and all 38 GB targets, so every declared constant of spec section 4.7 is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from domainia.density import ComplexGaussian, RealGaussian
from scripts.golden.goldens_io import read_json, read_npz, to_float
from scripts.golden.paths import GOLDEN
from scripts.golden.s2_cases import record, recording_build

pytestmark = pytest.mark.golden

SIN = read_json(GOLDEN / "sinusoid" / "intermediates" / "manuscript_mid_truth.json")
EXACT_HERE = recording_build()


def _check(record_property, item: str, got: float, ref: float) -> None:
    d = abs(got - ref)
    gate = 0.0 if EXACT_HERE else 1e-12 * abs(ref)
    record(record_property, item, d, gate)
    assert d <= gate, (item, got, ref)


@pytest.mark.parametrize("model", sorted(SIN["parts"]))
def test_sinusoid_density_reproduces_the_golden_lnl(model, record_property):
    p = SIN["parts"][model]
    if model.startswith("whittle"):
        got = RealGaussian().lnl(to_float(p["quad"]), to_float(p["logdet"]), int(p["n_keep"]))
    elif model == "exact":
        # FastFullCovariance (likelihood.py:577-584): A folded into each term first
        log10A = np.float64(np.log10(400.0))
        got = RealGaussian(0.0, "log10_A_n").lnl_folded(to_float(p["quad_unit"]), to_float(p["logdet_unit"]),
                                                        int(p["n_keep"]), log10A)
    else:
        log10A = np.float64(np.log10(400.0))
        assert 10.0 ** log10A == to_float(p["A_n"])
        if model.endswith("_pad"):
            assert to_float(p["log_jac"]) == 0.0
        else:   # 1/2 N ln dt with the FULL N = 16384, also for the cut models (spec section 4.7)
            assert to_float(p["log_jac"]) == 0.5 * 16384 * float(np.log(2.0 ** -10))
        got = RealGaussian(to_float(p["log_jac"]), "log10_A_n").lnl(
            to_float(p["quad"]), to_float(p["logdet"]), int(p["n_eff"]), log10A)
    _check(record_property, f"sinusoid/manuscript_mid/parts/{model}/density", float(got), to_float(p["lnL"]))


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo", "gb_12mo"])
def test_gb_density_reproduces_the_golden_lnl(case, record_property):
    g = read_json(GOLDEN / "galactic_binary" / f"intermediates_{case}.json")
    ql = read_npz(GOLDEN / "galactic_binary" / f"intermediates_{case}.npz")["quad_logdet"]
    for i, model in enumerate(g["quad_logdet_models"]):
        total = 0.0                                   # channels summed in order from 0.0
        for c in range(2):
            if model.startswith("whittle") or model == "exact":
                total = total + ComplexGaussian().lnl(ql[i, c, 0], ql[i, c, 1])
            else:
                P = int(g["scalars"]["V_" + model.split("_")[1]]["P"])
                total = total + RealGaussian().lnl(ql[i, c, 0], ql[i, c, 1], P)
        ref = to_float(g["lnl_at_anchor"][model]["lnl_make_lnlike"])
        _check(record_property, f"galactic_binary/{case}/quad_logdet/{model}/density", float(total), ref)
