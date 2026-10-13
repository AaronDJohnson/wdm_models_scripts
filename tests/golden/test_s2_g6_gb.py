"""D-D7 (S2 gate, G6): the GB models assembled from domainia blocks reproduce
golden/galactic_binary/intermediates_gb_3mo.{json,npz} at the anchor.

A preview of the S6 wiring (spec S2 gate): whittle_raw (complex Whittle on the nominal
bins), band_raw (separable diagonal), tri_exact (Eq. 45 separable + exact edge rows,
dense assembly) and exact (Hermitian M', expanded strategy).  Gates: lnL at the approved
G3 NumPy gate (1e-8), each channel's (quad, logdet) relative 1e-12; the GB numbers are
not claimed bitwise (spec section 3.6).  Also checked: the data cells (V_raw_cells), the
band variances at the anchor and the M' noise state per channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.goldens_io import to_float
from scripts.golden.s2_cases import GB_CASE, GB_MODELS, gb_compare, gb_gate, gb_inputs, gb_likelihood, record

pytestmark = pytest.mark.golden
ROOT = f"galactic_binary/{GB_CASE}"


@pytest.mark.parametrize("model", GB_MODELS)
def test_quad_logdet_lnl_at_the_anchor(model, record_property):
    out = gb_compare(GB_CASE, model)
    gates = {k: (gb_gate(GB_CASE, model) if k == "lnL" else 1e-12 * abs(ref)) for k, (_, ref, _) in out.items()}
    for key, (val, ref, d) in out.items():
        record(record_property, f"{ROOT}/quad_logdet/{model}/composition/{key}", d, gates[key])
    for key, (val, ref, d) in out.items():
        assert d <= gates[key], (model, key, val, ref)


def test_data_cells_and_band_variances(record_property):
    g = gb_inputs(GB_CASE)
    lik = gb_likelihood(GB_CASE, "band_raw")
    cells = np.asarray(lik.parts.y_data)
    d = float(np.max(np.abs(cells - g.npz["V_raw_cells"])))
    record(record_property, f"{ROOT}/arrays/V_raw_cells", d, 1e-12)
    assert d <= 1e-12
    state = lik.noise_state(lik.noise_slice_host(g.anchor))
    var = np.stack([np.asarray(state.factors[c].v) for c in range(2)])
    ref = g.npz["var_band_raw"]
    d = float(np.max(np.abs(var - ref) / np.abs(ref)))
    record(record_property, f"{ROOT}/arrays/var_band_raw", d, 1e-12)
    assert d <= 1e-12


def test_exact_noise_state_is_mprime(record_property):
    g = gb_inputs(GB_CASE)
    lik = gb_likelihood(GB_CASE, "exact")
    state = lik.noise_state(lik.noise_slice_host(g.anchor))
    for c in range(2):
        ref = g.json["scalars"][f"Mprime_c{c}"]
        dq = abs(float(state.q0[c]) - to_float(ref["q0"])) / abs(to_float(ref["q0"]))
        dl = abs(float(state.factors[c].logdet) - to_float(ref["logdet"])) / abs(to_float(ref["logdet"]))
        record(record_property, f"{ROOT}/scalars/Mprime_c{c}", max(dq, dl), 1e-12)
        assert max(dq, dl) <= 1e-12, (c, dq, dl)
