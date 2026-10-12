"""B4 (G6, GB part): per-module intermediates at the anchor, one npz + json per case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden" / "galactic_binary"
#: P per channel, exact-edge rows, NT, padded NT, store bins.
GEOMETRY = {"gb_3mo": (720, 27, 240, 256, 1139), "gb_6mo": (1314, 27, 438, 454, 2081),
            "gb_12mo": (2700, 0, 900, 916, 4275)}
N_MODELS = {"gb_3mo": 15, "gb_6mo": 15, "gb_12mo": 8}
FINGERPRINT_KEYS = {"shape", "dtype", "sha256", "fro", "idx"}


@pytest.mark.parametrize("case", sorted(GEOMETRY))
def test_intermediates_schema_and_shapes(case):
    meta = json.loads((GOLDEN / f"intermediates_{case}.json").read_text())
    P, nE, NT, NTp, n_store = GEOMETRY[case]
    assert meta["schema"] == 1 and meta["case"] == case
    for e in ("raw", "cut", "exact", "pad"):
        for name in ("C", "U", "b", "n"):
            assert FINGERPRINT_KEYS <= set(meta["fingerprints"][f"V_{e}_{name}"])
    assert meta["scalars"]["V_raw"]["P"] == P and meta["scalars"]["V_pad"]["NT"] == NTp
    assert meta["scalars"]["V_pad"]["cols"] == [0, n_store]
    assert meta["scalars"]["pad_mask"]["n_rows"] == 99 and meta["scalars"]["pad_mask"]["R_nB"] == 11
    assert meta["fingerprints"]["KI"]["shape"] == [2, P, P]
    assert meta["fingerprints"]["V_raw_C"]["dtype"] == "<c16"
    assert len(meta["quad_logdet_models"]) == N_MODELS[case]
    for model, chk in meta["lnl_at_anchor"].items():
        assert chk["abs_diff"] <= 1e-9 * max(1.0, abs(chk["lnl_make_lnlike"])), model
    with np.load(GOLDEN / f"intermediates_{case}.npz") as z:
        assert sorted(z.files) == meta["arrays"]
        assert z["V_raw_cells"].shape == (2, P) and z["V_pad_cells"].shape == (2, P)
        assert z["V_exact_rows"].shape == (nE,)
        assert z["pad_E"].shape == (99,) and z["pad_Kpos"].shape == (P,)
        assert z["KI_diag"].shape == (2, P) and z["var_band_raw"].shape == (2, P)
        assert z["cov_tri_raw_diag"].shape == (2, P) and z["cov_tri_pad_diag"].shape == (2, P)
        assert z["ghat_c0"].dtype == np.complex128 and z["store_template"].shape == (2, n_store)
        assert z["quad_logdet"].shape == (N_MODELS[case], 2, 2)
        assert np.all(np.isfinite(z["quad_logdet"]))
        if nE:
            assert z["exact_rows_c0_sel"].shape == (3, P)
        # the diagonal of the tri_raw covariance is K^I + g K^G on the diagonal: positive
        assert np.all(z["cov_tri_raw_diag"] > 0) and np.all(z["var_mid_raw"] > 0)


def test_anchor_lnl_matches_g3_spot():
    meta = json.loads((GOLDEN / "intermediates_gb_3mo.json").read_text())
    # the 3-month exact chain (seed 3009) starts at the anchor: row 0 = -600.5155231449987
    assert abs(meta["lnl_at_anchor"]["exact"]["lnl_make_lnlike"] - (-600.5155231449987)) < 5e-10
