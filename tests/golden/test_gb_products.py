"""B3 (G5, GB part): templates, describe() counts, snr_full, priors, injection, Table III."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden" / "galactic_binary"
#: Spec 5.2: sum|h|^2 / Re<z,h> (gbonfly, BW = 4096, prepared units), 10 decimals.
TEMPLATE = {"gb_3mo": ((47.0844510309, 44.6213355729), (47.0992463067, 44.6206986397)),
            "gb_6mo": ((75.3784950251, 78.7189953021), (75.4114941003, 78.6860544692)),
            "gb_12mo": ((160.9964196341, 167.5473945629), (160.9964198590, 167.5474286859))}
#: Spec 4.9: whittle, whittle_cut, WDM, cut, exact-edge rows, pad mask rows (per channel).
COUNTS = {"gb_3mo": (360, 347, 720, 693, 27, 99), "gb_6mo": (657, 644, 1314, 1287, 27, 99),
          "gb_12mo": (1350, 1350, 2700, 2700, 0, 99)}
SNR_FULL = {"gb_3mo": 12.24709552247082, "gb_6mo": 14.522176865090799, "gb_12mo": 20.25921338411754}
HALF_WIDTH = {"gb_3mo": 121.96, "gb_6mo": 21.40, "gb_12mo": 8.0}
TABLE_III_3MO_A = (-7.380, 5.177, 8.298, 0.552, -2.282)


@pytest.fixture(scope="module")
def products():
    return json.loads((GOLDEN / "products.json").read_text())


def test_schema(products):
    assert products["schema"] == 1 and set(products["cases"]) == set(TEMPLATE)
    assert len(products["models"]) == 15 and "whittle_win" not in products["models"]
    assert products["constants"]["TEMPLATE_BW"] == 4096
    assert products["constants"]["CUT"] == {"gb_3mo": [5, 4], "gb_6mo": [5, 4], "gb_12mo": [0, 0]}


@pytest.mark.parametrize("case", sorted(TEMPLATE))
def test_template_sums_match_spec(products, case):
    t = products["cases"][case]["template"]
    (w_h2, w_zh), (s_h2, s_zh) = TEMPLATE[case]
    assert round(t["window"]["sum_h2"], 10) == w_h2 and round(t["window"]["re_zh"], 10) == w_zh
    assert round(t["store"]["sum_h2"], 10) == s_h2 and round(t["store"]["re_zh"], 10) == s_zh


@pytest.mark.parametrize("case", sorted(COUNTS))
def test_counts_and_flags(products, case):
    c = products["cases"][case]
    k = c["counts"]
    assert (k["whittle"], k["whittle_cut"], k["wdm"], k["cut"], k["exact_edge_rows"],
            k["pad_mask_rows"]) == COUNTS[case]
    assert k["lambda_R"].startswith("11 bands (store) x ")
    assert c["snr_full"]["recomputed"] == SNR_FULL[case] == c["snr_full"]["report"]
    assert c["prior"]["equal_ucb_prior"] and c["inject"]["equal"]
    assert c["envelope"]["basis_L_equals_ghat_grid"]
    assert round(c["envelope"]["prior_half_width"], 2) == HALF_WIDTH[case]
    assert c["prior"]["lo"][3] == -54.0 and c["prior"]["hi"][3] == -46.0
    assert c["psd_check"]["max_rel_sI"] < 1e-14 and c["psd_check"]["max_rel_sG"] < 1e-14


def test_table_iii_physical_coefficients(products):
    phys = products["cases"]["gb_3mo"]["envelope"]["physical_A"]
    assert tuple(round(x, 3) for x in phys) == TABLE_III_3MO_A


def test_psd_window_arrays():
    with np.load(GOLDEN / "psd_window.npz") as z:
        for case, n in {"gb_3mo": 419, "gb_6mo": 767, "gb_12mo": 1575}.items():
            assert z[f"{case}__freqs"].shape == (n,)
            assert z[f"{case}__S_inst"].shape == (2, n) and z[f"{case}__S_fg"].shape == (2, n)
            assert np.all(z[f"{case}__S_inst"] > 0) and np.all(z[f"{case}__S_fg"] > 0)
