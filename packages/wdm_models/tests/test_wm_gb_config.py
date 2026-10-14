"""F-A2: the galactic-binary configuration is the legacy paper subset, verbatim (spec §5.2
config.py), pinned against golden/galactic_binary/products.json ``constants`` (lineage C)."""

from __future__ import annotations

import math

import pytest

from wdm_models.common import params
from wdm_models.galactic_binary import config as C

#: products.json ``constants`` key -> the release name, where they differ.
SAME = ("ANALYSIS_NT", "C_F", "C_T", "DT", "ENVELOPE_PERIOD", "ENV_NAMES", "GHAT_L", "GLASS_PARAMS",
        "MU_BOX", "NOISE_MOVE_FRACTION", "N_BANDS", "N_HARM", "PAD_COLUMNS", "PAD_EDGE_TOL",
        "TEMPLATE_BW")


def _plain(v):
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


@pytest.mark.golden
def test_constants_are_the_legacy_values(gb_products):
    legacy = gb_products["constants"]
    for name in SAME:
        assert _plain(getattr(C, name)) == legacy[name], name
    assert _plain(C.CUT) == legacy["CUT"]
    assert _plain(C.PERIODIC) == legacy["PERIODIC"]
    assert legacy["gbonfly_version"] == "0.1.0"          # the generator the chains used


def test_the_paper_subset():
    assert C.CASES == ("gb_3mo", "gb_6mo", "gb_12mo")
    assert dict(C.ANALYSIS_NT) == {"gb_3mo": 240, "gb_6mo": 438, "gb_12mo": 900}
    assert dict(C.CUT) == {"gb_3mo": (5, 4), "gb_6mo": (5, 4), "gb_12mo": (0, 0)}
    assert (C.PAD_COLUMNS, C.PAD_EDGE_TOL, C.DT, C.N_HARM, C.GHAT_L, C.N_BANDS) == (16, 1e-8, 5.0, 2, 65536, 3)
    assert C.ENVELOPE_PERIOD == 31536000.0 and C.NT_BAND == (0.98e-3, 1.02e-3)
    assert (C.TEMPLATE_BW, C.MU_BOX, C.C_F, C.C_T, C.NOISE_MOVE_FRACTION) == (4096, (-8.0, 8.0), 0.208, 0.639, 0.1)
    assert dict(C.PERIODIC) == {2: (0.0, 2 * math.pi), 5: (0.0, math.pi), 6: (0.0, 2 * math.pi)}
    with pytest.raises(TypeError):
        C.CUT["gb_3mo"] = (0, 0)                          # read-only tables


def test_parameter_names_follow_the_chains():
    assert C.parameter_names("whittle_cut") == C.GLASS_PARAMS + ("log_gbar_A", "log_gbar_E")
    names = C.parameter_names("tri_exact")
    assert len(names) == 18 and names[8:13] == tuple(f"{n}_A" for n in C.ENV_NAMES)
    assert names[13:] == tuple(f"{n}_E" for n in C.ENV_NAMES)
    assert [m for m in params.MODELS if C.is_stationary(m)] == ["whittle_raw", "whittle_cut"]
    with pytest.raises(ValueError, match="unknown model 'band_win'"):
        C.parameter_names("band_win")
    with pytest.raises(KeyError, match="unknown galactic-binary case 'gb_9mo'"):
        C.check_case("gb_9mo")


@pytest.mark.golden
def test_spec_literals_are_the_golden_products(gb_products):
    """``TEMPLATE_SUMS`` (spec §5.2, 10 decimals) and ``SNR_FULL`` are products.json's."""
    for case in C.CASES:
        t = gb_products["cases"][case]["template"]
        for win in ("window", "store"):
            assert C.TEMPLATE_SUMS[case][win] == (round(t[win]["sum_h2"], 10), round(t[win]["re_zh"], 10))
        snr = gb_products["cases"][case]["snr_full"]
        assert C.SNR_FULL[case] == snr["recomputed"] == snr["report"]
