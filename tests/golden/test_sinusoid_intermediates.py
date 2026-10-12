"""G6: sinusoid per-module intermediates at the manuscript_mid truth (spec §7.2 G6; Task A9)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.golden import sinusoid_goldens as SG
from scripts.golden.goldens_io import read_json, read_npz

pytestmark = pytest.mark.golden
WDM_MODELS = tuple(m for m in SG.MODELS if not m.startswith("whittle") and m != "exact")


@pytest.fixture(scope="module")
def g6():
    return (read_json(SG.INTERMEDIATES / "manuscript_mid_truth.json"),
            read_npz(SG.INTERMEDIATES / "manuscript_mid_truth.npz"))


def test_variances(g6):
    rep, arr = g6
    for base in ("band", "mid"):
        v = arr[f"{base}_variance_raw"]
        assert v.shape == (16384,) and np.all(v > 0)
        assert rep["variances"][f"{base}_raw"]["shape"] == [16384]
        assert rep["variances"][f"{base}_pad"]["shape"] == [17408]


def test_tri_blocks(g6):
    rep, _ = g6
    assert set(rep["tri_blocks"]) == {"tri_raw", "tri_cut", "tri_pad", "tri_exact"}
    for model, b in rep["tri_blocks"].items():
        assert (b["n_diag"], b["n_up"], len(b["samples"])) == (65, 64, 50), model
        assert all(f > 0 for f in b["diag_fro"]), model


def test_exact_rows(g6):
    rep, arr = g6
    assert rep["exact_rows"]["shape"] == [959, 16384]
    assert (rep["exact_rows"]["n_edge"], rep["exact_rows"]["n_interior"]) == (959, 15425)
    assert arr["exact_rows_pick"].tolist() == [0, 106, 213, 319, 426, 532, 639, 745, 852, 958]
    assert arr["exact_rows_sample"].shape == (10, 16384)
    assert arr["exact_rows_l2"].shape == (959,)
    assert np.allclose(np.linalg.norm(arr["exact_rows_sample"], axis=1),
                       arr["exact_rows_l2"][arr["exact_rows_pick"]], rtol=1e-12, atol=0.0)


def test_maps(g6):
    rep, _ = g6
    for model, n in {"band_raw": 16384, "band_cut": 15681, "band_exact": 16384, "band_pad": 16384}.items():
        assert rep["maps"][model]["n_cells"] == n, model
        assert rep["maps"][model]["u"]["shape"] == [n, 128], model
    assert rep["maps"]["Wp"]["shape"] == [17408, 1024]
    assert rep["maps"]["band_pad"]["log_jac"] == 0.0
    assert rep["maps"]["band_cut"]["log_jac"] == rep["maps"]["band_raw"]["log_jac"] == 0.5 * 16384 * np.log(2.0 ** -10)
    # G5 edge constants (spec §5.1 CUT = (6, 5), EDGE_EXACT = (8, 7), PAD_COLUMNS = 16; §4.2 crop)
    e = rep["edges"]
    assert (e["cut"], e["edge_exact"], e["pad_columns"]) == ([6, 5], [8, 7], 16)
    assert e["cut_columns"] == list(range(6)) + list(range(251, 256))
    assert e["edge_columns"] == list(range(8)) + list(range(249, 256))
    assert (rep["parts"]["whittle_cut"]["crop"], rep["parts"]["whittle_cut"]["Nc"]) == ([384, 16064], 15680)


@pytest.mark.parametrize("model", SG.MODELS)
def test_parts_recombine_to_the_G1_truth(g6, model):
    parts = g6[0]["parts"][model]
    assert parts["lnL"] == parts["lnlike"]
    truth = read_json(SG.REFS / "INDEX.json")["files"][f"manuscript_mid_{model}_reference.npz"]["truth_lnlike"]
    # exact: PCG oracle self-noise 4e-9 (spec §3.6), so its spec floor 1e-7; every other model 1e-9
    assert abs(parts["lnL"] - truth) <= (1e-7 if model == "exact" else 1e-9)


def test_marginal_terms(g6):
    for model in WDM_MODELS:
        p = g6[0]["parts"][model]
        assert p["n_u"] == 128 and p["n_eff"] == p["n_cells"] - 128, model
        assert p["quad"] == p["quad_C"] - p["quad_H"], model
        assert p["logdet"] == p["logdet_C"] + p["logdet_H"], model


def test_exact_model_assets(g6):
    p = g6[0]["parts"]["exact"]
    assert p["c_N"] == 0.22629082684418084
    assert (p["bandcov_key"], p["bandcov_file"]) == ("4bfb466c191c79f9879a", "bandcov_v1_4bfb466c191c79f9879a.npz")
    assert p["n_keep"] == 16256
    assert p["bandcov_identity"]["bounds"] == [[-4.0, 0.0], [-2.0, 2.0]]
