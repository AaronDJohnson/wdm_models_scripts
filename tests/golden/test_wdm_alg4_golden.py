"""B6 (G7, lineage D): Paper I Alg 4 oracle values and the June literal Alg 1/2 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.golden.paths import ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden" / "wdm"
GRIDS = {"g8x16dt1p0": (8, 16, 1.0), "g16x32dt0p5": (16, 32, 0.5)}
PSDS = ("flat", "linear", "steep", "quadratic", "random")
EXACT_ORACLES = ("alg4", "B", "exact", "offdiag", "new_section")
LEAK = 0.26727914306174855            # spec 2.2 test_alg2_leak, June literal Alg 2


@pytest.fixture(scope="module")
def meta():
    return json.loads((GOLDEN / "alg4_reference.json").read_text())


@requires_legacy
def test_sources_and_conventions(meta):
    assert meta["schema"] == 1 and meta["lineage"] == "D"
    assert Path(meta["wdm_file"]).resolve() == (ROOT / "wdm" / "__init__.py").resolve()
    assert meta["phi_centered_signature"].startswith("phi_centered(Nt, Nf, dt")
    assert set(meta["sources_sha256"]) == {
        "wdm/paper_algorithms.py", "wdm/stationary.py", "wdm/filter.py", "wdm/transform.py",
        "wdm_paper_1/notes/psd_backward_offdiag.py", "wdm_paper_1/notes/inverse_psd_new_section.py"}


@pytest.mark.parametrize("g", sorted(GRIDS))
@pytest.mark.parametrize("psd", PSDS)
def test_alg4_oracles_exact_and_inputs_recorded(meta, g, psd):
    Nf, Nt, dt = GRIDS[g]
    N = Nf * Nt
    key = f"{g}__{psd}"
    with np.load(GOLDEN / "alg4_reference.npz") as z:
        S_in = z[f"{key}__S_in"]
        assert S_in.shape == (N // 2 + 1,)
        if psd == "random":
            assert np.array_equal(S_in, np.random.default_rng(0).uniform(0.5, 5.0, N // 2 + 1))
        if psd == "steep":
            assert np.array_equal(S_in, 1.0 / (np.abs(np.fft.rfftfreq(N, dt)) + 0.02))
        for oracle in EXACT_ORACLES + ("diag_only",):
            assert z[f"{key}__S_{oracle}"].shape == (N // 2 + 1,)
        assert z[f"{g}__phi_june"].shape == (Nt,)
        if g == "g8x16dt1p0":
            assert z[f"{key}__lam_diag_00"].shape == (Nt // 2, Nt // 2)
            assert z[f"{key}__lam_offd_00"].shape == (Nt // 2, Nt)
            assert z[f"{key}__lam_diag_01"].shape == (Nt, Nt)
    for oracle in EXACT_ORACLES:
        assert meta["relerr_interior"][key][oracle] <= 1e-14, oracle
    assert meta["lambda_literal_vs_fast"][key]["diag"] <= 1e-15
    assert meta["lambda_literal_vs_fast"][key]["offd"] <= 1e-15
    if psd == "flat":
        assert meta["relerr_interior"][key]["diag_only"] <= 1e-14
    else:                                   # why lambda_to_psd (June, diagonal-only) was retired
        assert meta["relerr_interior"][key]["diag_only"] > 1e-2


def test_alg12_literal_outputs():
    with np.load(GOLDEN / "paper_june_alg12.npz") as z:
        rng = np.random.default_rng(0)
        for g, (Nf, Nt, dt) in {"g8x16dt1p0": (8, 16, 1.0), "g16x32dt0p5": (16, 32, 0.5),
                                "g8x32dt1p7": (8, 32, 1.7), "g32x64dt1p0": (32, 64, 1.0),
                                "g4x8dt1p0": (4, 8, 1.0)}.items():
            x = rng.standard_normal(Nf * Nt)
            assert np.array_equal(z[f"{g}__x"], x)
            assert z[f"{g}__w_alg1"].shape == (Nf + 1, Nt)
            assert np.max(np.abs(z[f"{g}__x_alg2"] - x)) <= 1e-12
            assert np.max(np.abs(z[f"{g}__w_alg1"] - z[f"{g}__w_package"])) <= 1e-12
        assert np.max(np.abs(z["leak__x_alg2"])) == LEAK
