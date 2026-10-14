"""F-A6: ``PreparedCase`` (spec §5.2 prepared.py): the paper keys of the shipped prepared
files, sha256-checked and read-only; the 12-month block factor; the whittle_cut guard."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from domainia.assets import ArtifactError
from wdm_models.galactic_binary import config as C
from wdm_models.galactic_binary import prepared as P

DATA = Path(__file__).resolve().parents[3] / "data"
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
#: Spec §4.9's per-channel counts: window bins, nominal bins, store bins, bands.
SHAPES = {"gb_3mo": (419, 360, 1139, (65, 66, 67)), "gb_6mo": (767, 657, 2081, (71, 72, 73)),
          "gb_12mo": (1575, 1350, 4275, (69, 70, 71))}
#: Keys of the shipped files that only the dropped models read.
LEGACY_ONLY = {"bands", "g0", "G", "phi", "sI_grid", "sG_grid", "rho_win", "matrix_store"}


def test_the_paper_schema():
    assert len(P.BASE_KEYS) == 32 and len(P.MATRIX_KEYS) == 21 and len(P.PAPER_KEYS) == 53
    assert P.PAPER_KEYS == P.BASE_KEYS + P.MATRIX_KEYS and len(set(P.PAPER_KEYS)) == 53
    assert set(P.FIT_KEYS) <= set(P.BASE_KEYS) and not LEGACY_ONLY & set(P.PAPER_KEYS)
    assert tuple(P.PREPARED_SHA256) == C.CASES


@pytest.mark.golden
def test_shipped_files_hold_the_paper_keys_and_the_legacy_extras():
    for case in CASES:
        with np.load(DATA / P.prepared_rel(case), allow_pickle=False) as z:
            assert set(z.files) == set(P.PAPER_KEYS) | LEGACY_ONLY


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_load_is_sha_checked_and_restricted_to_the_paper_keys(case, gb_products):
    prep = P.PreparedCase.load(case)
    n_win, n_nom, n_store, bands = SHAPES[case]
    want = gb_products["cases"][case]
    assert prep.case == case and tuple(prep.arrays) == P.PAPER_KEYS
    assert (prep.n_bins, prep.freq.stop - prep.freq.start, prep.store_n, prep.bands) == (n_win, n_nom, n_store, bands)
    assert (prep.T, prep.N, prep.NT, prep.DT, prep.df) == (want["T"], want["N"], want["NT"], 5.0, want["df"])
    assert [prep.freq.start, prep.freq.stop] == want["freq"] and [prep.store_offset, prep.store_n] == want["store"]
    assert prep.matrix_cut == tuple(want["matrix_cut"]) == tuple(C.CUT[case])
    assert prep.periodic is want["periodic"] and prep.exact_block == want["exact_block"]
    assert np.array_equal(prep.sigma0, want["sigma0"]) and np.array_equal(prep.mu_fit, want["mu_fit"])
    assert np.array_equal(prep.inject, want["inject"]["inject_glass"])
    assert prep.z.shape == (2, n_win) and prep.store_z.shape == (2, n_store) and prep.basis_L.shape == (5, 65536)
    with pytest.raises(ValueError):
        prep.z[0, 0] = 0.0                                              # read-only
    assert prep.anchor("whittle_raw").shape == (10,) and prep.anchor("exact").shape == (18,)
    assert np.array_equal(prep.anchor("exact")[8:], prep.mu_fit.ravel())
    assert prep.bounds("band_raw").shape == (18, 2)
    nom = prep.nominal_domain()
    assert (nom.k0, nom.n, nom.units) == (prep.bin_offset + prep.freq.start, n_nom, "sigma0")
    assert prep.store_domain().n == n_store and prep.window_domain().k0 == prep.bin_offset
    env = prep.envelope()
    assert env.T == prep.T and np.array_equal(env.basis_L, prep.basis_L)


@pytest.mark.golden
def test_twelve_months_is_periodic_with_blocks_of_half_a_band():
    p12 = P.PreparedCase.load("gb_12mo")
    assert p12.periodic and p12.exact_block == 450 and (p12.freq.stop - p12.freq.start) % 450 == 0
    assert not P.PreparedCase.load("gb_6mo").periodic


@pytest.mark.golden
def test_a_tampered_or_foreign_file_is_refused(tmp_path):
    src = DATA / P.prepared_rel("gb_3mo")
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["matrix_cut"] = np.array([4, 4])
    bad = tmp_path / "gb_3mo.npz"
    np.savez(bad, **arrays)
    prep = P.PreparedCase(bad)                                          # any file with the schema
    with pytest.raises(ValueError, match=r"matrix_cut \(4, 4\) is not CUT\['gb_3mo'\] = \(5, 4\)"):
        prep.require_cut()
    assert P.PreparedCase.load("gb_3mo").require_cut() == (5, 4)
    with pytest.raises(ArtifactError, match="sha256"):
        P.PreparedCase(bad, sha256=P.PREPARED_SHA256["gb_3mo"])
    root = tmp_path / "data"
    (root / "galactic_binary" / "prepared").mkdir(parents=True)
    shutil.copy(DATA / "MANIFEST.json", root / "MANIFEST.json")
    shutil.copy(bad, root / P.prepared_rel("gb_3mo"))
    with pytest.raises(ArtifactError, match="sha256"):
        P.PreparedCase.load("gb_3mo", data_root=root)
    del arrays["crop_sI"]
    np.savez(tmp_path / "short.npz", **arrays)
    with pytest.raises(KeyError, match="missing paper keys \\['crop_sI'\\]"):
        P.PreparedCase(tmp_path / "short.npz")
