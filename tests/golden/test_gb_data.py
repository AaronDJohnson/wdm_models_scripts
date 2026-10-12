"""B1: the shipped galactic-binary inputs and their data/MANIFEST.json entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.golden.paths import ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")

REL = Path(__file__).resolve().parents[2]
DATA = REL / "data"
CASES = {"gb_3mo": 3, "gb_6mo": 6, "gb_12mo": 12}
EXPECTED = sorted(
    [f"galactic_binary/prepared/gb_{m}mo.npz" for m in (3, 6, 12)]
    + [f"galactic_binary/prepared/gb_{m}mo_report.json" for m in (3, 6, 12)]
    + [f"galactic_binary/segments/segment_snr7_{m}mo_3bands.npz" for m in (3, 6, 12)]
    + [f"galactic_binary/segments/store_snr7_{m}mo_3bands_pm3.npz" for m in (3, 6, 12)]
    + [f"galactic_binary/foreground/foreground_{m}mo_snr7.{e}" for m in (3, 6, 12)
       for e in ("json", "npz")])
PREPARED_KEYS = {"case", "T", "N", "DT", "df", "bin_offset", "n_bins", "sigma0", "S_ref", "z",
                 "resid", "injected", "sI", "sG", "basis_L", "inject_glass", "prior_lo",
                 "prior_hi", "mu_fit", "mu_freq", "NT", "first_band", "n_bands", "freq_lo",
                 "freq_n", "envelope_period", "template_bw", "fg_params", "store_offset",
                 "store_n", "store_z", "store_resid", "store_injected", "store_sI", "store_sG",
                 "pad_g0", "pad_G", "pad_sI", "pad_sG", "rho_pad", "pad_store_g0", "pad_store_G",
                 "pad_store_sI", "pad_store_sG", "crop_k0", "crop_n", "crop_sI", "crop_sG",
                 "matrix_cut", "rho_win"}
#: (NT, window bins, nominal bins, store bins, matrix_cut) per case.
COUNTS = {"gb_3mo": (240, 419, 360, 1139, (5, 4)), "gb_6mo": (438, 767, 657, 2081, (5, 4)),
          "gb_12mo": (900, 1575, 1350, 4275, (0, 0))}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads((DATA / "MANIFEST.json").read_text())


def test_every_file_shipped_and_hashed():
    m = _manifest()
    assert m["schema"] == 1
    assert all(isinstance(v, str) for v in m["files"].values())     # {relpath: sha256} (A4 schema)
    gb = sorted(k for k in m["files"] if k.startswith("galactic_binary/"))
    assert gb == EXPECTED
    for rel in EXPECTED:
        prov = m["provenance"][rel]
        assert _sha(DATA / rel) == m["files"][rel] == prov["sha256"], rel
        assert (DATA / rel).stat().st_size == prov["bytes"], rel
    for months in (3, 6, 12):
        prov = m["provenance"][f"galactic_binary/foreground/foreground_{months}mo_snr7.npz"]
        assert prov["identical_to"].endswith(f"foreground_fit/foreground_{months}mo.npz")


@requires_legacy
def test_sources_are_identical_frozen_files():
    prov = _manifest()["provenance"]
    for rel in EXPECTED:
        assert _sha(ROOT / prov[rel]["from"]) == prov[rel]["sha256"], prov[rel]["from"]
    for months in (3, 6, 12):
        p = prov[f"galactic_binary/foreground/foreground_{months}mo_snr7.npz"]
        assert _sha(ROOT / p["identical_to"]) == p["sha256"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_prepared_schema_and_counts(case):
    with np.load(DATA / f"galactic_binary/prepared/{case}.npz", allow_pickle=False) as z:
        assert PREPARED_KEYS <= set(z.files)
        NT, n_bins, freq_n, store_n, cut = COUNTS[case]
        assert (int(z["NT"]), int(z["n_bins"]), int(z["freq_n"]), int(z["store_n"])) == (
            NT, n_bins, freq_n, store_n)
        assert tuple(int(v) for v in z["matrix_cut"]) == cut
        assert z["z"].shape == (2, n_bins) and z["store_z"].shape == (2, store_n)
        assert z["basis_L"].shape == (5, 65536)


def test_data_size_budget():
    total = sum((DATA / rel).stat().st_size for rel in EXPECTED)
    assert 10 * 2**20 < total < 20 * 2**20
