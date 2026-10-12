"""Shipped sinusoid inputs (spec §1 'Shipped data', §7.2 G5; Task A4)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.goldens_io import read_json, sha256_array, sha256_file
from scripts.golden.paths import DATA, ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
EXPECTED = {
    "sinusoid/manuscript_low/toy_data.npz": "650cf032a6e30fc1ad3bd40248ea943e21a70f8243a27ff1ef40703f86d530e8",
    "sinusoid/manuscript_mid/toy_data.npz": "02cdb2595ce1058a063b131abbd1588070892bb0f18260f4cf545936785d222d",
    "sinusoid/manuscript_high/toy_data.npz": "a6db39e500be548d2d6453d04f1f01d3b891d815ba8c90ce2de2958d972bc05a",
    "sinusoid/cache/cN_N16384_dt0.0009765625_fmin4_fref1.npy": "d263bf25960ab6f4e371fd597da0e28123cba569ca085e2af5e6c3599257213f",
    "sinusoid/cache/bandcov_v1_4bfb466c191c79f9879a.npz": "752b3ea46cf14feaf5b8845ad39999a0b596a956d6af9e58072f691a8cd6c2c2",
}
NOISE_SHA256 = "1fc614025fe15440839e8e75512d6b6d42273c4bb7c16067177e5440ce55da23"
INJECTION = {"manuscript_low": (33.0, 1.6322595829099906),
             "manuscript_mid": (137.0, 0.30421576834679953),
             "manuscript_high": (385.0, 0.08729868783526586)}


def test_data_manifest_lists_the_sinusoid_files():
    m = read_json(DATA / "MANIFEST.json")
    assert m["schema"] == 1
    assert all(isinstance(v, str) for v in m["files"].values())     # {relpath: sha256} (shared schema)
    for rel, sha in EXPECTED.items():
        assert m["files"][rel] == sha
        prov = m["provenance"][rel]
        assert prov["sha256"] == sha
        assert prov["bytes"] == (DATA / rel).stat().st_size
        assert prov["bundle_member"].startswith("sinusoid_matrix_colab/")


@pytest.mark.parametrize("rel", sorted(EXPECTED))
def test_shipped_bytes(rel):
    assert sha256_file(DATA / rel) == EXPECTED[rel]


@pytest.mark.parametrize("case", CASES)
def test_toy_data_layout(case):
    with np.load(DATA / "sinusoid" / case / "toy_data.npz", allow_pickle=False) as z:
        assert {"d", "h", "n", "w_d", "snr", "seed", "noise_source", "_cfg_keys", "_cfg_vals"} <= set(z.files)
        assert z["d"].shape == z["h"].shape == z["n"].shape == (16384,)
        assert z["w_d"].shape == (65, 256)
        assert int(z["seed"]) == 1
        assert np.array_equal(z["d"], z["h"] + z["n"])
        assert sha256_array(z["n"]) == NOISE_SHA256
        cfg = dict(zip((str(k) for k in z["_cfg_keys"]), z["_cfg_vals"].tolist()))
    assert (cfg["f_0"], cfg["A_h"]) == INJECTION[case]
    assert (cfg["N"], cfg["dt"], cfg["Nf"], cfg["Nt"]) == (16384.0, 2.0 ** -10, 64.0, 256.0)
    assert (cfg["A_n"], cfg["alpha_c"], cfg["alpha_d"], cfg["f_ref"], cfg["f_min"], cfg["phi_0"]) == (
        400.0, -2.0, 1.2, 1.0, 4.0, 0.6)


def test_cN_value():
    assert float(np.load(DATA / "sinusoid/cache/cN_N16384_dt0.0009765625_fmin4_fref1.npy")) == 0.22629082684418084


@requires_legacy
def test_shipped_files_equal_their_canonical_sources():
    prov = read_json(DATA / "MANIFEST.json")["provenance"]
    for rel, sha in EXPECTED.items():
        assert sha256_file(ROOT / prov[rel]["from"]) == sha


@requires_legacy
@pytest.mark.legacy
def test_vendored_forward_time_reproduces_w_d_bitwise():
    from scripts.golden.lineage import run_worker
    out = run_worker("A", "sinusoid_wt", "wd-check", "--data-root", str(DATA / "sinusoid"))
    rep = read_json(out / "wd_check.json")
    assert rep["wdm_file"] == str(ROOT / "toy_models/vendor/wdm/src/wdm/__init__.py")
    for case in CASES:
        got = rep["cases"][case]
        assert (got["array_equal"], got["max_abs"], got["shape"]) == (True, 0.0, [65, 256])
