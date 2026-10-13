"""E-B4 (S3 gate, G5): the sinusoid data products through the model layer (spec section 7.2
G5, section 5.1): the shipped bytes, ``w_d`` bitwise from ``wdm_models.sinusoid.data.load``,
the 0.1% wrap rule (6, 5), c_N, the bandcov file name and identity, the prior box that keys
it, and the cell counts ``describe()`` reports (16384 / 15681 / 959 / 17408).  Each check is
recorded as an ``s3`` property (observed |delta| or mismatch count, gate 0)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import wdm
from domainia.assets import identity_key
from domainia.domains import CellSet
from scripts.golden.goldens_io import read_json, sha256_file
from scripts.golden.paths import DATA
from scripts.golden.s3_cases import G5_ITEMS, record
from wdm.evolutionary import wrap_cut_columns, wrap_fractions
from wdm_models import registry
from wdm_models.sinusoid import config, data, exact, priors

pytestmark = pytest.mark.golden
CN = "sinusoid/cache/cN_N16384_dt0.0009765625_fmin4_fref1.npy"
BANDCOV = "sinusoid/cache/bandcov_v1_4bfb466c191c79f9879a.npz"
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
#: Every item this module records (``check`` refuses any other name).
ITEMS = frozenset(
    [f"sinusoid/{c}/g5/{k}" for c in CASES for k in ("toy_data_sha256", "w_d")]
    + [f"sinusoid/g5/{k}" for k in ("cN_sha256", "bandcov_sha256", "wrap_cut", "wrap_first_dropped_lead",
                                    "wrap_first_dropped_trail", "wrap_worst_kept", "c_N", "bandcov_key",
                                    "bandcov_identity", "noise_bounds")]
    + [f"sinusoid/g5/describe/{k}" for k in ("raw_cells", "cut_cells", "exact_rows", "pad_cells")])


def check(record_property, item: str, observed: float) -> None:
    assert item in ITEMS, item
    record(record_property, item, observed, 0.0)
    assert observed == 0.0, (item, observed)


@pytest.mark.parametrize("case", CASES)
def test_toy_data_and_w_d_bitwise(case, record_property):
    manifest = read_json(DATA / "MANIFEST.json")["files"]
    d = data.load(case)                                   # sha256 + w_d check inside
    rel = f"sinusoid/{case}/toy_data.npz"
    check(record_property, f"sinusoid/{case}/g5/toy_data_sha256", float(sha256_file(d.path) != manifest[rel]))
    cfg = config.CASES[case]
    w = wdm.forward_time(d.d, cfg.Nf, cfg.Nt, cfg.dt)
    assert w.shape == d.w_d.shape == (65, 256)
    check(record_property, f"sinusoid/{case}/g5/w_d", float(np.max(np.abs(w - d.w_d))))
    assert np.array_equal(w, d.w_d)


def test_shipped_caches_are_the_manifest_bytes(record_property):
    manifest = read_json(DATA / "MANIFEST.json")["files"]
    for rel, item in ((CN, "sinusoid/g5/cN_sha256"), (BANDCOV, "sinusoid/g5/bandcov_sha256")):
        check(record_property, item, float(sha256_file(DATA / rel) != manifest[rel]))


def test_wrap_rule_is_the_configured_cut(record_property):
    """``wrap_cut_columns(64, 256, 2**-10) == (6, 5) == config.CUT``: the first dropped columns
    wrap 0.11200% (n = 5) and 0.11708% (n = 251) of their energy, every kept one <= 0.1%."""
    cut = wrap_cut_columns(64, 256, 2.0 ** -10)
    check(record_property, "sinusoid/g5/wrap_cut", float(cut != (6, 5)) + float(tuple(config.CUT) != (6, 5)))
    worst = np.nanmax(wrap_fractions(64, 256, 2.0 ** -10), axis=0)
    check(record_property, "sinusoid/g5/wrap_first_dropped_lead", float(round(100 * worst[5], 5) != 0.11200))
    check(record_property, "sinusoid/g5/wrap_first_dropped_trail", float(round(100 * worst[251], 5) != 0.11708))
    check(record_property, "sinusoid/g5/wrap_worst_kept", float(not np.nanmax(worst[6:251]) <= 1e-3))


def test_c_N_and_the_bandcov_artifact(record_property):
    check(record_property, "sinusoid/g5/c_N", abs(exact.ExactLogDet.load().c_N - 0.22629082684418084))
    corr = exact.BandCorrection.load()
    ident = exact.bandcov_identity()
    check(record_property, "sinusoid/g5/bandcov_key",
          float(identity_key(ident) != "4bfb466c191c79f9879a")
          + float(corr.path.name != "bandcov_v1_4bfb466c191c79f9879a.npz"))
    with np.load(DATA / BANDCOV, allow_pickle=False) as z:
        stored = json.loads(str(z["metadata"]))["identity"]
    check(record_property, "sinusoid/g5/bandcov_identity",
          float(stored != json.loads(json.dumps(ident))) + float(corr.metadata["identity"] != stored))
    check(record_property, "sinusoid/g5/noise_bounds",
          float(not np.array_equal(priors.noise_bounds(), np.asarray([[-4.0, 0.0], [-2.0, 2.0]]))))


def test_describe_counts(record_property):
    """``registry.describe`` (spec section 5.3), cross-checked against ``CellSet`` counts."""
    cells = CellSet.all_active(wdm.Grid(64, 256, 2.0 ** -10))
    ext = CellSet.all_active(wdm.Grid(64, 256 + config.PAD_COLUMNS, 2.0 ** -10))
    got = {"raw_cells": registry.describe("sinusoid", "manuscript_mid", "band_raw")["P"],
           "cut_cells": registry.describe("sinusoid", "manuscript_mid", "tri_cut")["P"],
           "exact_rows": registry.describe("sinusoid", "manuscript_mid", "mid_exact")["exact_rows"],
           "pad_cells": registry.describe("sinusoid", "manuscript_mid", "band_pad")["lambda_cells"]}
    want = {"raw_cells": 16384, "cut_cells": 15681, "exact_rows": 959, "pad_cells": 17408}
    assert (cells.size, cells.drop_columns(*config.CUT).size, cells.columns(*config.EDGE_EXACT).size,
            ext.size) == tuple(want.values())
    for key, value in want.items():
        check(record_property, f"sinusoid/g5/describe/{key}", float(got[key] != value))


def test_g5_items_are_the_report_inventory():
    """scripts/golden/s3_report.py counts exactly these items as G5 (uncovered ones fail it)."""
    assert len(ITEMS) == len(G5_ITEMS) == 20 and set(G5_ITEMS) == ITEMS
