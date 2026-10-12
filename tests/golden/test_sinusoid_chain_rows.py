"""G2: sinusoid chain rows (spec §7.2 G2; Task A8)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.golden import sinusoid_goldens as SG
from scripts.golden.goldens_io import read_json, read_npz, sha256_file
from scripts.golden.paths import legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")


def _index() -> dict:
    return read_json(SG.CHAIN_ROWS / "INDEX.json")


def test_chain_row_index_is_deterministic_and_bracketed():
    idx = SG.chain_row_index(25000)
    assert idx.shape == (20,) and idx[0] == 0 and idx[-1] == 24999
    assert np.all(np.diff(idx) > 0)
    assert np.array_equal(idx, SG.chain_row_index(25000))


def test_one_file_per_target():
    names = {f"{c}_{m}.npz" for c in SG.CASES for m in SG.MODELS}
    assert {p.name for p in SG.CHAIN_ROWS.glob("*.npz")} == names
    assert set(_index()["runs"]) == {n[:-4] for n in names}


@pytest.mark.parametrize("case", SG.CASES)
@pytest.mark.parametrize("model", SG.MODELS)
def test_rows(case, model):
    entry = _index()["runs"][f"{case}_{model}"]
    rows = read_npz(SG.CHAIN_ROWS / f"{case}_{model}.npz")
    ndim = 5 if model.startswith("whittle") else 6
    assert entry["ndim"] == ndim and len(entry["parameter_names"]) == ndim
    assert np.array_equal(rows["row_index"], SG.chain_row_index(entry["nrows"]))
    assert rows["theta"].shape == (20, ndim)
    assert np.all(np.isfinite(rows["lnlike"])) and np.all(rows["temperature"] == 1.0)
    assert np.array_equal(rows["lnlike"], rows["lnprob"])
    assert np.all(np.abs(rows["theta"][:, 2]) <= np.pi)


@requires_legacy
def test_rows_are_the_stored_chain_rows():
    for key, entry in _index()["runs"].items():
        path = SG.SINUSOID_FINAL / entry["run"] / "chain_0.bin"
        assert sha256_file(path) == entry["chain_0_sha256"], key
        rows = read_npz(SG.CHAIN_ROWS / f"{key}.npz")
        chain = np.memmap(path, dtype="<f8", mode="r", shape=(entry["nrows"], entry["ndim"] + 4))
        stored = np.asarray(chain[rows["row_index"]])
        assert np.array_equal(stored[:, :entry["ndim"]], rows["theta"]), key
        assert np.array_equal(stored[:, entry["ndim"]], rows["lnlike"]), key


@pytest.mark.parametrize("lineage", ["A", "B"])
def test_fast_chain_row_reports(lineage):
    rep = read_json(SG.EVAL / f"{lineage}_chain_rows_fast.json")
    assert (rep["lineage"], rep["tier"], rep["kind"]) == (lineage, "fast", "chain_rows")
    assert set(rep["models"]) == set(SG.MODELS)
    for model, m in rep["models"].items():
        assert (m["n_rows"], m["inf_mismatch"]) == (9, 0), model
        assert m["max_abs"] <= (1e-4 if model == "exact" else 1e-5), model
