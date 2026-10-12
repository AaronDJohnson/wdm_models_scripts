"""Each frozen lineage runs isolated in its own subprocess (spec §7.1; Task A3)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from scripts.golden import lineage as L
from scripts.golden.paths import REL, ROOT, WORKERS, legacy_available, tmp_root
from tools import frozen_tree_hashes as fth

requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
VENDORED_WDM = ROOT / "toy_models" / "vendor" / "wdm" / "src" / "wdm" / "__init__.py"


def _report(out_dir: Path) -> dict:
    return json.loads((out_dir / "selftest.json").read_text())


def test_lineage_table_is_the_specified_one():
    a = [str(ROOT / "toy_models/vendor/wdm/src"), str(ROOT / "wdm_paper_2/toy_models/src")]
    assert set(L.LINEAGES) == {"A", "B", "C", "D", "R"}
    assert L.LINEAGES["A"] == a
    assert L.LINEAGES["B"] == a + [str(ROOT / "wdm_paper_2/toy_models/gpu_sinusoid")]
    assert L.LINEAGES["C"] == [str(ROOT / p) for p in (
        "toy_models/galactic_binary", "toy_models/galactic_binary/gpu", "wdm_paper_2/gbonfly/src",
        "wdm_paper_2/toy_models/package/src", "wdm_paper_2/toy_models/src")]
    assert L.LINEAGES["D"] == [str(ROOT), str(ROOT / "wdm_paper_1/notes")]
    assert L.LINEAGES["R"] == [str(REL / p) for p in (
        "packages/wdm/src", "packages/domainia/src", "packages/wdm_models/src")]
    assert all(Path(p).is_absolute() for paths in L.LINEAGES.values() for p in paths)
    assert L.EXPECTED_WDM["C"] is None


def test_worker_env_replaces_rather_than_inherits(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/must/not/leak")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("WDM_GOLDEN_EXPECT_WDM", "/stale")
    # Inherited variables that worker_env does NOT set again: only the prefix filter removes them.
    leaked = {"XLA_FLAGS": "--xla_cpu_enable_fast_math=true", "JAX_DISABLE_JIT": "1",
              "PYTHONWARNINGS": "ignore", "PYTHONHASHSEED": "0", "WDM_GOLDEN_STALE": "1"}
    for key, value in leaked.items():
        monkeypatch.setenv(key, value)
    env = L.worker_env("A")
    assert not set(leaked) & set(env)
    assert env["PYTHONPATH"].split(os.pathsep) == L.LINEAGES["A"]
    assert (env["JAX_PLATFORMS"], env["JAX_ENABLE_X64"]) == ("cpu", "true")
    assert (env["PYTHONDONTWRITEBYTECODE"], env["PYTHONSAFEPATH"], env["PYTHONNOUSERSITE"]) == ("1", "1", "1")
    assert env["PYTHONPYCACHEPREFIX"] == str(Path(tempfile.gettempdir()).resolve() / "wdmrel_pyc")
    assert env["WDM_GOLDEN_EXPECT_WDM"] == L.EXPECTED_WDM["A"]
    with pytest.raises(KeyError):
        L.worker_env("Z")


@requires_legacy
@pytest.mark.legacy
def test_lineage_A_subprocess_is_isolated():
    out = L.run_worker("A", "selftest_env", "--import", "wdm", "--import", "wdm_toys.matrix")
    rep = _report(out)
    assert out.is_relative_to(tmp_root())
    assert rep["cwd"] == str(tmp_root())
    assert rep["sys_path"][:2] == L.LINEAGES["A"]
    assert str(WORKERS) not in rep["sys_path"] and str(REL) not in rep["sys_path"]
    assert rep["flags"] == {"safe_path": True, "dont_write_bytecode": True, "no_user_site": True}
    assert rep["wdm_file"] == str(VENDORED_WDM)
    assert rep["imported"]["wdm_toys.matrix"] == str(ROOT / "wdm_paper_2/toy_models/src/wdm_toys/matrix.py")
    assert rep["env"]["JAX_PLATFORMS"] == "cpu" and rep["jax_imported"] is False


@requires_legacy
@pytest.mark.legacy
def test_lineage_D_resolves_the_june_package():
    rep = _report(L.run_worker("D", "selftest_env", "--import", "wdm"))
    assert rep["wdm_file"] == str(ROOT / "wdm" / "__init__.py")


@requires_legacy
@pytest.mark.legacy
def test_lineage_C_sees_no_wdm():
    rep = _report(L.run_worker("C", "selftest_env", "--import", "config", "--import", "matrix"))
    assert rep["wdm_file"] is None
    assert rep["imported"]["config"] == str(ROOT / "toy_models/galactic_binary/config.py")
    with pytest.raises(L.WorkerError, match="No module named 'wdm'"):
        L.run_worker("C", "selftest_env", "--import", "wdm")


@requires_legacy
@pytest.mark.legacy
def test_failing_worker_raises_with_its_stderr():
    with pytest.raises(L.WorkerError, match="selftest asked to fail"):
        L.run_worker("A", "selftest_env", "--fail")


@requires_legacy
@pytest.mark.legacy
def test_workers_write_nothing_into_the_frozen_trees():
    dirs = sorted({d for paths in L.SOURCE_DIRS.values() for d in paths})
    before = (fth.scan(ROOT, hash_files=False), fth.pycache_fingerprint(dirs))
    imports = {"A": ["wdm", "wdm_toys.matrix", "wdm_toys.bandcov", "wdm_toys.fastcov"],
               "B": ["wdmtoysgpu", "wdmtoysgpu.matrix"],
               "C": ["config", "matrix", "fullcache", "likelihoods"],
               "D": ["wdm", "wdm.paper_algorithms"]}
    for lineage, modules in imports.items():
        L.run_worker(lineage, "selftest_env", *[x for m in modules for x in ("--import", m)])
    assert (fth.scan(ROOT, hash_files=False), fth.pycache_fingerprint(dirs)) == before
