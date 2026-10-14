"""E-A1: which ``wdm_models``, ``domainia`` and ``wdm`` are imported, the sentinels, no JAX
on import, the distribution metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
PKG = TESTS.parent                                    # packages/wdm_models
SRC = PKG / "src"
DOMAINIA_SRC = PKG.parent / "domainia" / "src"
WDM_SRC = PKG.parent / "wdm" / "src"
WORKSPACE = PKG.parents[2]                            # holds the June wdm/ tree


def _probe(code: str, *paths: Path, cwd: Path = TESTS) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    return subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(cwd),
                          capture_output=True, text=True)


_INFO = ("import json, sys, importlib, pkgutil, wdm_models, domainia, wdm\n"
         "mods = sorted(m.name for m in pkgutil.walk_packages(wdm_models.__path__, 'wdm_models.'))\n"
         "for name in mods:\n"
         "    importlib.import_module(name)\n"
         "print(json.dumps({'wdm_models': wdm_models.__file__, 'domainia': domainia.__file__,"
         " 'wdm': wdm.__file__, 'version': wdm_models.__version__, 'modules': mods,"
         " 'jax': 'jax' in sys.modules}))")


def _info() -> dict:
    out = _probe(_INFO, SRC, DOMAINIA_SRC, WDM_SRC)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_wdm_models_domainia_and_wdm_resolve_to_the_release():
    info = _info()
    assert Path(info["wdm_models"]).resolve().is_relative_to(SRC.resolve())
    assert Path(info["domainia"]).resolve().is_relative_to(DOMAINIA_SRC.resolve())
    assert Path(info["wdm"]).resolve().is_relative_to(WDM_SRC.resolve())
    assert info["version"] == "1.0.0"


def test_importing_every_module_does_not_import_jax():
    """The sinusoid layer is NumPy-first: no module imports jax at import time (the GB
    waveform, which needs binary_galactica on JAX, is S5's and revisits this)."""
    info = _info()
    assert {"wdm_models.common", "wdm_models.sinusoid"} <= set(info["modules"])
    assert info["jax"] is False


def test_a_foreign_wdm_is_refused(tmp_path):
    fake = tmp_path / "wdm"
    fake.mkdir()
    (fake / "__init__.py").write_text("__version__ = '0.2.0'\n")
    out = _probe("import wdm_models", SRC, DOMAINIA_SRC, tmp_path)
    assert out.returncode != 0
    assert f"ImportError: wdm resolved to {fake / '__init__.py'}: not the paper release" in out.stderr


def test_a_wdm_without_the_batched_api_is_refused(tmp_path):
    fake = tmp_path / "wdm"
    fake.mkdir()
    (fake / "__init__.py").write_text("__release__ = 'wdm-paper-1'\n")
    out = _probe("import wdm_models", SRC, DOMAINIA_SRC, tmp_path)
    assert out.returncode != 0
    assert "not the paper release" in out.stderr


def test_a_foreign_domainia_is_refused(tmp_path):
    fake = tmp_path / "domainia"
    fake.mkdir()
    (fake / "__init__.py").write_text("__version__ = '0.1.0'\n")
    out = _probe("import wdm_models", SRC, tmp_path, WDM_SRC)
    assert out.returncode != 0
    assert (f"ImportError: domainia resolved to {fake / '__init__.py'}: not the paper release"
            in out.stderr)


def test_the_june_wdm_tree_is_refused():
    """From the workspace root without -P the June ``wdm/`` shadows everything;
    wdm_models must refuse it rather than run on it."""
    if not (WORKSPACE / "wdm" / "__init__.py").is_file():
        pytest.skip("no June wdm/ tree next to the release")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(SRC), str(DOMAINIA_SRC)]))
    out = subprocess.run([sys.executable, "-c", "import wdm_models"], env=env, cwd=str(WORKSPACE),
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "not the paper release" in out.stderr


def test_distribution_metadata():
    meta = tomllib.loads((PKG / "pyproject.toml").read_text())
    proj = meta["project"]
    assert proj["name"] == "wdm-models"
    assert proj["version"] == "1.0.0"
    assert proj["requires-python"] == ">=3.12"
    assert proj["dependencies"] == ["domainia==1.0.0", "jax>=0.9"]
    extras = proj["optional-dependencies"]
    assert extras["gb"] == ["binary_galactica>=0.5.0"]
    assert extras["pipeline"] == ["h5py"]
    # impulse v2 (964283c, env/constraints.txt) is not on PyPI, whose 1.0.0 is the pre-rewrite API
    assert extras["sampling"] == ["impulse-mcmc>=2"]
    ini = meta["tool"]["pytest"]["ini_options"]
    assert ini["pythonpath"] == ["src", "../domainia/src", "../wdm/src"]
    assert "-p no:cacheprovider" in ini["addopts"]
    assert '-m "not slow"' in ini["addopts"]


def test_tests_directory_is_not_a_package():
    """packages/wdm/tests owns the top-level name ``tests`` in a root run."""
    assert not (TESTS / "__init__.py").exists()
    names = sorted(p.name for p in TESTS.glob("test_*.py"))
    assert names and all(n.startswith("test_wm_") for n in names)
