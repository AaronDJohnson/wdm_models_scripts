"""D-A1: which ``domainia`` and ``wdm`` are imported, the wdm sentinel, no JAX on import."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
PKG = TESTS.parent                                    # packages/domainia
SRC = PKG / "src"
WDM_SRC = PKG.parent / "wdm" / "src"
WORKSPACE = PKG.parents[2]                            # holds the June wdm/ tree


def _probe(code: str, *paths: Path, cwd: Path = TESTS) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    return subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(cwd),
                          capture_output=True, text=True)


_INFO = ("import json, sys, domainia, wdm\n"
         "print(json.dumps({'domainia': domainia.__file__, 'wdm': wdm.__file__,"
         " 'version': domainia.__version__, 'jax': 'jax' in sys.modules}))")


def test_domainia_and_wdm_resolve_to_the_release():
    out = _probe(_INFO, SRC, WDM_SRC)
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(info["domainia"]).resolve().is_relative_to(SRC.resolve())
    assert Path(info["wdm"]).resolve().is_relative_to(WDM_SRC.resolve())
    assert info["version"] == "1.0.0"


def test_import_domainia_does_not_import_jax():
    out = _probe(_INFO, SRC, WDM_SRC)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])["jax"] is False


def test_a_foreign_wdm_is_refused(tmp_path):
    fake = tmp_path / "wdm"
    fake.mkdir()
    (fake / "__init__.py").write_text("__version__ = '0.2.0'\n")
    out = _probe("import domainia", SRC, tmp_path)
    assert out.returncode != 0
    assert f"ImportError: wdm resolved to {fake / '__init__.py'}: not the paper release" in out.stderr


def test_a_wdm_without_the_batched_api_is_refused(tmp_path):
    fake = tmp_path / "wdm"
    fake.mkdir()
    (fake / "__init__.py").write_text("__release__ = 'wdm-paper-1'\n")
    out = _probe("import domainia", SRC, tmp_path)
    assert out.returncode != 0
    assert "not the paper release" in out.stderr


def test_the_june_wdm_tree_is_refused():
    """From the workspace root without -P the June ``wdm/`` shadows everything; domainia
    must refuse it rather than run on it."""
    if not (WORKSPACE / "wdm" / "__init__.py").is_file():
        pytest.skip("no June wdm/ tree next to the release")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(SRC))
    out = subprocess.run([sys.executable, "-c", "import domainia"], env=env, cwd=str(WORKSPACE),
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "not the paper release" in out.stderr


def test_distribution_metadata():
    meta = tomllib.loads((PKG / "pyproject.toml").read_text())
    proj = meta["project"]
    assert proj["name"] == "domainia"
    assert proj["version"] == "1.0.0"
    assert proj["requires-python"] == ">=3.12"
    assert proj["dependencies"] == ["wdm-paper==1.0.0", "numpy>=2.1,<3", "scipy>=1.14.1,<2"]
    assert proj["optional-dependencies"]["jax"] == ["jax>=0.9", "jaxlib>=0.9"]
    ini = meta["tool"]["pytest"]["ini_options"]
    assert ini["pythonpath"] == ["src", "../wdm/src"]
    assert "-p no:cacheprovider" in ini["addopts"]


def test_tests_directory_is_not_a_package():
    """packages/wdm/tests owns the top-level name ``tests`` in a root run."""
    assert not (TESTS / "__init__.py").exists()
    names = sorted(p.name for p in TESTS.glob("test_*.py"))
    assert names and all(n.startswith("test_dom_") for n in names)
