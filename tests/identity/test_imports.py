"""Release-level import identity (spec section 1, guard 6).

S1 covers ``wdm`` only; S2 adds ``domainia``, S5 ``wdm_models`` and the
binary_galactica version/normalisation probe.  Every probe runs in a fresh
``python -P`` subprocess so nothing this pytest session imported can mask the
answer.
"""

from __future__ import annotations

import configparser
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REL = Path(__file__).resolve().parents[2]             # wdm_models_scripts/ (scripts.golden.paths.REL)
WDM_SRC = REL / "packages" / "wdm" / "src"
WORKSPACE = REL.parent                                # holds the June wdm/ tree (paths.ROOT)


def _probe(code: str, *, cwd: Path = REL, safe_path: bool = True,
           pythonpath: str | None = str(WDM_SRC)) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    cmd = [sys.executable, *(["-P"] if safe_path else []), "-c", code]
    out = subprocess.run(cmd, env=env, cwd=str(cwd), capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


_WDM = ("import json, sys, wdm\n"
        "print(json.dumps({'file': wdm.__file__, 'version': getattr(wdm, '__version__', None),"
        " 'api': getattr(wdm, '__wdm_api__', None), 'release': getattr(wdm, '__release__', None),"
        " 'many': hasattr(wdm, 'forward_time_many'), 'jax': 'jax' in sys.modules}))")


def test_wdm_is_the_paper_release():
    info = _probe(_WDM)
    assert Path(info["file"]).resolve().is_relative_to(WDM_SRC.resolve())
    assert (info["version"], info["api"], info["release"]) == ("1.0.0", "functions/1", "wdm-paper-1")
    assert info["many"] is True
    assert info["jax"] is False                       # `import wdm` never imports jax


def test_safe_path_keeps_the_june_tree_out():
    """From the workspace root (where the June ``wdm/`` lives) ``-P`` still gives the release."""
    if not (WORKSPACE / "wdm" / "__init__.py").is_file():
        pytest.skip("no June wdm/ tree next to the release")
    info = _probe(_WDM, cwd=WORKSPACE)
    assert Path(info["file"]).resolve().is_relative_to(WDM_SRC.resolve())


def test_without_safe_path_the_june_tree_would_shadow_the_release():
    """Why every entry point is ``python -P``: cwd = workspace root shadows any install."""
    if not (WORKSPACE / "wdm" / "__init__.py").is_file():
        pytest.skip("no June wdm/ tree next to the release")
    code = "import json, wdm; print(json.dumps({'file': wdm.__file__}))"
    info = _probe(code, cwd=WORKSPACE, safe_path=False)
    assert Path(info["file"]).resolve().is_relative_to((WORKSPACE / "wdm").resolve())


def test_no_foreign_wdm_is_installed_in_this_env():
    """Without PYTHONPATH, ``wdm`` is either absent or this release (editable install);
    never lisa/wdm 0.2.0 or any other distribution."""
    code = ("import importlib.util, json\n"
            "s = importlib.util.find_spec('wdm')\n"
            "print(json.dumps({'origin': None if s is None else s.origin}))")
    info = _probe(code, pythonpath=None)
    if info["origin"] is not None:
        assert Path(info["origin"]).resolve().is_relative_to(WDM_SRC.resolve())


def test_distribution_metadata():
    meta = tomllib.loads((REL / "packages" / "wdm" / "pyproject.toml").read_text())["project"]
    assert meta["name"] == "wdm-paper"
    assert meta["version"] == "1.0.0"
    assert meta["requires-python"] == ">=3.12"
    assert meta["dependencies"] == ["numpy>=2.1,<3", "scipy>=1.14.1,<2"]
    assert meta["optional-dependencies"]["jax"] == ["jax>=0.9", "jaxlib>=0.9"]


def test_root_pytest_ini_collects_the_wdm_package_tests():
    """Section A owns the root pytest.ini (Task A1); S1 appends packages/wdm/tests to
    its testpaths so a bare `pytest` from the release root runs the package tests too."""
    ini = REL / "pytest.ini"
    if not ini.is_file():
        pytest.skip("no root pytest.ini: run section A, Task A1 first")
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(ini)
    assert "packages/wdm/tests" in cfg["pytest"]["testpaths"].split()
    assert "packages/wdm/src" in cfg["pytest"]["pythonpath"].split()
