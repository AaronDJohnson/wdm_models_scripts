"""Release-level import identity (spec section 1, guard 6).

S1 covers ``wdm``; S2 adds ``domainia`` (task D-A1); S3 adds ``wdm_models`` (task E-A1);
S5 adds the binary_galactica version/normalisation probe.  Every probe runs in a fresh
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
DOMAINIA_SRC = REL / "packages" / "domainia" / "src"
WDM_MODELS_SRC = REL / "packages" / "wdm_models" / "src"
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


_DOMAINIA = ("import json, sys, domainia, wdm\n"
             "print(json.dumps({'file': domainia.__file__, 'wdm': wdm.__file__,"
             " 'version': domainia.__version__, 'jax': 'jax' in sys.modules}))")


def test_domainia_is_the_release_on_the_release_wdm():
    info = _probe(_DOMAINIA, pythonpath=os.pathsep.join([str(DOMAINIA_SRC), str(WDM_SRC)]))
    assert Path(info["file"]).resolve().is_relative_to(DOMAINIA_SRC.resolve())
    assert Path(info["wdm"]).resolve().is_relative_to(WDM_SRC.resolve())
    assert info["version"] == "1.0.0"
    assert info["jax"] is False                       # `import domainia` never imports jax


def test_root_pytest_ini_collects_the_domainia_package_tests():
    """S2 appends packages/domainia/tests to the root testpaths and S3 (E-A1)
    packages/wdm_models/tests (neither tests/ has an __init__.py, so packages/wdm/tests
    keeps the top-level name ``tests``)."""
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(REL / "pytest.ini")
    assert cfg["pytest"]["testpaths"].split() == ["tests", "packages/wdm/tests",
                                                  "packages/domainia/tests",
                                                  "packages/wdm_models/tests"]
    assert "packages/domainia/src" in cfg["pytest"]["pythonpath"].split()
    assert not (REL / "packages" / "domainia" / "tests" / "__init__.py").exists()


_WDM_MODELS = ("import json, sys, wdm_models, domainia, wdm\n"
               "print(json.dumps({'file': wdm_models.__file__, 'domainia': domainia.__file__,"
               " 'wdm': wdm.__file__, 'version': wdm_models.__version__, 'jax': 'jax' in sys.modules}))")


def test_wdm_models_is_the_release_on_the_release_domainia_and_wdm():
    info = _probe(_WDM_MODELS, pythonpath=os.pathsep.join([str(WDM_MODELS_SRC), str(DOMAINIA_SRC),
                                                           str(WDM_SRC)]))
    assert Path(info["file"]).resolve().is_relative_to(WDM_MODELS_SRC.resolve())
    assert Path(info["domainia"]).resolve().is_relative_to(DOMAINIA_SRC.resolve())
    assert Path(info["wdm"]).resolve().is_relative_to(WDM_SRC.resolve())
    assert info["version"] == "1.0.0"
    assert info["jax"] is False                       # `import wdm_models` never imports jax


def test_root_pytest_ini_collects_the_wdm_models_package_tests():
    """S3 (E-A1): packages/wdm_models/tests is a root testpath, its src is on the root
    pythonpath, and its tests/ is not a package."""
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(REL / "pytest.ini")
    assert "packages/wdm_models/tests" in cfg["pytest"]["testpaths"].split()
    assert "packages/wdm_models/src" in cfg["pytest"]["pythonpath"].split()
    assert not (REL / "packages" / "wdm_models" / "tests" / "__init__.py").exists()


# --- binary_galactica (S5, task F-A1; spec section 1 guard 6 and section 5.2) -----------------
CONSTRAINTS = REL / "env" / "constraints.txt"
_BG = ("import json, sys\n"
       "import jax\n"
       "x64_before = bool(jax.config.read('jax_enable_x64'))\n"
       "import binary_galactica as bg\n"
       "from packaging.version import Version\n"
       "print(json.dumps({'file': bg.__file__, 'version': bg.__version__,"
       " 'at_least': Version(bg.__version__) >= Version('0.5.0'),"
       " 'x64_untouched': bool(jax.config.read('jax_enable_x64')) == x64_before}))")
#: The normalisation probe: the 3-month injection's template on the store, binary_galactica
#: alone (no release code), against the spec's gbonfly literals (prepared units, BW 4096).
_PROBE = ("import json, numpy as np, jax\n"
          "jax.config.update('jax_enable_x64', True)\n"
          "import binary_galactica as bg\n"
          "a = np.load({path!r})\n"
          "T = float(a['T'])\n"
          "with jax.ensure_compile_time_eval():\n"
          "    orbit = bg.orbit_cache(T, 0.0, 4096)\n"
          "out = {{}}\n"
          "for be in ('fourier', 'onfly'):\n"
          "    h = np.asarray(bg.template_on_window(a['inject_glass'], T=T, bin_lo=int(a['store_offset']),"
          " n_bins=int(a['store_n']), BW=4096, positions=orbit, backend=be))"
          " / (float(a['DT']) * a['sigma0'][:, None])\n"
          "    out[be] = [float(np.sum(np.abs(h) ** 2)), float(np.real(np.vdot(a['store_z'], h)))]\n"
          "print(json.dumps(out))")
STORE_SUMS_3MO = (47.0992463067, 44.6206986397)       # spec section 5.2 (gbonfly)


def test_constraints_pin_binary_galactica():
    """The release pins the PyPI release binary-galactica 0.5.0 (controller override of
    2026-10-13: the user's PyPI upload replaces the plan's floor + git-commit pins)."""
    lines = [ln.strip() for ln in CONSTRAINTS.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    assert "binary-galactica==0.5.0" in lines


def test_binary_galactica_is_the_physical_amplitude_release():
    pytest.importorskip("binary_galactica", reason="binary_galactica not installed (spec: wdm_models[gb])")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "JAX_"))}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", JAX_ENABLE_X64="false", JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-P", "-c", _BG], env=env, cwd=str(REL), capture_output=True,
                         text=True, check=True)
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["at_least"] is True, info              # 0.4.0 templates are exactly half
    assert info["x64_untouched"] is True               # bg never flips x64 on import (gbonfly did)
    assert not Path(info["file"]).resolve().is_relative_to(REL.resolve())


@pytest.mark.golden
def test_binary_galactica_normalisation_probe(record_property):
    """Spec section 1 guard 6: Sum|h|^2 and Re<z,h> at the 3-month injection on the store
    match gbonfly's at 1e-6 (``fourier``) and 1e-3 (``onfly``); a power ratio of 1/4 is
    binary_galactica < 0.5.0.  Each backend is one ``s5`` record (the S5 report)."""
    pytest.importorskip("binary_galactica", reason="binary_galactica not installed (spec: wdm_models[gb])")
    path = REL / "data" / "galactic_binary" / "prepared" / "gb_3mo.npz"
    sums = _probe(_PROBE.format(path=str(path)), pythonpath=None)
    for backend, rtol in (("fourier", 1e-6), ("onfly", 1e-3)):
        (s2, zh), (r2, rz) = sums[backend], STORE_SUMS_3MO
        record_property("s5", json.dumps({"item": f"G5/probe/gb_3mo/{backend}", "gate": rtol,
                                          "observed": max(abs(s2 / r2 - 1.0), abs(zh / rz - 1.0))}))
        assert not 0.2 < s2 / r2 < 0.3, f"{backend}: binary_galactica < 0.5.0 (half amplitude)"
        assert abs(s2 / r2 - 1.0) <= rtol and abs(zh / rz - 1.0) <= rtol, (backend, s2, zh)
