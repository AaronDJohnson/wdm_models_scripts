"""The release interpreter is the pinned `wdmrel` env (Task A1; spec §1 Python, §3.6)."""
from __future__ import annotations

import csv
import importlib.metadata as md
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PINS = {"numpy": "2.3.5", "scipy": "1.16.3", "jax": "0.9.0.1", "jaxlib": "0.9.0.1",
        "ml_dtypes": "0.5.4", "opt_einsum": "3.4.0", "h5py": "3.14.0", "pytest": "8.4.2"}
FEW = Path.home() / "miniconda3" / "envs" / "few"
SITE = Path("lib") / "python3.12" / "site-packages"
needs_few = pytest.mark.skipif(not FEW.is_dir() or platform.system() != "Darwin",
                               reason="build identity is checked against the local `few` env only")


def test_python_is_conda_forge_3_12_4():
    assert sys.version_info[:3] == (3, 12, 4)
    assert "conda-forge" in sys.version


def test_interpreter_is_wdmrel():
    assert Path(sys.prefix).name == "wdmrel"


@pytest.mark.parametrize("dist", sorted(PINS))
def test_pinned_distribution(dist):
    assert md.version(dist) == PINS[dist]


def test_cpu_jax_is_float64_when_x64_is_on():
    env = dict(os.environ, JAX_ENABLE_X64="true", JAX_PLATFORMS="cpu")
    out = subprocess.run(
        [sys.executable, "-P", "-c",
         "import jax, jax.numpy as jnp; print(jax.default_backend(), jnp.ones(1).dtype)"],
        env=env, cwd=tempfile.gettempdir(), capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["cpu", "float64"]


@needs_few
def test_python_build_identical_to_few():
    out = subprocess.run([str(FEW / "bin" / "python"), "-I", "-B", "-c", "import sys; print(sys.version)"],
                         cwd=tempfile.gettempdir(), capture_output=True, text=True, check=True)
    assert out.stdout.strip() == sys.version


def _compiled(site: Path, dist_info: str) -> dict[str, str]:
    with (site / dist_info / "RECORD").open(newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if row and row[0].endswith((".so", ".dylib"))}


@needs_few
@pytest.mark.parametrize("dist_info", ["numpy-2.3.5.dist-info", "scipy-1.16.3.dist-info",
                                       "jaxlib-0.9.0.1.dist-info", "h5py-3.14.0.dist-info",
                                       "ml_dtypes-0.5.4.dist-info"])
def test_compiled_libraries_identical_to_few(dist_info):
    ours = _compiled(Path(sys.prefix) / SITE, dist_info)
    assert ours, f"{dist_info} lists no compiled libraries"
    assert ours == _compiled(FEW / SITE, dist_info)
