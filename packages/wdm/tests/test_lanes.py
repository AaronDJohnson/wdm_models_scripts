"""C9: the NumPy-only CI lane (spec section 1, "CI lanes" 1).

``WDM_TESTS_HIDE_JAX=1`` makes ``import jax`` fail inside the test session, so
every JAX test must skip -- and every NumPy test must still pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def test_numpy_only_lane_skips_every_jax_test_and_passes_the_rest():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PKG / "src"),
               WDM_TESTS_HIDE_JAX="1")
    out = subprocess.run(
        [sys.executable, "-P", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs",
         "tests/test_backend.py", "tests/test_transform_shim.py"],
        env=env, cwd=str(PKG), capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "hidden by WDM_TESTS_HIDE_JAX=1" in out.stdout or "jax not installed" in out.stdout
    assert " skipped" in out.stdout and " failed" not in out.stdout
