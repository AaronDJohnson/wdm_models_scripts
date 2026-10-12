"""C3 / G5: forward_time(d) reproduces the stored w_d of every sinusoid case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import wdm

DATA = Path(__file__).resolve().parents[3] / "data" / "sinusoid"
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
NF, NT, DT = 64, 256, 2.0**-10


def _load(case):
    path = DATA / case / "toy_data.npz"
    if not path.is_file():
        pytest.skip(f"{path} is missing: run section A, Task A4 (it ships "
                    "data/sinusoid/manuscript_*/toy_data.npz) first")
    return np.load(path)


@pytest.mark.parametrize("case", CASES)
def test_wd_bitwise(case):
    z = _load(case)
    assert np.array_equal(wdm.forward_time(z["d"], NF, NT, DT), z["w_d"])


@pytest.mark.parametrize("case", CASES)
def test_wd_jax(case, jax_be):
    z = _load(case)
    w = np.asarray(wdm.forward_time(jax_be.asarray(z["d"]), NF, NT, DT))
    assert np.max(np.abs(w - z["w_d"])) <= 1e-13 * np.max(np.abs(z["w_d"]))
