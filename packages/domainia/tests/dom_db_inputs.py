"""Shared inputs of the D-B tests (maps, spectra, variances): golden paths, the sinusoid
and GB test inputs, a dense covariance stand-in and the JAX helper.

The model layer (S3 sinusoid, S5 GB) owns the real configurations; these builders are
test-local copies of the few numbers the S2 gates need, read from ``data/`` and
``golden/``.  Golden numbers are read with ``scripts.golden.goldens_io`` (``read_json`` +
``to_float``).  A golden array that a BLAS call touched (``@``, ``np.linalg.norm``) is
compared bitwise only on its recording build (:func:`same_build`, D-A's convention) and by
value at the fingerprint's ``idx`` with a tolerance everywhere else (:func:`check_blas_golden`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REL = Path(__file__).resolve().parents[3]
if str(REL) not in sys.path:          # activate.sh exports it; a bare run still finds the harness
    sys.path.insert(0, str(REL))

from scripts.golden.goldens_io import read_json, sha256_array, to_float  # noqa: E402
from scripts.golden.workers._common import runtime_fingerprint  # noqa: E402

import wdm  # noqa: E402

GOLDEN = REL / "golden"
DATA = REL / "data"
SIN_G6 = GOLDEN / "sinusoid" / "intermediates" / "manuscript_mid_truth"
GB_CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
#: GB cut columns per case (toy_models/galactic_binary/matrix.py CUT; S5 owns the config).
GB_CUT = {"gb_3mo": (5, 4), "gb_6mo": (5, 4), "gb_12mo": (0, 0)}
GB_PAD_COLUMNS = 16
#: The sinusoid grid (S3 owns SinusoidConfig): N = 16384, dt = 2^-10, T = 16 s.
SIN_NF, SIN_NT, SIN_DT, SIN_T = 64, 256, 2.0**-10, 16.0
SIN_CUT, SIN_PAD_COLUMNS, SIN_FMIN = (6, 5), 16, 4.0

__all__ = ["DATA", "GB_CASES", "GB_CUT", "GB_PAD_COLUMNS", "GOLDEN", "REL", "SIN_CUT", "SIN_DT",
           "SIN_FMIN", "SIN_G6", "SIN_NF", "SIN_NT", "SIN_PAD_COLUMNS", "SIN_T", "DenseCov",
           "TableEnvelope", "check_blas_golden", "fingerprint_gap", "gb_case", "jax_or_skip",
           "read_json", "same_build", "sha256_array", "sinusoid_g6", "sinusoid_data", "to_float"]


def jax_or_skip():
    """``(jax, JAX backend)`` with float64 on; skips in the NumPy-only lane."""
    jax = pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    from wdm.backend import enable_x64, jax_backend

    enable_x64()
    return jax, jax_backend()


def sinusoid_g6() -> tuple[dict, dict]:
    """The G6 json (via ``read_json``) and npz of the manuscript_mid truth."""
    meta = read_json(SIN_G6.with_suffix(".json"))
    with np.load(SIN_G6.with_suffix(".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def sinusoid_data(case: str = "manuscript_mid") -> np.ndarray:
    with np.load(DATA / "sinusoid" / case / "toy_data.npz") as z:
        return np.asarray(z["d"], dtype=float)


def fingerprint_gap(fp: dict, arr) -> float:
    """``max |arr.flat[idx] - values| / max |values|`` for a G6 fingerprint (real or complex)."""
    flat = np.asarray(arr).reshape(-1)[np.asarray(fp["idx"] if "idx" in fp else fp["sample_index"])]
    if "values_re" in fp:
        ref = np.asarray([to_float(v) for v in fp["values_re"]]) \
            + 1j * np.asarray([to_float(v) for v in fp["values_im"]])
    else:
        ref = np.asarray([to_float(v) for v in fp.get("values", fp.get("sample_values"))])
    return float(np.max(np.abs(flat - ref)) / np.max(np.abs(ref)))


def same_build(recorded: dict) -> bool:
    """D-A's ``same_build`` (tests/conftest.py) as a plain function: True when this process
    matches a golden's recording build on every key the record has among machine / python /
    numpy / scipy / blas.  Only there must a BLAS-touched golden reproduce bit for bit."""
    current = runtime_fingerprint()
    keys = [k for k in ("machine", "python", "numpy", "scipy", "blas") if k in recorded]
    return all(recorded[k] == current.get(k) for k in keys)


def check_blas_golden(fp: dict, arr, recorded: dict, rtol: float) -> bool:
    """A G6 array a BLAS call touched: its sha256 on the recording build, else its
    fingerprint values to ``rtol`` of max (spec rule: never sha256 across BLAS builds)."""
    if tuple(np.shape(arr)) != tuple(fp["shape"]):
        return False
    if same_build(recorded):
        return sha256_array(arr) == fp["sha256"]
    return fingerprint_gap(fp, arr) <= rtol


@dataclass(frozen=True, eq=False)
class TableEnvelope:
    """A test-local ``Envelope`` on the G6 table ``basis_half`` (the GB basis at
    ``t = k dT / 2``, ``k = 0 .. 2 NT - 2``); S5 owns the real ``AnnualEnvelope``."""

    table: np.ndarray
    dT: float

    @property
    def n_params(self) -> int:
        return self.table.shape[0]

    def basis(self, t) -> np.ndarray:
        k = np.asarray(t, dtype=float) * 2.0 / self.dT
        ki = np.rint(k).astype(np.int64)
        if not np.array_equal(ki.astype(float), k) or ki.min() < 0 or ki.max() >= self.table.shape[1]:
            raise ValueError("t is not on the half-column table")
        return self.table[:, ki]

    def log_g(self, t, mu, be=None):
        return mu @ self.basis(t)

    def ghat(self, lags, mu, be=None):
        raise NotImplementedError("D-B tests never need ghat")


def gb_case(case: str) -> dict:
    """The prepared GB arrays and the G6 golden of one case, plus the derived geometry."""
    with np.load(DATA / "galactic_binary" / "prepared" / f"{case}.npz") as z:
        prep = {k: z[k] for k in z.files}
    meta = read_json(GOLDEN / "galactic_binary" / f"intermediates_{case}.json")
    with np.load(GOLDEN / "galactic_binary" / f"intermediates_{case}.npz") as z:
        g6 = {k: z[k] for k in z.files}
    N, NT, DT = int(prep["N"]), int(prep["NT"]), float(prep["DT"])
    first = int(prep["first_band"])
    return {"prep": prep, "meta": meta, "g6": g6, "N": N, "NT": NT, "DT": DT,
            "bands": tuple(range(first, first + int(prep["n_bands"]))),
            "grid": wdm.Grid(N // NT, NT, DT)}


class DenseCov:
    """A dense covariance stand-in: ``matvec`` (real, row-batched) and ``dense``."""

    def __init__(self, S: np.ndarray) -> None:
        self.S = np.asarray(S)

    def matvec(self, X, be=None):
        return X @ self.S

    def dense(self, be=None):
        return self.S
