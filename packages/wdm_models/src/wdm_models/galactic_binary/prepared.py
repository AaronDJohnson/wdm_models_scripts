r"""One prepared GB case (spec §5.2 prepared.py): legacy ``likelihoods.Prepared`` restricted
to the paper keys, loaded fail-closed.

The shipped files ``data/galactic_binary/prepared/gb_{3,6,12}mo.npz`` (sha256 pinned here and
in ``data/MANIFEST.json``) were written by the legacy ``prepare.py`` (+ ``--matrix``).  They also
carry keys only the dropped models read -- the frequency-padded geometry (``bands``, ``g0``,
``G``, ``phi``, ``sI_grid``, ``sG_grid``), the windowed models' ``rho_win`` and the absolute
``matrix_store`` path -- which :class:`PreparedCase` never reads.  :data:`PAPER_KEYS` is the
schema :func:`prepare.prepare` + :func:`prepare.add_matrix` write and the regeneration check
compares.

Units: ``z = (dt rfft x) / (DT sigma0)`` with ``sigma0 = sqrt(N S_ref / (2 DT))``, so
``E|z|^2 = s_I + g s_G`` with ``s_X = S_X / S_ref`` (window bins ``[bin_offset, +n_bins)``,
store bins ``[store_offset, +store_n)``; ``freq`` the nominal bins inside the window).

``periodic`` (12 months: ``T`` equals the 365-d envelope period) selects the block-tridiagonal
exact factor with blocks ``exact_block = NT/2`` (450 bins); :meth:`require_cut` is the
``whittle_cut`` guard (the crop PSDs were tabulated for ``matrix_cut``, so a file whose
``matrix_cut`` is not ``config.CUT[case]`` is refused).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
from domainia.assets import Artifact

from ..common.params import check_model
from ..common.validation import resolve_data_root, shipped
from .config import CASES, CUT, DT, MU_BOX, N_HARM, check_case, is_stationary

__all__ = ["BASE_KEYS", "FIT_KEYS", "MATRIX_KEYS", "PAPER_KEYS", "PREPARED_SHA256", "PreparedCase",
           "prepared_rel"]

#: The shipped prepared files (data/MANIFEST.json records the same digests).
PREPARED_SHA256 = MappingProxyType({
    "gb_3mo": "ad440fc145b693c46bed9f16a90ba1d64b4fff60f2bdb6209c9f58482d0b6afc",
    "gb_6mo": "2ced666ca77f564a40ae0a40baf707db6f913ee063e3df291baf2a0844dc9bd6",
    "gb_12mo": "6033977b0dcd3d4c94c281af4a84afbbed4f4a1145bf56cd2e2f9fb020e71711"})
#: Written by ``prepare.prepare`` (the analysis window, its PSDs, the injection, the prior,
#: the envelope fit and the tiling).
BASE_KEYS: tuple[str, ...] = (
    "case", "T", "N", "DT", "df", "bin_offset", "n_bins", "sigma0", "S_ref", "z", "resid",
    "injected", "sI", "sG", "basis_L", "inject_glass", "prior_lo", "prior_hi", "mu_fit", "mu_freq",
    "NT", "nt_star", "envelope_period", "template_bw", "window_band", "first_band", "n_bands",
    "freq_lo", "freq_n", "segment_bin_offset", "segment_n_bins", "fg_params")
#: Added by ``prepare.add_matrix`` (store, padded grids, cropped record, edge rule).
MATRIX_KEYS: tuple[str, ...] = (
    "store_offset", "store_n", "store_z", "store_resid", "store_injected", "store_sI", "store_sG",
    "pad_g0", "pad_G", "pad_sI", "pad_sG", "rho_pad", "pad_store_g0", "pad_store_G",
    "pad_store_sI", "pad_store_sG", "crop_k0", "crop_n", "crop_sI", "crop_sG", "matrix_cut")
PAPER_KEYS: tuple[str, ...] = BASE_KEYS + MATRIX_KEYS
#: The keys a fit sets (``fit_mu`` / ``calc_Nt``: L-BFGS, compared at 1e-6 on regeneration);
#: every other paper key is deterministic and regenerates ``np.array_equal``.
FIT_KEYS: tuple[str, ...] = ("mu_fit", "mu_freq", "nt_star")


def prepared_rel(case: str) -> str:
    """``galactic_binary/prepared/<case>.npz`` (relative to the data root)."""
    return f"galactic_binary/prepared/{check_case(case)}.npz"


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a)
    a.setflags(write=False)
    return a


class PreparedCase:
    """The paper keys of one prepared file, read-only.

    ``PreparedCase.load(case)`` checks the shipped file's pinned sha256 (and the manifest);
    ``PreparedCase(path)`` reads any file with the paper schema (a regenerated one), checking
    ``sha256`` only when given.  Missing paper keys raise ``KeyError`` naming them.
    """

    def __init__(self, path: str | os.PathLike, *, sha256: str | None = None) -> None:
        art = Artifact(Path(path), sha256=sha256)
        raw = art.load_npz()
        missing = [k for k in PAPER_KEYS if k not in raw]
        if missing:
            raise KeyError(f"{path}: not a prepared GB file; missing paper keys {missing}")
        self.path = art.path
        self.arrays = MappingProxyType({k: _readonly(raw[k]) for k in PAPER_KEYS})
        a = self.arrays
        self.case = check_case(str(a["case"]))
        self.T, self.N, self.DT = float(a["T"]), int(a["N"]), float(a["DT"])
        if self.DT != DT:
            raise ValueError(f"{path}: DT = {self.DT}, the toy's cadence is {DT}")
        self.df = float(a["df"])
        self.NT = int(a["NT"])
        self.bin_offset, self.n_bins = int(a["bin_offset"]), int(a["n_bins"])
        self.store_offset, self.store_n = int(a["store_offset"]), int(a["store_n"])
        self.sigma0, self.S_ref = a["sigma0"], a["S_ref"]
        self.z, self.resid, self.injected = a["z"], a["resid"], a["injected"]
        self.sI, self.sG = a["sI"], a["sG"]
        self.store_z, self.store_sI, self.store_sG = a["store_z"], a["store_sI"], a["store_sG"]
        self.basis_L = a["basis_L"]
        self.n_env = int(self.basis_L.shape[0])
        if self.n_env != 1 + 2 * N_HARM:
            raise ValueError(f"{path}: basis_L has {self.n_env} rows; the envelope has {1 + 2 * N_HARM}")
        self.inject = a["inject_glass"]
        self.prior_lo, self.prior_hi = a["prior_lo"], a["prior_hi"]
        self.mu_fit, self.mu_freq = a["mu_fit"], a["mu_freq"]
        self.envelope_period = float(a["envelope_period"])
        self.first_band, self.n_bands = int(a["first_band"]), int(a["n_bands"])
        self.window_band = int(a["window_band"])
        lo = int(a["freq_lo"])
        #: The nominal bins (Fourier likelihoods) inside the window: 360 / 657 / 1350.
        self.freq = slice(lo, lo + int(a["freq_n"]))
        self.matrix_cut = tuple(int(v) for v in a["matrix_cut"])

    @classmethod
    def load(cls, case: str, *, data_root: str | os.PathLike | None = None) -> "PreparedCase":
        """The shipped ``gb_{3,6,12}mo`` file, sha256-checked (fail-closed)."""
        art = shipped(resolve_data_root(data_root), prepared_rel(case), PREPARED_SHA256[case])
        out = cls(art.path, sha256=art.sha256)
        if out.case != case:
            raise ValueError(f"{art.path} holds case {out.case!r}, not {case!r}")
        return out

    # -- geometry ------------------------------------------------------------------------------
    @property
    def periodic(self) -> bool:
        """The envelope is periodic over the record (12 months): the exact factor is blocked."""
        return bool(np.isclose(self.T, self.envelope_period, rtol=0, atol=1e-6))

    @property
    def exact_block(self) -> int | None:
        """``NT // 2`` bins per block when :attr:`periodic`, else ``None`` (dense Cholesky)."""
        return self.NT // 2 if self.periodic else None

    @property
    def bands(self) -> tuple[int, ...]:
        """The analysis bands' absolute indices (65-67 / 71-73 / 69-71)."""
        return tuple(self.first_band + b for b in range(self.n_bands))

    @property
    def source_band(self) -> tuple[int, int]:
        """``[lo, hi)`` absolute bins of the source band's nominal width (the ``f0`` prior)."""
        step = self.NT // 2
        m = self.window_band
        return m * step - step // 2, m * step + step - step // 2

    def require_cut(self) -> tuple[int, int]:
        """``matrix_cut`` when it is ``config.CUT[case]``; ``ValueError`` otherwise (the
        ``whittle_cut`` guard, legacy ``matrix.whittle_map``)."""
        if self.matrix_cut != tuple(CUT[self.case]):
            raise ValueError(f"{self.path}: matrix_cut {self.matrix_cut} is not CUT[{self.case!r}] = "
                             f"{tuple(CUT[self.case])}; re-run prepare.add_matrix")
        return self.matrix_cut

    # -- parameters ----------------------------------------------------------------------------
    def ndim(self, model: str) -> int:
        check_model(model)
        return 8 + (2 if is_stationary(model) else 2 * self.n_env)

    def anchor(self, model: str) -> np.ndarray:
        """The injection plus the fitted noise (``mu_freq`` for Whittle, ``mu_fit`` otherwise)."""
        mu = self.mu_freq if is_stationary(model) else self.mu_fit.ravel()
        return np.concatenate([self.inject, mu])

    def bounds(self, model: str) -> np.ndarray:
        """``(ndim, 2)`` ``[lo, hi]``: the stored source box, then ``MU_BOX`` per noise entry."""
        k = self.ndim(model) - 8
        lo = np.concatenate([self.prior_lo, np.full(k, MU_BOX[0])])
        hi = np.concatenate([self.prior_hi, np.full(k, MU_BOX[1])])
        return np.stack([lo, hi], axis=1)

    # -- domainia pieces -------------------------------------------------------------------------
    def domain(self, k0: int, n: int):
        """``FourierDomain(N, DT, k0, n, "sigma0")``: absolute bins of this record."""
        from domainia.domains import FourierDomain

        return FourierDomain(self.N, self.DT, int(k0), int(n), "sigma0")

    def store_domain(self):
        return self.domain(self.store_offset, self.store_n)

    def window_domain(self):
        return self.domain(self.bin_offset, self.n_bins)

    def nominal_domain(self):
        return self.domain(self.bin_offset + self.freq.start, self.freq.stop - self.freq.start)

    def envelope(self):
        """The case's :class:`envelope.AnnualEnvelope` on the file's ``basis_L``."""
        from .envelope import AnnualEnvelope

        return AnnualEnvelope(self.T, self.envelope_period, N_HARM, self.basis_L)

    def __repr__(self) -> str:
        return f"PreparedCase({self.case!r}, NT={self.NT}, path={str(self.path)!r})"


if CASES != tuple(PREPARED_SHA256):                                  # pragma: no cover
    raise ImportError("PREPARED_SHA256 must list config.CASES in order")
