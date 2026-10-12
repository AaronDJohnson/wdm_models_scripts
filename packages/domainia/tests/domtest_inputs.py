"""Test-local inputs for the covariance tests (section D-C of the S2 plan).

The model layer owns these constructions later (S3 ``wdm_models.sinusoid``, S5
``wdm_models.galactic_binary``: configs, signals, the annual envelope).  S2 builds them
here from ``data/`` and ``golden/`` so the covariance operators can be gated against
G6 without the model layer.  Nothing here is imported by ``src/``.

Golden JSON is read with the semantics of ``scripts.golden.goldens_io.read_json`` /
``to_float`` (the package lane cannot import ``scripts/``).
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
import wdm
from wdm.grid import active_mask
from wdm.transform import band_atoms

REL = Path(__file__).resolve().parents[3]
GOLDEN = REL / "golden"
DATA = REL / "data"

# --------------------------------------------------------------------------------------
# golden readers
# --------------------------------------------------------------------------------------


def golden_json(rel_path: str) -> dict:
    """``read_json`` of ``golden/<rel_path>``."""
    return json.loads((GOLDEN / rel_path).read_text())


def num(v: object) -> float:
    """``to_float``: JSON numbers and the strings "inf", "-inf", "nan"."""
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v]
    return float(v)


def assert_summary(arr: np.ndarray, summary: dict, *, rtol: float = 1e-12) -> None:
    """``arr`` against a G6 summary (``array_summary`` keys ``sample_index``/``sample_values``/
    ``l2``, or ``fingerprint`` keys ``idx``/``values[_re,_im]``/``fro``): every sampled value
    within ``rtol * norm`` and the norm within ``rtol * norm`` (BLAS-produced arrays are
    compared by value, never by sha256)."""
    arr = np.ascontiguousarray(arr)
    flat = arr.reshape(-1)
    assert list(arr.shape) == list(summary["shape"])
    norm = num(summary["l2"] if "l2" in summary else summary["fro"])
    idx = np.asarray(summary["sample_index"] if "sample_index" in summary else summary["idx"], int)
    if "values_re" in summary:
        want = np.asarray(summary["values_re"], float) + 1j * np.asarray(summary["values_im"], float)
    else:
        want = np.asarray(summary["sample_values"] if "sample_values" in summary else summary["values"], float)
    assert np.max(np.abs(flat[idx] - want), initial=0.0) <= rtol * norm
    assert abs(np.linalg.norm(flat) - norm) <= rtol * norm


# --------------------------------------------------------------------------------------
# sinusoid (manuscript_mid truth; WT ToyConfig values, spec §5.1)
# --------------------------------------------------------------------------------------

SIN_NF, SIN_NT, SIN_DT = 64, 256, 2.0 ** -10
SIN_N = SIN_NF * SIN_NT
SIN_T = SIN_N * SIN_DT                       # 16.0 s
SIN_F_MIN, SIN_F_REF, SIN_FMIN_CUT = 4.0, 1.0, 4.0
SIN_NU = np.array([-2.0, 1.2])               # (alpha_c, alpha_d) at the truth
SIN_A_N = 400.0


@functools.cache
def sinusoid_mid() -> dict:
    """``d``, ``h`` (the truth signal, shipped with the data), the G6 report and arrays."""
    with np.load(DATA / "sinusoid" / "manuscript_mid" / "toy_data.npz", allow_pickle=False) as z:
        d, h = np.asarray(z["d"], float), np.asarray(z["h"], float)
    with np.load(GOLDEN / "sinusoid" / "intermediates" / "manuscript_mid_truth.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return {"d": d, "h": h, "g6": golden_json("sinusoid/intermediates/manuscript_mid_truth.json"),
            "arrays": arrays}


def sinusoid_cells_residual() -> np.ndarray:
    """WT ``y = v.map(d) - v.map(h)`` on every active cell of the 64 x 256 grid."""
    s = sinusoid_mid()
    mask = active_mask(SIN_NF, SIN_NT)
    return (wdm.forward_time(s["d"], SIN_NF, SIN_NT, SIN_DT)[mask]
            - wdm.forward_time(s["h"], SIN_NF, SIN_NT, SIN_DT)[mask])


def log_f_clip(f: np.ndarray) -> np.ndarray:
    """WT ``noise.log_f_clip`` (f_min = 4, f_ref = 1)."""
    return np.log(np.maximum(np.asarray(f, dtype=float), SIN_F_MIN) / SIN_F_REF)


def wt_band_variance(Nf: int, Nt: int, dt: float, ac: float, ad: float) -> np.ndarray:
    """WT ``matrix.band_variance`` on an (Nf, Nt) grid, A_n = 1, alpha over the 16 s record."""
    from wdm.filter import phi_centered
    from wdm.grid import packed_cells

    f_nyq = 0.5 / dt
    m, n = packed_cells(Nf, Nt)
    t = n * wdm.delta_T(Nf, dt)
    phi2 = phi_centered(Nf, Nt, dt) ** 2
    sup = np.flatnonzero(phi2)
    phi2 = phi2[sup]
    f_off = (np.arange(Nt) - Nt // 2)[sup] / (Nf * Nt * dt)
    fb = wdm.band_frequencies(Nf, dt)[:, None] + f_off[None, :]
    ff = np.abs(fb)
    ff = np.where(ff > f_nyq, 2.0 * f_nyq - ff, ff)
    Lb = log_f_clip(ff)
    a = ac + ad * (np.asarray(t, float) / SIN_T - 0.5)
    S = np.exp(Lb[m] * a[:, None])
    return 0.5 * (S * phi2[None, :]).sum(axis=1) / phi2.sum()


def wt_mid_variance(Nf: int, Nt: int, dt: float, ac: float, ad: float) -> np.ndarray:
    """WT ``matrix.mid_variance`` on an (Nf, Nt) grid, A_n = 1."""
    from wdm.grid import packed_cells

    m, n = packed_cells(Nf, Nt)
    t = n * wdm.delta_T(Nf, dt)
    Lm = log_f_clip(wdm.band_frequencies(Nf, dt))
    return 0.5 * np.exp((ac + ad * (np.asarray(t, float) / SIN_T - 0.5)) * Lm[m])


def sha256(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# galactic binary (prepared npz + G6 at the anchor)
# --------------------------------------------------------------------------------------


def raw_basis(t: np.ndarray, n_harm: int, period: float) -> np.ndarray:
    """GB ``envelope.raw_basis`` verbatim."""
    t = np.asarray(t, float)
    rows = [np.ones_like(t)]
    for h in range(1, n_harm + 1):
        w = 2.0 * np.pi * h / period
        rows += [np.cos(w * t), np.sin(w * t)]
    return np.stack(rows)


def orthonormal_coefficients(T: float, n_harm: int, period: float, L: int = 1 << 20) -> np.ndarray:
    """GB ``envelope.orthonormal_coefficients`` verbatim."""
    t = np.arange(L) * (T / L)
    B = raw_basis(t, n_harm, period)
    n = B.shape[0]
    C = np.zeros((n, n))
    C[0, 0] = 1.0
    E = [np.ones(L)]
    for i in range(1, n):
        c = np.zeros(n)
        c[i] = 1.0
        v = B[i].copy()
        for j, e in enumerate(E):
            p = float(np.mean(v * e) / np.mean(e * e))
            v -= p * e
            c -= p * C[j]
        s = float(np.sqrt(np.mean(v * v)))
        E.append(v / s)
        C[i] = c / s
    return C


def gb_phase(bands: np.ndarray, NT: int, phi: np.ndarray) -> np.ndarray:
    """GB wdm_bands._phase verbatim (the per-band phase of wdm_full.analyse)."""
    from wdm.grid import carrier_phase

    n = np.arange(NT)
    out = np.empty((bands.size, NT), complex)
    for b, m in enumerate(bands):
        quad = np.where((int(m) + n) % 2 == 1, -1j, 1.0)
        out[b] = carrier_phase(int(m), NT) * quad * (-1.0) ** n * NT / np.linalg.norm(phi)
    return out


class AnnualEnvelope:
    """Stand-in for ``wdm_models.galactic_binary.envelope`` (S5): the ``Envelope`` protocol
    plus the host ``basis(t)`` and the decimation-grid ``basis_L`` D-C reads."""

    def __init__(self, T: float, period: float, n_harm: int, basis_L: np.ndarray) -> None:
        self.T, self.period, self.n_harm = float(T), float(period), int(n_harm)
        self.C = orthonormal_coefficients(self.T, self.n_harm, self.period)
        self.n_params = int(self.C.shape[0])
        self.basis_L = np.asarray(basis_L, float)

    def basis(self, t: np.ndarray) -> np.ndarray:
        return self.C @ raw_basis(t, self.n_harm, self.period)

    def log_g(self, t: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.asarray(mu) @ self.basis(t)

    def ghat(self, lags: np.ndarray, mu: np.ndarray) -> np.ndarray:
        L = self.basis_L.shape[1]
        R = np.fft.rfft(np.exp(np.asarray(mu) @ self.basis_L)) / L
        lags = np.asarray(lags)
        out = R[np.abs(lags)]
        return np.where(lags < 0, np.conj(out), out)


class GBCase:
    """One prepared GB case with its G6 report: grids, cells, spectra, envelope, anchor."""

    def __init__(self, case: str) -> None:
        self.case = case
        with np.load(DATA / "galactic_binary" / "prepared" / f"{case}.npz", allow_pickle=False) as z:
            self.P = {k: z[k] for k in z.files}
        self.g6 = golden_json(f"galactic_binary/intermediates_{case}.json")
        with np.load(GOLDEN / "galactic_binary" / f"intermediates_{case}.npz", allow_pickle=False) as z:
            self.A = {k: z[k] for k in z.files}
        P = self.P
        self.NT, self.N, self.DT = int(P["NT"]), int(P["N"]), float(P["DT"])
        self.Nf = self.N // self.NT
        self.bands = int(P["first_band"]) + np.arange(int(P["n_bands"]))
        self.grid = wdm.Grid(self.Nf, self.NT, self.DT)
        self.mu = np.asarray(P["mu_fit"], float)
        self.store_k0, self.store_n = int(P["store_offset"]), int(P["store_n"])
        self.NTp = self.NT + 16
        self.pad_grid = wdm.Grid(self.Nf, self.NTp, self.DT)
        h = self.NT // 2
        self.R_bands = np.arange(self.store_k0 // h, -(-(self.store_k0 + self.store_n) // h) + 1)
        self.k0 = int(np.flatnonzero(self.R_bands == self.bands[0])[0])

    @functools.cached_property
    def cells(self):
        """The analysis bands x NT columns of the record grid (band order)."""
        from domainia.domains import CellSet

        return CellSet.bands(self.grid, self.bands)

    @functools.cached_property
    def env(self) -> AnnualEnvelope:
        return AnnualEnvelope(float(self.P["T"]), float(self.P["envelope_period"]), 2, self.P["basis_L"])

    def domain(self, N: int, k0: int, n: int):
        from domainia.domains import FourierDomain

        return FourierDomain(N=N, dt=self.DT, k0=k0, n=n, units="sigma0")

    @functools.cached_property
    def psd(self):
        """SeparablePSD tabulated on the store bins of the record (s_I, s_G with the envelope)."""
        from domainia.spectra import SeparablePSD, TabulatedSpectrum

        store = self.domain(self.N, self.store_k0, self.store_n)
        return SeparablePSD(((TabulatedSpectrum(store, self.P["store_sI"]), None),
                             (TabulatedSpectrum(store, self.P["store_sG"]), self.env)), ("A", "E"))

    @functools.cached_property
    def pad_psd(self):
        """SeparablePSD tabulated on the padded record's store-band window (Lambda_R)."""
        from domainia.spectra import SeparablePSD, TabulatedSpectrum

        R = self.domain(self.pad_grid.N, int(self.P["pad_store_g0"]), int(self.P["pad_store_G"]))
        return SeparablePSD(((TabulatedSpectrum(R, self.P["pad_store_sI"]), None),
                             (TabulatedSpectrum(R, self.P["pad_store_sG"]), self.env)), ("A", "E"))

    @property
    def window(self):
        """The analysis window: the prepared n_bins bins (419 / 767 / 1575)."""
        return self.domain(self.N, int(self.P["bin_offset"]), int(self.P["n_bins"]))

    @property
    def nominal(self):
        """The nominal bins of the Fourier likelihoods (360 / 657 / 1350)."""
        lo = int(self.P["freq_lo"])
        return self.domain(self.N, int(self.P["bin_offset"]) + lo, int(self.P["freq_n"]))

    def model_index(self, model: str) -> int:
        return self.g6["quad_logdet_models"].index(model)

    def quad_logdet(self, model: str) -> np.ndarray:
        """``(2, 2)``: ``[[quad_A, logdet_A], [quad_E, logdet_E]]`` at the anchor (G6)."""
        return self.A["quad_logdet"][self.model_index(model)]

    def grid_window(self) -> tuple[int, int]:
        """``(g0, G)`` of the record grid's analysis-band windows (GB ``make_grid``)."""
        return (self.NT // 2) * int(self.bands[0]) - self.NT // 2, (self.NT // 2) * int(self.bands[-1] - self.bands[0]) + self.NT

    def atoms(self, cells, k0: int, n: int) -> np.ndarray:
        """Unit-norm atoms of ``cells`` on the absolute bins ``[k0, k0 + n)`` (GB ``V.U``)."""
        g = cells.grid
        m, nn = np.asarray(cells.m), np.asarray(cells.n)
        bands = np.unique(m)
        at = band_atoms(bands, g.Nf, g.Nt, g.dt)
        U = np.zeros((m.size, n), complex)
        for i, (mm, col) in enumerate(zip(m, nn, strict=True)):
            cols = (g.Nt // 2) * int(mm) - g.Nt // 2 + np.arange(g.Nt) - k0
            ok = (cols >= 0) & (cols < n)
            U[i, cols[ok]] = at[int(mm - bands[0]), col][ok]
        return U

    def residual_cells(self, cells) -> np.ndarray:
        """``(2, P)``: store data minus the anchor template on ``cells`` (record grid):
        ``Re(z[:, cols] @ U^T) - Re(h[:, cols] @ U^T)`` (GB ``make_lnlike``)."""
        g0, G = self.grid_window()
        U = self.atoms(cells, g0, G)
        cols = slice(g0 - self.store_k0, g0 - self.store_k0 + G)
        d = np.real(self.P["store_z"][:, cols] @ U.T)
        return d - np.real(self.A["store_template"][:, cols] @ U.T)

    def pad_map_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(C, band, n)``: the padded variant's kept cells ``Re(C z_store)`` (GB
        ``variant(prep, "pad")``: ``_cells(grid, *pad_maps(...)) * sqrt(N/N')``, ``n < NT``)."""
        from domainia.maps.fourier import pad_maps
        from wdm.filter import phi_centered

        NTp, gp = self.NTp, self.pad_grid
        g0p = (NTp // 2) * int(self.bands[0]) - NTp // 2
        Gp = (NTp // 2) * int(self.bands[-1] - self.bands[0]) + NTp
        A, B = pad_maps(self.N, gp.N, g0p + np.arange(Gp), self.store_k0 + np.arange(self.store_n))
        phi = phi_centered(self.Nf, NTp, self.DT)
        phase = gb_phase(self.bands, NTp, phi)
        idx = ((NTp // 2) * (self.bands - self.bands[0]))[:, None] + np.arange(NTp)[None, :]

        def analyse(X: np.ndarray) -> np.ndarray:                      # GB wdm_full.analyse
            return phase[:, :, None] * np.fft.ifft(phi[None, :, None] * X, axis=1)

        nB = self.bands.size
        C = analyse(A[idx]).reshape(nB * NTp, -1) + np.conj(analyse(B[idx]).reshape(nB * NTp, -1))
        C = C * np.sqrt(self.N / gp.N)
        b = np.repeat(np.arange(nB), NTp)
        n = np.tile(np.arange(NTp), nB)
        keep = n < self.NT
        return C[keep], b[keep], n[keep]


@functools.cache
def gb_case(case: str) -> GBCase:
    """One :class:`GBCase` per process."""
    return GBCase(case)
