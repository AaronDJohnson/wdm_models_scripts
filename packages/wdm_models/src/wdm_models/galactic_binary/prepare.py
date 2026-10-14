r"""Prepare one GB case from its shipped segment (spec §5.2 prepare.py): the legacy
``toy_models/galactic_binary/prepare.py`` functions, ported verbatim on release pieces.

Per case (``prepare`` then ``add_matrix``):

1. :func:`load_case` reads the duration-study segment (``data/galactic_binary/segments``,
   sha256-pinned) and evaluates ``S_inst`` (:mod:`instrument`) and the fitted GLASS foreground
   (:mod:`foreground`) on its bins: :func:`psd_components`;
2. :func:`analysis_window` keeps the bins under the Meyer filters of the ``N_BANDS`` bands
   around the source at the fixed tiling ``config.ANALYSIS_NT`` (never recomputed), and
   :func:`restrict` renormalises them to unit variance at ``g = 1``;
3. :func:`fit_mu` fits ``mu`` per channel to the residual with the exact model on the
   nominal bins (JAX gradients, L-BFGS) and the Whittle model's single level;
4. :func:`calc_Nt` reports Sec. VII A's ``NT*`` on the fixed band ``config.NT_BAND``
   (:func:`nt_star`) and the nearest valid tiling (:func:`candidates`) -- a report only;
5. :func:`snr_full` -- the injection's optimal SNR under the exact model;
6. :func:`add_matrix` adds the likelihood matrix's inputs: the store (the window band-passed
   to 3 extra bands per side; inner bins verbatim), the padded grids' PSDs, ``rho_pad``, the
   cropped record's PSDs and ``matrix_cut``.

Everything is written under ``out_dir`` (never into ``data/``); the keys are
``prepared.PAPER_KEYS``.  Deviations from the legacy file, all because the dropped models were
their only readers: no frequency-padded geometry (``bands``, ``g0``, ``G``, ``phi``,
``sI_grid``, ``sG_grid``), no ``rho_win``, no absolute ``matrix_store``/``segment`` paths
(the report records the shipped relpaths and their sha256), and :func:`load_case` no longer
sets the segment-wide source prior that ``prepare`` immediately replaced.  ``geometry.py``
is not ported: only its ``even_divisors``/``candidates`` are transcribed for ``calc_Nt``.

The exact model's noise state is the legacy ``fullcache.noise_state`` on ``domainia``'s
ports (``linalg.cblock_factor``/``cblock_forward``, ``be.cholesky(impl="numpy")``,
``be.solve_triangular``), so :func:`snr_full` reproduces ``config.SNR_FULL`` bit for bit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
from wdm.backend import NUMPY, Backend
from wdm.filter import phi_centered
from wdm.transform import band_atoms

from ..common.validation import resolve_data_root, shipped
from .config import (ANALYSIS_NT, CUT, C_F, C_T, DT, ENVELOPE_PERIOD, MONTHS, MU_BOX, NT_BAND, N_BANDS,
                     PAD_COLUMNS, TEMPLATE_BW, check_case)
from .envelope import basis as envelope_basis
from .envelope import ghat, ghat_grid
from .foreground import glass_psd
from .instrument import SANGRIA_NOISE_PARAMS, tdi_instrumental_psd_AE
from .prepared import BASE_KEYS, MATRIX_KEYS

__all__ = ["SEGMENT_SHA256", "STORE_SHA256", "add_matrix", "analysis_window", "calc_Nt",
           "candidates", "even_divisors", "fit_mu", "load_case", "noise_state", "nt_star", "prepare",
           "psd_components", "restrict", "segment_rel", "snr_full", "store_rel"]

SEGMENT_SHA256 = MappingProxyType({
    "gb_3mo": "df8b528e4f06cfebd75f42a003dfe06000afd4d972013827f303aef04a586bfd",
    "gb_6mo": "0814c7af8d7f9f5bf7cc55a24e80e89f7594f49942eecbd34a50e599fbae1393",
    "gb_12mo": "7597415b9b1fb10fe1a2a0a5f1a350f919f3cd99c05a92fe9465e622769cb241"})
STORE_SHA256 = MappingProxyType({
    "gb_3mo": "f4cc7856ad49a2bbeed033f8740b2325f3a53c08eee9fdad50f45388c39b4516",
    "gb_6mo": "ab2b60b739d6e2621233627b4821f278075621bd825ad1ec2feb170bb3dbf9f1",
    "gb_12mo": "f4fafec2350822760f1f8e6012f29b9b7bf08bf6659e4a0a2867bcd9b8faa0b7"})
_INJECT_KEYS = ("f0", "fdot", "amp", "phi0", "colat", "lon", "psi", "cos_iota")


def segment_rel(case: str) -> str:
    return f"galactic_binary/segments/segment_snr7_{MONTHS[check_case(case)]}mo_3bands.npz"


def store_rel(case: str) -> str:
    return f"galactic_binary/segments/store_snr7_{MONTHS[check_case(case)]}mo_3bands_pm3.npz"


# ---------------------------------------------------------------------------------------------
# data and PSDs
# ---------------------------------------------------------------------------------------------
def psd_components(freqs, fg_params) -> tuple[np.ndarray, np.ndarray]:
    """``(S_inst, S_fg)``, each ``(2, nf)``: the Sangria A/E instrument noise and the duration's
    fitted GLASS foreground (``fg_params`` in GLASS's coordinates)."""
    if fg_params is None:
        raise ValueError("the paper's segments carry their fitted GLASS foreground (fg_params); "
                         "the one-year Sangria formula is not part of the release")
    P = SANGRIA_NOISE_PARAMS
    inst = np.atleast_2d(tdi_instrumental_psd_AE(freqs, S_oms=P.S_oms_A, S_a=P.S_a_A))
    gal = np.atleast_2d(glass_psd(freqs, np.asarray(fg_params, float)))
    inst = np.repeat(inst, 2, 0) if inst.shape[0] == 1 else inst
    gal = np.repeat(gal, 2, 0) if gal.shape[0] == 1 else gal
    return inst, gal


def load_case(case: str, *, data_root: str | os.PathLike | None = None) -> dict:
    """The shipped segment of ``case`` in prepared units (legacy ``prepare.load_case``)."""
    rel = segment_rel(case)
    art = shipped(resolve_data_root(data_root), rel, SEGMENT_SHA256[case])
    seg = art.load_npz()
    T, df = float(seg["t_obs"]), float(seg["df"])
    N = int(round(T / DT))
    bo0, n = int(seg["bin_offset"]), int(seg["n_bins"])
    freqs = (bo0 + np.arange(n)) * df
    fg = np.array(seg["fg_params"], float)
    inst, gal = psd_components(freqs, fg)
    S_ref = np.mean(inst + gal, axis=1)                                    # (2,)
    sigma0 = np.sqrt(N * S_ref / (2 * DT))

    def norm(key: str) -> np.ndarray:
        return np.stack([seg[f"{key}_{c}"] for c in "AE"]) / (DT * sigma0[:, None])

    inject = np.array([float(seg[f"inject_{k}"]) for k in _INJECT_KEYS])
    f0, fdot, amp, phi0, colat, lon, psi, cos_iota = inject
    glass = np.array([f0 * T, np.cos(colat), lon, np.log(amp), cos_iota, psi, phi0, fdot * T * T])
    return dict(case=case, T=T, df=df, N=N, bin_offset=bo0, n_bins=n, freqs=freqs,
                S_ref=S_ref, sigma0=sigma0, sI=inst / S_ref[:, None], sG=gal / S_ref[:, None],
                z=norm("data"), resid=norm("residual"), injected=norm("injected"),
                inject_glass=glass, basis_L=ghat_grid(T), fg_params=fg,
                segment=rel, segment_sha256=art.sha256)


def analysis_window(d: dict, NT: int, n_bands: int = N_BANDS) -> dict:
    """The ``n_bands`` WDM bands around the source at tiling ``NT``, in absolute rfft bins
    (legacy ``prepare.analysis_window``): ``band`` (the source's), ``first``/``last``,
    ``lo``/``n`` (the bins under the bands' nonzero Meyer filters), ``nominal`` and ``source``
    (``[lo, hi)`` of the bands' and the source band's nominal widths)."""
    if NT % 2 or d["N"] % NT:
        raise ValueError(f"NT={NT} must be even and divide N={d['N']}")
    step = NT // 2
    m = int(round(d["inject_glass"][0] / step))
    first = m - (n_bands - 1) // 2
    last = first + n_bands - 1
    nz = np.nonzero(phi_centered(d["N"] // NT, NT, DT))[0] - NT // 2
    lo, hi = first * step + int(nz.min()), last * step + int(nz.max())
    if lo < d["bin_offset"] or hi >= d["bin_offset"] + d["n_bins"]:
        raise ValueError(f"bands {first}..{last} at NT={NT} need bins {lo}..{hi}, outside the segment")
    below, above = step // 2, step - step // 2
    return dict(band=m, first=first, last=last, lo=lo, n=hi - lo + 1,
                nominal=(first * step - below, last * step + above),
                source=(m * step - below, m * step + above))


def restrict(d: dict, lo: int, n: int) -> dict:
    """Keep the absolute bins ``lo .. lo + n - 1``, renormalised to their own ``S_ref``."""
    k = slice(lo - d["bin_offset"], lo - d["bin_offset"] + n)
    S_ref = d["S_ref"] * np.mean(d["sI"][:, k] + d["sG"][:, k], axis=1)
    sigma0 = np.sqrt(d["N"] * S_ref / (2 * DT))
    rescale = (d["sigma0"] / sigma0)[:, None]
    out = dict(d, bin_offset=lo, n_bins=n, freqs=d["freqs"][k], S_ref=S_ref, sigma0=sigma0,
               sI=d["sI"][:, k] * (d["S_ref"] / S_ref)[:, None],
               sG=d["sG"][:, k] * (d["S_ref"] / S_ref)[:, None])
    for key in ("z", "resid", "injected"):
        out[key] = d[key][:, k] * rescale
    return out


# ---------------------------------------------------------------------------------------------
# the exact model's noise state (legacy fullcache.noise_state on domainia's ports)
# ---------------------------------------------------------------------------------------------
def _mprime(mu: Any, sI: Any, sG: Any, basis_L: Any, be: Backend) -> Any:
    """Dense ``M'`` (legacy ``likelihoods.full_covariance``; g-hat with the lag guard)."""
    xp = be.xp
    n = sI.shape[0]
    gh = ghat(mu, basis_L, np.arange(-(n - 1), n), be=be)
    lag = np.arange(n)[None, :] - np.arange(n)[:, None] + n - 1
    root = xp.sqrt(sG)
    return xp.diag(sI.astype(complex)) + (root[:, None] * root[None, :]) * xp.conj(gh)[lag]


def noise_state(mu: Any, sI: Any, sG: Any, basis_L: Any, z: Any, block: int | None,
                be: Backend = NUMPY) -> SimpleNamespace:
    """One channel's ``q0 = |L^-1 z|^2``, ``logdet`` and ``forward(h) = L^-1 h``: dense complex
    Cholesky, or the block-tridiagonal one in blocks of ``block`` bins (legacy
    ``fullcache.noise_state(..., with_factor=False)`` and ``forward``)."""
    from domainia.linalg import cblock_factor, cblock_forward

    xp = be.xp
    M = _mprime(mu, sI, sG, basis_L, be)
    if block:
        F = cblock_factor(M, block, be=be)
        logdet = F.logdet

        def forward(h: Any) -> Any:
            return cblock_forward(F, h, be=be)
    else:
        L, _ = be.cholesky(M, impl="numpy")
        logdet = 2.0 * xp.sum(xp.log(xp.real(xp.diagonal(L, axis1=-2, axis2=-1))))

        def forward(h: Any) -> Any:
            return be.solve_triangular(L, h, lower=True)
    u = forward(z)
    return SimpleNamespace(q0=xp.real(xp.vdot(u, u)), logdet=logdet, forward=forward)


def _lnl_whittle(r: np.ndarray, mu_c: float, sI: np.ndarray, sG: np.ndarray) -> float:
    m = sI + np.exp(mu_c) * sG
    return -np.sum(np.abs(r) ** 2 / m) - np.sum(np.log(m))


def fit_mu(d: dict, freq: slice = slice(None), block: int | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel maximum-likelihood ``mu`` on the residual's bins ``freq``: the exact model
    (L-BFGS-B with JAX gradients, inside ``MU_BOX``; restarts from a failed line search) and the
    Whittle model's single level (bounded scalar fit).  Needs float64 JAX (``require_x64``)."""
    from scipy.optimize import minimize, minimize_scalar
    from wdm.backend import jax_backend

    import jax

    be = jax_backend()                       # require_x64()
    jnp = be.xp
    mu_full, mu_freq = [], []
    basis = jnp.asarray(d["basis_L"])
    for c in range(2):
        r, sI, sG = d["resid"][c][freq], d["sI"][c][freq], d["sG"][c][freq]
        level = minimize_scalar(lambda m, r=r, sI=sI, sG=sG: -_lnl_whittle(r, m, sI, sG),
                                bounds=MU_BOX, method="bounded").x
        mu_freq.append(level)
        n = r.size

        def neg(mu, r=jnp.asarray(r), sI=jnp.asarray(sI), sG=jnp.asarray(sG)):
            st = noise_state(mu, sI, sG, basis, r, block, be)
            return (st.q0 + st.logdet) / n                # per bin: O(1) tolerances

        fg = jax.jit(jax.value_and_grad(neg))

        def f(mu):
            v, g = fg(jnp.asarray(mu))
            v, g = float(v), np.asarray(g, dtype=float)
            # an extreme trial envelope can make M' numerically indefinite: back off
            return (v, g) if np.isfinite(v) and np.all(np.isfinite(g)) else (1e10, np.zeros_like(g))

        start = np.zeros(d["basis_L"].shape[0])
        start[0] = level
        for _ in range(4):                                # a failed line search restarts from its point
            res = minimize(f, start, jac=True, method="L-BFGS-B", bounds=[MU_BOX] * start.size,
                           options=dict(ftol=1e-15, gtol=1e-9, maxiter=500))
            if res.success:
                break
            start = res.x
        if not (res.success or np.max(np.abs(f(res.x)[1])) < 1e-7):
            raise RuntimeError(f"mu fit did not converge for channel {c}: {res.message}")
        mu_full.append(res.x)
    return np.array(mu_full), np.array(mu_freq)


# ---------------------------------------------------------------------------------------------
# the tiling (Sec. VII A) and the SNR
# ---------------------------------------------------------------------------------------------
def nt_star(d: dict, mu: np.ndarray, freq: slice = slice(None), nt_grid: int = 4096) -> dict:
    """Sec. VII A optimum ``NT* = T sqrt(2 C_T r_t / (C_F r_f))`` from the fitted S(f, t), the
    RMS log-derivatives averaged over the bins ``freq`` and the whole record."""
    T, f = d["T"], d["freqs"][freq]
    t = (np.arange(nt_grid) + 0.5) * (T / nt_grid)
    E = envelope_basis(t, T)
    rf2, rt2 = [], []
    for c in range(2):
        g = np.exp(mu[c] @ E)
        S = d["sI"][c][freq][:, None] + d["sG"][c][freq][:, None] * g[None, :]    # (nf, nt)
        logS = np.log(S)
        rf2.append(np.mean(np.gradient(logS, f, axis=0) ** 2))
        rt2.append(np.mean(np.gradient(logS, t, axis=1) ** 2))
    r_f, r_t = np.sqrt(np.mean(rf2)), np.sqrt(np.mean(rt2))
    star = T * np.sqrt(2 * C_T * r_t / (C_F * r_f))
    per = [T * np.sqrt(2 * C_T * np.sqrt(rt2[c]) / (C_F * np.sqrt(rf2[c]))) for c in range(2)]
    return dict(nt_star=float(star), nt_star_A=float(per[0]), nt_star_E=float(per[1]),
                r_f=float(r_f), r_t=float(r_t))


def even_divisors(N: int) -> np.ndarray:
    """The even divisors of ``N`` (legacy ``geometry.even_divisors``)."""
    small = [d for d in range(2, int(np.sqrt(N)) + 1) if N % d == 0]
    divisors = sorted(set(small) | {N // d for d in small})
    return np.array([d for d in divisors if d % 2 == 0])


def candidates(N: int, nt_star: float, count: int = 5, span=(0.5, 2.0)) -> list[int]:
    """The ``count`` even divisors of ``N`` nearest ``nt_star`` in log distance (legacy
    ``geometry.candidates``)."""
    d = even_divisors(N)
    d = d[(d >= span[0] * nt_star) & (d <= span[1] * nt_star)]
    order = np.argsort(np.abs(np.log(d / nt_star)))
    return sorted(int(x) for x in d[order[:count]])


def calc_Nt(case: str, *, data_root: str | os.PathLike | None = None,
            band: tuple[float, float] = NT_BAND) -> dict:
    """NT* on the fixed band ``band`` (Hz) and the nearest valid tiling (a report: the
    analysis tiling is ``config.ANALYSIS_NT``).  Fits ``mu`` with the dense exact model."""
    d = load_case(case, data_root=data_root)
    lo, hi = int(np.ceil(band[0] / d["df"])), int(np.floor(band[1] / d["df"]))
    if lo < d["bin_offset"] or hi >= d["bin_offset"] + d["n_bins"]:
        raise ValueError(f"band {band} Hz needs bins {lo}..{hi}, outside the segment")
    d = restrict(d, lo, hi - lo + 1)
    mu, _ = fit_mu(d)
    star = nt_star(d, mu)
    nt = candidates(d["N"], star["nt_star"], count=1)[0]
    x = nt / star["nt_star"]
    return dict(star, nt_band_hz=list(band), nt_band_bins=[lo, hi - lo + 1], nt_mu=mu.tolist(),
                nt_valid=int(nt), nt_loss_ratio=float(0.5 * (x ** 2 + x ** -2)))


def snr_full(d: dict, mu: np.ndarray, freq: slice = slice(None), block: int | None = None) -> float:
    """Optimal SNR of the injection under the exact model on the bins ``freq`` (NumPy)."""
    s2 = 0.0
    for c in range(2):
        st = noise_state(mu[c], d["sI"][c][freq], d["sG"][c][freq], d["basis_L"],
                         d["resid"][c][freq], block, NUMPY)
        t = st.forward(d["injected"][c][freq])
        s2 += 2.0 * np.real(np.vdot(t, t))
    return float(np.sqrt(s2))


# ---------------------------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------------------------
def prepare(case: str, *, out_dir: str | os.PathLike, data_root: str | os.PathLike | None = None,
            fits: dict | None = None) -> Path:
    """Write ``out_dir/<case>.npz`` (``prepared.BASE_KEYS``) and ``<case>_report.json``.

    ``fits``: ``{"mu_fit", "mu_freq", "nt": calc_Nt dict}`` to reuse fitted values (the
    deterministic-keys check); ``None`` runs :func:`fit_mu` and :func:`calc_Nt` (minutes).
    """
    out_dir = Path(out_dir)
    d = load_case(case, data_root=data_root)
    NT = int(ANALYSIS_NT[case])
    win = analysis_window(d, NT)
    segment = (d["bin_offset"], d["n_bins"])
    d = restrict(d, win["lo"], win["n"])
    freq_lo = win["nominal"][0] - d["bin_offset"]
    freq = slice(freq_lo, freq_lo + win["nominal"][1] - win["nominal"][0])
    src = win["source"]
    import binary_galactica as bg

    prior = bg.ucb_prior(T=d["T"], fmin=src[0] * d["df"], fmax=(src[1] - 1) * d["df"])
    prior_lo, prior_hi = np.asarray(prior.lo, float), np.asarray(prior.hi, float)
    periodic = bool(np.isclose(d["T"], ENVELOPE_PERIOD, rtol=0, atol=1e-6))
    block = NT // 2 if periodic else None
    if fits is None:
        mu, mu_freq = fit_mu(d, freq, block)
        star = calc_Nt(case, data_root=data_root)
    else:
        mu, mu_freq, star = (np.asarray(fits["mu_fit"], float), np.asarray(fits["mu_freq"], float),
                             dict(fits["nt"]))
    chi2 = [float(noise_state(mu[c], d["sI"][c][freq], d["sG"][c][freq], d["basis_L"],
                              d["resid"][c][freq], block, NUMPY).q0) / (freq.stop - freq.start)
            for c in range(2)]
    snr = snr_full(d, mu, freq, block)
    arrays = dict(case=case, T=d["T"], N=d["N"], DT=DT, df=d["df"], bin_offset=d["bin_offset"],
                  n_bins=d["n_bins"], sigma0=d["sigma0"], S_ref=d["S_ref"], z=d["z"],
                  resid=d["resid"], injected=d["injected"], sI=d["sI"], sG=d["sG"],
                  basis_L=d["basis_L"], inject_glass=d["inject_glass"], prior_lo=prior_lo,
                  prior_hi=prior_hi, mu_fit=mu, mu_freq=mu_freq, NT=NT, nt_star=star["nt_star"],
                  envelope_period=ENVELOPE_PERIOD, template_bw=TEMPLATE_BW, window_band=win["band"],
                  first_band=win["first"], n_bands=win["last"] - win["first"] + 1,
                  freq_lo=freq.start, freq_n=freq.stop - freq.start,
                  segment_bin_offset=segment[0], segment_n_bins=segment[1], fg_params=d["fg_params"])
    if tuple(arrays) != BASE_KEYS:                                  # pragma: no cover
        raise RuntimeError("prepare writes prepared.BASE_KEYS, in order")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case}.npz"
    np.savez(path, **arrays)
    report = dict(case=case, segment=d["segment"], segment_sha256=d["segment_sha256"],
                  T_days=d["T"] / 86400, N=d["N"], n_bins=d["n_bins"], window_band=win["band"],
                  bands=[win["first"], win["last"]], window_bins=[win["lo"], win["n"]],
                  frequency_bins=[int(win["nominal"][0]), int(win["nominal"][1] - win["nominal"][0])],
                  source_band_bins=[int(src[0]), int(src[1] - src[0])],
                  mu_fit=mu.tolist(), mu_freq=mu_freq.tolist(), residual_chi2_per_bin=chi2,
                  **star, NT=NT, snr_full=snr, envelope_period_days=ENVELOPE_PERIOD / 86400,
                  foreground={"model": "GLASS (foreground.glass_psd)",
                              "params_lnA_lnf1_alpha_lnfk_lnf2": list(map(float, d["fg_params"]))})
    (out_dir / f"{case}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------------------------
# add_matrix: the store, the padded grids, the cropped record
# ---------------------------------------------------------------------------------------------
def _band_grid(N_rec: int, NT_rec: int, NT: int, bands: np.ndarray) -> SimpleNamespace:
    """Legacy ``matrix.make_grid``: ``bands`` on a record of ``NT`` columns with the analysis
    record's ``Nf``; ``g0``/``G`` the window bins (of that record's rfft grid) the bands span."""
    if NT % 2:
        raise ValueError("NT must be even")
    Nf = N_rec // NT_rec
    N = NT * Nf
    bands = np.asarray(bands)
    g0 = (NT // 2) * int(bands[0]) - NT // 2
    G = (NT // 2) * int(bands[-1] - bands[0]) + NT
    idx = ((NT // 2) * (bands - bands[0]))[:, None] + np.arange(NT)[None, :]
    return SimpleNamespace(NT=NT, N=N, T=N * DT, Nf=Nf, bands=bands, g0=g0, G=G, idx=idx)


def _store_bands(NT: int, offset: int, n: int) -> np.ndarray:
    """Every band whose window ``[(b - 1) NT/2, (b + 1) NT/2)`` overlaps the store's bins."""
    h = NT // 2
    return np.arange(offset // h, -(-(offset + n) // h) + 1)


def _energy_fraction(grid: SimpleNamespace, w2: np.ndarray) -> np.ndarray:
    """``(nB, NT)`` share of each cell's atom energy under ``w2`` (legacy
    ``matrix.energy_fraction``; atoms ``wdm.transform.band_atoms``, bit-identical to the GB
    ``geometry.band_atoms``)."""
    N, NT, Nf = grid.N, grid.NT, grid.Nf
    if w2.size != N:
        raise ValueError("w2 must cover the grid's record")
    atoms = band_atoms(grid.bands, Nf, NT, DT)
    W2 = np.fft.rfft(w2)
    rho = np.empty((grid.bands.size, NT))
    for b in range(grid.bands.size):
        start = grid.g0 + int(grid.idx[b, 0])
        for a in (0, 1):
            V = np.zeros(N // 2 + 1, complex)
            V[start:start + NT] = np.conj(atoms[b, a])
            a2 = np.fft.irfft(V, n=N) ** 2
            corr = np.fft.irfft(W2 * np.conj(np.fft.rfft(a2)), n=N)     # sum_t w2[t+s] a2[t]
            ns = np.arange(a, NT, 2)
            rho[b, ns] = corr[((ns - a) * Nf) % N] / a2.sum()
    return rho


def add_matrix(case: str, *, out_dir: str | os.PathLike, data_root: str | os.PathLike | None = None) -> Path:
    """Add ``prepared.MATRIX_KEYS`` to ``out_dir/<case>.npz`` (written by :func:`prepare`;
    rewritten in place, every key already there kept byte for byte).  Legacy
    ``prepare.add_matrix(case, out_dir)`` without the windowed models' ``rho_win`` and the
    ``matrix_store`` path."""
    path = Path(out_dir) / f"{check_case(case)}.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    if str(arrays["case"]) != case:
        raise ValueError(f"{path} holds case {str(arrays['case'])!r}, not {case!r}")
    T, N, NT = float(arrays["T"]), int(arrays["N"]), int(arrays["NT"])
    bin_offset, n_bins = int(arrays["bin_offset"]), int(arrays["n_bins"])
    sigma0, S_ref = np.asarray(arrays["sigma0"], float), np.asarray(arrays["S_ref"], float)
    fg = np.asarray(arrays["fg_params"], float)
    st = shipped(resolve_data_root(data_root), store_rel(case), STORE_SHA256[case]).load_npz()
    if not np.isclose(float(st["t_obs"]), T, rtol=0, atol=1e-6):
        raise ValueError(f"{store_rel(case)} is not {case}")
    off, n = int(st["bin_offset"]), int(st["n_bins"])
    inner = slice(bin_offset - off, bin_offset - off + n_bins)
    new: dict[str, Any] = dict(store_offset=off, store_n=n)

    def splice(name: str, value: np.ndarray, old: np.ndarray) -> None:
        err = np.max(np.abs(value[:, inner] - old)) / np.max(np.abs(old))
        if err > 1e-12:
            raise RuntimeError(f"{case}: the store's inner {name} differs from the prepared one by {err:.2e}")
        value[:, inner] = old
        new[name] = value

    for key, name in (("data", "z"), ("residual", "resid"), ("injected", "injected")):
        value = np.stack([st[f"{key}_{c}"] for c in "AE"]) / (DT * sigma0[:, None])
        splice(f"store_{name}", value, arrays[name])
    inst, gal = psd_components((off + np.arange(n)) / T, fg)
    splice("store_sI", inst / S_ref[:, None], arrays["sI"])
    splice("store_sG", gal / S_ref[:, None], arrays["sG"])

    bands = int(arrays["first_band"]) + np.arange(int(arrays["n_bands"]))
    grid = _band_grid(N, NT, NT + PAD_COLUMNS, bands)
    inst, gal = psd_components((grid.g0 + np.arange(grid.G)) / grid.T, fg)
    new.update(pad_g0=grid.g0, pad_G=grid.G, pad_sI=inst / S_ref[:, None], pad_sG=gal / S_ref[:, None])
    w = np.zeros(grid.N)
    w[:N] = 1.0
    new["rho_pad"] = _energy_fraction(grid, w)[:, :NT]
    gR = _band_grid(N, NT, grid.NT, _store_bands(NT, off, n))
    inst, gal = psd_components((gR.g0 + np.arange(gR.G)) / gR.T, fg)
    new.update(pad_store_g0=gR.g0, pad_store_G=gR.G, pad_store_sI=inst / S_ref[:, None],
               pad_store_sG=gal / S_ref[:, None])

    lead, trail = CUT[case]
    Nc = (NT - lead - trail) * (N // NT)
    lo = bin_offset + int(arrays["freq_lo"])
    hi = lo + int(arrays["freq_n"])
    k0, k1 = -((-lo * Nc) // N), -((-hi * Nc) // N)             # f in [lo / T, hi / T)
    inst, gal = psd_components(np.arange(k0, k1) / (Nc * DT), fg)
    new.update(crop_k0=k0, crop_n=k1 - k0, crop_sI=inst / S_ref[:, None], crop_sG=gal / S_ref[:, None])
    new["matrix_cut"] = np.array([lead, trail])
    if tuple(new) != MATRIX_KEYS:                                   # pragma: no cover
        raise RuntimeError("add_matrix writes prepared.MATRIX_KEYS, in order")
    clash = set(new) & set(arrays)
    if clash:
        raise RuntimeError(f"{case}: {path} already holds {sorted(clash)}")
    arrays.update(new)
    np.savez(path, **arrays)
    return path
