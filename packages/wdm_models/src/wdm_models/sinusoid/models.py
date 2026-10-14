r"""The 15 sinusoid likelihoods assembled from ``domainia`` pieces (spec §4.8, §5.3).

A port of the S2 gate's working wiring (``scripts/golden/s2_cases.py::sinusoid_likelihood``,
bitwise against G1/G6 at the truth) with the edge constants and the strategy choices made
explicit, so the same builder serves the manuscript cases (``registry.build``) and the small
grids the dense-oracle tests use.

=====================  =====================================================================
model                  composition (``A_n`` outside; density ``RealGaussian``)
=====================  =====================================================================
``whittle_raw``        ``FourierWhittle("real-orthonormal")``, ``fmin <= f < f_Nyq``
``whittle_cut``        the same on samples ``[cut_lead Nf, (Nt - cut_trail) Nf)`` (own rfft grid)
``{b}_raw``            ``TimeToWDM`` on every active cell; ``b`` = band (Eq. 36 diagonal), mid
                       (Eq. 39 diagonal) or tri (Eq. 45 block-tridiagonal); flat marginal over
                       the discarded Fourier modes; log-Jacobian ``N/2 ln dt``
``{b}_cut``            the same on the cells outside the ``cut`` columns (same Jacobian, full N)
``{b}_exact``          ``SchurEdge``: exact rows (``PushedRows`` of the operator covariance) on
                       the ``edge_exact`` columns over the approximate interior
``{b}_pad``            ``ZeroPadded("time_marginal")``: the base on every cell of the grid with
                       ``pad_columns`` more columns (alpha continued, ``t_max = Nt' dT``),
                       the residual samples as data, ``Identity`` map, log-Jacobian 0
``exact``              ``ProjectedExact`` (``exact.py``): projected PCG + ``ExactLogDet`` + the
                       bandcov correction; ``density_order="folded"`` (WT order)
=====================  =====================================================================

Both backends (S4): ``resolved.backend`` selects NumPy (WT column, every domainia form
``solve``) or JAX (WG column: gram + padded, whitened tri marginal, adaptive rows (staged on CPU),
``on_fail="nan"``, the alpha_d clamp, the ``table`` correction, the pads' ``reduce`` marginal
and the ``scaled`` Schur complement over a diagonal interior -- E-C's WG-shaped states); on
JAX the ``exact`` model also carries WG's validity mask as ``row_ok`` (:func:`_exact_row_ok`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import wdm
from domainia.covariance import (DiagonalCovariance, Eq45Covariance, ExactTimeCovariance, PushedRows,
                                 SchurEdge, ZeroPadded)
from domainia.data import TimeSeries
from domainia.density import RealGaussian
from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.likelihood import FourierWhittle, GaussianLikelihood
from domainia.maps import Identity, TimeToWDM
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.variance import BandAveragedVariance, MidpointVariance
from wdm.backend import get_backend

from ..common.params import split_model
from ..options import Resolved, resolve
from . import priors
from .config import ANALYSIS_FMIN, CUT, EDGE_EXACT, EXACT_P, PAD_COLUMNS, SinusoidConfig
from .exact import ProjectedExact
from .psd import as_loglinear
from .signal import Sinusoid

__all__ = ["Geometry", "MANUSCRIPT_GEOMETRY", "likelihood"]


@dataclass(frozen=True)
class Geometry:
    """The edge constants of one grid: the cut and exact-edge columns ``(lead, trail)``, the
    appended pad columns and the analysis cut (Hz; ``0.0`` = no Fourier marginal)."""

    cut: tuple[int, int] = CUT
    edge_exact: tuple[int, int] = EDGE_EXACT
    pad_columns: int = PAD_COLUMNS
    analysis_fmin: float = ANALYSIS_FMIN


MANUSCRIPT_GEOMETRY = Geometry()


def _base(base: str, psd: Any, cells: CellSet, r: Resolved, t_max: float | None = None) -> Any:
    if base == "band":
        return DiagonalCovariance(BandAveragedVariance(psd, cells), len(cells.m))
    if base == "mid":
        return DiagonalCovariance(MidpointVariance(psd, cells), len(cells.m))
    return Eq45Covariance(psd, cells, builder=r.tri_build, layout=r.layout, t_max=t_max, P=EXACT_P)


def _exact_row_ok(params: Any, box: np.ndarray) -> Any:
    """WG ``BandFullCovarianceGPU._lnlike_batch``'s validity (``bandcov.py:117-126``): the noise
    inside the correction's box and ``|ln A_n| < 700``; traceable.  Rows it rejects are ``-inf``
    without a PD flag (the JAX ``exact`` model, whose ``table`` correction cannot refuse under
    tracing)."""
    i_ac, i_ad = params.noise
    i_amp = params.index("log10_A_n")
    lo, hi = np.asarray(box[:, 0], dtype=float), np.asarray(box[:, 1], dtype=float)

    def ok(theta: Any) -> Any:
        xp = get_backend(theta).xp
        ac, ad = theta[i_ac], theta[i_ad]
        loga = theta[i_amp] * np.log(10.0)
        return ((ac >= lo[0]) & (ac <= hi[0]) & (ad >= lo[1]) & (ad <= hi[1])
                & xp.isfinite(loga) & (loga > -700.0) & (loga < 700.0))

    return ok


def likelihood(model: str, cfg: SinusoidConfig, d: Any, *, geometry: Geometry = MANUSCRIPT_GEOMETRY,
               resolved: Resolved | None = None, exact: ProjectedExact | None = None) -> Any:
    """``model`` on the record ``d`` (``cfg.N`` samples of ``cfg.dt``).

    ``resolved`` defaults to the NumPy historical strategies
    (``options.resolve("sinusoid", model, "numpy")``); ``exact`` (the fitted assets, loaded in
    ``resolved.correction``'s order -- ``ValueError`` otherwise) is required for the ``exact``
    model.  Returns a ``domainia`` ``GaussianLikelihood`` (``FourierWhittle`` for the Whittle
    forms) whose rows are ``priors.names(model)``, on ``resolved.backend``; in the JAX column
    (``exact_on_fail="nan"``) the ``exact`` model's ``row_ok`` is WG's validity mask
    (:func:`_exact_row_ok`).
    """
    base, edge = split_model(model)
    r = resolve("sinusoid", model, "numpy") if resolved is None else resolved
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (cfg.N,):
        raise ValueError(f"d must hold the {cfg.N} samples of the record; got shape {d.shape}")
    time = TimeDomain(cfg.N, cfg.dt)
    data = TimeSeries(d, time)
    psd = as_loglinear(cfg)
    signal = Sinusoid.for_config(cfg)
    params = priors.params(model)
    fmin = float(geometry.analysis_fmin)
    if base == "whittle":
        if not fmin > 0.0:
            raise ValueError("the Whittle forms need a positive analysis cut (geometry.analysis_fmin)")
        crop = None if edge == "raw" else (geometry.cut[0] * cfg.Nf, (cfg.Nt - geometry.cut[1]) * cfg.Nf)
        return FourierWhittle("real-orthonormal", data, psd=psd, fmin=fmin, crop=crop, signal=signal,
                              params=params, backend=r.backend)
    amplitude = RealGaussian(0.0, "log10_A_n")
    clamp = None if r.noise_clamp is None else dict(r.noise_clamp)
    if model == "exact":
        if exact is None:
            raise ValueError("the exact model needs its fitted assets: pass "
                             "exact=ProjectedExact.load(cfg, algorithm=resolved.correction) (shipped for "
                             "the manuscript grid)")
        if exact.correction.algorithm != r.correction:
            raise ValueError(f"the exact assets evaluate the correction in {exact.correction.algorithm!r} "
                             f"order; the resolved strategies need {r.correction!r} "
                             "(ProjectedExact.load(..., algorithm=resolved.correction))")
        row_ok = _exact_row_ok(params, exact.correction.bounds) if r.exact_on_fail == "nan" else None
        return GaussianLikelihood(data, Identity(time), exact.covariance(on_fail=r.exact_on_fail), amplitude,
                                  signal=signal, params=params, noise_clamp=clamp, row_ok=row_ok,
                                  density_order=r.density_order, backend=r.backend)
    nuis = FourierBandNuisance(cfg.N, cfg.dt, fmin) if fmin > 0.0 else None
    if edge == "pad":
        ext = wdm.Grid(cfg.Nf, cfg.Nt + geometry.pad_columns, cfg.dt)
        cov = ZeroPadded(_base(base, psd, CellSet.all_active(ext), r,
                               t_max=ext.Nt * wdm.delta_T(cfg.Nf, cfg.dt)),
                         strategy="time_marginal", grid=ext, pad_columns=geometry.pad_columns,
                         form=r.pad_form)
        U = None if nuis is None else nuis.discarded_basis().T
        if U is not None and r.pad_form == "reduce":          # WG: the marginal reduces, too
            cov, U = FlatMarginal(cov, U, form="reduce"), None
        return GaussianLikelihood(data, Identity(time), cov, amplitude, signal=signal, params=params,
                                  nuisance=U,
                                  noise_clamp=clamp, density_order=r.density_order, backend=r.backend)
    cells = CellSet.all_active(cfg.grid())
    if edge == "cut":
        cells = cells.drop_columns(*geometry.cut)
    tmap = TimeToWDM(time, WDMDomain(cells))
    if edge == "exact":
        e = cells.columns(*geometry.edge_exact)
        rows = PushedRows(ExactTimeCovariance(psd, cfg.N, cfg.dt, form="operator", P=EXACT_P), tmap, e,
                          order=r.rows_order, phi_cache=r.phi_cache, staged=r.staged, chunk=r.exact_chunk)
        cov = SchurEdge(_base(base, psd, cells.complement(e), r), rows, e, form=r.schur_form)
    else:
        cov = _base(base, psd, cells, r)
    U = None if nuis is None else tmap.apply(nuis.discarded_basis()).T
    if U is not None and r.marginal_form == "whiten":
        cov, U = FlatMarginal(cov, U, form="whiten"), None
    return GaussianLikelihood(data, tmap, cov, RealGaussian(0.5 * cfg.N * float(np.log(cfg.dt)), "log10_A_n"),
                              signal=signal, params=params, nuisance=U, noise_clamp=clamp,
                              density_order=r.density_order, backend=r.backend)
