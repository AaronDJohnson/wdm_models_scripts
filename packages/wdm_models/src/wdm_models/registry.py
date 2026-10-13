r"""The model registry (spec §5.3): the 15 paper models, the targets, ``build`` and ``describe``.

``TARGETS`` lists every (toy, case, model) the release reproduces.  S3 registers the sinusoid
(3 cases x 15 models = 45); the galactic binary's 38 targets are added with S5.

``build(toy, case, model, *, backend, options, data_root)`` loads the shipped, sha256-checked
inputs and returns the ``domainia`` likelihood (rows in ``priors.names(model)`` order);
``describe(toy, case, model)`` reports what a run analyses (cells or Fourier coordinates, grids,
exact rows, padded cells) without building anything heavy.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import wdm
from domainia.domains import CellSet
from domainia.nuisance import FourierBandNuisance

from .common.params import MODELS, check_model, split_model
from .options import ModelOptions, resolve
from .sinusoid import config as sin_config
from .sinusoid import data as sin_data
from .sinusoid import models as sin_models
from .sinusoid import priors as sin_priors
from .sinusoid.exact import ProjectedExact

__all__ = ["MODELS", "TARGETS", "TOYS", "build", "describe", "targets"]

TOYS: tuple[str, ...] = ("sinusoid",)
TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "sinusoid": {case: MODELS for case in sin_config.CASE_NAMES},
}


def targets(toy: str | None = None) -> tuple[tuple[str, str, str], ...]:
    """Every ``(toy, case, model)``, in ``TARGETS`` order."""
    toys = TOYS if toy is None else (toy,)
    return tuple((t, c, m) for t in toys for c, ms in _toy(t).items() for m in ms)


def _toy(toy: str) -> dict[str, tuple[str, ...]]:
    if toy not in TARGETS:
        raise KeyError(f"unknown toy {toy!r}; registered: {TOYS} (the galactic binary arrives with S5)")
    return TARGETS[toy]


def _check(toy: str, case: str, model: str) -> None:
    cases = _toy(toy)
    if case not in cases:
        raise KeyError(f"unknown {toy} case {case!r}; the cases are {tuple(cases)}")
    check_model(model)
    if model not in cases[case]:
        raise ValueError(f"{model!r} is not a target of {toy}/{case}")


def build(toy: str, case: str, model: str, *, backend: str = "numpy",
          options: ModelOptions = ModelOptions(), data_root: str | os.PathLike | None = None) -> Any:
    """The likelihood of ``model`` for ``toy``/``case`` on the shipped data."""
    _check(toy, case, model)
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax'; got {backend!r}")
    if backend != "numpy":
        raise NotImplementedError("S3 builds the sinusoid on NumPy; backend='jax' is sub-project S4's")
    resolved = resolve(toy, model, backend, options)
    s = sin_data.load(case, data_root=data_root)
    exact = ProjectedExact.load(s.cfg, data_root=data_root) if model == "exact" else None
    return sin_models.likelihood(model, s.cfg, s.d, resolved=resolved, exact=exact)


def describe(toy: str, case: str, model: str) -> dict[str, Any]:
    """What a run of ``model`` analyses (spec §5.3; the WT oracle's counts, pinned by tests)."""
    _check(toy, case, model)
    cfg, geo = sin_config.config(case), sin_models.MANUSCRIPT_GEOMETRY
    base, edge = split_model(model)
    out: dict[str, Any] = {"toy": toy, "case": case, "model": model, "params": sin_priors.names(model),
                           "N": cfg.N, "dt": cfg.dt}
    nuis = FourierBandNuisance(cfg.N, cfg.dt, geo.analysis_fmin)
    if base == "whittle":
        if edge == "raw":
            out.update(P=nuis.n_keep, samples=cfg.N,
                       grid=f"real orthonormal Fourier coordinates, {geo.analysis_fmin:g} Hz <= f < f_Nyq")
        else:
            start, stop = geo.cut[0] * cfg.Nf, (cfg.Nt - geo.cut[1]) * cfg.Nf
            out.update(P=FourierBandNuisance(stop - start, cfg.dt, geo.analysis_fmin).n_keep,
                       samples=stop - start, crop=(start, stop),
                       grid="real orthonormal Fourier coordinates of the cropped record")
        return out
    common = {"n_nuisance": nuis.n_drop}
    if model == "exact":
        out.update(P=cfg.N, grid="time samples (exact covariance)", n_eff=cfg.N - nuis.n_drop,
                   log_jacobian=0.0, **common)
        return out
    if edge == "pad":
        ext = wdm.Grid(cfg.Nf, cfg.Nt + geo.pad_columns, cfg.dt)
        out.update(P=cfg.N, NT=ext.Nt, grid=f"{cfg.Nf + 1} bands x {ext.Nt} columns (padded grid)",
                   lambda_cells=CellSet.all_active(ext).size, pad_columns=geo.pad_columns,
                   covariance="M Lambda M", n_eff=cfg.N - nuis.n_drop, log_jacobian=0.0, **common)
    else:
        cells = CellSet.all_active(cfg.grid())
        if edge == "cut":
            cells = cells.drop_columns(*geo.cut)
            out["cut"] = geo.cut
        out.update(P=cells.size, NT=cfg.Nt, grid=f"{cfg.Nf + 1} bands x {cfg.Nt} columns",
                   n_eff=cells.size - nuis.n_drop, log_jacobian=0.5 * cfg.N * float(np.log(cfg.dt)),
                   **common)
        if edge == "exact":
            e = cells.columns(*geo.edge_exact)
            out.update(exact_rows=e.size, interior=cells.size - e.size, exact_columns=geo.edge_exact)
    if base == "tri":
        out["lags"] = "all"
    return out
