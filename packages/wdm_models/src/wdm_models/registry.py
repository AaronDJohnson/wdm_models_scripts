r"""The model registry (spec §5.3): the 15 paper models, the targets, ``build`` and ``describe``.

``TARGETS`` lists every (toy, case, model) the release reproduces: the sinusoid (3 cases x 15
models = 45) and the galactic binary (15 + 15 + 8 = 38: at 12 months the periodic envelope
makes the cut and the exact edges identical to raw).

``build(toy, case, model, *, backend, options, data_root)`` loads the shipped, sha256-checked
inputs and returns the ``domainia`` likelihood (rows in ``priors.names(model)`` order): bare on
NumPy (WT), inside WG's evaluator on JAX (``sinusoid.evaluation``: a ``NoiseStateCache`` for the
eight cached models, a ``BatchedLikelihood`` otherwise; the domainia likelihood is ``.lik``);
``describe(toy, case, model)`` reports what a run analyses (cells or Fourier coordinates, grids,
exact rows, padded cells) without building anything heavy.

The galactic binary (S6) builds through :func:`wdm_models.galactic_binary.models.build` (NumPy:
lineage C's reference, or its cache classes with ``ModelOptions(evaluator="cache")``; the
template is ``waveform.GBTemplate`` with ``options.waveform_backend``, ``"onfly"`` by default --
golden checks pass ``"fourier"``); its ``describe`` is lineage C's ``matrix.describe`` key for key.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import wdm
from domainia.domains import CellSet
from domainia.nuisance import FourierBandNuisance

from .common.params import MODELS, check_model, split_model
from .galactic_binary import config as gb_config
from .galactic_binary import models as gb_models
from .galactic_binary.prepared import PreparedCase
from .options import ModelOptions, resolve
from .sinusoid import config as sin_config
from .sinusoid import data as sin_data
from .sinusoid import models as sin_models
from .sinusoid import priors as sin_priors
from .sinusoid.evaluation import wrap
from .sinusoid.exact import ProjectedExact

__all__ = ["MODELS", "TARGETS", "TOYS", "build", "describe", "targets"]

TOYS: tuple[str, ...] = ("sinusoid", "galactic_binary")
TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "sinusoid": {case: MODELS for case in sin_config.CASE_NAMES},
    "galactic_binary": {case: gb_models.TARGET_MODELS[case] for case in gb_config.CASES},
}


def targets(toy: str | None = None) -> tuple[tuple[str, str, str], ...]:
    """Every ``(toy, case, model)``, in ``TARGETS`` order."""
    toys = TOYS if toy is None else (toy,)
    return tuple((t, c, m) for t in toys for c, ms in _toy(t).items() for m in ms)


def _toy(toy: str) -> dict[str, tuple[str, ...]]:
    if toy not in TARGETS:
        raise KeyError(f"unknown toy {toy!r}; registered: {TOYS}")
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
    if toy == "galactic_binary":
        return gb_models.build(case, model, backend=backend, options=options, data_root=data_root)
    resolved = resolve(toy, model, backend, options)
    s = sin_data.load(case, data_root=data_root)
    exact = (ProjectedExact.load(s.cfg, data_root=data_root, algorithm=resolved.correction)
             if model == "exact" else None)
    return wrap(sin_models.likelihood(model, s.cfg, s.d, resolved=resolved, exact=exact), model, resolved)


def describe(toy: str, case: str, model: str) -> dict[str, Any]:
    """What a run of ``model`` analyses (spec §5.3; the WT oracle's counts / lineage C's
    ``matrix.describe``, pinned by tests)."""
    _check(toy, case, model)
    if toy == "galactic_binary":
        return _describe_gb(case, model)
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


def _describe_gb(case: str, model: str) -> dict[str, Any]:
    """Lineage C ``matrix.describe`` (``toy_models/galactic_binary/matrix.py:1082-1104``) key
    for key, plus ``toy``/``case``/``model``/``params``; from the shipped prepared file only
    (the pad's mask rows are counted from ``rho_pad``, not built)."""
    a = PreparedCase.load(case).arrays
    NT = int(a["NT"])
    out: dict[str, Any] = {"toy": "galactic_binary", "case": case, "model": model,
                           "params": gb_config.parameter_names(model), "NT": NT}
    base, edge = split_model(model)
    if model in ("whittle_raw", "exact"):
        out.update(P=int(a["freq_n"]), grid="Fourier bins")
        return out
    if model == "whittle_cut":
        out.update(P=int(a["crop_n"]), grid="Fourier bins of the cropped record")
        return out
    nb = int(a["n_bands"])
    lead, trail = gb_config.CUT[case]
    if edge == "pad":
        NTp = NT + gb_config.PAD_COLUMNS
        h, k0, n = NT // 2, int(a["store_offset"]), int(a["store_n"])
        n_R = len(range(k0 // h, -(-(k0 + n) // h) + 1))
        rho = np.asarray(a["rho_pad"], float)
        out.update(P=nb * NT, NT=NTp, grid=f"{nb} bands x {NTp} columns", covariance="M Lambda M",
                   mask_rows=int(np.sum(1.0 - rho > gb_config.PAD_EDGE_TOL)),
                   lambda_cells=f"{n_R} bands (store) x {NTp} columns (padded grid)")
    else:
        P = nb * (NT - lead - trail) if edge == "cut" else nb * NT
        out.update(P=P, grid=f"{nb} bands x {NT} columns")
        if edge == "exact":
            out.update(exact_rows=nb * (lead + trail), exact_columns="all")
    if base == "tri":
        out["lags"] = "all"
    return out
