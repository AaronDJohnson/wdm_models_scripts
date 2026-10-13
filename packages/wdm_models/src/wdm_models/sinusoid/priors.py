r"""The sinusoid prior table: the single source of every bound (spec §5.1 priors.py).

Uniform box priors, the manuscript spectral-study table (WT ``priors.for_model`` with
``cfg.frequency_study``: ``STUDY_SOURCE_BLOCK`` + ``MANUSCRIPT_NOISE_*``,
``wdm_paper_2/toy_models/src/wdm_toys/priors.py:229-260``), which the runner hard-coded again as
``_DRIFT_BOUNDS`` / ``_FREQUENCY_BOUNDS`` (``toy_models/sinusoid/gpu/run.py:40-48``) and every
final run recorded in its ``run_meta.json``:

==============  ==================  =========================================
parameter       prior               note
==============  ==================  =========================================
``A_h``         U[0, 3]
``f_0``         U[16, 496] Hz       common spectral-study box
``phi_0``       U[-pi, pi)          periodic (wrapped, never rejected)
``log10_A_n``   U[1, 4]             the amplitude, outside every factorisation
``alpha``       U[-4, 0]            Whittle models only (= ``alpha_c``'s box)
``alpha_c``     U[-4, 0]
``alpha_d``     U[-2, 2]
==============  ==================  =========================================

:func:`noise_bounds` -- ``[[alpha_c lo, hi], [alpha_d lo, hi]]`` -- is part of the bandcov
artifact's identity, so the exact model's interpolant only loads while these floats are
unchanged (WT ``bandcov.noise_bounds``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from domainia.likelihood import Params

from ..common.params import is_whittle

__all__ = ["Bound", "NOISE_DRIFT", "NOISE_WHITTLE", "PERIODIC", "PHASE_INDEX", "SOURCE", "bounds",
           "in_prior", "names", "noise_bounds", "params", "periodic", "table", "wrap"]


@dataclass(frozen=True)
class Bound:
    """One uniform interval ``[lo, hi]``; ``periodic`` coordinates wrap into ``[lo, hi)``."""

    name: str
    lo: float
    hi: float
    periodic: bool = False


SOURCE: tuple[Bound, ...] = (
    Bound("A_h", 0.0, 3.0),
    Bound("f_0", 16.0, 496.0),
    Bound("phi_0", -math.pi, math.pi, periodic=True),
    Bound("log10_A_n", 1.0, 4.0),
)
NOISE_WHITTLE: tuple[Bound, ...] = (Bound("alpha", -4.0, 0.0),)
NOISE_DRIFT: tuple[Bound, ...] = (Bound("alpha_c", -4.0, 0.0), Bound("alpha_d", -2.0, 2.0))
PHASE_INDEX = 2
#: The wrapped coordinates of every sinusoid layout, ``{2: (-pi, pi)}`` (read-only; the phase
#: sits at index 2 in both the Whittle and the drift layouts).
PERIODIC = MappingProxyType({PHASE_INDEX: (-math.pi, math.pi)})


def table(model: str) -> tuple[Bound, ...]:
    """The bounds of ``model``'s parameters, in its row order."""
    return SOURCE + (NOISE_WHITTLE if is_whittle(model) else NOISE_DRIFT)


def names(model: str) -> tuple[str, ...]:
    return tuple(b.name for b in table(model))


def bounds(model: str) -> np.ndarray:
    """``(ndim, 2)`` float64 ``[lo, hi]`` rows (the runner's ``prior_bounds``)."""
    return np.array([[b.lo, b.hi] for b in table(model)], dtype=np.float64)


def periodic(model: str) -> dict[int, tuple[float, float]]:
    """``{index: (lo, hi)}`` of the wrapped coordinates (``{2: (-pi, pi)}``)."""
    return {i: (b.lo, b.hi) for i, b in enumerate(table(model)) if b.periodic}


def noise_bounds() -> np.ndarray:
    """``np.asarray([[-4., 0.], [-2., 2.]])``: the ``(alpha_c, alpha_d)`` box (a fresh copy)."""
    return np.asarray([[b.lo, b.hi] for b in NOISE_DRIFT], dtype=np.float64)


def params(model: str) -> Params:
    """The ``domainia`` row layout: source ``(A_h, f_0, phi_0)``; noise ``(log10_A_n, alpha)``
    for the Whittle forms, ``(alpha_c, alpha_d)`` otherwise (``log10_A_n`` is then the
    density's amplitude, read by name)."""
    n = names(model)
    return Params(n, (0, 1, 2), (3, 4) if is_whittle(model) else (4, 5))


def _rows(model: str, theta: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    ndim = len(table(model))
    if X.ndim != 2 or X.shape[1] != ndim:
        raise ValueError(f"{model}: expected rows of shape (R, {ndim}); got {np.shape(theta)}")
    return X


def wrap(model: str, theta: np.ndarray) -> np.ndarray:
    """A copy with the periodic coordinates wrapped into ``[lo, hi)`` (runner ``_wrap_phase``)."""
    X = np.array(_rows(model, theta), copy=True)
    for i, (lo, hi) in periodic(model).items():
        X[:, i] = lo + np.mod(X[:, i] - lo, hi - lo)
    return X


def in_prior(model: str, theta: np.ndarray) -> np.ndarray:
    """``(R,)`` bool: finite and inside the box after wrapping (runner ``_prior_valid``)."""
    X = wrap(model, theta)
    b = bounds(model)
    return np.isfinite(X).all(axis=1) & (X >= b[:, 0]).all(axis=1) & (X <= b[:, 1]).all(axis=1)
