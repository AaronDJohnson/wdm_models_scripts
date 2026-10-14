r"""The galactic-binary priors (spec §5.2 priors.py; WDM II Table IV).

* **Source** (the GLASS eight, ``config.GLASS_PARAMS``): binary_galactica's GLASS box on the
  source band's nominal width, :func:`source_prior` =
  ``bg.ucb_prior(T=T, fmin=lo * df, fmax=(hi - 1) * df)`` with ``[lo, hi)`` the band's bins
  (``df = 1/T`` exactly).  It equals the prepared files' ``prior_lo/hi`` bit for bit
  (``ln A`` in ``[-54, -46]``; ``f0 T`` the band; ``fdot T^2`` GLASS's empirical envelope;
  the cosines in ``[-1, 1]``; ``lon``, ``psi``, ``phi0`` periodic).  The GLASS SNR >= 5
  rejection is not applied (as in the chains).
* **Noise**: :func:`noise_box` -- ``mu ~ U[-8, 8]`` per orthonormal envelope coefficient
  (5 per channel, A then E), or one log level per channel for the Whittle models.
* :func:`wrap_source` folds the periodic angles into range (as GLASS does) and
  :func:`source_ok` is the row's in-box test -- the GB likelihoods return ``-inf`` outside it,
  which is how the stored chain rows were computed (legacy ``likelihoods.wrap_source`` /
  ``source_ok``, verbatim, on any backend).

binary_galactica is imported only inside :func:`source_prior` (it is JAX-only; importing this
module never imports JAX).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from domainia.likelihood import Params
from wdm.backend import Backend, get_backend

from ..common.params import check_model
from .config import ENV_NAMES, GLASS_PARAMS, MU_BOX, PERIODIC, is_stationary, parameter_names

__all__ = ["bounds", "names", "noise_box", "params", "periodic", "source_ok", "source_prior", "wrap_source"]


def source_prior(prep):
    """``binary_galactica.ucb_prior`` on ``prep.source_band`` (a ``UCBPrior``; ``.lo``/``.hi``
    are the 8 source bounds; host NumPy float64, whatever JAX's x64 flag)."""
    import binary_galactica as bg

    lo, hi = prep.source_band
    return bg.ucb_prior(T=prep.T, fmin=lo * prep.df, fmax=(hi - 1) * prep.df)


def noise_box(model: str) -> np.ndarray:
    """``(k, 2)`` ``[lo, hi]`` of the noise block: ``MU_BOX`` for every entry (10 envelope
    coefficients, or 2 Whittle log levels)."""
    check_model(model)
    k = 2 if is_stationary(model) else 2 * len(ENV_NAMES)
    return np.tile(np.array([MU_BOX], dtype=np.float64), (k, 1))


def bounds(prep, model: str) -> np.ndarray:
    """``(ndim, 2)``: the prepared source box (``prior_lo/hi``) then :func:`noise_box`."""
    src = np.stack([np.asarray(prep.prior_lo, float), np.asarray(prep.prior_hi, float)], axis=1)
    return np.concatenate([src, noise_box(model)])


def names(model: str) -> tuple[str, ...]:
    """The sampler columns of ``model``: ``config.parameter_names`` (GLASS eight, then
    ``log_gbar_{A,E}`` or ``<env>_{A,E}``)."""
    return parameter_names(model)


def periodic(model: str) -> dict[int, tuple[float, float]]:
    """``{index: (lo, hi)}`` of the wrapped source coordinates (lon, psi, phi0)."""
    check_model(model)
    return dict(PERIODIC)


def params(model: str) -> Params:
    """The ``domainia`` row layout: source ``0..7``, noise ``8..`` (A block then E block)."""
    names = parameter_names(model)
    return Params(names, tuple(range(len(GLASS_PARAMS))), tuple(range(len(GLASS_PARAMS), len(names))))


def wrap_source(p8: Any, *, be: Backend | None = None) -> Any:
    """``p8`` with lon, psi and phi0 folded into ``[lo, hi)`` (legacy ``wrap_source``)."""
    be = be or get_backend(p8)
    xp = be.xp
    numpy = be.name == "numpy"
    out = np.array(p8, dtype=np.float64, copy=True) if numpy else be.asarray(p8)
    for i, (lo, hi) in PERIODIC.items():
        out = be.set(out, i, lo + xp.mod(p8[i] - lo, hi - lo), copy=not numpy)
    return out


def source_ok(p8: Any, lo: Any, hi: Any, *, be: Backend | None = None) -> Any:
    """Finite and inside ``[lo, hi]`` after :func:`wrap_source` (legacy ``source_ok``)."""
    be = be or get_backend(p8)
    xp = be.xp
    w = wrap_source(p8, be=be)
    return xp.all(xp.isfinite(w)) & xp.all(w >= lo) & xp.all(w <= hi)
