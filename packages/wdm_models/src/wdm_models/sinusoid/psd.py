r"""The sinusoid's noise PSD (spec §5.1 psd.py), A_n outside every factorisation.

.. math::  S(f, t) = A_n\,\bigl(\max(f, f_{\min})/f_{\rm ref}\bigr)^{\alpha(t)},\qquad
           \alpha(t) = \alpha_c + \alpha_d\,(t/T - 1/2),\quad T = N\,\delta t.

:func:`log_f_clip` and :func:`psd` are WT ``noise.log_f_clip`` / ``noise.psd`` verbatim
(``wdm_paper_2/toy_models/src/wdm_toys/noise.py:105-131``); :func:`as_loglinear` is the
``domainia.spectra.LogLinearPSD`` every covariance is built from (unit amplitude, the same
``L(f)`` and ``alpha(t)`` bit for bit).

``T`` is the record length ``N dt``, not ``(N - 1) dt``.  The exact covariance's dense form
(``domainia ExactTimeCovariance.sigma_tilde``) evaluates ``alpha`` at each sample pair's
midpoint through the index ``s/(2N) - 1/2`` (``s = i + j``): that is
``alpha((t_i + t_j)/2)`` with the same ``T``, and it is the only place the pair index appears.
On a padded grid ``alpha`` continues linearly past ``T`` (``Eq45Covariance(t_max=T')``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from domainia.spectra import LogLinearPSD

from .config import SinusoidConfig

__all__ = ["alpha", "as_loglinear", "log_f_clip", "psd"]

_MANUSCRIPT = SinusoidConfig()


def log_f_clip(f: Any, cfg: SinusoidConfig = _MANUSCRIPT) -> np.ndarray:
    r""":math:`L(f) = \log(\max(f, f_{\min}) / f_{\rm ref})`."""
    return np.log(np.maximum(np.asarray(f, dtype=float), cfg.f_min) / cfg.f_ref)


def alpha(t: Any, alpha_c: float, alpha_d: float, cfg: SinusoidConfig = _MANUSCRIPT) -> np.ndarray:
    """The spectral index at time ``t``: ``alpha_c + alpha_d (t/T - 1/2)``, ``T = N dt``."""
    return alpha_c + alpha_d * (np.asarray(t, dtype=float) / cfg.T_obs - 0.5)


def psd(f: Any, t: Any = None, *, cfg: SinusoidConfig = _MANUSCRIPT, A_n: float | None = None,
        alpha_c: float | None = None, alpha_d: float | None = None) -> np.ndarray:
    """One-sided ``S(f, t)``, shape ``shape(t) + shape(f)`` (outer product); ``t=None`` is the
    record midpoint (the stationary index ``alpha_c``).  Parameters default to ``cfg``'s."""
    A_n = cfg.A_n if A_n is None else A_n
    alpha_c = cfg.alpha_c if alpha_c is None else alpha_c
    alpha_d = cfg.alpha_d if alpha_d is None else alpha_d
    L = log_f_clip(f, cfg)
    tt = cfg.T_obs / 2.0 if t is None else t
    a = alpha_c + alpha_d * (np.asarray(tt, dtype=float) / cfg.T_obs - 0.5)
    return A_n * np.exp(np.multiply.outer(a, L))


def as_loglinear(cfg: SinusoidConfig = _MANUSCRIPT) -> LogLinearPSD:
    """The unit-amplitude ``domainia`` form: ``LogLinearPSD(f_floor=f_min, f_ref, T=N dt)``."""
    return LogLinearPSD(f_floor=cfg.f_min, f_ref=cfg.f_ref, T=cfg.T_obs)
