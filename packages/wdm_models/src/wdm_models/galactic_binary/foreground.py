r"""The galactic foreground spectrum :math:`S_{\rm fg}(f)` in its two forms (spec §5.2
foreground.py).

* :func:`glass_psd` -- GLASS's model (Digman & Cornish 2022; ldasoft
  ``noise/src/glass_noise_model.c``), the form every prepared file and chain used
  (``toy_models/galactic_binary/fit_foreground.py::s_fg_glass``, verbatim, without GLASS's
  :math:`f > 10 f_1` zeroing):

  .. math::  S_{\rm fg}(f) = \tfrac32 A f^{5/3} e^{-(f/f_1)^\alpha}[1 + \tanh((f_k - f)/f_2)],
             \qquad x = (\ln A, \ln f_1, \alpha, \ln f_k, \ln f_2).

* :func:`ms_eq90_psd` -- the manuscript's Eq. (90) (Katz et al. form) in
  :math:`x = f/f_*`:

  .. math::  S_{\rm fg}(f) = 3\tilde A\,x^{-1/3}\sin^2x\;e^{-\tilde s_1 x^\alpha}
             [1 + \tanh(-\tilde s_2(x - x_k))].

:func:`glass_to_ms` maps the first onto the second's parameters (Table III):
:math:`\tilde A = A f_*^{5/3}/2`, :math:`\tilde s_1 = (f_*/f_1)^\alpha`,
:math:`\tilde s_2 = f_*/f_2`, :math:`x_k = f_k/f_*`, :math:`\alpha` unchanged.  The two shapes
then differ only by :math:`\sin^2x/x^2` (0.99909 at 1 mHz), so Eq. (90) is the GLASS form up to
that factor -- an erratum candidate recorded for the user, not a code change: the release keeps
the GLASS form the chains used.
"""

from __future__ import annotations

import numpy as np

from .instrument import F_STAR

__all__ = ["GLASS_NAMES", "GLASS_PRIOR", "GLASS_START", "MS_NAMES", "glass_psd", "glass_to_ms",
           "ms_eq90_psd"]

#: GLASS's coordinates and its MCMC prior box (``glass_noise_sampler.c``; fit_foreground.py).
GLASS_NAMES = ("ln A", "ln f1", "alpha", "ln fk", "ln f2")
GLASS_PRIOR = np.array([[-86.0, -82.0], [np.log(1e-4), np.log(1e-2)], [0.0, 3.0],
                        [np.log(1e-4), np.log(0.1)], [np.log(1e-5), np.log(1e-2)]])
#: GLASS's starting values (``initialize_foreground_model``).
GLASS_START = np.array([np.log(3.86677e-37), np.log(0.00344439), 1.629667,
                        np.log(0.0102644), np.log(4.810781e-4)])
#: The parameters of Eq. (90), Table III's order.
MS_NAMES = ("A_fg", "alpha", "s1", "s2", "x_k")


def glass_psd(f, x):
    r"""GLASS's A/E foreground PSD at ``x = (ln A, ln f1, alpha, ln fk, ln f2)``
    (``fit_foreground.s_fg_glass`` verbatim: the chains' form)."""
    f = np.asarray(f, float)
    A, f1, alpha, fk, f2 = np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), np.exp(x[4])
    return 1.5 * A * f ** (5.0 / 3.0) * np.exp(-(f / f1) ** alpha) * (1.0 + np.tanh((fk - f) / f2))


def glass_to_ms(x, *, f_star: float = F_STAR) -> dict[str, float]:
    """Table III's ``(A_fg, alpha, s1, s2, x_k)`` of the GLASS parameters ``x``."""
    A, f1, alpha, fk, f2 = (float(np.exp(x[0])), float(np.exp(x[1])), float(x[2]),
                            float(np.exp(x[3])), float(np.exp(x[4])))
    return {"A_fg": 0.5 * A * f_star ** (5.0 / 3.0), "alpha": alpha, "s1": (f_star / f1) ** alpha,
            "s2": f_star / f2, "x_k": fk / f_star}


def ms_eq90_psd(f, ms, *, f_star: float = F_STAR):
    """Eq. (90) at ``ms`` (a :func:`glass_to_ms` dict or a sequence in :data:`MS_NAMES` order)."""
    if isinstance(ms, dict):
        ms = tuple(ms[k] for k in MS_NAMES)
    A_fg, alpha, s1, s2, x_k = (float(v) for v in ms)
    x = np.asarray(f, float) / f_star
    return (3.0 * A_fg * x ** (-1.0 / 3.0) * np.sin(x) ** 2 * np.exp(-s1 * x ** alpha)
            * (1.0 + np.tanh(-s2 * (x - x_k))))
