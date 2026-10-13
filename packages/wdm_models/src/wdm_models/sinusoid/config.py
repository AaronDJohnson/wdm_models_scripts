r"""The sinusoid toy's configuration and cases (spec §5.1; WDM II Sec. VII B).

Signal :math:`h(t) = A_h\cos(2\pi f_0 t + \phi_0)` on :math:`t_j = j\,\delta t`; noise with
one-sided PSD :math:`S(f, t) = A_n (\max(f, f_{\min})/f_{\rm ref})^{\alpha(t)}`,
:math:`\alpha(t) = \alpha_c + \alpha_d (t/T - 1/2)`, :math:`T = N\,\delta t`.

The three cases are WT ``config.CONFIGS["manuscript_{low,mid,high}"]``
(``wdm_paper_2/toy_models/src/wdm_toys/config.py:369-377``): the manuscript grid and noise
(:math:`N = 16384`, :math:`\delta t = 2^{-10}` s, tiling :math:`64 \times 256`,
:math:`A_n = 400`, :math:`\alpha_c = -2`, :math:`\alpha_d = 1.2`, :math:`f_{\rm ref} = 1`,
:math:`f_{\min} = 4` Hz) with the source at 33 / 137 / 385 Hz and :math:`A_h` rescaled to an
optimal SNR of 15 (``0.5 * 15 / SNR(A_h = 0.5)``, kept here as the literal floats).  The
chirp (``fit_fdot``) is not part of the release (spec §0 rule 7) and is refused.

Edge constants (all pinned by tests):

* ``CUT = (6, 5)`` -- every column where some atom wraps more than ``WRAP_TOL = 1e-3`` of its
  energy across the record boundary, ``wdm.evolutionary.wrap_cut_columns(64, 256, 2**-10)``
  (worst kept wraps 0.11200% in column 5 and 0.11708% in column Nt - 5 are the dropped ones);
* ``EDGE_EXACT = (8, 7)`` -- the edge-exact models' exact columns (WT ``matrix.EDGE_START/END``);
* ``PAD_COLUMNS = 16`` -- zero columns appended by the padded models;
* ``ANALYSIS_FMIN = 4.0`` Hz -- the analysis cut (a flat marginal over the 128 Fourier
  coordinates below it, DC and Nyquist), separate from the PSD floor ``f_min``;
* ``EXACT_P = 24``, ``EXACT_TOL = 1e-7`` -- the exact operator's Chebyshev order and PCG
  tolerance (runner ``_FULL_ORDER`` / ``_FULL_TOL``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import wdm

from ..common.params import check_model, is_whittle

__all__ = ["ANALYSIS_FMIN", "CASES", "CASE_NAMES", "CUT", "EDGE_EXACT", "EXACT_P", "EXACT_TOL",
           "PAD_COLUMNS", "SinusoidConfig", "WRAP_TOL", "config"]

ANALYSIS_FMIN = 4.0
CUT = (6, 5)
EDGE_EXACT = (8, 7)
PAD_COLUMNS = 16
WRAP_TOL = 1e-3
EXACT_P = 24
EXACT_TOL = 1e-7


@dataclass(frozen=True)
class SinusoidConfig:
    """One sinusoid configuration (frozen).  Defaults are ``manuscript_mid``.

    ``N``, ``T_obs`` (``= N dt``), ``fs``, ``f_nyq``, ``DF`` and ``DT`` are derived as in WT
    ``ToyConfig.__post_init__`` (``DF``/``DT`` through ``wdm.delta_F``/``wdm.delta_T``).
    """

    Nf: int = 64
    Nt: int = 256
    dt: float = 2.0 ** -10
    A_h: float = 0.30421576834679953
    f_0: float = 137.0
    phi_0: float = 0.6
    A_n: float = 400.0
    alpha_c: float = -2.0
    alpha_d: float = 1.2
    f_ref: float = 1.0
    f_min: float = 4.0
    fit_fdot: bool = False

    N: int = field(init=False)
    T_obs: float = field(init=False)
    fs: float = field(init=False)
    f_nyq: float = field(init=False)
    DF: float = field(init=False)
    DT: float = field(init=False)

    def __post_init__(self) -> None:
        wdm.validate_grid(self.Nf, self.Nt)
        wdm.validate_dt(self.dt)
        if self.fit_fdot:
            raise ValueError("the release sinusoid is fixed-frequency (A_h, f_0, phi_0); the chirp "
                             "(fit_fdot) is not one of the 15 paper models (spec §0 rule 7)")
        set_ = object.__setattr__
        set_(self, "N", self.Nf * self.Nt)
        set_(self, "T_obs", self.Nf * self.Nt * self.dt)
        set_(self, "fs", 1.0 / self.dt)
        set_(self, "f_nyq", 0.5 / self.dt)
        set_(self, "DF", wdm.delta_F(self.Nf, self.dt))
        set_(self, "DT", wdm.delta_T(self.Nf, self.dt))
        if not self.A_n > 0.0:
            raise ValueError(f"A_n must be positive; got {self.A_n}")
        if not self.f_ref > 0.0:
            raise ValueError(f"f_ref must be positive; got {self.f_ref}")
        if not 0.0 < self.f_min < self.f_nyq:
            raise ValueError(f"f_min must lie in (0, f_Nyq) = (0, {self.f_nyq}); got {self.f_min}")

    # -- grids ------------------------------------------------------------------------------
    def grid(self) -> wdm.Grid:
        return wdm.Grid(self.Nf, self.Nt, self.dt)

    def times(self) -> np.ndarray:
        """``t_j = j dt`` (WT ``ToyConfig.times``)."""
        return np.arange(self.N) * self.dt

    # -- the injection --------------------------------------------------------------------
    @property
    def log10_A_n(self) -> float:
        return float(np.log10(self.A_n))

    def truth(self, model: str) -> np.ndarray:
        """The injected parameters in ``model``'s layout: ``(A_h, f_0, phi_0, log10 A_n, alpha)``
        for the Whittle models (``alpha = alpha_c``, the midpoint index), else
        ``(A_h, f_0, phi_0, log10 A_n, alpha_c, alpha_d)``."""
        check_model(model)
        head = [self.A_h, self.f_0, self.phi_0, self.log10_A_n, self.alpha_c]
        return np.array(head if is_whittle(model) else head + [self.alpha_d], dtype=np.float64)


CASES: dict[str, SinusoidConfig] = {
    "manuscript_low": SinusoidConfig(A_h=1.6322595829099906, f_0=33.0),
    "manuscript_mid": SinusoidConfig(A_h=0.30421576834679953, f_0=137.0),
    "manuscript_high": SinusoidConfig(A_h=0.08729868783526586, f_0=385.0),
}
CASE_NAMES: tuple[str, ...] = tuple(CASES)


def config(case: str) -> SinusoidConfig:
    """The configuration of ``case`` (``KeyError`` naming the cases otherwise)."""
    try:
        return CASES[case]
    except KeyError:
        raise KeyError(f"unknown sinusoid case {case!r}; the cases are {CASE_NAMES}") from None
