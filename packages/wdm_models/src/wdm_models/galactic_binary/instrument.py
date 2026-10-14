r"""The LISA A/E instrument-noise PSD (WDM II Eq. 87-89), re-homed verbatim from
``wdm_toymodels/galactic_binary/_nora/noise/model.py`` (spec §5.2 instrument.py).

.. math::

   S_{\rm inst}(f) = 8\sin^2x\,[S_{\rm oms}(f)(2+\cos x) + 2S_{\rm acc}(f)(3+2\cos x+\cos2x)],
   \qquad x = f/f_*,\ f_* = c/(2\pi L),

   S_{\rm oms}(f) = S_{\rm oms,d}[1+(2\,{\rm mHz}/f)^4](2\pi f/c)^2,\qquad
   S_{\rm acc}(f) = S_a[1+(0.4\,{\rm mHz}/f)^2][1+(f/8\,{\rm mHz})^4]/((2\pi f)^2c^2).

Every expression keeps the original's operation order (``_nora/constants.py``:
``TWO_PI = 2.0 * np.pi``, ``F_STAR = C_LIGHT / (ARM_LENGTH * TWO_PI)``), so the release PSD is
``np.array_equal`` to the original on every grid the toy uses (pinned against
``golden/galactic_binary/psd_window.npz`` and a fresh lineage-C evaluation).  Only the A/E
instrument part is carried: the Sangria one-year foreground formula, the low-frequency excess
and the 16-parameter packing are not used by the paper (``foreground.glass_psd`` is the
foreground the chains used).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ARM_LENGTH", "C_LIGHT", "F_STAR", "InstrumentNoise", "SANGRIA_NOISE_PARAMS", "TWO_PI",
           "acceleration_noise", "oms_noise", "tdi_instrumental_psd_AE"]

C_LIGHT = 2.99792458e8           # speed of light [m/s]
TWO_PI = 2.0 * np.pi
ARM_LENGTH = 2.5e9               # LISA arm length [m]
#: Transfer frequency f* = c / (2 pi L) ~ 19.09 mHz (``_nora`` operation order).
F_STAR = C_LIGHT / (ARM_LENGTH * TWO_PI)


@dataclass(frozen=True)
class InstrumentNoise:
    """The instrument part of ``_nora``'s ``NoiseParams`` (A and E channels)."""

    S_oms_A: float
    S_a_A: float
    S_oms_E: float
    S_a_E: float


#: Sangria (LDC2a) instrument noise: S_oms,d = 6.241e-23 m^2/Hz, S_a = 5.76e-30 m^2/s^4/Hz
#: (WDM II Table III, "held fixed").
SANGRIA_NOISE_PARAMS = InstrumentNoise(S_oms_A=6.241e-23, S_a_A=5.76e-30,
                                       S_oms_E=6.241e-23, S_a_E=5.76e-30)


def oms_noise(f, S_oms_d: float):
    """OMS displacement noise ``S_oms_d [1 + (2 mHz/f)^4] (2 pi f / c)^2``."""
    ratio = 2.0e-3 / f
    omega_over_c = TWO_PI * f / C_LIGHT
    return S_oms_d * (1.0 + ratio**4) * omega_over_c**2


def acceleration_noise(f, S_a: float):
    """Test-mass acceleration noise ``S_a [1 + (0.4 mHz/f)^2][1 + (f/8 mHz)^4] / ((2 pi f)^2 c^2)``."""
    lo = 0.4e-3 / f
    hi = f / 8.0e-3
    omega = TWO_PI * f
    return S_a * (1.0 + lo**2) * (1.0 + hi**4) / (omega**2 * C_LIGHT**2)


def tdi_instrumental_psd_AE(f, *, S_oms: float = SANGRIA_NOISE_PARAMS.S_oms_A,
                            S_a: float = SANGRIA_NOISE_PARAMS.S_a_A, L: float = ARM_LENGTH):
    """The A (= E) channel instrument PSD at frequencies ``f`` [Hz] (vectorised).

    ``L`` other than :data:`ARM_LENGTH` recomputes ``f* = c / (L 2 pi)`` in the same order.
    """
    f_star = F_STAR if L == ARM_LENGTH else C_LIGHT / (L * TWO_PI)
    x = f / f_star
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    cos_2x = np.cos(2.0 * x)
    return 8.0 * sin_x * sin_x * (
        oms_noise(f, S_oms) * (2.0 + cos_x)
        + 2.0 * acceleration_noise(f, S_a) * (3.0 + 2.0 * cos_x + cos_2x)
    )
