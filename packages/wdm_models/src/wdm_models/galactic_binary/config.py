r"""The galactic-binary toy's fixed settings: the paper subset of the legacy
``toy_models/galactic_binary/config.py`` and ``matrix.py`` constants, verbatim (spec §5.2).

One GB injection at 1 mHz in the three WDM bands around it, at three durations
(``gb_3mo`` / ``gb_6mo`` / ``gb_12mo``: T = 91.25 / 182.5 / 365 d, ``DT = 5`` s).  Noise:
:math:`S_c(f, t) = S_{\rm inst}(f) + g_c(t) S_{\rm fg}(f)` (additive: only the foreground is
modulated), :math:`\ln g_c = \sum_k \mu_{ck} e_k(t)` with ``e`` the orthonormalised
``[1, cos, sin, cos 2, sin 2]`` of the 365-d period (``envelope``).

Constants (each pinned by a test against ``golden/galactic_binary/products.json``
``constants``, which the frozen code wrote):

* ``ANALYSIS_NT`` -- Sec. VII A's tiling, a fixed value (never recomputed: user ruling
  2026-09-28); ``CUT`` -- the 0.1% wrap rule's lead/trail columns (none at 12 months, where
  the envelope is periodic); ``PAD_COLUMNS`` / ``PAD_EDGE_TOL`` -- the pad models;
* ``N_BANDS`` WDM bands per window, ``NT_BAND`` the fixed band of ``prepare.calc_Nt``,
  ``C_F``, ``C_T`` the atom's kernel moments (Sec. VII A);
* ``TEMPLATE_BW`` -- the inference template bandwidth; ``MU_BOX`` -- the orthonormal
  envelope prior; ``GLASS_PARAMS`` / ``ENV_NAMES`` -- the sampled columns; ``PERIODIC`` --
  the wrapped source angles (as GLASS wraps them).

``TEMPLATE_SUMS`` and ``SNR_FULL`` are the spec's §5.2 literals (gbonfly at ``TEMPLATE_BW``,
prepared units): the waveform's construction guard and ``prepare.snr_full`` check against
them without reading ``golden/``.
"""

from __future__ import annotations

from types import MappingProxyType

from ..common.params import check_model, is_whittle

__all__ = ["ANALYSIS_NT", "CASES", "CUT", "C_F", "C_T", "DT", "ENVELOPE_PERIOD", "ENV_NAMES",
           "GHAT_L", "GLASS_PARAMS", "MONTHS", "MU_BOX", "NOISE_MOVE_FRACTION", "NT_BAND",
           "N_BANDS", "N_HARM", "PAD_COLUMNS", "PAD_EDGE_TOL", "PERIODIC", "SNR_FULL",
           "TEMPLATE_BW", "TEMPLATE_SUMS", "check_case", "is_stationary",
           "parameter_names"]

CASES: tuple[str, ...] = ("gb_3mo", "gb_6mo", "gb_12mo")
MONTHS = MappingProxyType({"gb_3mo": 3, "gb_6mo": 6, "gb_12mo": 12})
ANALYSIS_NT = MappingProxyType({"gb_3mo": 240, "gb_6mo": 438, "gb_12mo": 900})
CUT = MappingProxyType({"gb_3mo": (5, 4), "gb_6mo": (5, 4), "gb_12mo": (0, 0)})
PAD_COLUMNS = 16
PAD_EDGE_TOL = 1e-8
DT = 5.0
#: The Sangria record length, exactly 365 d: g is periodic over the 12-month record.  The
#: binary_galactica orbit keeps its own sidereal year; only the noise envelope uses this.
ENVELOPE_PERIOD = 365.0 * 86400.0
N_HARM = 2
#: Decimation length of the envelope transform g-hat.
GHAT_L = 65536
N_BANDS = 3
#: The band (Hz) on which ``prepare.calc_Nt`` fits the envelope for NT* (fixed before any tiling).
NT_BAND = (0.98e-3, 1.02e-3)
TEMPLATE_BW = 4096
MU_BOX = (-8.0, 8.0)
C_F, C_T = 0.208, 0.639
GLASS_PARAMS = ("f0T", "cos_colat", "lon", "lnA", "cos_iota", "psi", "phi0", "fdotT2")
ENV_NAMES = ("logg_mean", "ann_c", "ann_s", "semi_c", "semi_s")
#: Periodic source coordinates (index -> (low, high)), as GLASS wraps them.
PERIODIC = MappingProxyType({2: (0.0, 6.283185307179586), 5: (0.0, 3.141592653589793),
                             6: (0.0, 6.283185307179586)})
NOISE_MOVE_FRACTION = 0.1
#: Spec §5.2: sum|h|^2 and Re<z, h> of the injection template (gbonfly = binary_galactica
#: ``backend="fourier"``, BW 4096, prepared units) on the analysis window and on the store.
TEMPLATE_SUMS = MappingProxyType({
    "gb_3mo": MappingProxyType({"window": (47.0844510309, 44.6213355729),
                                "store": (47.0992463067, 44.6206986397)}),
    "gb_6mo": MappingProxyType({"window": (75.3784950251, 78.7189953021),
                                "store": (75.4114941003, 78.6860544692)}),
    "gb_12mo": MappingProxyType({"window": (160.9964196341, 167.5473945629),
                                 "store": (160.9964198590, 167.5474286859)})})
#: Spec §5.2: the injection's optimal SNR under the exact model at ``mu_fit`` (the prepared
#: reports' ``snr_full``; ``prepare.snr_full`` reproduces them bit for bit).
SNR_FULL = MappingProxyType({"gb_3mo": 12.24709552247082, "gb_6mo": 14.522176865090799,
                             "gb_12mo": 20.25921338411754})


def check_case(case: str) -> str:
    """``case`` if it is one of :data:`CASES`, else ``KeyError`` naming them."""
    if case not in CASES:
        raise KeyError(f"unknown galactic-binary case {case!r}; the cases are {CASES}")
    return case


def is_stationary(model: str) -> bool:
    """The Whittle models: one stationary log level per channel instead of the envelope."""
    return is_whittle(model)


def parameter_names(model: str) -> tuple[str, ...]:
    """Sampler columns: the GLASS eight, then the noise block per channel (A, then E)."""
    check_model(model)
    if is_stationary(model):
        return GLASS_PARAMS + ("log_gbar_A", "log_gbar_E")
    return GLASS_PARAMS + tuple(f"{n}_{c}" for c in "AE" for n in ENV_NAMES)
