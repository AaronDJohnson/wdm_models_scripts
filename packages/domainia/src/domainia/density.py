"""Gaussian densities with declared constants (spec section 4.7).

A density turns the two numbers a covariance factor gives, the quadratic form
and the log-determinant, into a log-likelihood.  Its constants -- the
log-Jacobian, whether an amplitude ``A`` sits outside the factorisation, and
which normalisation is used -- are *declared* by the model registry (one line
per model, spec section 4.7), never derived.

Every method is the oracle's expression, operand for operand, so the NumPy path
keeps the stored references bitwise where they are bitwise today:

* :meth:`RealGaussian.lnl` with ``amplitude`` set: the last line of
  ``wdm_toys.matrix.marginal_lnlike`` (wdm_paper_2/toy_models/src/wdm_toys/matrix.py:669);
* :meth:`RealGaussian.lnl` without amplitude: ``galactic_binary/matrix.py:919`` and
  ``:922`` (``-0.5 * (q + logdet + P * _LOG_2PI)``), and the WT Whittle forms
  (``likelihood.py:250``, ``matrix.py:898``);
* :meth:`RealGaussian.lnl_expanded`: ``MatrixCache.signal`` / ``EdgeSchurCache.signal``
  (``galactic_binary/matrix.py:991`` and ``:1070``), where the noise state carries
  ``logdet + P ln 2 pi`` (``:983``, ``:1059``);
* :meth:`ComplexGaussian.lnl`: ``likelihoods.lnl_complex`` / ``lnl_frequency``
  (``galactic_binary/likelihoods.py:142-162``): ``-quad - logdet``;
* :meth:`ComplexGaussian.lnl_expanded`: ``fullcache.lnl_from_state``
  (``galactic_binary/fullcache.py:141-144``).

Nothing here imports JAX; the methods take NumPy or JAX scalars alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wdm.backend import Backend, get_backend

#: ``float(np.log(2 pi))``: the constant of every oracle (WT ``_LOG_2PI``, GB ``_LOG_2PI``).
LOG_2PI = float(np.log(2.0 * np.pi))
#: ``float(np.log(pi))``: the canonical complex normalisation (never applied on the golden path).
LOG_PI = float(np.log(np.pi))

__all__ = ["LOG_2PI", "LOG_PI", "ComplexGaussian", "RealGaussian"]


@dataclass(frozen=True)
class RealGaussian:
    """Real Gaussian density ``N(y; 0, A C)`` on ``n_eff`` coordinates.

    ``log_jacobian`` is added last (the sinusoid WDM models declare
    ``0.5 * N * ln(dt)`` with the full ``N = 16384``, also for the cut models).
    ``amplitude`` names the parameter holding ``log10 A`` (``"log10_A_n"``);
    ``None`` means no amplitude outside the factorisation (the GB models).
    """

    log_jacobian: float = 0.0
    amplitude: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.log_jacobian):
            raise ValueError(f"log_jacobian must be finite; got {self.log_jacobian!r}")
        if self.amplitude is not None and not (isinstance(self.amplitude, str) and self.amplitude):
            raise ValueError(f"amplitude must be a parameter name or None; got {self.amplitude!r}")

    def log_norm(self, n_eff: int) -> float:
        """``n_eff ln 2 pi`` (the GB noise states fold it into their log-determinant)."""
        return n_eff * LOG_2PI

    def lnl(self, quad: Any, logdet: Any, n_eff: int, log10_amplitude: Any = None, *,
            be: Backend | None = None) -> Any:
        """``-0.5 (quad/A + logdet + n_eff ln A + n_eff ln 2 pi) + log_jacobian`` with
        ``A = 10**log10_amplitude`` (WT ``marginal_lnlike``), or, without an amplitude,
        ``-0.5 (quad + logdet + n_eff ln 2 pi) + log_jacobian`` (GB matrix models)."""
        if self.amplitude is None:
            if log10_amplitude is not None:
                raise ValueError("this density has no amplitude; pass log10_amplitude=None")
            return -0.5 * (quad + logdet + n_eff * LOG_2PI) + self.log_jacobian
        if log10_amplitude is None:
            raise ValueError(f"this density needs the {self.amplitude!r} value")
        be = be or get_backend(quad, logdet, log10_amplitude)
        A = 10.0 ** log10_amplitude
        return (-0.5 * (quad / A + logdet + n_eff * be.xp.log(A) + n_eff * LOG_2PI)
                + self.log_jacobian)

    def lnl_folded(self, quad: Any, logdet: Any, n_eff: int, log10_amplitude: Any, *,
                   be: Backend | None = None) -> Any:
        """The amplitude folded into each term first, then summed:
        ``-0.5 ((quad/A) + (logdet + n_eff ln A) + n_eff ln 2 pi) + log_jacobian``
        (WT ``FastFullCovariance.lnlike``, ``wdm_toys/likelihood.py:577-584``: the exact
        model's order, which differs from :meth:`lnl` in the association)."""
        if self.amplitude is None:
            raise ValueError("lnl_folded needs an amplitude density")
        if log10_amplitude is None:
            raise ValueError(f"this density needs the {self.amplitude!r} value")
        be = be or get_backend(quad, logdet, log10_amplitude)
        A = 10.0 ** log10_amplitude
        q = quad / A
        ld = logdet + n_eff * be.xp.log(A)
        return -0.5 * (q + ld + n_eff * LOG_2PI) + self.log_jacobian

    def lnl_expanded(self, q0: Any, hy: Any, tt: Any, logdet_norm: Any) -> Any:
        """``-0.5 (q0 - 2 h.y + |L^-1 h|^2 + logdet_norm) + log_jacobian`` (GB
        ``MatrixCache`` / ``EdgeSchurCache``); ``logdet_norm`` already holds
        ``n ln 2 pi``.  There is no amplitude form: no oracle evaluated one."""
        if self.amplitude is not None:
            raise ValueError("the expanded form has no amplitude oracle; use strategy='residual'")
        return -0.5 * (q0 - 2.0 * hy + tt + logdet_norm) + self.log_jacobian


@dataclass(frozen=True)
class ComplexGaussian:
    """Circular complex Gaussian ``CN(r; 0, M)`` without the ``n ln pi`` constant.

    The GB Fourier models (whittle, exact) were sampled as ``-r^H M^-1 r - ln|M|``;
    :meth:`offset_to_canonical` documents the missing constant and is never
    applied on the golden path.  ``include_pi=True`` subtracts it (tests only).
    """

    include_pi: bool = False

    def offset_to_canonical(self, n: int) -> float:
        """What to add to :meth:`lnl` for the canonical density: ``-n ln pi``
        (``0.0`` when ``include_pi`` is already set)."""
        return 0.0 if self.include_pi else -n * LOG_PI

    def lnl(self, quad: Any, logdet: Any, n: int | None = None) -> Any:
        """``-quad - logdet`` (GB ``lnl_complex`` / ``lnl_frequency``)."""
        val = -quad - logdet
        if self.include_pi:
            if n is None:
                raise ValueError("include_pi=True needs n")
            val = val - n * LOG_PI
        return val

    def lnl_expanded(self, q0: Any, hy: Any, tt: Any, logdet: Any, n: int | None = None) -> Any:
        """``-(q0 - 2 Re h^H y + |L^-1 h|^2) - logdet`` (GB ``fullcache.lnl_from_state``)."""
        val = -(q0 - 2.0 * hy + tt) - logdet
        if self.include_pi:
            if n is None:
                raise ValueError("include_pi=True needs n")
            val = val - n * LOG_PI
        return val
