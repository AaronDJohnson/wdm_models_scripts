r"""The annual noise envelope (spec §5.2 envelope.py; legacy ``galactic_binary/envelope.py``).

:math:`\ln g(t) = \sum_k \mu_k e_k(t)` with ``e`` the constant plus the 1/yr and 2/yr harmonics
of :data:`config.ENVELOPE_PERIOD` (365 d), Gram--Schmidt orthonormalised on the record
:math:`[0, T)` under the continuous mean (``L = 2**20`` points), so ``mu[0]`` is
:math:`\langle\ln g\rangle` and ``e = C @ raw`` with ``C`` lower triangular.

* Sampling is always in the orthonormal :math:`\mu \sim U[-8, 8]` (``config.MU_BOX``).
* The manuscript's coefficients (Eq. 91, Table III: :math:`a_0, a_1, b_1, a_2, b_2` of
  ``[1, cos, sin, cos 2, sin 2]``) are :func:`physical_coefficients` = ``C.T @ mu``.
* :func:`prior_half_widths` -- ``8 sum_j |C_ji|`` per physical coefficient -- bounds the box
  the :math:`\mu` prior maps to; its minimum (:func:`prior_half_width`, 122 / 21.4 / 8 at
  3 / 6 / 12 months) is Table IV's :math:`X` (the true prior is the image parallelotope).
* :func:`ghat` -- :math:`\hat g[q] = \frac1L\sum_i g(t_i)e^{-2\pi iqi/L}` on the
  ``GHAT_L = 65536`` decimation grid (:func:`ghat_grid`), with the lag guard
  :math:`|q| \le L/2` the legacy ``likelihoods.ghat_at`` skipped.

:class:`AnnualEnvelope` is the ``domainia.spectra.Envelope`` the GB likelihoods compose
(``basis(t)`` host NumPy, ``basis_L`` the decimation-grid basis, ``ghat``/``log_g`` on any
backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from wdm.backend import Backend, get_backend

from .config import ENVELOPE_PERIOD, GHAT_L, MU_BOX, N_HARM

__all__ = ["AnnualEnvelope", "basis", "ghat", "ghat_grid", "orthonormal_coefficients",
           "physical_coefficients", "prior_half_width", "prior_half_widths", "raw_basis"]


def raw_basis(t, n_harm: int = N_HARM, period: float = ENVELOPE_PERIOD) -> np.ndarray:
    """``[1, cos w t, sin w t, cos 2w t, sin 2w t, ...]``, shape ``(1 + 2 n_harm, t.size)``."""
    t = np.asarray(t, float)
    rows = [np.ones_like(t)]
    for h in range(1, n_harm + 1):
        w = 2.0 * np.pi * h / period
        rows += [np.cos(w * t), np.sin(w * t)]
    return np.stack(rows)


def orthonormal_coefficients(T: float, n_harm: int = N_HARM, period: float = ENVELOPE_PERIOD,
                             L: int = 1 << 20) -> np.ndarray:
    """``C`` with ``e = C @ raw``: e_0 = 1, every other e_k zero-mean, unit-RMS, orthogonal."""
    t = np.arange(L) * (T / L)
    B = raw_basis(t, n_harm, period)
    n = B.shape[0]
    C = np.zeros((n, n))
    C[0, 0] = 1.0
    E = [np.ones(L)]
    for i in range(1, n):
        c = np.zeros(n)
        c[i] = 1.0
        v = B[i].copy()
        for j, e in enumerate(E):
            p = float(np.mean(v * e) / np.mean(e * e))
            v -= p * e
            c -= p * C[j]
        s = float(np.sqrt(np.mean(v * v)))
        if not s > 0:
            raise ValueError(f"harmonic {i} is degenerate on a {T:.4g} s window")
        E.append(v / s)
        C[i] = c / s
    return C


def basis(t, T: float, n_harm: int = N_HARM, period: float = ENVELOPE_PERIOD) -> np.ndarray:
    """Orthonormal envelope basis ``(1 + 2 n_harm, t.size)`` at times ``t`` (seconds)."""
    return orthonormal_coefficients(T, n_harm, period) @ raw_basis(t, n_harm, period)


def ghat_grid(T: float, n_harm: int = N_HARM, L: int = GHAT_L,
              period: float = ENVELOPE_PERIOD) -> np.ndarray:
    """The basis on the decimation grid ``t_i = i T / L`` used for g-hat (``prep.basis_L``)."""
    return basis(np.arange(L) * (T / L), T, n_harm, period)


def ghat(mu: Any, basis_L: Any, lags: Any, *, be: Backend | None = None) -> Any:
    r""":math:`\hat g` at the static integer ``lags`` (``rfft(exp(mu @ basis_L)) / L``; negative
    lags by Hermitian symmetry).  ``ValueError`` when a lag exceeds the grid's Nyquist ``L/2``."""
    lags = np.asarray(lags)
    L = int(np.shape(basis_L)[1])
    if np.any(np.abs(lags) > L // 2):
        raise ValueError("lag beyond the decimation grid's Nyquist; raise GHAT_L")
    be = be or get_backend(mu)
    xp = be.xp
    R = be.fft.rfft(xp.exp(be.asarray(mu) @ be.asarray(basis_L))) / L
    out = R[np.abs(lags)]
    return xp.where(lags < 0, xp.conj(out), out)


def physical_coefficients(mu, T: float, n_harm: int = N_HARM,
                          period: float = ENVELOPE_PERIOD) -> np.ndarray:
    """Table III's ``(a_0, a_1, b_1, a_2, b_2)``: ``C.T @ mu`` (``mu`` ``(n,)``, or ``(R, n)``
    rows -> ``(R, n)``)."""
    C = orthonormal_coefficients(T, n_harm, period)
    mu = np.asarray(mu, float)
    return C.T @ mu if mu.ndim == 1 else (C.T @ mu.T).T


def prior_half_widths(T: float, n_harm: int = N_HARM, period: float = ENVELOPE_PERIOD,
                      box: float = MU_BOX[1]) -> np.ndarray:
    """``box * sum_j |C_ji|`` per physical coefficient ``i``: the largest ``|a_i|`` the
    orthonormal box ``|mu_j| <= box`` reaches."""
    C = orthonormal_coefficients(T, n_harm, period)
    return np.array([box * float(np.sum(np.abs(C[:, i]))) for i in range(C.shape[1])])


def prior_half_width(T: float, n_harm: int = N_HARM, period: float = ENVELOPE_PERIOD) -> float:
    """Table IV's :math:`X`: ``min_i 8 sum_j |C_ji|`` (122 / 21.4 / 8 at 3 / 6 / 12 months)."""
    return float(min(prior_half_widths(T, n_harm, period)))


class AnnualEnvelope:
    """The ``domainia`` envelope of one case: ``basis(t)`` (host NumPy), ``basis_L`` (the
    ``(n_params, GHAT_L)`` decimation-grid basis, normally the prepared file's), ``log_g`` and
    ``ghat`` on any backend.  ``C`` is built once (``orthonormal_coefficients``, L = 2**20)."""

    def __init__(self, T: float, period: float = ENVELOPE_PERIOD, n_harm: int = N_HARM,
                 basis_L: np.ndarray | None = None) -> None:
        self.T, self.period, self.n_harm = float(T), float(period), int(n_harm)
        self.C = orthonormal_coefficients(self.T, self.n_harm, self.period)
        self.C.setflags(write=False)
        self.n_params = int(self.C.shape[0])
        bL = ghat_grid(self.T, self.n_harm, GHAT_L, self.period) if basis_L is None else basis_L
        self.basis_L = np.array(bL, dtype=float)
        if self.basis_L.ndim != 2 or self.basis_L.shape[0] != self.n_params:
            raise ValueError(f"basis_L must be ({self.n_params}, L); got {self.basis_L.shape}")
        self.basis_L.setflags(write=False)

    def basis(self, t) -> np.ndarray:
        """``C @ raw_basis(t)`` = :func:`basis` (``(n_params, t.size)``, host constant)."""
        return self.C @ raw_basis(t, self.n_harm, self.period)

    def log_g(self, t, mu, be: Backend | None = None):
        be = be or get_backend(mu)
        return be.asarray(mu) @ be.asarray(self.basis(t))

    def ghat(self, lags, mu, be: Backend | None = None):
        return ghat(mu, self.basis_L, lags, be=be)

    def physical(self, mu) -> np.ndarray:
        """``C.T @ mu`` (Table III's coefficients; ``(R, n)`` rows -> ``(R, n)``)."""
        mu = np.asarray(mu, float)
        return self.C.T @ mu if mu.ndim == 1 else (self.C.T @ mu.T).T
