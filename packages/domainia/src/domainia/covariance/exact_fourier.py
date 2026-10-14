r"""The exact covariance of complex Fourier bins (GB, additive foreground): ``ExactFourierCovariance``.

In the prepared units (``z = rfft(x) / sigma_0``) one channel's exact bin covariance is

.. math::  M' = \mathrm{diag}(s_I) + D_{\sqrt{s_G}}\,T(\overline{\hat g})\,D_{\sqrt{s_G}},
           \qquad T_{jl} = \overline{\hat g[l - j]},

with :math:`\hat g[q] = \frac1L \sum_i g(t_i) e^{-2\pi i q i/L}` on the ``L = 65536``
decimation grid (GB ``likelihoods.full_covariance`` / ``ghat_at``).  Unlike ``ghat_at``,
every lag is checked against the grid's Nyquist ``L/2`` once, at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from wdm.backend import NUMPY, Backend, get_backend, require_x64

__all__ = ["ExactFourierAt", "ExactFourierCovariance", "ghat"]


def ghat(mu: Any, basis_L: Any, lags: np.ndarray, be: Backend, *, absl: Any = None, neg: Any = None) -> Any:
    """``ghat`` at the static integer ``lags`` (GB ``likelihoods.ghat_at`` verbatim).  ``absl`` /
    ``neg``: ``|lags|`` and ``lags < 0`` as arrays (constants passed to ``jit``, never literals);
    built from ``lags`` when not given.  The same gather either way."""
    xp = be.xp
    L = basis_L.shape[1]
    R = be.fft.rfft(xp.exp(mu @ basis_L)) / L
    if absl is None or neg is None:
        absl, neg = np.abs(lags), np.asarray(lags) < 0
    out = R[absl]
    return xp.where(neg, xp.conj(out), out)


class ExactFourierCovariance:
    """``M'(mu)`` of one channel on the absolute bins of ``bins`` (an evaluator: complex
    ``(n, n)`` Hermitian; factorise it with ``HermitianCovariance`` or push it to cells
    with ``PushedRows``).

    ``psd`` is a ``SeparablePSD`` with components ``((s_I, None), (s_G, envelope))``;
    ``bins`` a ``FourierDomain`` inside their tabulation; ``envelope.basis_L`` the
    ``(n_env, L)`` envelope basis on the decimation grid.
    """

    def __init__(self, psd: Any, bins: Any, *, channel: int = 0) -> None:
        if "fourier_toeplitz" not in getattr(psd, "capabilities", frozenset()):
            raise ValueError(f"ExactFourierCovariance needs a separable PSD; got {type(psd).__name__}")
        comps = tuple(psd.components)
        if len(comps) != 2 or comps[0][1] is not None or comps[1][1] is None:
            raise ValueError("ExactFourierCovariance needs components ((s_I, None), (s_G, envelope))")
        (tab_I, _), (_, env_G) = comps
        if not 0 <= int(channel) < tab_I.channels:
            raise ValueError(f"channel {channel} is not one of the PSD's {tab_I.channels} channels")
        self.psd, self.bins, self.channel, self.envelope = psd, bins, int(channel), env_G
        self.n = int(bins.n)
        L = int(np.shape(env_G.basis_L)[1])
        if self.n - 1 > L // 2:
            raise ValueError(f"lag {self.n - 1} is beyond the decimation grid's Nyquist {L // 2}; "
                             "raise the envelope's L")

    def host_constants(self) -> dict[str, Any]:
        (tab_I, _), (tab_G, _) = self.psd.components
        sI = np.asarray(tab_I.on(self.bins), float)[self.channel]        # (C, n) -> one channel
        sG = np.asarray(tab_G.on(self.bins), float)[self.channel]
        n = self.n
        lags = np.arange(-(n - 1), n)
        return {"sI": sI, "sG": sG, "basis_L": np.asarray(self.envelope.basis_L, float),
                "lag": np.arange(n)[None, :] - np.arange(n)[:, None] + n - 1,
                "absl": np.abs(lags), "neg": lags < 0}           # constants, not program literals

    def evaluate(self, C: Any, mu: Any, be: Backend) -> Any:
        """``M'(mu)``: GB ``full_covariance`` verbatim (the lag tables from ``C``)."""
        xp = be.xp
        gh = ghat(mu, C["basis_L"], np.arange(-(self.n - 1), self.n), be, absl=C["absl"], neg=C["neg"])
        root = xp.sqrt(C["sG"])
        return xp.diag(C["sI"].astype(complex)) + (root[:, None] * root[None, :]) * xp.conj(gh)[C["lag"]]

    def at(self, mu: Any) -> "ExactFourierAt":
        """``M'(mu)`` at a fixed noise state as ``maps.SupportsDense`` (section D-B), so
        ``FourierToWDM(bins, cells, restrict=True).push(cov.at(mu)).rows(e)`` is the exact
        cell rows ``0.5 Re(U_E M' U^H)`` (spec §4.2).  ``mu`` (``n_env`` envelope
        coefficients of this channel) must be concrete."""
        mu = np.array(mu, dtype=float)
        n_env = int(np.shape(self.envelope.basis_L)[0])
        if mu.shape != (n_env,):
            raise ValueError(f"mu must hold the envelope's {n_env} coefficients; got shape {mu.shape}")
        return ExactFourierAt(self, mu)

    def ghat(self, mu: Any, lags: np.ndarray, be: Backend | None = None) -> Any:
        """``ghat`` at ``lags`` (``|lags| <= L/2``, checked)."""
        be = be or get_backend(mu)
        lags = np.asarray(lags)
        L = int(np.shape(self.envelope.basis_L)[1])
        if np.any(np.abs(lags) > L // 2):
            raise ValueError("lag beyond the decimation grid's Nyquist; raise the envelope's L")
        return ghat(be.asarray(mu), be.asarray(np.asarray(self.envelope.basis_L, float)), lags, be)


@dataclass(frozen=True, eq=False)
class ExactFourierAt:
    """:meth:`ExactFourierCovariance.at`: ``dense(be)`` is ``M'(mu)`` (GB ``full_covariance``).
    The covariance's host constants are built once per backend and cached on it (JAX:
    ``require_x64()``, then one ``jax.device_put``), so every ``at(mu)`` shares them."""

    cov: ExactFourierCovariance
    mu: np.ndarray

    def dense(self, be: Backend | None = None) -> Any:
        be = be or NUMPY
        cache = self.cov.__dict__.setdefault("_at_constants", {})
        if be.name not in cache:
            C = self.cov.host_constants()
            if be.name != "numpy":
                require_x64()
                import jax

                C = jax.device_put(C)
            cache[be.name] = C
        return self.cov.evaluate(cache[be.name], be.asarray(self.mu), be)
