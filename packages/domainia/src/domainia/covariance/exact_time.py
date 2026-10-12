r"""The exact covariance in time (WDM II Eq. 47, :math:`\theta = 0`): ``ExactTimeCovariance``.

.. math::  \tilde\Sigma_{ij} = \sum_{k=0}^{N/2} \eta_k\,\delta f\,
           e^{\bar\alpha_{ij} L_k}\cos 2\pi f_k (t_i - t_j),\qquad
           \bar\alpha_{ij} = \tfrac12(\alpha(t_i) + \alpha(t_j)),

for the log-linear family :math:`S = A_n e^{\alpha(t) L(f)}` (``A_n`` outside).

* ``form="operator"`` -- WT ``fastcov.CovOperator``:
  :math:`\tilde\Sigma x = \sum_p T_p \odot \mathrm{irfft}(g\,E_p \sum_q E_q\,\mathrm{rfft}(T_q x))`
  with exact Bessel coefficients :math:`E_p` of :math:`e^{a u}`, :math:`a = \alpha_d L/4`.
  The endpoint weights are NOT applied: ``irfft`` already counts DC and Nyquist once and
  the interior twice, which is exactly that weight (a factor-2 trap).  The preconditioner
  is the same operator at :math:`(-\alpha_c, -\alpha_d)` scaled by :math:`4\,\delta t^2`.
  ``weyl_logdet`` is the closed-form leading term; :math:`c_N\alpha_d^2` completes it.
* ``form="dense"`` -- WT ``noise.sigma_tilde`` (:math:`O(N^2\log N)`, tests and the
  :math:`c_N` fit), ``voice_factor`` and ``draw`` (cosine voices ``k = 0..N/2``, then
  sine voices ``k = 1..N/2-1``).

Every FFT of the operator goes through ``be.fft_pinned`` (``scipy.fft``, ``workers=4``
on NumPy, as ``CovOperator``; ``jnp.fft`` on JAX).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from wdm.backend import Backend, get_backend, pytree_dataclass
from wdm.grid import endpoint_weights
from wdm.special import chebyshev_basis

from ..linalg import pcg
from .base import CholFactor, Covariance, QuadForm

__all__ = ["CovOperator", "ExactTimeCovariance", "OperatorFactor", "bessel_coefficients"]


@pytree_dataclass(data=("E", "g", "T"), meta=("N",))
@dataclass(frozen=True, eq=False)
class CovOperator:
    """``y = Op x`` (WT ``fastcov.CovOperator``, ``x`` 1-D): ``E (P, K)``, ``g (K,)``,
    ``T (P, N)``."""

    E: Any
    g: Any
    T: Any
    N: int

    def __call__(self, x: Any, be: Backend | None = None) -> Any:
        be = be or get_backend(x, self.E)
        xp = be.xp
        Fp = be.fft_pinned("rfft", self.T * x[None, :], axis=1)
        F = xp.einsum("pk,pk->k", self.E, Fp)
        G = (self.g * F)[None, :] * self.E
        Y = be.fft_pinned("irfft", G, n=self.N, axis=1) * (0.5 * self.N)
        return xp.einsum("pn,pn->n", self.T, Y)


def bessel_coefficients(a: Any, P: int, be: Backend) -> Any:
    """``(P, K)``: ``E_p = (2 - delta_p0) I_p(a)``, the Chebyshev coefficients of ``e^{a u}``
    (WT ``CovOperator``: ``ive(p, a) e^{|a|}``, then ``E[1:] *= 2``)."""
    E = be.bessel_i(P, a)
    return be.set(E, slice(1, None), E[1:] * 2.0, copy=False)


@pytree_dataclass(data=("A", "M", "logdet", "ok"), meta=("tol", "maxit", "on_fail"))
@dataclass(frozen=True, eq=False)
class OperatorFactor:
    """``C = Sigma~`` as an operator: ``solve`` by PCG (``linalg.pcg``), ``logdet`` closed form.

    ``on_fail="raise"`` (WT, eager) raises when PCG does not converge; ``"nan"`` returns a
    NaN solution (the likelihood is then ``-inf``).
    """

    A: Any
    M: Any
    logdet: Any
    ok: Any
    tol: float
    maxit: int
    on_fail: str

    n_u = 0

    def solve(self, b: Any, *, be: Backend | None = None) -> Any:
        """``C^-1 b`` by ``linalg.pcg`` on the rows ``b[None]`` (1-D) or ``b.T`` (``(N, k)``); the
        operator is applied row by row (``be.map``), so a single row is WT ``fastcov.pcg``."""
        be = be or get_backend(b, self.A.E)
        one = b.ndim == 1
        res = pcg(lambda V: be.map(lambda v: self.A(v, be), V), lambda V: be.map(lambda v: self.M(v, be), V),
                  b[None, :] if one else b.T, tol=self.tol, maxit=self.maxit, on_fail=self.on_fail, be=be)
        return res.x[0] if one else res.x.T

    def whiten(self, b: Any, *, be: Backend | None = None) -> Any:
        raise NotImplementedError("an operator covariance has no whitening; use form='solve'")

    def quad(self, b: Any, *, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        if form != "solve":
            raise ValueError("an operator covariance has only form='solve'")
        return b @ self.solve(b, be=be)


class ExactTimeCovariance(Covariance):
    """The exact time-domain covariance of a ``LogLinearPSD`` on ``N`` samples of ``dt``.

    ``nu = (alpha_c, alpha_d)``.  ``c_N`` (the fitted ``alpha_d^2`` coefficient of
    ``log|Sigma~|``, owned by the model layer) completes the operator log-determinant;
    without it :meth:`factor` raises for ``form="operator"``.
    """

    def __init__(self, psd: Any, N: int, dt: float, *, form: Literal["operator", "dense"] = "operator",
                 P: int = 24, c_N: float | None = None, tol: float = 1e-7, maxit: int = 200,
                 on_fail: Literal["raise", "nan"] = "raise") -> None:
        if "operator" not in getattr(psd, "capabilities", frozenset()):
            raise ValueError(f"ExactTimeCovariance needs a log-linear PSD; got {type(psd).__name__}")
        if form not in ("operator", "dense"):
            raise ValueError(f"form must be 'operator' or 'dense'; got {form!r}")
        if on_fail not in ("raise", "nan"):
            raise ValueError(f"on_fail must be 'raise' or 'nan'; got {on_fail!r}")
        if int(N) < 2 or int(N) % 2:
            raise ValueError(f"N must be even and positive; got {N}")
        if not float(dt) > 0.0:
            raise ValueError(f"dt must be positive; got {dt}")
        if int(P) < 1 or not float(tol) > 0.0 or int(maxit) < 1:
            raise ValueError(f"need P >= 1, tol > 0 and maxit >= 1; got P={P}, tol={tol}, maxit={maxit}")
        self.psd, self.N, self.dt, self.form, self.P = psd, int(N), float(dt), form, int(P)
        self.n = self.N
        self.c_N = None if c_N is None else float(c_N)
        self.tol, self.maxit, self.on_fail = float(tol), int(maxit), on_fail
        self.T_obs = self.N * self.dt
        f2 = np.abs(np.fft.fftfreq(self.N, d=self.dt))
        self.S_L = float(np.sum(psd.L(f2)))       # WT FastCovariance._SL
        self.jittable = form == "operator" and on_fail == "nan"

    # -- host constants -------------------------------------------------------------
    def host_constants(self) -> dict[str, Any]:
        f = np.fft.rfftfreq(self.N, d=self.dt)
        out = {"L": np.asarray(self.psd.L(f), float)}
        if self.form == "operator":
            out["T"] = chebyshev_basis(self.N, self.P)
        return out

    # -- the operator ---------------------------------------------------------------
    def operator(self, C: Any, alpha_c: Any, alpha_d: Any, be: Backend, *, scale: float = 1.0,
                 P: int | None = None) -> CovOperator:
        """WT ``CovOperator(cfg, alpha_c, alpha_d, P, T, scale)``; ``P`` (static) at most the
        constructor's ``P`` (the rows of the Chebyshev basis in ``C``)."""
        if self.form != "operator":
            raise ValueError("operator() needs form='operator' (its constants carry the Chebyshev basis)")
        P = self.P if P is None else int(P)
        if not 1 <= P <= self.P:
            raise ValueError(f"P must be in [1, {self.P}]; got {P}")
        L = C["L"]
        E = bessel_coefficients(0.25 * alpha_d * L, P, be)
        g = scale / self.T_obs * be.xp.exp(alpha_c * L)
        return CovOperator(E, g, C["T"][:P], self.N)

    def weyl_logdet(self, nu: Any) -> Any:
        r""":math:`\alpha_c S_L - \alpha_d S_L/(2N) - N\ln(2\,\delta t)` (WT ``_weyl``)."""
        S_L, N = self.S_L, self.N
        return nu[0] * S_L - nu[1] * S_L / (2.0 * N) - N * np.log(2.0 * self.dt)

    def logdet_model(self, nu: Any) -> Any:
        """``weyl_logdet + c_N alpha_d^2`` (WT ``FastCovariance.logdet``)."""
        if self.c_N is None:
            raise ValueError("the operator log-determinant needs c_N (model-layer asset)")
        return self.weyl_logdet(nu) + self.c_N * nu[1] * nu[1]

    # -- dense forms (WT noise.py) ----------------------------------------------------
    def sigma_tilde(self, nu: Any) -> np.ndarray:
        """Exact ``Sigma~ = Sigma / A_n`` ``(N, N)``: WT ``noise.sigma_tilde`` verbatim."""
        N = self.N
        L = np.asarray(self.psd.L(np.fft.rfftfreq(N, d=self.dt)), float)   # = host_constants()["L"]
        w = endpoint_weights(L.size) * (1.0 / self.T_obs)
        w[1:-1] *= 0.5
        s_over_2N = np.arange(2 * N - 1) / (2.0 * N) - 0.5
        i = np.arange(N)
        flat = ((i[:, None] + i[None, :]) * N + np.abs(i[:, None] - i[None, :])).ravel()
        abar = nu[0] + nu[1] * s_over_2N
        A = np.multiply.outer(abar, L)
        np.exp(A, out=A)
        A *= w[None, :]
        R = np.fft.irfft(A, n=N, axis=1)
        Sig = R.ravel()[flat].reshape(N, N)
        Sig *= N
        return Sig

    def voice_factor(self, nu: Any, A_n: float = 1.0) -> np.ndarray:
        """``G`` with ``Sigma = G G^T``: WT ``noise.factor`` verbatim (cosine voices
        ``k = 0..N/2``, then sine voices ``k = 1..N/2-1``)."""
        f = np.fft.rfftfreq(self.N, d=self.dt)
        t = np.arange(self.N) * self.dt
        df = 1.0 / self.T_obs
        c = endpoint_weights(f.size)
        L = np.asarray(self.psd.L(f), float)
        a = nu[0] + nu[1] * (t / self.T_obs - 0.5)
        amp = np.sqrt(A_n * np.exp(np.outer(a, L)) * df * c[None, :])
        ph = 2.0 * np.pi * np.outer(t, f)
        return np.hstack([amp * np.cos(ph), (amp * np.sin(ph))[:, 1:-1]])

    def draw(self, nu: Any, rng: np.random.Generator, A_n: float = 1.0) -> np.ndarray:
        """One realisation (WT ``noise.draw``: normals for the cosine voices first)."""
        G = self.voice_factor(nu, A_n)
        z = rng.standard_normal(G.shape[1])
        return G @ z

    # -- Covariance protocol ----------------------------------------------------------
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return be.asarray(self.sigma_tilde(np.asarray(nu)))

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.form == "dense":
            return CholFactor.build(self._matrix(C, nu, be), impl="scipy", be=be)
        A = self.operator(C, nu[0], nu[1], be)
        M = self.operator(C, -nu[0], -nu[1], be, scale=4.0 * self.dt ** 2)
        logdet = self.logdet_model(nu)
        return OperatorFactor(A, M, logdet, be.xp.isfinite(logdet), self.tol, self.maxit, self.on_fail)
