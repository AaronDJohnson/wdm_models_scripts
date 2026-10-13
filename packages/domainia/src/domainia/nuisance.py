r"""Flat marginals over nuisance directions (spec §4.6): the analysis cut below ``f_min``.

* :class:`FourierBandNuisance` -- WT ``band.FourierBand``: the retained real orthonormal
  Fourier coordinates ``fmin <= f < f_Nyq`` (DC and Nyquist always discarded) and the
  discarded ones (16256 kept / 128 discarded on the sinusoid grid).  ``fmin`` is the
  analysis cut, not the PSD floor (both 4 Hz in the paper; kept separate).
* :func:`nuisance_basis` -- WT verbatim: the discarded modes' images kept as given when
  full rank, else an orthonormal basis of their range (SVD, ``rtol = 1e-10``).
* :class:`FlatMarginal` -- WT ``marginal_lnlike`` without the amplitude:
  ``quad = y^T C^-1 y - b^T H^-1 b``, ``logdet = log|C| + log|H|``, ``b = U^T C^-1 y``,
  ``H = U^T C^-1 U``; ``n_u = U.shape[1]`` directions removed.  ``form="solve"`` is the
  WT order (NumPy golden); ``form="whiten"`` the WG forward-only order
  (``H = Z_u^T Z_u``, ``Z_u = L^-1 U``; the JAX tri raw/cut golden); ``form="reduce"`` the
  WG pad order over ``ZeroPadded(form="reduce")`` (``W_u`` the nuisance's extended cells,
  ``G_u = Lq^-1 W_p^T Lambda^-1 W_u``, ``H = W_u^T Lambda^-1 W_u - G_u^T G_u``; the JAX pad
  golden).
* :class:`ProjectedOperatorMarginal` -- WT ``bandcov.BandCovariance`` evaluation path
  (the sinusoid ``exact`` model): projected PCG with a true-residual check, and
  ``log|Sigma~| + correction(nu)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import wdm
from wdm.backend import Backend, get_backend, is_tracer, pytree_dataclass
from wdm.grid import active_mask

from .covariance.base import CholFactor, Covariance, QuadForm, concrete_failure, factor_constants
from .covariance.eq45 import keep_mask

__all__ = ["FlatMarginal", "FourierBandNuisance", "MarginalFactor", "ProjectedFactor", "ProjectedOperatorMarginal",
           "nuisance_basis"]


class FourierBandNuisance:
    """WT ``band.FourierBand`` on ``N`` samples of ``dt`` with the cut at ``fmin > 0``.

    ``keep_rfft`` marks the retained rfft bins; ``n_keep`` / ``n_drop`` count real
    coordinates (two per retained bin).
    """

    def __init__(self, N: int, dt: float, fmin: float) -> None:
        self.N, self.dt, self.fmin = int(N), float(dt), float(fmin)
        if not np.isfinite(self.fmin) or self.fmin <= 0:
            raise ValueError("fmin must be finite and positive")
        f = np.fft.rfftfreq(self.N, d=self.dt)
        self.keep_rfft = (f >= self.fmin) & (f < 0.5 / self.dt)
        # Endpoint frequencies may round below f_nyq for a non-binary dt; DC and Nyquist
        # are real coordinates, excluded by convention.
        self.keep_rfft[[0, -1]] = False
        if not np.any(self.keep_rfft):
            raise ValueError("fmin leaves no interior Fourier frequencies")
        self.n_keep = 2 * int(self.keep_rfft.sum())
        self.n_drop = self.N - self.n_keep

    def discarded_spectra(self) -> np.ndarray:
        """``(n_drop, N/2 + 1)``: rfft of the discarded orthonormal real coordinates."""
        spectra = np.zeros((self.n_drop, self.N // 2 + 1), dtype=complex)
        row = 0
        for k in np.flatnonzero(~self.keep_rfft):
            if k in (0, self.N // 2):
                spectra[row, k] = np.sqrt(self.N)
                row += 1
            else:
                spectra[row, k] = np.sqrt(self.N / 2)
                spectra[row + 1, k] = -1j * np.sqrt(self.N / 2)
                row += 2
        return spectra

    def discarded_basis(self) -> np.ndarray:
        """``(n_drop, N)``: the discarded coordinates as orthonormal time series."""
        return np.fft.irfft(self.discarded_spectra(), n=self.N, axis=-1)

    def project(self, x: Any, be: Backend | None = None) -> Any:
        """Remove the discarded coordinates: ``irfft(rfft(x) * keep)`` (WT ``project``)."""
        be = be or get_backend(x)
        x = be.asarray(x)                   # WT: np.asarray(x)
        if x.shape[-1] != self.N:
            raise ValueError(f"last axis must have length {self.N}")
        spectrum = be.fft.rfft(x, axis=-1)
        return be.fft.irfft(spectrum * be.asarray(self.keep_rfft), n=self.N, axis=-1)

    def push_to_cells(self, cells: Any) -> np.ndarray:
        """``(n_cells, n_drop)``: the discarded basis on ``cells`` of an ``N``-sample grid,
        ``forward_time_many(B)[:, active][:, keep].T`` (WT ``v.map_many(B).T``)."""
        g = cells.grid
        if g.N != self.N or g.dt != self.dt:
            raise ValueError("the cells' grid is not this record's")
        W = wdm.forward_time_many(self.discarded_basis(), g.Nf, g.Nt, g.dt)[:, active_mask(g.Nf, g.Nt)]
        return W[:, keep_mask(cells)].T


def nuisance_basis(u: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """WT ``matrix.nuisance_basis``: ``u`` itself when full rank, else an orthonormal basis of
    its range (a direction the kept data cannot see carries no information, and a flat
    marginal depends only on the range up to a parameter-independent constant)."""
    s = np.linalg.svd(u, compute_uv=False)
    if s.min() > rtol * s.max():
        return u
    U, s, _ = np.linalg.svd(u, full_matrices=False)
    return U[:, s > rtol * s.max()]


@pytree_dataclass(data=("base", "H", "Zu", "Gu", "logdet", "ok"), meta=("form", "n_u"))
@dataclass(frozen=True, eq=False)
class MarginalFactor:
    """A factor with ``n_u`` flat directions marginalised: ``quad`` / ``logdet`` / ``ok``.

    Per-state arrays only: the base factor, ``H``'s Cholesky, ``Zu = L^-1 U``
    (``form="whiten"``) and ``Gu = Lq^-1 W_p^T Lambda^-1 W_u`` (``form="reduce"``).  ``U``
    and ``W_u`` are constants (``C["U"]``, ``C["Wu"]``); ``base`` reads ``C["cov"]``."""

    base: Any
    H: Any
    Zu: Any
    Gu: Any
    logdet: Any
    ok: Any
    form: str
    n_u: int

    def parts(self, y: Any, *, C: Any = None, be: Backend | None = None) -> dict[str, Any]:
        """``quad_C, quad_H, logdet_C, logdet_H`` (the G6 split) in this factor's order."""
        if self.H is None:
            raise ValueError("the factorisation failed (not positive definite): check ok first")
        C = factor_constants(self, C, "parts")
        be = be or get_backend(y, self.H.L)
        if self.form == "reduce":
            # WG _evaluate pad branch: t = Lambda^-1 w, cb = Lq^-1 W_p^T t,
            # quad = w.t - cb.cb - |cH^-1 (W_u^T t - G_u^T cb)|^2
            xp, F, Cc = be.xp, self.base, C["cov"]
            w = F.lift(y, C=Cc, be=be)
            t = F.Lam.solve(w, C=Cc["base"], be=be)
            cb = F.reduce(t, C=Cc, be=be)
            cu = self.H.whiten(C["Wu"].T @ t - self.Gu.T @ cb, be=be)
            return {"quad_C": xp.dot(w, t) - xp.dot(cb, cb), "quad_H": xp.dot(cu, cu),
                    "logdet_C": F.logdet, "logdet_H": self.H.logdet}
        if self.form == "solve":
            Ciy = self.base.solve(y, C=C["cov"], be=be)
            quad_C = y @ Ciy
            b = C["U"].T @ Ciy
        else:
            zy = self.base.whiten(y, C=C["cov"], be=be)
            quad_C = zy @ zy
            b = self.Zu.T @ zy
        return {"quad_C": quad_C, "quad_H": b @ self.H.solve(b, be=be),
                "logdet_C": self.base.logdet, "logdet_H": self.H.logdet}

    def quad(self, y: Any, *, C: Any = None, form: QuadForm | None = None,
             be: Backend | None = None) -> Any:
        """``y^T C^-1 y - b^T H^-1 b`` (WT: ``quad -= b @ cho_solve(cf, b)``)."""
        if form is not None and form != self.form:
            raise ValueError(f"this marginal was factorised for form={self.form!r}")
        p = self.parts(y, C=C, be=be)
        return p["quad_C"] - p["quad_H"]

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        raise NotImplementedError("a marginal exposes quad and logdet only")

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        raise NotImplementedError("a marginal exposes quad and logdet only")


class FlatMarginal(Covariance):
    """``cov`` marginalised over the columns of ``U`` (``(n, n_u)``: the nuisance in the
    data coordinates) under a flat prior.  ``U`` goes through :func:`nuisance_basis`
    (unchanged when full rank, so the raw models are WT's exactly).  ``form`` (module
    docstring): ``"solve"`` (WT), ``"whiten"`` (WG forward-only) or ``"reduce"`` (WG pad: ``cov``
    a ``ZeroPadded(strategy="time_marginal", form="reduce")``)."""

    def __init__(self, cov: Covariance, U: np.ndarray, *,
                 form: Literal["solve", "whiten", "reduce"] = "solve", rtol: float = 1e-10) -> None:
        if form not in ("solve", "whiten", "reduce"):
            raise ValueError(f"form must be 'solve', 'whiten' or 'reduce'; got {form!r}")
        if form == "reduce" and not (getattr(cov, "strategy", None) == "time_marginal"
                                     and getattr(cov, "form", None) == "reduce"):
            raise ValueError("form='reduce' marginalises a ZeroPadded(strategy='time_marginal', form='reduce')")
        U = np.asarray(U, float)
        if U.ndim != 2 or U.shape[0] != cov.n:
            raise ValueError(f"U must be ({cov.n}, n_u); got {U.shape}")
        if U.shape[1] == 0:                 # nuisance_basis needs a column (WT skipped u.shape[1] == 0)
            raise ValueError(f"U must have at least one column; got {U.shape}: use the covariance itself")
        self.cov, self.form = cov, form
        self.cells = getattr(cov, "cells", None)          # the likelihood's map/cells check
        self.U = nuisance_basis(U, rtol)
        self.n = cov.n
        self.n_u = int(self.U.shape[1]) + cov.n_u
        self.jittable = cov.jittable

    def host_constants(self) -> dict[str, Any]:
        if self.form == "reduce":                         # WG Wu: the nuisance on the extended cells
            return {"U": self.U, "Wu": np.ascontiguousarray(self.cov.lift(self.U.T).T)}
        return {"U": self.U}

    def children(self) -> dict[str, Covariance]:
        return {"cov": self.cov}

    def _factor(self, C: Any, nu: Any, be: Backend) -> MarginalFactor:
        """WT ``marginal_lnlike``'s noise-only half: ``H = U^T C^-1 U``, ``cho_factor(0.5 (H + H^T))``."""
        F = self.cov._factor(C["cov"], nu, be)
        U = C["U"]
        if concrete_failure(F.ok):          # WT: LinAlgError in the base -> -inf; H is never formed
            return MarginalFactor(F, None, None, None, float("nan"), False, self.form, self.n_u)
        Zu = Gu = None
        impl = "scipy"
        if self.form == "solve":
            H = U.T @ F.solve(U, C=C["cov"], be=be)
        elif self.form == "whiten":
            Zu = F.whiten(U, C=C["cov"], be=be)
            H = Zu.T @ Zu
        else:                               # WG _factor pad branch (jnp.linalg.cholesky: numpy)
            Wu = C["Wu"]
            LiWu = F.Lam.solve(Wu, C=C["cov"]["base"], be=be)
            Gu = F.reduce(LiWu, C=C["cov"], be=be)
            H = Wu.T @ LiWu - Gu.T @ Gu
            impl = "numpy"
        Hf = CholFactor.build(0.5 * (H + H.T), impl=impl, be=be)
        return MarginalFactor(F, Hf, Zu, Gu, F.logdet + Hf.logdet, F.ok & Hf.ok, self.form, self.n_u)


@pytree_dataclass(data=("A", "M", "logdet", "ok"), meta=("N", "tol", "maxit", "on_fail", "n_drop"))
@dataclass(frozen=True, eq=False)
class ProjectedFactor:
    """WT ``BandCovariance``: ``quad`` by projected PCG with a true-residual check.

    ``quad`` raises (``on_fail="raise"``, WT) or returns NaN (``"nan"``, jit-able; the
    likelihood maps a non-finite value to ``-inf``) when PCG fails.  ``A`` / ``M`` hold the
    per-state Bessel coefficients and weights only; the Chebyshev basis
    (``C["op"]["T"]``) and the retained-bin mask (``C["keep"]``) are constants.
    """

    A: Any
    M: Any
    logdet: Any
    ok: Any
    N: int
    tol: float
    maxit: int
    on_fail: str
    n_drop: int

    @property
    def n_u(self) -> int:
        return self.n_drop

    def project(self, x: Any, keep: Any, be: Backend) -> Any:
        return be.fft.irfft(be.fft.rfft(x, axis=-1) * keep, n=self.N, axis=-1)

    def quad(self, r: Any, *, C: Any = None, form: QuadForm | None = None,
             be: Backend | None = None) -> Any:
        """WT ``BandCovariance.quad`` after its domain check (the model layer owns the box)."""
        from .covariance.exact_time import with_basis
        from .linalg import pcg

        be = be or get_backend(r, self.A.E)
        xp = be.xp
        if form not in (None, "solve"):
            raise ValueError("the projected operator marginal has only form='solve'")
        C = factor_constants(self, C, "quad")
        keep = C["keep"]
        Aop, Mop = with_basis(self.A, C["op"]), with_basis(self.M, C["op"])
        r = be.asarray(r, dtype=float)      # WT: band.project(np.asarray(r, dtype=float))
        if r.shape[-1:] != (self.N,):       # WT band.project; irfft(n=N) would resize silently
            raise ValueError(f"last axis must have length {self.N}")
        r = self.project(r, keep, be)
        if r.shape != (self.N,) or (not is_tracer(r) and not np.all(np.isfinite(be.to_host(r)))):
            raise ValueError("residual must be a finite length-N vector")

        def A(v: Any) -> Any:
            return self.project(Aop(self.project(v, keep, be), be), keep, be)

        def M(v: Any) -> Any:
            return self.project(Mop(self.project(v, keep, be), be), keep, be)

        # One row: WT's 1-D pcg.  linalg.pcg also runs WT's true-residual check
        # (|r - A x| <= max(5 tol, 1e-12) |r|, bandcov.py:198-201): with on_fail="raise" it
        # raises RuntimeError, with "nan" the row is NaN (-inf downstream).
        res = pcg(lambda V: be.map(A, V), lambda V: be.map(M, V), r[None, :], tol=self.tol,
                  maxit=self.maxit, on_fail=self.on_fail, be=be)
        return xp.where(res.converged[0], r @ res.x[0], xp.nan)

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        raise NotImplementedError("the projected marginal exposes quad and logdet only")

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        raise NotImplementedError("the projected marginal exposes quad and logdet only")


class ProjectedOperatorMarginal(Covariance):
    """The sinusoid ``exact`` model: ``ExactTimeCovariance(form="operator")`` restricted to
    the retained Fourier coordinates of ``nuisance`` (WT ``bandcov.BandCovariance``
    evaluation path; the offline ``prepare`` stays in ``scripts/``).

    ``logdet = op_cov.logdet_model(nu) + correction(nu, be)``: ``correction`` is the
    validated log-det correction of the discarded block (the model layer's bandcov
    ``Artifact`` interpolant; identity ``log|Q C Q^T| = log|C| + log|B C^-1 B^T|``).
    ``n_u = n_drop`` (the data are the ``N`` samples; ``n_eff = n_keep``).
    """

    def __init__(self, op_cov: Any, nuisance: FourierBandNuisance,
                 correction: Callable[[Any, Backend], Any]) -> None:
        if getattr(op_cov, "form", None) != "operator":
            raise ValueError("ProjectedOperatorMarginal needs ExactTimeCovariance(form='operator')")
        if nuisance.N != op_cov.N or nuisance.dt != op_cov.dt:
            raise ValueError("the nuisance and the operator differ in N or dt")
        self.op, self.nuisance, self.correction = op_cov, nuisance, correction
        self.n = op_cov.N
        self.n_u = nuisance.n_drop
        self.jittable = op_cov.jittable

    def host_constants(self) -> dict[str, Any]:
        return {"keep": self.nuisance.keep_rfft}

    def children(self) -> dict[str, Covariance]:
        return {"op": self.op}

    def _factor(self, C: Any, nu: Any, be: Backend) -> ProjectedFactor:
        """WT ``FastCovariance._operators`` (A and its Weyl-inverse preconditioner) +
        ``BandCovariance.logdet``."""
        from .covariance.exact_time import without_basis

        op = self.op
        A = op.operator(C["op"], nu[0], nu[1], be)
        M = op.operator(C["op"], -nu[0], -nu[1], be, scale=4.0 * op.dt ** 2)
        logdet = op.logdet_model(nu) + self.correction(nu, be)
        return ProjectedFactor(without_basis(A), without_basis(M), logdet, be.xp.isfinite(logdet), op.N,
                               op.tol, op.maxit, op.on_fail, self.nuisance.n_drop)
