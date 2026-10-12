r"""``PushedRows``: rows of an exact covariance pushed to WDM cells (edge-exact models).

* Time (sinusoid): :math:`R = W\,\tilde\Sigma\,g_e` for every edge atom :math:`g_e`, on
  every destination cell -- WT ``MatrixWdm._exact_rows`` (``order="fixed"`` P = 24, one
  ``CovOperator`` per atom, the NumPy golden) or WG ``_exact_rows_staged`` (``"adaptive"``
  even ``P >= 8`` from :func:`cheb_order`, ``phi_cache``: the noise-free
  :math:`\Phi = \mathrm{rfft}(T_q g_e)` precomputed, ``staged``: separately compiled
  stages on XLA-CPU; the JAX golden).
* Fourier (GB): :math:`R = \tfrac12\mathrm{Re}(U_E M' U^{\mathsf H})` with ``U`` the
  destination cells' unit-norm atoms on the window bins -- GB ``exact_rows_builder``.

``order`` and ``P`` are static: the adaptive order is a deterministic host function of
:math:`\alpha_d` alone, so cached noise states never depend on the batch.
"""

from __future__ import annotations

import functools
from math import exp, lgamma, log
from typing import Any, Literal

import numpy as np
from wdm.backend import Backend
from wdm.transform import atom_spectrum, base_atom_spectra, forward_time_many

from .exact_fourier import ExactFourierCovariance
from .exact_time import CovOperator, ExactTimeCovariance, bessel_coefficients

__all__ = ["CHEB_TOL", "PushedRows", "cheb_order", "cheb_tail", "positions"]

#: Truncation bound of the exact rows' Chebyshev expansion (WG ``MatrixGPU.CHEB_TOL``).
CHEB_TOL = 1e-14


def cheb_tail(P: int, x: float) -> float:
    """``2 e^x sum_{p >= P} I_p(x)``: the relative truncation bound of ``e^{x tau}``'s
    Chebyshev series at order ``P`` (WG ``MatrixGPU._cheb_tail`` verbatim)."""
    if x <= 0.0:
        return 0.0
    lh = log(0.5 * x)
    return 2.0 * exp(x) * sum(exp((2 * m + p) * lh - lgamma(m + 1) - lgamma(m + p + 1))
                              for p in range(P, P + 60) for m in range(60))


@functools.cache
def _cheb_limits(P_max: int, tol: float) -> tuple[tuple[int, float], ...]:
    """Largest ``x`` each even order ``8 <= P < P_max`` meets ``tol`` at (bisection)."""
    out = []
    for P in range(8, P_max, 2):
        lo, hi = 0.0, 16.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cheb_tail(P, mid) < tol else (lo, mid)
        out.append((P, lo))
    return tuple(out)


def cheb_order(alpha_d: float, L_max: float, P_max: int = 24, tol: float = CHEB_TOL) -> int:
    """The smallest even order (from 8) whose bound at ``x = |alpha_d| max|L| / 4`` is
    below ``tol``; at most ``P_max`` (WG ``MatrixGPU._cheb_order`` verbatim)."""
    x = 0.25 * abs(float(alpha_d)) * L_max
    return next((P for P, lim in _cheb_limits(int(P_max), float(tol)) if x <= lim), int(P_max))


def positions(sub: Any, cells: Any) -> np.ndarray:
    """Positions of the cells of ``sub`` inside ``cells`` (``CellSet``s on one grid)."""
    if sub.grid != cells.grid:
        raise ValueError(f"the cells live on {sub.grid}, not on the destination grid {cells.grid}")
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cells.m, cells.n, strict=True))}
    try:
        return np.array([index[int(a), int(b)] for a, b in zip(sub.m, sub.n, strict=True)], int)
    except KeyError as err:
        raise ValueError(f"cell {err.args[0]} is not among the destination cells") from None


class PushedRows:
    """Rows ``R (n_e, n)`` of an exact covariance on the destination cells.

    Parameters
    ----------
    exact
        ``ExactTimeCovariance`` (time) or ``ExactFourierCovariance`` (Fourier).
    map
        The data map; its ``dst.cells`` are the destination cells (time:
        ``maps.TimeToWDM``) and its ``C`` the cells' unit-norm atoms on the exact
        covariance's bins (Fourier: ``maps.FourierToWDM(bins, dst, restrict=True)``).
        ``cells=`` / ``atoms=`` override them.
    rows
        The edge cells (a ``CellSet`` inside the destination cells).
    order, phi_cache, staged, chunk
        Time only (module docstring).  ``chunk`` atoms per batch with ``phi_cache``
        (default 64).
    """

    def __init__(self, exact: Any, map: Any, rows: Any, *, cells: Any = None, atoms: Any = None,
                 order: Literal["fixed", "adaptive"] = "fixed", phi_cache: bool = False,
                 staged: bool = False, chunk: int | None = None) -> None:
        self.exact, self.rows_cells = exact, rows
        if cells is None and map is None:
            raise ValueError("pass a map (its dst.cells are the destination cells) or cells=")
        self.cells = cells if cells is not None else map.dst.cells
        self.e = positions(rows, self.cells)
        self.n = len(self.cells.m)
        if isinstance(exact, ExactTimeCovariance):
            self.kind = "time"
            if order not in ("fixed", "adaptive"):
                raise ValueError(f"order must be 'fixed' or 'adaptive'; got {order!r}")
            g = self.cells.grid
            if g.N != exact.N or g.dt != exact.dt:
                raise ValueError("the destination grid and the exact covariance differ")
            if exact.form != "operator":
                raise ValueError("time rows need ExactTimeCovariance(form='operator') "
                                 "(its constants carry the Chebyshev basis)")
            if staged and not phi_cache:
                raise ValueError("staged=True runs the phi_cache stages: pass phi_cache=True")
            if chunk is not None and int(chunk) < 1:
                raise ValueError(f"chunk must be a positive number of atoms; got {chunk}")
            if self.e.size == 0:
                raise ValueError("time rows need at least one edge cell")
            self.order, self.phi_cache, self.staged = order, bool(phi_cache), bool(staged)
            self.chunk = int(chunk or 64)
            self.P_max = exact.P
            self.L_max = float(np.max(np.abs(exact.host_constants()["L"])))
            self.atoms = None
        elif isinstance(exact, ExactFourierCovariance):
            self.kind = "fourier"
            if order != "fixed" or phi_cache or staged or chunk is not None:
                raise ValueError("order, phi_cache, staged and chunk apply to time rows only")
            if atoms is None:
                if map is None:
                    raise ValueError("Fourier rows need a FourierToWDM map or atoms=")
                # D-B's FourierToWDM: C (P, width) unit-norm atoms on the src bins [lo, lo + width);
                # FourierToWDM(exact.bins, dst, restrict=True) gives lo = 0, width = exact.n.
                src, bins = map.src, exact.bins
                if (src.N, src.dt, src.k0, src.n) != (bins.N, bins.dt, bins.k0, bins.n):
                    raise ValueError("the map's src must be the exact covariance's bins: "
                                     "use FourierToWDM(bins, dst, restrict=True)")
                if (map.lo, map.width) != (0, exact.n):
                    raise ValueError("the map's atoms must cover exactly the exact covariance's bins: "
                                     "use FourierToWDM(bins, dst, restrict=True)")
                atoms = map.C
            self.atoms = np.asarray(atoms)
            if self.atoms.shape != (self.n, exact.n):
                raise ValueError(f"atoms must be ({self.n}, {exact.n}); got {self.atoms.shape}")
            self.order, self.phi_cache, self.staged = "fixed", False, False
        else:
            raise TypeError(f"cannot push {type(exact).__name__}")

    # -- constants ----------------------------------------------------------------------
    def host_constants(self) -> dict[str, Any]:
        if self.kind == "fourier":
            return {"exact": self.exact.host_constants(), "U": self.atoms, "UE": self.atoms[self.e]}
        g = self.cells.grid
        base = base_atom_spectra(g.Nf, g.Nt, g.dt)
        m = np.asarray(self.cells.m, int)[self.e]
        n = np.asarray(self.cells.n, int)[self.e]
        atoms = np.stack([np.fft.irfft(atom_spectrum(int(a), int(b), g.Nf, g.Nt, g.dt, base=base), n=g.N)
                          for a, b in zip(m, n, strict=True)])
        C = dict(self.exact.host_constants())
        C["atoms"] = atoms
        C["flat"] = np.asarray(self.cells.m, int) * g.Nt + np.asarray(self.cells.n, int)
        if self.phi_cache:
            import scipy.fft

            pad = (-atoms.shape[0]) % self.chunk
            A = np.concatenate([atoms, np.zeros((pad, g.N))]).reshape(-1, self.chunk, g.N)
            T = C["T"]
            C["Phi"] = np.stack([scipy.fft.rfft(T[None] * a[:, None, :], axis=-1, workers=4) for a in A])
        return C

    # -- the order ------------------------------------------------------------------------
    def order_for(self, nu: Any) -> int:
        """The static Chebyshev order at ``nu`` (``P_max`` when ``order="fixed"``)."""
        if self.kind == "fourier" or self.order == "fixed":
            return getattr(self, "P_max", 0)
        return cheb_order(float(np.asarray(nu)[1]), self.L_max, self.P_max)

    def _check_P(self, P: int) -> int:
        P = int(P)
        if not 1 <= P <= self.P_max:
            raise ValueError(f"P must be in [1, {self.P_max}]; got {P}")
        return P

    # -- evaluation -------------------------------------------------------------------------
    def evaluate(self, C: Any, nu: Any, be: Backend, P: int | None = None) -> Any:
        """``R`` at ``nu`` (pure in ``(C, nu)``; ``P`` static)."""
        xp = be.xp
        if self.kind == "fourier":
            M = self.exact.evaluate(C["exact"], nu, be)
            return 0.5 * xp.real((C["UE"] @ M) @ xp.conj(C["U"]).T)
        P = self._check_P(self.order_for(nu) if P is None else P)
        g = self.cells.grid
        n_e = self.e.size
        L = C["L"]
        E = bessel_coefficients(0.25 * nu[1] * L, P, be)
        if not self.phi_cache:
            op = CovOperator(E, 1.0 / self.exact.T_obs * xp.exp(nu[0] * L), C["T"][:P], g.N)
            Sg = be.map(lambda x: op(x, be), C["atoms"])
        else:
            gk = xp.exp(nu[0] * L) / self.exact.T_obs
            Sg = xp.concatenate([self._spectra_to_series(self._spectra(Phi, E, gk, be), C["T"], be)
                                 for Phi in C["Phi"]])[:n_e]
        return self._cells(Sg, C["flat"], be)

    def _spectra(self, Phi: Any, E: Any, gk: Any, be: Backend) -> Any:
        xp = be.xp
        Fk = xp.sum(E[None] * Phi[:, :E.shape[0]], axis=1)
        return (gk * Fk)[:, None, :] * E[None]

    def _spectra_to_series(self, V: Any, T: Any, be: Backend) -> Any:
        xp = be.xp
        N = self.cells.grid.N
        Y = be.fft_pinned("irfft", V, n=N, axis=-1)
        return (0.5 * N) * xp.sum(Y * T[None, :Y.shape[1]], axis=1)

    def _cells(self, Sg: Any, flat: Any, be: Backend) -> Any:
        g = self.cells.grid
        W = forward_time_many(Sg, g.Nf, g.Nt, g.dt, be=be)
        return W.reshape(W.shape[0], -1)[:, flat]

    def evaluate_staged(self, C: Any, nu: Any, be: Backend, P: int) -> Any:
        """:meth:`evaluate` (``phi_cache``) as separately compiled stages per chunk: the same
        numbers; XLA's CPU inverse FFT is ~6x slower on an input computed inside the
        same program (WG ``_exact_rows_staged``)."""
        if not self.staged:
            raise ValueError("evaluate_staged needs PushedRows(..., phi_cache=True, staged=True)")
        P = self._check_P(P)
        xp = be.xp
        stages = self.__dict__.setdefault("_stages", {})
        if be.name not in stages:
            T_obs = self.exact.T_obs

            def coeffs(nu: Any, L: Any, P: int) -> Any:
                return bessel_coefficients(0.25 * nu[1] * L, P, be), xp.exp(nu[0] * L) / T_obs

            stages[be.name] = (be.jit(coeffs, static_argnums=2),
                               be.jit(lambda Phi, E, gk: self._spectra(Phi, E, gk, be)),
                               be.jit(lambda V, T: self._spectra_to_series(V, T, be)),
                               be.jit(lambda S, flat: self._cells(S, flat, be)))
        coeffs, spectra, series, cells = stages[be.name]
        E, gk = coeffs(nu, C["L"], int(P))
        out = [cells(series(spectra(Phi, E, gk), C["T"]), C["flat"]) for Phi in C["Phi"]]
        return xp.concatenate(out)[:self.e.size]
