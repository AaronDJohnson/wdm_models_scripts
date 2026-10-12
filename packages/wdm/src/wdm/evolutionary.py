r"""Building blocks for nonstationary (evolutionary) noise analyses.

Everything in the stationary modules (:mod:`wdm.noise`, :mod:`wdm.stationary`,
:mod:`wdm.inner_product`) assumes a time-independent PSD.  This module collects
the pieces that the WDM II analyses need beyond that, none of which depends on a
particular noise model:

* :func:`endpoint_weights` -- the one-sided trapezoid weights :math:`\eta_k`
  (``1/2`` at DC and Nyquist), WDM II eq:freq_edge_weight.
* :class:`PackedGrid` -- the active cells of an ``(Nf, Nt, dt)`` grid in packed
  band-major order, their atoms, the time mask :math:`M = W\chi W^{\mathsf T}/\delta t`
  in the WDM domain, and per-cell energy fractions of a time weight.
* :class:`BlockTri`, :class:`Solver`, :class:`Schur` -- :math:`C^{-1}` and
  :math:`\log|C|` for a diagonal, block tri-diagonal or dense covariance, and for
  one whose edge rows/columns are replaced by an exact block (the edge-exact
  treatment).
* :class:`PaddedMarginal` -- the zero-padded covariance :math:`M\Lambda M`,
  handled as the extended Gaussian with the padding samples marginalised out.
* :func:`marginal_lnlike`, :func:`nuisance_basis`, :class:`FourierBand`,
  :func:`wdm_discarded` -- a Gaussian likelihood marginalised (flat prior) over
  discarded Fourier modes, e.g. an analysis cut below some :math:`f_{\min}`.
* :class:`Variant`, :func:`make_variant` -- the time-edge treatments of WDM II
  (raw, cut, padded, exact edges) as one record map ``x -> y``.

and, for a general evolutionary PSD ``Sn(f, t)`` (elementwise, broadcasting):

* :func:`band_averaged_evolutionary_variance` -- WDM II
  eq:wdm_variance_band_average at every cell (the cell-centre form is
  :func:`wdm.noise.evolutionary_variance`).
* :class:`TriCovariance` -- the nonstationary tri-diagonal covariance, WDM II
  Eq. (45), as band blocks for :class:`BlockTri`.
* :func:`time_domain_factor`, :func:`time_domain_covariance`, :func:`draw_noise`,
  :func:`exact_covariance` -- the exact covariance (WDM II
  eq:wdm_generic_covariance, phase :math:`\theta = 0`) and noise realisations.
* :func:`wrap_fractions`, :func:`wrap_cut_columns` -- the 0.1% wrap rule that
  sets how many time columns the cut treatment drops.

Provenance: ported on 2026-10-10 from the WDM II toy code
(``wdm_paper_2/toy_models/src/wdm_toys``, modules ``matrix``, ``band`` and
``noise``; reference copy ``wdm_paper_2/toy_models_reference_2026-10-10``).  The
numerical code is unchanged; only the toy configuration object was replaced by the
grid arguments ``(Nf, Nt, dt)``, and the cut and padding sizes, which the toy fixed
as module constants, are now arguments.

Conventions follow the rest of the package: grid arguments come first as
``(Nf, Nt, dt)``; packed vectors list the active cells in band-major order
(:func:`wdm.grid.active_mask`); the transform satisfies
:math:`W_a W_a^{\mathsf T} = \delta t\, I`, so :math:`W_a^{-1} = W_a^{\mathsf T}/\delta t`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .filter import phi_centered
from .grid import active_mask, band_frequencies, delta_T, sanitize_psd, validate_grid
from .transform import (
    forward_many,
    forward_time,
    forward_time_many,
    inverse_many,
    inverse_time_many,
)

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type CArray = NDArray[np.complexfloating]
type PsdFtFunc = Callable[[Array, Array], Array]

__all__ = [
    "BlockTri",
    "TriCovariance",
    "band_averaged_evolutionary_variance",
    "draw_noise",
    "exact_covariance",
    "time_domain_covariance",
    "time_domain_factor",
    "wrap_cut_columns",
    "wrap_fractions",
    "FourierBand",
    "PackedGrid",
    "PaddedMarginal",
    "Schur",
    "Solver",
    "Variant",
    "edge_columns",
    "endpoint_weights",
    "make_variant",
    "marginal_lnlike",
    "nuisance_basis",
    "wdm_discarded",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Frequency-domain weights
# ---------------------------------------------------------------------------


def endpoint_weights(n_rfft: int) -> Array:
    r"""Trapezoid weights :math:`\eta_k` for the one-sided sum: ``1/2`` at DC and
    Nyquist, ``1`` in between (WDM II eq:freq_edge_weight).

    Getting these wrong is not a small error: :math:`\eta = 1` everywhere
    overstates the variance, and dropping the DC voice altogether makes the
    time-domain covariance singular.
    """
    c = np.ones(n_rfft)
    c[0] = 0.5
    c[-1] = 0.5
    return c


# ---------------------------------------------------------------------------
# The packed grid
# ---------------------------------------------------------------------------


class PackedGrid:
    """``Nf`` bands and ``Nt`` columns at ``dt``; the active cells in packed
    (band-major) order, their atoms' spectra, and the forward transform.

    Attributes: ``mask`` (``(Nf+1, Nt)`` active cells), ``m`` and ``n`` (band and
    column of every active cell), ``size`` (number of active cells, ``N``), ``t``
    (cell times :math:`n\\Delta T`), ``f`` (rfft frequencies) and ``eta``
    (:func:`endpoint_weights`).
    """

    def __init__(self, Nf: int, Nt: int, dt: float) -> None:
        validate_grid(Nf, Nt)
        self.Nf, self.Nt, self.dt = int(Nf), int(Nt), float(dt)
        self.N = self.Nf * self.Nt
        self.T = self.N * self.dt
        self.mask = active_mask(self.Nf, self.Nt)                         # (Nf+1, Nt)
        m, n = np.nonzero(self.mask)                                      # band-major
        self.m, self.n = m, n
        self.size = m.size
        self.t = n * delta_T(self.Nf, self.dt)                            # cell times
        self.K = self.N // 2 + 1
        self.f = np.fft.rfftfreq(self.N, d=self.dt)
        self.eta = np.ones(self.K)
        self.eta[[0, -1]] = 0.5
        self._base = self._base_spectra()

    def forward(self, x: Array) -> Array:
        """Packed active WDM coefficients of ``x`` (length ``N``)."""
        return forward_time(np.asarray(x, float), self.Nf, self.Nt, self.dt)[self.mask]

    def forward_many(self, X: Array) -> Array:
        """Packed active WDM coefficients of each row of ``X``."""
        X = np.atleast_2d(X)
        return forward_time_many(X, self.Nf, self.Nt, self.dt)[:, self.mask]

    def inverse_many(self, Y: Array) -> Array:
        """Time series of packed coefficients ``Y`` (rows): ``W^{-1} = W^T / dt``."""
        Y = np.atleast_2d(Y)
        full = np.zeros((Y.shape[0], self.Nf + 1, self.Nt))
        full[:, self.mask] = Y
        return inverse_time_many(full, self.Nf, self.Nt, self.dt)

    def apply_mask(self, Y: Array, chi: Array) -> Array:
        """``M y`` for each row ``y`` of ``Y``, ``M = W chi W^T / dt``: the time mask ``chi``
        in the WDM domain (transform back, mask, transform)."""
        return self.forward_many(self.inverse_many(Y) * chi)

    def _base_spectra(self) -> dict:
        """rfft of the atoms of cells ``(m, a)``, ``a in {0, 1}``; every other atom of a
        band is a circular shift of these by an even number of columns."""
        out = {}
        for m in range(self.Nf + 1):
            for a in (0, 1):
                if not self.mask[m, a]:
                    continue
                e = np.zeros((self.Nf + 1, self.Nt))
                e[m, a] = 1.0
                # inverse_many returns rfft(W_a^{-1} e) = rfft(atom) / dt.
                out[m, a] = self.dt * inverse_many(e[None], self.Nf, self.Nt, self.dt)[0]
        return out

    def atom_spectrum(self, p: int) -> CArray:
        """rfft of the atom of packed cell ``p``."""
        m, n = int(self.m[p]), int(self.n[p])
        a = n % 2
        k = np.arange(self.K)
        return self._base[m, a] * np.exp(-2j * np.pi * k * (n - a) / self.Nt)

    def atom(self, p: int) -> Array:
        """The atom (basis function) of packed cell ``p``, length ``N``."""
        return np.fft.irfft(self.atom_spectrum(p), n=self.N)

    def energy_fraction(self, w2: Array) -> Array:
        """``rho_p = sum_t w(t)^2 g_p(t)^2 / dt`` for every active cell."""
        rho = np.empty(self.size)
        W2 = np.fft.rfft(w2)
        for (m, a), G in self._base.items():
            g2 = np.fft.irfft(G, n=self.N) ** 2
            corr = np.fft.irfft(W2 * np.conj(np.fft.rfft(g2)), n=self.N)  # sum_t w2[t+s] g2[t]
            sel = np.flatnonzero((self.m == m) & (self.n % 2 == a))
            shifts = (self.n[sel] - a) * self.Nf
            rho[sel] = corr[shifts % self.N] / self.dt
        return rho


def edge_columns(Nt: int, lead: int, trail: int) -> tuple[int, ...]:
    """The first ``lead`` and last ``trail`` time columns of an ``Nt``-column grid."""
    return tuple(range(lead)) + tuple(range(Nt - trail, Nt))


# ---------------------------------------------------------------------------
# Structured covariances: C^{-1} and log|C|
# ---------------------------------------------------------------------------


class BlockTri:
    """Cholesky of a symmetric matrix that is block tri-diagonal in the band index.

    ``blocks`` are the diagonal blocks and ``upper[i]`` the block coupling block
    ``i`` to block ``i + 1`` (``None`` where there is none).
    """

    def __init__(self, blocks: list[Array], upper: list[Array | None]) -> None:
        self.sizes = [b.shape[0] for b in blocks]
        self.off = np.concatenate([[0], np.cumsum(self.sizes)])
        Ls, Xs = [], []
        prev = None
        for i, A in enumerate(blocks):
            A = A.copy()
            if i and prev is not None and upper[i - 1] is not None and A.size and prev.size:
                X = solve_triangular(prev, upper[i - 1], lower=True)           # L_{i-1}^{-1} B
                A -= X.T @ X
                Xs.append(X)
            else:
                Xs.append(None)
            L = np.linalg.cholesky(A) if A.size else A
            Ls.append(L)
            prev = L
        self.Ls, self.Xs = Ls, Xs

    def logdet(self) -> float:
        return 2.0 * sum(float(np.sum(np.log(np.diag(L)))) for L in self.Ls if L.size)

    def solve(self, b: Array) -> Array:
        b = np.asarray(b, float)
        one = b.ndim == 1
        B = b[:, None] if one else b
        z = []
        for i, L in enumerate(self.Ls):
            r = B[self.off[i]:self.off[i + 1]].copy()
            if self.Xs[i] is not None:
                r -= self.Xs[i].T @ z[i - 1]
            z.append(solve_triangular(L, r, lower=True) if L.size else r)
        x = [None] * len(self.Ls)
        for i in range(len(self.Ls) - 1, -1, -1):
            r = z[i]
            if i + 1 < len(self.Ls) and self.Xs[i + 1] is not None:
                r = r - self.Xs[i + 1] @ x[i + 1]
            x[i] = solve_triangular(self.Ls[i], r, lower=True, trans="T") if self.Ls[i].size else r
        out = np.concatenate(x, axis=0)
        return out[:, 0] if one else out


class Solver:
    """``C^{-1}`` and ``log|C|`` for a diagonal, block tri-diagonal or dense ``C``.

    Give exactly one of ``diag`` (the diagonal), ``blocktri`` (a :class:`BlockTri`)
    or ``dense`` (the matrix).
    """

    def __init__(self, *, diag=None, blocktri=None, dense=None) -> None:
        self.diag, self.bt, self.dense = diag, blocktri, dense
        if dense is not None:
            self.cf = cho_factor(dense, lower=True, check_finite=False)

    def solve(self, b):
        if self.diag is not None:
            return b / (self.diag if b.ndim == 1 else self.diag[:, None])
        if self.bt is not None:
            return self.bt.solve(b)
        return cho_solve(self.cf, b, check_finite=False)

    def logdet(self) -> float:
        if self.diag is not None:
            return float(np.sum(np.log(self.diag)))
        if self.bt is not None:
            return self.bt.logdet()
        return 2.0 * float(np.sum(np.log(np.diag(self.cf[0]))))


class Schur:
    """``C = [[A, B], [B^T, T]]`` with ``T`` from a :class:`Solver` (edge-exact).

    ``e`` and ``i`` index the edge and interior cells in the packed vector; ``A``
    is the exact edge block, ``B`` the exact edge-interior coupling and ``T`` the
    approximate interior covariance.  The Schur complement ``A - B T^{-1} B^T``
    must be positive definite; when it is not, :func:`scipy.linalg.cho_factor`
    raises ``LinAlgError``.
    """

    def __init__(self, e: Array, i: Array, A: Array, B: Array, T: Solver) -> None:
        self.e, self.i, self.T = e, i, T
        self.TiBt = T.solve(B.T)                                   # (n_i, n_e)
        Sc = A - B @ self.TiBt
        self.cf = cho_factor(0.5 * (Sc + Sc.T), lower=True, check_finite=False)
        self.B = B

    def solve(self, b):
        one = b.ndim == 1
        b = b[:, None] if one else b
        be, bi = b[self.e], b[self.i]
        ti = self.T.solve(bi)
        xe = cho_solve(self.cf, be - self.B @ ti, check_finite=False)
        xi = ti - self.TiBt @ xe
        x = np.empty_like(b)
        x[self.e], x[self.i] = xe, xi
        return x[:, 0] if one else x

    def logdet(self) -> float:
        return self.T.logdet() + 2.0 * float(np.sum(np.log(np.diag(self.cf[0]))))


class PaddedMarginal:
    r"""``C_dd^{-1}`` and ``log|C_dd|`` for the zero-padded treatment.

    ``C_dd`` is the extended grid's covariance :math:`W^{\mathsf T}\Lambda W/\delta t^2`
    on the data samples, i.e. the extended Gaussian with the padding samples
    marginalised out; in the WDM domain this is the :math:`M\Lambda M` of WDM II.
    ``grid`` is the extended :class:`PackedGrid`, ``Lam`` a :class:`Solver` on every
    extended cell, ``Wp`` ``(size, N_pad)`` the transforms of the padding samples'
    impulses, and vectors are time series of length ``n`` (the data samples, which
    come first in the extended record).
    """

    def __init__(self, grid: PackedGrid, Lam: Solver, Wp: Array, n: int) -> None:
        self.g, self.Lam, self.Wp, self.n = grid, Lam, Wp, int(n)
        self.LiWp = Lam.solve(Wp)                                          # (size, N_pad)
        Qpp = Wp.T @ self.LiWp
        self.cf = cho_factor(0.5 * (Qpp + Qpp.T), lower=True, check_finite=False)

    def solve(self, x):
        g = self.g
        one = x.ndim == 1
        X = np.atleast_2d(x.T if not one else x)                           # (k, n)
        ext = np.zeros((X.shape[0], g.N))
        ext[:, :self.n] = X
        Liw = self.Lam.solve(g.forward_many(ext).T)                        # (size, k)
        c = cho_solve(self.cf, self.Wp.T @ Liw, check_finite=False)
        # W^T z = dt W^{-1} z
        out = g.dt * g.inverse_many((Liw - self.LiWp @ c).T)[:, :self.n]   # (k, n)
        return out[0] if one else out.T

    def logdet(self) -> float:
        return (self.Lam.logdet() - self.g.N * np.log(self.g.dt)
                + 2.0 * float(np.sum(np.log(np.diag(self.cf[0])))))


# ---------------------------------------------------------------------------
# Likelihood marginalised over discarded modes
# ---------------------------------------------------------------------------


def marginal_lnlike(y: Array, u: Array | None, solver, A_n: float, n_cells: int,
                    log_jac: float) -> float:
    r"""``ln N(y; 0, A_n C)`` marginalised over the columns of ``u`` (flat prior).

    ``solver`` provides :math:`C^{-1}` and :math:`\log|C|` (:class:`Solver`,
    :class:`Schur` or :class:`PaddedMarginal`); ``A_n`` scales the covariance,
    ``n_cells`` is the dimension of ``y`` and ``log_jac`` a constant added to the
    result (e.g. the Jacobian of the transform).  With ``u = None`` this is the
    plain Gaussian log-likelihood.
    """
    Ciy = solver.solve(y)
    quad = float(y @ Ciy)
    logdet = solver.logdet()
    n_eff = n_cells
    if u is not None and u.shape[1]:
        Ciu = solver.solve(u)
        H = u.T @ Ciu
        cf = cho_factor(0.5 * (H + H.T), lower=True, check_finite=False)
        b = u.T @ Ciy
        quad -= float(b @ cho_solve(cf, b, check_finite=False))
        logdet += 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
        n_eff -= u.shape[1]
    return -0.5 * (quad / A_n + logdet + n_eff * np.log(A_n) + n_eff * _LOG_2PI) + log_jac


def nuisance_basis(u: Array, rtol: float = 1e-10) -> Array:
    """The discarded modes' images on the kept cells, as marginalisation directions.

    Kept as given when they have full rank.  When cutting cells leaves a mode with
    no image -- it lived entirely in the removed cells -- the columns are replaced
    by an orthonormal basis of their range: a direction the kept data cannot see
    carries no information, and a flat marginal only depends on the range (up to a
    parameter-independent constant).
    """
    s = np.linalg.svd(u, compute_uv=False)
    if s.min() > rtol * s.max():
        return u
    U, s, _ = np.linalg.svd(u, full_matrices=False)
    return U[:, s > rtol * s.max()]


class FourierBand:
    """An analysis band in frequency: positive cuts retain ``fmin <= f < Nyquist``;
    ``analysis_fmin = 0`` keeps everything.

    Basis vectors are rows in time coordinates.  Spectra use NumPy's unscaled
    forward FFT convention, so their inverse FFT is an orthonormal real basis.
    DC and Nyquist are real coordinates and are excluded by any positive cut.
    """

    def __init__(self, Nf: int, Nt: int, dt: float, analysis_fmin: float = 0.0) -> None:
        validate_grid(Nf, Nt)
        self.Nf, self.Nt, self.dt = int(Nf), int(Nt), float(dt)
        self.N = self.Nf * self.Nt
        self.fmin = float(analysis_fmin)
        if not np.isfinite(self.fmin) or self.fmin < 0:
            raise ValueError('analysis_fmin must be finite and nonnegative')
        f = np.fft.rfftfreq(self.N, d=self.dt)
        f_nyq = 0.5 / self.dt
        self.keep_rfft = np.ones(len(f), dtype=bool)
        if self.fmin:
            self.keep_rfft = (f >= self.fmin) & (f < f_nyq)
            # Endpoint frequencies may round below f_nyq for nonbinary dt.
            # DC and Nyquist are real coordinates, excluded by convention.
            self.keep_rfft[[0, -1]] = False
            if not np.any(self.keep_rfft):
                raise ValueError('analysis_fmin leaves no interior Fourier frequencies')
        self.n_keep = (self.N if not self.fmin else 2 * int(self.keep_rfft.sum()))
        self.n_drop = self.N - self.n_keep

    def discarded_spectra(self) -> CArray:
        """rfft spectra of the discarded real Fourier modes, ``(n_drop, N//2 + 1)``."""
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

    def discarded_basis(self) -> Array:
        """The discarded modes as orthonormal time series, ``(n_drop, N)``."""
        return np.fft.irfft(self.discarded_spectra(), n=self.N, axis=-1)

    def project(self, x: Array) -> Array:
        """``x`` with the discarded modes removed (last axis of length ``N``)."""
        x = np.asarray(x)
        if x.shape[-1] != self.N:
            raise ValueError(f'last axis must have length {self.N}')
        if not self.fmin:
            return x.copy()
        spectrum = np.fft.rfft(x, axis=-1)
        return np.fft.irfft(spectrum * self.keep_rfft, n=self.N, axis=-1)


def wdm_discarded(band: FourierBand) -> tuple[NDArray[np.intp], Array]:
    """Return exact nonzero active-pixel rows and their discarded WDM basis.

    Sparse input spectra preserve exact zeros outside overlapping WDM filters;
    no threshold or approximate row truncation enters the marginal precision.
    """
    transformed = forward_many(band.discarded_spectra(), band.Nf, band.Nt, band.dt)
    u = transformed[:, active_mask(band.Nf, band.Nt)].T
    rows = np.flatnonzero(np.any(u != 0, axis=1))
    return rows, u[rows]


# ---------------------------------------------------------------------------
# Time-edge treatments
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """The record map ``x -> y`` of one time-edge treatment.

    ``grid`` is the (possibly extended) :class:`PackedGrid`, ``keep`` selects the
    packed cells the likelihood uses, and ``w`` is an optional time weight on the
    grid's record (zeros over the padding).
    """

    grid: PackedGrid
    keep: BoolArray             # bool over grid.size
    w: Array | None             # time weight on the grid's record (padding)

    def extend(self, x: Array) -> Array:
        x = np.asarray(x, float)
        N = self.grid.N
        out = np.zeros(x.shape[:-1] + (N,))
        out[..., :x.shape[-1]] = x
        if self.w is not None:
            out = out * self.w
        return out

    def map(self, x: Array) -> Array:
        return self.grid.forward(self.extend(x))[self.keep]

    def map_many(self, X: Array) -> Array:
        return self.grid.forward_many(self.extend(X))[:, self.keep]


def make_variant(Nf: int, Nt: int, dt: float, edge: str, *,
                 cut: tuple[int, int] | None = None,
                 pad_columns: int | None = None) -> Variant:
    """The time-edge treatment ``edge`` of WDM II on an ``(Nf, Nt, dt)`` record.

    * ``"raw"`` and ``"exact"``: the record's own grid, every cell kept (the exact
      edges are applied to the covariance, e.g. with :class:`Schur`).
    * ``"cut"``: drop the first ``cut[0]`` and last ``cut[1]`` time columns.
    * ``"pad"``: append ``pad_columns`` columns of zeros (equivalent, by the
      transform's periodicity, to half at each end) and keep the record's
      ``Nt`` columns; use with :class:`PaddedMarginal`.
    """
    if edge in ("raw", "exact"):
        g = PackedGrid(Nf, Nt, dt)
        return Variant(g, np.ones(g.size, bool), None)
    if edge == "cut":
        if cut is None:
            raise ValueError('edge "cut" needs cut=(leading, trailing) columns')
        g = PackedGrid(Nf, Nt, dt)
        return Variant(g, ~np.isin(g.n, edge_columns(Nt, *cut)), None)
    if edge == "pad":
        if pad_columns is None:
            raise ValueError('edge "pad" needs pad_columns')
        g = PackedGrid(Nf, Nt + int(pad_columns), dt)
        w = np.zeros(g.N)
        w[:Nf * Nt] = 1.0
        keep = g.n < Nt
        return Variant(g, keep, w)
    raise ValueError(f"unknown edge treatment {edge!r}")


# ===========================================================================
# Beyond the toy's powerlaw: everything below takes a general S(f, t)
# ===========================================================================
#
# A PSD here is a callable ``Sn(f, t)`` that is ELEMENTWISE and broadcasts like a
# NumPy ufunc: given arrays ``f`` and ``t`` of broadcastable shapes it returns the
# one-sided evolutionary PSD at every broadcast pair (note the package argument
# order (f, t); WDM II writes S(t, f)). Samples pass through
# :func:`wdm.grid.sanitize_psd` (inf -> 0, nan and negative values raise).


def _psd_at(Sn: PsdFtFunc, f: Array, t: Array, context: str) -> Array:
    shape = np.broadcast_shapes(np.shape(f), np.shape(t))
    out = sanitize_psd(Sn(f, t), context)
    if out.shape != shape:
        raise ValueError(f"{context}: Sn(f, t) must be elementwise and broadcast to "
                         f"{shape}; got {out.shape}")
    return out


def band_averaged_evolutionary_variance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
                                        bands: Array | None = None) -> Array:
    r"""The filter-weighted band average of :math:`S(f, t_n)` at every cell centre,

    .. math::  \sigma^2_{mn} = \frac12\,
               \frac{\sum_j \tilde\phi^2(f_j - f_m)\,S(f_j, t_n)}{\sum_j \tilde\phi^2(f_j - f_m)},

    WDM II eq:wdm_variance_band_average (the edge-band factor
    :math:`\varepsilon_{mn}` is not applied: inactive edge cells carry the same
    value, as in :func:`wdm.noise.evolutionary_variance`). In the stationary limit
    this is :func:`wdm.noise.band_averaged_variance`, i.e. the exact
    :math:`\mathrm{diag}\,\Lambda`. The DC band's slab folds about 0 and the
    Nyquist band's about :math:`f_{\rm Nyq}`, where the discrete spectrum aliases.
    Cell times are :math:`t_n = n\Delta T`; the cell-centre (midpoint) variance
    :math:`S(f_m, t_n)/2` is :func:`wdm.noise.evolutionary_variance`.

    Returns shape ``(Nf + 1, Nt)``, or ``(len(bands), Nt)`` when ``bands`` selects
    band indices (useful when ``Nf`` is large and only a few bands are analysed).
    """
    validate_grid(Nf, Nt)
    N = Nf * Nt
    phi2 = phi_centered(Nf, Nt, dt) ** 2
    # The Meyer window has compact support: a slab weight of exactly 0 cannot
    # contribute, so dropping those bins is an exact restriction.
    sup = np.flatnonzero(phi2)
    phi2 = phi2[sup]
    f_off = (np.arange(Nt) - Nt // 2)[sup] / (N * dt)
    m = np.arange(Nf + 1) if bands is None else np.asarray(bands, int)
    fb = band_frequencies(Nf, dt)[m][:, None] + f_off[None, :]
    f_nyq = 0.5 / dt
    ff = np.abs(fb)
    ff = np.where(ff > f_nyq, 2.0 * f_nyq - ff, ff)                       # (B, J)
    t = np.arange(Nt) * delta_T(Nf, dt)
    S = _psd_at(Sn, ff[:, None, :], t[None, :, None],
                "band_averaged_evolutionary_variance")                    # (B, Nt, J)
    return 0.5 * (S * phi2).sum(axis=-1) / phi2.sum()


# ---------------------------------------------------------------------------
# Exact covariance and noise simulation (theta = 0)
# ---------------------------------------------------------------------------


def time_domain_factor(N: int, dt: float, Sn: PsdFtFunc) -> Array:
    r"""A square root :math:`G` of the exact time-domain covariance, :math:`\Sigma = G G^{\mathsf T}`.

    The noise is the oscillatory process of WDM II with amplitude
    :math:`\sqrt{S(f, t)}` and zero phase, :math:`\theta(t, f) = 0`:

    .. math::  x(t_i) = \sum_k \sqrt{\eta_k\,\Delta f\,S(f_k, t_i)}\,
               \big[\xi_k \cos(2\pi f_k t_i) + \zeta_k \sin(2\pi f_k t_i)\big],

    so :math:`\Sigma_{ij} = \Delta f \sum_k \eta_k \sqrt{S(f_k, t_i) S(f_k, t_j)}
    \cos(2\pi f_k (t_i - t_j))`, with :math:`t_i = i\,\delta t`,
    :math:`f_k = k/(N\delta t)` and :func:`endpoint_weights` :math:`\eta_k`.
    Columns are the cosine voices :math:`k = 0 \ldots N/2` followed by the sine
    voices :math:`k = 1 \ldots N/2 - 1` (the DC and Nyquist sines vanish), so
    :math:`G` is square. The exact covariance depends on the phase
    :math:`\theta`, which a model of :math:`S(t, f)` alone does not fix; this is
    the :math:`\theta = 0` member. :math:`O(N^2)` memory.
    """
    if N % 2:
        raise ValueError("N must be even")
    f = np.fft.rfftfreq(N, d=dt)
    t = np.arange(N) * dt
    df = 1.0 / (N * dt)
    c = endpoint_weights(f.size)
    S = _psd_at(Sn, f[None, :], t[:, None], "time_domain_factor")          # (N, K)
    amp = np.sqrt(S * df * c[None, :])
    ph = 2.0 * np.pi * np.outer(t, f)
    G_cos = amp * np.cos(ph)
    G_sin = (amp * np.sin(ph))[:, 1:-1]
    return np.hstack([G_cos, G_sin])


def time_domain_covariance(N: int, dt: float, Sn: PsdFtFunc) -> Array:
    r"""The exact time-domain covariance :math:`\Sigma = G G^{\mathsf T}`
    (:func:`time_domain_factor`), shape ``(N, N)``. :math:`O(N^3)`."""
    G = time_domain_factor(N, dt, Sn)
    return G @ G.T


def draw_noise(N: int, dt: float, Sn: PsdFtFunc, rng: np.random.Generator) -> Array:
    """One realisation of length ``N`` from the exact covariance (:func:`time_domain_factor`).

    The standard normals are consumed in a fixed order (cosine coefficients
    first, then sine), so a given ``rng`` state always yields the same series.
    """
    G = time_domain_factor(N, dt, Sn)
    z = rng.standard_normal(G.shape[1])
    return G @ z


def exact_covariance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc) -> Array:
    r"""The exact WDM covariance :math:`\Lambda = W_a \Sigma W_a^{\mathsf T}` on the
    active cells (packed, band-major), shape ``(N, N)``.

    Built in Gram form, :math:`\Lambda = (W_a G)(W_a G)^{\mathsf T}`, so it is
    positive semi-definite by construction (WDM II eq:wdm_generic_covariance with
    :math:`\theta = 0`). :math:`O(N^3)`; for tests and small grids.
    """
    validate_grid(Nf, Nt)
    N = Nf * Nt
    G = time_domain_factor(N, dt, Sn)
    A = forward_time_many(np.ascontiguousarray(G.T), Nf, Nt, dt)[:, active_mask(Nf, Nt)].T
    Lam = A @ A.T
    return 0.5 * (Lam + Lam.T)


# ---------------------------------------------------------------------------
# The nonstationary tri-diagonal covariance, WDM II Eq. (45)
# ---------------------------------------------------------------------------


class TriCovariance:
    r"""WDM II Eq. (45) on the cells ``keep`` of ``grid``, as dense band blocks.

    For cells :math:`p = (m, n)` and :math:`q = (m', n')`, :math:`m' \in \{m, m+1\}`,
    with base atoms of parities :math:`a = n \bmod 2`, :math:`b = n' \bmod 2`,

    .. math::  \Lambda_{pq} = \frac1T\,\mathrm{Re}\sum_k \eta_k\,
               S(f_k, \bar t)\, B_{ab}(k)\, e^{-2\pi i k d / N_t},
               \quad d = (n - a) - (n' - b),\quad \bar t = (t_n + t_{n'})/2,

    :math:`B_{ab} = \hat g_{m a}\hat g^*_{m' b}` (rfft of the base atoms). Blocks
    are dense in time (no lag truncation). :meth:`blocks` expands the time
    dependence of :math:`S(f_k, \bar t)` in ``P`` Chebyshev polynomials on
    :math:`\bar t \in [0, t_{\max}]` (interpolated at Chebyshev nodes, so any
    smooth :math:`S` works; raise ``P`` for faster time variation), and each block
    costs one length-:math:`N_t` FFT per order. :meth:`values` and :meth:`dense`
    evaluate the sum directly and are the reference. ``t_max`` defaults to the
    grid's duration; pair mid-times of the kept cells must lie in
    :math:`[0, t_{\max}]`. Use with :class:`BlockTri` and :class:`Solver`.
    """

    def __init__(self, grid: PackedGrid, keep: BoolArray | None = None, P: int = 24,
                 t_max: float | None = None) -> None:
        g = grid
        self.grid, self.P = g, int(P)
        self.keep = np.ones(g.size, bool) if keep is None else np.asarray(keep, bool)
        self.t_max = float(g.T if t_max is None else t_max)
        cells = np.flatnonzero(self.keep)
        self.cells = cells
        self.pos = -np.ones(g.size, int)
        self.pos[cells] = np.arange(cells.size)
        self.band_cells = [cells[g.m[cells] == m] for m in range(g.Nf + 1)]
        # Chebyshev basis on the pair mid-times t_bar = s dT / 2, s = n + n'
        s_grid = np.arange(2 * g.Nt - 1)
        u = 2.0 * (s_grid * 0.5 * delta_T(g.Nf, g.dt) / self.t_max - 0.5)
        used = s_grid <= 2 * int(g.n[cells].max())
        if np.any(np.abs(u[used]) > 1 + 1e-12):
            raise ValueError("pair mid-times must lie inside [0, t_max]")
        Tp = np.empty((self.P, s_grid.size))
        Tp[0] = 1.0
        if self.P > 1:
            Tp[1] = u
        for q in range(2, self.P):
            Tp[q] = 2 * u * Tp[q - 1] - Tp[q - 2]
        self.Tp = Tp
        self.specs = [self._block_spec(m, m2) for m in range(g.Nf + 1)
                      for m2 in (m, m + 1) if m2 <= g.Nf]
        k_all = [k for spec in self.specs for (k, _) in spec["B"].values()]
        self.k_used = np.unique(np.concatenate(k_all)) if k_all else np.zeros(0, int)

    def _block_spec(self, m: int, m2: int) -> dict:
        """Static index arrays for block ``(m, m2)``: one entry per (row, col) cell pair."""
        g = self.grid
        R, C = self.band_cells[m], self.band_cells[m2]
        nr, nc = g.n[R], g.n[C]
        a = (nr % 2)[:, None] + 0 * nc[None, :]
        b = 0 * nr[:, None] + (nc % 2)[None, :]
        combo = 2 * a + b                                              # (nR, nC)
        d = ((nr[:, None] - a) - (nc[None, :] - b)) % g.Nt
        s = nr[:, None] + nc[None, :]
        B = {}
        for aa in (0, 1):
            for bb in (0, 1):
                Ga, Gb = g._base.get((m, aa)), g._base.get((m2, bb))
                if Ga is None or Gb is None:
                    continue
                k = np.flatnonzero((np.abs(Ga) > 0) & (np.abs(Gb) > 0))
                B[2 * aa + bb] = (k, g.eta[k] * Ga[k] * np.conj(Gb[k]) / g.T)
        return dict(m=m, m2=m2, R=R, C=C, combo=combo, d=d, s=s, B=B)

    def chebyshev_coefficients(self, Sn: PsdFtFunc) -> Array:
        r"""``(P, K)``: Chebyshev coefficients :math:`c_p(f_k)` of :math:`S(f_k, \bar t)` in
        :math:`u = 2\bar t/t_{\max} - 1`, by interpolation at the ``P`` Chebyshev
        nodes (zero for bins no block uses)."""
        g, P = self.grid, self.P
        j = np.arange(P)
        x = np.cos(np.pi * (j + 0.5) / P)                                 # nodes in u
        t_nodes = 0.5 * (x + 1.0) * self.t_max
        k = self.k_used
        S = _psd_at(Sn, g.f[k][None, :], t_nodes[:, None], "TriCovariance")   # (P, |k|)
        Cmat = np.cos(np.pi * np.outer(np.arange(P), j + 0.5) / P)          # (p, j)
        c = (2.0 / P) * (Cmat @ S)
        c[0] *= 0.5
        out = np.zeros((P, g.K))
        out[:, k] = c
        return out

    def block(self, spec: dict, coeffs: Array) -> Array:
        """One band block from the Chebyshev coefficients (:meth:`chebyshev_coefficients`)."""
        g = self.grid
        F = np.zeros((4, self.P, g.Nt), complex)
        for cb, (k, Bk) in spec["B"].items():
            V = coeffs[:, k] * Bk[None, :]                                   # (P, |k|)
            folded = np.zeros((self.P, g.Nt), complex)
            np.add.at(folded, (slice(None), k % g.Nt), V)
            F[cb] = np.fft.fft(folded, axis=1)                          # F_p(d)
        Fg = F[spec["combo"], :, spec["d"]]                             # (nR, nC, P)
        Tg = self.Tp[:, spec["s"]]                                      # (P, nR, nC)
        return np.real(np.einsum("ijp,pij->ij", Fg, Tg))

    def blocks(self, Sn: PsdFtFunc):
        """Diagonal and upper off-diagonal band blocks, for :class:`BlockTri`."""
        g = self.grid
        coeffs = self.chebyshev_coefficients(Sn)
        diag = [None] * (g.Nf + 1)
        up = [None] * g.Nf
        for spec in self.specs:
            blk = self.block(spec, coeffs)
            if spec["m2"] == spec["m"]:
                diag[spec["m"]] = 0.5 * (blk + blk.T)
            else:
                up[spec["m"]] = blk
        return diag, up

    # -- direct evaluation: the reference ------------------------------------
    def pairs(self):
        """Every ``(p, q)``, ``m_q in {m_p, m_p + 1}`` (upper triangle within a band)."""
        g = self.grid
        P, Q = [], []
        for m in range(g.Nf + 1):
            A = self.band_cells[m]
            for m2 in (m, m + 1):
                if m2 > g.Nf:
                    continue
                B = self.band_cells[m2]
                if not A.size or not B.size:
                    continue
                ok = np.ones((A.size, B.size), bool)
                if m2 == m:
                    ok = A[:, None] <= B[None, :]
                i, j = np.nonzero(ok)
                P.append(A[i])
                Q.append(B[j])
        return np.concatenate(P), np.concatenate(Q)

    def values(self, P: Array, Q: Array, Sn: PsdFtFunc) -> Array:
        """Eq. (45) at the cell pairs ``(P, Q)`` by the direct sum over Fourier bins."""
        g = self.grid
        out = np.empty(P.size)
        tbar = 0.5 * (g.t[P] + g.t[Q])
        k = np.arange(g.K)
        key = ((g.m[P] * 2 + g.n[P] % 2) * (2 * g.Nf + 4) + g.m[Q] * 2 + g.n[Q] % 2)
        for kk in np.unique(key):
            sel = np.flatnonzero(key == kk)
            p0, q0 = P[sel[0]], Q[sel[0]]
            mp, ap, mq, aq = int(g.m[p0]), int(g.n[p0] % 2), int(g.m[q0]), int(g.n[q0] % 2)
            Gp, Gq = g._base[mp, ap], g._base[mq, aq]
            s = np.flatnonzero((np.abs(Gp) > 0) & (np.abs(Gq) > 0))
            if not s.size:
                out[sel] = 0.0
                continue
            prod = g.eta[s] * Gp[s] * np.conj(Gq[s]) / g.T
            d = (g.n[P[sel]] - ap) - (g.n[Q[sel]] - aq)
            ph = np.exp(-2j * np.pi * np.outer(d, k[s]) / g.Nt)
            S = _psd_at(Sn, g.f[s][None, :], tbar[sel][:, None], "TriCovariance.values")
            out[sel] = np.real(np.sum(S * ph * prod[None, :], axis=1))
        return out

    def dense(self, Sn: PsdFtFunc) -> Array:
        """The full matrix on the kept cells by the direct sum (tests, small grids)."""
        P, Q = self.pairs()
        v = self.values(P, Q, Sn)
        n = self.cells.size
        C = np.zeros((n, n))
        C[self.pos[P], self.pos[Q]] = v
        C[self.pos[Q], self.pos[P]] = v
        return C


# ---------------------------------------------------------------------------
# Time edges: how much of each atom wraps around the record
# ---------------------------------------------------------------------------


def wrap_fractions(Nf: int, Nt: int, dt: float, *, bands: Array | None = None) -> Array:
    r"""Fraction of each atom's energy that wraps across the record boundary.

    The transform is periodic, so the atom of a cell near the start of the record
    extends past :math:`t = 0` and reappears at the end (and vice versa). For the
    cell :math:`(m, n)`, centred on sample :math:`n N_f`, this is the fraction of
    :math:`\sum_j g_{mn}(t_j)^2` at samples whose unwrapped position, measured from
    the centre along the shorter way round, lies outside :math:`[0, N)`.

    Returns ``(Nf + 1, Nt)`` (or ``(len(bands), Nt)``), ``nan`` at inactive cells.
    """
    validate_grid(Nf, Nt)
    N = Nf * Nt
    mask = active_mask(Nf, Nt)
    m_list = np.arange(Nf + 1) if bands is None else np.asarray(bands, int)
    out = np.full((m_list.size, Nt), np.nan)
    n = np.arange(Nt)
    for row, m in enumerate(m_list):
        for a in (0, 1):
            cols = n[(n % 2 == a) & mask[m]]
            if not cols.size:
                continue
            e = np.zeros((1, Nf + 1, Nt))
            e[0, m, a] = 1.0
            g2 = inverse_time_many(e, Nf, Nt, dt)[0] ** 2               # base atom, centre a Nf
            delta = (np.arange(N) - a * Nf + N // 2) % N - N // 2       # signed offset from centre
            order = np.argsort(delta, kind="stable")
            ds, cum = delta[order], np.concatenate([[0.0], np.cumsum(g2[order])])
            total = cum[-1]
            c = cols * Nf                                               # each cell's centre
            left = cum[np.searchsorted(ds, -c, side="left")]            # delta < -c
            right = total - cum[np.searchsorted(ds, N - c, side="left")]  # delta >= N - c
            out[row, cols] = (left + right) / total
    return out


def wrap_cut_columns(Nf: int, Nt: int, dt: float, tol: float = 1e-3, *,
                     bands: Array | None = None) -> tuple[int, int]:
    """``(leading, trailing)``: the time columns to cut so that no kept atom of
    ``bands`` (all bands by default) wraps more than ``tol`` of its energy across
    the record boundary (:func:`wrap_fractions`; WDM II uses ``tol = 1e-3``).

    The rule is geometric: it ignores the noise. A record whose noise is periodic
    over its length (e.g. a full year of LISA's annual modulation) needs no cut.
    """
    frac = np.nanmax(wrap_fractions(Nf, Nt, dt, bands=bands), axis=0)    # worst band per column
    bad = np.flatnonzero(frac > tol)
    half = Nt // 2
    lead = int(bad[bad < half].max()) + 1 if np.any(bad < half) else 0
    trail = Nt - int(bad[bad >= half].min()) if np.any(bad >= half) else 0
    return lead, trail
