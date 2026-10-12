r"""Stateless WDM II kernels for evolutionary (nonstationary) noise.

Everything here is a pure function of ``(grid, spectrum, arrays)``.  Anything with
state, a factorisation or a density (the wdm_dev ``PackedGrid``, ``BlockTri``,
``Solver``, ``Schur``, ``PaddedMarginal``, ``marginal_lnlike``, ``FourierBand``,
``Variant``) lives in ``domainia``.

* Time masks: :func:`energy_fraction`, :func:`apply_time_mask`
  (:math:`M = W\chi W^{\mathsf T}/\delta t`).
* The 0.1% wrap rule: :func:`wrap_fractions`, :func:`wrap_cut_columns`.
* The exact covariance (WDM II eq:wdm_generic_covariance, :math:`\theta = 0`):
  :func:`time_domain_factor`, :func:`time_domain_covariance`,
  :func:`exact_covariance`, :func:`draw_noise` (dense, NumPy, small grids).
* WDM II Eq. (45), the nonstationary block tri-diagonal covariance
  :math:`\Lambda_{pq} = \frac1T\,\mathrm{Re}\sum_k \eta_k S(f_k, \bar t_{pq})
  \hat g_p(k)\hat g_q^*(k)`, :math:`\bar t = (t_n + t_{n'})/2` used literally:

  - :func:`eq45_plan` -- host constants (one plan per grid and kept-cell set);
  - Chebyshev-in-time builders: :func:`eq45_coeffs_bessel` (exact Bessel-I
    coefficients of the log-linear family; the WT ``TriCovariance._cheb_weights``
    that wrote the NumPy references) or :func:`eq45_coeffs_interp` (any
    ``Sn(f, t)``, interpolated at ``P`` Chebyshev nodes; wdm_dev), then
    :func:`eq45_blocks_cheb`;
  - the Gram form for the log-linear family (the JAX chains' WG build):
    :func:`eq45_gram_plan`, :func:`eq45_gram_gather`, :func:`eq45_blocks_gram`;
  - the separable form :math:`S_c = s_I + g(t) s_G` (GB):
    :func:`eq45_separable_kernels`, :func:`eq45_separable_assemble`;
  - :func:`eq45_direct` -- the direct sum, the oracle.

PSDs are callables ``Sn(f, t)``: elementwise, broadcasting, argument order
``(f, t)`` (WDM II writes :math:`S(t, f)`), samples through
:func:`wdm.grid.sanitize_psd`.  Cell times are :math:`t_n = n\Delta T`.

Numerics: every NumPy path is an operation-order-preserving transcription of its
source (wdm_dev ``evolutionary.py``, WT ``wdm_toys/matrix.py``, WG
``wdmtoysgpu/matrix.py``, GB ``galactic_binary/matrix.py``), so it is
bit-identical to it; the JAX paths agree to round-off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend import NUMPY, Backend, get_backend, pytree_dataclass
from .grid import (
    active_mask,
    delta_T,
    endpoint_weights,
    validate_dt,
    validate_grid,
)
from .noise import _psd_at
from .transform import (
    atom_spectrum,
    base_atom_spectra,
    forward_time_many,
    inverse_time_many,
)

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type CArray = NDArray[np.complexfloating]
type IntArray = NDArray[np.integer]
type PsdFtFunc = Callable[[Array, Array], Array]

__all__ = [
    "Eq45BlockSpec",
    "Eq45GramPlan",
    "Eq45Plan",
    "apply_time_mask",
    "draw_noise",
    "energy_fraction",
    "eq45_blocks_cheb",
    "eq45_blocks_gram",
    "eq45_coeffs_bessel",
    "eq45_coeffs_interp",
    "eq45_direct",
    "eq45_gram_gather",
    "eq45_gram_plan",
    "eq45_plan",
    "eq45_separable_assemble",
    "eq45_separable_kernels",
    "exact_covariance",
    "time_domain_covariance",
    "time_domain_factor",
    "wrap_cut_columns",
    "wrap_fractions",
]


# ---------------------------------------------------------------------------
# Time masks in the WDM domain
# ---------------------------------------------------------------------------


def energy_fraction(w2: Array, Nf: int, Nt: int, dt: float) -> Array:
    """``rho_p = sum_t w2(t) g_p(t)^2 / dt`` for every active cell ``p`` (band order).

    One FFT correlation per base atom (wdm_dev ``PackedGrid.energy_fraction``).
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = Nf * Nt
    m_arr, n_arr = np.nonzero(active_mask(Nf, Nt))
    base = base_atom_spectra(Nf, Nt, dt)
    rho = np.empty(m_arr.size)
    W2 = np.fft.rfft(w2)
    for (m, a), G in base.items():
        g2 = np.fft.irfft(G, n=N) ** 2
        corr = np.fft.irfft(W2 * np.conj(np.fft.rfft(g2)), n=N)  # sum_t w2[t+s] g2[t]
        sel = np.flatnonzero((m_arr == m) & (n_arr % 2 == a))
        shifts = (n_arr[sel] - a) * Nf
        rho[sel] = corr[shifts % N] / dt
    return rho


def apply_time_mask(Y: Array, chi: Array, Nf: int, Nt: int, dt: float) -> Array:
    """``M y`` for each packed row ``y`` of ``Y``, ``M = W chi W^T / dt``: back to time,
    mask, forward again (wdm_dev ``PackedGrid.apply_mask``)."""
    validate_grid(Nf, Nt)
    mask = active_mask(Nf, Nt)
    Y = np.atleast_2d(Y)
    full = np.zeros((Y.shape[0], Nf + 1, Nt))
    full[:, mask] = Y
    X = np.atleast_2d(inverse_time_many(full, Nf, Nt, dt) * chi)
    return forward_time_many(X, Nf, Nt, dt)[:, mask]


# ---------------------------------------------------------------------------
# The 0.1% wrap rule
# ---------------------------------------------------------------------------


def wrap_fractions(Nf: int, Nt: int, dt: float, *, bands: Array | None = None) -> Array:
    r"""Fraction of each atom's energy that wraps across the record boundary.

    For the cell :math:`(m, n)`, centred on sample :math:`n N_f`, the fraction of
    :math:`\sum_j g_{mn}(t_j)^2` at samples whose unwrapped position (measured from
    the centre along the shorter way round) lies outside :math:`[0, N)`.
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
    """``(leading, trailing)`` time columns to cut so that no kept atom of ``bands``
    (all by default) wraps more than ``tol`` of its energy (WDM II: ``tol = 1e-3``).

    The rule is geometric (it ignores the noise).  ``(6, 5)`` on the sinusoid grid
    ``(64, 256, 2**-10)``; ``(5, 4)`` on the 3- and 6-month GB analysis bands.

    Examples
    --------
    >>> import wdm
    >>> wdm.evolutionary.wrap_cut_columns(64, 256, 2**-10)
    (6, 5)
    """
    frac = np.nanmax(wrap_fractions(Nf, Nt, dt, bands=bands), axis=0)    # worst band per column
    bad = np.flatnonzero(frac > tol)
    half = Nt // 2
    lead = int(bad[bad < half].max()) + 1 if np.any(bad < half) else 0
    trail = Nt - int(bad[bad >= half].min()) if np.any(bad >= half) else 0
    return lead, trail


# ---------------------------------------------------------------------------
# The exact covariance (theta = 0) and noise draws -- dense, NumPy
# ---------------------------------------------------------------------------


def time_domain_factor(N: int, dt: float, Sn: PsdFtFunc) -> Array:
    r"""A square root :math:`G` of the exact time-domain covariance, :math:`\Sigma = G G^{\mathsf T}`.

    :math:`x(t_i) = \sum_k \sqrt{\eta_k\,\Delta f\,S(f_k, t_i)}\,[\xi_k \cos 2\pi f_k t_i
    + \zeta_k \sin 2\pi f_k t_i]`, :math:`t_i = i\,\delta t`, :math:`f_k = k/(N\delta t)`.
    Columns: cosine voices :math:`k = 0..N/2`, then sine voices :math:`k = 1..N/2-1`.
    :math:`O(N^2)` memory.
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
    r""":math:`\Sigma = G G^{\mathsf T}` (:func:`time_domain_factor`), ``(N, N)``."""
    G = time_domain_factor(N, dt, Sn)
    return G @ G.T


def draw_noise(N: int, dt: float, Sn: PsdFtFunc, rng: np.random.Generator) -> Array:
    """One realisation from the exact covariance; normals consumed cosine voices
    first, then sine voices, so a given ``rng`` state always gives the same series."""
    G = time_domain_factor(N, dt, Sn)
    z = rng.standard_normal(G.shape[1])
    return G @ z


def exact_covariance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc) -> Array:
    r"""Exact :math:`\Lambda = W_a \Sigma W_a^{\mathsf T}` on the active cells (band order),
    built in Gram form :math:`(W_a G)(W_a G)^{\mathsf T}`, so PSD by construction."""
    validate_grid(Nf, Nt)
    N = Nf * Nt
    G = time_domain_factor(N, dt, Sn)
    A = forward_time_many(np.ascontiguousarray(G.T), Nf, Nt, dt)[:, active_mask(Nf, Nt)].T
    Lam = A @ A.T
    return 0.5 * (Lam + Lam.T)


# ---------------------------------------------------------------------------
# WDM II Eq. (45): plans (host constants)
# ---------------------------------------------------------------------------


@pytree_dataclass(data=("R", "C", "combo", "d", "s", "B"), meta=("m", "m2"))
@dataclass(frozen=True, eq=False)
class Eq45BlockSpec:
    """One band block ``(m, m2)``, ``m2 in {m, m+1}``, of an :class:`Eq45Plan`.

    ``R``/``C`` (row/column packed cells), ``combo = 2a + b`` (base-atom parities),
    ``d`` (circular column lag), ``s = n + n'`` (mid-time index; ``t_bar = s dT/2``)
    and ``B = {combo: (k, eta_k G_ma(k) G*_m2b(k) / T)}`` over the support bins:
    the entries of WT ``TriCovariance._block_spec``'s dict, as fields.

    A JAX pytree: the band indices ``m``/``m2`` are static (they pick the block's
    slot and whether it is symmetrised); every array is a leaf.  ``R``/``C`` are
    leaves too, because JAX rejects arrays as static fields; only their shapes,
    which stay static under tracing, matter to a kernel.
    """

    m: int
    m2: int
    R: IntArray
    C: IntArray
    combo: IntArray
    d: IntArray
    s: IntArray
    B: dict[int, tuple[IntArray, CArray]]


@pytree_dataclass(
    data=("f", "eta", "m", "n", "t", "base", "keep", "cells", "pos", "band_cells", "Tp",
          "specs", "k_used"),
    meta=("Nf", "Nt", "dt", "N", "T", "K", "P", "t_max", "fold_has_duplicates"),
)
@dataclass(frozen=True, eq=False)
class Eq45Plan:
    """Host (NumPy float64) constants of Eq. (45) on the kept cells of one grid.

    ``specs`` holds one :class:`Eq45BlockSpec` per band block ``(m, m2)``,
    ``m2 in {m, m+1}``.  Mirrors WT ``TriCovariance.__init__``/``_block_spec``
    line for line.

    A JAX pytree: the grid scalars (``Nf, Nt, dt, N, T, K, P, t_max``) and
    ``fold_has_duplicates`` are static; every array is a leaf.

    .. warning::
       Under ``jax.jit`` pass the plan as an **argument**
       (``jax.jit(lambda plan, c: eq45_blocks_cheb(plan, c))(plan, coeffs)``);
       never close over it.  A closed-over plan bakes every array (219 MB at
       64 x 256) into the compiled program: 404 MB of HLO text and ~25 s to
       lower and compile, against 2.5 MB and ~4 s as an argument (measured,
       CPU).  ``jax.device_put(plan)`` once keeps the arrays on the device
       between calls.
    """

    Nf: int
    Nt: int
    dt: float
    N: int
    T: float
    K: int
    P: int
    t_max: float
    f: Array
    eta: Array
    m: IntArray
    n: IntArray
    t: Array
    base: dict[tuple[int, int], CArray] = field(repr=False)
    keep: BoolArray = field(repr=False)
    cells: IntArray = field(repr=False)
    pos: IntArray = field(repr=False)
    band_cells: list[IntArray] = field(repr=False)
    Tp: Array = field(repr=False)
    specs: list[Eq45BlockSpec] = field(repr=False)
    k_used: IntArray = field(repr=False)
    fold_has_duplicates: bool = False


def _block_spec(Nf: int, Nt: int, T: float, n: IntArray, eta: Array,
                base: dict[tuple[int, int], CArray], band_cells: list[IntArray],
                m: int, m2: int) -> dict[str, Any]:
    R, C = band_cells[m], band_cells[m2]
    nr, nc = n[R], n[C]
    a = (nr % 2)[:, None] + 0 * nc[None, :]
    b = 0 * nr[:, None] + (nc % 2)[None, :]
    combo = 2 * a + b                                              # (nR, nC)
    d = ((nr[:, None] - a) - (nc[None, :] - b)) % Nt
    s = nr[:, None] + nc[None, :]
    B = {}
    for aa in (0, 1):
        for bb in (0, 1):
            Ga, Gb = base.get((m, aa)), base.get((m2, bb))
            if Ga is None or Gb is None:
                continue
            k = np.flatnonzero((np.abs(Ga) > 0) & (np.abs(Gb) > 0))
            B[2 * aa + bb] = (k, eta[k] * Ga[k] * np.conj(Gb[k]) / T)
    return dict(m=m, m2=m2, R=R, C=C, combo=combo, d=d, s=s, B=B)


def eq45_plan(Nf: int, Nt: int, dt: float, keep: BoolArray | None = None, *,
              P: int = 24, t_max: float | None = None) -> Eq45Plan:
    """Static plan of Eq. (45) on the cells ``keep`` (a mask over the ``N`` packed
    cells, band order; all by default), Chebyshev order ``P`` in time on
    ``t_bar in [0, t_max]`` (default: this grid's ``T = N dt``).

    Pair mid-times of the kept cells must lie in ``[0, t_max]``.  A padded grid
    passes its extended ``t_max``; the spectrum's time dependence is then
    continued over the padding by the caller (``r = t_max / T_obs`` in
    :func:`eq45_coeffs_bessel`).
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    Nf, Nt, dt, P = int(Nf), int(Nt), float(dt), int(P)
    N = Nf * Nt
    T = N * dt
    mask = active_mask(Nf, Nt)
    m, n = np.nonzero(mask)                                       # band-major
    t = n * delta_T(Nf, dt)                                       # cell times
    K = N // 2 + 1
    f = np.fft.rfftfreq(N, d=dt)
    eta = np.ones(K)
    eta[[0, -1]] = 0.5
    base = base_atom_spectra(Nf, Nt, dt)
    keep = np.ones(m.size, bool) if keep is None else np.asarray(keep)
    if keep.dtype != np.bool_:   # an index array (e.g. np.arange(N)) would silently drop cell 0
        raise TypeError("keep must be a boolean mask over the packed cells")
    if keep.shape != (m.size,):
        raise ValueError(f"keep must be a mask over the {m.size} packed cells; got {keep.shape}")
    t_max = float(T if t_max is None else t_max)
    cells = np.flatnonzero(keep)
    pos = -np.ones(m.size, int)
    pos[cells] = np.arange(cells.size)
    band_cells = [cells[m[cells] == mm] for mm in range(Nf + 1)]
    # Chebyshev basis on the pair mid-times t_bar = s dT / 2, s = n + n'
    s_grid = np.arange(2 * Nt - 1)
    u = 2.0 * (s_grid * 0.5 * delta_T(Nf, dt) / t_max - 0.5)
    used = s_grid <= 2 * int(n[cells].max())
    if np.any(np.abs(u[used]) > 1 + 1e-12):
        raise ValueError("pair mid-times must lie inside [0, t_max]")
    Tp = np.empty((P, s_grid.size))
    Tp[0] = 1.0
    if P > 1:
        Tp[1] = u
    for q in range(2, P):
        Tp[q] = 2 * u * Tp[q - 1] - Tp[q - 2]
    specs = [Eq45BlockSpec(**_block_spec(Nf, Nt, T, n, eta, base, band_cells, mm, m2))
             for mm in range(Nf + 1) for m2 in (mm, mm + 1) if m2 <= Nf]
    k_all = [k for spec in specs for (k, _) in spec.B.values()]
    k_used = np.unique(np.concatenate(k_all)) if k_all else np.zeros(0, int)
    dup = any(np.unique(k % Nt).size != k.size for spec in specs for (k, _) in spec.B.values())
    return Eq45Plan(Nf, Nt, dt, N, T, K, P, t_max, f, eta, m, n, t, base, keep, cells, pos,
                    band_cells, Tp, specs, k_used, dup)


# ---------------------------------------------------------------------------
# Eq. (45): Chebyshev-in-time coefficients and blocks
# ---------------------------------------------------------------------------


def eq45_coeffs_bessel(L: Array, alpha_c: Any, alpha_d: Any, P: int, *, r: float,
                       be: Backend | None = None) -> Array:
    r"""``(P, len(L))``: Chebyshev coefficients of :math:`e^{\alpha(\bar t) L}` in
    :math:`u = 2\bar t/t_{\max} - 1` for the log-linear family
    :math:`\alpha(t) = \alpha_c + \alpha_d (t/T_{\rm obs} - 1/2)`, exactly (Bessel I):
    with :math:`r = t_{\max}/T_{\rm obs}`,
    :math:`\alpha = \alpha_c + \alpha_d (r/2 - 1/2) + (\alpha_d r/2)\,u`.

    ``r`` is a required keyword: ``plan.t_max / T_obs``, i.e. ``1.0`` on the
    data grid and ``Nt' / Nt`` on a padded grid (``272/256`` for the sinusoid
    pad).  It used to default to ``1.0``, which is silently wrong on a padded
    plan.  WT ``TriCovariance._cheb_weights`` verbatim (the spec's 4-argument
    signature omitted ``r``).  On JAX the Bessel functions are the 40-term
    series of :meth:`wdm.backend.Backend.bessel_i`.
    """
    be = be or get_backend(L, alpha_c, alpha_d)
    xp = be.xp
    a = 0.5 * alpha_d * r * L
    c = be.bessel_i(P, a)
    c = be.set(c, slice(1, None), c[1:] * 2.0, copy=False)          # c[1:] *= 2.0
    return c * xp.exp((alpha_c + alpha_d * (0.5 * r - 0.5)) * L)[None, :]


def eq45_coeffs_interp(Sn: PsdFtFunc, f: Array, t_max: float, P: int, *,
                       k: IntArray | None = None) -> Array:
    r"""``(P, len(f))``: Chebyshev coefficients :math:`c_p(f_k)` of a general
    :math:`S(f_k, \bar t)` in :math:`u = 2\bar t/t_{\max} - 1`, by interpolation at
    the ``P`` Chebyshev nodes; zero outside the bins ``k`` (all by default; pass
    ``plan.k_used``).  wdm_dev ``TriCovariance.chebyshev_coefficients`` verbatim.
    NumPy (the PSD is a host callable).
    """
    f = np.asarray(f, float)
    k = np.arange(f.size) if k is None else np.asarray(k, int)
    j = np.arange(P)
    x = np.cos(np.pi * (j + 0.5) / P)                                 # nodes in u
    t_nodes = 0.5 * (x + 1.0) * t_max
    S = _psd_at(Sn, f[k][None, :], t_nodes[:, None], "eq45_coeffs_interp")   # (P, |k|)
    Cmat = np.cos(np.pi * np.outer(np.arange(P), j + 0.5) / P)          # (p, j)
    c = (2.0 / P) * (Cmat @ S)
    c[0] *= 0.5
    out = np.zeros((P, f.size))
    out[:, k] = c
    return out


def eq45_blocks_cheb(plan: Eq45Plan, coeffs: Array | Callable[[IntArray], Array], *,
                     be: Backend | None = None) -> tuple[list[Array], list[Array]]:
    """Diagonal (symmetrised) and upper band blocks of Eq. (45) from Chebyshev coefficients.

    ``coeffs`` is either a ``(P, K)`` array over every rfft bin
    (:func:`eq45_coeffs_interp`; read as ``coeffs[:, k]``) or a callable
    ``k -> (P, len(k))`` evaluated per block (the WT call pattern:
    ``lambda k: eq45_coeffs_bessel(L[k], ac, ad, P, r=r)``).  Each block costs
    one length-``Nt`` FFT per order.  The NumPy path is WT ``TriCovariance.blocks``
    / wdm_dev ``TriCovariance.blocks`` line for line (``np.add.at`` fold).
    Returns ragged lists (``diag[m]``: kept cells of band ``m`` squared).

    Under ``jax.jit`` pass ``plan`` as an argument (see :class:`Eq45Plan`) and
    ``coeffs`` as the ``(P, K)`` array (a callable would index host arrays with
    the traced ``k``).  The backend follows ``coeffs`` and the plan's arrays.
    """
    if be is None:
        be = get_backend(plan.Tp) if callable(coeffs) else get_backend(coeffs, plan.Tp)
    xp = be.xp
    P, Nt = plan.P, plan.Nt
    Tp = be.asarray(plan.Tp)
    diag: list[Array] = [None] * (plan.Nf + 1)  # type: ignore[list-item]
    up: list[Array] = [None] * plan.Nf  # type: ignore[list-item]
    for spec in plan.specs:
        F = xp.zeros((4, P, Nt), complex)
        for cb, (k, Bk) in spec.B.items():
            Ck = coeffs(k) if callable(coeffs) else coeffs[:, k]
            V = Ck * Bk[None, :]                                        # (P, |k|)
            folded = xp.zeros((P, Nt), complex)
            folded = be.add_at(folded, (slice(None), k % Nt), V, copy=False)
            F = be.set(F, cb, be.fft.fft(folded, axis=1), copy=False)   # F_p(d)
        Fg = F[spec.combo, :, spec.d]                                   # (nR, nC, P)
        Tg = Tp[:, spec.s]                                              # (P, nR, nC)
        blk = xp.real(xp.einsum("ijp,pij->ij", Fg, Tg))
        if spec.m2 == spec.m:
            diag[spec.m] = 0.5 * (blk + blk.T)
        else:
            up[spec.m] = blk
    return diag, up


# ---------------------------------------------------------------------------
# Eq. (45): Gram form for the log-linear family (WG ``_gram_setup``/``_tri_blocks``)
# ---------------------------------------------------------------------------


@pytree_dataclass(data=("sup", "slot", "gG", "gW", "gT", "uI1", "uI2", "uW"),
                  meta=("nb", "B", "Kb"))
@dataclass(frozen=True, eq=False)
class Eq45GramPlan:
    """Host constants of the Gram form, bands laid out ``(nb, B)`` with dummy slots.

    ``slot[m, i]`` is the kept-cell position of slot ``i`` of band ``m`` (``-1`` for
    a dummy); ``gG`` the cells' atom spectra on their band's support bins
    ``sup[m]``; ``gW = eta/T``; ``gT`` the cell times; ``uI1``/``uI2``/``uW`` the
    overlap bins of adjacent bands.  WG ``MatrixGPU._gram_setup`` line for line.

    A JAX pytree: the layout sizes ``nb``, ``B``, ``Kb`` are static; every array
    is a leaf.

    .. warning::
       Under ``jax.jit`` pass the plan as an **argument**
       (``jax.jit(lambda gp, a: eq45_blocks_gram(gp, a, gL))(gplan, alpha_cells)``);
       never close over it.  A closed-over plan bakes ``gG`` (51 MB at 64 x 256)
       into the compiled program: 102 MB of HLO text, against 0.2 MB as an
       argument (measured, CPU).
    """

    nb: int
    B: int
    Kb: int
    sup: list[IntArray] = field(repr=False)
    slot: IntArray = field(repr=False)
    gG: CArray = field(repr=False)
    gW: Array = field(repr=False)
    gT: Array = field(repr=False)
    uI1: NDArray[np.int32] = field(repr=False)
    uI2: NDArray[np.int32] = field(repr=False)
    uW: Array = field(repr=False)


def eq45_gram_plan(plan: Eq45Plan) -> Eq45GramPlan:
    r"""Gram constants for :func:`eq45_blocks_gram`.

    For :math:`S(t, f) = e^{\alpha(t) L(f)}` with :math:`\alpha` linear in
    :math:`t`, :math:`S(\bar t_{pq}) = \sqrt{S(t_p) S(t_q)}` exactly, so
    :math:`\Lambda_{pq} = \frac1T \mathrm{Re}\sum_k \eta_k v_p(k) v_q^*(k)` with
    :math:`v_p = e^{\alpha(t_p) L/2}\hat g_p` -- one matrix product per band block.
    """
    nb = len(plan.band_cells)
    B = max(len(c) for c in plan.band_cells)
    slot = np.full((nb, B), -1)
    for mm, c in enumerate(plan.band_cells):
        slot[mm, :len(c)] = plan.pos[c]
    sup = []
    for mm in range(nb):
        ks = [np.flatnonzero(np.abs(G) > 0) for (m2, _), G in plan.base.items() if m2 == mm]
        sup.append(np.unique(np.concatenate(ks)) if ks else np.zeros(0, int))
    Kb = max(len(k) for k in sup)
    gG = np.zeros((nb, B, Kb), complex)
    gW = np.zeros((nb, Kb))
    gT = np.zeros((nb, B))
    for mm, cells in enumerate(plan.band_cells):
        k = sup[mm]
        for i, p in enumerate(cells):
            gG[mm, i, :len(k)] = atom_spectrum(int(plan.m[p]), int(plan.n[p]), plan.Nf,
                                               plan.Nt, plan.dt, base=plan.base)[k]
            gT[mm, i] = plan.t[p]
        gW[mm, :len(k)] = plan.eta[k] / plan.T
    ov = [np.intersect1d(sup[mm], sup[mm + 1], return_indices=True) for mm in range(nb - 1)]
    Ko = max(1, max(len(k) for k, _, _ in ov))
    uI1 = np.zeros((nb - 1, Ko), np.int32)
    uI2 = np.zeros((nb - 1, Ko), np.int32)
    uW = np.zeros((nb - 1, Ko))
    for mm, (k, i1, i2) in enumerate(ov):
        uI1[mm, :len(k)], uI2[mm, :len(k)] = i1, i2
        uW[mm, :len(k)] = plan.eta[k] / plan.T
    return Eq45GramPlan(nb, B, Kb, sup, slot, gG, gW, gT, uI1, uI2, uW)


def eq45_gram_gather(gplan: Eq45GramPlan, values: Array) -> Array:
    """``(nb, Kb)``: a per-rfft-bin array (e.g. ``L(f)``) on each band's support bins,
    zero-padded (WG ``gL[m, :len(k)] = L[k]``)."""
    out = np.zeros((gplan.nb, gplan.Kb))
    for mm, k in enumerate(gplan.sup):
        out[mm, :len(k)] = values[k]
    return out


def eq45_blocks_gram(gplan: Eq45GramPlan, alpha_cells: Array, gL: Array, *,
                     be: Backend | None = None) -> tuple[Array, Array]:
    """Eq. (45) blocks in the Gram form: diagonal ``(nb, B, B)`` and upper ``(nb-1, B, B)``.

    ``alpha_cells`` is ``alpha(t_p)`` per slot, ``(nb, B)``; the caller computes it
    from ``gplan.gT`` with the **data record's** ``T_obs`` (WG:
    ``ac + ad * (gT / T_obs - 0.5)``).  ``gL`` is :func:`eq45_gram_gather` of
    ``L(f)``.  Dummy slots give zero rows and columns.  WG ``_tri_blocks`` verbatim.
    Under ``jax.jit`` pass ``gplan`` as an argument (see :class:`Eq45GramPlan`).
    The backend follows ``alpha_cells``, ``gL`` and the plan's arrays.
    """
    be = be or get_backend(alpha_cells, gL, gplan.gG)
    xp = be.xp
    gG, gW = be.asarray(gplan.gG), be.asarray(gplan.gW)
    uI1, uI2, uW = be.asarray(gplan.uI1), be.asarray(gplan.uI2), be.asarray(gplan.uW)
    al = be.asarray(alpha_cells)
    V = xp.exp(0.5 * al[:, :, None] * be.asarray(gL)[:, None, :]) * gG          # (nb, B, Kb)
    diag = xp.real(xp.einsum("bik,bjk->bij", V * gW[:, None, :], xp.conj(V)))
    Va = xp.take_along_axis(V[:-1], uI1[:, None, :], axis=2) * uW[:, None, :]
    Vb = xp.take_along_axis(V[1:], uI2[:, None, :], axis=2)
    up = xp.real(xp.einsum("bik,bjk->bij", Va, xp.conj(Vb)))
    return diag, up


# ---------------------------------------------------------------------------
# Eq. (45): separable spectra S_c(f, t) = s_I(f) + g(t) s_G(f)  (GB ``tri_kernels``)
# ---------------------------------------------------------------------------


def eq45_separable_kernels(U: CArray, s: Array) -> Array:
    r""":math:`K^X = \tfrac12 \mathrm{Re}(U\,\mathrm{diag}(s)\,U^{\mathsf H})` in real arithmetic.

    ``U`` (cells x window bins) holds the unit-norm atoms; ``s`` a stationary
    spectrum on the window bins.  GB ``matrix._kernel`` verbatim (host NumPy,
    built once per noise component).
    """
    Ur, Ui = U.real, U.imag
    return 0.5 * ((Ur * s) @ Ur.T + (Ui * s) @ Ui.T)


def eq45_separable_assemble(KI: Array, KG: Array, g_half: Array, half_index: IntArray, *,
                            be: Backend | None = None) -> Array:
    r"""Eq. (45) for :math:`S = s_I + g(t)\,s_G`: :math:`K^I + g(\bar t)\,K^G`.

    ``g_half`` is the envelope on the half-step grid :math:`t = j\,\Delta T/2`,
    ``half_index = n_p + n_q`` (non-circular), so ``g_half[half_index]`` is
    :math:`g(\bar t_{pq})`.  GB ``covariance_builder``: ``KI[c] + g[half] * KG[c]``.
    """
    be = be or get_backend(g_half)
    return be.asarray(KI) + be.asarray(g_half)[be.asarray(half_index)] * be.asarray(KG)


# ---------------------------------------------------------------------------
# Eq. (45): the direct sum (oracle)
# ---------------------------------------------------------------------------


def _eq45_pairs(plan: Eq45Plan) -> tuple[IntArray, IntArray]:
    P, Q = [], []
    for mm in range(plan.Nf + 1):
        A = plan.band_cells[mm]
        for m2 in (mm, mm + 1):
            if m2 > plan.Nf:
                continue
            B = plan.band_cells[m2]
            if not A.size or not B.size:
                continue
            ok = np.ones((A.size, B.size), bool)
            if m2 == mm:
                ok = A[:, None] <= B[None, :]
            i, j = np.nonzero(ok)
            P.append(A[i])
            Q.append(B[j])
    return np.concatenate(P), np.concatenate(Q)


def eq45_direct(plan: Eq45Plan, Sn: PsdFtFunc) -> Array:
    """Eq. (45) on the kept cells by the direct sum over rfft bins (dense; tests and
    small grids).  wdm_dev ``TriCovariance.pairs/values/dense`` verbatim."""
    g = plan
    Pp, Qq = _eq45_pairs(plan)
    out = np.empty(Pp.size)
    tbar = 0.5 * (g.t[Pp] + g.t[Qq])
    k = np.arange(g.K)
    key = ((g.m[Pp] * 2 + g.n[Pp] % 2) * (2 * g.Nf + 4) + g.m[Qq] * 2 + g.n[Qq] % 2)
    for kk in np.unique(key):
        sel = np.flatnonzero(key == kk)
        p0, q0 = Pp[sel[0]], Qq[sel[0]]
        mp, ap, mq, aq = int(g.m[p0]), int(g.n[p0] % 2), int(g.m[q0]), int(g.n[q0] % 2)
        Gp, Gq = g.base[mp, ap], g.base[mq, aq]
        s = np.flatnonzero((np.abs(Gp) > 0) & (np.abs(Gq) > 0))
        if not s.size:
            out[sel] = 0.0
            continue
        prod = g.eta[s] * Gp[s] * np.conj(Gq[s]) / g.T
        d = (g.n[Pp[sel]] - ap) - (g.n[Qq[sel]] - aq)
        ph = np.exp(-2j * np.pi * np.outer(d, k[s]) / g.Nt)
        S = _psd_at(Sn, g.f[s][None, :], tbar[sel][:, None], "eq45_direct")
        out[sel] = np.real(np.sum(S * ph * prod[None, :], axis=1))
    n = plan.cells.size
    C = np.zeros((n, n))
    C[plan.pos[Pp], plan.pos[Qq]] = out
    C[plan.pos[Qq], plan.pos[Pp]] = out
    return C
