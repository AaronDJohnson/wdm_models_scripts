"""Maps out of band-limited Fourier bins: zero padding, cropping, WDM cells (GB, spec §4.2).

Ported from ``toy_models/galactic_binary/matrix.py`` (frozen): ``_geometric``,
``pad_maps``, ``crop_maps`` verbatim (L270-306), the cell matrices of ``Grid.dense_atoms``
(L222-227) and ``_cells`` (L320-332, with ``wdm_full.analyse`` L142-145 and
``wdm_bands._phase`` L89-95), and the data map ``cells`` (L818-820).  The atoms are
``wdm.transform.band_atoms`` (bit-identical to ``geometry.band_atoms``, S1 Task C6).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import wdm
from wdm.backend import pytree_dataclass
from wdm.filter import phi_centered
from wdm.grid import carrier_phase
from wdm.transform import band_atoms

from ..domains import FourierDomain, WDMDomain
from .base import _BUILD, LinearMap

__all__ = ["FourierCrop", "FourierToWDM", "FourierZeroPad", "crop_maps", "pad_maps"]


# ---------------------------------------------------------------------------
# Dirichlet kernels (GB matrix.py:270-306, verbatim)
# ---------------------------------------------------------------------------


def _geometric(num, den: int, n: int, n_nu_mod2):
    r"""``sum_{s<n} exp(2 pi i s nu)`` with ``nu = num / den`` (integer ``num``), given
    ``n nu mod 2`` computed exactly by the caller."""
    num = np.asarray(num, np.int64)
    nu = num / den
    exact = np.mod(num, den) == 0
    ratio = np.sin(np.pi * n_nu_mod2) / np.where(exact, 1.0, np.sin(np.pi * nu))
    return np.where(exact, float(n), np.exp(1j * np.pi * (n_nu_mod2 - nu)) * ratio)


def pad_maps(N: int, Np: int, k_out, j_in):
    r"""Bins ``k_out`` of the record zero-padded from ``N`` to ``Np`` samples, from the
    record's bins ``j_in``: ``z'_k = A z + B conj(z)``, both ``(len(k_out), len(j_in))``,
    ``A_kj = (1/N) sum_{s<N} e^{2 pi i s (j/N - k/Np)}`` and ``B`` the same at ``-j``."""
    k = np.asarray(k_out, np.int64)[:, None]
    out = []
    for j in (np.asarray(j_in, np.int64)[None, :], -np.asarray(j_in, np.int64)[None, :]):
        num = j * Np - k * N                                       # nu = num / (N Np)
        # N nu = j - k N / Np, mod 2
        n_nu = np.mod(np.mod(j, 2) * Np - np.mod(k * N, 2 * Np), 2 * Np) / Np
        out.append(_geometric(num, N * Np, N, n_nu) / N)
    return out[0], out[1]


def crop_maps(N: int, start: int, Nc: int, k_out, j_in):
    r"""Bins ``k_out`` of the record cropped to samples ``[start, start + Nc)``, from the
    record's bins ``j_in``: ``z^c_k = A z + B conj(z)``,
    ``A_kj = (1/N) e^{2 pi i j start / N} sum_{s<Nc} e^{2 pi i s (j/N - k/Nc)}``."""
    k = np.asarray(k_out, np.int64)[:, None]
    out = []
    for j in (np.asarray(j_in, np.int64)[None, :], -np.asarray(j_in, np.int64)[None, :]):
        num = j * Nc - k * N                                       # nu = num / (N Nc)
        # Nc nu = j Nc / N - k, mod 2
        n_nu = np.mod(np.mod(j * Nc, 2 * N) - np.mod(k, 2) * N, 2 * N) / N
        shift = np.exp(2j * np.pi * np.mod(j * start, N) / N)
        out.append(shift * _geometric(num, N * Nc, Nc, n_nu) / N)
    return out[0], out[1]


class _WidelyLinear(LinearMap):
    """``z -> scale (A z + B conj z)`` (kind ``"widely"``): there is no complex matrix
    ``M`` with ``M z`` equal to it, so :meth:`rows` (and :meth:`LinearMap.dense`) refuse
    instead of returning ``conj(A^H e + B^T e)``; read ``.A``, ``.B`` (and ``.scale``).
    ``push`` raises in :class:`LinearMap`."""

    kind = "widely"

    def rows(self, idx):
        raise TypeError(f"{type(self).__name__} is widely linear (A z + B conj z): it has no "
                        "complex row matrix; use its .A, .B and .scale")


@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "scale"))
@dataclass(frozen=True, eq=False)
class FourierZeroPad(_WidelyLinear):
    """Bins ``dst`` of the record zero-padded from ``src.N`` to ``dst.N`` samples, from the
    bins ``src``: ``z' = scale (A z + B conj z)``, ``scale = sqrt(N / N')`` (kind
    ``"widely"``).  ``A, B`` are :func:`pad_maps`; ``scale`` keeps the padded record's
    unit-norm cells in the record's cell units (GB ``variant``, matrix.py:373-377)."""

    src: FourierDomain
    dst: FourierDomain
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)
    scale: float = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.A is _BUILD:
            s, d = self.src, self.dst
            if d.N <= s.N or d.dt != s.dt or d.units != s.units:
                raise ValueError("FourierZeroPad needs a longer record on the same dt and units")
            A, B = pad_maps(s.N, d.N, d.k0 + np.arange(d.n), s.k0 + np.arange(s.n))
            object.__setattr__(self, "A", A)
            object.__setattr__(self, "B", B)
            object.__setattr__(self, "scale", float(np.sqrt(s.N / d.N)))

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        return self.scale * ((z @ self.A.T) + (be.xp.conj(z) @ self.B.T))

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        return self.scale * ((y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B))


@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "start"))
@dataclass(frozen=True, eq=False)
class FourierCrop(_WidelyLinear):
    """Bins ``dst`` of the record cropped to samples ``[start, start + dst.N)``, from the
    bins ``src``: ``z^c = A z + B conj z`` (:func:`crop_maps`; GB ``whittle_map`` and
    ``mask_rows``).  ``apply`` is the template call site ``(h @ A.T) + (conj(h) @ B.T)``
    (matrix.py:897); :meth:`apply_data` the data call site (matrix.py:892)."""

    src: FourierDomain
    dst: FourierDomain
    start: int = 0
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)

    scale = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
        if self.A is _BUILD:
            s, d = self.src, self.dst
            if not (0 <= self.start and self.start + d.N <= s.N) or d.dt != s.dt or d.units != s.units:
                raise ValueError("FourierCrop needs [start, start + dst.N) inside the record")
            A, B = crop_maps(s.N, self.start, d.N, d.k0 + np.arange(d.n), s.k0 + np.arange(s.n))
            object.__setattr__(self, "A", A)
            object.__setattr__(self, "B", B)

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        return (z @ self.A.T) + (be.xp.conj(z) @ self.B.T)

    def apply_data(self, z, be=None):
        """The oracle's DATA call site ``(A @ z.T + B @ conj(z).T).T`` (matrix.py:892): the
        same map as :meth:`apply` in the other matmul orientation (<= 2e-15 apart)."""
        be = self._be(z, be)
        z = be.asarray(z)
        return (self.A @ z.T + self.B @ be.xp.conj(z).T).T

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        return (y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B)


# ---------------------------------------------------------------------------
# The cell geometry of a set of interior bands
# ---------------------------------------------------------------------------


def _band_phase(bands: np.ndarray, Nt: int, phi: np.ndarray) -> np.ndarray:
    """``(nb, Nt)`` carrier sign x quadrature x ``(-1)^n Nt / |phi|`` (GB ``wdm_bands._phase``)."""
    n = np.arange(Nt)
    out = np.empty((bands.size, Nt), complex)
    for b, m in enumerate(bands):
        quad = np.where((int(m) + n) % 2 == 1, -1j, 1.0)
        out[b] = carrier_phase(int(m), Nt) * quad * (-1.0) ** n * Nt / np.linalg.norm(phi)
    return out


def _analyse(phase: np.ndarray, phi: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``W_b X_b`` for a stack of bands, ``X`` ``(nb, Nt, k)`` (GB ``wdm_full.analyse``)."""
    Y = np.fft.ifft(np.asarray(phi)[None, :, None] * X, axis=1)
    return np.asarray(phase)[:, :, None] * Y


def _geometry(dom: WDMDomain) -> dict:
    """Bands, window, per-cell positions of a cell set of interior bands (GB ``make_grid``)."""
    cells = dom.cells
    g = cells.grid
    Nt = g.Nt
    m = np.asarray(cells.m, dtype=np.int64)
    n = np.asarray(cells.n, dtype=np.int64)
    bands = np.unique(m)
    if bands.min() < 1 or bands.max() > g.Nf - 1:
        raise ValueError(f"FourierToWDM needs interior bands 1..{g.Nf - 1}; got {bands.tolist()}")
    h = Nt // 2
    g0 = h * int(bands[0]) - h                                         # absolute bin of window bin 0
    G = h * int(bands[-1] - bands[0]) + Nt
    idx = (h * (bands - bands[0]))[:, None] + np.arange(Nt)[None, :]   # (nb, Nt) window index
    b_of = np.searchsorted(bands, m)
    sel = b_of * Nt + n                                                # row of the full band grid
    return {"bands": bands, "g0": g0, "G": G, "idx": idx, "sel": sel, "Nt": Nt}


def _dense_atoms(geo: dict, Nf: int, dt: float) -> np.ndarray:
    """``(nb Nt, G)``: row ``b Nt + n`` is the unit-norm atom of cell ``(bands[b], n)`` on the
    window bins (GB ``Grid.dense_atoms``, matrix.py:222-227)."""
    bands, Nt, G, idx = geo["bands"], geo["Nt"], geo["G"], geo["idx"]
    atoms = band_atoms(bands, Nf, Nt, dt)
    U = np.zeros((bands.size * Nt, G), complex)
    for b in range(bands.size):
        U[b * Nt:(b + 1) * Nt, idx[b]] = atoms[b]
    return U


@pytree_dataclass(data=("C",), meta=("src", "dst", "impl", "lo", "width", "restrict"))
@dataclass(frozen=True, eq=False)
class FourierToWDM(LinearMap):
    """Cells of interior bands from band-limited bins: ``cells = Re(C z)`` (kind ``"cells"``).

    ``src`` is the ``FourierDomain`` the data live on (the GB store, or the band window
    itself); the cells read the window bins ``[g0, g0 + G)``, ``g0 = (m_0 - 1) Nt/2``, which
    ``src`` must cover -- or, with ``restrict=True``, ``src`` may be a sub-window of it and
    the atoms are truncated to ``src`` (the GB exact rows on the analysis window,
    matrix.py:686).  ``C`` holds the unit-norm atoms (GB ``Variant.C``):
    ``impl="dense"`` evaluates ``Re(z[..., lo:lo+width] @ C.T)`` (GB ``cells``, the golden
    default); ``impl="fft"`` evaluates ``wdm.forward_bands`` (opt-in, <= 1e-12 relative).
    For a record ``z = rfft(x)`` the cells are ``sqrt(N / (2 dt))`` times
    ``TimeToWDM.apply(x)`` on the same cells (``WDMDomain.scale`` records that factor
    together with the data's own normalisation; it is not used in arithmetic).
    """

    src: FourierDomain
    dst: WDMDomain
    impl: str = "dense"
    restrict: bool = field(default=False, kw_only=True)
    lo: int = field(default=_BUILD, kw_only=True)
    width: int = field(default=_BUILD, kw_only=True)
    C: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.C is not _BUILD:
            return
        if self.impl not in ("dense", "fft"):
            raise ValueError(f"impl must be 'dense' or 'fft'; got {self.impl!r}")
        g = self.dst.cells.grid
        if (self.src.N, self.src.dt) != (g.N, g.dt):
            raise ValueError(f"bins of an N={self.src.N} record cannot feed cells of an N={g.N} grid")
        geo = _geometry(self.dst)
        g0, G = geo["g0"], geo["G"]
        k0, n = self.src.k0, self.src.n
        U = _dense_atoms(geo, g.Nf, g.dt)[geo["sel"]]                 # (P, G), GB V.U
        if k0 <= g0 and g0 + G <= k0 + n:
            lo, width, C = g0 - k0, G, U
        elif self.restrict and g0 <= k0 and k0 + n <= g0 + G:
            if self.impl != "dense":
                raise ValueError("restrict=True truncates the atoms: impl='dense' only")
            lo, width, C = 0, n, U[:, k0 - g0:k0 - g0 + n]
        else:
            raise ValueError(f"the bins [{k0}, {k0 + n}) do not cover the band windows "
                             f"[{g0}, {g0 + G}) (pass restrict=True for a sub-window)")
        object.__setattr__(self, "lo", int(lo))
        object.__setattr__(self, "width", int(width))
        object.__setattr__(self, "C", C)

    kind = "cells"

    @property
    def bands(self) -> np.ndarray:
        """The interior bands of the cells, ascending."""
        return _geometry(self.dst)["bands"]

    @property
    def window(self) -> FourierDomain:
        """The band windows' bins ``[g0, g0 + G)`` of the record (480 / 876 / 1800 for the GB)."""
        geo = _geometry(self.dst)
        return FourierDomain(self.src.N, self.src.dt, geo["g0"], geo["G"], self.src.units)

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        if self.impl == "dense":
            return be.xp.real(z[..., self.lo:self.lo + self.width] @ self.C.T)
        g = self.dst.cells.grid
        geo = _geometry(self.dst)
        w = wdm.transform.forward_bands(z, self.src.k0, geo["bands"], g.Nf, g.Nt, g.dt, be=be)
        return w.reshape(*w.shape[:-2], -1)[..., geo["sel"]]

    def adjoint(self, w, be=None):
        """``C^H w`` on ``src`` (complex), the real-inner-product adjoint."""
        be = self._be(w, be)
        w = be.asarray(w)
        part = w @ be.xp.conj(self.C)
        out = be.xp.zeros((*w.shape[:-1], self.src.n), dtype=complex)
        return be.set(out, (..., slice(self.lo, self.lo + self.width)), part, copy=False)

    def rows(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        C = np.asarray(self.C)
        out = np.zeros((idx.size, self.src.n), complex)
        out[:, self.lo:self.lo + self.width] = C[idx]
        return out

    def push_rows(self, Mp, idx, be=None):
        """``1/2 Re((C[idx] @ M') @ conj(C).T)`` (GB ``exact_rows_builder``, matrix.py:686-693);
        ``M'`` is ``(src.n, src.n)``; the ``width`` bins ``C`` reads are sliced out of it."""
        be = be or self._be(Mp, None)
        if Mp.shape != (self.src.n, self.src.n):
            raise ValueError(f"M' must be ({self.src.n}, {self.src.n}) on src's bins; got {Mp.shape}")
        Mp = Mp[self.lo:self.lo + self.width, self.lo:self.lo + self.width]
        C = self.C
        return 0.5 * be.xp.real((C[idx] @ Mp) @ be.xp.conj(C).T)

    def after(self, inner: Any) -> "FourierToWDM":
        """The dense cell matrix of ``self . inner`` for a widely linear ``inner``
        (:class:`FourierZeroPad`, :class:`FourierCrop`) -- GB ``_cells`` (matrix.py:320-332):
        ``C = (analyse(A[idx]) + conj(analyse(B[idx]))) * inner.scale``, rows of ``dst``."""
        if self.impl != "dense" or inner.dst != self.src or self.lo != 0 or self.width != self.src.n:
            raise ValueError("after() needs a dense map whose src is exactly inner.dst = the window")
        g = self.dst.cells.grid
        geo = _geometry(self.dst)
        phi = phi_centered(g.Nf, g.Nt, g.dt)
        phase = _band_phase(geo["bands"], g.Nt, phi)
        nb, Nt = geo["bands"].size, g.Nt
        A, B = np.asarray(inner.A), np.asarray(inner.B)
        C = _analyse(phase, phi, A[geo["idx"]]).reshape(nb * Nt, -1)
        C = C + np.conj(_analyse(phase, phi, B[geo["idx"]]).reshape(nb * Nt, -1))
        C = C * inner.scale
        return FourierToWDM(inner.src, self.dst, "dense", lo=0, width=inner.src.n, C=C[geo["sel"]])
