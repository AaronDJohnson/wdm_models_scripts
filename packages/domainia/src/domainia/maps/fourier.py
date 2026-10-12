"""Maps out of band-limited Fourier bins: zero padding, cropping, WDM cells (GB, spec §4.2).

Ported from ``toy_models/galactic_binary/matrix.py`` (frozen): ``_geometric``,
``pad_maps``, ``crop_maps`` verbatim (L270-306), the cell matrices of ``Grid.dense_atoms``
(L222-227) and ``_cells`` (L320-332, with ``wdm_full.analyse`` L142-145 and
``wdm_bands._phase`` L89-95), and the data map ``cells`` (L818-820).  The atoms are
``wdm.transform.band_atoms`` (bit-identical to ``geometry.band_atoms``, S1 Task C6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from wdm.backend import pytree_dataclass

from ..domains import FourierDomain
from .base import _BUILD, LinearMap

__all__ = ["FourierCrop", "FourierZeroPad", "crop_maps", "pad_maps"]


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


@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "scale"))
@dataclass(frozen=True, eq=False)
class FourierZeroPad(LinearMap):
    """Bins ``dst`` of the record zero-padded from ``src.N`` to ``dst.N`` samples, from the
    bins ``src``: ``z' = scale (A z + B conj z)``, ``scale = sqrt(N / N')`` (kind
    ``"widely"``).  ``A, B`` are :func:`pad_maps`; ``scale`` keeps the padded record's
    unit-norm cells in the record's cell units (GB ``variant``, matrix.py:373-377)."""

    src: FourierDomain
    dst: FourierDomain
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)
    scale: float = field(default=_BUILD, kw_only=True)

    kind = "widely"

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
class FourierCrop(LinearMap):
    """Bins ``dst`` of the record cropped to samples ``[start, start + dst.N)``, from the
    bins ``src``: ``z^c = A z + B conj z`` (:func:`crop_maps`; GB ``whittle_map`` and
    ``mask_rows``).  ``apply`` is the template call site ``(h @ A.T) + (conj(h) @ B.T)``
    (matrix.py:897); :meth:`apply_data` the data call site (matrix.py:892)."""

    src: FourierDomain
    dst: FourierDomain
    start: int = 0
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)

    kind = "widely"
    scale = 1.0

    def __post_init__(self) -> None:
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
