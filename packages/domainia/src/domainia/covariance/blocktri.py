"""``BlockTridiagonalCovariance``: band blocks factorised by the block Cholesky (spec §4.4).

Two layouts of the same factorisation (spec §3.2):

* ``"ragged"`` -- Python lists of per-band blocks, WT ``BlockTri`` order; eager on both
  backends (the NumPy golden default);
* ``"padded"`` -- ``(nb, B, B)`` arrays with identity dummy slots (``BandLayout``), WG
  ``_btri_*`` over ``be.scan``; jit-able (the JAX golden default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from wdm.backend import Backend, get_backend, pytree_dataclass

from ..linalg import BandLayout, blocktri_factor, blocktri_solve, blocktri_whiten
from .base import Covariance, QuadForm, as_evaluator, quad_form

__all__ = ["BlockTriCovFactor", "BlockTridiagonalCovariance", "blocks_to_dense", "blocktri_cov_factor",
           "dummy_slots", "to_layout"]

Layout = Literal["ragged", "padded"]


def dummy_slots(sizes: tuple[int, ...]) -> np.ndarray:
    """``(nb, B)``: 1.0 on the dummy slots of the padded layout, 0.0 on the cells."""
    B = max(sizes)
    return np.where(np.arange(B)[None, :] < np.asarray(sizes)[:, None], 0.0, 1.0)


def to_layout(diag: Any, up: Any, sizes: tuple[int, ...], layout: Layout, be: Backend,
              dummy: Any = None) -> tuple[Any, Any]:
    """``(diag, up)`` in ``layout``.

    Ragged input: lists (``diag[m]`` ``(s_m, s_m)``, ``up[m]`` ``(s_m, s_{m+1})``); padded
    input: ``(nb, B, B)`` / ``(nb - 1, B, B)`` arrays whose slot ``i < s_m`` of band ``m``
    is the band's ``i``-th cell.  Padded output symmetrises the diagonal blocks and puts
    the identity on dummy slots (WG ``MatrixGPU._tri_factor``); ``dummy`` is
    :func:`dummy_slots` (from the constants under ``jit``).
    """
    xp = be.xp
    ragged_in = isinstance(diag, (list, tuple))
    if layout == "ragged":
        if ragged_in:
            return list(diag), list(up)
        return ([diag[m][:s, :s] for m, s in enumerate(sizes)],
                [up[m][:sizes[m], :sizes[m + 1]] for m in range(len(sizes) - 1)])
    B = max(sizes)
    if ragged_in:
        diag = xp.stack([xp.pad(d, ((0, B - s), (0, B - s))) for d, s in zip(diag, sizes, strict=True)])
        up = xp.stack([xp.pad(u, ((0, B - sizes[m]), (0, B - sizes[m + 1])))
                       for m, u in enumerate(up)])
    dummy = be.asarray(dummy_slots(sizes)) if dummy is None else dummy
    diag = 0.5 * (diag + xp.swapaxes(diag, 1, 2))
    diag = diag + dummy[:, :, None] * xp.eye(B)[None]
    return diag, up


def blocks_to_dense(diag: list[Any], up: list[Any], be: Backend) -> Any:
    """The dense matrix of ragged band blocks (tests and dense assembly; ``O(n^2)`` memory)."""
    xp = be.xp
    nb = len(diag)
    sizes = [d.shape[0] for d in diag]

    def blk(i: int, j: int) -> Any:
        if i == j:
            return diag[i]
        if j == i + 1:
            return up[i]
        if i == j + 1:
            return up[j].T
        return xp.zeros((sizes[i], sizes[j]))

    return xp.block([[blk(i, j) for j in range(nb)] for i in range(nb)])


@pytree_dataclass(data=("bt", "lay", "logdet", "ok"), meta=("layout",))
@dataclass(frozen=True, eq=False)
class BlockTriCovFactor:
    """A :class:`domainia.linalg.BlockTriFactor` with vectors in cell order (band-major)."""

    bt: Any
    lay: Any
    logdet: Any
    ok: Any
    layout: str

    n_u = 0

    def _apply(self, fn: Any, b: Any, be: Backend) -> Any:
        # domainia.linalg (D-A) takes natural, band-major vectors on both layouts: the padded
        # layout packs, runs WG's _btri_* and unpacks inside (the dummy entries are exactly 0).
        return fn(self.bt, b, be=be)

    def solve(self, b: Any, *, be: Backend | None = None) -> Any:
        return self._apply(blocktri_solve, b, be or get_backend(b))

    def whiten(self, b: Any, *, be: Backend | None = None) -> Any:
        return self._apply(blocktri_whiten, b, be or get_backend(b))

    def quad(self, b: Any, *, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, be or get_backend(b))


def blocktri_cov_factor(diag: Any, up: Any, sizes: tuple[int, ...], layout: Layout, lay: Any,
                        be: Backend, dummy: Any = None) -> BlockTriCovFactor:
    """Factor band blocks in ``layout`` (``lay``: the :class:`BandLayout`, ``dummy``:
    :func:`dummy_slots`, both for ``"padded"``)."""
    d, u = to_layout(diag, up, sizes, layout, be, dummy)
    bt = blocktri_factor(d, u, layout=layout, band_layout=lay, be=be)
    return BlockTriCovFactor(bt, lay, bt.logdet, bt.ok, layout)


class BlockTridiagonalCovariance(Covariance):
    """``C`` block tri-diagonal in the band index.

    ``blocks`` is an :class:`~domainia.covariance.base.Evaluator` (or callable
    ``fn(nu, be)``) returning ``(diag, up)`` ragged or padded (:func:`to_layout`);
    ``sizes`` the cells per band.  ``layout="ragged"`` runs eagerly on both backends.
    """

    def __init__(self, blocks: Any, sizes: tuple[int, ...], *, layout: Layout = "ragged") -> None:
        if layout not in ("ragged", "padded"):
            raise ValueError(f"layout must be 'ragged' or 'padded'; got {layout!r}")
        self.blocks = as_evaluator(blocks)
        self.sizes = tuple(int(s) for s in sizes)
        self.n = int(sum(self.sizes))
        self.layout = layout
        self.jittable = layout == "padded"

    def host_constants(self) -> Any:
        padded = self.layout == "padded"
        return {"blocks": self.blocks.host_constants(),
                "layout": BandLayout.from_sizes(self.sizes) if padded else None,
                "dummy": dummy_slots(self.sizes) if padded else None}

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        return self.blocks.evaluate(C["blocks"], nu, be)

    def _factor(self, C: Any, nu: Any, be: Backend) -> BlockTriCovFactor:
        diag, up = self.band_blocks(C, nu, be)
        return blocktri_cov_factor(diag, up, self.sizes, self.layout, C["layout"], be, C["dummy"])

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.band_blocks(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)
