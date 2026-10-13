"""``DiagonalCovariance``: a per-cell variance as a covariance (spec §4.4)."""

from __future__ import annotations

from typing import Any

from wdm.backend import Backend

from ..domains import CellSet
from .base import Covariance, DiagFactor, as_evaluator

__all__ = ["DiagonalCovariance"]


class DiagonalCovariance(Covariance):
    """``C = diag(v(nu))`` -- WT ``Solver(diag=...)``, GB ``band``/``mid`` models.

    ``variance`` is an :class:`~domainia.covariance.base.Evaluator` (``host_constants()``,
    ``evaluate(C, nu, be) -> (n,)``), a ``domainia.variance`` pytree
    (``BandAveragedVariance``, ``MidpointVariance``: evaluated as ``v(nu, channel=channel,
    be=be)``, the object itself passed to ``jit`` as the constants; ``channel`` selects the
    GB channel), a callable ``fn(nu, be) -> (n,)`` or a fixed ``(n,)`` array.  ``n`` is the
    number of cells.  ``cells`` is the variance's ``CellSet`` when it has one (the
    ``domainia.variance`` classes), else ``None``; a likelihood checks it against its map.
    """

    def __init__(self, variance: Any, n: int, *, channel: int | None = None) -> None:
        cells = getattr(variance, "cells", None)
        self.cells = cells if isinstance(cells, CellSet) else None
        self.variance = as_evaluator(variance, **({} if channel is None else {"channel": int(channel)}))
        self.n = int(n)

    def host_constants(self) -> Any:
        return self.variance.host_constants()

    def variances(self, C: Any, nu: Any, be: Backend) -> Any:
        v = self.variance.evaluate(C, nu, be)
        if v.shape != (self.n,):
            raise ValueError(f"the variance has shape {v.shape}; expected ({self.n},)")
        return v

    def _factor(self, C: Any, nu: Any, be: Backend) -> DiagFactor:
        return DiagFactor.build(self.variances(C, nu, be), be)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return be.xp.diag(self.variances(C, nu, be))
