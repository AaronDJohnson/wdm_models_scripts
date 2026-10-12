"""Data vectors on a domain (spec section 4.1).

Each class is a JAX pytree (:func:`wdm.backend.pytree_dataclass`) whose only leaf is
``values``; ``domain`` and ``channels`` are static metadata, so a batch of data
vectors on one domain ``vmap``s and two different domains compile separately.

Shapes: ``values[..., N]`` (no channel axis) or ``values[..., C, N]`` with
``C = len(channels.names)``; ``N`` is ``domain.size``.  The shape is checked when
``values`` is a concrete array; JAX rebuilds pytrees with tracers or placeholder
leaves (``tree_map``, ``vmap`` axis specs), and those are never checked.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from wdm.backend import is_tracer, pytree_dataclass

from .domains import ChannelAxis, FourierDomain, TimeDomain, WDMDomain

__all__ = ["FourierBins", "TimeSeries", "WDMCells"]


def _check(values: Any, size: int, channels: ChannelAxis, what: str) -> None:
    if is_tracer(values) or not hasattr(values, "shape"):
        return
    shape = tuple(values.shape)
    need = 2 if channels.size else 1
    if len(shape) < need:
        raise ValueError(f"{what}: values need at least {need} axes; got shape {shape}")
    if shape[-1] != size:
        raise ValueError(f"{what}: last axis must be the domain size {size}; got shape {shape}")
    if channels.size and shape[-2] != channels.size:
        raise ValueError(f"{what}: axis -2 must be the {channels.size} channels "
                         f"{channels.names}; got shape {shape}")


class _Data:
    """Shared helpers (not a dataclass)."""

    def replace(self, **changes: Any) -> Any:
        """A copy with fields replaced (``dataclasses.replace``)."""
        return dataclasses.replace(self, **changes)  # type: ignore[type-var]


@pytree_dataclass(data=("values",), meta=("domain", "channels"))
@dataclass(frozen=True, eq=False)
class TimeSeries(_Data):
    """Real samples on a :class:`~domainia.domains.TimeDomain`."""

    values: Any
    domain: TimeDomain
    channels: ChannelAxis = ChannelAxis()

    def __post_init__(self) -> None:
        if not isinstance(self.domain, TimeDomain):
            raise TypeError(f"TimeSeries.domain must be a TimeDomain; got {type(self.domain).__name__}")
        _check(self.values, self.domain.N, self.channels, "TimeSeries")


@pytree_dataclass(data=("values",), meta=("domain", "channels"))
@dataclass(frozen=True, eq=False)
class FourierBins(_Data):
    """Complex rfft bins on a :class:`~domainia.domains.FourierDomain`."""

    values: Any
    domain: FourierDomain
    channels: ChannelAxis = ChannelAxis()

    def __post_init__(self) -> None:
        if not isinstance(self.domain, FourierDomain):
            raise TypeError(f"FourierBins.domain must be a FourierDomain; got "
                            f"{type(self.domain).__name__}")
        _check(self.values, self.domain.n, self.channels, "FourierBins")


@pytree_dataclass(data=("values",), meta=("domain", "channels"))
@dataclass(frozen=True, eq=False)
class WDMCells(_Data):
    """Real WDM cells on a :class:`~domainia.domains.WDMDomain` (band order)."""

    values: Any
    domain: WDMDomain
    channels: ChannelAxis = ChannelAxis()

    def __post_init__(self) -> None:
        if not isinstance(self.domain, WDMDomain):
            raise TypeError(f"WDMCells.domain must be a WDMDomain; got {type(self.domain).__name__}")
        _check(self.values, self.domain.size, self.channels, "WDMCells")
