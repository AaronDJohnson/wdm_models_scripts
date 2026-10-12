"""``ChannelStack``: independent channels (GB A and E), one covariance each (spec §4.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wdm.backend import Backend, get_backend, pytree_dataclass

from .base import Covariance

__all__ = ["ChannelStack", "StackFactor"]


@pytree_dataclass(data=("channels", "logdet", "ok"), meta=())
@dataclass(frozen=True, eq=False)
class StackFactor:
    """Per-channel factors; ``logdet`` summed in channel order from ``0.0``.

    A likelihood that must keep the oracle's per-channel association
    (``total - 0.5 (q_c + logdet_c + P log 2 pi)``) reads ``channels``.
    """

    channels: tuple[Any, ...]
    logdet: Any
    ok: Any

    @property
    def n_u(self) -> int:
        return sum(f.n_u for f in self.channels)

    def quad(self, b: Any, *, form: str = "solve", be: Backend | None = None) -> Any:
        """``sum_c quad_c(b[c])`` from ``0.0`` in channel order; ``b`` is ``(C, n)``."""
        total = 0.0
        for c, f in enumerate(self.channels):
            total = total + f.quad(b[c], form=form, be=be)
        return total

    def solve(self, b: Any, *, be: Backend | None = None) -> Any:
        be = be or get_backend(b)
        return be.xp.stack([f.solve(b[c], be=be) for c, f in enumerate(self.channels)])

    def whiten(self, b: Any, *, be: Backend | None = None) -> Any:
        be = be or get_backend(b)
        return be.xp.stack([f.whiten(b[c], be=be) for c, f in enumerate(self.channels)])


class ChannelStack(Covariance):
    """Independent channels: ``covs[c]`` sees ``nu[c * k:(c + 1) * k]``,
    ``k = params_per_channel`` (GB: ``n_env`` envelope coefficients per channel)."""

    def __init__(self, covs: tuple[Covariance, ...] | list[Covariance], params_per_channel: int) -> None:
        self.covs = tuple(covs)
        if not self.covs:
            raise ValueError("ChannelStack needs at least one channel")
        self.k = int(params_per_channel)
        self.n = sum(c.n for c in self.covs)
        self.n_u = sum(c.n_u for c in self.covs)
        self.jittable = all(c.jittable for c in self.covs)

    def split(self, nu: Any) -> list[Any]:
        if nu.shape[-1] != self.k * len(self.covs):
            raise ValueError(f"nu has {nu.shape[-1]} entries; expected {self.k * len(self.covs)}")
        return [nu[..., c * self.k:(c + 1) * self.k] for c in range(len(self.covs))]

    def children(self) -> dict[str, Covariance]:
        return {str(c): cov for c, cov in enumerate(self.covs)}

    def state_key(self, nu: Any) -> Any:
        """The channels' keys, each at its own slice of ``nu``."""
        keys = tuple(cov.state_key(nu_c) for cov, nu_c in zip(self.covs, self.split(nu), strict=True))
        return None if all(k is None for k in keys) else keys

    def _factor(self, C: Any, nu: Any, be: Backend) -> StackFactor:
        fs = tuple(cov._factor(C[str(c)], nu_c, be)
                   for c, (cov, nu_c) in enumerate(zip(self.covs, self.split(nu), strict=True)))
        logdet = 0.0
        ok = True
        for f in fs:
            logdet = logdet + f.logdet
            ok = ok & f.ok
        return StackFactor(fs, logdet, ok)
