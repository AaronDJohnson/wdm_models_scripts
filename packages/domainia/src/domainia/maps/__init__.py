"""Linear maps between domains (spec §4.2)."""

from .base import (BinSelect, Compose, Identity, LinearMap, PushedCovariance, Restrict,
                   SupportsDense, SupportsMatvec, domain_size, on_backend)
from .time import TimeToWDM, WDMToTime

__all__ = ["BinSelect", "Compose", "Identity", "LinearMap", "PushedCovariance", "Restrict",
           "SupportsDense", "SupportsMatvec", "TimeToWDM", "WDMToTime", "domain_size",
           "on_backend"]
