"""Linear maps between domains (spec §4.2)."""

from .base import (BinSelect, Compose, Identity, LinearMap, PushedCovariance, Restrict,
                   SupportsDense, SupportsMatvec, domain_size, on_backend)
from .time import CropTime, TimeToFourier, TimeToWDM, WDMToTime, ZeroPadTime

__all__ = ["BinSelect", "Compose", "CropTime", "Identity", "LinearMap", "PushedCovariance",
           "Restrict", "SupportsDense", "SupportsMatvec", "TimeToFourier", "TimeToWDM",
           "WDMToTime", "ZeroPadTime", "domain_size", "on_backend"]
