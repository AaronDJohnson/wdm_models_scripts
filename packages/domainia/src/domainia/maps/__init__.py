"""Linear maps between domains (spec §4.2)."""

from .base import (BinSelect, Compose, Identity, LinearMap, PushedCovariance, Restrict,
                   SupportsDense, SupportsMatvec, domain_size, on_backend)
from .fourier import FourierCrop, FourierToWDM, FourierZeroPad, crop_maps, pad_maps
from .time import CropTime, TimeToFourier, TimeToWDM, WDMToTime, ZeroPadTime

__all__ = ["BinSelect", "Compose", "CropTime", "FourierCrop", "FourierToWDM", "FourierZeroPad",
           "Identity", "LinearMap", "PushedCovariance", "Restrict", "SupportsDense",
           "SupportsMatvec", "TimeToFourier", "TimeToWDM", "WDMToTime", "ZeroPadTime",
           "crop_maps", "domain_size", "on_backend", "pad_maps"]
