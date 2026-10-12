"""domainia: data on domains, the maps between them, covariance operators and Gaussian
likelihoods over the WDM transform (WDM Paper II release, spec section 4).

Importing ``domainia`` checks that ``wdm`` is the paper release (spec section 1, guard 4)
and never imports JAX: the JAX backend is reached only through :mod:`wdm.backend`
(``jax_backend()`` / ``get_backend`` on a ``jax.Array``), which checks float64 first.

Modules
-------
* :mod:`domainia.domains`: ``TimeDomain``, ``FourierDomain``, ``CellSet``, ``WDMDomain``,
  ``ChannelAxis`` (frozen, hashable: static under ``jax.jit``)
* :mod:`domainia.data`: ``TimeSeries``, ``FourierBins``, ``WDMCells`` (pytrees; the
  domain and channels are static metadata)
* :mod:`domainia.linalg`: block-tridiagonal (ragged / padded), Hermitian block, dense,
  Schur and padded-marginal factors, preconditioned CG
* :mod:`domainia.assets`: ``Artifact`` (identity + sha256, fail-closed, never writes
  unless asked)
* :mod:`domainia.testing`: dense oracles for tests (not imported here)
"""

from __future__ import annotations

import wdm as _wdm

if getattr(_wdm, "__release__", None) != "wdm-paper-1" or not hasattr(_wdm, "forward_time_many"):
    raise ImportError(f"wdm resolved to {getattr(_wdm, '__file__', None)}: not the paper release")

from . import assets, data, domains, linalg  # noqa: E402
from .assets import Artifact, ArtifactError  # noqa: E402
from .data import FourierBins, TimeSeries, WDMCells  # noqa: E402
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain  # noqa: E402

__version__ = "1.0.0"

__all__ = [
    "Artifact",
    "ArtifactError",
    "CellSet",
    "ChannelAxis",
    "FourierBins",
    "FourierDomain",
    "TimeDomain",
    "TimeSeries",
    "WDMCells",
    "WDMDomain",
    "__version__",
    "assets",
    "data",
    "domains",
    "linalg",
]
