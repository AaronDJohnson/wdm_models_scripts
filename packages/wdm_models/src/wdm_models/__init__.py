"""wdm_models: the WDM Paper II toy models (sinusoid; galactic binary from S5) and the
15-model registry, built from ``domainia`` pieces (spec section 5).

Importing ``wdm_models`` checks that ``wdm`` and ``domainia`` are the paper release (spec
section 1, guard 4) and never imports JAX: the JAX backend is reached only through
:mod:`wdm.backend`, which checks float64 first.

Modules
-------
* :mod:`wdm_models.sinusoid`: ``config``, ``signal``, ``psd``, ``priors``, ``data``,
  ``exact``, ``models`` (spec section 5.1)
* :mod:`wdm_models.options`: ``ModelOptions`` and the per-backend strategy table
* :mod:`wdm_models.registry`: ``MODELS``, ``TARGETS``, ``build``, ``describe`` (spec 5.3)
* :mod:`wdm_models.common`: helpers shared by the toys (shipped-data paths, parameter
  layouts)
"""

from __future__ import annotations

import wdm as _wdm

if getattr(_wdm, "__release__", None) != "wdm-paper-1" or not hasattr(_wdm, "forward_time_many"):
    raise ImportError(f"wdm resolved to {getattr(_wdm, '__file__', None)}: not the paper release")

import domainia as _domainia  # noqa: E402  (its own sentinel re-checks wdm)

if getattr(_domainia, "__version__", None) != "1.0.0" or not hasattr(_domainia, "CellSet"):
    raise ImportError(f"domainia resolved to {getattr(_domainia, '__file__', None)}: "
                      "not the paper release")

__version__ = "1.0.0"

__all__ = ["__version__"]
