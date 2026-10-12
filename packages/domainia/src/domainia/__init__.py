"""domainia: data on domains, the maps between them, covariance operators and Gaussian
likelihoods over the WDM transform (WDM Paper II release, spec section 4).

Importing ``domainia`` checks that ``wdm`` is the paper release (spec section 1, guard 4)
and never imports JAX: the JAX backend is reached only through :mod:`wdm.backend`
(``jax_backend()`` / ``get_backend`` on a ``jax.Array``), which checks float64 first.
"""

from __future__ import annotations

import wdm as _wdm

if getattr(_wdm, "__release__", None) != "wdm-paper-1" or not hasattr(_wdm, "forward_time_many"):
    raise ImportError(f"wdm resolved to {getattr(_wdm, '__file__', None)}: not the paper release")

__version__ = "1.0.0"

__all__ = ["__version__"]
