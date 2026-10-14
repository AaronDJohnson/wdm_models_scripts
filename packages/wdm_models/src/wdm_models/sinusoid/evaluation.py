"""The sinusoid's evaluation policy per backend (spec sections 3.2 and 3.5): what
``registry.build`` puts around the ``domainia`` likelihood.

The algorithm choices themselves are :func:`wdm_models.options.resolve`'s columns; this
module only decides the host-side evaluator, from the same :class:`~wdm_models.options.Resolved`:

* NumPy (WT, the golden references): the bare likelihood -- every row factorises its own
  noise state, as ``wdm_toys`` did (and as ``scripts/sinusoid/validate.py`` counts it);
* JAX (WG ``MatrixGPU``, ``gpu_sinusoid/wdmtoysgpu/matrix.py:269-279, 726-789``): an LRU of
  noise states (``domainia.sampling.NoiseStateCache``, ``Resolved.cache_size`` = 16 GPU /
  32 CPU, the rows of one call batched when ``Resolved.batch_rows``: on a GPU, never for
  edge-exact) for the eight models WG cached; one batched call
  (``domainia.sampling.BatchedLikelihood``, a ``jit(vmap)``) for the rest -- the four
  uncached diagonals counting no noise state (WG ``:776-786``, "nothing to factor"), the
  Whittle forms and the ``exact`` model's batched PCG counting one per row.  A likelihood
  that cannot be traced (``strategy="reference"`` on JAX: the exact model's WT PCG raising
  on failure) goes through the eager LRU instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..options import Resolved

__all__ = ["CACHED", "EDGE_EXACT_MODELS", "UNCACHED_DIAGONAL", "Wrapper", "wrap", "wrapper_for"]

#: Models whose noise state WG cached per noise value (``matrix.py:269``: tri, pad, exact edges).
CACHED = ("tri_raw", "tri_cut", "band_pad", "mid_pad", "tri_pad", "band_exact", "mid_exact", "tri_exact")
EDGE_EXACT_MODELS = ("band_exact", "mid_exact", "tri_exact")
#: Diagonal models WG evaluated without a cache: no noise state is counted (``matrix.py:780``).
UNCACHED_DIAGONAL = ("band_raw", "mid_raw", "band_cut", "mid_cut")


@dataclass(frozen=True)
class Wrapper:
    """What :func:`wrap` puts around a model's likelihood (``"none"``: nothing)."""

    kind: Literal["none", "batched", "cache"]
    count_states: bool = True
    capacity: int | None = None
    batch_rows: bool = False


def wrapper_for(model: str, r: Resolved) -> Wrapper:
    """The evaluator of ``model`` under the resolved strategies ``r`` (module docstring)."""
    if r.backend == "numpy":
        return Wrapper("none")
    if model in CACHED:
        return Wrapper("cache", capacity=r.cache_size, batch_rows=r.batch_rows)
    return Wrapper("batched", count_states=model not in UNCACHED_DIAGONAL)


def wrap(lik: Any, model: str, r: Resolved) -> Any:
    """``lik`` (a ``domainia`` ``GaussianLikelihood`` / ``FourierWhittle``) in its evaluator."""
    w = wrapper_for(model, r)
    if w.kind == "none":
        return lik
    from domainia.sampling import BatchedLikelihood, NoiseStateCache

    if w.kind == "cache":
        return NoiseStateCache(lik, w.capacity, batch_rows=w.batch_rows)
    if not getattr(lik, "traceable", True):
        # strategy="reference" on JAX: the exact model's WT PCG (on_fail="raise") is not
        # jit-able, so its rows run eagerly, one noise state at a time
        return NoiseStateCache(lik, r.cache_size, batch_rows=False)
    return BatchedLikelihood(lik, count_states=w.count_states)
