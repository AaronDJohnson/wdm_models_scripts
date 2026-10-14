"""Prior sanitising for a registry likelihood (spec section 5; ``toy_models/sinusoid/gpu/run.py``
``_wrap_phase`` / ``_prior_valid``, lines 135-178): rows outside the prior box (after wrapping
the periodic coordinates) or non-finite are replaced by the target's truth before the call
and masked back to ``-inf`` after it (``domainia.sampling.PriorSanitized``).

This is how the runner evaluated chain rows (G2).  The G1 reference points were evaluated by
the bare likelihood (lineages A and B, ``scripts/golden/workers/sinusoid_w{t,g}.py``): the
phase wrap ``lo + mod(phi_0 - lo, 2 pi)`` rounds ``phi_0`` even inside ``[-pi, pi)``, so G1
never goes through here.
"""

from __future__ import annotations

from typing import Any

from domainia.sampling import PriorSanitized

from ..sinusoid import config, priors

__all__ = ["sanitize"]


def sanitize(lik: Any, toy: str, case: str, model: str) -> PriorSanitized:
    """``lik`` (``registry.build(toy, case, model, ...)``) behind the prior of ``model``:
    the box ``priors.bounds(model)``, the periodic ``priors.periodic(model)`` and the anchor
    ``config(case).truth(model)``."""
    if toy != "sinusoid":
        raise KeyError(f"no prior table for toy {toy!r} yet (galactic_binary arrives in S5/S6)")
    return PriorSanitized(lik, priors.bounds(model), config.config(case).truth(model),
                          periodic=priors.periodic(model))
