"""Prior sanitising for a registry likelihood (spec section 5; ``toy_models/sinusoid/gpu/run.py``
``_wrap_phase`` / ``_prior_valid``, lines 135-178): rows outside the prior box (after wrapping
the periodic coordinates) or non-finite are replaced by the target's truth before the call
and masked back to ``-inf`` after it (``domainia.sampling.PriorSanitized``).

This is how the runner evaluated chain rows (G2).  The G1 reference points were evaluated by
the bare likelihood (lineages A and B, ``scripts/golden/workers/sinusoid_w{t,g}.py``): the
phase wrap ``lo + mod(phi_0 - lo, 2 pi)`` rounds ``phi_0`` even inside ``[-pi, pi)``, so G1
never goes through here.

The galactic binary (S7) is the runner's ``gpu/run_flow_gb.py`` ``lnprior``; its G3/G4 rows
went to the bare likelihood (``source_ok`` as its ``row_ok``), so they never go through here
either, and ``mu`` outside ``[-8, 8]`` is refused by this prior only.
"""

from __future__ import annotations

from typing import Any

from domainia.sampling import PriorSanitized

from ..sinusoid import config, priors

__all__ = ["sanitize"]


def sanitize(lik: Any, toy: str, case: str, model: str) -> PriorSanitized:
    """``lik`` (``registry.build(toy, case, model, ...)``) behind the prior of ``model``.

    Sinusoid: the box ``priors.bounds(model)``, the periodic ``priors.periodic(model)`` and the
    anchor ``config(case).truth(model)``.  Galactic binary (S7; ``run_flow_gb.lnprior``): the box
    ``galactic_binary.priors.bounds(prep, model)`` -- the stored GLASS source box and
    ``mu ~ U[-8, 8]`` per noise entry --, lon/psi/phi0 wrapped (``galactic_binary.priors.periodic``)
    and the anchor ``prep.anchor(model)`` (injection + fitted noise) of the shipped prepared case;
    the bare likelihood keeps ``source_ok`` as its ``row_ok`` (how the stored rows were computed)."""
    if toy == "sinusoid":
        return PriorSanitized(lik, priors.bounds(model), config.config(case).truth(model),
                              periodic=priors.periodic(model))
    if toy == "galactic_binary":
        from ..galactic_binary import priors as gb_priors
        from ..galactic_binary.prepared import PreparedCase

        prep = PreparedCase.load(case)
        return PriorSanitized(lik, gb_priors.bounds(prep, model), prep.anchor(model),
                              periodic=gb_priors.periodic(model))
    raise KeyError(f"no prior table for toy {toy!r}; the toys are ('sinusoid', 'galactic_binary')")
