"""The 15 WDM II models (spec §0 rule 7, §5.3) and their names.

``MODELS`` is the one list both toys and the registry use (spec §5.3 order).  A model name is
``<base>_<edge>`` -- ``base`` in ``whittle``/``band``/``mid``/``tri``, ``edge`` in
``raw``/``cut``/``pad``/``exact`` -- or ``exact`` alone (the exact covariance).
"""

from __future__ import annotations

__all__ = ["MODELS", "check_model", "is_whittle", "split_model"]

MODELS: tuple[str, ...] = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad",
                           "band_exact", "mid_raw", "mid_cut", "mid_pad", "mid_exact", "tri_raw",
                           "tri_cut", "tri_pad", "tri_exact", "exact")


def check_model(model: str) -> str:
    """``model`` if it is one of :data:`MODELS`, else ``ValueError`` naming them."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; the paper models are {MODELS}")
    return model


def is_whittle(model: str) -> bool:
    """The stationary Fourier models (one spectral index, five parameters)."""
    return check_model(model).startswith("whittle_")


def split_model(model: str) -> tuple[str, str | None]:
    """``("band", "cut")`` for ``band_cut``; ``("exact", None)`` for ``exact``."""
    check_model(model)
    if model == "exact":
        return "exact", None
    base, edge = model.split("_")
    return base, edge
