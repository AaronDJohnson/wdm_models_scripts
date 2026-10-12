"""Imports of the frozen galactic-binary code (lineage C), each checked against its origin."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import gbw_common

GB_MODULES = ("config", "envelope", "likelihoods", "matrix", "fullcache", "prepare")
CHAINS = "wdm_paper_2/final_chains/gb_likelihood_matrix"
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")


def load(workspace, *, jax: bool = False) -> SimpleNamespace:
    ws = Path(workspace).resolve()
    gb = ws / "toy_models" / "galactic_binary"
    mods = {}
    for name in GB_MODULES:
        mods[name] = importlib.import_module(name)
        gbw_common.check_origin(mods[name], gb)
    import gbonfly

    gbw_common.check_origin(gbonfly, ws / "wdm_paper_2" / "gbonfly" / "src" / "gbonfly")
    mods["gbonfly"] = gbonfly
    if jax:
        mods["jax_models"] = importlib.import_module("jax_models")
        gbw_common.check_origin(mods["jax_models"], gb / "gpu")
    return SimpleNamespace(**mods, workspace=ws, gb_dir=gb)


def targets(manifest: dict, case: str) -> dict[str, list[str]]:
    """model -> run directories of one case, from final_chains/gb_likelihood_matrix/manifest.json."""
    return {model: (list(e["runs"]) if "runs" in e else [e["run"]])
            for model, e in sorted(manifest[case].items())}
