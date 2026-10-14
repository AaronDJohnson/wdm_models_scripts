"""F-D4 (S7): prior sanitising for the GB (spec sections 4.7 and 5.2; legacy ``run_flow_gb.lnprior``
and ``likelihoods.source_ok``).

``sanitize(lik, "galactic_binary", case, model)`` puts a registry build behind the prior box
``priors.bounds(prep, model)`` -- the stored GLASS source box (``prior_lo``/``prior_hi``) and
``mu ~ U[-8, 8]`` per noise entry -- with lon, psi and phi0 wrapped first (``priors.periodic``):
rows outside it (or non-finite) are replaced by the case's anchor before the call and come back
``-inf``; a ``-inf`` the likelihood flags as a PD failure passes; anything else non-finite raises.
The bare likelihood keeps ``source_ok`` (``row_ok``): it is how the stored rows were computed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from domainia.sampling import PriorSanitized  # noqa: E402
from scripts.golden.paths import GOLDEN  # noqa: E402
from wdm_models.galactic_binary import priors  # noqa: E402
from wdm_models.galactic_binary.prepared import PreparedCase  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import TARGETS, build  # noqa: E402
from wdm_models.sampling.prior_sanitize import sanitize  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
CASE = "gb_3mo"
GOLDEN_OPTIONS = ModelOptions(waveform_backend="fourier")


def _stress(model: str) -> dict:
    return {p["name"]: p for p in json.loads(
        (GOLDEN / "galactic_binary" / "stress" / CASE / f"{model}.fast.json").read_text())["points"]}


class _Rows:
    """A stand-in likelihood (the sanitiser reads ``ndim`` only until it is called)."""

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim


def test_the_box_the_periodic_angles_and_the_anchor_are_the_priors():
    prep = PreparedCase.load(CASE)
    for case, models in TARGETS["galactic_binary"].items():
        p = PreparedCase.load(case)
        for model in models:
            s = sanitize(_Rows(p.ndim(model)), "galactic_binary", case, model)
            assert isinstance(s, PriorSanitized) and s.ndim == p.ndim(model)
            assert np.array_equal(s.bounds, priors.bounds(p, model)) and np.array_equal(s.anchor, p.anchor(model))
            assert s.periodic == {2: (0.0, 2 * np.pi), 5: (0.0, np.pi), 6: (0.0, 2 * np.pi)}
    assert np.array_equal(priors.bounds(prep, "tri_raw")[8:], np.tile([[-8.0, 8.0]], (10, 1)))


def test_rows_outside_the_box_are_minus_inf_and_never_reach_the_device():
    prep = PreparedCase.load(CASE)
    lik = build("galactic_binary", CASE, "tri_raw", backend="jax", options=GOLDEN_OPTIONS)
    s = sanitize(lik, "galactic_binary", CASE, "tri_raw")
    x = prep.anchor("tri_raw")
    lnA_out, mu_out, nan_row = x.copy(), x.copy(), x.copy()
    lnA_out[3] = prep.prior_hi[3] + 1.0
    mu_out[9] = 8.5                                     # source_ok passes it: only the prior refuses
    nan_row[0] = np.nan
    got = s(np.stack([x, lnA_out, mu_out, nan_row]))
    assert np.isfinite(got[0]) and got[1:].tolist() == [-np.inf] * 3
    assert lik.counters()["noise_states"] == 1           # the replaced rows were the anchor: one state
    assert np.isfinite(build("galactic_binary", CASE, "tri_raw", backend="jax", options=GOLDEN_OPTIONS)(mu_out[None])[0])


def test_angles_are_wrapped_before_the_call():
    prep = PreparedCase.load(CASE)
    lik = build("galactic_binary", CASE, "band_raw", backend="jax", options=GOLDEN_OPTIONS)
    s = sanitize(lik, "galactic_binary", CASE, "band_raw")
    x = prep.anchor("band_raw")
    x[2], x[5], x[6] = 0.3, 0.7, 0.5
    y = x.copy()
    y[2], y[5], y[6] = 0.3 + 2 * np.pi, 0.7 - np.pi, 0.5 + 4 * np.pi
    Y = np.stack([x, y])
    got = s(Y)
    assert np.all(np.isfinite(got)) and np.all(s.valid(s.wrap(Y)))
    assert np.array_equal(got, lik(s.wrap(Y)))                 # the likelihood saw the wrapped rows
    assert abs(got[1] - got[0]) <= 1e-12 * abs(got[0])          # lon, phi0 2 pi- and psi pi-periodic


@pytest.mark.parametrize("model", ["tri_raw", "band_exact", "exact"])
def test_a_pd_failure_inside_the_box_passes_as_minus_inf(model):
    pts = _stress(model)
    failing = [p for n, p in pts.items() if not p["marginal"] and p["expected_neg_inf"] and not p["source_out"]]
    lik = build("galactic_binary", CASE, model, backend="jax", options=GOLDEN_OPTIONS)
    s = sanitize(lik, "galactic_binary", CASE, model)
    X = np.array([pts["anchor"]["theta"]] + [p["theta"] for p in failing])
    got = s(X)
    assert np.isfinite(got[0]) and np.all(got[1:] == -np.inf)
    assert lik.last_failed.tolist() == [False] + [True] * len(failing)
    if model != "exact":
        assert failing                                    # tri_raw: 3, band_exact: 5 PD failures at gb_3mo


def test_unknown_toys_are_refused():
    with pytest.raises(KeyError, match="toy"):
        sanitize(object(), "chirp", CASE, "band_raw")
