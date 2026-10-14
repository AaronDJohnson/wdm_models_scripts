"""F-A7: the GB priors (spec §5.2 priors.py): binary_galactica's GLASS box on the source band
is the prepared ``prior_lo/hi`` bit for bit (and Table IV), the noise box, the periodic
wrap and ``source_ok`` (legacy ``likelihoods.wrap_source``/``source_ok``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wdm_models.galactic_binary import config as C
from wdm_models.galactic_binary import priors as Pr

CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
T = {"gb_3mo": 7884000.0, "gb_6mo": 15768000.0, "gb_12mo": 31536000.0}
#: WDM II Table IV: f_min, f_max [mHz] and fdot_min, fdot_max [1e-18 Hz/s] per duration.
TABLE_IV = {"gb_3mo": ((0.99696, 1.01205), (-0.493, 8.36)),
            "gb_6mo": ((0.99309, 1.00691), (-0.485, 8.20)),
            "gb_12mo": ((0.99172, 1.00596), (-0.482, 8.18))}


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_source_prior_is_the_stored_box_bit_for_bit(case, bg, gb_products, s5):
    from wdm_models.galactic_binary.prepared import PreparedCase

    prep = PreparedCase.load(case)
    pr = Pr.source_prior(prep)
    s5(f"G5/prior/{case}/box", max(np.max(np.abs(np.asarray(pr.lo) - prep.prior_lo)),
                                   np.max(np.abs(np.asarray(pr.hi) - prep.prior_hi))), 0.0)
    assert type(pr).__name__ == "UCBPrior"
    assert np.array_equal(np.asarray(pr.lo), prep.prior_lo) and np.array_equal(np.asarray(pr.hi), prep.prior_hi)
    lo, n = gb_products["cases"][case]["prior"]["source_band_bins"]
    assert prep.source_band == (lo, lo + n)
    assert (prep.prior_lo[3], prep.prior_hi[3]) == (-54.0, -46.0)
    (f_lo, f_hi), (fd_lo, fd_hi) = TABLE_IV[case]
    assert round(prep.prior_lo[0] / prep.T * 1e3, 5) == f_lo and round(prep.prior_hi[0] / prep.T * 1e3, 5) == f_hi
    assert round(prep.prior_lo[7] / prep.T ** 2 * 1e18, 3) == fd_lo
    assert round(prep.prior_hi[7] / prep.T ** 2 * 1e18, 2) == fd_hi


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_the_injection_is_params_from_native(case, bg, gb_products, s5):
    inj = gb_products["cases"][case]["inject"]
    keys = ("f0", "fdot", "amp", "phi0", "colat", "lon", "psi", "cos_iota")
    got = np.asarray(bg.params_from_native(*(inj["native"][k] for k in keys), T[case]))
    s5(f"G5/prior/{case}/inject_glass", np.max(np.abs(got - np.asarray(inj["inject_glass"]))), 0.0)
    assert np.array_equal(got, np.asarray(inj["inject_glass"]))


def test_noise_box_bounds_and_layout():
    assert np.array_equal(Pr.noise_box("tri_exact"), np.tile([[-8.0, 8.0]], (10, 1)))
    assert np.array_equal(Pr.noise_box("whittle_raw"), np.tile([[-8.0, 8.0]], (2, 1)))
    p = Pr.params("band_pad")
    assert p.names == C.parameter_names("band_pad") == Pr.names("band_pad") and p.source == tuple(range(8))
    assert p.noise == tuple(range(8, 18))
    assert Pr.params("whittle_cut").noise == (8, 9)
    assert Pr.periodic("exact") == {2: (0.0, 2 * math.pi), 5: (0.0, math.pi), 6: (0.0, 2 * math.pi)}


@pytest.mark.golden
def test_bounds_are_the_legacy_prepared_bounds():
    from wdm_models.galactic_binary.prepared import PreparedCase

    prep = PreparedCase.load("gb_3mo")
    b = Pr.bounds(prep, "tri_raw")
    assert b.shape == (18, 2) and np.array_equal(b, prep.bounds("tri_raw"))
    assert np.array_equal(b[:8, 0], prep.prior_lo) and np.all(b[8:] == [-8.0, 8.0])


def test_wrap_source_and_source_ok():
    lo = np.array([7860.0, -1, 0, -54, -1, 0, 0, -3e-5])
    hi = np.array([7979.0, 1, 2 * math.pi, -46, 1, math.pi, 2 * math.pi, 5e-4])
    p = np.array([7900.0, 0.1, 7.0, -50.0, 0.2, -0.5, -1.0, 1e-4])
    w = Pr.wrap_source(p)
    assert w is not p and p[2] == 7.0                                    # a copy
    assert np.array_equal(w, [7900.0, 0.1, 7.0 - 2 * math.pi, -50.0, 0.2, math.pi - 0.5,
                              2 * math.pi - 1.0, 1e-4])
    assert bool(Pr.source_ok(p, lo, hi))
    for i, v in ((0, 7859.0), (1, 1.5), (3, -45.0), (7, 1e-3), (0, np.nan)):
        q = p.copy()
        q[i] = v
        assert not bool(Pr.source_ok(q, lo, hi)), (i, v)


@pytest.mark.jax
def test_wrap_and_source_ok_trace_under_jit(jax_be):
    import jax
    import jax.numpy as jnp

    lo = np.array([7860.0, -1, 0, -54, -1, 0, 0, -3e-5])
    hi = np.array([7979.0, 1, 2 * math.pi, -46, 1, math.pi, 2 * math.pi, 5e-4])
    p = np.array([7900.0, 0.1, 7.0, -50.0, 0.2, -0.5, -1.0, 1e-4])
    w = jax.jit(lambda x: Pr.wrap_source(x, be=jax_be))(jnp.asarray(p))
    assert np.allclose(np.asarray(w), Pr.wrap_source(p), rtol=0, atol=1e-15)
    # an explicit be= wins over a host row (get_backend's contract; legacy _set raised here)
    assert np.allclose(np.asarray(Pr.wrap_source(p, be=jax_be)), Pr.wrap_source(p), rtol=0, atol=1e-15)
    ok = jax.jit(lambda x: Pr.source_ok(x, jnp.asarray(lo), jnp.asarray(hi), be=jax_be))
    assert bool(ok(jnp.asarray(p))) and not bool(ok(jnp.asarray(p).at[0].set(7859.0)))
