"""F-C1: the complex Whittle form maps its data at the oracle's call site.

GB ``whittle_cut`` crops the store bins to the record without the edge columns
(``FourierCrop``); lineage C maps the *data* as ``A z^T + B conj(z)^T``
(``galactic_binary/matrix.py:892``, :meth:`FourierCrop.apply_data`) and the *template* as
``h A^T + conj(h) B^T`` (``apply``).  The two orientations differ by up to 2e-15, which made
the release whittle_cut 1.1e-13 off lineage C at the 3-month anchor; mapping the data with
``apply_data`` (where the map has one) makes it bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import domtest_inputs as T
from domainia.data import FourierBins
from domainia.domains import ChannelAxis, FourierDomain
from domainia.likelihood import FourierWhittle, Params, StaticSignal
from domainia.maps import FourierCrop

pytestmark = pytest.mark.golden
GLASS = ("f0T", "cos_colat", "lon", "lnA", "cos_iota", "psi", "phi0", "fdotT2")


def test_complex_whittle_maps_its_data_at_the_oracles_call_site(same_build):
    """3-month whittle_cut: the cropped data are ``FourierCrop.apply_data`` bit for bit; on the
    recording build (quad, logdet) per channel equal G6 bit for bit (1e-12 relative elsewhere)."""
    c = T.gb_case("gb_3mo")
    store = c.domain(c.N, c.store_k0, c.store_n)
    lead, trail = 5, 4
    Nc = (c.NT - lead - trail) * c.Nf
    crop = FourierDomain(Nc, c.DT, int(c.P["crop_k0"]), int(c.P["crop_n"]), "sigma0")
    fc = FourierCrop(store, crop, lead * c.Nf)
    factor = Nc / c.N
    lik = FourierWhittle("complex", FourierBins(c.P["store_z"], store, ChannelAxis(("A", "E"))), map=fc,
                         sI=factor * c.P["crop_sI"], sG=factor * c.P["crop_sG"],
                         signal=StaticSignal(lambda src, h=c.A["store_template"]: h),
                         params=Params(GLASS + ("log_gbar_A", "log_gbar_E"), tuple(range(8)), (8, 9)))
    assert np.array_equal(lik.parts.data, fc.apply_data(c.P["store_z"]))
    theta = np.concatenate([c.P["inject_glass"], c.P["mu_freq"]])
    got, want = lik.quad_logdet(theta), c.quad_logdet("whittle_cut")
    if same_build(c.g6["build"]):
        assert np.array_equal(got, want)
    assert np.allclose(got, want, rtol=1e-12, atol=0)
