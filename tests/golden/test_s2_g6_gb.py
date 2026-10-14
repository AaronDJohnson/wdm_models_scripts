"""D-D7 (S2 gate, G6): the GB models assembled from domainia blocks reproduce
golden/galactic_binary/intermediates_gb_3mo.{json,npz} at the anchor.

A preview of the S6 wiring (spec S2 gate): whittle_raw (complex Whittle on the nominal
bins), band_raw (separable diagonal), tri_exact (Eq. 45 separable + exact edge rows,
dense assembly) and exact (Hermitian M', expanded strategy).  Gates: lnL at the approved
G3 NumPy gate (1e-8), each channel's (quad, logdet) relative 1e-12; the GB numbers are
not claimed bitwise (spec section 3.6).  Also checked: the data cells (V_raw_cells), the
band variances at the anchor and the M' noise state per channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.goldens_io import to_float
from scripts.golden.s2_cases import GB_CASE, GB_MODELS, gb_compare, gb_gate, gb_inputs, gb_likelihood, record

pytestmark = pytest.mark.golden
ROOT = f"galactic_binary/{GB_CASE}"


@pytest.mark.parametrize("model", GB_MODELS)
def test_quad_logdet_lnl_at_the_anchor(model, record_property):
    out = gb_compare(GB_CASE, model)
    gates = {k: (gb_gate(GB_CASE, model) if k == "lnL" else 1e-12 * abs(ref)) for k, (_, ref, _) in out.items()}
    for key, (val, ref, d) in out.items():
        record(record_property, f"{ROOT}/quad_logdet/{model}/composition/{key}", d, gates[key])
    for key, (val, ref, d) in out.items():
        assert d <= gates[key], (model, key, val, ref)


def test_data_cells_and_band_variances(record_property):
    g = gb_inputs(GB_CASE)
    lik = gb_likelihood(GB_CASE, "band_raw")
    cells = np.asarray(lik.parts.y_data)
    d = float(np.max(np.abs(cells - g.npz["V_raw_cells"])))
    record(record_property, f"{ROOT}/arrays/V_raw_cells", d, 1e-12)
    assert d <= 1e-12
    state = lik.noise_state(lik.noise_slice_host(g.anchor))
    var = np.stack([np.asarray(state.factors[c].v) for c in range(2)])
    ref = g.npz["var_band_raw"]
    d = float(np.max(np.abs(var - ref) / np.abs(ref)))
    record(record_property, f"{ROOT}/arrays/var_band_raw", d, 1e-12)
    assert d <= 1e-12


def test_exact_noise_state_is_mprime(record_property):
    g = gb_inputs(GB_CASE)
    lik = gb_likelihood(GB_CASE, "exact")
    state = lik.noise_state(lik.noise_slice_host(g.anchor))
    for c in range(2):
        ref = g.json["scalars"][f"Mprime_c{c}"]
        dq = abs(float(state.q0[c]) - to_float(ref["q0"])) / abs(to_float(ref["q0"]))
        dl = abs(float(state.factors[c].logdet) - to_float(ref["logdet"])) / abs(to_float(ref["logdet"]))
        record(record_property, f"{ROOT}/scalars/Mprime_c{c}", max(dq, dl), 1e-12)
        assert max(dq, dl) <= 1e-12, (c, dq, dl)


def test_cells_or_bins_other_than_the_maps_are_refused():
    """Review regression (G-F1): equal-sized but different cell sets or bins used to pass
    every shape check.  band_raw with the covariances on the bands shifted by one (720
    cells, as many as the map's) gave lnL -1428.3264 against the golden -1428.2877, and a
    map reading the store shifted by one bin gave -1427.3035, both silently; the
    likelihood now refuses both at construction."""
    import wdm
    from domainia.covariance import ChannelStack, DiagonalCovariance
    from domainia.data import FourierBins
    from domainia.density import RealGaussian
    from domainia.domains import CellSet, ChannelAxis, FourierDomain, WDMDomain
    from domainia.likelihood import GaussianLikelihood, StaticSignal
    from domainia.maps import FourierToWDM
    from domainia.spectra import SeparablePSD, TabulatedSpectrum
    from domainia.variance import BandAveragedVariance
    from scripts.golden.s2_cases import ENV, GB_PARAMS, gb_envelope

    g = gb_inputs(GB_CASE)
    a = g.a
    Nr, dtg, nt = int(a["N"]), float(a["DT"]), int(a["NT"])
    store_dom = FourierDomain(Nr, dtg, int(a["store_offset"]), int(a["store_n"]), "sigma0")
    store = FourierBins(a["store_z"], store_dom, ChannelAxis(("A", "E")))
    grid = wdm.Grid(Nr // nt, nt, dtg)
    fb, nb = int(a["first_band"]), int(a["n_bands"])
    cells = CellSet.bands(grid, tuple(fb + b for b in range(nb)))
    shifted = CellSet.bands(grid, tuple(fb + 1 + b for b in range(nb)))
    assert shifted.size == cells.size and shifted != cells
    psd = SeparablePSD(((TabulatedSpectrum(store_dom, a["store_sI"]), None),
                        (TabulatedSpectrum(store_dom, a["store_sG"]), gb_envelope(GB_CASE))), ("A", "E"))

    def lik(cov_cells, map_src=store_dom):
        covs = ChannelStack(tuple(DiagonalCovariance(BandAveragedVariance(psd, cov_cells), cov_cells.size,
                                                     channel=c) for c in range(2)), len(ENV))
        return GaussianLikelihood(store, FourierToWDM(map_src, WDMDomain(cells), impl="dense"), covs,
                                  RealGaussian(), signal=StaticSignal(lambda src, h=g.npz["store_template"]: h),
                                  params=GB_PARAMS, quad_form="elementwise")

    want = to_float(g.json["lnl_at_anchor"]["band_raw"]["lnl_make_lnlike"])
    assert abs(float(lik(cells)(g.anchor[None, :])[0]) - want) <= gb_gate(GB_CASE, "band_raw")
    with pytest.raises(ValueError, match="cells"):
        lik(shifted)
    moved = FourierDomain(Nr, dtg, store_dom.k0 + 1, store_dom.n, "sigma0")
    with pytest.raises(ValueError, match="domain"):
        lik(cells, map_src=moved)


def test_exact_bins_other_than_the_maps_are_refused():
    """S5 regression (task F-A10; spec section 4 "As built in S2"): the ``exact`` model with its
    M' on the nominal bins shifted by one (360 bins, as many as the map's) gave lnL
    -600.5157309322215 against the golden -600.5155231450001, silently; the likelihood now
    refuses a ``HermitianCovariance`` whose ``bins`` are not the map's ``dst``."""
    from domainia.covariance import ChannelStack, ExactFourierCovariance, HermitianCovariance
    from domainia.data import FourierBins
    from domainia.density import ComplexGaussian
    from domainia.domains import ChannelAxis, FourierDomain
    from domainia.likelihood import GaussianLikelihood, StaticSignal
    from domainia.maps import BinSelect
    from domainia.spectra import SeparablePSD, TabulatedSpectrum
    from scripts.golden.s2_cases import ENV, GB_PARAMS, gb_envelope

    g = gb_inputs(GB_CASE)
    a = g.a
    Nr, dtg = int(a["N"]), float(a["DT"])
    store_dom = FourierDomain(Nr, dtg, int(a["store_offset"]), int(a["store_n"]), "sigma0")
    store = FourierBins(a["store_z"], store_dom, ChannelAxis(("A", "E")))
    nominal = FourierDomain(Nr, dtg, int(a["bin_offset"]) + int(a["freq_lo"]), int(a["freq_n"]), "sigma0")
    psd = SeparablePSD(((TabulatedSpectrum(store_dom, a["store_sI"]), None),
                        (TabulatedSpectrum(store_dom, a["store_sG"]), gb_envelope(GB_CASE))), ("A", "E"))

    def lik(shift):
        bins = FourierDomain(Nr, dtg, nominal.k0 + shift, nominal.n, "sigma0")
        covs = ChannelStack(tuple(HermitianCovariance(ExactFourierCovariance(psd, bins, channel=c))
                                  for c in range(2)), len(ENV))
        return GaussianLikelihood(store, BinSelect(store_dom, nominal), covs, ComplexGaussian(),
                                  signal=StaticSignal(lambda src, h=g.npz["store_template"]: h),
                                  params=GB_PARAMS, strategy="expanded")

    want = to_float(g.json["lnl_at_anchor"]["exact"]["lnl_make_lnlike"])
    assert abs(float(lik(0)(g.anchor[None, :])[0]) - want) <= gb_gate(GB_CASE, "exact")
    for shift in (1, -1):
        with pytest.raises(ValueError, match="bins"):
            lik(shift)
