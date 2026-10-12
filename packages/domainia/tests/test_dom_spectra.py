"""D-B6: LogLinearPSD, TabulatedSpectrum, SeparablePSD, GenericPSD, the Envelope protocol."""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from dom_db_inputs import GB_CASES, SIN_DT, SIN_FMIN, SIN_T, TableEnvelope, gb_case, jax_or_skip

from domainia.domains import FourierDomain
from domainia.spectra import (Envelope, GenericPSD, LogLinearPSD, SeparablePSD,
                              TabulatedSpectrum)

PSD = LogLinearPSD(f_floor=SIN_FMIN, f_ref=1.0, T=SIN_T)


def test_loglinear_L_and_alpha_are_the_wt_formulas():
    f = np.fft.rfftfreq(16384, d=SIN_DT)
    # WT noise.log_f_clip (noise.py:105-107) and matrix._alpha (matrix.py:252-254), verbatim
    L_wt = np.log(np.maximum(np.asarray(f, dtype=float), 4.0) / 1.0)
    assert np.array_equal(PSD.L(f), L_wt)
    t = np.arange(256) * wdm.delta_T(64, SIN_DT)
    a_wt = -2.0 + 1.2 * (np.asarray(t, float) / 16.0 - 0.5)
    assert np.array_equal(PSD.alpha(t, (-2.0, 1.2)), a_wt)
    assert PSD.capabilities == {"gram", "bessel", "operator"}
    assert PSD.nu_names == ("alpha_c", "alpha_d")


def test_loglinear_call_is_elementwise_unit_amplitude():
    f = np.array([1.0, 4.0, 16.0, 64.0])
    t = np.array([0.0, 4.0, 8.0, 16.0])
    S = PSD(f, t, (-2.0, 1.2))
    a = -2.0 + 1.2 * (t / 16.0 - 0.5)
    assert S.shape == (4,) and np.array_equal(S, np.exp(a * np.log(np.maximum(f, 4.0))))
    assert PSD(f[:, None], t[None, :], (-2.0, 1.2)).shape == (4, 4)       # broadcasting
    Sn = PSD.bind((-2.0, 0.0))
    assert np.array_equal(Sn(f, 3.0), np.exp(-2.0 * np.log(np.maximum(f, 4.0))))


def test_tabulated_spectrum_slices_by_absolute_bin_and_refuses_outside():
    dom = FourierDomain(4096, 5.0, 100, 50, "sigma0")
    v = np.arange(100.0).reshape(2, 50)
    s = TabulatedSpectrum(dom, v)
    assert s.channels == 2
    assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, "sigma0")), v[:, 10:15])
    assert np.array_equal(s.at([100, 149]), v[:, [0, 49]])
    assert np.array_equal(s.at([100.0]), v[:, [0]]) and s.at([]).shape == (2, 0)
    assert not np.shares_memory(s.values, v) and not s.values.flags.writeable   # a frozen copy
    with pytest.raises(ValueError, match="integers"):
        s.at([100.5])                                                            # never truncated
    with pytest.raises(ValueError, match="outside"):
        s.on(FourierDomain(4096, 5.0, 140, 20, "sigma0"))
    with pytest.raises(ValueError, match="outside"):
        s.at([99])
    with pytest.raises(ValueError, match="record"):
        s.on(FourierDomain(8192, 5.0, 110, 5, "sigma0"))
    with pytest.raises(ValueError, match="channels"):
        TabulatedSpectrum(dom, np.zeros((2, 49)))


def test_store_spectra_on_the_window_are_the_prepared_spectra():
    """Ported from GB tests/test_matrix.py:43-49 (the store's inner bins are the prepared data)."""
    for case in GB_CASES:
        c = gb_case(case)
        p = c["prep"]
        store = FourierDomain(c["N"], c["DT"], int(p["store_offset"]), int(p["store_n"]), "sigma0")
        window = FourierDomain(c["N"], c["DT"], int(p["bin_offset"]), int(p["n_bins"]), "sigma0")
        assert np.array_equal(TabulatedSpectrum(store, p["store_sI"]).on(window), p["sI"])
        assert np.array_equal(TabulatedSpectrum(store, p["store_sG"]).on(window), p["sG"])


def test_separable_psd_validates_its_components():
    c = gb_case("gb_3mo")
    p = c["prep"]
    store = FourierDomain(c["N"], c["DT"], int(p["store_offset"]), int(p["store_n"]), "sigma0")
    env = TableEnvelope(c["g6"]["basis_half"], c["grid"].dT)
    assert isinstance(env, Envelope)  # structural: basis, log_g, ghat, n_params
    sI, sG = TabulatedSpectrum(store, p["store_sI"]), TabulatedSpectrum(store, p["store_sG"])
    psd = SeparablePSD(((sI, None), (sG, env)), ("A", "E"))
    assert psd.domain == store and psd.envelope is env
    assert psd.capabilities == {"separable", "fourier_toeplitz"}
    with pytest.raises(ValueError, match="at most one modulated"):
        SeparablePSD(((sI, env), (sG, env)))
    with pytest.raises(ValueError, match="channel names"):
        SeparablePSD(((sI, None), (sG, env)), ("A",))
    other = TabulatedSpectrum(FourierDomain(c["N"], c["DT"], 0, 10, "sigma0"), np.ones((2, 10)))
    with pytest.raises(ValueError, match="share the domain"):
        SeparablePSD(((sI, None), (other, env)))


def test_generic_psd_binds():
    g = GenericPSD(lambda f, t, nu: nu[0] * f + t)
    assert g(2.0, 1.0, (3.0,)) == 7.0 and g.bind((3.0,))(2.0, 1.0) == 7.0
    assert g.capabilities == {"interp", "direct"}


@pytest.mark.jax
def test_loglinear_traced_parameters():
    jax, JB = jax_or_skip()
    t = np.arange(256) * wdm.delta_T(64, SIN_DT)
    a = jax.jit(lambda nu: PSD.alpha(t, nu))(JB.asarray([-2.0, 1.2]))
    assert isinstance(a, jax.Array)
    assert np.max(np.abs(np.asarray(a) - PSD.alpha(t, (-2.0, 1.2)))) <= 1e-15
