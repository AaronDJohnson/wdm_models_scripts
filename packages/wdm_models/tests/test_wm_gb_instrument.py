"""F-A3: the LISA A/E instrument PSD, re-homed from ``_nora`` (spec §5.2 instrument.py), is
the original bit for bit on every window grid (golden/galactic_binary/psd_window.npz is lineage
C's ``prepare.psd_components``)."""

from __future__ import annotations

import numpy as np
import pytest

from wdm_models.galactic_binary import instrument as I

CASES = ("gb_3mo", "gb_6mo", "gb_12mo")


def test_constants_follow_nora():
    assert (I.C_LIGHT, I.ARM_LENGTH, I.TWO_PI) == (2.99792458e8, 2.5e9, 2.0 * np.pi)
    assert I.F_STAR == I.C_LIGHT / (I.ARM_LENGTH * I.TWO_PI) == 0.01908538063694777
    P = I.SANGRIA_NOISE_PARAMS
    assert (P.S_oms_A, P.S_a_A, P.S_oms_E, P.S_a_E) == (6.241e-23, 5.76e-30, 6.241e-23, 5.76e-30)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_instrument_psd_is_the_original_bit_for_bit(case, gb_psd_window, s5):
    """psd_window.npz is lineage C's ``prepare.psd_components`` (``_nora`` model) on the
    window's bins; both channels carry the same PSD."""
    f = gb_psd_window[f"{case}__freqs"]
    got = I.tdi_instrumental_psd_AE(f)
    want = gb_psd_window[f"{case}__S_inst"]
    s5(f"G5/psd/{case}/S_inst", np.max(np.abs(got - want[0])), 0.0)
    assert np.array_equal(want[0], want[1])
    assert np.array_equal(got, want[0])
    assert np.array_equal(I.tdi_instrumental_psd_AE(f, L=2.5e9), got)
    other = I.tdi_instrumental_psd_AE(f, L=2.0e9)
    assert not np.array_equal(other, got)                 # L enters through f*
