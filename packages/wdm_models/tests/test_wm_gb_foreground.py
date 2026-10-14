"""F-A4: the GLASS foreground is the chains' form bit for bit, and Eq. (90)'s parameters
reproduce Table III (spec §5.2 foreground.py)."""

from __future__ import annotations

import numpy as np
import pytest

from wdm_models.galactic_binary import foreground as F
from wdm_models.galactic_binary import instrument as I

CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
#: WDM II Table III (tab:foreground-coeffs): A_fg [1/Hz], alpha, s1, s2, x_k per duration.
TABLE_III = {"gb_3mo": (1.158e-40, 1.716, 24.13, 1909.0, 0.2093),
             "gb_6mo": (1.302e-40, 1.713, 35.18, 43.77, 0.2027),
             "gb_12mo": (1.728e-40, 0.923, 8.972, 39.92, 0.1486)}
#: The one Table III entry the mapping misses at 0.1%: 6-month s1 is 35.1436 (table 35.18,
#: -0.104%).  Reported to the user as an erratum candidate; every other entry is within 0.1%.
TABLE_III_OFF = {("gb_6mo", "s1"): (35.1436, 1.1e-3)}


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_glass_foreground_is_the_chains_form_bit_for_bit(case, gb_psd_window, gb_products, s5):
    x = np.asarray(gb_products["cases"][case]["fg_params"], float)
    f = gb_psd_window[f"{case}__freqs"]
    want = gb_psd_window[f"{case}__S_fg"]
    s5(f"G5/psd/{case}/S_fg", np.max(np.abs(F.glass_psd(f, x) - want[0])), 0.0)
    assert np.array_equal(F.glass_psd(f, x), want[0]) and np.array_equal(want[0], want[1])


@pytest.mark.golden
def test_glass_to_ms_reproduces_table_iii(gb_products):
    worst = 0.0
    for case, row in TABLE_III.items():
        ms = F.glass_to_ms(np.asarray(gb_products["cases"][case]["fg_params"], float))
        assert tuple(ms) == F.MS_NAMES
        for name, table in zip(("A_fg", "alpha", "s1", "s2", "x_k"), row):
            rel = abs(ms[name] / table - 1.0)
            if (case, name) in TABLE_III_OFF:
                value, gate = TABLE_III_OFF[(case, name)]
                assert round(ms[name], 4) == value and 1e-3 < rel < gate, (case, name, ms[name])
                continue
            worst = max(worst, rel)
            assert rel <= 1e-3, (case, name, ms[name], table)
    assert worst < 9.3e-4


def test_eq90_is_the_glass_form_times_sin2x_over_x2():
    x = np.array([-84.66474739026393, -5.814244748161286, 1.71580595333524,
                  -5.522557541691449, -11.512925462027555])            # the 3-month fit
    f = np.linspace(2e-4, 4e-3, 64)
    u = f / I.F_STAR
    ratio = F.ms_eq90_psd(f, F.glass_to_ms(x)) / F.glass_psd(f, x)
    assert np.max(np.abs(ratio / (np.sin(u) ** 2 / u ** 2) - 1.0)) < 1e-13
    at_1mHz = F.ms_eq90_psd([1e-3], F.glass_to_ms(x))[0] / F.glass_psd([1e-3], x)[0]
    assert round(at_1mHz, 5) == 0.99909
    ms = F.glass_to_ms(x)
    assert np.array_equal(F.ms_eq90_psd(f, ms), F.ms_eq90_psd(f, [ms[k] for k in F.MS_NAMES]))
