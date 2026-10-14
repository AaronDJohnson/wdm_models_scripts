"""F-A5: the annual envelope (spec §5.2 envelope.py): ``C`` and the decimation-grid basis are
the legacy ones bit for bit, ``C.T @ mu_fit`` is Table III, the prior half-widths are Table
IV's X, g-hat has its lag guard, and ``AnnualEnvelope`` reproduces the G6 intermediates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wdm_models.galactic_binary import config as C
from wdm_models.galactic_binary import envelope as E

DATA = Path(__file__).resolve().parents[3] / "data"
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
T = {"gb_3mo": 7884000.0, "gb_6mo": 15768000.0, "gb_12mo": 31536000.0}
#: WDM II Table III, "Power modulation log g_c(t)": (a0, a1, b1, a2, b2) per channel.
TABLE_III = {
    "gb_3mo": ((-7.380, 5.177, 8.298, 0.552, -2.282), (-2.641, 2.656, 2.363, -0.691, -1.116)),
    "gb_6mo": ((-0.948, -0.480, 1.232, 0.047, 0.146), (0.636, -0.397, -0.951, -0.886, -0.047)),
    "gb_12mo": ((-0.039, -0.293, -0.136, -0.550, -0.031), (0.072, -0.366, -0.024, -0.485, -0.055))}
#: WDM II Table IV's X (the modulation coefficients' prior half-width).
TABLE_IV_X = {"gb_3mo": 122.0, "gb_6mo": 21.4, "gb_12mo": 8.0}


def _prepared(case):
    with np.load(DATA / "galactic_binary" / "prepared" / f"{case}.npz", allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in ("basis_L", "mu_fit", "T", "envelope_period")}


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_coefficients_and_decimation_basis_are_the_legacy_ones(case, gb_products, s5):
    env = gb_products["cases"][case]["envelope"]
    Cm = E.orthonormal_coefficients(T[case])
    s5(f"G5/envelope/{case}/C", np.max(np.abs(Cm - np.asarray(env["C"]))), 0.0)
    assert np.array_equal(Cm, np.asarray(env["C"]))
    assert np.array_equal(np.tril(Cm), Cm) and Cm[0, 0] == 1.0
    p = _prepared(case)
    assert float(p["T"]) == T[case] and float(p["envelope_period"]) == C.ENVELOPE_PERIOD
    assert np.array_equal(E.ghat_grid(T[case]), p["basis_L"])          # basis_L_equals_ghat_grid
    t = np.linspace(0.0, T[case], 333, endpoint=False)
    assert np.array_equal(E.basis(t, T[case]), Cm @ E.raw_basis(t))


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_physical_coefficients_are_table_iii(case, gb_products, s5):
    env = gb_products["cases"][case]["envelope"]
    mu = np.asarray(gb_products["cases"][case]["mu_fit"], float)
    for c, name in enumerate(("physical_A", "physical_E")):
        a = E.physical_coefficients(mu[c], T[case])
        s5(f"G5/table_iii/{case}/{'AE'[c]}", np.max(np.abs(a - np.asarray(env[name]))), 0.0)
        assert np.array_equal(a, np.asarray(env[name]))              # legacy C.T @ mu_fit[c]
        assert tuple(round(float(v), 3) for v in a) == TABLE_III[case][c]
    rows = E.physical_coefficients(mu, T[case])
    assert rows.shape == (2, 5) and np.allclose(rows[1], np.asarray(env["physical_E"]), rtol=0, atol=1e-15)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_prior_half_widths_are_table_iv(case, gb_products):
    env = gb_products["cases"][case]["envelope"]
    hw = E.prior_half_widths(T[case])
    assert np.allclose(hw, np.asarray(env["half_widths"]), rtol=1e-15, atol=0)
    assert E.prior_half_width(T[case]) == min(hw)
    assert abs(E.prior_half_width(T[case]) - env["prior_half_width"]) <= 1e-15 * env["prior_half_width"]
    assert round(E.prior_half_width(T[case]), 1 if case != "gb_3mo" else 0) == TABLE_IV_X[case]


def test_ghat_has_the_lag_guard():
    bL = E.ghat_grid(T["gb_3mo"], L=256)
    mu = np.array([0.1, -0.2, 0.05, 0.0, 0.01])
    ok = E.ghat(mu, bL, np.arange(-128, 129))
    R = np.fft.rfft(np.exp(mu @ bL)) / 256
    assert np.array_equal(ok[128:], R[:129]) and np.array_equal(ok[:128], np.conj(R[128:0:-1]))
    with pytest.raises(ValueError, match="Nyquist"):
        E.ghat(mu, bL, np.array([129]))
    with pytest.raises(ValueError, match="Nyquist"):
        E.AnnualEnvelope(T["gb_3mo"], basis_L=bL).ghat(np.array([-200]), mu)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES)
def test_annual_envelope_reproduces_the_g6_intermediates(case, gb_g6):
    """``basis_half`` (the basis at the half-step times Eq. 45 reads) bit for bit, and g-hat on
    the nominal lags at ``mu_fit`` as the D-C4 tests pin it (1e-15 absolute)."""
    _, A = gb_g6(case)
    p = _prepared(case)
    env = E.AnnualEnvelope(T[case], C.ENVELOPE_PERIOD, C.N_HARM, p["basis_L"])
    assert env.n_params == 5 and env.basis_L.shape == (5, C.GHAT_L) and not env.basis_L.flags.writeable
    NT = C.ANALYSIS_NT[case]
    half = np.arange(2 * NT - 1) * (T[case] / NT) / 2
    assert np.array_equal(env.basis(half), A["basis_half"])
    for c in range(2):
        gh = env.ghat(np.arange(A[f"ghat_c{c}"].size), p["mu_fit"][c])
        assert np.allclose(gh, A[f"ghat_c{c}"], rtol=0, atol=1e-15)
    assert np.array_equal(env.log_g(half, p["mu_fit"][0]), p["mu_fit"][0] @ env.basis(half))
    assert np.array_equal(env.physical(p["mu_fit"][0]), E.physical_coefficients(p["mu_fit"][0], T[case]))


@pytest.mark.jax
def test_annual_envelope_on_jax(jax_be):
    import jax
    import jax.numpy as jnp

    env = E.AnnualEnvelope(T["gb_3mo"])
    mu = np.array([0.1, -0.2, 0.05, 0.02, -0.01])
    lags = np.arange(-40, 41)
    want = env.ghat(lags, mu)
    got = jax.jit(lambda m: env.ghat(lags, m, be=jax_be))(jnp.asarray(mu))
    assert np.allclose(np.asarray(got), want, rtol=0, atol=1e-15)
