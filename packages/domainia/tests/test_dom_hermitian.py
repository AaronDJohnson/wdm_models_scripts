"""D-C4: ExactFourierCovariance (M') and HermitianCovariance (GB ``exact``, fullcache port).

* Dense and block-tridiagonal factors reproduce direct inversion (GB test_fullcache.py).
* G6: M' per channel (fingerprint), q0 = |L^-1 z|^2, log|M'| and ghat at the anchor; at
  12 months the block form (b = NT/2) is exact and is what the oracle used.
* G6: the ``exact`` model's per-channel (quad, logdet) by the expanded form
  q0 - 2 Re(h^H y) + |L^-1 h|^2 (GB ``lnl_from_state``).
* The lag guard; ok = False on a non-PD matrix; JAX parity.
"""

from __future__ import annotations

import numpy as np
import pytest

import domtest_inputs as T
from domainia.covariance import ExactFourierCovariance, HermitianCovariance, HermitianFactor

CASES = ["gb_3mo", "gb_6mo", "gb_12mo"]


def _block(case: str, c: T.GBCase) -> int | None:
    return c.NT // 2 if case == "gb_12mo" else None


@pytest.mark.parametrize("block", [None, 8])
def test_factor_reproduces_direct_inversion(backend, block):
    rng = np.random.default_rng(1)
    b, nb = 8, 6
    n = b * nb
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = A @ A.conj().T / n + 3 * np.eye(n)
    if block:
        M = np.where(np.abs(np.arange(n)[:, None] // b - np.arange(n)[None, :] // b) <= 1, M, 0)
    z, h = (rng.standard_normal(n) + 1j * rng.standard_normal(n) for _ in range(2))
    F = HermitianCovariance(M, n, block=block).factor(backend.asarray(np.zeros(1)), backend)
    assert isinstance(F, HermitianFactor) and bool(F.ok)
    Mi = np.linalg.inv(M)
    zz, hh = backend.asarray(z), backend.asarray(h)
    u = F.whiten(zz)
    assert abs(float(np.real(np.vdot(np.asarray(u), np.asarray(u)))) - (z.conj() @ Mi @ z).real) < 1e-10
    assert abs(float(F.logdet) - np.linalg.slogdet(M)[1]) < 1e-10
    assert np.max(np.abs(np.asarray(F.solve(zz)) - Mi @ z)) < 1e-12
    assert abs(float(F.quad(hh)) - (h.conj() @ Mi @ h).real) < 1e-10
    assert abs(float(F.quad(hh, form="solve")) - (h.conj() @ Mi @ h).real) < 1e-10


@pytest.mark.parametrize("case", CASES)
def test_g6_mprime_q0_logdet_and_ghat(case):
    c = T.gb_case(case)
    k = slice(int(c.P["freq_lo"]), int(c.P["freq_lo"]) + int(c.P["freq_n"]))
    for ch in range(2):
        ef = ExactFourierCovariance(c.psd, c.nominal, channel=ch)
        assert ef.n == int(c.P["freq_n"])
        T.assert_summary(ef.evaluate(ef.host_constants(), c.mu[ch], _numpy()), c.g6["fingerprints"][f"Mprime_c{ch}"])
        assert np.allclose(ef.ghat(c.mu[ch], np.arange(ef.n)), c.A[f"ghat_c{ch}"], rtol=0, atol=1e-15)
        F = HermitianCovariance(ef, block=_block(case, c)).factor(c.mu[ch])
        u = F.whiten(c.P["z"][ch, k])
        ref = c.g6["scalars"][f"Mprime_c{ch}"]
        assert float(np.real(np.vdot(u, u))) == pytest.approx(T.num(ref["q0"]), rel=1e-12, abs=0)
        assert float(F.logdet) == pytest.approx(T.num(ref["logdet"]), rel=1e-12, abs=1e-12)


def test_block_form_is_exact_at_twelve_months():
    c = T.gb_case("gb_12mo")
    assert c.g6["scalars"]["exact_block"] == c.NT // 2
    for ch in range(2):
        hc = HermitianCovariance(ExactFourierCovariance(c.psd, c.nominal, channel=ch), block=c.NT // 2)
        assert hc.block_is_exact(c.mu[ch])


@pytest.mark.parametrize("case", CASES)
def test_g6_exact_quad_and_logdet(case):
    """GB ``lnl_from_state``: q0 - 2 Re(h^H y) + |L^-1 h|^2 with y = M'^-1 z."""
    c = T.gb_case(case)
    k = slice(int(c.P["freq_lo"]), int(c.P["freq_lo"]) + int(c.P["freq_n"]))
    inner = slice(int(c.P["bin_offset"]) - c.store_k0, int(c.P["bin_offset"]) - c.store_k0 + int(c.P["n_bins"]))
    got = []
    for ch in range(2):
        F = HermitianCovariance(ExactFourierCovariance(c.psd, c.nominal, channel=ch), block=_block(case, c)).factor(c.mu[ch])
        z, h = c.P["z"][ch, k], c.A["store_template"][ch, inner][k]
        u, t = F.whiten(z), F.whiten(h)
        y = F.backward(u)
        got.append([np.real(np.vdot(u, u)) - 2.0 * np.real(np.vdot(h, y)) + np.real(np.vdot(t, t)), float(F.logdet)])
    assert np.allclose(np.array(got), c.quad_logdet("exact"), rtol=1e-12, atol=1e-10)


def test_lag_guard_and_not_positive_definite():
    c = T.gb_case("gb_3mo")
    small = T.AnnualEnvelope(float(c.P["T"]), float(c.P["envelope_period"]), 2, c.P["basis_L"][:, :256])
    from domainia.spectra import SeparablePSD

    (sI, _), (sG, _) = c.psd.components
    with pytest.raises(ValueError, match="Nyquist"):
        ExactFourierCovariance(SeparablePSD(((sI, None), (sG, small)), ("A", "E")), c.nominal)
    F = HermitianCovariance(-np.eye(4, dtype=complex), 4).factor(np.zeros(1))
    assert not bool(F.ok)


@pytest.mark.jax
@pytest.mark.parametrize("case", ["gb_3mo", "gb_12mo"])
def test_jax_matches_numpy(jax_be, case):
    import jax.numpy as jnp

    c = T.gb_case(case)
    k = slice(int(c.P["freq_lo"]), int(c.P["freq_lo"]) + int(c.P["freq_n"]))
    hc = HermitianCovariance(ExactFourierCovariance(c.psd, c.nominal, channel=0), block=_block(case, c))
    z = c.P["z"][0, k]
    Fn, Fj = hc.factor(c.mu[0]), hc.factor(jnp.asarray(c.mu[0]))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12, abs=1e-10)
    assert float(Fj.quad(jnp.asarray(z))) == pytest.approx(float(Fn.quad(z)), rel=1e-12)


def _numpy():
    from wdm.backend import NUMPY

    return NUMPY


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo"])
def test_tri_is_exact_for_a_constant_envelope(case):
    """Ported GB test_matrix.py::test_tri_is_exact_for_a_constant_envelope: with g(t) constant
    (mu = (0.3, 0, ...)) the separable Eq. (45) is the exact cell covariance 1/2 Re(U M' U^H)
    on the grid window, M' = diag(s_I) + D T(conj ghat) D (every lag kept), to 1e-9."""
    from domainia.covariance import Eq45Covariance

    c = T.gb_case(case)
    mu = np.zeros(c.mu.shape[1])
    mu[0] = 0.3
    g0, G = c.grid_window()
    ex = ExactFourierCovariance(c.psd, c.domain(c.N, g0, G), channel=1)
    U = c.atoms(c.cells, g0, G)
    exact = 0.5 * np.real((U @ ex.evaluate(ex.host_constants(), mu, _numpy())) @ np.conj(U).T)
    got = Eq45Covariance(c.psd, c.cells, builder="separable", assembly="dense", channel=1).dense(mu)
    assert np.max(np.abs(got - exact)) < 1e-9 * np.max(np.abs(exact))
