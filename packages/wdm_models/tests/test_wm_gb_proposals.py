"""F-B6: ``galactic_binary.proposals`` -- the source Fisher against finite differences, the GLASS
mix (weights, refresh, per-rung caches), binary_galactica's whole-vector rejection at the
cosine boundaries, the noise Fisher and the initial covariance.  No sampling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wdm_models.galactic_binary import proposals as P

REL = Path(__file__).resolve().parents[3]
PREPARED = REL / "data" / "galactic_binary" / "prepared"
#: A point off the integer carrier bin (``f0 T`` = 7884.3): ``q = trunc(f0 T)`` is constant
#: across every finite-difference step below.
OFFSET = np.array([0.3, 0.02, 0.05, 0.01, 0.03, 0.04, 0.05, 1e-6])
STEPS = np.array([1e-4, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8])

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def arrays():
    with np.load(PREPARED / "gb_3mo.npz", allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


@pytest.fixture(scope="module")
def bg_x64():
    bg = pytest.importorskip("binary_galactica", reason="the templates need binary_galactica (JAX)")
    from wdm.backend import enable_x64

    enable_x64()
    return bg


def _template(a, bg, backend):
    """The window template in prepared units, straight from binary_galactica."""
    import jax

    T, lo, n = float(a["T"]), int(a["bin_offset"]), int(a["n_bins"])
    scale = float(a["DT"]) * np.asarray(a["sigma0"])[:, None]
    with jax.ensure_compile_time_eval():
        orbit = bg.orbit_cache(T, 0.0, 4096)
    return lambda p: bg.template_on_window(p, T=T, bin_lo=lo, n_bins=n, BW=4096, positions=orbit,
                                           backend=backend) / scale


@pytest.mark.parametrize("backend", ["fourier", "onfly"])
def test_the_fisher_is_the_finite_difference_fisher(arrays, bg_x64, backend):
    """``jacfwd`` of the template against central differences (per-parameter steps): every
    Jacobian column and ``2 Re(J^H W J)`` agree to 1e-6 relative (measured <= 6.4e-8)."""
    tpl = _template(arrays, bg_x64, backend)
    fisher = P.SourceFisher.from_prepared(arrays, template=tpl)
    p = arrays["inject_glass"] + OFFSET
    J = fisher.jacobian(p)
    Jfd = np.empty_like(J)
    for i in range(8):
        e = np.zeros(8)
        e[i] = STEPS[i]
        Jfd[..., i] = (np.asarray(tpl(p + e)) - np.asarray(tpl(p - e))) / (2 * STEPS[i])
    for i in range(8):
        assert np.max(np.abs(J[..., i] - Jfd[..., i])) <= 1e-6 * np.max(np.abs(J[..., i])), i
    W = np.asarray(fisher.weight)
    Ffd = 2.0 * np.real(np.einsum("cki,ckj,ck->ij", np.conj(Jfd), Jfd, W))
    F = fisher(p)
    assert np.max(np.abs(F - 0.5 * (Ffd + Ffd.T))) <= 1e-6 * np.max(np.abs(F))
    assert np.array_equal(F, F.T) and np.linalg.eigvalsh(F).min() > 0


def test_the_fisher_weight_and_template_come_from_the_prepared_case(arrays, bg_x64):
    """``from_prepared``: ``W = 1/(s_I + e^{mu_freq} s_G)`` on the window; the default template is
    ``waveform.GBTemplate(prep, window="window")`` (a jit argument of the Jacobian), identical
    to binary_galactica's on the window; a bare mapping needs ``template=``."""
    from wdm_models.galactic_binary.prepared import PreparedCase
    from wdm_models.galactic_binary.waveform import GBTemplate

    import inspect

    assert inspect.signature(P.SourceFisher.from_prepared).parameters["backend"].default == "onfly"
    fisher = P.SourceFisher.from_prepared(PreparedCase(PREPARED / "gb_3mo.npz"), backend="fourier")
    assert isinstance(fisher.template, GBTemplate) and fisher._tpl is fisher.template
    assert (fisher.template.bin_lo, fisher.template.n_bins) == (int(arrays["bin_offset"]), int(arrays["n_bins"]))
    with pytest.raises(TypeError, match="PreparedCase"):
        P.SourceFisher.from_prepared(arrays)
    want = 1.0 / (arrays["sI"] + np.exp(arrays["mu_freq"])[:, None] * arrays["sG"])
    assert np.array_equal(np.asarray(fisher.weight), want)
    p = arrays["inject_glass"] + OFFSET
    direct = _template(arrays, bg_x64, "fourier")(p)
    assert np.max(np.abs(np.asarray(fisher.template(p)) - np.asarray(direct))) <= 1e-12 * np.max(np.abs(direct))
    ev, V = fisher.eigen(p)
    F = fisher(p)
    assert np.all(np.diff(ev) >= 0) and np.max(np.abs(V @ np.diag(ev) @ V.T - F)) <= 1e-12 * np.max(np.abs(F))


def test_a_non_finite_fisher_gives_the_self_proposal_oracle(arrays, bg_x64):
    nan = P.SourceFisher(lambda p: p[0] * np.nan * np.ones((2, 3)), np.ones((2, 3)))
    ev, V = nan.eigen(arrays["inject_glass"])
    assert np.all(np.isnan(ev)) and np.array_equal(V, np.eye(8))


class _Oracle:
    """A Fisher oracle with a fixed decomposition that counts its calls."""

    def __init__(self, ev, V):
        self.ev, self.V, self.calls = np.asarray(ev, float), np.asarray(V, float), 0

    def eigen(self, p8):
        self.calls += 1
        return self.ev, self.V


def test_the_mix_is_glass_single_source(arrays, bg_x64):
    """Weights fisher 0.7 / fm_shift 0.1 / prior 0.1 / psi_phi 0.1; the Fisher decomposition is
    refreshed on Fisher calls 1, 81, 161 (``FISHER_REFRESH_STEPS = 80``); one mix per rung."""
    oracle = _Oracle(np.full(8, 1e4), np.eye(8))
    prop = P.GlassProposal.from_prepared(arrays, oracle)
    assert prop.every == 80 and prop.__name__ == "glass_single_source"
    mix = prop.mix(0)
    assert mix.names == ["fisher", "fm_shift", "prior", "psi_phi"]
    assert np.allclose(mix.p, [0.7, 0.1, 0.1, 0.1], rtol=1e-15, atol=0)
    rng = np.random.default_rng(1)
    x = arrays["inject_glass"].copy()
    for _ in range(161):
        mix.proposals["fisher"](x, rng)
    assert oracle.calls == 3
    assert prop.mix(1) is not mix and prop.mix(0) is mix
    assert np.array_equal(prop.prior.lo, arrays["prior_lo"]) and np.array_equal(prop.prior.hi, arrays["prior_hi"])


def test_cosine_boundary_and_non_finite_jumps_are_whole_vector_self_proposals(arrays, bg_x64):
    """Every eigendirection moves ``cos_iota`` by ~1e3: each Fisher jump crosses +-1 and returns
    a copy of the current vector with ``log q = 0`` (binary_galactica), never a component revert."""
    V = np.zeros((8, 8))
    V[4, :] = 1.0                                               # each direction: all on cos_iota
    prop = P.GlassProposal.from_prepared(arrays, _Oracle(np.full(8, 1e-6), V))
    rng = np.random.default_rng(2)
    x = arrays["inject_glass"].copy()
    for _ in range(50):
        out, logq = prop.mix(0).proposals["fisher"](x, rng)
        assert np.array_equal(out, x) and out is not x and logq == 0.0
    nan = P.GlassProposal.from_prepared(arrays, _Oracle(np.full(8, np.nan), np.eye(8)))
    out, logq = nan.mix(0).proposals["fisher"](x, rng)
    assert np.array_equal(out, x) and logq == 0.0


def test_the_custom_jump_moves_only_the_source_block(arrays, bg_x64):
    prop = P.GlassProposal.from_prepared(arrays, _Oracle(np.full(8, 1e4), np.eye(8)))
    x = np.concatenate([arrays["inject_glass"], arrays["mu_fit"].ravel()])
    rng = np.random.default_rng(3)
    moved = 0
    for _ in range(40):
        stats = SimpleNamespace(current_sample=x, chain_index=2, rng=rng)
        y, logq = prop(stats)
        assert y.shape == (18,) and logq == 0.0 and np.array_equal(y[8:], x[8:])
        moved += not np.array_equal(y[:8], x[:8])
    assert moved > 0 and np.array_equal(x[:8], arrays["inject_glass"])    # the input is not mutated
    assert set(prop._mix) == {2}


def test_the_noise_fisher(arrays, bg_x64):
    """Whittle: one level per channel, ``sum (e^mu s_G / M)^2`` on the nominal bins.  Envelope
    models: ``tr(M^-1 dM_k M^-1 dM_l)`` with ``dM`` from ``jacfwd`` of domainia's exact
    covariance (checked against central differences), block-diagonal over channels."""
    a = arrays
    k = slice(int(a["freq_lo"]), int(a["freq_lo"]) + int(a["freq_n"]))
    Fw = P.noise_fisher(a, "whittle_cut")
    for c in range(2):
        level = np.exp(a["mu_freq"][c]) * a["sG"][c, k]
        assert Fw[c, c] == np.sum((level / (a["sI"][c, k] + level)) ** 2)
    assert Fw[0, 1] == Fw[1, 0] == 0.0
    Fn = P.noise_fisher(a, "tri_exact")
    assert Fn.shape == (10, 10) and np.linalg.eigvalsh(0.5 * (Fn + Fn.T)).min() > 0
    assert np.max(np.abs(Fn - Fn.T)) <= 1e-12 * np.max(np.abs(Fn))     # legacy: not symmetrised
    assert np.all(Fn[:5, 5:] == 0) and np.array_equal(P.noise_fisher(a, "band_raw"), Fn)
    import jax

    cov = P.exact_covariance_fn(a, 0)
    mu = a["mu_fit"][0]
    dM = np.asarray(jax.jacfwd(cov)(mu))
    for i in range(5):
        e = np.zeros(5)
        e[i] = 1e-6
        fd = (np.asarray(cov(mu + e)) - np.asarray(cov(mu - e))) / 2e-6
        assert np.max(np.abs(dM[..., i] - fd)) <= 1e-6 * np.max(np.abs(dM[..., i]))
    with pytest.raises(ValueError, match="unknown model"):
        P.noise_fisher(a, "frequency")


def test_the_initial_covariance(arrays, bg_x64):
    fisher = P.SourceFisher.from_prepared(arrays, template=_template(arrays, bg_x64, "fourier"))
    width = arrays["prior_hi"] - arrays["prior_lo"]
    for model, n in (("whittle_raw", 10), ("exact", 18)):
        C = P.initial_covariance(arrays, model, fisher)
        assert C.shape == (n, n) and np.array_equal(C, C.T) and np.linalg.eigvalsh(C).min() > 0
        src = np.linalg.inv(fisher(arrays["inject_glass"]) + np.diag(12.0 / width ** 2))
        assert np.allclose(C[:8, :8], 0.5 * (src + src.T), rtol=1e-12, atol=0)
        assert np.all(C[:8, 8:] == 0)
