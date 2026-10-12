"""C4: wdm.noise and wdm.stationary against the frozen oracle, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.noise as onoise
import vendored_2f7136c.stationary as ostat

import wdm.noise as noise
import wdm.stationary as stat

GRIDS = [(4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10)]


def _psd(f):
    f = np.asarray(f, dtype=float)
    return 1.0 + 0.3 / (np.abs(f) + 0.05) ** 2 + 5.0 * f**2


def _psd32(f):
    return _psd(f).astype(np.float32)


def _psd_ft(f, t):
    return np.outer(_psd(f), 1.0 + 0.1 * np.asarray(t) / (1.0 + np.max(t)))


def test_dropped_and_renamed_names():
    for gone in ("build_lambda_blocks", "build_lambda_blocks_packed", "unpack_diag_block",
                 "assemble_lambda", "lambda_to_psd", "lambda_to_psd_slab"):
        assert not hasattr(stat, gone), gone
    assert not hasattr(noise, "_generate_colored_noise")
    assert {"psd_from_K", "psd_from_lambda_blocks"} <= set(stat.__all__)
    assert "colored_noise" in noise.__all__


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("psd", [_psd, _psd32])
def test_variance_builders_are_bitwise_the_oracle(grid, psd):
    Nf, Nt, dt = grid
    for name in ("whittle_variance", "band_averaged_variance"):
        got = getattr(noise, name)(Nf, Nt, dt, psd)
        ref = getattr(onoise, name)(Nf, Nt, dt, psd)
        assert got.dtype == ref.dtype and np.array_equal(got, ref), name
    assert np.array_equal(noise.evolutionary_variance(Nf, Nt, dt, _psd_ft),
                          onoise.evolutionary_variance(Nf, Nt, dt, _psd_ft))
    band = noise.evaluate_psd_on_band_grid(Nf, dt, psd)
    assert np.array_equal(band, onoise.evaluate_psd_on_band_grid(Nf, dt, psd))
    assert np.array_equal(noise.whittle_variance_from_array(Nt, band),
                          onoise.whittle_variance_from_array(Nt, band))


def test_colored_noise_and_empirical_variance_are_bitwise_the_oracle():
    Nf, Nt, dt = 8, 16, 0.5
    a = noise.colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
    b = onoise._generate_colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert np.array_equal(noise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3),
                          onoise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3))


@pytest.mark.parametrize(("N", "dt", "match"), [
    (9, 1.0, "N must be even, got 9"),
    (7, 0.5, "N must be even, got 7"),
    (1, 1.0, "N must be even, got 1"),
    (0, 1.0, "N must be positive, got 0"),
    (-8, 1.0, "N must be positive, got -8"),
    (True, 1.0, "N must be an integer, got bool"),
    (8.0, 1.0, "N must be an integer, got float"),
    (8, 0.0, "dt must be a positive finite number"),
    (8, -1.0, "dt must be a positive finite number"),
    (8, np.inf, "dt must be a positive finite number"),
    (8, np.nan, "dt must be a positive finite number"),
])
def test_colored_noise_rejects_bad_N_and_dt(N, dt, match):
    """``colored_noise`` is public since C4, so it validates like every entry point.

    Unvalidated, an odd N has no Nyquist bin yet bin N//2 was drawn real-only,
    so the noise was not stationary (per-sample variance 0.79--1.22 of its mean
    at N=9, flat PSD, 40k draws); dt < 0 returned NaN with only a warning; dt=0
    and N=0 raised ZeroDivisionError; N=1 raised "negative dimensions".  The
    checks run before the PSD is evaluated or the generator is touched.
    """
    def psd(f):
        raise AssertionError("PSD evaluated before the input checks")

    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match=match):
        noise.colored_noise(N, dt, psd, rng)
    assert rng.bit_generator.state == state


def test_colored_noise_valid_edge_inputs_are_bitwise_the_oracle():
    """The checks only raise: N=2 (DC + Nyquist, no interior) and a NumPy-int N
    still give the oracle's draws bit for bit."""
    a = noise.colored_noise(2, 1.0, _psd, np.random.default_rng(5))
    b = onoise._generate_colored_noise(2, 1.0, _psd, np.random.default_rng(5))
    assert a.shape == (2,) and np.array_equal(a, b)
    c = noise.colored_noise(np.int64(16), 0.5, _psd, np.random.default_rng(5))
    d = onoise._generate_colored_noise(16, 0.5, _psd, np.random.default_rng(5))
    assert np.array_equal(c, d)


@pytest.mark.parametrize("grid", GRIDS)
def test_stationary_builders_are_bitwise_the_oracle(grid, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    S_rfft = _psd(np.fft.rfftfreq(N, d=dt))
    for noise_in in (_psd, S_rfft):
        d1, o1, n1 = stat.build_lambda_blocks_full_active(Nf, Nt, dt, noise_in)
        d2, o2, n2 = ostat.build_lambda_blocks_full_active(Nf, Nt, dt, noise_in)
        assert all(np.array_equal(a, b) for a, b in zip(d1, d2, strict=True))
        assert all(np.array_equal(a, b) for a, b in zip(o1, o2, strict=True))
        assert all(np.array_equal(a, b) for a, b in zip(n1, n2, strict=True))
        assert np.array_equal(stat.assemble_lambda_full_active(d1, o1, n1, Nf, Nt),
                              ostat.assemble_lambda_full_active(d2, o2, n2, Nf, Nt))
        K1, L1 = stat.build_K_full_active(Nf, Nt, dt, noise_in)
        K2, L2 = ostat.build_K_full_active(Nf, Nt, dt, noise_in)
        assert all(np.array_equal(a, b) for a, b in zip(K1 + L1, K2 + L2, strict=True))
        assert np.array_equal(stat.psd_from_K(K1, Nf, Nt, dt),
                              ostat.lambda_to_psd_slab(K2, Nf, Nt, dt))
        assert np.array_equal(stat.psd_from_lambda_blocks(d1, o1, n1, Nf, Nt, dt),
                              ostat.lambda_to_psd(d2, o2, n2, Nf, Nt, dt))
    x = rng.standard_normal((Nf + 1, Nt))
    assert np.array_equal(stat.apply_lambda_from_psd(x, Nf, Nt, dt, S_rfft),
                          ostat.apply_lambda_from_psd(x, Nf, Nt, dt, S_rfft))


@pytest.mark.parametrize("grid", GRIDS)
def test_psd_round_trips(grid):
    Nf, Nt, dt = grid
    N = Nf * Nt
    S = _psd(np.fft.rfftfreq(N, d=dt))
    K, _ = stat.build_K_full_active(Nf, Nt, dt, S)
    d, o, n = stat.build_lambda_blocks_full_active(Nf, Nt, dt, S)
    assert np.max(np.abs(stat.psd_from_K(K, Nf, Nt, dt) / S - 1)) < 1e-12
    assert np.max(np.abs(stat.psd_from_lambda_blocks(d, o, n, Nf, Nt, dt) / S - 1)) < 1e-12


def _complex_psd_forms(Nf, Nt, dt, imag):
    """Every accepted PSD form, made complex: callable, rfft / full 1D, both 2D sizes."""
    N = Nf * Nt
    rf = _psd(np.fft.rfftfreq(N, d=dt)) + imag
    full = _psd(np.abs(np.fft.fftfreq(N, d=dt))) + imag
    return {"callable": lambda f: _psd(f) + imag, "rfft": rf, "full": full,
            "rfft matrix": np.diag(rf), "full matrix": np.diag(full)}


@pytest.mark.parametrize("imag", [1e-3j, 0j])
@pytest.mark.parametrize("build", ["build_lambda_blocks_full_active", "build_K_full_active"])
def test_complex_psd_is_rejected(build, imag):
    """A PSD is real: a complex one (even with a zero imaginary part) raises ValueError
    in every input form, instead of being truncated to its real part (1D and callables
    used to warn ComplexWarning and go on; the 2D form took np.real silently)."""
    import warnings

    Nf, Nt, dt = 8, 16, 1.0
    for form, noise_in in _complex_psd_forms(Nf, Nt, dt, imag).items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # a ComplexWarning must not stand in
            with pytest.raises(ValueError, match="real-valued"):
                getattr(stat, build)(Nf, Nt, dt, noise_in)
                pytest.fail(f"{build} accepted a complex {form} PSD")


def _on_device(out):
    """``out`` on the host, after checking it came back as a ``jax.Array``."""
    import jax

    assert isinstance(out, jax.Array), type(out)      # no eager NumPy fallback
    return np.asarray(out)


def test_apply_lambda_from_psd_on_jax(jax_be, rng):
    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    x = rng.standard_normal((Nf + 1, Nt))
    ref = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    got = _on_device(stat.apply_lambda_from_psd(jax_be.asarray(x), Nf, Nt, dt, jax_be.asarray(S)))
    assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))


def _jit_apply(Nf, Nt, dt):
    """``apply_lambda_from_psd`` under ``jax.jit`` with both x and S traced."""
    import jax

    return jax.jit(lambda x, S: stat.apply_lambda_from_psd(x, Nf, Nt, dt, S))


def test_apply_lambda_from_psd_under_jit(jax_be, rng):
    """The tracer branch (x and S both traced) matches eager JAX and NumPy."""
    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    x = rng.standard_normal((Nf + 1, Nt))
    ref = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    xj, Sj = jax_be.asarray(x), jax_be.asarray(S)
    eager = _on_device(stat.apply_lambda_from_psd(xj, Nf, Nt, dt, Sj))
    got = _on_device(_jit_apply(Nf, Nt, dt)(xj, Sj))
    assert np.max(np.abs(got - eager)) <= 1e-13 * np.max(np.abs(eager))
    assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))


def test_apply_lambda_from_psd_inf_psd_under_jit(jax_be, rng):
    """An inf PSD sample (the LISA f = 0 case) is zero power on every path.

    The host path maps it to 0.0 through ``sanitize_psd``; the tracer branch
    must apply the same policy or the inf leaks into the coefficients.
    """
    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    S[0] = np.inf
    x = rng.standard_normal((Nf + 1, Nt))
    ref = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    assert np.all(np.isfinite(ref))
    xj, Sj = jax_be.asarray(x), jax_be.asarray(S)
    eager = _on_device(stat.apply_lambda_from_psd(xj, Nf, Nt, dt, Sj))
    got = _on_device(_jit_apply(Nf, Nt, dt)(xj, Sj))
    assert np.all(np.isfinite(got)), f"{np.count_nonzero(~np.isfinite(got))} non-finite"
    assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))
    assert np.max(np.abs(got - eager)) <= 1e-13 * np.max(np.abs(eager))


def test_apply_lambda_from_psd_explicit_numpy_backend(jax_be, rng):
    """``be=NUMPY`` on JAX inputs runs the host path: bitwise the NumPy result."""
    from wdm.backend import NUMPY

    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    x = rng.standard_normal((Nf + 1, Nt))
    ref = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    got = stat.apply_lambda_from_psd(jax_be.asarray(x), Nf, Nt, dt, jax_be.asarray(S), be=NUMPY)
    assert isinstance(got, np.ndarray)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
