"""C3: the transform over the backend shim.

NumPy: bit-identical (np.array_equal) to the frozen oracle vendored_2f7136c on
random grids, dt != 1, float32 and float64, batched and unbatched.
JAX: <= 1e-13 * max|w| (float64), and jit/vmap-able.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.transform as oracle

import wdm.transform as T

GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
         (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]
DTYPES = [np.float64, np.float32]


def _x(rng, shape, dtype):
    return rng.standard_normal(shape).astype(dtype)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_family_is_bitwise_the_oracle(grid, dtype, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = _x(rng, N, dtype)
    X = _x(rng, (3, N), dtype)
    pairs = [
        (T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        (T.forward(np.fft.rfft(x), Nf, Nt, dt), oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        (T.forward_time_many(X, Nf, Nt, dt), oracle.forward_time_many(X, Nf, Nt, dt)),
        (T.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt),
         oracle.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_family_is_bitwise_the_oracle(grid, dtype, rng):
    Nf, Nt, dt = grid
    w = _x(rng, (Nf + 1, Nt), dtype)          # junk in inactive pixels on purpose
    W = _x(rng, (3, Nf + 1, Nt), dtype)
    pairs = [
        (T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        (T.inverse_many(W, Nf, Nt, dt), oracle.inverse_many(W, Nf, Nt, dt)),
        (T.inverse_time_many(W, Nf, Nt, dt), oracle.inverse_time_many(W, Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref)


def test_inputs_are_not_mutated(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = rng.standard_normal((Nf + 1, Nt))
    w0 = w.copy()
    T.inverse(w, Nf, Nt, dt)
    assert np.array_equal(w, w0)


@pytest.mark.parametrize("grid", [(4, 8, 1.0), (8, 16, 0.5), (16, 32, 2.0**-10), (64, 256, 2.0**-10)])
def test_jax_matches_numpy(grid, jax_be, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    X = rng.standard_normal((3, N))
    W_np = T.forward_time_many(X, Nf, Nt, dt)
    W_jx = np.asarray(T.forward_time_many(jax_be.asarray(X), Nf, Nt, dt))
    assert np.max(np.abs(W_jx - W_np)) <= 1e-13 * np.max(np.abs(W_np))
    w_np = T.forward_time(X[0], Nf, Nt, dt)
    w_jx = np.asarray(T.forward_time(jax_be.asarray(X[0]), Nf, Nt, dt))
    assert np.max(np.abs(w_jx - w_np)) <= 1e-13 * np.max(np.abs(w_np))
    x_np = T.inverse_time_many(W_np, Nf, Nt, dt)
    x_jx = np.asarray(T.inverse_time_many(jax_be.asarray(W_np), Nf, Nt, dt))
    assert np.max(np.abs(x_jx - x_np)) <= 1e-13 * np.max(np.abs(x_np))
    r_np = T.inverse(W_np[0], Nf, Nt, dt)
    r_jx = np.asarray(T.inverse(jax_be.asarray(W_np[0]), Nf, Nt, dt))
    assert np.max(np.abs(r_jx - r_np)) <= 1e-13 * np.max(np.abs(r_np))


def test_jax_jit_and_vmap(jax_be, rng):
    import jax

    Nf, Nt, dt = 8, 16, 0.5
    X = rng.standard_normal((5, Nf * Nt))
    fwd = jax.jit(lambda x: T.forward_time(x, Nf, Nt, dt))
    batched = jax.vmap(fwd)(jax_be.asarray(X))
    ref = T.forward_time_many(X, Nf, Nt, dt)
    assert np.max(np.abs(np.asarray(batched) - ref)) <= 1e-13 * np.max(np.abs(ref))
    inv = jax.jit(lambda w: T.inverse_time(w, Nf, Nt, dt))
    back = jax.vmap(inv)(batched)
    assert np.max(np.abs(np.asarray(back) - X)) <= 1e-12


def test_jax_rejects_nonfinite_concrete_input(jax_be):
    x = np.ones(128)
    x[5] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        T.forward_time(jax_be.asarray(x), 8, 16, 1.0)
