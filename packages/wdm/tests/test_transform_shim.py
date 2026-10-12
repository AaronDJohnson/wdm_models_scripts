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


def _on_device(out):
    """``out`` as a host array, after checking it really came back as a ``jax.Array``
    (an eager NumPy fallback would otherwise pass every parity check)."""
    import jax

    assert isinstance(out, jax.Array), type(out)
    return np.asarray(out)


@pytest.mark.parametrize("grid", [(4, 8, 1.0), (8, 16, 0.5), (16, 32, 2.0**-10), (64, 256, 2.0**-10)])
def test_jax_matches_numpy(grid, jax_be, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    X = rng.standard_normal((3, N))
    W_np = T.forward_time_many(X, Nf, Nt, dt)
    W_jx = _on_device(T.forward_time_many(jax_be.asarray(X), Nf, Nt, dt))
    assert np.max(np.abs(W_jx - W_np)) <= 1e-13 * np.max(np.abs(W_np))
    w_np = T.forward_time(X[0], Nf, Nt, dt)
    w_jx = _on_device(T.forward_time(jax_be.asarray(X[0]), Nf, Nt, dt))
    assert np.max(np.abs(w_jx - w_np)) <= 1e-13 * np.max(np.abs(w_np))
    x_np = T.inverse_time_many(W_np, Nf, Nt, dt)
    x_jx = _on_device(T.inverse_time_many(jax_be.asarray(W_np), Nf, Nt, dt))
    assert np.max(np.abs(x_jx - x_np)) <= 1e-13 * np.max(np.abs(x_np))
    r_np = T.inverse(W_np[0], Nf, Nt, dt)
    r_jx = _on_device(T.inverse(jax_be.asarray(W_np[0]), Nf, Nt, dt))
    assert np.max(np.abs(r_jx - r_np)) <= 1e-13 * np.max(np.abs(r_np))


def test_jax_jit_and_vmap(jax_be, rng):
    import jax

    Nf, Nt, dt = 8, 16, 0.5
    X = rng.standard_normal((5, Nf * Nt))
    fwd = jax.jit(lambda x: T.forward_time(x, Nf, Nt, dt))
    batched = jax.vmap(fwd)(jax_be.asarray(X))
    ref = T.forward_time_many(X, Nf, Nt, dt)
    assert np.max(np.abs(_on_device(batched) - ref)) <= 1e-13 * np.max(np.abs(ref))
    inv = jax.jit(lambda w: T.inverse_time(w, Nf, Nt, dt))
    back = jax.vmap(inv)(batched)
    assert np.max(np.abs(_on_device(back) - X)) <= 1e-12


def test_jax_rejects_nonfinite_concrete_input(jax_be):
    x = np.ones(128)
    x[5] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        T.forward_time(jax_be.asarray(x), 8, 16, 1.0)


# Single precision on JAX: every transform keeps float32 / complex64 end to end
# and emits no FutureWarning ("scatter inputs have incompatible types: cannot
# safely cast value from dtype=float64 to dtype=float32", an error in a future
# JAX).  Run in a fresh interpreter: JAX warns at trace time, and a scatter of
# the same shapes and dtypes cached earlier in this session would hide it.
_F32_NO_WARNINGS = """
import warnings
import numpy as np, jax, jax.numpy as jnp
from wdm.backend import enable_x64
enable_x64()
import wdm.transform as T
warnings.simplefilter("error", FutureWarning)
for Nf, Nt, dt in GRIDS:
    rng = np.random.default_rng(Nf * Nt)
    X = rng.standard_normal((3, Nf * Nt))
    X32 = jnp.asarray(X.astype(np.float32))
    W32 = jnp.asarray(T.forward_time_many(X, Nf, Nt, dt).astype(np.float32))
    calls = {
        "forward_time": (lambda: T.forward_time(X32[0], Nf, Nt, dt), np.float32),
        "forward_time_many": (lambda: T.forward_time_many(X32, Nf, Nt, dt), np.float32),
        "forward": (lambda: T.forward(jnp.fft.rfft(X32[0]), Nf, Nt, dt), np.float32),
        "forward_many": (lambda: T.forward_many(jnp.fft.rfft(X32, axis=-1), Nf, Nt, dt),
                         np.float32),
        "inverse": (lambda: T.inverse(W32[0], Nf, Nt, dt), np.complex64),
        "inverse_many": (lambda: T.inverse_many(W32, Nf, Nt, dt), np.complex64),
        "inverse_time": (lambda: T.inverse_time(W32[0], Nf, Nt, dt), np.float32),
        "inverse_time_many": (lambda: T.inverse_time_many(W32, Nf, Nt, dt), np.float32),
        "jit forward_time": (lambda: jax.jit(lambda x: T.forward_time(x, Nf, Nt, dt))(X32[0]),
                             np.float32),
        "jit inverse_time": (lambda: jax.jit(lambda w: T.inverse_time(w, Nf, Nt, dt))(W32[0]),
                             np.float32),
    }
    for name, (call, dtype) in calls.items():
        try:
            out = call()
        except FutureWarning as warn:
            print("WARNED", (Nf, Nt, dt), name, str(warn)[:120])
            continue
        if not isinstance(out, jax.Array) or out.dtype != dtype:
            print("DTYPE", (Nf, Nt, dt), name, type(out).__name__, out.dtype)
print("done")
"""


def test_jax_float32_keeps_single_precision_without_future_warnings(jax_be):
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(src),
               JAX_ENABLE_X64="true")
    # the smallest grid, a non-power-of-two one and the sinusoid grid: the warning
    # depends on dtypes, not shapes, and each grid costs ~2 s of eager JAX compiles
    grids = [(2, 2, 1.0), (6, 10, 0.3), (64, 256, 2.0**-10)]
    code = f"GRIDS = {grids!r}\n" + _F32_NO_WARNINGS
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["done"], out.stdout
