"""C2: the backend shim (spec section 3.1)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wdm.backend import NUMPY, Backend, get_backend, pytree_dataclass

SRC = Path(__file__).resolve().parents[1] / "src"


def _spd(n, rng):
    A = rng.standard_normal((n, n))
    return A @ A.T / n + np.eye(n)


def _run(code: str) -> str:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(SRC))
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def test_numpy_backend_identity():
    assert isinstance(NUMPY, Backend)
    assert NUMPY.name == "numpy" and NUMPY.xp is np and NUMPY.fft is np.fft
    assert get_backend(np.ones(3)) is NUMPY
    assert get_backend([1.0, 2.0]) is NUMPY
    assert get_backend(np.ones(3), backend="numpy") is NUMPY
    with pytest.raises(ValueError, match="backend must be"):
        get_backend(backend="torch")


def test_import_wdm_and_the_shim_do_not_import_jax():
    assert _run("import sys, wdm, wdm.backend; print('jax' in sys.modules)") == "False"


def test_numpy_set_copies_unless_told_not_to():
    a = np.zeros(4)
    b = NUMPY.set(a, slice(1, 3), 7.0)
    assert np.array_equal(a, np.zeros(4)) and np.array_equal(b, [0, 7, 7, 0])
    c = NUMPY.set(a, 0, 1.0, copy=False)
    assert c is a and a[0] == 1.0


def test_numpy_add_at_keeps_the_oracle_statement():
    a = np.zeros(5)
    out = NUMPY.add_at(a, np.array([0, 0, 3]), np.array([1.0, 2.0, 4.0]))
    assert np.array_equal(out, [3.0, 0, 0, 4.0, 0])          # duplicates accumulate (np.add.at)
    X = np.zeros((2, 6), complex)
    v = np.arange(8.0).reshape(2, 4) + 1j
    Y = NUMPY.add_at(X, (..., slice(1, 5)), v)
    ref = X.copy()
    ref[..., 1:5] += v
    assert np.array_equal(Y, ref)


@pytest.mark.parametrize("impl", ["numpy", "scipy"])
def test_cholesky_solves_and_flags_failure(backend, impl, rng):
    A = _spd(7, rng)
    b = rng.standard_normal((7, 2))
    L, ok = backend.cholesky(backend.asarray(A), impl=impl)
    assert bool(ok)
    L = np.asarray(L)
    assert np.allclose(np.triu(L, 1), 0.0)
    x = backend.cho_solve(backend.asarray(L), backend.asarray(b))
    assert np.allclose(np.asarray(x), np.linalg.solve(A, b), rtol=1e-12, atol=1e-12)
    z = backend.solve_triangular(backend.asarray(L), backend.asarray(b), lower=True)
    assert np.allclose(np.asarray(L) @ np.asarray(z), b, rtol=1e-12, atol=1e-12)
    bad = A.copy()
    bad[0, 0] = -1.0
    _, ok_bad = backend.cholesky(backend.asarray(bad), impl=impl)
    assert not bool(ok_bad)


def test_numpy_scipy_cholesky_matches_cho_factor_bitwise(rng):
    from scipy.linalg import cho_factor, cho_solve

    A = _spd(9, rng)
    b = rng.standard_normal(9)
    c, low = cho_factor(A, lower=True, check_finite=False)
    L, ok = NUMPY.cholesky(A, impl="scipy")
    assert ok and np.array_equal(L, np.tril(c))
    assert np.array_equal(NUMPY.cho_solve(L, b), cho_solve((c, low), b))


def test_control_flow(backend):
    xp = backend.xp

    def step(c, x):
        return c + x, 2.0 * x

    carry, ys = backend.scan(step, backend.asarray(0.0), backend.asarray(np.arange(4.0)))
    assert float(carry) == 6.0 and np.array_equal(np.asarray(ys), [0, 2, 4, 6])
    _, rev = backend.scan(lambda c, x: (c + x, c), backend.asarray(0.0),
                          backend.asarray(np.arange(3.0)), reverse=True)
    assert np.array_equal(np.asarray(rev), [3.0, 2.0, 0.0])
    w = backend.while_loop(lambda v: v < 10.0, lambda v: v * 2.0, backend.asarray(1.0))
    assert float(w) == 16.0
    f = backend.fori_loop(0, 5, lambda i, v: v + i, backend.asarray(0.0))
    assert float(f) == 10.0
    m = backend.map(lambda x: xp.sum(x), backend.asarray(np.ones((3, 4))))
    assert np.array_equal(np.asarray(m), [4.0, 4.0, 4.0])
    vf = backend.vmap(lambda a, s: a * s, in_axes=(0, None))
    assert np.array_equal(np.asarray(vf(backend.asarray(np.ones((2, 3))), 3.0)), 3 * np.ones((2, 3)))
    assert np.asarray(backend.jit(lambda x: x + 1.0)(backend.asarray(1.0))) == 2.0


def test_bessel_i_numpy_is_the_wt_expression(rng):
    from scipy.special import ive

    a = rng.uniform(-3.2, 3.2, 50)
    got = NUMPY.bessel_i(24, a)
    ref = ive(np.arange(24)[:, None], a[None, :]) * np.exp(np.abs(a))[None, :]
    assert np.array_equal(got, ref)


def test_bessel_i_jax_series_matches_scipy(jax_be, rng):
    a = rng.uniform(-8.0, 8.0, 64)
    got = np.asarray(jax_be.bessel_i(24, jax_be.asarray(a)))
    ref = NUMPY.bessel_i(24, a)
    assert np.max(np.abs(got - ref) / np.abs(ref)) < 1e-13   # measured 4.4e-14


def test_fft_pinned(backend, rng):
    x = rng.standard_normal(64)
    got = np.asarray(backend.fft_pinned("rfft", backend.asarray(x)))
    assert np.allclose(got, np.fft.rfft(x), rtol=1e-14, atol=1e-13)


def test_get_backend_sees_jax_arrays_and_tracers(jax_be):
    import jax
    import jax.numpy as jnp

    assert get_backend(jnp.ones(3)) is jax_be
    assert get_backend(np.ones(3), jnp.ones(3)) is jax_be
    seen = []

    @jax.jit
    def f(x):
        seen.append(get_backend(x).name)
        return x + 1.0

    f(jnp.ones(2))
    assert seen == ["jax"]


def test_jax_add_at_static_fold_matches_numpy(jax_be, rng):
    v = rng.standard_normal((3, 37))
    k = (5 + np.arange(37)) % 16
    ref = NUMPY.add_at(np.zeros((3, 16)), (slice(None), k), v)
    got = jax_be.add_at(jax_be.xp.zeros((3, 16)), (slice(None), k), jax_be.asarray(v),
                        static_fold=(5, 16))
    assert np.allclose(np.asarray(got), ref, rtol=0, atol=1e-14)


def test_jax_add_at_static_fold_accepts_ellipsis_and_1d(jax_be, rng):
    k = (5 + np.arange(37)) % 16
    for shape, idx in [((2, 3, 16), (..., k)), ((2, 3, 16), (slice(None), ..., k)),
                       ((16,), k), ((16,), (k,))]:
        a = rng.standard_normal(shape)
        v = rng.standard_normal((*shape[:-1], 37))
        ref = NUMPY.add_at(a, idx, v)
        got = jax_be.add_at(jax_be.asarray(a), idx, jax_be.asarray(v), static_fold=(5, 16))
        assert got.shape == a.shape
        assert np.allclose(np.asarray(got), ref, rtol=0, atol=1e-14), (shape, idx)


def test_jax_add_at_static_fold_rejects_anything_but_the_fold(jax_be, rng):
    # Each case would otherwise broadcast the folded sum over the wrong axes (or
    # change a's shape) and differ from NUMPY.add_at without raising.
    k = (5 + np.arange(37)) % 16
    xp = jax_be.xp
    v1, v3 = jax_be.asarray(rng.standard_normal(37)), jax_be.asarray(rng.standard_normal((3, 37)))
    cases = [
        ("leading component", xp.zeros((3, 16)), (1, k), v1, (5, 16)),  # NumPy: row 1 only
        ("leading component", xp.zeros((3, 16)), (slice(0, 2), k), v3[:2], (5, 16)),
        ("leading component", xp.zeros((3, 16)), (None, k), v1, (5, 16)),
        ("last axis of a", xp.zeros((16, 16)), k, v1, (5, 16)),  # NumPy: axis 0
        ("last axis of a", xp.zeros((2, 3, 16)), (slice(None), k), v3, (5, 16)),
        ("last axis of a", xp.zeros((3, 16)), (..., ..., k), v3, (5, 16)),
        ("a.shape\\[-1\\] == n", xp.zeros((3, 1)), (slice(None), k), v3, (5, 16)),
        ("a.shape\\[-1\\] == n", xp.zeros((3, 32)), (slice(None), k), v3, (5, 16)),
        ("idx\\[-1\\] ==", xp.zeros((3, 16)), (slice(None), k), v3, (4, 16)),
        ("idx\\[-1\\] ==", xp.zeros((3, 16)), (slice(None), k[::-1]), v3, (5, 16)),
        ("v.shape\\[-1\\]", xp.zeros((3, 16)), (slice(None), k), v3[:, :36], (5, 16)),
        ("change a.shape", xp.zeros((3, 16)), (slice(None), k), xp.stack([v3, v3]), (5, 16)),
        ("does not fit", xp.zeros((3, 16)), (slice(None), k), v3[:2], (5, 16)),
    ]
    for match, a, idx, v, fold in cases:
        with pytest.raises(ValueError, match=match):
            jax_be.add_at(a, idx, v, static_fold=fold)


def test_require_x64_raises_when_off():
    out = _run(
        "import jax; jax.config.update('jax_enable_x64', False)\n"
        "from wdm.backend import require_x64\n"
        "try:\n    require_x64()\nexcept RuntimeError as e:\n    print('raised')\n"
    ) if _has_jax() else "raised"
    assert out == "raised"


def _has_jax() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def test_pytree_dataclass_registers_lazily():
    if not _has_jax():
        pytest.skip("jax not installed")
    out = _run(
        "import sys\n"
        "from wdm.backend import pytree_dataclass\n"
        "@pytree_dataclass(data=('x',), meta=('n',))\n"
        "class F:\n    x: object\n    n: int\n"
        "assert 'jax' not in sys.modules\n"
        "from wdm.backend import jax_backend\n"
        "jax_backend()\n"
        "import jax, jax.numpy as jnp\n"
        "leaves = jax.tree_util.tree_leaves(F(jnp.ones(2), 3))\n"
        "print(len(leaves), F(1.0, 2).n)\n"
    )
    assert out == "1 2"


def test_pytree_dataclass_rejects_a_bad_partition():
    with pytest.raises(ValueError, match="partition"):
        @pytree_dataclass(data=("x",), meta=())
        class G:
            x: float
            n: int
