"""D-A7: the linear algebra on JAX -- jit/vmap with the constants passed as arguments,
parity with NumPy, the eager-only oracle modes refusing to trace, x64 enforced."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import eq45_blocks_cheb, eq45_coeffs_bessel, eq45_plan, exact_covariance
from domainia import linalg as la
from domainia import testing as T
from domainia.domains import CellSet

pytestmark = pytest.mark.jax

NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
GRID = wdm.Grid(NF, NT, DT)
TESTS = Path(__file__).resolve().parent


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def eq45_blocks(cells, ac, ad):
    plan = eq45_plan(NF, NT, DT, cells.keep_mask(), P=24)
    Lf = L_of(plan.f)
    return eq45_blocks_cheb(plan, lambda k: eq45_coeffs_bessel(Lf[k], ac, ad, 24, r=1.0))


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


def test_device_constants_need_x64():
    pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    code = ("import numpy as np\n"
            "from domainia import linalg as la\n"
            "la.device_constants(la.BandLayout.from_sizes([2, 3]))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(PYTHONDONTWRITEBYTECODE="1", JAX_ENABLE_X64="0",
               PYTHONPATH=os.pathsep.join(str(p) for p in (TESTS.parent / "src",
                                                           TESTS.parent.parent / "wdm" / "src")))
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(TESTS),
                         capture_output=True, text=True)
    assert out.returncode != 0 and "float64 under JAX" in out.stderr


def test_padded_factor_jits_and_vmaps_with_the_layout_as_an_argument(jax_be, rng):
    import jax
    import jax.numpy as jnp

    cells = CellSet.all_active(GRID).drop_columns(3, 1)
    lay = la.BandLayout.from_cells(cells)
    sets = [eq45_blocks(cells, ac, ad) for ac, ad in ((-2.0, 1.2), (-1.7, 0.9), (-0.5, -1.0))]
    packed = [lay.pack_blocks(d, u, be=NUMPY) for d, u in sets]
    D = jnp.asarray(np.stack([p[0] for p in packed]))
    U = jnp.asarray(np.stack([p[1] for p in packed]))
    b = rng.standard_normal(cells.size)

    def logdet_quad(D, U, lay, b):
        f = la.blocktri_factor(D, U, layout="padded", band_layout=lay)
        w = la.blocktri_whiten(f, b)
        return f.logdet, w @ w, f.ok

    layd = la.device_constants(lay)
    assert isinstance(layd.src, jax.Array)
    ld, q, ok = jax.jit(jax.vmap(logdet_quad, in_axes=(0, 0, None, None)))(D, U, layd,
                                                                         jnp.asarray(b))
    assert isinstance(ld, jax.Array) and bool(jnp.all(ok))
    for r, (d, u) in enumerate(sets):
        f = la.blocktri_factor(d, u, layout="ragged")
        assert abs(float(ld[r]) - f.logdet) <= 1e-12 * abs(f.logdet)
        want = b @ la.blocktri_solve(f, b)
        assert abs(float(q[r]) - want) <= 1e-12 * abs(want)


def test_constants_passed_as_arguments_are_not_compiled_in(jax_be):
    import jax
    import jax.numpy as jnp

    lay = la.device_constants(la.BandLayout.from_sizes([128] + [256] * 63 + [128]))
    b = jnp.ones(lay.n)
    as_arg = jax.jit(lambda lay, b: lay.unpack(lay.pack(b[:, None]))).lower(lay, b).as_text()
    closed = jax.jit(lambda b: lay.unpack(lay.pack(b[:, None]))).lower(b).as_text()
    assert len(as_arg) < 20_000 < len(closed)


def test_padded_solve_matches_ragged_on_jax(jax_be, rng):
    import jax
    import jax.numpy as jnp

    cells = CellSet.all_active(GRID)
    diag, up = eq45_blocks(cells, -2.0, 1.2)
    lay = la.BandLayout.from_cells(cells)
    D, U = lay.pack_blocks(diag, up, be=NUMPY)
    b = rng.standard_normal((cells.size, 3))
    solve = jax.jit(lambda D, U, lay, b: la.blocktri_solve(
        la.blocktri_factor(D, U, layout="padded", band_layout=lay), b))
    got = solve(jnp.asarray(D), jnp.asarray(U), la.device_constants(lay), jnp.asarray(b))
    assert isinstance(got, jax.Array)
    assert rel(got, la.blocktri_solve(la.blocktri_factor(diag, up, layout="ragged"), b)) <= 1e-12


def test_ragged_runs_eagerly_on_jax_and_refuses_to_trace(jax_be, rng):
    import jax
    import jax.numpy as jnp

    cells = CellSet.all_active(GRID)
    diag, up = eq45_blocks(cells, -2.0, 1.2)
    fj = la.blocktri_factor([jnp.asarray(d) for d in diag], [jnp.asarray(u) for u in up],
                            layout="ragged")
    fn = la.blocktri_factor(diag, up, layout="ragged")
    assert abs(fj.logdet - fn.logdet) <= 1e-13 * abs(fn.logdet)
    b = rng.standard_normal(cells.size)
    assert rel(la.blocktri_solve(fj, jnp.asarray(b)), la.blocktri_solve(fn, b)) <= 1e-13
    with pytest.raises(ValueError, match="eagerly only"):
        jax.jit(lambda d0: la.blocktri_factor([d0] + diag[1:], up, layout="ragged"))(
            jnp.asarray(diag[0]))


def test_dense_cblock_schur_and_padded_factors_on_jax(jax_be, rng):
    import jax
    import jax.numpy as jnp

    # dense real and complex
    A = rng.standard_normal((20, 20))
    C = A @ A.T + 20 * np.eye(20)
    r = rng.standard_normal(20)
    q = jax.jit(lambda C, r: (lambda f: (f.logdet, la.chol_whiten(f, r)))(
        la.chol_factor(C, impl="numpy", symmetrize=True)))(jnp.asarray(C), jnp.asarray(r))
    fn = la.chol_factor(C, impl="numpy", symmetrize=True)
    assert abs(float(q[0]) - float(fn.logdet)) < 1e-12 * abs(float(fn.logdet))
    assert rel(q[1], la.chol_whiten(fn, r)) < 1e-12
    b, nb = 6, 4
    Z = rng.standard_normal((b * nb, b * nb)) + 1j * rng.standard_normal((b * nb, b * nb))
    M = Z @ Z.conj().T / (b * nb) + 2 * np.eye(b * nb)
    z = rng.standard_normal(b * nb) + 1j * rng.standard_normal(b * nb)
    out = jax.jit(lambda M, z: (lambda f: (f.logdet, la.cblock_backward(f, la.cblock_forward(f, z))))(
        la.cblock_factor(M, b)))(jnp.asarray(M), jnp.asarray(z))
    fc = la.cblock_factor(M, b)
    assert abs(float(out[0]) - float(fc.logdet)) < 1e-12 * abs(float(fc.logdet))
    assert rel(out[1], la.factor_solve(fc, z)) < 1e-12
    # Schur over a diagonal interior (the GB band/mid_exact JAX default)
    cells = CellSet.all_active(GRID)
    edge = cells.columns(3, 1)
    e, i = cells.index_of(edge), cells.index_of(cells.complement(edge))
    ex = exact_covariance(NF, NT, DT, lambda f, t: np.exp((-1.7 + 0.9 * (t / 8.0 - 0.5)) * L_of(f)))
    R = ex[e]
    v = np.diag(ex)[i]
    y = rng.standard_normal(N)

    def schur(v, Ree, Rei, e, i, y):
        f = la.schur_factor(la.diag_factor(v), Ree, Rei, e=e, i=i, form="scaled", impl="numpy")
        return f.logdet, la.schur_solve(f, y)

    got = jax.jit(schur)(*(jnp.asarray(a) for a in (v, R[:, e], R[:, i], e, i, y)))
    want = schur(v, R[:, e], R[:, i], e, i, y)
    assert abs(float(got[0]) - float(want[0])) < 1e-12 * abs(float(want[0]))
    assert rel(got[1], want[1]) < 1e-12
    # padded marginal
    Np = NF * (NT + 2)
    ext = CellSet.all_active(wdm.Grid(NF, NT + 2, DT))
    W = T.dense_W(ext)
    vext = rng.uniform(0.5, 2.0, Np)
    w = rng.standard_normal(Np)

    def pad(vext, Wp, w):
        f = la.padded_factor(la.diag_factor(vext), Wp, n_ext=Np, dt=DT, impl="numpy")
        return f.logdet, la.padded_solve(f, w)

    gp = jax.jit(pad)(jnp.asarray(vext), jnp.asarray(W[:, N:]), jnp.asarray(w))
    wp = pad(vext, W[:, N:], w)
    assert abs(float(gp[0]) - float(wp[0])) < 1e-12 * abs(float(wp[0]))
    assert rel(gp[1], wp[1]) < 1e-12


def test_pcg_on_jax(jax_be):
    import jax
    import jax.numpy as jnp

    n = 128
    rng = np.random.default_rng(7)
    d = np.exp(0.3 * rng.standard_normal(n))
    s = np.exp(1.5 * np.cos(np.linspace(0, np.pi, n // 2 + 1)))

    def make(xp, fft):
        def A(V):
            return d * fft.irfft(fft.rfft(d * V, axis=-1) * s, n=n, axis=-1)

        def M(V):
            return fft.irfft(fft.rfft(V, axis=-1) / s, n=n, axis=-1)
        return A, M

    B = rng.standard_normal((3, n))
    An, Mn = make(np, np.fft)
    ref = la.pcg(An, Mn, B, tol=1e-10, maxit=300)
    Aj, Mj = make(jnp, jnp.fft)
    res = jax.jit(lambda B: la.pcg(Aj, Mj, B, tol=1e-10, maxit=300, on_fail="nan"))(jnp.asarray(B))
    assert isinstance(res.x, jax.Array) and bool(jnp.all(res.converged))
    # two runs to tol 1e-10 with different FFT rounding (numpy.fft vs XLA): measured 4.6e-11
    assert rel(res.x, ref.x) < 1e-9
    bad = jax.jit(lambda B: la.pcg(Aj, Mj, B, tol=1e-10, maxit=2, on_fail="nan"))(jnp.asarray(B))
    assert not bool(jnp.any(bad.converged)) and bool(jnp.all(jnp.isnan(bad.x)))
    with pytest.raises(ValueError, match="eagerly only"):
        jax.jit(lambda B: la.pcg(Aj, Mj, B, tol=1e-10, maxit=300).x)(jnp.asarray(B))
