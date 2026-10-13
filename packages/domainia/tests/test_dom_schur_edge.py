"""D-C8: SchurEdge (edge-exact models) and ``cut``.

* Small grid: SchurEdge is the dense Gaussian of the exact covariance with the interior
  block replaced by the approximation (WT test_matrix.py), for a diagonal and an Eq. (45)
  interior; ``schur`` and ``dense`` assemblies agree; solve and whiten orders agree.
* A PD interior with a non-PD Schur complement gives ok = False (the -inf truncation).
* G6 (GB 3 and 6 months): band/mid/tri_exact per-channel (quad, logdet) at the anchor,
  dense assembly (golden) and Schur (the JAX default) within the G3 gate.
* G6 (sinusoid, ``slow``: 959 exact rows, ~3 s each): band/mid/tri_exact parts.
* JAX: fixed-order rows and the staged adaptive rows, jitted rest.
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import exact_covariance

import domtest_inputs as T
from domainia.covariance import (BlockTridiagonalCovariance, DiagonalCovariance, Eq45Covariance,
                                 ExactFourierCovariance, ExactTimeCovariance, PushedRows, SchurEdge, SchurFactor,
                                 cut, keep_mask, positions)
from domainia.domains import CellSet
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.spectra import LogLinearPSD

LOG_2PI = float(np.log(2.0 * np.pi))
NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
CELLS = CellSet.all_active(wdm.Grid(NF, NT, DT))
EDGE = CELLS.columns(3, 1)
INTERIOR = CELLS.complement(EDGE)


def _rows(**kw) -> PushedRows:
    return PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS, **kw)


def _interior(kind: str):
    if kind == "tri":
        return Eq45Covariance(PSD, INTERIOR, builder="bessel")
    v = np.exp(np.linspace(-1.0, 1.0, len(INTERIOR.m)))
    return DiagonalCovariance(v, v.size)


@pytest.mark.parametrize("kind", ["diag", "tri"])
def test_schur_edge_is_the_dense_hybrid(rng, kind):
    """NumPy (the jitted JAX path is ``test_jax_schur_edge``; eager JAX rows cost ~2.5 s here)."""
    backend = NUMPY
    interior = _interior(kind)
    exact = exact_covariance(NF, NT, DT, lambda f, t: PSD(f, t, NU))
    e, i = positions(EDGE, CELLS), positions(INTERIOR, CELLS)
    C = exact.copy()
    C[np.ix_(i, i)] = interior.dense(NU)
    y = rng.standard_normal(N)
    want_q, want_ld = y @ np.linalg.solve(C, y), np.linalg.slogdet(C)[1]
    for assembly, impl in (("schur", "scipy"), ("dense", "numpy")):
        se = SchurEdge(interior, _rows(), EDGE, assembly=assembly, chol_impl=impl)
        F = se.factor(backend.asarray(NU), backend, jit=False)
        assert bool(F.ok)
        assert float(F.logdet) == pytest.approx(want_ld, rel=1e-10)
        for form in ("solve", "whiten"):
            assert float(F.quad(backend.asarray(y), form=form)) == pytest.approx(want_q, rel=1e-9), (assembly, form)
    F = SchurEdge(interior, _rows(), EDGE).factor(NU)
    assert isinstance(F, SchurFactor)
    B = rng.standard_normal((N, 2))
    assert np.allclose(F.solve(B), np.linalg.solve(0.5 * (C + C.T), B), rtol=1e-8, atol=1e-10)


def test_non_pd_schur_complement_is_minus_infinity(backend):
    """A tiny interior variance makes S = A - B T^-1 B^T indefinite while T is PD."""
    interior = DiagonalCovariance(np.full(len(INTERIOR.m), 1e-9), len(INTERIOR.m))
    F = SchurEdge(interior, _rows(), EDGE).factor(backend.asarray(NU), backend)
    assert bool(F.T.ok) and not bool(F.S.ok) and not bool(F.ok)
    F = SchurEdge(interior, _rows(), EDGE, assembly="dense", chol_impl="numpy").factor(backend.asarray(NU), backend)
    assert not bool(F.ok)
    # a failed (ragged) interior: WT's LinAlgError before the complement -> ok False, no exception
    sizes = tuple(c for _, c in INTERIOR.band_sizes())
    bad = BlockTridiagonalCovariance(lambda nu, be: ([be.asarray(-np.eye(s)) for s in sizes],
                                                     [be.asarray(np.zeros((a, b))) for a, b in zip(sizes, sizes[1:])]),
                                     sizes)
    F = SchurEdge(bad, _rows(), EDGE).factor(backend.asarray(NU), backend)
    assert not bool(F.ok) and F.S is None
    with pytest.raises(ValueError, match="failed"):
        F.quad(backend.asarray(np.ones(N)))


def test_arguments_are_checked():
    with pytest.raises(ValueError, match="interior has"):
        SchurEdge(DiagonalCovariance(np.ones(3), 3), _rows(), EDGE)
    with pytest.raises(ValueError, match="rows' cells"):
        SchurEdge(_interior("diag"), _rows(), CELLS.columns(2, 1))
    with pytest.raises(ValueError, match="assembly"):
        SchurEdge(_interior("diag"), _rows(), EDGE, assembly="blocktri")
    with pytest.raises(ValueError, match="chol_impl"):
        SchurEdge(_interior("diag"), _rows(), EDGE, chol_impl="lapack")


def test_cut_rebuilds_on_the_kept_cells():
    cov = cut(lambda cs: Eq45Covariance(PSD, cs, builder="bessel"), CELLS, 3, 1)
    assert cov.n == len(INTERIOR.m) == 96
    assert np.array_equal(keep_mask(cov.cells), keep_mask(INTERIOR))


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo"])
@pytest.mark.parametrize("base", ["band", "mid", "tri"])
def test_g6_gb_edge_exact(case, base):
    c = T.gb_case(case)
    edge = c.cells.columns(5, 4)
    interior_cells = c.cells.complement(edge)
    keep = np.isin(np.arange(len(c.cells.m)), positions(interior_cells, c.cells))
    r = c.residual_cells(c.cells)
    U = c.atoms(c.cells, c.window.k0, c.window.n)
    got, schur = [], []
    for ch in range(2):
        if base == "tri":
            interior = Eq45Covariance(c.psd, interior_cells, builder="separable", assembly="dense", channel=ch)
        else:
            v = c.A[f"var_{base}_raw"][ch][keep]
            interior = DiagonalCovariance(v, v.size)
        rows = PushedRows(ExactFourierCovariance(c.psd, c.window, channel=ch), None, edge, cells=c.cells, atoms=U)
        F = SchurEdge(interior, rows, edge, assembly="dense", chol_impl="numpy").factor(c.mu[ch])
        got.append([float(F.quad(r[ch], form="whiten")), float(F.logdet)])
        Fs = SchurEdge(interior, rows, edge, assembly="schur", chol_impl="numpy").factor(c.mu[ch])
        schur.append([float(Fs.quad(r[ch], form="whiten")), float(Fs.logdet)])
    want = c.quad_logdet(f"{base}_exact")
    assert np.allclose(np.array(got), want, rtol=1e-12, atol=0)
    assert np.max(np.abs(np.array(schur).sum(axis=1) - want.sum(axis=1))) <= 2e-8   # G3 gate, both channels


@pytest.mark.slow
@pytest.mark.parametrize("base", ["band", "mid", "tri"])
def test_g6_sinusoid_edge_exact_parts(base):
    s = T.sinusoid_mid()
    ref = s["g6"]["parts"][f"{base}_exact"]
    grid = wdm.Grid(T.SIN_NF, T.SIN_NT, T.SIN_DT)
    cells = CellSet.all_active(grid)
    edge = cells.columns(8, 7)
    interior_cells = cells.complement(edge)
    psd = LogLinearPSD(f_floor=4.0, f_ref=1.0, T=T.SIN_T)
    if base == "tri":
        interior = Eq45Covariance(psd, interior_cells, builder="bessel")
    else:
        v = s["arrays"][f"{base}_variance_raw"][keep_mask(interior_cells)]
        interior = DiagonalCovariance(v, v.size)
    rows = PushedRows(ExactTimeCovariance(psd, T.SIN_N, T.SIN_DT, P=24), None, edge, cells=cells)
    nb = FourierBandNuisance(T.SIN_N, T.SIN_DT, T.SIN_FMIN_CUT)
    F = FlatMarginal(SchurEdge(interior, rows, edge), nb.push_to_cells(cells)).factor(T.SIN_NU)
    p = F.parts(T.sinusoid_cells_residual())
    for key in ("quad_C", "quad_H", "logdet_C", "logdet_H"):
        assert float(p[key]) == pytest.approx(T.num(ref[key]), rel=1e-12, abs=0), key


def test_state_key_is_the_adaptive_order_and_n_u_is_static():
    """What ``GaussianLikelihood`` reads before factorising (section D-D): ``state_key`` is
    ``(P, None)`` for adaptive rows (``P = cheb_order``), ``None`` for fixed rows, and
    ``FlatMarginal`` forwards it; ``n_u`` is static and equals the factor's."""
    from domainia.covariance import cheb_order

    fixed = SchurEdge(_interior("diag"), _rows(), EDGE)
    adaptive = SchurEdge(_interior("diag"), _rows(order="adaptive"), EDGE)
    P = cheb_order(NU[1], adaptive.rows.L_max, adaptive.rows.P_max)
    assert fixed.state_key(NU) is None and adaptive.state_key(NU) == (P, None) and P < adaptive.rows.P_max
    U = np.linalg.qr(np.random.default_rng(3).standard_normal((N, 2)))[0]
    fm = FlatMarginal(adaptive, U)
    assert fm.state_key(NU) == ((P, None),) and FlatMarginal(fixed, U).state_key(NU) is None
    assert (fm.n_u, fm.factor(NU).n_u, fixed.n_u) == (2, 2, 0)


@pytest.mark.jax
def test_jax_flat_marginal_over_adaptive_rows_runs_eagerly(jax_be):
    """A combinator over adaptive rows has a static key, so ``factor`` runs it eagerly on JAX
    (the order needs a concrete ``nu``) instead of failing inside ``jit``."""
    import jax.numpy as jnp

    U = np.linalg.qr(np.random.default_rng(3).standard_normal((N, 2)))[0]
    fm = FlatMarginal(SchurEdge(_interior("diag"), _rows(order="adaptive"), EDGE), U)
    Fn, Fj = fm.factor(NU), fm.factor(jnp.asarray(NU))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)


@pytest.mark.jax
@pytest.mark.parametrize("kw", [{}, {"order": "adaptive", "phi_cache": True, "staged": True, "chunk": 8}])
def test_jax_schur_edge(jax_be, rng, kw):
    import jax.numpy as jnp

    se = SchurEdge(Eq45Covariance(PSD, INTERIOR, builder="gram", layout="padded"), _rows(**kw), EDGE)
    assert se.jittable
    y = rng.standard_normal(N)
    Fn, Fj = se.factor(NU), se.factor(jnp.asarray(NU))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
    assert float(Fj.quad(jnp.asarray(y))) == pytest.approx(float(Fn.quad(y)), rel=1e-11)


#: The second envelope state of the GB edge-rows test: mu_fit plus a stronger, asymmetric
#: modulation (Im M' is 10-37% of max |M'| over the two cases and two states, so a
#: transposed or conjugated M' and conjugated atoms change the cell covariance at O(1)).
GB_MU_SHIFT = np.array([0.3, -0.4, 0.25, 0.2, -0.15])


def _gb_exact_cells_oracle(c: T.GBCase, mu: np.ndarray, ch: int) -> np.ndarray:
    r"""The exact cell covariance :math:`\tfrac12\mathrm{Re}(U M' U^H)` on the analysis window,
    built without ``ExactFourierCovariance``, ``FourierToWDM`` or ``PushedRows``.

    * ``U``: the unit-norm band atoms placed on the window bins by the test-local
      ``GBCase.atoms`` (``wdm.transform.band_atoms``, as ``test_dom_hermitian.py``).
    * ``M'[j, k] = s_I(j) delta_jk + sqrt(s_G(j) s_G(k)) ghat[j - k]`` (``= conj(ghat)[k - j]``,
      ``g`` is real): the modulated process ``sqrt(g) n_G`` couples bins ``j``, ``k`` through
      the ``(j - k)``-th Fourier coefficient of ``g = exp(mu @ basis_L)``.  ``s_I``, ``s_G``
      are the prepared store tables sliced to the window; ``ghat`` is
      ``domainia.testing.hhat`` at ``2 mu`` (``hhat`` transforms ``sqrt(g)``:
      ``exp(0.5 (2 mu) @ B) = exp(mu @ B)`` exactly).
    """
    from domainia.testing import hhat

    w = c.window
    lo = w.k0 - c.store_k0
    sI = np.asarray(c.P["store_sI"], float)[ch, lo:lo + w.n]
    sG = np.asarray(c.P["store_sG"], float)[ch, lo:lo + w.n]
    j = np.arange(w.n)
    gh = hhat(2.0 * np.asarray(mu, float), np.asarray(c.P["basis_L"], float), j[:, None] - j[None, :])
    M = np.diag(sI).astype(complex) + np.sqrt(sG)[:, None] * np.sqrt(sG)[None, :] * gh
    U = c.atoms(c.cells, w.k0, w.n)
    return 0.5 * np.real(U @ M @ np.conj(U).T)


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo"])
def test_gb_edge_rows_are_the_exact_covariance(case):
    """Ported GB test_matrix.py::test_edge_exact_rows_are_the_exact_covariance with an
    independent oracle (:func:`_gb_exact_cells_oracle`; the legacy test used
    ``wdm_full.dense_lambda``): the 27 edge rows of the assembled mid/tri_exact covariance
    (rows pushed through D-B's ``FourierToWDM(window, cells, restrict=True)``) are the exact
    cell covariance's rows (1e-12 of max), the matrix is symmetric (1e-14), and with every
    cell a row the pushed rows are the exact cell covariance -- at mu_fit and at a second
    state ``mu_fit + GB_MU_SHIFT``."""
    from domainia.domains import WDMDomain
    from domainia.maps import FourierToWDM

    c = T.gb_case(case)
    edge = c.cells.columns(5, 4)
    assert len(edge.m) == 3 * (5 + 4)
    n = np.asarray(c.cells.n)
    e = np.flatnonzero((n < 5) | (n >= c.NT - 4))                   # the edge rows, in cell order
    fmap = FourierToWDM(c.window, WDMDomain(c.cells), restrict=True)
    ex = ExactFourierCovariance(c.psd, c.window, channel=0)
    rows = PushedRows(ex, fmap, edge)
    assert np.array_equal(rows.e, e)
    interior_cells = c.cells.complement(edge)
    v = np.delete(c.A["var_mid_raw"][0], e)        # mu_fit's variances: only the edge rows are compared
    full = PushedRows(ex, fmap, c.cells)
    for mu in (c.mu[0], c.mu[0] + GB_MU_SHIFT):
        exact = _gb_exact_cells_oracle(c, mu, 0)
        tol = 1e-12 * np.max(np.abs(exact))
        for interior in (DiagonalCovariance(v, v.size),
                         Eq45Covariance(c.psd, interior_cells, builder="separable", assembly="dense", channel=0)):
            C = SchurEdge(interior, rows, edge, assembly="dense", chol_impl="numpy").dense(mu)
            assert np.max(np.abs(C[e] - exact[e])) < tol
            assert np.max(np.abs(C - C.T)) < 1e-14 * np.max(np.abs(C))
        R = np.asarray(full.evaluate(full.host_constants(), mu, NUMPY))
        assert np.max(np.abs(R - exact)) < tol
