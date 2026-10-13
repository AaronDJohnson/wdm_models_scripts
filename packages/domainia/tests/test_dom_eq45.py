"""D-C3: Eq. (45) for the log-linear family -- builders bessel, gram, interp, direct.

* Small grid (8 x 16, dt 1/16): every builder against the direct sum (dense oracle), on
  all cells, a cut cell set and a padded grid (t_max = T', alpha continued); at
  alpha_d = 0 Eq. (45) is the exact stationary covariance (WT test_matrix.py).
* Capabilities and arguments are checked.
* G6 (64 x 256, manuscript_mid truth): the tri_raw / tri_cut / tri_exact-interior /
  tri_pad blocks (Frobenius norms + 50 sampled entries) and tri_raw / tri_cut
  (quad_C, logdet_C).
* Strategy cross-check: gram vs bessel <= 1e-11, normalised by max|block| over all bands
  and per band (spec §3.2; prior corners under ``slow``).
* JAX: gram/padded jitted with the Gram plan as an argument (HLO stays small).
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.evolutionary import eq45_blocks_cheb, eq45_coeffs_interp, eq45_plan
from wdm.matrix import build_C_wdm_active

import domtest_inputs as T
from domainia.covariance import Eq45Covariance, keep_mask
from domainia.covariance.blocktri import blocks_to_dense, to_layout
from domainia.domains import CellSet
from domainia.spectra import LogLinearPSD

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
T_REC = N * DT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=T_REC)


def _cells(Nt: int = NT) -> CellSet:
    return CellSet.all_active(wdm.Grid(NF, Nt, DT))


def _dense_blocks(cov: Eq45Covariance, nu: np.ndarray) -> np.ndarray:
    from wdm.backend import NUMPY

    d, u = to_layout(*cov.blocks(nu), cov.sizes, "ragged", NUMPY)
    return np.asarray(blocks_to_dense(d, u, NUMPY))


@pytest.mark.parametrize("which", ["all", "cut", "padded"])
@pytest.mark.parametrize("builder", ["bessel", "gram", "interp"])
def test_builders_match_the_direct_sum(builder, which):
    Nt = NT + 4 if which == "padded" else NT
    cells = _cells(Nt)
    if which == "cut":
        cells = cells.drop_columns(3, 1)
    t_max = Nt * NF * DT if which == "padded" else None
    ref = Eq45Covariance(PSD, cells, builder="direct", assembly="dense", t_max=t_max).dense(NU)
    got = _dense_blocks(Eq45Covariance(PSD, cells, builder=builder, t_max=t_max), NU)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_assemblies_agree(rng):
    cells = _cells()
    blk = Eq45Covariance(PSD, cells, builder="bessel")
    den = Eq45Covariance(PSD, cells, builder="bessel", assembly="dense", chol_impl="scipy", symmetrize=False)
    y = rng.standard_normal(N)
    Fb, Fd = blk.factor(NU), den.factor(NU)
    assert np.array_equal(blk.dense(NU), den.dense(NU))
    assert float(Fb.logdet) == pytest.approx(float(Fd.logdet), rel=1e-13)
    assert float(Fb.quad(y)) == pytest.approx(float(Fd.quad(y)), rel=1e-12)


def test_interp_is_the_wdm_dev_chebyshev_path():
    plan = eq45_plan(NF, NT, DT)
    Sn = lambda f, t: PSD(f, t, NU)                                 # noqa: E731
    want = eq45_blocks_cheb(plan, eq45_coeffs_interp(Sn, plan.f, plan.t_max, plan.P, k=plan.k_used))
    got = Eq45Covariance(PSD, _cells(), builder="interp").blocks(NU)
    assert all(np.array_equal(a, b) for a, b in zip(got[0], want[0], strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(got[1], want[1], strict=True))


def test_stationary_limit_is_the_exact_covariance():
    """At alpha_d = 0 Eq. (45) is the exact WDM covariance (WT test_eq45_is_exact...)."""
    for ac in (-2.0, -0.7):
        nu = np.array([ac, 0.0])
        exact = build_C_wdm_active(NF, NT, DT, lambda f: PSD(f, 0.0, nu))
        got = _dense_blocks(Eq45Covariance(PSD, _cells(), builder="bessel"), nu)
        assert np.max(np.abs(got - exact)) < 1e-10 * np.max(np.abs(exact))


def test_capabilities_and_arguments_are_checked():
    cells = _cells()

    class NoSn:
        capabilities = frozenset({"separable"})

    with pytest.raises(ValueError, match="capability 'bessel'"):
        Eq45Covariance(NoSn(), cells, builder="bessel")
    with pytest.raises(ValueError, match="capability 'separable'"):
        Eq45Covariance(PSD, cells, builder="separable")
    with pytest.raises(ValueError, match="callable"):
        Eq45Covariance(NoSn(), cells, builder="interp")
    with pytest.raises(ValueError, match="assembly='dense'"):
        Eq45Covariance(PSD, cells, builder="direct")
    with pytest.raises(ValueError, match="unknown"):
        Eq45Covariance(PSD, cells, builder="chebyshev")
    with pytest.raises(ValueError, match="band order"):
        CellSet(cells.grid, cells.m[::-1], cells.n[::-1])               # D-A refuses it when built
    sub = cells.drop_columns(1, 1)
    assert np.array_equal(keep_mask(sub), sub.keep_mask())               # D-A's CellSet.keep_mask


# -- G6 at the manuscript_mid truth (64 x 256) --------------------------------------------

SIN_GRID = wdm.Grid(T.SIN_NF, T.SIN_NT, T.SIN_DT)
SIN_PSD = LogLinearPSD(f_floor=T.SIN_F_MIN, f_ref=T.SIN_F_REF, T=T.SIN_T)


def _sin_cells(model: str) -> tuple[CellSet, float | None]:
    if model == "tri_pad":
        return CellSet.all_active(wdm.Grid(T.SIN_NF, T.SIN_NT + 16, T.SIN_DT)), (T.SIN_NT + 16) * T.SIN_NF * T.SIN_DT
    cells = CellSet.all_active(SIN_GRID)
    if model == "tri_cut":
        return cells.drop_columns(6, 5), None
    if model == "tri_exact":
        return cells.drop_columns(8, 7), None
    return cells, None


@pytest.mark.parametrize("model", ["tri_raw", "tri_cut", "tri_exact", "tri_pad"])
def test_g6_tri_blocks(model):
    ref = T.sinusoid_mid()["g6"]["tri_blocks"][model]
    cells, t_max = _sin_cells(model)
    diag, up = Eq45Covariance(SIN_PSD, cells, builder="bessel", t_max=t_max).blocks(T.SIN_NU)
    assert (len(diag), len(up)) == (ref["n_diag"], ref["n_up"])
    assert [list(b.shape) for b in diag] == ref["diag_shapes"]
    fro_d = np.array([np.linalg.norm(b) for b in diag])
    fro_u = np.array([np.linalg.norm(b) for b in up])
    assert np.allclose(fro_d, np.asarray(ref["diag_fro"], float), rtol=1e-12, atol=0)
    assert np.allclose(fro_u, np.asarray(ref["up_fro"], float), rtol=1e-12, atol=0)
    for kind, k, i, j, v in ref["samples"]:
        blk = diag[k] if kind == "diag" else up[k]
        assert abs(blk[i, j] - T.num(v)) <= 1e-12 * max(fro_d[k], fro_u[k] if k < len(up) else 0.0)


@pytest.mark.parametrize("model", ["tri_raw", "tri_cut"])
def test_g6_tri_quad_and_logdet(model, same_build):
    """y^T C^-1 y and log|C| (A_n = 1) of tri_raw / tri_cut, WT order (ragged BlockTri).

    Bitwise on the golden's recording build: tri_cut's log|C| moves by 2.9e-11 (1.7e-16
    relative, invisible to any tolerance) when the per-block sums are added as numpy scalars
    instead of Python floats (Python 3.12's builtin sum is compensated)."""
    s = T.sinusoid_mid()
    ref = s["g6"]["parts"][model]
    cells, _ = _sin_cells(model)
    y = T.sinusoid_cells_residual()[keep_mask(cells)]
    F = Eq45Covariance(SIN_PSD, cells, builder="bessel").factor(T.SIN_NU)
    q, ld = float(F.quad(y, form="solve")), float(F.logdet)
    if same_build(s["g6"]["runtime"]):
        assert (q, ld) == (T.num(ref["quad_C"]), T.num(ref["logdet_C"]))
    assert q == pytest.approx(T.num(ref["quad_C"]), rel=1e-13, abs=0)
    assert ld == pytest.approx(T.num(ref["logdet_C"]), rel=1e-13, abs=0)


def _gram_vs_bessel(cells: CellSet, nu: np.ndarray, t_max: float | None) -> tuple[float, float]:
    """(global, per-band) normalised max |gram - bessel| over every block."""
    bd, bu = Eq45Covariance(SIN_PSD, cells, builder="bessel", t_max=t_max).blocks(nu)
    gd, gu = Eq45Covariance(SIN_PSD, cells, builder="gram", t_max=t_max).blocks(nu)
    scale = max(np.abs(b).max() for b in bd)
    worst_g = worst_b = 0.0
    for m, b in enumerate(bd):
        s = b.shape[0]
        d = np.abs(np.asarray(gd[m])[:s, :s] - b).max()
        worst_g, worst_b = max(worst_g, d / scale), max(worst_b, d / np.abs(b).max())
        if m < len(bu):
            s2 = bu[m].shape[1]
            d = np.abs(np.asarray(gu[m])[:s, :s2] - bu[m]).max()
            worst_g = max(worst_g, d / scale)
            worst_b = max(worst_b, d / max(np.abs(b).max(), np.abs(bu[m]).max()))
    return worst_g, worst_b


def test_gram_matches_bessel_at_the_truth():
    worst_g, worst_b = _gram_vs_bessel(CellSet.all_active(SIN_GRID), T.SIN_NU, None)
    assert worst_g <= 1e-11 and worst_b <= 1e-11


@pytest.mark.slow
@pytest.mark.parametrize("nu", [(-4.0, -2.0), (-4.0, 2.0), (0.0, -2.0), (0.0, 2.0)])
@pytest.mark.parametrize("pad", [False, True])
def test_gram_matches_bessel_at_the_prior_corners(nu, pad):
    cells, t_max = _sin_cells("tri_pad" if pad else "tri_raw")
    worst_g, worst_b = _gram_vs_bessel(cells, np.asarray(nu), t_max)
    assert worst_g <= 1e-11 and worst_b <= 1e-11


@pytest.mark.jax
def test_jax_gram_padded_matches_numpy_bessel(jax_be, rng):
    import jax.numpy as jnp

    cells = _cells()
    y = rng.standard_normal(N)
    Fn = Eq45Covariance(PSD, cells, builder="bessel").factor(NU)
    cov = Eq45Covariance(PSD, cells, builder="gram", layout="padded")
    Fj = cov.factor(jnp.asarray(NU))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
    for form in ("solve", "whiten"):
        assert float(Fj.quad(jnp.asarray(y), form=form)) == pytest.approx(float(Fn.quad(y)), rel=1e-12)


@pytest.mark.jax
def test_jax_gram_plan_is_a_jit_argument(jax_be):
    """At 64 x 256 the Gram constants are ~51 MB; closed over they would be ~100 MB of HLO."""
    import jax.numpy as jnp

    cov = Eq45Covariance(SIN_PSD, CellSet.all_active(SIN_GRID), builder="gram", layout="padded")
    F = cov.factor(jnp.asarray(T.SIN_NU))
    text = cov._jit_cache["factor", None].lower(cov.constants(jax_be), jnp.asarray(T.SIN_NU), None).as_text()
    assert len(text) < 5_000_000
    assert float(F.logdet) == pytest.approx(T.num(T.sinusoid_mid()["g6"]["parts"]["tri_raw"]["logdet_C"]), rel=1e-12)
