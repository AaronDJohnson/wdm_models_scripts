"""D-C9: ZeroPadded -- the padded models' M Lambda M, both strategies.

* time_marginal (WT PaddedMarginal): the density of the data samples under
  C_dd = (W^T Lambda W)_dd / dt^2 (dense oracle, WT test_matrix.py), diagonal and Eq. (45)
  bases.  G6: W_p and the sinusoid band/mid/tri_pad parts and lnL.
* JAX: both strategies jitted against NumPy.
* cell_congruence (GB pad_covariance): A Lambda_R A^T for A = identity except on the E
  rows, diagonal and block-tridiagonal bases; ``schur`` (opt-in) equals ``dense``.
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY
from wdm.grid import active_mask

import domtest_inputs as T
from domainia.covariance import (BlockTridiagonalCovariance, DiagonalCovariance, Eq45Covariance, PaddedFactor,
                                 ZeroPadded)
from domainia.domains import CellSet
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.spectra import LogLinearPSD

LOG_2PI = float(np.log(2.0 * np.pi))
NF, NT, DT, PAD = 8, 16, 1 / 16, 4
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
EXT = wdm.Grid(NF, NT + PAD, DT)


def _base(kind: str):
    """Every cell of the extended grid: Eq. (45) (t_max = T') or a midpoint-like diagonal,
    alpha continued over the padding."""
    if kind == "tri":
        return Eq45Covariance(PSD, CellSet.all_active(EXT), builder="bessel", t_max=EXT.N * DT)
    m, n = wdm.grid.packed_cells(NF, NT + PAD)
    L = np.log(np.maximum(wdm.band_frequencies(NF, DT), 0.5))
    v = 0.5 * np.exp((NU[0] + NU[1] * (n * EXT.dT / (N * DT) - 0.5)) * L[m])
    return DiagonalCovariance(v, v.size)


@pytest.mark.parametrize("kind", ["diag", "tri"])
def test_time_marginal_is_the_data_density(rng, kind):
    """NumPy (the JAX path is ``test_jax_matches_numpy``: eager JAX transforms cost ~3 s here)."""
    backend = NUMPY
    base = _base(kind)
    zp = ZeroPadded(base, strategy="time_marginal", grid=EXT, pad_columns=PAD)
    assert zp.n == N
    W = wdm.forward_time_many(np.eye(EXT.N), NF, NT + PAD, DT)[:, active_mask(NF, NT + PAD)].T
    Wd = W[:, :N]
    Cdd = Wd.T @ base.dense(NU) @ Wd / DT ** 2
    F = zp.factor(backend.asarray(NU), backend)
    assert isinstance(F, PaddedFactor) and bool(F.ok)
    x = rng.standard_normal(N)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(Cdd)[1], rel=1e-10)
    assert float(F.quad(backend.asarray(x))) == pytest.approx(x @ np.linalg.solve(Cdd, x), rel=1e-9)
    X = rng.standard_normal((N, 2))
    assert np.allclose(np.asarray(F.solve(backend.asarray(X))), np.linalg.solve(Cdd, X), rtol=1e-8, atol=1e-10)
    with pytest.raises(NotImplementedError):
        F.whiten(backend.asarray(x))


def test_time_marginal_arguments_are_checked():
    with pytest.raises(ValueError, match="extended grid"):
        ZeroPadded(_base("diag"), strategy="time_marginal")
    with pytest.raises(ValueError, match="cells"):
        ZeroPadded(DiagonalCovariance(np.ones(3), 3), strategy="time_marginal", grid=EXT)
    with pytest.raises(ValueError, match="strategy"):
        ZeroPadded(_base("diag"), strategy="taper")
    # a failed (ragged) Lambda: WT's LinAlgError before Q_pp -> ok False, no exception
    sizes = tuple(c for _, c in CellSet.all_active(EXT).band_sizes())
    bad = BlockTridiagonalCovariance(lambda nu, be: ([be.asarray(-np.eye(s)) for s in sizes],
                                                     [be.asarray(np.zeros((a, b))) for a, b in zip(sizes, sizes[1:])]),
                                     sizes)
    F = ZeroPadded(bad, strategy="time_marginal", grid=EXT, pad_columns=PAD).factor(NU)
    assert not bool(F.ok) and F.LiWp is None
    with pytest.raises(ValueError, match="failed"):
        F.quad(np.ones(N))


def test_g6_padding_impulses():
    g6 = T.sinusoid_mid()["g6"]["maps"]["Wp"]
    zp = ZeroPadded(DiagonalCovariance(np.ones(64 * 272), 64 * 272), strategy="time_marginal",
                    grid=wdm.Grid(T.SIN_NF, T.SIN_NT + 16, T.SIN_DT))
    Wp = zp.constants()["Wp"]
    assert Wp.shape == (17408, 1024)
    T.assert_summary(Wp, g6)


@pytest.mark.parametrize("base", ["band", "mid", "tri"])
def test_g6_sinusoid_pad_parts(base):
    s = T.sinusoid_mid()
    ref = s["g6"]["parts"][f"{base}_pad"]
    ext = wdm.Grid(T.SIN_NF, T.SIN_NT + 16, T.SIN_DT)
    if base == "tri":
        cov = Eq45Covariance(LogLinearPSD(f_floor=4.0, f_ref=1.0, T=T.SIN_T), CellSet.all_active(ext),
                             builder="bessel", t_max=ext.N * T.SIN_DT)
    else:
        fn = T.wt_band_variance if base == "band" else T.wt_mid_variance
        v = fn(T.SIN_NF, T.SIN_NT + 16, T.SIN_DT, *T.SIN_NU)
        assert T.sha256(v) == s["g6"]["variances"][f"{base}_pad"]["sha256"]
        cov = DiagonalCovariance(v, v.size)
    nb = FourierBandNuisance(T.SIN_N, T.SIN_DT, T.SIN_FMIN_CUT)
    F = FlatMarginal(ZeroPadded(cov, strategy="time_marginal", grid=ext), nb.discarded_basis().T).factor(T.SIN_NU)
    y = s["d"] - s["h"]
    p = F.parts(y)
    for key in ("quad_C", "quad_H", "logdet_C", "logdet_H"):
        assert float(p[key]) == pytest.approx(T.num(ref[key]), rel=1e-12, abs=0), key
    A_n, n_eff = T.num(ref["A_n"]), T.SIN_N - F.n_u
    lnl = -0.5 * (float(F.quad(y)) / A_n + float(F.logdet) + n_eff * np.log(A_n) + n_eff * LOG_2PI)
    assert n_eff == ref["n_eff"] and abs(lnl - T.num(ref["lnL"])) <= 1e-9


class _Mask:
    """A synthetic PadMask: 3 store bands x 6 columns, 2 analysis bands x 4 data columns."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.NT_ext, self.k0 = 6, 1
        b = np.repeat(np.arange(2), 4)
        n = np.tile(np.arange(4), 2)
        self.Kpos = (self.k0 + b) * self.NT_ext + n
        self.E = np.flatnonzero((n == 0) | (n == 3))
        R = 3 * self.NT_ext
        self.AE = np.eye(R)[self.Kpos[self.E]] + 0.1 * rng.standard_normal((self.E.size, R))

    def A(self) -> np.ndarray:
        A = np.eye(3 * self.NT_ext)[self.Kpos]
        A[self.E] = self.AE
        return A


def _blocks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((18, 18))
    M = X @ X.T / 18 + 2 * np.eye(18)
    band = np.arange(18) // 6
    M = np.where(np.abs(band[:, None] - band[None, :]) <= 1, M, 0.0)
    d = np.stack([M[6 * b:6 * b + 6, 6 * b:6 * b + 6] for b in range(3)])
    u = np.stack([M[6 * b:6 * b + 6, 6 * b + 6:6 * b + 12] for b in range(2)])
    return d, u


@pytest.mark.parametrize("kind", ["diag", "blocktri"])
def test_cell_congruence_is_a_lambda_a_transpose(backend, rng, kind):
    mask = _Mask(rng)
    if kind == "diag":
        v = rng.uniform(0.5, 2.0, 18)
        base = DiagonalCovariance(v, 18)
        Lam = np.diag(v)
    else:
        d, u = _blocks(rng)
        base = BlockTridiagonalCovariance(lambda nu, be: (be.asarray(d), be.asarray(u)), (6, 6, 6), layout="ragged")
        Lam = np.asarray(base.dense(np.zeros(1)))
    A = mask.A()
    want = A @ Lam @ A.T
    y = rng.standard_normal(8)
    dense = ZeroPadded(base, strategy="cell_congruence", mask=mask, n_bands=2, n_columns=4)
    assert np.allclose(dense.dense(np.zeros(1)), want, rtol=0, atol=1e-13)
    for factor in ("dense", "schur"):
        zp = ZeroPadded(base, strategy="cell_congruence", mask=mask, n_bands=2, n_columns=4, factor=factor)
        F = zp.factor(backend.asarray(np.zeros(1)), backend, jit=False)
        assert bool(F.ok)
        assert float(F.logdet) == pytest.approx(np.linalg.slogdet(want)[1], rel=1e-11)
        assert float(F.quad(backend.asarray(y), form="whiten")) == pytest.approx(y @ np.linalg.solve(want, y), rel=1e-10)


def test_cell_congruence_arguments_are_checked(rng):
    with pytest.raises(ValueError, match="mask"):
        ZeroPadded(DiagonalCovariance(np.ones(18), 18), strategy="cell_congruence")
    with pytest.raises(ValueError, match="factor"):
        ZeroPadded(DiagonalCovariance(np.ones(18), 18), strategy="cell_congruence", mask=_Mask(rng), n_bands=2,
                   n_columns=4, factor="lu")


@pytest.mark.jax
@pytest.mark.parametrize("strategy", ["time_marginal", "cell_congruence"])
def test_jax_matches_numpy(jax_be, rng, strategy):
    import jax.numpy as jnp

    if strategy == "time_marginal":
        zp = ZeroPadded(Eq45Covariance(PSD, CellSet.all_active(EXT), builder="gram", layout="padded",
                                       t_max=EXT.N * DT), strategy="time_marginal", grid=EXT, pad_columns=PAD)
        y, nu, form = rng.standard_normal(N), NU, "solve"
    else:
        d, u = _blocks(rng)
        base = BlockTridiagonalCovariance(lambda nu, be: (be.asarray(d), be.asarray(u)), (6, 6, 6), layout="padded")
        zp = ZeroPadded(base, strategy="cell_congruence", mask=_Mask(rng), n_bands=2, n_columns=4)
        y, nu, form = rng.standard_normal(8), np.zeros(1), "whiten"
    assert zp.jittable
    Fn, Fj = zp.factor(nu), zp.factor(jnp.asarray(nu))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
    assert float(Fj.quad(jnp.asarray(y), form=form)) == pytest.approx(float(Fn.quad(y, form=form)), rel=1e-11)


def test_padding_decouples_the_record_ends():
    """Ported WT test_matrix.py::test_padding_decouples_the_record_ends (8 x 64, dt 1/32): the raw
    Eq. (45) couples the first and the last column through the wrap (t_bar literal); on the
    grid padded by 16 columns the kept cells' coupling drops by more than 100x."""
    nf, nt, dt = 8, 64, 1 / 32
    psd = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=nf * nt * dt)
    ends = {}
    for edge, Nt in (("raw", nt), ("pad", nt + 16)):
        cells = CellSet.all_active(wdm.Grid(nf, Nt, dt)).drop_columns(0, Nt - nt)
        C = Eq45Covariance(psd, cells, builder="bessel").dense(np.array([-2.0, 1.2]))
        n = np.asarray(cells.n)
        ends[edge] = np.abs(C[np.ix_(n == 0, n == nt - 1)]).max() / np.abs(C).max()
    assert ends["raw"] > 1e-2 and ends["pad"] < 1e-2 * ends["raw"]


def test_time_marginal_is_m_lambda_m_on_the_kept_cells():
    """Ported WT test_matrix.py::test_pad_mask_is_the_projection_onto_the_data, the energy-fraction
    half of test_pad_map_and_scaling and test_pad_is_m_lambda_m_on_the_kept_cells (8 x 64,
    dt 1/32, 16 padding columns): M = W chi W^T / dt is a projection whose diagonal is
    wdm's energy fraction rho and which ``wdm.evolutionary.apply_time_mask`` applies; the kept
    cells' rho lies in (0, 1], is 1 away from the edges and about 1/2 in column 0; and the
    time-marginal density is the kept cells' M Lambda M density,
    ln N(x; 0, C_dd) = ln N(A x; 0, (M Lambda M)_KK) + ln|det A|."""
    from wdm.evolutionary import apply_time_mask, energy_fraction

    nf, nt, dt = 8, 64, 1 / 32
    g = wdm.Grid(nf, nt + 16, dt)
    chi = (np.arange(g.N) < nf * nt).astype(float)
    W = wdm.forward_time_many(np.eye(g.N), nf, nt + 16, dt)[:, active_mask(nf, nt + 16)].T   # (cells, N)
    Mm = (W * chi[None, :]) @ W.T / dt
    rho = energy_fraction(chi, nf, nt + 16, dt)
    assert np.allclose(Mm @ Mm, Mm, atol=1e-10)
    assert np.allclose(np.diag(Mm), rho, atol=1e-12)
    Y = np.random.default_rng(3).standard_normal((3, Mm.shape[0]))
    assert np.allclose(apply_time_mask(Y, chi, nf, nt + 16, dt), Y @ Mm, atol=1e-10)
    assert np.allclose(np.sum((Mm - np.eye(Mm.shape[0])) ** 2, axis=0), 1.0 - np.diag(Mm), atol=1e-12)
    _, n = wdm.grid.packed_cells(nf, nt + 16)
    kept = n < nt
    rk, nk = rho[kept], n[kept]
    assert np.all((rk > 0) & (rk <= 1 + 1e-12))
    assert np.allclose(rk[(nk >= 8) & (nk <= nt - 9)], 1.0, atol=1e-3)
    assert 0.3 < np.median(rk[nk == 0]) < 0.7
    psd = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=nf * nt * dt)
    nu = np.array([-1.7, 0.9])
    base = Eq45Covariance(psd, CellSet.all_active(g), builder="bessel", t_max=g.N * dt)
    F = ZeroPadded(base, strategy="time_marginal", grid=g, pad_columns=16).factor(nu)
    A = W[np.ix_(kept, np.arange(nf * nt))]                                  # y_K = A x
    MLM = (Mm @ base.dense(nu) @ Mm)[np.ix_(kept, kept)]
    x = np.random.default_rng(7).standard_normal(nf * nt)
    y = A @ x
    assert float(F.quad(x)) == pytest.approx(y @ np.linalg.solve(MLM, y), rel=1e-6)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(MLM)[1] - 2.0 * np.linalg.slogdet(A)[1], rel=1e-6)
