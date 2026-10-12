"""D-C3: Eq. (45) for separable spectra S_c = s_I + g(t) s_G (galactic binary).

* The envelope stand-in reproduces the golden half-step basis bit for bit.
* G6 at the anchor: K^I, K^G (fingerprints + diagonals), the half-step index, the dense
  tri_raw covariance per channel, and (quad, logdet) of tri_raw / tri_cut (GB order:
  np.linalg.cholesky of 0.5 (S + S^T), forward solve).
* The block-tridiagonal assembly (GB ``pad_static``) equals the dense one on full bands.
* JAX: both assemblies jitted with the kernels as arguments.
"""

from __future__ import annotations

import numpy as np
import pytest
from wdm.backend import NUMPY

import domtest_inputs as T
from domainia.covariance import Eq45Covariance
from domainia.covariance.blocktri import blocks_to_dense, to_layout
from domainia.domains import CellSet

CASES = ["gb_3mo", "gb_6mo", "gb_12mo"]


def _tri(c: T.GBCase, cells: CellSet, ch: int, assembly: str = "dense", **kw) -> Eq45Covariance:
    return Eq45Covariance(c.psd, cells, builder="separable", assembly=assembly, channel=ch, **kw)


@pytest.mark.parametrize("case", CASES)
def test_envelope_stand_in_is_the_golden_basis(case):
    c = T.gb_case(case)
    t = np.arange(2 * c.NT - 1) * c.grid.dT / 2
    assert np.array_equal(c.env.basis(t), c.A["basis_half"])


@pytest.mark.parametrize("case", CASES)
def test_g6_kernels_half_and_tri_raw(case):
    c = T.gb_case(case)
    covs = [_tri(c, c.cells, ch) for ch in range(2)]
    C = [cov.constants() for cov in covs]
    KI, KG = np.stack([Cc["KI"] for Cc in C]), np.stack([Cc["KG"] for Cc in C])
    fp = c.g6["fingerprints"]
    T.assert_summary(KI, fp["KI"])
    T.assert_summary(KG, fp["KG"])
    assert np.allclose(np.stack([np.diagonal(K) for K in KI]), c.A["KI_diag"], rtol=1e-13, atol=0)
    assert np.allclose(np.stack([np.diagonal(K) for K in KG]), c.A["KG_diag"], rtol=1e-13, atol=0)
    assert np.array_equal(C[0]["half"], C[1]["half"])
    T.assert_summary(C[0]["half"], fp["half"], rtol=0)
    for ch in range(2):
        D = covs[ch].dense(c.mu[ch])
        T.assert_summary(D, fp[f"cov_tri_raw_c{ch}"])
        assert np.allclose(np.diagonal(D), c.A["cov_tri_raw_diag"][ch], rtol=1e-13, atol=0)


@pytest.mark.parametrize("case,model", [("gb_3mo", "tri_raw"), ("gb_3mo", "tri_cut"), ("gb_6mo", "tri_raw"),
                                        ("gb_6mo", "tri_cut"), ("gb_12mo", "tri_raw")])
def test_g6_quad_and_logdet(case, model):
    """Per channel (quad, logdet) at the anchor; the cut drops (5, 4) columns (the periodic
    12-month record is not cut)."""
    c = T.gb_case(case)
    cells = c.cells.drop_columns(5, 4) if model == "tri_cut" else c.cells
    r = c.residual_cells(cells)
    got = np.array([[float(F.quad(r[ch], form="whiten")), float(F.logdet)]
                    for ch, F in ((ch, _tri(c, cells, ch).factor(c.mu[ch])) for ch in range(2))])
    want = c.quad_logdet(model)
    assert np.allclose(got, want, rtol=1e-12, atol=0)
    lnl = -0.5 * (got.sum(axis=1) + len(cells.m) * np.log(2 * np.pi))
    ref = -0.5 * (want.sum(axis=1) + len(cells.m) * np.log(2 * np.pi))
    assert np.max(np.abs(lnl - ref)) <= 1e-8                      # the approved G3 NumPy gate


def test_blocktri_assembly_equals_dense_on_full_bands():
    c = T.gb_case("gb_3mo")
    dense = _tri(c, c.cells, 0).dense(c.mu[0])
    blk = _tri(c, c.cells, 0, assembly="blocktri", layout="ragged")
    d, u = to_layout(*blk.blocks(c.mu[0]), blk.sizes, "ragged", NUMPY)
    assembled = np.asarray(blocks_to_dense(d, u, NUMPY))
    assert np.max(np.abs(assembled - dense)) <= 1e-13 * np.abs(dense).max()
    r = c.residual_cells(c.cells)[0]
    Fd, Fb = _tri(c, c.cells, 0).factor(c.mu[0]), blk.factor(c.mu[0])
    assert float(Fb.logdet) == pytest.approx(float(Fd.logdet), rel=1e-12)
    assert float(Fb.quad(r)) == pytest.approx(float(Fd.quad(r, form="whiten")), rel=1e-11)


def test_separable_arguments_are_checked():
    c = T.gb_case("gb_3mo")
    with pytest.raises(ValueError, match="every column"):
        _tri(c, c.cells.drop_columns(5, 4), 0, assembly="blocktri").constants()
    gap = CellSet(c.grid, tuple(m for m in c.cells.m if m != 66), tuple(n for m, n in zip(c.cells.m, c.cells.n) if m != 66))
    for assembly in ("dense", "blocktri"):          # blocktri would pair bands 65 and 67
        with pytest.raises(ValueError, match="contiguous"):
            _tri(c, gap, 0, assembly=assembly).constants()


@pytest.mark.jax
@pytest.mark.parametrize("assembly", ["dense", "blocktri"])
def test_jax_matches_numpy(jax_be, assembly):
    import jax.numpy as jnp

    c = T.gb_case("gb_3mo")
    cov = _tri(c, c.cells, 1, assembly=assembly)
    r = c.residual_cells(c.cells)[1]
    Fn, Fj = cov.factor(c.mu[1]), cov.factor(jnp.asarray(c.mu[1]))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
    assert float(Fj.quad(jnp.asarray(r), form="whiten")) == pytest.approx(float(Fn.quad(r, form="whiten")), rel=1e-11)
