"""C8: the stateless WDM II kernels (wdm.evolutionary, wdm.special, wdm.noise).

Every kernel is checked against its source: the frozen wdm_dev oracle
(``wdm_dev_ev``, bit for bit), the WT ``TriCovariance`` in a lineage-A
subprocess (bit for bit), dense references, and across backends.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from wdm_dev_ev import evolutionary as ev

from wdm import evolutionary as E
from wdm import special
from wdm.grid import active_mask
from wdm.matrix import active_W
from wdm.noise import evolutionary_band_variance

NF, NT, DT = 8, 16, 0.5
N = NF * NT
T_REC = N * DT
F_NYQ = 0.5 / DT
WORKSPACE = Path(__file__).resolve().parents[4]
LINEAGE_A = [WORKSPACE / "toy_models" / "vendor" / "wdm" / "src",
             WORKSPACE / "wdm_paper_2" / "toy_models" / "src"]


def _S_stat(f):
    u = np.asarray(f, float) / F_NYQ
    return 1.0 + 0.3 / (u + 0.05) ** 2 + 5.0 * u**4


def _S_evo(f, t):
    return _S_stat(f) * (1.0 + 0.4 * np.sin(2 * np.pi * np.asarray(t, float) / T_REC))


def _L(f, f_min=0.2, f_ref=1.0):
    return np.log(np.maximum(np.asarray(f, dtype=float), f_min) / f_ref)


def _loglinear(alpha_c, alpha_d, T_obs, f_min=0.2):
    def S(f, t):
        return np.exp((alpha_c + alpha_d * (np.asarray(t, float) / T_obs - 0.5)) * _L(f, f_min))
    return S


# -- moved functions: bit for bit against the wdm_dev oracle --------------------------

def test_moved_functions_are_bitwise_the_oracle(rng):
    g = ev.PackedGrid(NF, NT, DT)
    w2 = rng.uniform(0.0, 1.0, N)
    Y = rng.standard_normal((2, N))
    assert np.array_equal(E.energy_fraction(w2, NF, NT, DT), g.energy_fraction(w2))
    assert np.array_equal(E.apply_time_mask(Y, w2, NF, NT, DT), g.apply_mask(Y, w2))
    assert np.array_equal(E.wrap_fractions(NF, NT, DT), ev.wrap_fractions(NF, NT, DT),
                          equal_nan=True)
    for tol in (1e-2, 1e-3, 1e-4):
        assert E.wrap_cut_columns(NF, NT, DT, tol) == ev.wrap_cut_columns(NF, NT, DT, tol)
    assert np.array_equal(E.time_domain_factor(N, DT, _S_evo), ev.time_domain_factor(N, DT, _S_evo))
    assert np.array_equal(E.time_domain_covariance(N, DT, _S_evo),
                          ev.time_domain_covariance(N, DT, _S_evo))
    assert np.array_equal(E.exact_covariance(NF, NT, DT, _S_evo), ev.exact_covariance(NF, NT, DT, _S_evo))
    assert np.array_equal(E.draw_noise(N, DT, _S_evo, np.random.default_rng(5)),
                          ev.draw_noise(N, DT, _S_evo, np.random.default_rng(5)))
    assert np.array_equal(evolutionary_band_variance(NF, NT, DT, _S_evo),
                          ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo))
    assert np.array_equal(evolutionary_band_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF]),
                          ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF]))


def test_energy_fraction_and_time_mask_match_the_dense_mask(rng):
    Wa = active_W(NF, NT, DT)
    w2 = rng.uniform(0.0, 1.0, N)
    Mm = Wa @ np.diag(w2) @ Wa.T / DT
    Y = rng.standard_normal((2, N))
    assert np.allclose(E.apply_time_mask(Y, w2, NF, NT, DT), Y @ Mm.T, rtol=0, atol=1e-12)
    assert np.allclose(E.energy_fraction(w2, NF, NT, DT), np.diag(Mm), rtol=0, atol=1e-12)


def test_wrap_rule_on_the_sinusoid_grid():
    """G5: (6, 5) at (64, 256, 2**-10).  The last cut columns wrap 0.11200% (col 5)
    and 0.11708% (col 251); the first kept ones 0.0631% (cols 6 and 250)."""
    frac = np.nanmax(E.wrap_fractions(64, 256, 2.0**-10), axis=0)
    assert E.wrap_cut_columns(64, 256, 2.0**-10) == (6, 5)
    assert frac[5] == pytest.approx(1.1200e-3, abs=5e-8)
    assert frac[251] == pytest.approx(1.1708e-3, abs=5e-8)
    assert frac[6] == pytest.approx(6.311e-4, abs=5e-8)
    assert frac[250] == pytest.approx(6.311e-4, abs=5e-8)


@pytest.mark.parametrize("Nf, Nt, bands, cut", [
    (6570, 240, (65, 66, 67), (5, 4)),     # GB 3 months, 5 s
    (7200, 438, (71, 72, 73), (5, 4)),     # GB 6 months
])
def test_wrap_rule_on_the_gb_grids(Nf, Nt, bands, cut):
    assert E.wrap_cut_columns(Nf, Nt, 5.0, bands=np.array(bands)) == cut


# -- Eq. (45) ----------------------------------------------------------------------

def _assemble(plan, diag, up):
    n = plan.cells.size
    C = np.zeros((n, n))
    for m, c in enumerate(plan.band_cells):
        C[np.ix_(plan.pos[c], plan.pos[c])] = diag[m]
        if m < plan.Nf:
            c2 = plan.band_cells[m + 1]
            C[np.ix_(plan.pos[c], plan.pos[c2])] = up[m]
            C[np.ix_(plan.pos[c2], plan.pos[c])] = up[m].T
    return C


@pytest.mark.parametrize("keep", [None, "cut"])
def test_interp_blocks_and_direct_sum_are_bitwise_the_oracle(keep):
    g = ev.PackedGrid(NF, NT, DT)
    mask = None if keep is None else ~np.isin(g.n, (0, 1, NT - 1))
    tri = ev.TriCovariance(g, mask)
    plan = E.eq45_plan(NF, NT, DT, mask)
    coeffs = E.eq45_coeffs_interp(_S_evo, plan.f, plan.t_max, plan.P, k=plan.k_used)
    assert np.array_equal(coeffs, tri.chebyshev_coefficients(_S_evo))
    diag, up = E.eq45_blocks_cheb(plan, coeffs)
    d0, u0 = tri.blocks(_S_evo)
    assert all(np.array_equal(a, b) for a, b in zip(diag, d0, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(up, u0, strict=True))
    assert np.array_equal(E.eq45_direct(plan, _S_evo), tri.dense(_S_evo))
    assert not plan.fold_has_duplicates


def test_eq45_is_exact_for_stationary_noise():
    from wdm.matrix import build_C_wdm_active

    plan = E.eq45_plan(NF, NT, DT)
    Sst = lambda f, t: _S_stat(f) + 0 * t  # noqa: E731
    ref = build_C_wdm_active(NF, NT, DT, _S_stat)
    assert np.max(np.abs(E.eq45_direct(plan, Sst) - ref)) <= 1e-12 * np.abs(ref).max()
    coeffs = E.eq45_coeffs_interp(Sst, plan.f, plan.t_max, plan.P, k=plan.k_used)
    C = _assemble(plan, *E.eq45_blocks_cheb(plan, coeffs))
    assert np.max(np.abs(C - ref)) <= 1e-11 * np.abs(ref).max()


def test_eq45_plan_rejects_a_non_boolean_keep():
    # np.arange(N) has the mask's shape: cast to bool it would silently drop cell 0
    N = NF * NT
    for keep in (np.arange(N), np.ones(N, int), list(range(N))):
        with pytest.raises(TypeError, match="boolean mask"):
            E.eq45_plan(NF, NT, DT, keep)
    assert E.eq45_plan(NF, NT, DT, [True] * N).cells.size == N


@pytest.mark.parametrize("Nt_grid, keep_cols, r",[(NT, None, 1.0), (NT + 4, None, (NT + 4) / NT)])
def test_bessel_blocks_match_the_direct_sum_and_gram(Nt_grid, keep_cols, r):
    """bessel (exact Chebyshev) vs direct sum vs gram, log-linear S, both <= 1e-12."""
    ac, ad = -2.0, 1.2
    plan = E.eq45_plan(NF, Nt_grid, DT, t_max=Nt_grid * NF * DT)
    T_obs = N * DT
    L = _L(plan.f)
    diag, up = E.eq45_blocks_cheb(plan, lambda k: E.eq45_coeffs_bessel(L[k], ac, ad, plan.P, r=r))
    C = _assemble(plan, diag, up)
    D = E.eq45_direct(plan, _loglinear(ac, ad, T_obs))
    assert np.max(np.abs(C - D)) <= 1e-12 * np.abs(D).max()
    gp = E.eq45_gram_plan(plan)
    al = ac + ad * (gp.gT / T_obs - 0.5)
    gd, gu = E.eq45_blocks_gram(gp, al, E.eq45_gram_gather(gp, L))
    for m, c in enumerate(plan.band_cells):
        n = len(c)
        assert np.max(np.abs(gd[m][:n, :n] - diag[m])) <= 1e-12 * np.abs(D).max()
        if m < plan.Nf:
            n2 = len(plan.band_cells[m + 1])
            assert np.max(np.abs(gu[m][:n, :n2] - up[m])) <= 1e-12 * np.abs(D).max()


def test_bessel_blocks_are_bitwise_the_wt_toy(tmp_path):
    """Lineage A (frozen WT wdm_toys + vendored wdm) in a subprocess: WT
    TriCovariance.blocks == eq45_blocks_cheb(eq45_coeffs_bessel) bit for bit,
    on the data grid (r = 1) and on a padded grid (t_max = T', r = T'/T_obs)."""
    if not all(p.is_dir() for p in LINEAGE_A):
        pytest.skip("lineage-A trees not present")
    out = tmp_path / "wt_blocks.npz"
    code = (
        "import numpy as np, wdm\n"
        f"assert wdm.__file__.startswith({str(LINEAGE_A[0])!r}), wdm.__file__\n"
        "from wdm_toys.config import ToyConfig\n"
        "from wdm_toys import matrix as M\n"
        "cfg = ToyConfig(Nf=8, Nt=16, dt=0.5, f_min=0.2)\n"
        "res = {}\n"
        "for tag, Nt in (('raw', 16), ('pad', 20)):\n"
        "    g = M.Grid(cfg, Nt)\n"
        "    tri = M.TriCovariance(g, np.ones(g.size, bool), P=24, t_max=Nt * cfg.DT)\n"
        "    d, u = tri.blocks(-2.0, 1.2)\n"
        "    for m, b in enumerate(d): res[f'{tag}_d{m}'] = b\n"
        "    for m, b in enumerate(u): res[f'{tag}_u{m}'] = b\n"
        f"np.savez({str(out)!r}, **res)\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(str(p) for p in LINEAGE_A))
    subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(tmp_path), check=True)
    ref = np.load(out)
    T_obs = N * DT
    for tag, Nt in (("raw", NT), ("pad", NT + 4)):
        t_max = Nt * NF * DT
        plan = E.eq45_plan(NF, Nt, DT, t_max=t_max)
        L = _L(plan.f)
        r = t_max / T_obs
        diag, up = E.eq45_blocks_cheb(plan, lambda k: E.eq45_coeffs_bessel(L[k], -2.0, 1.2, 24, r=r))
        for m, b in enumerate(diag):
            assert np.array_equal(b, ref[f"{tag}_d{m}"]), (tag, m)
        for m, b in enumerate(up):
            assert np.array_equal(b, ref[f"{tag}_u{m}"]), (tag, m)


def test_eq45_kernels_on_jax(jax_be):
    ac, ad = -2.0, 1.2
    plan = E.eq45_plan(NF, NT, DT)
    L = _L(plan.f)
    diag, up = E.eq45_blocks_cheb(plan, lambda k: E.eq45_coeffs_bessel(L[k], ac, ad, 24))
    jd, ju = E.eq45_blocks_cheb(
        plan, lambda k: E.eq45_coeffs_bessel(jax_be.asarray(L[k]), ac, ad, 24), be=jax_be)
    scale = max(np.abs(b).max() for b in diag)
    for a, b in zip(diag + up, jd + ju, strict=True):
        assert np.max(np.abs(np.asarray(b) - a)) <= 1e-13 * scale
    gp = E.eq45_gram_plan(plan)
    al = ac + ad * (gp.gT / (N * DT) - 0.5)
    gL = E.eq45_gram_gather(gp, L)
    nd, nu = E.eq45_blocks_gram(gp, al, gL)
    xd, xu = E.eq45_blocks_gram(gp, jax_be.asarray(al), jax_be.asarray(gL))
    assert np.max(np.abs(np.asarray(xd) - nd)) <= 1e-13 * scale
    assert np.max(np.abs(np.asarray(xu) - nu)) <= 1e-13 * scale


def test_separable_kernels_and_assembly(backend, rng):
    P, G = 12, 20
    U = rng.standard_normal((P, G)) + 1j * rng.standard_normal((P, G))
    sI, sG = rng.uniform(0.5, 2.0, G), rng.uniform(0.5, 2.0, G)
    KI = E.eq45_separable_kernels(U, sI)
    # GB matrix._kernel, verbatim (toy_models/galactic_binary/matrix.py L503-506)
    Ur, Ui = U.real, U.imag
    assert np.array_equal(KI, 0.5 * ((Ur * sI) @ Ur.T + (Ui * sI) @ Ui.T))
    assert np.allclose(KI, 0.5 * np.real(U @ np.diag(sI) @ U.conj().T), rtol=1e-14, atol=1e-14)
    KG = E.eq45_separable_kernels(U, sG)
    n = rng.integers(0, 8, P)
    half = n[:, None] + n[None, :]
    g_half = np.exp(rng.standard_normal(15))
    ref = KI + g_half[half] * KG
    got = E.eq45_separable_assemble(backend.asarray(KI), backend.asarray(KG),
                                    backend.asarray(g_half), half, be=backend)
    assert np.max(np.abs(np.asarray(got) - ref)) <= 1e-15 * np.abs(ref).max()


# -- special ----------------------------------------------------------------------

def test_special(backend, rng):
    a = rng.uniform(-3.2, 3.2, 40)
    got = np.asarray(special.bessel_i(24, backend.asarray(a), be=backend))
    from scipy.special import ive
    ref = ive(np.arange(24)[:, None], a[None, :]) * np.exp(np.abs(a))[None, :]
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-13
    T = special.chebyshev_basis(64, 5)
    u = 2.0 * (np.arange(64) / 64 - 0.5)
    assert np.allclose(T[3], 4 * u**3 - 3 * u, rtol=0, atol=1e-14)


def test_evolutionary_has_no_classes_left():
    for gone in ("PackedGrid", "BlockTri", "Solver", "Schur", "PaddedMarginal", "TriCovariance",
                 "Variant", "make_variant", "marginal_lnlike", "nuisance_basis", "FourierBand",
                 "wdm_discarded", "band_averaged_evolutionary_variance"):
        assert not hasattr(E, gone), gone


def test_wdm_dev_ev_oracle_is_the_seed_with_imports_rewritten():
    import hashlib
    import re

    oracle = Path(__file__).resolve().parent / "oracle" / "wdm_dev_ev" / "evolutionary.py"
    assert hashlib.sha256(oracle.read_bytes()).hexdigest() == (
        "679ef3cfb37d388fe704e0c87386b36c76c888dfb06e96abd85723de9cc0af88")
    seed = WORKSPACE / "wdm_dev" / "src" / "wdm" / "evolutionary.py"
    if not seed.is_file():
        pytest.skip(f"seed not present: {seed}")
    assert hashlib.sha256(seed.read_bytes()).hexdigest() == (
        "3a8140452e83269b46bc2f70382e76c229358134648a71a7c1faa9f5209e3cde")
    rewritten = re.sub(r"^( *)from \.(filter|grid|transform) import",
                       r"\1from vendored_2f7136c.\2 import", seed.read_text(), flags=re.MULTILINE)
    assert rewritten == oracle.read_text()
