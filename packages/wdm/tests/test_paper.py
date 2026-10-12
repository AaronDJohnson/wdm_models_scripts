"""C7: the literal Paper I algorithms (wdm.paper) against the fast package (spec 2.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import wdm
from wdm import paper
from wdm.filter import phi_hz
from wdm.matrix import build_C_time, build_W
from wdm.stationary import (
    assemble_lambda_full_active,
    build_lambda_blocks_full_active,
    psd_from_K,
    psd_from_lambda_blocks,
    build_K_full_active,
)

DATA_GRIDS = [(8, 16, 1.0), (16, 32, 0.5), (8, 32, 1.7), (32, 64, 1.0), (4, 8, 1.0)]
PSD_GRIDS = [(8, 16, 1.0), (16, 32, 0.5), (8, 32, 1.7), (32, 64, 1.0)]
PSDS = {
    "flat": lambda f, dt: np.full_like(np.asarray(f, float), 2.3),
    "linear": lambda f, dt: 1.0 + 3.0 * np.abs(f) / (0.5 / dt),
    "steep": lambda f, dt: 1.0 / (np.abs(f) + 0.02),
    "quadratic": lambda f, dt: 1.0 + 40.0 * (np.abs(f) - 0.25) ** 2,
}
NAMES = (*PSDS, "random")
#: section B, Task B6 (lineage D, June ``paper_algorithms``): golden/wdm/
GOLDEN_WDM = Path(__file__).resolve().parents[3] / "golden" / "wdm"
B6_PSD_GRIDS = {"g8x16dt1p0": (8, 16, 1.0), "g16x32dt0p5": (16, 32, 0.5)}
B6_DATA_GRIDS = {"g8x16dt1p0": (8, 16, 1.0), "g16x32dt0p5": (16, 32, 0.5),
                 "g8x32dt1p7": (8, 32, 1.7), "g32x64dt1p0": (32, 64, 1.0), "g4x8dt1p0": (4, 8, 1.0)}
JUNE_LEAK = 0.26727914306174855                  # max|x| of the June literal Alg 2 leak


def _psd_cases():
    """The June _test sequence: one default_rng(0) for every random PSD, in grid order."""
    rng = np.random.default_rng(0)
    for g, (Nf, Nt, dt) in enumerate(PSD_GRIDS):
        fk = np.fft.rfftfreq(Nf * Nt, dt)
        for name in NAMES:
            S = rng.uniform(0.5, 5.0, size=Nf * Nt // 2 + 1) if name == "random" else PSDS[name](fk, dt)
            yield g, (Nf, Nt, dt), name, S


@pytest.mark.parametrize("grid", DATA_GRIDS)
def test_alg1_is_the_fast_forward_and_alg2_round_trips(grid):
    Nf, Nt, dt = grid
    x = np.random.default_rng(0).standard_normal(Nf * Nt)
    w = paper.alg1_forward(x, Nf, Nt, dt)
    assert np.max(np.abs(w - wdm.forward_time(x, Nf, Nt, dt))) <= 1e-14
    assert np.all(w[~wdm.active_mask(Nf, Nt)] == 0.0)          # Alg 1 zeroes inactive pixels
    assert np.max(np.abs(paper.alg2_inverse(w, Nf, Nt, dt) - x)) <= 1e-12


def test_alg2_leak():
    """Printed Alg 2 leaks a unit inactive DC pixel into x (0.267); wdm.inverse masks it (0.0)."""
    Nf, Nt, dt = 8, 16, 0.5
    x = np.random.default_rng(1).standard_normal(Nf * Nt)
    w = wdm.forward_time(x, Nf, Nt, dt)
    w2 = w.copy()
    w2[0, 1] += 1.0                                       # (0, 1): an inactive DC pixel
    leak = np.max(np.abs(paper.alg2_inverse(w2, Nf, Nt, dt) - paper.alg2_inverse(w, Nf, Nt, dt)))
    assert leak == pytest.approx(0.267, abs=5e-4)
    assert np.max(np.abs(wdm.inverse_time(w2, Nf, Nt, dt) - wdm.inverse_time(w, Nf, Nt, dt))) == 0.0


def test_k_recipes_has_one_function_per_table_row():
    assert sorted(paper.K_RECIPES) == sorted([
        "interior_diag", "interior_offdiag", "edge_diag_dc", "edge_diag_nyquist",
        "edge_cross_dc", "edge_cross_nyquist"])


@pytest.mark.parametrize("case", list(_psd_cases()), ids=lambda c: f"{c[2]}-{c[1]}")
def test_alg3_and_alg4(case):
    _, (Nf, Nt, dt), name, S = case
    N = Nf * Nt
    d, o, na = paper.alg3_lambda_from_psd(S, Nf, Nt, dt)
    dp, op, nap = build_lambda_blocks_full_active(Nf, Nt, dt, S)
    scale = max(float(np.max(np.abs(b))) for b in dp)
    # vs the fast slab-IFFT: measured 2.6e-16 (diag) / 1.6e-17 (off-diag) relative.
    assert max(np.max(np.abs(a - b)) for a, b in zip(d, dp, strict=True)) <= 1e-15 * scale
    assert max(np.max(np.abs(a - b)) for a, b in zip(o, op, strict=True)) <= 1e-15 * scale
    assert all(np.array_equal(a, b) for a, b in zip(na, nap, strict=True))
    if name != "random" and N <= 512:
        Wm = build_W(Nf, Nt, dt)
        dense = Wm @ build_C_time(N, dt, lambda f: PSDS[name](f, dt)) @ Wm.T
        L = assemble_lambda_full_active(d, o, na, Nf, Nt)
        assert np.max(np.abs(L - dense)) <= 1e-10 * np.max(np.abs(dense))
    sl = slice(Nt // 2, N // 2 - Nt // 2)
    S4 = paper.alg4_psd_from_lambda(d, o, na, Nf, Nt, dt)
    assert np.max(np.abs(S4[sl] - S[sl])) <= 1e-10 * np.max(np.abs(S[sl]))
    # the production backward maps agree with the literal one on the interior slice
    So = psd_from_lambda_blocks(dp, op, nap, Nf, Nt, dt)
    K, _ = build_K_full_active(Nf, Nt, dt, S)
    Sk = psd_from_K(K, Nf, Nt, dt)
    for fast in (So, Sk):
        assert np.max(np.abs(fast[sl] - S4[sl])) <= 1e-10 * np.max(np.abs(S[sl]))


def _b6(name):
    """One of B6's lineage-D files (section B, Task B6), or skip naming that task."""
    path = GOLDEN_WDM / name
    if not path.is_file():
        pytest.skip(f"{path} missing: run section B, Task B6 "
                    "(scripts/golden/capture_paper1.py, lineage D) first")
    return np.load(path)


@pytest.mark.parametrize("g", sorted(B6_PSD_GRIDS))
def test_window_alg3_and_alg4_are_bitwise_the_lineage_d_golden(g):
    """June paper_algorithms in lineage D: phi_hz, Alg 3 and Alg 4 are bit-identical."""
    z = _b6("alg4_reference.npz")
    Nf, Nt, dt = B6_PSD_GRIDS[g]
    N = Nf * Nt
    sl = slice(Nt // 2, N // 2 - Nt // 2)
    assert np.array_equal(phi_hz(Nf, Nt, dt), z[f"{g}__phi_june"])
    for name in NAMES:
        S = z[f"{g}__{name}__S_in"]
        if name == "random":                  # B6 draws a fresh default_rng(0) per grid
            assert np.array_equal(S, np.random.default_rng(0).uniform(0.5, 5.0, N // 2 + 1))
        d, o, na = paper.alg3_lambda_from_psd(S, Nf, Nt, dt)
        assert all(np.array_equal(na[m], z[f"{g}__n_active_{m:02d}"]) for m in range(Nf + 1))
        if f"{g}__{name}__lam_diag_00" in z.files:          # the full June Lambda (8x16)
            dj = [z[f"{g}__{name}__lam_diag_{m:02d}"] for m in range(Nf + 1)]
            oj = [z[f"{g}__{name}__lam_offd_{m:02d}"] for m in range(Nf)]
            assert all(np.array_equal(a, b) for a, b in zip(d, dj, strict=True)), name
            assert all(np.array_equal(a, b) for a, b in zip(o, oj, strict=True)), name
            assert np.array_equal(paper.alg4_psd_from_lambda(dj, oj, na, Nf, Nt, dt),
                                  z[f"{g}__{name}__S_alg4"]), name
        got = paper.alg4_psd_from_lambda(d, o, na, Nf, Nt, dt)
        assert np.array_equal(got, z[f"{g}__{name}__S_alg4"]), name
        ref = z[f"{g}__{name}__S_offdiag"]                   # psd_backward_offdiag (June notes)
        assert np.max(np.abs(got[sl] - ref[sl])) <= 1e-14 * np.max(np.abs(ref[sl])), name
        dp, op, nap = build_lambda_blocks_full_active(Nf, Nt, dt, S)
        K, _ = build_K_full_active(Nf, Nt, dt, S)
        for fast in (psd_from_lambda_blocks(dp, op, nap, Nf, Nt, dt), psd_from_K(K, Nf, Nt, dt)):
            assert np.max(np.abs(fast[sl] - S[sl])) <= 1e-10 * np.max(np.abs(S[sl])), name


def test_alg1_alg2_and_the_leak_are_bitwise_the_lineage_d_golden():
    """June literal Alg 1 / Alg 2 outputs and the Alg 2 leak, bit for bit."""
    a = _b6("paper_june_alg12.npz")
    for g, (Nf, Nt, dt) in B6_DATA_GRIDS.items():
        x = a[f"{g}__x"]
        assert np.array_equal(paper.alg1_forward(x, Nf, Nt, dt), a[f"{g}__w_alg1"]), g
        assert np.array_equal(paper.alg2_inverse(a[f"{g}__w_alg1"], Nf, Nt, dt), a[f"{g}__x_alg2"]), g
        # The June package transform used the Hz window; the release's omega form
        # agrees to rounding (measured <= 1.4e-15), not bit for bit.
        assert np.max(np.abs(wdm.forward_time(x, Nf, Nt, dt) - a[f"{g}__w_package"])) <= 1e-14, g
    leak = paper.alg2_inverse(a["leak__w"], 8, 16, 0.5)
    assert np.array_equal(leak, a["leak__x_alg2"])
    assert np.max(np.abs(leak)) == JUNE_LEAK
    assert np.max(np.abs(wdm.inverse_time(a["leak__w"], 8, 16, 0.5))) == 0.0
