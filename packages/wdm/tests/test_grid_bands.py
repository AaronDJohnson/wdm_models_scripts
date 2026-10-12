"""C6: packing helpers, phi_hz, atoms and band-limited analysis."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wdm
from wdm.filter import phi_centered, phi_hz
from wdm.grid import (
    Grid,
    active_mask,
    edge_columns,
    endpoint_weights,
    pack,
    packed_cells,
    unpack,
)
from wdm.matrix import active_W
from wdm.transform import (
    atom_spectrum,
    band_atoms,
    base_atom_spectra,
    forward_bands,
    forward_time,
    inverse_bands,
)

GB_DIR = Path(__file__).resolve().parents[4] / "toy_models" / "galactic_binary"
#: (Nf, Nt, dt): small grids plus the three GB analysis grids (N // NT, NT, 5 s).
ATOM_GRIDS = [(4, 8, 1.0), (16, 32, 0.5), (6570, 240, 5.0), (7200, 438, 5.0), (7008, 900, 5.0)]


def test_grid_is_frozen_positional_and_unpacks():
    g = Grid(8, 16, 0.5)
    assert tuple(g) == (8, 16, 0.5) and g.N == 128
    with pytest.raises(AttributeError):
        g.Nf = 4  # type: ignore[misc]


@pytest.mark.parametrize("grid", [(2, 2), (4, 8), (8, 16), (64, 256)])
def test_packed_cells_is_band_order(grid):
    Nf, Nt = grid
    m, n = packed_cells(Nf, Nt)
    ref_m, ref_n = np.nonzero(active_mask(Nf, Nt))
    assert np.array_equal(m, ref_m) and np.array_equal(n, ref_n)
    assert m.size == Nf * Nt and np.all(np.diff(m) >= 0)


def test_pack_unpack_round_trip_and_active_W_order(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = forward_time(rng.standard_normal(Nf * Nt), Nf, Nt, dt)
    y = pack(w, Nf, Nt)
    assert np.array_equal(y, w[active_mask(Nf, Nt)])
    back = unpack(y, Nf, Nt)
    assert np.array_equal(back[active_mask(Nf, Nt)], y) and np.all(back[~active_mask(Nf, Nt)] == 0)
    x = rng.standard_normal(Nf * Nt)
    assert np.allclose(active_W(Nf, Nt, dt) @ x, pack(forward_time(x, Nf, Nt, dt), Nf, Nt),
                       rtol=0, atol=1e-12)
    Y = rng.standard_normal((3, Nf * Nt))
    assert np.array_equal(pack(unpack(Y, Nf, Nt), Nf, Nt), Y)


def test_pack_unpack_on_jax(jax_be, rng):
    Nf, Nt = 8, 16
    w = rng.standard_normal((2, Nf + 1, Nt))
    assert np.array_equal(np.asarray(pack(jax_be.asarray(w), Nf, Nt)), pack(w, Nf, Nt))
    y = pack(w, Nf, Nt)
    assert np.array_equal(np.asarray(unpack(jax_be.asarray(y), Nf, Nt)), unpack(y, Nf, Nt))


def test_edge_columns_and_endpoint_weights():
    assert edge_columns(256, 6, 5) == (0, 1, 2, 3, 4, 5, 251, 252, 253, 254, 255)
    assert edge_columns(16, 0, 0) == ()
    c = endpoint_weights(9)
    assert c[0] == c[-1] == 0.5 and np.all(c[1:-1] == 1.0)


@pytest.mark.parametrize("grid", [(8, 16, 1.0), (16, 32, 0.5), (8, 32, 1.7), (64, 256, 2.0**-10)])
def test_hz_identity(grid):
    """kappa_omega * phi_omega == kappa_Hz * phi_Hz (Paper I eq:Meyer-def) to 1e-15."""
    Nf, Nt, dt = grid
    w = (2.0 * np.sqrt(np.pi) / Nf) * phi_centered(Nf, Nt, dt)
    h = (np.sqrt(2.0) / Nf) * phi_hz(Nf, Nt, dt)
    nz = w != 0
    assert np.array_equal(nz, h != 0)
    assert np.max(np.abs(h[nz] / w[nz] - 1.0)) <= 1e-15
    assert np.max(np.abs(phi_hz(Nf, Nt, dt) / np.sqrt(2 * np.pi) - phi_centered(Nf, Nt, dt))) \
        <= 1e-15 * np.max(phi_centered(Nf, Nt, dt))


def test_atoms_are_the_rows_of_the_transform():
    Nf, Nt, dt = 8, 16, 0.5
    Wa = active_W(Nf, Nt, dt)
    base = base_atom_spectra(Nf, Nt, dt)
    assert sorted(base) == [(m, a) for m in range(Nf + 1) for a in (0, 1)
                            if active_mask(Nf, Nt)[m, a]]
    m, n = packed_cells(Nf, Nt)
    for p in range(m.size):
        atom = np.fft.irfft(atom_spectrum(int(m[p]), int(n[p]), Nf, Nt, dt, base=base), n=Nf * Nt)
        assert np.allclose(atom, Wa[p], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="not an active cell"):
        atom_spectrum(0, 1, Nf, Nt, dt, base=base)


def _gb_reference(tmp_path: Path, grids) -> dict:
    """GB phi and atoms computed by the frozen GB code, from a temporary copy."""
    if not GB_DIR.is_dir():
        pytest.skip(f"GB toy tree not present: {GB_DIR}")
    copy = tmp_path / "gb"
    copy.mkdir()
    for name in ("config.py", "wdm_atoms.py", "geometry.py"):
        shutil.copyfile(GB_DIR / name, copy / name)
    out = tmp_path / "gb_ref.npz"
    code = (
        "import sys, numpy as np\n"
        "from wdm_atoms import phi_centered\n"
        "from geometry import band_atoms\n"
        f"grids = {grids!r}\n"
        "res = {}\n"
        "for i, (Nf, Nt, dt) in enumerate(grids):\n"
        "    N = Nt * Nf\n"
        "    phi = phi_centered(Nt, Nf, dt, N * dt)\n"
        "    res[f'phi{i}'] = phi\n"
        "    res[f'atoms{i}'] = band_atoms(np.array([1, 2, Nf // 2, Nf - 1]), Nt, phi)\n"
        f"np.savez({str(out)!r}, **res)\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(copy))
    subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(copy), check=True)
    return dict(np.load(out))


def test_window_and_atoms_are_bitwise_the_gb_copies(tmp_path):
    ref = _gb_reference(tmp_path, ATOM_GRIDS)
    for i, (Nf, Nt, dt) in enumerate(ATOM_GRIDS):
        assert np.array_equal(phi_centered(Nf, Nt, dt), ref[f"phi{i}"])
        bands = np.array([1, 2, Nf // 2, Nf - 1])
        assert np.array_equal(band_atoms(bands, Nf, Nt, dt), ref[f"atoms{i}"])


@pytest.mark.parametrize("grid", [(8, 16, 1.0), (16, 32, 0.5), (6570, 240, 5.0)])
def test_forward_bands_is_the_dense_atom_analysis(grid, backend, rng):
    Nf, Nt, dt = grid
    bands = np.array([2, 3, 4]) if Nf < 100 else np.array([65, 66, 67])
    k0 = (bands[0] - 1) * (Nt // 2) - 3
    n_bins = (bands[-1] + 1) * (Nt // 2) - k0 + 5
    z = rng.standard_normal((2, n_bins)) + 1j * rng.standard_normal((2, n_bins))
    atoms = band_atoms(bands, Nf, Nt, dt)
    dense = np.stack([
        np.real(np.einsum("ns,...s->...n", atoms[b],
                          z[..., (m - 1) * (Nt // 2) - k0 + np.arange(Nt)]))
        for b, m in enumerate(bands)], axis=-2)
    got = np.asarray(forward_bands(backend.asarray(z), k0, bands, Nf, Nt, dt))
    assert got.shape == (2, bands.size, Nt)
    assert np.max(np.abs(got - dense)) <= 1e-12 * np.max(np.abs(dense))


@pytest.mark.parametrize("grid", [(8, 16, 0.5), (16, 32, 1.0), (8, 16, 2.0)])
def test_forward_bands_is_the_transform_times_sqrt_N_over_2dt(grid, rng):
    """On a whole record (z = rfft(x)) the unit-norm-atom cells of the interior
    bands are the WDM coefficients times sqrt(N / (2 dt)) -- the GB comment's
    "orthonormal coefficient times sqrt(N/2)" (orthonormal = w / sqrt(dt))."""
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = rng.standard_normal(N)
    w = forward_time(x, Nf, Nt, dt)
    cells = forward_bands(np.fft.rfft(x), 0, np.arange(1, Nf), Nf, Nt, dt)
    scale = np.sqrt(N / (2.0 * dt))
    assert np.max(np.abs(cells - scale * w[1:Nf])) <= 1e-12 * np.max(np.abs(cells))


def test_inverse_bands_is_the_adjoint(backend, rng):
    Nf, Nt, dt = 16, 32, 0.5
    bands = np.array([3, 4, 5, 6])
    k0, n_bins = 20, 120
    z = rng.standard_normal(n_bins) + 1j * rng.standard_normal(n_bins)
    w = rng.standard_normal((bands.size, Nt))
    lhs = np.sum(np.asarray(forward_bands(backend.asarray(z), k0, bands, Nf, Nt, dt)) * w)
    rhs = np.real(np.sum(np.conj(z) * np.asarray(
        inverse_bands(backend.asarray(w), k0, n_bins, bands, Nf, Nt, dt))))
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="do not cover"):
        forward_bands(z, k0 + 100, bands, Nf, Nt, dt)
    with pytest.raises(ValueError, match="interior bands"):
        band_atoms([0, 1], Nf, Nt, dt)


def test_time_pixel_centers_doctest_value():
    assert wdm.time_pixel_centers(8, 4, 1.0).tolist() == [0.0, 8.0, 16.0, 24.0]
