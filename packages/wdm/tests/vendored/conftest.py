"""Shared pytest fixtures for the wdm test suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class Grid:
    """Bundle of (Nf, Nt, dt) parameters for tests."""

    Nf: int
    Nt: int
    dt: float

    @property
    def N(self) -> int:
        return self.Nf * self.Nt

    @property
    def T(self) -> float:
        return self.N * self.dt

    @property
    def dF(self) -> float:
        return 1.0 / (2.0 * self.dt * self.Nf)

    @property
    def dT(self) -> float:
        return self.Nf * self.dt

    @property
    def f_nyq(self) -> float:
        return 1.0 / (2.0 * self.dt)


@pytest.fixture
def simple_grid() -> Grid:
    """Small grid (N = 128) for fast tests."""
    return Grid(Nf=8, Nt=16, dt=1.0)


@pytest.fixture
def medium_grid() -> Grid:
    """Medium grid (N = 512) for tests that need more resolution."""
    return Grid(Nf=16, Nt=32, dt=1.0)


@pytest.fixture
def lisa_grid() -> Grid:
    """LISA-like grid (N = 2048) for noise tests."""
    return Grid(Nf=32, Nt=64, dt=10.0)


@pytest.fixture
def flat_psd():
    """Constant PSD ``S_n(f) = 1e-3`` (no Whittle approximation error)."""

    def _flat(f):
        return 1e-3 * np.ones_like(np.asarray(f, dtype=float))

    return _flat


@pytest.fixture
def lisa_psd():
    """Simplified LISA sensitivity curve."""

    def _lisa(f):
        f = np.asarray(f, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            S_acc = 9e-30 / (2 * np.pi * f) ** 4
            S_shot = 2.25e-22
            f_star = 19.09e-3
            R = 1.0 + (f / f_star) ** 2
            Sn = (S_acc + S_shot) * R
        return np.where(np.isfinite(Sn), Sn, np.inf)

    return _lisa


@pytest.fixture
def chirp_signal(simple_grid):
    """Linear chirp on ``simple_grid``."""
    g = simple_grid
    t = np.arange(g.N) * g.dt
    f0 = 0.05 * g.f_nyq
    f1 = 0.45 * g.f_nyq
    fdot = (f1 - f0) / g.T
    return np.cos(2 * np.pi * (f0 * t + 0.5 * fdot * t**2))


@pytest.fixture
def white_noise(simple_grid):
    """Deterministic white-noise realization on ``simple_grid``."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(simple_grid.N)


@pytest.fixture
def gaussian_pulse(simple_grid):
    """Gaussian-windowed monochromatic signal localized in interior bands."""
    g = simple_grid
    t = np.arange(g.N) * g.dt
    f_center = 0.25 * g.f_nyq
    return np.exp(-(((t - g.T / 2) / (g.T / 8)) ** 2)) * np.cos(
        2 * np.pi * f_center * t
    )
