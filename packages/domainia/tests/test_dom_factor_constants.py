"""S4 deferral (b), spec section 3.3: a factor (a noise state) holds per-state arrays only.

The constants a factor's methods read -- edge/interior positions, the padding impulses
``W_p`` and the active-cell index, the nuisance ``U``, the padded band layout, the
Chebyshev basis and the retained-bin mask -- stay in ``cov.constants(be)`` and are passed
as ``C=`` (``Covariance.factor`` attaches them to the factor it returns, outside every
pytree).  Checked on NumPy for every factor class: no factor array shares memory with a
constants array, the explicit and the attached constants give the same numbers, and a
bare factor (as unstacked from a cache) refuses to guess.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY

from domainia.covariance import (ChannelStack, DiagonalCovariance, Eq45Covariance, ExactTimeCovariance,
                                 PushedRows, SchurEdge, ZeroPadded)
from domainia.domains import CellSet
from domainia.nuisance import FlatMarginal, FourierBandNuisance, ProjectedOperatorMarginal
from domainia.spectra import LogLinearPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NF, NT, DT, PAD = 8, 16, 1 / 16, 4
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
GRID, EXT = wdm.Grid(NF, NT, DT), wdm.Grid(NF, NT + PAD, DT)
CELLS = CellSet.all_active(GRID)
EDGE = CELLS.columns(3, 1)
INTERIOR = CELLS.complement(EDGE)
NUIS = FourierBandNuisance(N, DT, 1.0)


def _arrays(obj):
    """Every ndarray reachable through a factor's / constants' dataclass fields, tuples,
    lists and dicts (scalars excluded)."""
    if isinstance(obj, np.ndarray):
        if obj.size > 1:
            yield obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _arrays(getattr(obj, f.name))
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _arrays(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _arrays(x)


def _diag(cells):
    return DiagonalCovariance(BandAveragedVariance(PSD, cells), len(cells.m))


def _rows():
    return PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS)


def _pad():
    ext = CellSet.all_active(EXT)
    return ZeroPadded(DiagonalCovariance(MidpointVariance(PSD, ext), len(ext.m)), strategy="time_marginal",
                      grid=EXT, pad_columns=PAD)


COVS = {
    "diag": lambda: _diag(CELLS),
    "tri_ragged": lambda: Eq45Covariance(PSD, CELLS, builder="bessel"),
    "tri_padded": lambda: Eq45Covariance(PSD, CELLS, builder="gram"),
    "schur_diag": lambda: SchurEdge(_diag(INTERIOR), _rows(), EDGE),
    "schur_tri": lambda: SchurEdge(Eq45Covariance(PSD, INTERIOR, builder="bessel"), _rows(), EDGE),
    "marginal_solve": lambda: FlatMarginal(SchurEdge(_diag(INTERIOR), _rows(), EDGE), NUIS.push_to_cells(CELLS)),
    "marginal_whiten": lambda: FlatMarginal(Eq45Covariance(PSD, CELLS, builder="gram"), NUIS.push_to_cells(CELLS),
                                            form="whiten"),
    "pad": lambda: FlatMarginal(_pad(), NUIS.discarded_basis().T),
    "projected": lambda: ProjectedOperatorMarginal(ExactTimeCovariance(PSD, N, DT, c_N=0.1), NUIS,
                                                   lambda nu, be: 0.0),
    "stack": lambda: ChannelStack((_diag(CELLS), Eq45Covariance(PSD, CELLS, builder="gram")), 2),
}
#: The factor classes whose methods read constants (the others ignore ``C``).
NEEDS_C = {"tri_padded", "schur_diag", "schur_tri", "marginal_solve", "marginal_whiten", "pad", "projected",
           "stack"}


def _nu(name):
    return np.concatenate([NU, NU]) if name == "stack" else NU


def _y(name, cov, rng):
    return rng.standard_normal((2, CELLS.size)) if name == "stack" else rng.standard_normal(cov.n)


@pytest.mark.parametrize("name", list(COVS))
def test_no_factor_array_is_a_constant(name):
    cov = COVS[name]()
    F = cov.factor(_nu(name), NUMPY)
    consts = list(_arrays(cov.constants(NUMPY)))
    shared = [a.shape for a in _arrays(F) for c in consts if np.shares_memory(a, c)]
    assert not shared, f"{type(F).__name__} stores constants of shapes {shared}"


@pytest.mark.parametrize("name", list(COVS))
def test_explicit_constants_equal_the_attached_ones(name, rng):
    cov = COVS[name]()
    F = cov.factor(_nu(name), NUMPY)
    y = _y(name, cov, rng)
    attached = float(F.quad(y))
    bare = dataclasses.replace(F)                      # a new instance: nothing attached
    assert float(bare.quad(y, C=cov.constants(NUMPY))) == attached
    if name in NEEDS_C:
        with pytest.raises(ValueError, match="needs its covariance's constants"):
            bare.quad(y)
    else:
        assert float(bare.quad(y)) == attached
