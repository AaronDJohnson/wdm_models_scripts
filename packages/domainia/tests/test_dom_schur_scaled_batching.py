"""S4 deferral (b): the WG-shaped edge-exact state and the WG batching policy.

* ``SchurEdge(form="scaled")`` over a diagonal interior stores ``B``, ``v`` and ``S`` only and
  forms ``B / v`` on the fly (WG ``_factor`` / ``_solve`` diagonal branch); it equals the WT
  ``"solve"`` form (which stores ``T^-1 B^T``) to round-off.
* ``Covariance.batchable``: ``SchurEdge`` over time rows is not (WG never batched the
  sinusoid edge-exact states); everything else inherits its children's.
* ``NoiseStateCache(batch_rows=None)`` follows WG: batched on an accelerator only, never
  for a non-batchable likelihood; ``batch_rows=True`` there is refused.
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY

from dom_toys import ToyMap, ToySignal
from domainia.covariance import (ChannelStack, DiagonalCovariance, Eq45Covariance, ExactTimeCovariance, PushedRows,
                                 SchurEdge, ZeroPadded)
from domainia.density import RealGaussian
from domainia.domains import CellSet
from domainia.likelihood import GaussianLikelihood, Params
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.sampling import NoiseStateCache
from domainia.spectra import LogLinearPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
CELLS = CellSet.all_active(wdm.Grid(NF, NT, DT))
EDGE = CELLS.columns(3, 1)
INTERIOR = CELLS.complement(EDGE)
U = FourierBandNuisance(N, DT, 1.0).push_to_cells(CELLS)


def _rows():
    return PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS)


def _diag(cells):
    return DiagonalCovariance(BandAveragedVariance(PSD, cells), len(cells.m))


@pytest.mark.parametrize("marginal", [False, True])
def test_scaled_equals_the_wt_solve_form(marginal, rng):
    def make(form):
        cov = SchurEdge(_diag(INTERIOR), _rows(), EDGE, form=form)
        return FlatMarginal(cov, U) if marginal else cov

    y = rng.standard_normal(N)
    wt, wg = make("solve").factor(NU, NUMPY), make("scaled").factor(NU, NUMPY)
    S = wg.base if marginal else wg
    assert S.form == "scaled" and S.TiBt is None and S.B.shape == (EDGE.size, INTERIOR.size)
    assert abs(float(wg.logdet) - float(wt.logdet)) <= 1e-13 * abs(float(wt.logdet))
    assert abs(float(wg.quad(y)) - float(wt.quad(y))) <= 1e-12 * abs(float(wt.quad(y)))
    if not marginal:
        X = rng.standard_normal((N, 2))
        assert np.allclose(wg.solve(X), wt.solve(X), rtol=1e-12, atol=0.0)
        assert np.allclose(wg.whiten(y) @ wg.whiten(y), wt.quad(y), rtol=1e-12)


def test_scaled_arguments_are_checked():
    with pytest.raises(ValueError, match="form must be"):
        SchurEdge(_diag(INTERIOR), _rows(), EDGE, form="whiten")
    with pytest.raises(ValueError, match="DiagonalCovariance interior"):
        SchurEdge(Eq45Covariance(PSD, INTERIOR, builder="bessel"), _rows(), EDGE, form="scaled")
    with pytest.raises(ValueError, match="assembly='schur'"):
        SchurEdge(_diag(INTERIOR), _rows(), EDGE, assembly="dense", form="scaled")


def test_batchable_follows_wg():
    ext = wdm.Grid(NF, NT + 4, DT)
    pad = ZeroPadded(DiagonalCovariance(MidpointVariance(PSD, CellSet.all_active(ext)), ext.N),
                     strategy="time_marginal", grid=ext, pad_columns=4)
    edge = SchurEdge(_diag(INTERIOR), _rows(), EDGE)
    assert _diag(CELLS).batchable and pad.batchable and FlatMarginal(_diag(CELLS), U).batchable
    assert not edge.batchable and not FlatMarginal(edge, U).batchable
    assert not ChannelStack((_diag(CELLS), edge), 2).batchable


def _lik(cov, backend):
    rng = np.random.default_rng(0)
    return GaussianLikelihood(rng.standard_normal(N), ToyMap(np.eye(N)), cov, RealGaussian(),
                              signal=ToySignal(rng.standard_normal((1, N))),
                              params=Params(("s", "alpha_c", "alpha_d"), (0,), (1, 2)), backend=backend)


@pytest.mark.jax
def test_batch_rows_default_is_the_wg_policy(jax_be, monkeypatch):
    import domainia.sampling as sampling

    diag = _lik(FlatMarginal(_diag(CELLS), U), "jax")
    edge = _lik(SchurEdge(_diag(INTERIOR), _rows(), EDGE, form="scaled"), "jax")
    assert diag.batchable and not edge.batchable
    assert not NoiseStateCache(diag, 4).batch_rows                    # CPU: WG batch_rungs off
    assert NoiseStateCache(diag, 4, batch_rows=True).batch_rows
    with pytest.raises(ValueError, match="not batchable"):
        NoiseStateCache(edge, 4, batch_rows=True)
    monkeypatch.setattr(sampling, "_accelerator", lambda: True)
    assert NoiseStateCache(diag, 4).batch_rows                        # GPU: batched ...
    assert not NoiseStateCache(edge, 4).batch_rows                    # ... except edge-exact
    X = np.column_stack([np.linspace(-1, 1, 4), np.full(4, NU[0]), np.full(4, NU[1])])
    got = NoiseStateCache(edge, 4)(X)
    want = _lik(SchurEdge(_diag(INTERIOR), _rows(), EDGE), "numpy")(X)
    assert np.max(np.abs(got - want) / np.abs(want)) <= 1e-12
