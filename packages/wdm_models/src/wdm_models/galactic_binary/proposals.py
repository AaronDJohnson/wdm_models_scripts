r"""GLASS's single-source proposals and the initial proposal covariance (spec §5.2
``proposals.py``; port of ``toy_models/galactic_binary/gpu/proposals.py``).

* :class:`SourceFisher`: the Fisher of the eight source parameters,
  :math:`F = 2\,\mathrm{Re}(J^H W J)` in the prepared units, with :math:`J` the forward-mode JAX
  Jacobian (``jax.jacfwd``) of the template on the analysis window and
  :math:`W = 1/(s_I + e^{\mu_c} s_G)` at the Whittle fit's level (``mu_freq``); the template
  (``waveform.GBTemplate``, a pytree) is a ``jit`` argument, never closed over (spec §3.3);
* :class:`GlassProposal`: an impulse custom jump running binary_galactica's
  :func:`single_source_mix` (Fisher eigendirection 0.7, ``fm_shift`` 0.1, ``psi_phi`` 0.1,
  prior draw 0.1; the Fisher decomposition refreshed every ``FISHER_REFRESH_STEPS = 80``
  Fisher calls) on ``x[:8]``, one mix per temperature rung;
* :func:`noise_fisher`, :func:`initial_covariance`: the block-diagonal inverse Fisher that
  starts the sampler's proposal covariance.

**Boundary semantics (user ruling 2026-10-11).**  binary_galactica's ``draw_from_fisher`` turns
a non-finite jump, or a jump that takes either cosine to ``|x| >= 1``, into a *whole-vector*
self-proposal (``log q = 0``); gbonfly reverted only the offending component.  The release
uses binary_galactica's rule, so the paper's chains cannot be replayed proposal for proposal;
they are compared statistically (spec §7.4).

The prepared inputs are read by key (``sI``, ``sG``, ``mu_freq``, ``mu_fit``, ``prior_lo``,
``prior_hi``, ``inject_glass``, ``basis_L``, ``envelope_period``, ``T``, ``N``, ``DT``,
``bin_offset``, ``n_bins``, ``freq_lo``, ``freq_n``: all ``prepared.PAPER_KEYS``) from
``prep.arrays`` (a ``prepared.PreparedCase``) or from a mapping of the prepared npz; only the
default template of :meth:`SourceFisher.from_prepared` needs the ``PreparedCase`` itself (its
construction guard reads ``z``, ``store_z`` and the injection).  ``jax`` and
``binary_galactica`` are imported lazily (importing this module imports neither).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..common.params import check_model, is_whittle
from .config import TEMPLATE_BW

__all__ = ["GlassProposal", "SourceFisher", "block_diag", "exact_covariance_fn", "initial_covariance",
           "noise_fisher", "prepared_arrays"]


def prepared_arrays(prep: Any) -> Mapping[str, np.ndarray]:
    """The prepared arrays of ``prep``: its ``.arrays`` (``PreparedCase``) or ``prep`` itself."""
    arrays = getattr(prep, "arrays", prep)
    if not isinstance(arrays, Mapping):
        raise TypeError(f"expected a PreparedCase or a mapping of prepared arrays; got {type(prep).__name__}")
    return arrays


class SourceFisher:
    """``p8 -> (8, 8)`` Fisher of the source block, and its eigen-oracle.

    ``template`` maps the eight GLASS parameters to the ``(2, n)`` complex template in
    prepared units (traceable by JAX); ``weight`` is the ``(2, n)`` inverse variance.  A
    ``waveform.GBTemplate`` (a pytree: orbit and scale are leaves) is passed to the compiled
    Jacobian as an argument; any other callable is wrapped in ``jax.tree_util.Partial``."""

    def __init__(self, template: Callable, weight: Any):
        import jax
        import jax.numpy as jnp
        from wdm.backend import require_x64

        from .waveform import GBTemplate

        require_x64()
        self.template = template
        self.weight = jnp.asarray(np.asarray(weight, dtype=np.float64))           # (2, n)
        self._tpl = template if isinstance(template, GBTemplate) else jax.tree_util.Partial(template)
        self._jac = jax.jit(lambda tpl, p: jax.jacfwd(tpl)(p))

    @classmethod
    def from_prepared(cls, prep: Any, *, backend: str = "onfly", BW: int = TEMPLATE_BW,
                      template: Callable | None = None) -> "SourceFisher":
        """The Fisher on the analysis window (``bin_offset``, ``n_bins``) at ``mu_freq``; the
        template defaults to ``waveform.GBTemplate(prep, BW=BW, backend=backend,
        window="window")`` (with its construction guard), which needs a ``PreparedCase``."""
        a = prepared_arrays(prep)
        if template is None:
            from .waveform import GBTemplate

            if prep is a:
                raise TypeError("the default template needs a prepared.PreparedCase; pass template= "
                                "with a mapping of prepared arrays")
            template = GBTemplate(prep, BW=BW, backend=backend, window="window")
        level = np.exp(np.asarray(a["mu_freq"], float))[:, None]
        return cls(template, 1.0 / (np.asarray(a["sI"], float) + level * np.asarray(a["sG"], float)))

    def jacobian(self, p8) -> np.ndarray:
        """``J = d template / d p8``, ``(2, n, 8)`` complex."""
        import jax.numpy as jnp

        return np.asarray(self._jac(self._tpl, jnp.asarray(p8, dtype=jnp.float64)))

    def __call__(self, p8) -> np.ndarray:
        import jax.numpy as jnp

        J = self._jac(self._tpl, jnp.asarray(p8, dtype=jnp.float64))           # (2, n, 8)
        F = 2.0 * jnp.real(jnp.einsum("cki,ckj,ck->ij", jnp.conj(J), J, self.weight))
        F = np.asarray(F)
        return 0.5 * (F + F.T)

    def eigen(self, p8) -> tuple[np.ndarray, np.ndarray]:
        """``(evalues, evectors)`` of the Fisher at ``p8``; ``(nan, I)`` when it is not finite
        (``draw_from_fisher`` then self-proposes)."""
        F = self(p8)
        if not np.all(np.isfinite(F)):
            return np.full(8, np.nan), np.eye(8)
        return np.linalg.eigh(F)


class GlassProposal:
    """impulse custom jump: GLASS's single-source mix on ``x[:8]``, one mix (and Fisher cache)
    per temperature rung; the noise block ``x[8:]`` is never moved."""

    __name__ = "glass_single_source"

    def __init__(self, prior_lo, prior_hi, T: float, fisher: Any, every: int | None = None):
        from binary_galactica import UCBPrior
        from binary_galactica.glass.proposals import FISHER_REFRESH_STEPS

        self.prior = UCBPrior(lo=np.asarray(prior_lo, float), hi=np.asarray(prior_hi, float))
        self.T, self.fisher = float(T), fisher
        self.every = int(FISHER_REFRESH_STEPS if every is None else every)
        self._mix: dict[int, Any] = {}

    @classmethod
    def from_prepared(cls, prep: Any, fisher: Any, every: int | None = None) -> "GlassProposal":
        a = prepared_arrays(prep)
        return cls(a["prior_lo"], a["prior_hi"], float(a["T"]), fisher, every)

    def mix(self, chain_index: int):
        """The rung's ``single_source_mix`` (built on first use)."""
        m = self._mix.get(chain_index)
        if m is None:
            from binary_galactica import single_source_mix

            m = self._mix[chain_index] = single_source_mix(
                prior=self.prior, fisher=self.fisher.eigen, T=self.T, every=self.every)
        return m

    def __call__(self, chain_stats) -> tuple[np.ndarray, float]:
        x = np.array(chain_stats.current_sample, dtype=np.float64, copy=True)
        p8, logq, _ = self.mix(int(chain_stats.chain_index))(x[:8], chain_stats.rng)
        x[:8] = p8
        return x, float(logq)


def exact_covariance_fn(a: Mapping[str, np.ndarray], c: int) -> Callable:
    """``mu -> M'(mu)`` of channel ``c`` on the window bins (JAX-traceable): domainia's
    ``ExactFourierCovariance.evaluate`` (GB ``likelihoods.full_covariance``) with the case's
    ``envelope.AnnualEnvelope`` (the prepared ``basis_L``)."""
    from domainia.covariance.exact_fourier import ExactFourierCovariance
    from domainia.domains import FourierDomain
    from domainia.spectra import SeparablePSD, TabulatedSpectrum
    from wdm.backend import jax_backend

    from .envelope import AnnualEnvelope

    dom = FourierDomain(int(a["N"]), float(a["DT"]), int(a["bin_offset"]), int(a["n_bins"]), "sigma0")
    env = AnnualEnvelope(float(a["T"]), period=float(a["envelope_period"]), basis_L=a["basis_L"])
    psd = SeparablePSD(((TabulatedSpectrum(dom, a["sI"]), None), (TabulatedSpectrum(dom, a["sG"]), env)))
    cov = ExactFourierCovariance(psd, dom, channel=c)
    be = jax_backend()
    C = {k: (v if k == "lag" else be.asarray(v)) for k, v in cov.host_constants().items()}
    return lambda mu: cov.evaluate(C, mu, be)


def noise_fisher(prep: Any, model: str) -> np.ndarray:
    """Fisher of the noise block at the fitted noise (block-diagonal over channels).

    ``tr(M^-1 dM_k M^-1 dM_l)`` of the exact covariance at ``mu_fit`` for every envelope model;
    ``sum_j (e^mu s_G / M_jj)^2`` over the nominal bins at ``mu_freq`` for the one level per
    channel of ``whittle_raw``/``whittle_cut``."""
    check_model(model)
    a = prepared_arrays(prep)
    sI, sG = np.asarray(a["sI"], float), np.asarray(a["sG"], float)
    if is_whittle(model):
        lo = int(a["freq_lo"])
        k = slice(lo, lo + int(a["freq_n"]))
        blocks = []
        for c in range(2):
            level = np.exp(a["mu_freq"][c]) * sG[c, k]
            blocks.append(np.array([[np.sum((level / (sI[c, k] + level)) ** 2)]]))
        return block_diag(blocks)
    import jax
    import jax.numpy as jnp

    blocks = []
    for c in range(2):
        cov = exact_covariance_fn(a, c)
        mu = jnp.asarray(np.asarray(a["mu_fit"][c], float))
        M = cov(mu)
        dM = jnp.moveaxis(jax.jacfwd(cov)(mu), -1, 0)                         # (k, n, n)
        X = jnp.stack([jnp.linalg.solve(M, d) for d in dM])                  # M^-1 dM_k
        F = jnp.real(jnp.einsum("kab,lba->kl", X, X))
        blocks.append(np.asarray(F))
    return block_diag(blocks)


def block_diag(blocks) -> np.ndarray:
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        out[i:i + k, i:i + k] = b
        i += k
    return out


def initial_covariance(prep: Any, model: str, fisher: SourceFisher) -> np.ndarray:
    """Block-diagonal inverse Fisher at the injection and the fitted noise, each block
    regularised by its prior box (``12 / width^2``; the noise box is 16 wide)."""
    a = prepared_arrays(prep)
    width = np.asarray(a["prior_hi"], float) - np.asarray(a["prior_lo"], float)
    Fs = fisher(np.asarray(a["inject_glass"], float)) + np.diag(12.0 / width ** 2)
    Fn = noise_fisher(prep, model)
    Fn = Fn + np.diag(np.full(Fn.shape[0], 12.0 / 16.0 ** 2))
    cov = block_diag([np.linalg.inv(Fs), np.linalg.inv(Fn)])
    return 0.5 * (cov + cov.T)
