"""The galactic-binary toy of WDM II Sec. VII D (spec section 5.2): ``config``, ``instrument``,
``foreground``, ``envelope``, ``priors``, ``prepared``, ``prepare`` and ``waveform``.
Submodules are imported explicitly (``from wdm_models.galactic_binary import config``); this
package imports none of them, and no module imports JAX or binary_galactica at import time
(``waveform``, ``priors.source_prior`` and ``prepare.fit_mu`` import them when called)."""
