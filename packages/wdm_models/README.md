# wdm_models

The WDM Paper II toy models on top of `domainia` (`wdm` <- `domainia` <- `wdm_models`):

* `wdm_models.sinusoid`: the drifting power-law sinusoid of WDM II Sec. VII B --
  configuration and cases (`config`), the signal, the PSD, the single prior table, the
  shipped data (sha256-checked), the exact model's fitted assets (`exact`) and the 15
  likelihood models (`models`).
* `wdm_models.registry`: `MODELS`, `TARGETS`, `build(toy, case, model, ...)`, `describe(...)`.
* `wdm_models.options`: `ModelOptions` and the per-backend historical defaults.

The galactic binary (`wdm_models.galactic_binary`) arrives with sub-project S5.
`import wdm_models` refuses any `wdm` or `domainia` that is not the paper release and never
imports JAX.

Run the tests with the release environment (`env/activate.sh` must be sourced from the
release root):

    cd ../.. && source env/activate.sh && cd packages/wdm_models && $PY -m pytest -q
