# WDM Paper I + II code release

The release has four parts:

| part | what | where |
|---|---|---|
| `wdm` (distribution `wdm-paper`) | Paper I transforms and PSD <-> Lambda algorithms, stateless Paper II kernels | `packages/wdm` |
| `domainia` | domains, linear maps, covariance operators, edge combinators, likelihoods | `packages/domainia` |
| `wdm_models` | the sinusoid and galactic-binary toy models, model registry, sampling glue | `packages/wdm_models` |
| `wdm_models_scripts` | this repository: runners, golden harness, shipped data, goldens, tests | `scripts/`, `data/`, `golden/`, `tests/` |

Status (v0.10.1): `wdm`, `domainia`, the golden harness and both toy models are complete. The sinusoid
(15 models x 3 injections) and the galactic binary (15 models at 3/6/12 months, binary-galactica 0.5.0)
reproduce the papers' stored likelihood values on NumPy (bitwise where the frozen code is bitwise) and on
CPU-JAX at the approved gates. GPU sampling runners, the statistical comparison with the papers' chains,
and the remaining documentation follow in v1.0.0.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23347526.svg)](https://doi.org/10.5281/zenodo.23347526)

License: MIT (see `LICENSE`). Citation: see `CITATION.cff`.

Environment: `conda env create -f env/environment-cpu.yml -n wdmrel`, then `source env/activate.sh`
from this directory; the galactic binary's waveform package, binary-galactica (JAX-only, PyPI release 0.5.0),
goes on top at its pinned release: `$PY -m pip install --no-deps --force-reinstall -r env/requirements-bg.txt`.
The GLASS foreground MAP (and, from v1.0.0, the sampling runners) needs impulse v2, the
`wdm-models[sampling]` extra (`impulse-mcmc>=2`); it is not on PyPI, whose impulse-mcmc 1.0.0 is the
pre-rewrite API and is refused: `$PY -m pip install -c env/constraints.txt "impulse-mcmc>=2"` installs
commit 964283c from git.
Tests: `$PY -m pytest` (fast tier; `-m golden_full` / `-m slow` are opt-in).

## GPU validation of the sinusoid (user-run; sub-project S4)

The JAX likelihoods are gated locally on CPU-JAX (`$PY -m scripts.golden.s4_report --full`).
The GPU strategy column (exact-row chunk 192, noise-state LRU 16, the rows of one call
batched, unstaged exact rows) is validated by the user on a Colab GPU runtime. Nothing here
samples; the run evaluates the G1/G2 goldens only (45 targets, a few minutes).

1. Locally, from this directory: `git archive --format=zip -o "$TMPDIR/wdm_models_scripts_s4.zip" HEAD`,
   then upload the zip to `MyDrive/wdm_release/` on Google Drive.
2. In a Colab notebook with a GPU runtime, one cell each:

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   ```
   !rm -rf /content/wdm_models_scripts && mkdir -p /content/wdm_models_scripts && cd /content/wdm_models_scripts && unzip -q /content/drive/MyDrive/wdm_release/wdm_models_scripts_s4.zip
   !pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt
   !pip install -q -r /content/wdm_models_scripts/env/requirements-colab-cuda12.txt
   ```
   ```
   %%bash
   cd /content/wdm_models_scripts
   export PYTHONPATH=$PWD:$PWD/packages/wdm/src:$PWD/packages/domainia/src:$PWD/packages/wdm_models/src
   export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_FLAGS=--xla_gpu_deterministic_ops=true
   export JAX_PLATFORMS=cuda
   python -P -m scripts.sinusoid.validate --backend jax --tier fast --report /content/drive/MyDrive/wdm_release/validate_sinusoid_jax_gpu_fast.json
   ```
   Exit status 0 and `"ok": true` in the report mean every target passed its approved JAX gate.
3. Download the report and summarise it locally next to the CPU gates:
   `$PY -m scripts.golden.s4_report --validate ~/Downloads/validate_sinusoid_jax_gpu_fast.json --out "$TMPDIR/s4_gpu.json"`
   (its `run_meta` must read `"device_kind": "gpu"` and `"deterministic_ops": true`; `--validate`
   refuses a report from any other device, a CPU fallback included -- the cell's
   `JAX_PLATFORMS=cuda` makes a runtime without a usable GPU fail instead of falling back).


## Validating the galactic binary on NumPy (sub-project S6)

The 38 galactic-binary targets (`wdm_models.registry.TARGETS["galactic_binary"]`) are gated
against the frozen lineage-C goldens; nothing here samples. From this directory, after
`source env/activate.sh`:

* `$PY -P -m scripts.galactic_binary.validate --tier fast --report "$TMPDIR/gb_validate.json"` --
  G3 (174 stored chain rows), G4 (the stress points of the 30 three- and six-month targets),
  G5 (data products) and G6 (the anchors); about 1.5 min. The golden checks use
  binary_galactica `fourier` (the chains' gbonfly templates); the release default `onfly` is
  audited, not gated: `--golden audit`.
* `--templates gbonfly` swaps in gbonfly templates streamed from the frozen lineage C (needs the
  `wdm_test` workspace): the likelihood numerics alone, bitwise against lineage C on the
  recording build.
* The S6 gate: `$PY -m scripts.golden.s6_report --full --strict --out "$TMPDIR/s6.json"`
  (every G3/G4/G5/G6 item, the onfly audit, and the S2 gate; about 9 min).
* The full tier (32 rows per run directory, 12 stress points) needs the user's approval: its
  goldens are not captured yet (`tests/golden/test_s6_full_tier.py` lists the commands).

## GPU validation of the galactic binary (user-run; sub-project S7)

The GB JAX likelihoods are gated locally on CPU-JAX (`$PY -m scripts.golden.s7_report --full`).
Their strategies are the same on every device (lineage C's `gpu/jax_models.build`: the cache
classes batch the rows of one call everywhere, `exact` loops), so the GPU run checks XLA's GPU
arithmetic against the same approved gates (G3 1e-8; G4 8.32e-8 at well-conditioned points,
identical -inf masks). Nothing here samples; the run evaluates the G3 rows and G4 stress points
only (38 targets, about a minute after compilation).

1. Locally, from this directory: `git archive --format=zip -o "$TMPDIR/wdm_models_scripts_s7.zip" HEAD`,
   then upload the zip to `MyDrive/wdm_release/` on Google Drive.
2. In a Colab notebook with a GPU runtime, one cell each:

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   ```
   !rm -rf /content/wdm_models_scripts && mkdir -p /content/wdm_models_scripts && cd /content/wdm_models_scripts && unzip -q /content/drive/MyDrive/wdm_release/wdm_models_scripts_s7.zip
   !pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt
   !pip install -q -r /content/wdm_models_scripts/env/requirements-colab-cuda12.txt
   !pip install -q --no-deps -r /content/wdm_models_scripts/env/requirements-bg.txt
   ```
   ```
   %%bash
   cd /content/wdm_models_scripts
   export PYTHONPATH=$PWD:$PWD/packages/wdm/src:$PWD/packages/domainia/src:$PWD/packages/wdm_models/src
   export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_FLAGS=--xla_gpu_deterministic_ops=true
   export JAX_PLATFORMS=cuda
   python -P -m scripts.galactic_binary.validate --backend jax --tier fast --golden g3 g4 --report /content/drive/MyDrive/wdm_release/validate_gb_jax_gpu_fast.json
   ```
   Exit status 0 and `"ok": true` in the report mean every target passed its approved JAX gate
   (binary-galactica, the PyPI release 0.5.0, is installed with `--no-deps` on top of the CUDA-12
   JAX pin: it is JAX-only, and its `jax` requirement must not pull another JAX).
3. Download the report and summarise it locally next to the CPU gates:
   `$PY -m scripts.golden.s7_report --validate ~/Downloads/validate_gb_jax_gpu_fast.json --out "$TMPDIR/s7_gpu.json"`
   (its `run_meta` must read `"device_kind": "gpu"` and `"deterministic_ops": true`; `--validate`
   refuses a report from any other device, a CPU fallback included -- the cell's
   `JAX_PLATFORMS=cuda` makes a runtime without a usable GPU fail instead of falling back).

A G4 failure **only** at `gb_6mo/exact` `edge_h1` is conditioning, not a defect, and is accepted in
advance: that stress point's PD margin is 1.86e-6, just above the 1e-6 well-conditioned cut, so it
is gated at 8.32e-8 although its roundoff bound (eps |lnL| / margin) is about 10.7x that gate; CPU-JAX
sits at 31% of the gate (2.56e-8). A failure at any other point, or at this one by more than its
roundoff bound (about 8.9e-7), is a defect.
