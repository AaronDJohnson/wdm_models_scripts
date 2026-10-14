# WDM Paper I + II code release

The release has four parts:

| part | what | where |
|---|---|---|
| `wdm` (distribution `wdm-paper`) | Paper I transforms and PSD <-> Lambda algorithms, stateless Paper II kernels | `packages/wdm` |
| `domainia` | domains, linear maps, covariance operators, edge combinators, likelihoods | `packages/domainia` |
| `wdm_models` | the sinusoid and galactic-binary toy models, model registry, sampling glue | `packages/wdm_models` |
| `wdm_models_scripts` | this repository: runners, golden harness, shipped data, goldens, tests | `scripts/`, `data/`, `golden/`, `tests/` |

Status (v0.9.0, preprint snapshot): `wdm`, `domainia` and the golden harness are complete and
verified against the papers' frozen code; the sinusoid model layer is in progress, and the
galactic-binary model layer, sampling runners and documentation follow in v1.0.0.

License: MIT (see `LICENSE`). Citation: see `CITATION.cff`.

Environment: `conda env create -f env/environment-cpu.yml -n wdmrel`, then `source env/activate.sh`
from this directory; the galactic binary's waveform package, binary-galactica (JAX-only, PyPI release 0.5.0),
goes on top at its pinned release: `$PY -m pip install --no-deps --force-reinstall -r env/requirements-bg.txt`.
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
