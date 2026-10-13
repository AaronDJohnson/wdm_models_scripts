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
from this directory. Tests: `$PY -m pytest` (fast tier; `-m golden_full` / `-m slow` are opt-in).
