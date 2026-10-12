# WDM Paper I + II code release

The release has four parts (design: `docs/superpowers/specs/2026-10-11-wdm-code-release-design.md`
in the parent `wdm_test` workspace):

| part | what | where |
|---|---|---|
| `wdm` (distribution `wdm-paper`) | Paper I transforms and PSD <-> Lambda algorithms, stateless Paper II kernels | `packages/wdm` |
| `domainia` | domains, linear maps, covariance operators, edge combinators, likelihoods | `packages/domainia` |
| `wdm_models` | the sinusoid and galactic-binary toy models, model registry, sampling glue | `packages/wdm_models` |
| `wdm_models_scripts` | this repository: runners, golden harness, shipped data, goldens, tests | `scripts/`, `data/`, `golden/`, `tests/` |

Status: sub-project S0 (golden harness, captured from the frozen legacy code) in progress.

Environment: `conda env create -f env/environment-cpu.yml -n wdmrel`, then `source env/activate.sh`
from this directory. Tests: `$PY -m pytest` (fast tier; `-m golden_full` / `-m slow` are opt-in).
