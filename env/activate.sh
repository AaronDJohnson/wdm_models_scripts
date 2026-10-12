# Release environment variables (spec §1). Source it from the wdm_models_scripts root:
#     source env/activate.sh
# The release uses the conda env `wdmrel` only; never install anything into emri313 or few.
if [ ! -f "$PWD/env/environment-cpu.yml" ]; then
    echo "env/activate.sh: source it from the wdm_models_scripts root" >&2
    return 1
fi
export WDMREL_ROOT="$PWD"
export PY="$HOME/miniconda3/envs/wdmrel/bin/python"
# PYTHONSAFEPATH stops `python -m` from adding the cwd, so the harness packages (scripts/,
# tools/) are put on the path explicitly. Lineage workers REPLACE PYTHONPATH entirely
# (scripts/golden/lineage.py::worker_env), so none of this leaks into a legacy subprocess.
export PYTHONPATH="$WDMREL_ROOT"
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
