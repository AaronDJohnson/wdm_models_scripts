# wdm-paper (import name `wdm`)

The Wilson-Daubechies-Meyer (WDM) transform as used by WDM Paper I and II:

* the fast transforms (`forward*`, `inverse*`; NumPy, or JAX through `wdm.backend`),
* the Paper I likelihoods (`wdm.inner_product`) and stationary covariance (`wdm.stationary`),
* the literal Paper I algorithms 1-4 (`wdm.paper`),
* the stateless WDM II kernels (`wdm.evolutionary`: wrap rule, exact covariance, Eq. 45 builders).

Release sentinels: `wdm.__version__ == "1.0.0"`, `wdm.__wdm_api__ == "functions/1"`,
`wdm.__release__ == "wdm-paper-1"`.  Seeded from lisa/wdm@2f7136c (as vendored in
`toy_models/vendor/wdm`) plus `wdm_dev/src/wdm/evolutionary.py`, with the lisa/wdm
b5973b7 fixes cherry-picked.  Tests: `tests/` (the frozen oracle lives in `tests/oracle`).

Run the tests from this directory with the release environment:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu \
        ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -p no:cacheprovider -q
