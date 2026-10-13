# domainia

Data on domains, the linear maps between them, covariance operators and Gaussian
likelihoods over the WDM transform: the middle layer of the WDM Paper II release
(`wdm` <- `domainia` <- `wdm_models`).

* `domainia.domains`, `domainia.data`: time / Fourier / WDM-cell domains (static, hashable)
  and the data vectors on them (JAX pytrees).
* `domainia.linalg`: array kernels -- block-tridiagonal (ragged host loop or padded
  `scan`), complex Hermitian block, preconditioned CG; each a port of one legacy oracle
  with its library call pinned.  The factor objects (diagonal, dense, Schur, padded
  marginal) are `domainia.covariance`'s: one family (S4).
* `domainia.assets`: fail-closed artifacts (identity + sha256; never written implicitly).
* `domainia.testing`: dense oracles for tests.

`import domainia` refuses any `wdm` that is not the paper release
(`wdm.__release__ == "wdm-paper-1"`) and never imports JAX.

Run the tests with the release environment (`env/activate.sh` must be sourced from the
release root):

    cd ../.. && source env/activate.sh && cd packages/domainia && $PY -m pytest -q
