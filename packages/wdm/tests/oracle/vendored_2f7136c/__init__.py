"""Frozen oracle: lisa/wdm@2f7136c as vendored in toy_models/vendor/wdm/src/wdm.

``filter``, ``grid``, ``noise``, ``stationary`` and ``transform`` are byte copies
of the frozen tree except that every relative import ``from .x import`` was
rewritten to ``from vendored_2f7136c.x import`` so this package can never be
imported under the name ``wdm``.  Never edit these files; their bytes are pinned
by ``tests/test_package_identity.py`` and the release modules are compared with
them bit for bit (``tests/test_transform_shim.py``, ``tests/test_noise_stationary.py``).
"""
