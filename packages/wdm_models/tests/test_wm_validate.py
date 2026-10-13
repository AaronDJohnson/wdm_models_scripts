"""wdm_models.sampling.validate: the golden-comparison rules, fail-closed (spec sections 6,
7.3).  Synthetic inputs only; the real goldens are tests/golden/test_s3_*.py."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from wdm_models.sampling.validate import (COUNTER_KEYS, Gate, ValidationError, build_fingerprint, check_counters,
                                          check_reference, compare, evaluate, jsonable, same_build,
                                          wrap_periodic, write_report)

NEAR = np.array([True, True, False, False])


def test_gate_near_is_atol_and_stress_adds_rtol():
    g = Gate(1e-9, 1e-12)
    ref = np.array([-100.0, -1e5, -1e5, -np.inf])
    assert g.allowed(ref, NEAR).tolist() == [1e-9, 1e-9, 1e-9 + 1e-12 * 1e5, 1e-9]
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValidationError):
            Gate(bad)
        with pytest.raises(ValidationError):
            Gate(1e-9, bad)


def test_compare_passes_inside_the_gate_and_matching_neg_inf():
    ref = np.array([-10.0, -20.0, -3e4, -np.inf])
    got = ref + np.array([5e-10, -5e-10, 2e-8, 0.0])
    got[3] = -np.inf
    c = compare(got, ref, NEAR, Gate(1e-9, 1e-12))
    assert c["ok"] and c["inf_mismatch"] == 0 and c["invalid"] == 0
    assert c["abs_delta"][3] == 0.0 and c["passed"].all()
    assert c["max_abs"] == pytest.approx(2e-8, rel=1e-3)          # -3e4 + 2e-8 rounds


@pytest.mark.parametrize("bad,what", [(np.nan, "invalid"), (np.inf, "invalid"), (-np.inf, "inf_mismatch")])
def test_compare_never_passes_nan_inf_or_a_mask_change(bad, what):
    ref = np.array([-10.0, -20.0, -30.0, -40.0])
    got = ref.copy()
    got[1] = bad
    c = compare(got, ref, NEAR, Gate(1.0, 1.0))
    assert not c["ok"] and c[what] == 1 and c["abs_delta"][1] == np.inf and c["max_abs"] == np.inf


def test_compare_a_finite_value_at_a_neg_inf_reference_fails():
    ref = np.array([-10.0, -np.inf])
    c = compare(np.array([-10.0, -5.0]), ref, np.array([True, False]), Gate(1e9))
    assert not c["ok"] and c["inf_mismatch"] == 1


def test_compare_exact_demands_equality():
    ref = np.array([-10.0, -20.0])
    assert compare(ref.copy(), ref, np.array([True, False]), Gate(1.0), exact=True)["ok"]
    c = compare(ref + [0.0, 1e-12], ref, np.array([True, False]), Gate(1.0), exact=True)
    assert not c["ok"] and c["allowed"].tolist() == [0.0, 0.0]


def test_compare_refuses_empty_and_misshapen_input():
    with pytest.raises(ValidationError, match="nothing to compare"):
        compare(np.zeros(0), np.zeros(0), np.zeros(0, bool), Gate(1.0))
    with pytest.raises(ValidationError):
        compare(np.zeros(3), np.zeros(4), np.zeros(4, bool), Gate(1.0))


def test_check_reference_refuses_what_cannot_be_validated():
    theta, ll, near = np.zeros((2, 3)), np.array([-1.0, -np.inf]), np.array([True, False])
    check_reference(theta, ll, near, 3, what="ok")
    for args, msg in (((np.zeros((0, 3)), np.zeros(0), np.zeros(0, bool), 3), "no points"),
                      ((np.zeros((2, 4)), ll, near, 3), "shapes"),
                      ((np.full((2, 3), np.nan), ll, near, 3), "non-finite"),
                      ((theta, np.array([-1.0, np.nan]), near, 3), "NaN or \\+inf"),
                      ((theta, np.array([-np.inf, -np.inf]), near, 3), "every reference")):
        with pytest.raises(ValidationError, match=msg):
            check_reference(*args, what="t")


def test_wrap_periodic_is_the_runner_arithmetic():
    th = np.array([[0.3, 137.0, 3.5, 2.6], [0.3, 137.0, -np.pi, 2.6], [0.3, 137.0, np.pi, 2.6]])
    out = wrap_periodic(th, {2: (-np.pi, np.pi)})
    lo, hi = -np.pi, np.pi
    assert np.array_equal(out[:, 2], lo + np.mod(th[:, 2] - lo, hi - lo))      # gpu/run.py::_wrap_phase
    assert out[0, 2] == pytest.approx(3.5 - 2 * np.pi) and out[1, 2] == -np.pi and out[2, 2] == -np.pi
    assert th[0, 2] == 3.5 and np.array_equal(out[:, [0, 1, 3]], th[:, [0, 1, 3]])


class Stub:
    """A likelihood with the domainia surface: __call__, last_failed, counters, reset_counters."""

    ndim = 2

    def __init__(self, values, *, failed=None, counters=None):
        self.values, self.failed, self.fixed = np.asarray(values, float), failed, counters
        self.reset_counters()

    def reset_counters(self):
        self.c = {k: 0 for k in COUNTER_KEYS}

    def counters(self):
        return dict(self.fixed) if self.fixed is not None else dict(self.c, cache_hits=0)

    def __call__(self, Z):
        n = len(Z)
        self.c["evaluations"] += n
        self.c["noise_states"] += n
        bad = int(np.sum(self.values == -np.inf))
        self.c["pd_failed_rows"] += bad
        self.c["pd_failed_states"] += bad
        self.last_failed = (self.values == -np.inf) if self.failed is None else np.asarray(self.failed)
        return self.values


def test_evaluate_and_counters_consistent():
    ref = np.array([-1.0, -np.inf, -3.0])
    lik = Stub(ref)
    lik.c["evaluations"] = 99                                            # stale: evaluate() resets
    res = evaluate(lik, np.zeros((3, 2)))
    assert res["counters"]["evaluations"] == 3 and res["seconds"] >= 0.0
    assert check_counters(res, ref, noise_states=3) == []
    assert check_counters(res, ref, noise_states=None) == []


def test_counters_and_last_failed_inconsistencies_are_reported():
    ref = np.array([-1.0, -np.inf, -3.0])
    res = evaluate(Stub(ref, failed=[False, False, False]), np.zeros((3, 2)))
    assert any("last_failed" in p for p in check_counters(res, ref, noise_states=3))
    fixed = {"noise_states": 3, "pd_failed_states": 0, "pd_failed_rows": 0, "evaluations": 3}
    res = evaluate(Stub(ref, counters=fixed), np.zeros((3, 2)))
    assert check_counters(res, ref, noise_states=3) == ["counter pd_failed_rows = 0, expected 1",
                                                        "counter pd_failed_states = 0, expected 1"]
    res = evaluate(Stub(ref), np.zeros((3, 2)))
    assert check_counters(res, ref, noise_states=0) == ["counter noise_states = 3, expected 0"]
    res = evaluate(Stub(ref, counters={"evaluations": 3}), np.zeros((3, 2)))
    assert check_counters(res, ref, noise_states=None)[-1].startswith("counters missing")


def test_evaluate_refuses_a_wrong_shape():
    with pytest.raises(ValidationError, match="shape"):
        evaluate(Stub(np.zeros(2)), np.zeros((3, 2)))


def test_fingerprint_and_same_build():
    fp = build_fingerprint("numpy")
    for key in ("python", "machine", "numpy", "scipy", "blas", "threads", "packages", "backend"):
        assert key in fp
    assert same_build(fp, dict(fp))
    assert not same_build(fp, dict(fp, numpy="0.0"))
    assert not same_build({}, {})                                       # missing keys never match


def test_report_is_strict_json(tmp_path):
    rep = {"a": np.array([-np.inf, 1.0]), "b": np.float64(np.nan), "c": np.bool_(True), "d": (1, 2)}
    assert jsonable(rep) == {"a": ["-inf", 1.0], "b": "nan", "c": True, "d": [1, 2]}
    write_report(tmp_path / "sub" / "r.json", rep)
    assert json.loads((tmp_path / "sub" / "r.json").read_text())["a"] == ["-inf", 1.0]
    with pytest.raises(TypeError):
        jsonable(object())
