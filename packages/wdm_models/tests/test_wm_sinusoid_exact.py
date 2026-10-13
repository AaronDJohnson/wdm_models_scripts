"""E-A7: the exact model's fitted assets (c_N, the bandcov interpolant) and their composition
(spec §5.1 exact.py), against the shipped artifacts and the G6 intermediates at the truth."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from domainia.assets import ArtifactError, ArtifactMissing, identity_key
from domainia.covariance import ExactTimeCovariance
from domainia.nuisance import ProjectedOperatorMarginal
from wdm_models.sinusoid import config as C
from wdm_models.sinusoid import exact as E

DATA = Path(__file__).resolve().parents[3] / "data"
CACHE = DATA / "sinusoid" / "cache"
NU = (-2.0, 1.2)                       # the truth's (alpha_c, alpha_d)
pytestmark = pytest.mark.golden


def _same(got: float, want: float, exact: bool) -> None:
    assert got == want if exact else abs(got - want) <= 1e-12 * abs(want), (got, want)


def test_the_c_N_asset(g6, recording_build):
    cfg = C.config("manuscript_mid")
    assert E.cn_filename(cfg) == "cN_N16384_dt0.0009765625_fmin4_fref1.npy"
    ld = E.ExactLogDet.load(cfg)
    assert ld.c_N == 0.22629082684418084 == g6["parts"]["exact"]["c_N"]
    manifest = json.loads((DATA / "MANIFEST.json").read_text())["files"]
    assert manifest[f"sinusoid/cache/{E.cn_filename(cfg)}"] == E.CN_SHA256
    _same(ld.weyl(NU), g6["parts"]["exact"]["weyl"], recording_build)
    _same(ld(NU), g6["parts"]["exact"]["full_logdet"], recording_build)
    cov = ld.covariance()
    assert isinstance(cov, ExactTimeCovariance) and (cov.form, cov.P, cov.tol, cov.c_N) == (
        "operator", 24, 1e-7, ld.c_N)


def test_the_bandcov_identity_is_kept_verbatim(g6):
    ident = E.bandcov_identity(C.config("manuscript_mid"))
    assert ident == {"version": 1, "N": 16384, "dt": 0.0009765625, "f_min": 4.0, "f_ref": 1.0,
                     "analysis_fmin": 4.0, "n_keep": 16256, "n_drop": 128,
                     "support": "real-orthonormal;f>=fmin;f<Nyquist;noDC", "P": 24,
                     "bounds": [[-4.0, 0.0], [-2.0, 2.0]], "drift_transform": "asinh",
                     "drift_scale": 0.003,
                     "algorithm": "latent-C0-PCG-discarded-precision-chebyshev"}
    assert ident == g6["parts"]["exact"]["bandcov_identity"]
    with np.load(CACHE / "bandcov_v1_4bfb466c191c79f9879a.npz", allow_pickle=False) as z:
        assert ident == json.loads(str(z["metadata"]))["identity"]
    assert identity_key(ident) == E.BANDCOV_KEY == "4bfb466c191c79f9879a" == g6["parts"]["exact"]["bandcov_key"]
    assert E.bandcov_filename(C.config("manuscript_mid")) == "bandcov_v1_4bfb466c191c79f9879a.npz"


def test_the_correction_at_the_truth_and_at_the_validation_points(g6, recording_build):
    corr = E.BandCorrection.load(C.config("manuscript_mid"))
    assert corr.path.name == "bandcov_v1_4bfb466c191c79f9879a.npz"
    assert corr.identity == E.bandcov_identity(C.config("manuscript_mid"))
    assert (corr.drift_scale, corr.bounds.tolist(), corr.c_N) == (0.003, [[-4.0, 0.0], [-2.0, 2.0]],
                                                                  0.22629082684418084)
    ex = g6["parts"]["exact"]
    # G6 stores the correction as the difference logdet_unit - full_logdet (sinusoid_wt.py
    # _exact_parts), which rounds: the interpolant itself is pinned through the sum.
    assert abs(float(corr(np.array(NU))) - ex["correction"]) <= 1e-14 * abs(ex["full_logdet"])
    _same(float(E.ExactLogDet.load()(NU) + corr(np.array(NU))), ex["logdet_unit"], recording_build)
    with np.load(corr.path, allow_pickle=False) as z:
        points, exact = np.array(z["validation_points"]), np.array(z["validation_exact"])
    got = np.array([corr(p) for p in points])
    assert np.max(np.abs(got - exact)) <= corr.metadata["validation_tol"] == 2e-5


def test_the_correction_refuses_noise_outside_the_validated_box():
    corr = E.BandCorrection.load(C.config("manuscript_mid"))
    for nu in ((-4.1, 0.0), (0.5, 0.0), (-2.0, 2.01), (-2.0, np.nan)):
        with pytest.raises(ValueError, match="outside the validated"):
            corr(np.array(nu))
    assert np.isfinite(corr(np.array([-4.0, -2.0]))) and np.isfinite(corr(np.array([0.0, 2.0])))


def test_the_correction_is_numpy_only_until_s4(jax_be):
    corr = E.BandCorrection.load(C.config("manuscript_mid"))
    with pytest.raises(NotImplementedError, match="S4"):
        corr(jax_be.asarray(np.array(NU)), jax_be)


def test_the_artifact_checks_port_wt_load():
    """WT ``BandCovariance._load``: identity, validated flag, solver tolerance, finite c_N,
    validation tolerance, coefficient shape, independent validation points, re-validation."""
    ident = E.bandcov_identity(C.config("manuscript_mid"))
    with np.load(CACHE / "bandcov_v1_4bfb466c191c79f9879a.npz", allow_pickle=False) as z:
        good = {k: np.array(z[k]) for k in z.files}
    E.check_bandcov(good, ident)

    def with_meta(**change):
        meta = json.loads(str(good["metadata"]))
        meta.update(change)
        return {**good, "metadata": np.asarray(json.dumps(meta, sort_keys=True))}

    for arrays, why in (
            (with_meta(validated=False), "not validated"),
            (with_meta(solver_tol=1e-8), "solver"),
            (with_meta(c_N=float("nan")), "c_N"),
            (with_meta(validation_tol=1e-4), "validation tolerance"),
            (with_meta(degree=[40, 63]), "degree"),
            (with_meta(max_error=3e-5), "failed validation"),
            ({**good, "validation_points": good["validation_points"][:8],
              "validation_exact": good["validation_exact"][:8]}, "validation points"),
            ({**good, "validation_exact": good["validation_exact"] + 1e-3}, "failed validation"),
            ({k: v for k, v in good.items() if k != "coefficients"}, "lacks")):
        with pytest.raises(ArtifactError, match=why):
            E.check_bandcov(arrays, ident)
    with pytest.raises(ArtifactError, match="identity"):
        E.check_bandcov(good, {**ident, "P": 23})


def test_the_artifacts_are_fail_closed_on_the_files(tmp_path):
    shutil.copytree(CACHE, tmp_path / "sinusoid" / "cache")
    shutil.copy(DATA / "MANIFEST.json", tmp_path / "MANIFEST.json")
    cfg = C.config("manuscript_mid")
    E.ProjectedExact.load(cfg, data_root=tmp_path)
    band = tmp_path / "sinusoid" / "cache" / "bandcov_v1_4bfb466c191c79f9879a.npz"
    body = bytearray(band.read_bytes())
    body[200] ^= 1
    band.write_bytes(bytes(body))
    with pytest.raises(ArtifactError, match="sha256"):
        E.BandCorrection.load(cfg, data_root=tmp_path)
    (tmp_path / "sinusoid" / "cache" / E.cn_filename(cfg)).unlink()
    with pytest.raises(ArtifactMissing):
        E.ExactLogDet.load(cfg, data_root=tmp_path)
    with pytest.raises(ArtifactError, match="not in"):         # another identity -> another key
        E.BandCorrection.load(cfg, data_root=tmp_path, P=23)


def test_the_projected_exact_composition(g6, recording_build):
    pe = E.ProjectedExact.load(C.config("manuscript_mid"))
    cov = pe.covariance()
    assert isinstance(cov, ProjectedOperatorMarginal)
    assert (cov.n, cov.n_u, cov.nuisance.n_keep) == (16384, 128, 16256)
    assert (cov.op.P, cov.op.tol, cov.op.on_fail) == (24, 1e-7, "raise")
    F = cov.factor(np.array(NU))
    assert bool(F.ok)
    _same(float(F.logdet), g6["parts"]["exact"]["logdet_unit"], recording_build)
    assert pe.covariance(on_fail="nan").op.on_fail == "nan"
    off = dataclasses.replace(pe.logdet, c_N=pe.logdet.c_N + 1e-9)
    with pytest.raises(ArtifactError, match="c_N"):
        E.ProjectedExact(off, pe.correction, pe.nuisance)
