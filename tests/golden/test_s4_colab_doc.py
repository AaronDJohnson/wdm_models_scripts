"""E-D9: the user-run Colab GPU validation is documented once, consistently (README,
validate.py's ``COLAB`` text, the CUDA-12 pins); nothing here runs on a GPU."""

from __future__ import annotations

from scripts.golden.paths import REL

VALIDATE = ("python -P -m scripts.sinusoid.validate --backend jax --tier fast "
            "--report /content/drive/MyDrive/wdm_release/validate_sinusoid_jax_gpu_fast.json")


def test_cuda12_pins_match_the_cpu_env():
    lines = [ln.strip() for ln in (REL / "env" / "requirements-colab-cuda12.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert lines == ["jax[cuda12]==0.9.0.1", "numpy==2.3.5", "scipy==1.16.3", "pytest==8.4.2"]
    pip = (REL / "env" / "requirements-pip.txt").read_text().splitlines()
    assert {"jax==0.9.0.1", "numpy==2.3.5", "scipy==1.16.3", "pytest==8.4.2"} <= {ln.strip() for ln in pip}


def test_readme_and_validate_name_the_same_gpu_command():
    from scripts.sinusoid import validate

    readme = (REL / "README.md").read_text()
    colab = " ".join(validate.COLAB.replace("\\\n", " ").split())
    for text in (" ".join(readme.split()), colab):
        assert VALIDATE in text
        assert "pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt" in text
        assert "XLA_FLAGS=--xla_gpu_deterministic_ops=true" in text
        assert "JAX_ENABLE_X64=true" in text
        assert "PYTHONPATH=$PWD:$PWD/packages/wdm/src:$PWD/packages/domainia/src:$PWD/packages/wdm_models/src" in text
    assert "scripts.golden.s4_report --validate" in readme
