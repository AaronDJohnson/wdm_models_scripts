"""F-D10: the user-run Colab GPU validation of the galactic binary is documented once, consistently
(README, ``scripts/galactic_binary/validate.py``'s ``COLAB`` text, the CUDA-12 pins and the
binary-galactica pin, the user's PyPI release 0.5.0); nothing here runs on a GPU."""

from __future__ import annotations

from scripts.golden.paths import REL

HEADER = "## GPU validation of the galactic binary (user-run; sub-project S7)"
VALIDATE = ("python -P -m scripts.galactic_binary.validate --backend jax --tier fast --golden g3 g4 "
            "--report /content/drive/MyDrive/wdm_release/validate_gb_jax_gpu_fast.json")
BG = "pip install -q --no-deps -r /content/wdm_models_scripts/env/requirements-bg.txt"


def test_the_bg_pin_is_the_pypi_release():
    lines = [ln.strip() for ln in (REL / "env" / "requirements-bg.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert lines == ["binary-galactica==0.5.0"]


def test_readme_and_validate_name_the_same_gpu_command():
    from scripts.galactic_binary import validate

    readme = (REL / "README.md").read_text()
    assert HEADER in readme
    section = readme.split(HEADER, 1)[1].split("\n## ", 1)[0]          # this section only
    colab = " ".join(validate.COLAB.replace("\\\n", " ").split())
    for text in (" ".join(section.split()), colab):
        assert VALIDATE in text and BG in text
        assert "pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt" in text
        assert "pip install -q -r /content/wdm_models_scripts/env/requirements-colab-cuda12.txt" in text
        assert text.index("requirements-colab-cuda12.txt") < text.index("requirements-bg.txt")   # bg on top of JAX
        assert "XLA_FLAGS=--xla_gpu_deterministic_ops=true" in text
        assert "JAX_ENABLE_X64=true" in text
        assert "export JAX_PLATFORMS=cuda" in text and text.index("export JAX_PLATFORMS=cuda") < text.index(VALIDATE)
        assert "PYTHONPATH=$PWD:$PWD/packages/wdm/src:$PWD/packages/domainia/src:$PWD/packages/wdm_models/src" in text
    assert "scripts.golden.s7_report --validate" in section
    assert '"device_kind": "gpu"' in section and '"deterministic_ops": true' in section
