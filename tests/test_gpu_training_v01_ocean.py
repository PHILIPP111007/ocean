from pathlib import Path
import math
import re
import shutil
import subprocess

import pytest

from main import compile_c, compile_pipeline


ROOT = Path(__file__).resolve().parents[1]


def opencl_available():
    if shutil.which("pkg-config") is None:
        return False
    probe = subprocess.run(["pkg-config", "--exists", "OpenCL"], check=False)
    if probe.returncode != 0:
        return False
    if shutil.which("clinfo"):
        info = subprocess.run(
            ["clinfo", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
        if info.returncode != 0:
            return False
        if "device" not in (info.stdout + info.stderr).lower():
            return False
    return True


@pytest.mark.skipif(
    not opencl_available(),
    reason="OpenCL development/runtime device is unavailable",
)
def test_gpu_training_v01(tmp_path):
    source = ROOT / "examples/ML/gpu_training_v01.oc"
    generated_c = tmp_path / "gpu_training_v01.c"
    binary = tmp_path / "gpu_training_v01"

    compile_pipeline(source.parent, source, generated_c, quiet=True)
    command = compile_c(generated_c, binary)

    assert "-DOCEAN_TENSOR_ENABLE_OPENCL" in command
    assert "-lOpenCL" in command

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert "model parameter device = gpu" in result.stdout
    assert "prediction device = gpu" in result.stdout
    assert "[ok] Ocean GPU training v0.1" in result.stdout

    initial = re.search(r"initial loss\s*=\s*([-+0-9.eE]+)", result.stdout)
    final = re.search(r"final loss\s*=\s*([-+0-9.eE]+)", result.stdout)

    assert initial
    assert final

    initial_value = float(initial.group(1))
    final_value = float(final.group(1))

    assert math.isfinite(initial_value)
    assert math.isfinite(final_value)
    assert final_value < initial_value * 0.1
