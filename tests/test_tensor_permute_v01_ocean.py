from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_tensor_permute_v01_ocean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/tensor_permute_v01.oc"
    c_path = tmp_path / "tensor_permute_v01.generated.c"
    binary = tmp_path / "tensor_permute_v01"

    compile_pipeline(
        source.parent,
        source,
        c_path,
        quiet=True,
    )
    compile_c(c_path, binary)

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "y shape = 2 2 3 2" in result.stdout
    assert "x has grad = 1" in result.stdout.lower()
    assert "[ok] Ocean Tensor.permute v0.1" in result.stdout
