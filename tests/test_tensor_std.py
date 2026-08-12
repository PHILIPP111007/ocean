from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_standard_tensor_cpu_facade_runs(tmp_path):
    source = tmp_path / "tensor_example.oc"
    source.write_text(
        """
import "std/tensor/tensor.oc"

def main() -> int:
    var left: Tensor = Tensor.zeros(2, 3, "cpu")
    var right: Tensor = Tensor.zeros(3, 2, "cpu")
    var result: Tensor = left.matmul(right)
    var copy: Tensor = result.to("cpu")
    print(copy.shape(0))
    print(copy.shape(1))
    print(copy.device())
    return 0
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "tensor.json"
    c_path = tmp_path / "tensor.generated.c"
    binary_path = tmp_path / "tensor"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]),
        source,
        json_path,
        c_path,
        quiet=True,
    )
    compile_c(c_path, binary_path)
    result = subprocess.run(
        [str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["2", "2", "cpu"]
