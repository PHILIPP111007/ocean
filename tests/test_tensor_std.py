from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_standard_tensor_cpu_facade_runs(tmp_path):
    source = tmp_path / "tensor_example.oc"
    source.write_text(
        """
import <std/tensor/tensor.oc>

def main() -> int:
    var native: tensor[float32] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    var left: Tensor[float32] = Tensor[float32].from_tensor(native, "cpu")
    var right: Tensor[float32] = Tensor[float32].zeros(3, 2, "cpu")
    var result: Tensor[float32] = left.matmul(right)
    var copy: Tensor[float32] = result.to("cpu")
    var restored: tensor[float32] = copy.to_tensor()
    var native_int: tensor[int32] = [[[7, 8], [9, 10]], [[11, 12], [13, 14]]]
    var int_tensor: Tensor[int32] = Tensor[int32].from_tensor(native_int, "cpu")
    var int_cube: Tensor[int32] = Tensor[int32].zeros(2, 2, 2, "cpu")
    var restored_int: tensor[int32] = int_tensor.to_tensor()
    print(copy.shape(0))
    print(copy.shape(1))
    print(copy.device())
    print(restored[0, 0])
    print(int_tensor.ndim())
    print(int_cube.shape(2))
    print(restored_int[1, 0, 1])
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

    assert result.stdout.splitlines() == [
        "2",
        "2",
        "cpu",
        "0.000000",
        "3",
        "2",
        "12",
    ]
