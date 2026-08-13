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


def test_standard_tensor_operations_and_metadata(tmp_path):
    source = tmp_path / "tensor_operations.oc"
    source.write_text(
        """
import <std/tensor/tensor.oc>

def main() -> int:
    var native_left: tensor[int32] = [[1, 2], [3, 4]]
    var native_right: tensor[int32] = [[5, 6], [7, 8]]
    var native_bias: tensor[int32] = [[10, 20]]
    var left: Tensor[int32] = Tensor[int32].from_tensor(native_left, "cpu")
    var right: Tensor[int32] = Tensor[int32].from_tensor(native_right, "cpu")
    var bias: Tensor[int32] = Tensor[int32].from_tensor(native_bias, "cpu")
    var added: Tensor[int32] = left.add(right)
    var subtracted: Tensor[int32] = right.sub(left)
    var multiplied: Tensor[int32] = left.mul(right)
    var transposed: Tensor[int32] = left.transpose()
    var reshaped: Tensor[int32] = left.reshape(1, 4)
    var scaled: Tensor[int32] = left.mul_scalar(2.0)
    var broadcast: Tensor[int32] = left.add(bias)
    left.fill(9.0)
    var restored_added: tensor[int32] = added.to_tensor()
    var restored_subtracted: tensor[int32] = subtracted.to_tensor()
    var restored_multiplied: tensor[int32] = multiplied.to_tensor()
    var restored_transposed: tensor[int32] = transposed.to_tensor()
    var restored_reshaped: tensor[int32] = reshaped.to_tensor()
    var restored_scaled: tensor[int32] = scaled.to_tensor()
    var restored_broadcast: tensor[int32] = broadcast.to_tensor()
    print(restored_added[1, 0])
    print(restored_subtracted[0, 1])
    print(restored_multiplied[1, 1])
    print(restored_transposed[0, 1])
    print(restored_reshaped.shape[1])
    print(restored_scaled[1, 1])
    print(restored_broadcast[0, 1])
    print(restored_broadcast[1, 0])
    print(left.sum())
    return 0
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "tensor_operations.json"
    c_path = tmp_path / "tensor_operations.generated.c"
    binary_path = tmp_path / "tensor_operations"

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
        "10",
        "4",
        "32",
        "3",
        "4",
        "8",
        "22",
        "13",
        "36.000000",
    ]
