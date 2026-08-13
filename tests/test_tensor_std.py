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
    var left: Tensor[float32] = Tensor.from_tensor(native, "cpu")
    var right: Tensor[float32] = Tensor.zeros(3, 2, "cpu")
    var result: Tensor[float32] = left.matmul(right)
    var copy: Tensor[float32] = result.to("cpu")
    var restored: tensor[float32] = copy.to_tensor()
    var native_int: tensor[int32] = [[[7, 8], [9, 10]], [[11, 12], [13, 14]]]
    var int_tensor: Tensor[int32] = Tensor.from_tensor(native_int, "cpu")
    var int_cube: Tensor[int32] = Tensor.zeros(2, 2, 2, "cpu")
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
    var left: Tensor[int32] = Tensor.from_list([[1, 2], [3, 4]], "cpu")
    var right: Tensor[int32] = Tensor.from_list([[5, 6], [7, 8]], "cpu")
    var bias: Tensor[int32] = Tensor.from_list([[10, 20]], "cpu")
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


def test_standard_tensor_constructor_uses_function_return_dtype(tmp_path):
    source = tmp_path / "tensor_return.oc"
    source.write_text(
        """
import <std/tensor/tensor.oc>

def make_labels() -> Tensor[int32]:
    return Tensor.zeros(1, 2, "cpu")

def main() -> int:
    var labels: Tensor[int32] = make_labels()
    var native: tensor[int32] = labels.to_tensor()
    print(native.shape[1])
    return 0
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "tensor_return.json"
    c_path = tmp_path / "tensor_return.generated.c"
    binary_path = tmp_path / "tensor_return"

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

    assert result.stdout.splitlines() == ["2"]


def test_standard_tensor_multidimensional_augmented_index_assignment(tmp_path):
    source = tmp_path / "tensor_augmented_index.oc"
    source.write_text(
        """
import <std/tensor/tensor.oc>

def main() -> int:
    var value: Tensor[float32] = Tensor.from_list([[1.0, 3.0]], "cpu")
    value[0, 1] *= 2
    print(value[0, 1])
    return 0
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "tensor_augmented_index.json"
    c_path = tmp_path / "tensor_augmented_index.generated.c"
    binary_path = tmp_path / "tensor_augmented_index"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]),
        source,
        json_path,
        c_path,
        quiet=True,
    )
    generated_c = c_path.read_text(encoding="utf-8")
    assert "ocean_tensor_device_native" not in generated_c
    assert "ocean_tensor_float32" not in generated_c
    compile_c(c_path, binary_path)
    result = subprocess.run(
        [str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["6.000000"]


def test_standard_tensor_row_column_and_slice_operations(tmp_path):
    source = tmp_path / "tensor_views.oc"
    source.write_text(
        """
import <std/tensor/tensor.oc>

def main() -> int:
    var matrix: Tensor[float32] = Tensor.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "cpu")
    var row: Tensor[float32] = matrix.row(1)
    var column: Tensor[float32] = matrix.column(1)
    var sliced: Tensor[float32] = matrix.slice(1, 1, 3, 1)
    print(row.ndim())
    print(row.shape(0))
    print(row.sum())
    print(column.shape(0))
    print(column.sum())
    print(sliced.shape(0))
    print(sliced.shape(1))
    print(sliced.sum())
    return 0
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "tensor_views.json"
    c_path = tmp_path / "tensor_views.generated.c"
    binary_path = tmp_path / "tensor_views"

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
        "1",
        "3",
        "15.000000",
        "2",
        "7.000000",
        "2",
        "2",
        "16.000000",
    ]
