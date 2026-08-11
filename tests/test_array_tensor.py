from tests.constants import base_path
from src.parser import Parser
from src.compiler import CCodeGenerator

def generate(source: str) -> str:
    data = Parser(base_path=base_path).parse_code(source)
    return CCodeGenerator().generate_from_json(data)


def test_array_lowering_and_index_mutation():
    code = generate(
        """
def main() -> int:
    var values: array[float32] = [1.0, 2.0, 3.0]
    values[1] = values[0] + 2.0
    print(values[1])
    return 0
"""
    )

    assert "typedef struct ocean_array_float32" in code
    assert "ocean_array_float32_create" in code
    assert "ocean_array_float32_get" in code
    assert "ocean_array_float32_set" in code
    assert "ocean_array_float32_free(values);" in code
    assert "#include <stddef.h>" in code
    assert "ocean_list_" not in code


def test_tensor_lowering_shape_and_index_mutation():
    code = generate(
        """
def main() -> int:
    var matrix: tensor[float32] = [[1.0, 2.0], [3.0, 4.0]]
    var value: float32 = matrix[0, 1]
    matrix[1, 0] = value
    var elements: int = len(matrix)
    var rows: int = matrix.shape[0]
    print(matrix[1, 0])
    return 0
"""
    )

    assert "typedef struct ocean_tensor_float32" in code
    assert "ocean_tensor_float32_create" in code
    assert "ocean_tensor_float32_get" in code
    assert "ocean_tensor_float32_set" in code
    assert "ocean_tensor_float32_shape_at" in code
    assert "ocean_tensor_float32_free(matrix);" in code
    assert "ocean_tensor_float32_get(matrix, (size_t[])" in code





def test_tensor_the_big_code():
    P = r"""
# Большой пример работы с array и tensor в языке Ocean.
#
# В примере показаны:
# - owned array и tensor;
# - immutable и mutable borrow;
# - поэлементное изменение array;
# - матричное умножение для tensor;
# - трехмерный tensor и доступ по нескольким индексам;
# - shape, ndim, size и len.

def scale_array(values: &mut array[float32], factor: float32) -> None:
    var count: int = len(values)

    for i in range(count):
        values[i] = values[i] * factor

    return None


def dot_product(left: &array[float32], right: &array[float32]) -> float32:
    var count: int = len(left)
    var result: float32 = 0.0

    for i in range(count):
        result = result + left[i] * right[i]

    return result


def tensor_sum(values: &tensor[float32]) -> float32:
    var total: float32 = 0.0
    var rows: int = values.shape[0]
    var cols: int = values.shape[1]

    for i in range(rows):
        for j in range(cols):
            total = total + values[i, j]

    return total


def matmul(left: &tensor[float32], right: &tensor[float32], result: &mut tensor[float32]) -> None:
    var rows: int = left.shape[0]
    var shared: int = left.shape[1]
    var cols: int = right.shape[1]

    for i in range(rows):
        for j in range(cols):
            var value: float32 = 0.0

            for k in range(shared):
                value = value + left[i, k] * right[k, j]

            result[i, j] = value

    return None


def add_bias(matrix: &mut tensor[float32], bias: &array[float32]) -> None:
    var rows: int = matrix.shape[0]
    var cols: int = matrix.shape[1]

    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = matrix[i, j] + bias[j]

    return None


def main() -> int:
    # Одномерные owned arrays.
    var input: array[float32] = [1.0, 2.0, 3.0]
    var weights: array[float32] = [0.5, 1.0, 1.5]

    scale_array(input, 2.0)

    var score: float32 = dot_product(input, weights)
    print("dot product =", score)

    # Двумерные tensors хранятся в contiguous row-major buffer.
    var left: tensor[float32] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    var right: tensor[float32] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    var product: tensor[float32] = [[0.0, 0.0], [0.0, 0.0]]

    matmul(left, right, product)

    var bias: array[float32] = [0.25, 0.5]
    add_bias(product, bias)

    var first_row: int = product.shape[0]
    var first_col: int = product.shape[1]
    var product_size: int = len(product)
    var product_sum: float32 = tensor_sum(product)

    print("product rows =", first_row)
    print("product cols =", first_col)
    print("product elements =", product_size)
    print("product sum =", product_sum)
    print("product[0, 0] =", product[0, 0])
    print("product[1, 1] =", product[1, 1])

    # Трехмерный tensor: batch x height x width.
    var batch: tensor[float32] = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]

    var batches: int = batch.shape[0]
    var height: int = batch.shape[1]
    var width: int = batch.shape[2]
    var batch_elements: int = len(batch)
    var depth: int = batch.ndim
    var total_size: int = batch.size

    print("batch ndim =", depth)
    print("batch size =", total_size)
    print("batch shape =", batches, height, width)
    print("batch elements =", batch_elements)
    print("batch[1, 0, 1] =", batch[1, 0, 1])

    return 0
"""

    code = generate(P)

    assert "typedef struct ocean_array_float32" in code
    assert "typedef struct ocean_tensor_float32" in code
    assert "ocean_array_float32_create" in code
    assert "ocean_tensor_float32_create" in code
    assert "ocean_tensor_float32_get" in code
    assert "ocean_tensor_float32_set" in code
    assert "ocean_tensor_float32_shape_at" in code
    assert "ocean_tensor_float32_free(batch);" in code
    assert "ocean_array_float32_free(input);" in code

def test_tensor_zeros_dynamic_shape():
    code = generate(
        """
def main() -> int:
    var rows: int = 2
    var cols: int = 3
    var matrix: tensor[float32] = tensor.zeros(rows, cols)
    matrix[1, 2] = 7.5
    var elements: int = len(matrix)
    print(elements)
    print(matrix[1, 2])
    return 0
"""
    )

    assert "ocean_tensor_float32_zeros" in code
    assert "size_t ocean_tensor_matrix_" in code
    assert "ocean_tensor_float32_set(matrix" in code
    assert "ocean_tensor_float32_free(matrix);" in code
