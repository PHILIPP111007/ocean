from tests.constants import base_path
from src.parser import Parser
from src.compiler import CCodeGenerator
from tests.base import run

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

    C = r"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
// =========================================
// Вспомогательные структуры и функции
// =========================================

typedef struct ocean_array_float32 {
    float* data;
    size_t size;
    size_t capacity;
} ocean_array_float32;

typedef struct ocean_tensor_float32 {
    float* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
} ocean_tensor_float32;

static ocean_array_float32* ocean_array_float32_create(const float* values, size_t size) {
    ocean_array_float32* array = (ocean_array_float32*)calloc(1, sizeof(ocean_array_float32));
    if (!array) { fprintf(stderr, "Ocean allocation error: ocean_array_float32\n"); exit(1); }
    array->size = size;
    array->capacity = size;
    if (size > 0) {
        array->data = (float*)malloc(size * sizeof(float));
        if (!array->data) { free(array); fprintf(stderr, "Ocean allocation error: ocean_array_float32 data\n"); exit(1); }
        memcpy(array->data, values, size * sizeof(float));
    }
    return array;
}
static void ocean_array_float32_free(ocean_array_float32* array) {
    if (!array) return;
    free(array->data);
    free(array);
}
static inline size_t ocean_array_float32_len(const ocean_array_float32* array) {
    return array ? array->size : 0;
}
static inline float ocean_array_float32_get(const ocean_array_float32* array, size_t index) {
    if (!array || index >= array->size) {
        fprintf(stderr, "Index out of bounds in ocean_array_float32\n"); exit(1);
    }
    return array->data[index];
}
static inline void ocean_array_float32_set(ocean_array_float32* array, size_t index, float value) {
    if (!array || index >= array->size) {
        fprintf(stderr, "Index out of bounds in ocean_array_float32\n"); exit(1);
    }
    array->data[index] = value;
}

static ocean_tensor_float32* ocean_tensor_float32_create(const float* values, size_t value_count, const size_t* shape, size_t ndim) {
    ocean_tensor_float32* tensor = (ocean_tensor_float32*)calloc(1, sizeof(ocean_tensor_float32));
    if (!tensor) { fprintf(stderr, "Ocean allocation error: ocean_tensor_float32\n"); exit(1); }
    tensor->ndim = ndim;
    tensor->size = 1;
    if (ndim == 0) tensor->size = 0;
    tensor->shape = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    tensor->strides = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    if ((ndim && !tensor->shape) || (ndim && !tensor->strides)) {
        free(tensor->shape); free(tensor->strides); free(tensor);
        fprintf(stderr, "Ocean allocation error: ocean_tensor_float32 metadata\n"); exit(1);
    }
    for (size_t i = 0; i < ndim; ++i) tensor->shape[i] = shape[i];
    size_t stride = 1;
    for (size_t i = ndim; i-- > 0;) {
        tensor->strides[i] = stride;
        if (tensor->shape[i] != 0 && stride > (size_t)-1 / tensor->shape[i]) {
            fprintf(stderr, "Tensor size overflow in ocean_tensor_float32\n"); exit(1);
        }
        stride *= tensor->shape[i];
    }
    tensor->size = ndim ? stride : 0;
    if (tensor->size != value_count) {
        fprintf(stderr, "Tensor literal size mismatch in ocean_tensor_float32\n"); exit(1);
    }
    if (tensor->size) {
        tensor->data = (float*)malloc(tensor->size * sizeof(float));
        if (!tensor->data) { fprintf(stderr, "Ocean allocation error: ocean_tensor_float32 data\n"); exit(1); }
        memcpy(tensor->data, values, tensor->size * sizeof(float));
    }
    return tensor;
}
static void ocean_tensor_float32_free(ocean_tensor_float32* tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}
static inline size_t ocean_tensor_float32_len(const ocean_tensor_float32* tensor) {
    return tensor ? tensor->size : 0;
}
static inline size_t ocean_tensor_float32_shape_at(const ocean_tensor_float32* tensor, size_t axis) {
    if (!tensor || axis >= tensor->ndim) {
        fprintf(stderr, "Tensor shape index out of bounds in ocean_tensor_float32\n"); exit(1);
    }
    return tensor->shape[axis];
}
static size_t ocean_tensor_float32_offset(const ocean_tensor_float32* tensor, const size_t* indices, size_t rank) {
    if (!tensor || rank != tensor->ndim || (rank && !indices)) {
        fprintf(stderr, "Tensor rank mismatch in ocean_tensor_float32\n"); exit(1);
    }
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i) {
        if (indices[i] >= tensor->shape[i]) {
            fprintf(stderr, "Tensor index out of bounds in ocean_tensor_float32\n"); exit(1);
        }
        offset += indices[i] * tensor->strides[i];
    }
    return offset;
}
static inline float ocean_tensor_float32_get(const ocean_tensor_float32* tensor, const size_t* indices, size_t rank) {
    return tensor->data[ocean_tensor_float32_offset(tensor, indices, rank)];
}
static inline void ocean_tensor_float32_set(ocean_tensor_float32* tensor, const size_t* indices, size_t rank, float value) {
    tensor->data[ocean_tensor_float32_offset(tensor, indices, rank)] = value;
}

int main(void);

void* ocean_scale_array(ocean_array_float32* values, float factor) {
    int count = ocean_array_float32_len(values);
    for (int i = 0; ((1) > 0 ? i < count : i > count); i += 1) {
        ocean_array_float32_set(values, (size_t)(i), (ocean_array_float32_get(values, (size_t)(i)) * factor));
    }
    return NULL;
}

float ocean_dot_product(ocean_array_float32* left, ocean_array_float32* right) {
    int count = ocean_array_float32_len(left);
    float result = 0.0;
    for (int i = 0; ((1) > 0 ? i < count : i > count); i += 1) {
        result = (result + (ocean_array_float32_get(left, (size_t)(i)) * ocean_array_float32_get(right, (size_t)(i))));
    }
    float ocean_return_0 = result;
    return ocean_return_0;
}

float ocean_tensor_sum(ocean_tensor_float32* values) {
    float total = 0.0;
    int rows = ocean_tensor_float32_shape_at(values, (size_t)(0));
    int cols = ocean_tensor_float32_shape_at(values, (size_t)(1));
    for (int i = 0; ((1) > 0 ? i < rows : i > rows); i += 1) {
        for (int j = 0; ((1) > 0 ? j < cols : j > cols); j += 1) {
            total = (total + ocean_tensor_float32_get(values, (size_t[]){(size_t)(i), (size_t)(j)}, 2));
        }
    }
    float ocean_return_1 = total;
    return ocean_return_1;
}

void* ocean_matmul(ocean_tensor_float32* left, ocean_tensor_float32* right, ocean_tensor_float32* result) {
    int rows = ocean_tensor_float32_shape_at(left, (size_t)(0));
    int shared = ocean_tensor_float32_shape_at(left, (size_t)(1));
    int cols = ocean_tensor_float32_shape_at(right, (size_t)(1));
    for (int i = 0; ((1) > 0 ? i < rows : i > rows); i += 1) {
        for (int j = 0; ((1) > 0 ? j < cols : j > cols); j += 1) {
            float value = 0.0;
            for (int k = 0; ((1) > 0 ? k < shared : k > shared); k += 1) {
                value = (value + (ocean_tensor_float32_get(left, (size_t[]){(size_t)(i), (size_t)(k)}, 2) * ocean_tensor_float32_get(right, (size_t[]){(size_t)(k), (size_t)(j)}, 2)));
            }
            ocean_tensor_float32_set(result, (size_t[]){(size_t)(i), (size_t)(j)}, 2, value);
        }
    }
    return NULL;
}

void* ocean_add_bias(ocean_tensor_float32* matrix, ocean_array_float32* bias) {
    int rows = ocean_tensor_float32_shape_at(matrix, (size_t)(0));
    int cols = ocean_tensor_float32_shape_at(matrix, (size_t)(1));
    for (int i = 0; ((1) > 0 ? i < rows : i > rows); i += 1) {
        for (int j = 0; ((1) > 0 ? j < cols : j > cols); j += 1) {
            ocean_tensor_float32_set(matrix, (size_t[]){(size_t)(i), (size_t)(j)}, 2, (ocean_tensor_float32_get(matrix, (size_t[]){(size_t)(i), (size_t)(j)}, 2) + ocean_array_float32_get(bias, (size_t)(j))));
        }
    }
    return NULL;
}

int main(void) {
    float ocean_array_input_2_values[3] = { 1.0, 2.0, 3.0 };
    ocean_array_float32* input = ocean_array_float32_create(ocean_array_input_2_values, 3);
    float ocean_array_weights_3_values[3] = { 0.5, 1.0, 1.5 };
    ocean_array_float32* weights = ocean_array_float32_create(ocean_array_weights_3_values, 3);
    ocean_scale_array(input, 2.0);
    float score = ocean_dot_product(input, weights);
    printf("%f\n", score);
    float ocean_tensor_left_4_data[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    size_t ocean_tensor_left_4_shape[2] = { 2, 3 };
    ocean_tensor_float32* left = ocean_tensor_float32_create(ocean_tensor_left_4_data, 6, ocean_tensor_left_4_shape, 2);
    float ocean_tensor_right_5_data[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    size_t ocean_tensor_right_5_shape[2] = { 3, 2 };
    ocean_tensor_float32* right = ocean_tensor_float32_create(ocean_tensor_right_5_data, 6, ocean_tensor_right_5_shape, 2);
    float ocean_tensor_product_6_data[4] = { 0.0, 0.0, 0.0, 0.0 };
    size_t ocean_tensor_product_6_shape[2] = { 2, 2 };
    ocean_tensor_float32* product = ocean_tensor_float32_create(ocean_tensor_product_6_data, 4, ocean_tensor_product_6_shape, 2);
    ocean_matmul(left, right, product);
    float ocean_array_bias_7_values[2] = { 0.25, 0.5 };
    ocean_array_float32* bias = ocean_array_float32_create(ocean_array_bias_7_values, 2);
    ocean_add_bias(product, bias);
    int first_row = ocean_tensor_float32_shape_at(product, (size_t)(0));
    int first_col = ocean_tensor_float32_shape_at(product, (size_t)(1));
    int product_size = ocean_tensor_float32_len(product);
    float product_sum = ocean_tensor_sum(product);
    printf("%d\n", first_row);
    printf("%d\n", first_col);
    printf("%d\n", product_size);
    printf("%f\n", product_sum);
    printf("%f\n", ocean_tensor_float32_get(product, (size_t[]){(size_t)(0), (size_t)(0)}, 2));
    printf("%f\n", ocean_tensor_float32_get(product, (size_t[]){(size_t)(1), (size_t)(1)}, 2));
    float ocean_tensor_batch_8_data[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    size_t ocean_tensor_batch_8_shape[3] = { 2, 2, 2 };
    ocean_tensor_float32* batch = ocean_tensor_float32_create(ocean_tensor_batch_8_data, 8, ocean_tensor_batch_8_shape, 3);
    int batches = ocean_tensor_float32_shape_at(batch, (size_t)(0));
    int height = ocean_tensor_float32_shape_at(batch, (size_t)(1));
    int width = ocean_tensor_float32_shape_at(batch, (size_t)(2));
    int batch_elements = ocean_tensor_float32_len(batch);
    int depth = batch->ndim;
    int total_size = batch->size;
    printf("%d\n", depth);
    printf("%d\n", total_size);
    printf("%d %d %d\n", batches, height, width);
    printf("%d\n", batch_elements);
    printf("%f\n", ocean_tensor_float32_get(batch, (size_t[]){(size_t)(1), (size_t)(0), (size_t)(1)}, 3));
    int ocean_return_9 = 0;
    ocean_tensor_float32_free(batch);
    batch = NULL;
    ocean_array_float32_free(bias);
    bias = NULL;
    ocean_tensor_float32_free(product);
    product = NULL;
    ocean_tensor_float32_free(right);
    right = NULL;
    ocean_tensor_float32_free(left);
    left = NULL;
    ocean_array_float32_free(weights);
    weights = NULL;
    ocean_array_float32_free(input);
    input = NULL;
    return ocean_return_9;
    ocean_tensor_float32_free(batch);
    batch = NULL;
    ocean_array_float32_free(bias);
    bias = NULL;
    ocean_tensor_float32_free(product);
    product = NULL;
    ocean_tensor_float32_free(right);
    right = NULL;
    ocean_tensor_float32_free(left);
    left = NULL;
    ocean_array_float32_free(weights);
    weights = NULL;
    ocean_array_float32_free(input);
    input = NULL;
}
"""
    run(P, C)
