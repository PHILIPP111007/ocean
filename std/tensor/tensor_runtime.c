#include "std/tensor/tensor_runtime.h"
#include "std/tensor/tensor_backend.h"

#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include <CL/cl.h>
#endif

typedef enum ocean_tensor_dtype {
    OCEAN_TENSOR_BOOL,
    OCEAN_TENSOR_INT8,
    OCEAN_TENSOR_INT16,
    OCEAN_TENSOR_INT32,
    OCEAN_TENSOR_INT64,
    OCEAN_TENSOR_UINT8,
    OCEAN_TENSOR_UINT16,
    OCEAN_TENSOR_UINT32,
    OCEAN_TENSOR_UINT64,
    OCEAN_TENSOR_FLOAT16,
    OCEAN_TENSOR_FLOAT32,
    OCEAN_TENSOR_FLOAT64,
} ocean_tensor_dtype;

enum {
    OCEAN_TENSOR_ADD = 0,
    OCEAN_TENSOR_SUB = 1,
    OCEAN_TENSOR_MUL = 2,
    OCEAN_TENSOR_DIV = 3,
};

struct ocean_tensor_handle {
    uint64_t identity;
    ocean_tensor_backend_kind device;
    ocean_tensor_dtype dtype;
    size_t item_size;
    size_t ndim;
    size_t size;
    size_t *shape;
    size_t *strides;
    void *cpu_data;
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    cl_mem gpu_data;
#endif
};

#define OCEAN_TENSOR_CPU OCEAN_TENSOR_BACKEND_CPU
#define OCEAN_TENSOR_GPU OCEAN_TENSOR_BACKEND_OPENCL

static uint64_t ocean_tensor_next_identity = 1;



static const ocean_tensor_backend_ops *ocean_tensor_backend_for_device(
    ocean_tensor_backend_kind device
);
static ocean_tensor_handle_t ocean_tensor_matmul_cpu(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
static ocean_tensor_handle_t ocean_tensor_matmul_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static ocean_tensor_handle_t ocean_tensor_matmul_opencl_batched(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
);
#endif
static ocean_tensor_handle_t ocean_tensor_binary_cpu(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_binary_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_scalar_cpu(
    const ocean_tensor_handle_t tensor,
    double scalar,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_scalar_opencl(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_ternary_quantize_cpu(
    const ocean_tensor_handle_t tensor
);
static ocean_tensor_handle_t ocean_tensor_gelu_cpu(
    const ocean_tensor_handle_t tensor
);
static ocean_tensor_handle_t ocean_tensor_gelu_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t input
);
static ocean_tensor_handle_t ocean_tensor_embedding_forward_cpu(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices
);
static ocean_tensor_handle_t ocean_tensor_embedding_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
);
static ocean_tensor_handle_t ocean_tensor_cross_entropy_forward_cpu(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
);
static ocean_tensor_handle_t ocean_tensor_cross_entropy_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets
);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static void ocean_tensor_opencl_ternary_quantize(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output
);
static void ocean_tensor_opencl_embedding_forward(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t output,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
);
static void ocean_tensor_opencl_embedding_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
);
static void ocean_tensor_opencl_cross_entropy_forward(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t row_losses,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
);
static void ocean_tensor_opencl_cross_entropy_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
);
static void ocean_tensor_opencl_permute(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output,
    const int *axes
);
static void ocean_tensor_opencl_gelu(
    const ocean_tensor_handle_t first,
    const ocean_tensor_handle_t second,
    ocean_tensor_handle_t output,
    int key
);
#endif
static void ocean_tensor_write_scalar(
    const ocean_tensor_handle_t tensor,
    size_t index,
    long double value
);
static uint16_t ocean_tensor_float_to_half(float value);

static void ocean_tensor_fill_cpu(ocean_tensor_handle_t tensor, double value);
static void ocean_tensor_fill_opencl(ocean_tensor_handle_t tensor, double value);

static ocean_tensor_handle_t ocean_tensor_restore_device(
    const ocean_tensor_handle_t source,
    ocean_tensor_handle_t cpu_result
);

static ocean_tensor_handle_t ocean_tensor_permute_cpu(
    const ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
);

_Noreturn void ocean_tensor_fail(const char *message) {
    fprintf(stderr, "Ocean Tensor error: %s\n", message);
    exit(EXIT_FAILURE);
}

void ocean_tensor_validate_list_length(size_t actual, size_t expected) {
    if (actual != expected) {
        ocean_tensor_fail("Tensor.from_list requires rectangular lists");
    }
}

static int ocean_tensor_parse_device(const char *device) {
    if (device && strcmp(device, "cpu") == 0) return OCEAN_TENSOR_CPU;
    if (device && strcmp(device, "gpu") == 0) return OCEAN_TENSOR_GPU;
    ocean_tensor_fail("device must be \"cpu\" or \"gpu\"");
    return OCEAN_TENSOR_CPU;
}

static ocean_tensor_dtype ocean_tensor_parse_dtype(const char *name) {
    if (!name || strcmp(name, "float32") == 0) {
        return OCEAN_TENSOR_FLOAT32;
    }
    if (strcmp(name, "bool") == 0) return OCEAN_TENSOR_BOOL;
    if (strcmp(name, "int8") == 0 || strcmp(name, "int8_t") == 0) {
        return OCEAN_TENSOR_INT8;
    }
    if (strcmp(name, "int16") == 0 || strcmp(name, "int16_t") == 0) {
        return OCEAN_TENSOR_INT16;
    }
    if (strcmp(name, "int") == 0 || strcmp(name, "int32") == 0) {
        return OCEAN_TENSOR_INT32;
    }
    if (strcmp(name, "int32_t") == 0) return OCEAN_TENSOR_INT32;
    if (strcmp(name, "int64") == 0 || strcmp(name, "int64_t") == 0) {
        return OCEAN_TENSOR_INT64;
    }
    if (strcmp(name, "uint8") == 0 || strcmp(name, "uint8_t") == 0) {
        return OCEAN_TENSOR_UINT8;
    }
    if (strcmp(name, "uint16") == 0 || strcmp(name, "uint16_t") == 0) {
        return OCEAN_TENSOR_UINT16;
    }
    if (strcmp(name, "uint") == 0 || strcmp(name, "uint32") == 0) {
        return OCEAN_TENSOR_UINT32;
    }
    if (strcmp(name, "uint32_t") == 0) return OCEAN_TENSOR_UINT32;
    if (strcmp(name, "uint64") == 0 || strcmp(name, "uint64_t") == 0) {
        return OCEAN_TENSOR_UINT64;
    }
    if (strcmp(name, "size_t") == 0 || strcmp(name, "uintptr_t") == 0) {
        return OCEAN_TENSOR_UINT64;
    }
    if (strcmp(name, "intptr_t") == 0) return OCEAN_TENSOR_INT64;
    if (strcmp(name, "float16") == 0) return OCEAN_TENSOR_FLOAT16;
    if (strcmp(name, "float") == 0 ||
        strcmp(name, "float64") == 0 ||
        strcmp(name, "double") == 0) {
        return OCEAN_TENSOR_FLOAT64;
    }
    ocean_tensor_fail("unsupported Tensor dtype");
    return OCEAN_TENSOR_FLOAT32;
}

static size_t ocean_tensor_dtype_size(ocean_tensor_dtype dtype) {
    switch (dtype) {
        case OCEAN_TENSOR_BOOL:
        case OCEAN_TENSOR_INT8:
        case OCEAN_TENSOR_UINT8:
            return 1;
        case OCEAN_TENSOR_INT16:
        case OCEAN_TENSOR_UINT16:
        case OCEAN_TENSOR_FLOAT16:
            return 2;
        case OCEAN_TENSOR_INT32:
        case OCEAN_TENSOR_UINT32:
        case OCEAN_TENSOR_FLOAT32:
            return 4;
        case OCEAN_TENSOR_INT64:
        case OCEAN_TENSOR_UINT64:
        case OCEAN_TENSOR_FLOAT64:
            return 8;
    }
    ocean_tensor_fail("invalid Tensor dtype");
    return 0;
}

static size_t ocean_tensor_elements_from_shape(const size_t *shape, size_t ndim) {
    if (!shape || ndim == 0) ocean_tensor_fail("Tensor must have at least one dimension");
    size_t elements = 1;
    for (size_t axis = 0; axis < ndim; ++axis) {
        if (shape[axis] != 0 && elements > SIZE_MAX / shape[axis]) {
            ocean_tensor_fail("Tensor shape is too large");
        }
        elements *= shape[axis];
    }
    return elements;
}

static size_t ocean_tensor_bytes(const ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("invalid null Tensor handle");
    if (tensor->size != 0 && tensor->item_size > SIZE_MAX / tensor->size) {
        ocean_tensor_fail("Tensor allocation is too large");
    }
    return tensor->size * tensor->item_size;
}

static ocean_tensor_handle_t ocean_tensor_alloc(
    const size_t *shape,
    size_t ndim,
    ocean_tensor_dtype dtype,
    int device
) {
    ocean_tensor_handle_t tensor =
        (ocean_tensor_handle_t)calloc(1, sizeof(*tensor));
    if (!tensor) ocean_tensor_fail("out of memory allocating Tensor handle");

    tensor->identity = ocean_tensor_next_identity++;
    if (ocean_tensor_next_identity == 0) {
        ocean_tensor_fail("Tensor identity counter overflow");
    }

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->item_size = ocean_tensor_dtype_size(dtype);
    tensor->size = ocean_tensor_elements_from_shape(shape, ndim);
    if (ndim > SIZE_MAX / sizeof(size_t)) {
        ocean_tensor_fail("Tensor metadata is too large");
    }
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        ocean_tensor_fail("out of memory allocating Tensor metadata");
    }
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->strides[ndim - 1] = 1;
    for (size_t axis = ndim - 1; axis > 0; --axis) {
        if (tensor->strides[axis] != 0 &&
            tensor->shape[axis] > SIZE_MAX / tensor->strides[axis]) {
            free(tensor->shape);
            free(tensor->strides);
            free(tensor);
            ocean_tensor_fail("Tensor strides are too large");
        }
        tensor->strides[axis - 1] =
            tensor->strides[axis] * tensor->shape[axis];
    }
    tensor->device = device;
    return tensor;
}

static ocean_tensor_handle_t ocean_tensor_alloc_zeros(
    const size_t *shape,
    size_t ndim,
    ocean_tensor_dtype dtype,
    int device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc(shape, ndim, dtype, device);
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device((ocean_tensor_backend_kind)device);
    backend->allocate(tensor);
    backend->zero(tensor);
    return tensor;
}

static ocean_tensor_handle_t ocean_tensor_alloc_uninitialized(
    const size_t *shape,
    size_t ndim,
    ocean_tensor_dtype dtype,
    int device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc(shape, ndim, dtype, device);
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device((ocean_tensor_backend_kind)device);
    backend->allocate(tensor);
    return tensor;
}


#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static const char *ocean_tensor_matmul_kernel_source =
    "__kernel void ocean_tensor_matmul("
    "__global const float *a, __global const float *b, __global float *c, "
    "const int rows_a, const int cols_a, const int cols_b) {"
    "int local_row = (int)get_local_id(0);"
    "int local_col = (int)get_local_id(1);"
    "int row = (int)get_group_id(0) * 8 + local_row;"
    "int col = (int)get_group_id(1) * 8 + local_col;"
    "__local float tile_a[8][8];"
    "__local float tile_b[8][8];"
    "float sum = 0.0f;"
    "int tile_count = (cols_a + 7) / 8;"
    "for (int tile = 0; tile < tile_count; ++tile) {"
    "int a_col = tile * 8 + local_col;"
    "int b_row = tile * 8 + local_row;"
    "tile_a[local_row][local_col] = "
    "row < rows_a && a_col < cols_a ? a[row * cols_a + a_col] : 0.0f;"
    "tile_b[local_row][local_col] = "
    "b_row < cols_a && col < cols_b ? b[b_row * cols_b + col] : 0.0f;"
    "barrier(CLK_LOCAL_MEM_FENCE);"
    "for (int k = 0; k < 8; ++k)"
    "sum += tile_a[local_row][k] * tile_b[k][local_col];"
    "barrier(CLK_LOCAL_MEM_FENCE);"
    "}"
    "if (row < rows_a && col < cols_b) c[row * cols_b + col] = sum;"
    "}"
    "__kernel void ocean_tensor_binary("
    "__global const float *a, __global const float *b, __global float *out, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "float left = a[index]; float right = b[index];"
    "if (operation == 0) out[index] = left + right;"
    "else if (operation == 1) out[index] = left - right;"
    "else if (operation == 2) out[index] = left * right;"
    "else out[index] = left / right;"
    "}"
    "}"
    "__kernel void ocean_tensor_scalar("
    "__global const float *input, __global float *out, const float scalar, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "float value = input[index];"
    "if (operation == 0) out[index] = value + scalar;"
    "else if (operation == 1) out[index] = value - scalar;"
    "else if (operation == 2) out[index] = value * scalar;"
    "else out[index] = value / scalar;"
    "}"
    "}"
    "__kernel void ocean_tensor_matmul_int32("
    "__global const int *a, __global const int *b, __global int *c, "
    "const int rows_a, const int cols_a, const int cols_b) {"
    "int local_row = (int)get_local_id(0);"
    "int local_col = (int)get_local_id(1);"
    "int row = (int)get_group_id(0) * 8 + local_row;"
    "int col = (int)get_group_id(1) * 8 + local_col;"
    "__local int tile_a[8][8];"
    "__local int tile_b[8][8];"
    "int sum = 0;"
    "int tile_count = (cols_a + 7) / 8;"
    "for (int tile = 0; tile < tile_count; ++tile) {"
    "int a_col = tile * 8 + local_col;"
    "int b_row = tile * 8 + local_row;"
    "tile_a[local_row][local_col] = "
    "row < rows_a && a_col < cols_a ? a[row * cols_a + a_col] : 0;"
    "tile_b[local_row][local_col] = "
    "b_row < cols_a && col < cols_b ? b[b_row * cols_b + col] : 0;"
    "barrier(CLK_LOCAL_MEM_FENCE);"
    "for (int k = 0; k < 8; ++k)"
    "sum += tile_a[local_row][k] * tile_b[k][local_col];"
    "barrier(CLK_LOCAL_MEM_FENCE);"
    "}"
    "if (row < rows_a && col < cols_b) c[row * cols_b + col] = sum;"
    "}"
    "__kernel void ocean_tensor_binary_int32("
    "__global const int *a, __global const int *b, __global int *out, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "int left = a[index]; int right = b[index];"
    "if (operation == 0) out[index] = left + right;"
    "else if (operation == 1) out[index] = left - right;"
    "else if (operation == 2) out[index] = left * right;"
    "else out[index] = right == 0 ? 0 : left / right;"
    "}"
    "}"
    "__kernel void ocean_tensor_scalar_int32("
    "__global const int *input, __global int *out, const int scalar, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "int value = input[index];"
    "if (operation == 0) out[index] = value + scalar;"
    "else if (operation == 1) out[index] = value - scalar;"
    "else if (operation == 2) out[index] = value * scalar;"
    "else out[index] = scalar == 0 ? 0 : value / scalar;"
    "}"
    "}";

static const char *ocean_tensor_batched_matmul_kernel_source =
    "__kernel void ocean_tensor_batched_matmul("
    "__global const float *a, __global const float *b, "
    "__global float *c, __global const int *a_shape, "
    "__global const int *b_shape, __global const int *out_shape, "
    "__global const int *a_strides, __global const int *b_strides, "
    "const int batch_ndim, const int rows, const int inner, "
    "const int cols, const int transpose_a, const int transpose_b, "
    "const int output_size) {"
    "int index = (int)get_global_id(0);"
    "if (index >= output_size) return;"
    "int matrix_size = rows * cols;"
    "int batch_linear = index / matrix_size;"
    "int matrix_index = index - batch_linear * matrix_size;"
    "int row = matrix_index / cols;"
    "int col = matrix_index - row * cols;"
    "int a_batch_offset = 0;"
    "int b_batch_offset = 0;"
    "int remaining = batch_linear;"
    "for (int axis = batch_ndim - 1; axis >= 0; --axis) {"
    "int coordinate = remaining % out_shape[axis];"
    "remaining /= out_shape[axis];"
    "if (a_shape[axis] != 1) a_batch_offset += coordinate * a_strides[axis];"
    "if (b_shape[axis] != 1) b_batch_offset += coordinate * b_strides[axis];"
    "}"
    "float sum = 0.0f;"
    "for (int k = 0; k < inner; ++k) {"
    "int a_index = transpose_a"
    "? a_batch_offset + k * rows + row"
    ": a_batch_offset + row * inner + k;"
    "int b_index = transpose_b"
    "? b_batch_offset + col * inner + k"
    ": b_batch_offset + k * cols + col;"
    "sum += a[a_index] * b[b_index];"
    "}"
    "c[index] = sum;"
    "}";

static const char *ocean_tensor_permute_kernel_source =
    "__kernel void ocean_tensor_permute("
    "__global const uchar *input, __global uchar *output, "
    "__global const int *output_shape, __global const int *input_strides, "
    "__global const int *axes, const int rank, const int item_size, "
    "const int output_size) {"
    "int index = (int)get_global_id(0);"
    "if (index >= output_size) return;"
    "int remaining = index;"
    "int input_offset = 0;"
    "for (int axis = rank - 1; axis >= 0; --axis) {"
    "int coordinate = remaining % output_shape[axis];"
    "remaining /= output_shape[axis];"
    "input_offset += coordinate * input_strides[axes[axis]];"
    "}"
    "int output_offset = index * item_size;"
    "int input_byte_offset = input_offset * item_size;"
    "for (int byte = 0; byte < item_size; ++byte) {"
    "output[output_offset + byte] = input[input_byte_offset + byte];"
    "}"
    "}";

static const char *ocean_tensor_hotpath_kernel_source =
    "__kernel void ocean_tensor_softmax_last_dim("
    "__global const float *input, __global float *output, "
    "const int rows, const int width) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "int offset = row * width;"
    "float max_value = -3.402823466e+38f;"
    "for (int j = 0; j < width; ++j) {"
    "float value = input[offset + j];"
    "if (value > max_value) max_value = value;"
    "}"
    "float denominator = 0.0f;"
    "for (int j = 0; j < width; ++j) {"
    "denominator += exp(input[offset + j] - max_value);"
    "}"
    "for (int j = 0; j < width; ++j) {"
    "output[offset + j] = exp(input[offset + j] - max_value) / denominator;"
    "}"
    "}"
    "__kernel void ocean_tensor_layer_norm_last_dim("
    "__global const float *input, __global float *output, "
    "const int rows, const int width, const float epsilon) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "int offset = row * width;"
    "float mean = 0.0f;"
    "for (int j = 0; j < width; ++j) mean += input[offset + j];"
    "mean /= (float)width;"
    "float variance = 0.0f;"
    "for (int j = 0; j < width; ++j) {"
    "float delta = input[offset + j] - mean;"
    "variance += delta * delta;"
    "}"
    "variance /= (float)width;"
    "float inverse_std = rsqrt(variance + epsilon);"
    "for (int j = 0; j < width; ++j) {"
    "output[offset + j] = (input[offset + j] - mean) * inverse_std;"
    "}"
    "}"
    "__kernel void ocean_tensor_reduce_last_dim("
    "__global const float *input, __global float *output, "
    "const int rows, const int width, const int mean) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "int offset = row * width;"
    "float value = 0.0f;"
    "for (int j = 0; j < width; ++j) value += input[offset + j];"
    "output[row] = mean ? value / (float)width : value;"
    "}"
    "__kernel void ocean_tensor_sgd_update("
    "__global float *parameter, __global const float *gradient, "
    "const int size, const float learning_rate) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) parameter[index] -= learning_rate * gradient[index];"
    "}"
    "__kernel void ocean_tensor_adamw_update("
    "__global float *parameter, __global const float *gradient, "
    "__global float *first_moment, __global float *second_moment, "
    "const int size, const float learning_rate, const float beta1, "
    "const float beta2, const float epsilon, const float weight_decay, "
    "const float bias_correction1, const float bias_correction2) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "float gradient_value = gradient[index];"
    "float first = beta1 * first_moment[index] + (1.0f - beta1) * gradient_value;"
    "float second = beta2 * second_moment[index] + "
    "(1.0f - beta2) * gradient_value * gradient_value;"
    "first_moment[index] = first;"
    "second_moment[index] = second;"
    "float first_unbiased = first / bias_correction1;"
    "float second_unbiased = second / bias_correction2;"
    "float adaptive = first_unbiased / (sqrt(second_unbiased) + epsilon);"
    "float value = parameter[index];"
    "parameter[index] = value - learning_rate * weight_decay * value "
    "- learning_rate * adaptive;"
    "}"
    "}"
    "__kernel void ocean_tensor_ternary_quantize("
    "__global const float *input, __global float *output, "
    "__local float *partial, const int size) {"
    "int local_index = (int)get_local_id(0);"
    "int local_size = (int)get_local_size(0);"
    "float sum_abs = 0.0f;"
    "for (int index = local_index; index < size; index += local_size) {"
    "float value = input[index];"
    "sum_abs += value < 0.0f ? -value : value;"
    "}"
    "partial[local_index] = sum_abs;"
    "barrier(CLK_LOCAL_MEM_FENCE);"
    "for (int stride = local_size / 2; stride > 0; stride /= 2) {"
    "if (local_index < stride) partial[local_index] += partial[local_index + stride];"
    "barrier(CLK_LOCAL_MEM_FENCE);"
    "}"
    "float scale = partial[0] / (float)size;"
    "if (scale < 1.0e-8f) scale = 1.0e-8f;"
    "float threshold = 0.5f * scale;"
    "for (int index = local_index; index < size; index += local_size) {"
    "float value = input[index];"
    "output[index] = value > threshold ? scale : "
    "(value < -threshold ? -scale : 0.0f);"
    "}"
    "}";

static const char *ocean_tensor_gelu_kernel_source =
    "__kernel void ocean_tensor_gelu("
    "__global const float *input, __global float *output, "
    "const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index >= size) return;"
    "const float coefficient = 0.7978845608028654f;"
    "const float cubic = 0.044715f;"
    "float value = input[index];"
    "float value_squared = value * value;"
    "float argument = coefficient * (value + cubic * value * value_squared);"
    "output[index] = 0.5f * value * (1.0f + tanh(argument));"
    "}"
    "__kernel void ocean_tensor_gelu_backward("
    "__global const float *upstream, __global const float *input, "
    "__global float *output, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index >= size) return;"
    "const float coefficient = 0.7978845608028654f;"
    "const float cubic = 0.044715f;"
    "float value = input[index];"
    "float argument = coefficient * (value + cubic * value * value * value);"
    "float tanh_argument = tanh(argument);"
    "float derivative = 0.5f * (1.0f + tanh_argument);"
    "derivative += 0.5f * value * (1.0f - tanh_argument * tanh_argument)"
    " * coefficient * (1.0f + 3.0f * cubic * value * value);"
    "output[index] = upstream[index] * derivative;"
    "}";

static const char *ocean_tensor_cache_kernel_source =
    "__kernel void ocean_tensor_cache_write("
    "__global float *cache, __global const float *value, "
    "const int heads, const int sequence, const int width, const int position) {"
    "size_t linear = get_global_id(0);"
    "size_t head_width = (size_t)heads * (size_t)width;"
    "size_t batch = linear / head_width;"
    "size_t remainder = linear % head_width;"
    "size_t head = remainder / (size_t)width;"
    "size_t column = remainder % (size_t)width;"
    "size_t destination = (batch * (size_t)heads + head) * "
    "(size_t)sequence * (size_t)width + (size_t)position * (size_t)width + "
    "column;"
    "cache[destination] = value[linear];"
    "}"
    "__kernel void ocean_tensor_cache_slice("
    "__global const float *cache, __global float *output, "
    "const int heads, const int source_sequence, const int output_sequence, "
    "const int width, const int start) {"
    "size_t linear = get_global_id(0);"
    "size_t row_width = (size_t)output_sequence * (size_t)width;"
    "size_t head_span = (size_t)heads * row_width;"
    "size_t batch = linear / head_span;"
    "size_t remainder = linear % head_span;"
    "size_t head = remainder / row_width;"
    "remainder = remainder % row_width;"
    "size_t position = remainder / (size_t)width;"
    "size_t column = remainder % (size_t)width;"
    "size_t source = (batch * (size_t)heads + head) * "
    "(size_t)source_sequence * (size_t)width + "
    "((size_t)start + position) * (size_t)width + column;"
    "output[linear] = cache[source];"
    "}";

static const char *ocean_tensor_embedding_kernel_source =
    "inline void ocean_tensor_atomic_add_f32("
    "volatile __global float *address, const float value) {"
    "volatile __global int *bits = (volatile __global int *)address;"
    "int old = *bits;"
    "int assumed;"
    "do {"
    "assumed = old;"
    "old = atomic_cmpxchg(bits, assumed, "
    "as_int(as_float(assumed) + value));"
    "} while (old != assumed);"
    "}"
    "__kernel void ocean_tensor_embedding_forward("
    "__global const float *weight, __global const long *indices, "
    "__global float *output, __global int *error, "
    "const int index_count, const int vocab, const int dim) {"
    "int index = (int)get_global_id(0);"
    "int total = index_count * dim;"
    "if (index >= total) return;"
    "int token_position = index / dim;"
    "int feature = index - token_position * dim;"
    "long token = indices[token_position];"
    "if (token < 0 || token >= (long)vocab) {"
    "atomic_or(error, 1);"
    "output[index] = 0.0f;"
    "return;"
    "}"
    "output[index] = weight[(int)token * dim + feature];"
    "}"
    "__kernel void ocean_tensor_embedding_backward("
    "__global const float *upstream, __global const long *indices, "
    "__global float *gradient, __global int *error, "
    "const int index_count, const int vocab, const int dim) {"
    "int index = (int)get_global_id(0);"
    "int total = index_count * dim;"
    "if (index >= total) return;"
    "int token_position = index / dim;"
    "int feature = index - token_position * dim;"
    "long token = indices[token_position];"
    "if (token < 0 || token >= (long)vocab) {"
    "atomic_or(error, 1);"
    "return;"
    "}"
    "ocean_tensor_atomic_add_f32("
    "gradient + (int)token * dim + feature, upstream[index]);"
    "}";

static const char *ocean_tensor_cross_entropy_kernel_source =
    "__kernel void ocean_tensor_cross_entropy_forward("
    "__global const float *logits, __global const long *targets, "
    "__global float *probabilities, __global float *row_losses, "
    "__global int *error, const int rows, const int vocab) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "int offset = row * vocab;"
    "float maximum = -3.402823466e+38f;"
    "for (int cls = 0; cls < vocab; ++cls) {"
    "float value = logits[offset + cls];"
    "if (value > maximum) maximum = value;"
    "}"
    "float denominator = 0.0f;"
    "for (int cls = 0; cls < vocab; ++cls) {"
    "float exponential = exp(logits[offset + cls] - maximum);"
    "probabilities[offset + cls] = exponential;"
    "denominator += exponential;"
    "}"
    "long target = targets[row];"
    "if (target < 0 || target >= (long)vocab) {"
    "atomic_or(error, 1);"
    "row_losses[row] = 0.0f;"
    "} else {"
    "float target_logit = logits[offset + (int)target];"
    "row_losses[row] = log(denominator) + maximum - target_logit;"
    "}"
    "for (int cls = 0; cls < vocab; ++cls) {"
    "probabilities[offset + cls] /= denominator;"
    "}"
    "}"
    "__kernel void ocean_tensor_cross_entropy_backward("
    "__global const float *upstream, "
    "__global const float *probabilities, "
    "__global const long *targets, __global float *gradient, "
    "__global int *error, const int rows, const int vocab) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "long target = targets[row];"
    "int offset = row * vocab;"
    "if (target < 0 || target >= (long)vocab) {"
    "atomic_or(error, 1);"
    "for (int cls = 0; cls < vocab; ++cls) gradient[offset + cls] = 0.0f;"
    "return;"
    "}"
    "float scale = upstream[0] / (float)rows;"
    "for (int cls = 0; cls < vocab; ++cls) {"
    "float value = probabilities[offset + cls];"
    "if (cls == (int)target) value -= 1.0f;"
    "gradient[offset + cls] = value * scale;"
    "}"
    "}";

static const char *ocean_tensor_backward_kernel_source =
    "__kernel void ocean_tensor_softmax_backward_last_dim("
    "__global const float *upstream, __global const float *output, "
    "__global float *gradient, const int rows, const int width) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "int offset = row * width;"
    "float dot = 0.0f;"
    "for (int j = 0; j < width; ++j) {"
    "dot += upstream[offset + j] * output[offset + j];"
    "}"
    "for (int j = 0; j < width; ++j) {"
    "float y = output[offset + j];"
    "gradient[offset + j] = y * (upstream[offset + j] - dot);"
    "}"
    "}"
    "__kernel void ocean_tensor_layer_norm_backward_last_dim("
    "__global const float *upstream, __global const float *input, "
    "__global float *gradient, const int rows, const int width, "
    "const float epsilon) {"
    "int row = (int)get_global_id(0);"
    "if (row >= rows) return;"
    "int offset = row * width;"
    "float mean = 0.0f;"
    "for (int j = 0; j < width; ++j) mean += input[offset + j];"
    "mean /= (float)width;"
    "float variance = 0.0f;"
    "float sum_g = 0.0f;"
    "float sum_g_centered = 0.0f;"
    "for (int j = 0; j < width; ++j) {"
    "float centered = input[offset + j] - mean;"
    "variance += centered * centered;"
    "sum_g += upstream[offset + j];"
    "sum_g_centered += upstream[offset + j] * centered;"
    "}"
    "variance /= (float)width;"
    "float inverse_std = 1.0f / sqrt(variance + epsilon);"
    "float inverse_variance = inverse_std * inverse_std;"
    "float mean_g = sum_g / (float)width;"
    "float mean_g_centered = sum_g_centered / (float)width;"
    "for (int j = 0; j < width; ++j) {"
    "float centered = input[offset + j] - mean;"
    "gradient[offset + j] = inverse_std * ("
    "upstream[offset + j] - mean_g - centered * inverse_variance * mean_g_centered);"
    "}"
    "}";

typedef struct ocean_tensor_opencl_runtime {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_kernel;
    cl_kernel matmul_int32_kernel;
    cl_kernel batched_matmul_kernel;
    cl_kernel permute_kernel;
    cl_kernel binary_kernel;
    cl_kernel binary_int32_kernel;
    cl_kernel scalar_kernel;
    cl_kernel scalar_int32_kernel;
    cl_kernel softmax_kernel;
    cl_kernel layer_norm_kernel;
    cl_kernel reduce_kernel;
    cl_kernel sgd_update_kernel;
    cl_kernel adamw_update_kernel;
    cl_kernel ternary_quantize_kernel;
    cl_kernel gelu_kernel;
    cl_kernel gelu_backward_kernel;
    cl_kernel embedding_forward_kernel;
    cl_kernel embedding_backward_kernel;
    cl_kernel cross_entropy_forward_kernel;
    cl_kernel cross_entropy_backward_kernel;
    cl_kernel softmax_backward_kernel;
    cl_kernel layer_norm_backward_kernel;
    cl_kernel cache_write_kernel;
    cl_kernel cache_slice_kernel;
} ocean_tensor_opencl_runtime;

typedef enum ocean_tensor_opencl_kernel_key {
    OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_INT32,
    OCEAN_TENSOR_OPENCL_KERNEL_BATCHED_MATMUL_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PERMUTE,
    OCEAN_TENSOR_OPENCL_KERNEL_BINARY_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_BINARY_INT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_INT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SGD_UPDATE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_ADAMW_UPDATE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_QUANTIZE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_GELU_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_FORWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_BACKWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_FORWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_BACKWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_BACKWARD_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_BACKWARD_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_CACHE_WRITE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_CACHE_SLICE_FLOAT32,
} ocean_tensor_opencl_kernel_key;

static ocean_tensor_opencl_runtime ocean_tensor_opencl;
static int ocean_tensor_opencl_initialized = 0;

static void ocean_tensor_opencl_shutdown(void) {
    if (!ocean_tensor_opencl_initialized) return;
    if (ocean_tensor_opencl.scalar_kernel) {
        clReleaseKernel(ocean_tensor_opencl.scalar_kernel);
        ocean_tensor_opencl.scalar_kernel = NULL;
    }
    if (ocean_tensor_opencl.binary_kernel) {
        clReleaseKernel(ocean_tensor_opencl.binary_kernel);
        ocean_tensor_opencl.binary_kernel = NULL;
    }
    if (ocean_tensor_opencl.matmul_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matmul_kernel);
        ocean_tensor_opencl.matmul_kernel = NULL;
    }
    if (ocean_tensor_opencl.matmul_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matmul_int32_kernel);
        ocean_tensor_opencl.matmul_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.batched_matmul_kernel) {
        clReleaseKernel(ocean_tensor_opencl.batched_matmul_kernel);
        ocean_tensor_opencl.batched_matmul_kernel = NULL;
    }
    if (ocean_tensor_opencl.permute_kernel) {
        clReleaseKernel(ocean_tensor_opencl.permute_kernel);
        ocean_tensor_opencl.permute_kernel = NULL;
    }
    if (ocean_tensor_opencl.scalar_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.scalar_int32_kernel);
        ocean_tensor_opencl.scalar_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.binary_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.binary_int32_kernel);
        ocean_tensor_opencl.binary_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.softmax_kernel) {
        clReleaseKernel(ocean_tensor_opencl.softmax_kernel);
        ocean_tensor_opencl.softmax_kernel = NULL;
    }
    if (ocean_tensor_opencl.layer_norm_kernel) {
        clReleaseKernel(ocean_tensor_opencl.layer_norm_kernel);
        ocean_tensor_opencl.layer_norm_kernel = NULL;
    }
    if (ocean_tensor_opencl.reduce_kernel) {
        clReleaseKernel(ocean_tensor_opencl.reduce_kernel);
        ocean_tensor_opencl.reduce_kernel = NULL;
    }
    if (ocean_tensor_opencl.sgd_update_kernel) {
        clReleaseKernel(ocean_tensor_opencl.sgd_update_kernel);
        ocean_tensor_opencl.sgd_update_kernel = NULL;
    }
    if (ocean_tensor_opencl.adamw_update_kernel) {
        clReleaseKernel(ocean_tensor_opencl.adamw_update_kernel);
        ocean_tensor_opencl.adamw_update_kernel = NULL;
    }
    if (ocean_tensor_opencl.ternary_quantize_kernel) {
        clReleaseKernel(ocean_tensor_opencl.ternary_quantize_kernel);
        ocean_tensor_opencl.ternary_quantize_kernel = NULL;
    }
    if (ocean_tensor_opencl.embedding_forward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.embedding_forward_kernel);
        ocean_tensor_opencl.embedding_forward_kernel = NULL;
    }
    if (ocean_tensor_opencl.embedding_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.embedding_backward_kernel);
        ocean_tensor_opencl.embedding_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.cross_entropy_forward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cross_entropy_forward_kernel);
        ocean_tensor_opencl.cross_entropy_forward_kernel = NULL;
    }
    if (ocean_tensor_opencl.cross_entropy_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cross_entropy_backward_kernel);
        ocean_tensor_opencl.cross_entropy_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.softmax_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.softmax_backward_kernel);
        ocean_tensor_opencl.softmax_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.layer_norm_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.layer_norm_backward_kernel);
        ocean_tensor_opencl.layer_norm_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.cache_write_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cache_write_kernel);
        ocean_tensor_opencl.cache_write_kernel = NULL;
    }
    if (ocean_tensor_opencl.cache_slice_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cache_slice_kernel);
        ocean_tensor_opencl.cache_slice_kernel = NULL;
    }
    if (ocean_tensor_opencl.program) {
        clReleaseProgram(ocean_tensor_opencl.program);
        ocean_tensor_opencl.program = NULL;
    }
    if (ocean_tensor_opencl.queue) {
        clReleaseCommandQueue(ocean_tensor_opencl.queue);
        ocean_tensor_opencl.queue = NULL;
    }
    if (ocean_tensor_opencl.context) {
        clReleaseContext(ocean_tensor_opencl.context);
        ocean_tensor_opencl.context = NULL;
    }
    ocean_tensor_opencl.device = NULL;
    ocean_tensor_opencl_initialized = 0;
}

static void ocean_tensor_opencl_check(cl_int status, const char *operation) {
    if (status != CL_SUCCESS) {
        char message[256];
        snprintf(message, sizeof(message),
                 "%s failed (OpenCL error %d)", operation, status);
        ocean_tensor_fail(message);
    }
}

static void ocean_tensor_opencl_wait_event(
    cl_event event,
    const char *operation
) {
    if (!event) return;
    ocean_tensor_opencl_check(clWaitForEvents(1, &event), operation);
    ocean_tensor_opencl_check(clReleaseEvent(event), "clReleaseEvent");
}

static void ocean_tensor_opencl_release_event(cl_event event) {
    if (!event) return;
    ocean_tensor_opencl_check(clReleaseEvent(event), "clReleaseEvent");
}

static void ocean_tensor_opencl_init(void) {
    if (ocean_tensor_opencl_initialized) return;
    ocean_tensor_opencl_initialized = 1;
    atexit(ocean_tensor_opencl_shutdown);

    cl_uint platform_count = 0;
    ocean_tensor_opencl_check(
        clGetPlatformIDs(0, NULL, &platform_count), "clGetPlatformIDs"
    );
    if (platform_count == 0) ocean_tensor_fail("no OpenCL platform is available");

    cl_platform_id *platforms = (cl_platform_id *)calloc(
        platform_count, sizeof(*platforms)
    );
    if (!platforms) ocean_tensor_fail("out of memory enumerating OpenCL platforms");
    ocean_tensor_opencl_check(
        clGetPlatformIDs(platform_count, platforms, NULL), "clGetPlatformIDs"
    );

    /* A Tensor selected as "gpu" must never silently bind to a CPU OpenCL
       device through CL_DEVICE_TYPE_DEFAULT. Search every platform and fail
       explicitly when no real GPU device is exposed by the ICD. */
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    for (cl_uint index = 0; index < platform_count; ++index) {
        cl_device_id candidate = NULL;
        cl_int device_status = clGetDeviceIDs(
            platforms[index], CL_DEVICE_TYPE_GPU, 1, &candidate, NULL
        );
        if (device_status == CL_SUCCESS) {
            platform = platforms[index];
            device = candidate;
            break;
        }
    }
    free(platforms);
    if (!device || !platform) {
        ocean_tensor_fail(
            "no OpenCL GPU device is available; refusing CPU fallback for device= gpu"
        );
    }

    cl_int status = CL_SUCCESS;
    ocean_tensor_opencl.device = device;

    ocean_tensor_opencl.context =
        clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    ocean_tensor_opencl_check(status, "clCreateContext");
#ifdef CL_VERSION_2_0
    const cl_queue_properties queue_properties[] = {0};
    ocean_tensor_opencl.queue = clCreateCommandQueueWithProperties(
        ocean_tensor_opencl.context, device, queue_properties, &status
    );
#else
    ocean_tensor_opencl.queue =
        clCreateCommandQueue(ocean_tensor_opencl.context, device, 0, &status);
#endif
    ocean_tensor_opencl_check(status, "clCreateCommandQueue");

    const char *sources[] = {
        ocean_tensor_matmul_kernel_source,
        ocean_tensor_batched_matmul_kernel_source,
        ocean_tensor_permute_kernel_source,
        ocean_tensor_hotpath_kernel_source,
        ocean_tensor_gelu_kernel_source,
        ocean_tensor_embedding_kernel_source,
        ocean_tensor_cross_entropy_kernel_source,
        ocean_tensor_backward_kernel_source,
        ocean_tensor_cache_kernel_source,
    };
    ocean_tensor_opencl.program = clCreateProgramWithSource(
        ocean_tensor_opencl.context, 9, sources, NULL, &status
    );
    ocean_tensor_opencl_check(status, "clCreateProgramWithSource");
    status = clBuildProgram(
        ocean_tensor_opencl.program, 1, &device, NULL, NULL, NULL
    );
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(
            ocean_tensor_opencl.program, device, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size
        );
        char *log = (char *)calloc(log_size + 1, 1);
        if (log) {
            clGetProgramBuildInfo(
                ocean_tensor_opencl.program, device, CL_PROGRAM_BUILD_LOG,
                log_size, log, NULL
            );
            fprintf(stderr, "Ocean Tensor OpenCL build log:\n%s\n", log);
            free(log);
        }
        ocean_tensor_opencl_check(status, "clBuildProgram");
    }
}

static cl_kernel ocean_tensor_opencl_get_kernel(
    ocean_tensor_opencl_kernel_key key
) {
    ocean_tensor_opencl_init();
    cl_kernel *slot = NULL;
    const char *name = NULL;
    switch (key) {
        case OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_FLOAT32:
            slot = &ocean_tensor_opencl.matmul_kernel;
            name = "ocean_tensor_matmul";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_INT32:
            slot = &ocean_tensor_opencl.matmul_int32_kernel;
            name = "ocean_tensor_matmul_int32";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_BATCHED_MATMUL_FLOAT32:
            slot = &ocean_tensor_opencl.batched_matmul_kernel;
            name = "ocean_tensor_batched_matmul";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PERMUTE:
            slot = &ocean_tensor_opencl.permute_kernel;
            name = "ocean_tensor_permute";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_BINARY_FLOAT32:
            slot = &ocean_tensor_opencl.binary_kernel;
            name = "ocean_tensor_binary";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_BINARY_INT32:
            slot = &ocean_tensor_opencl.binary_int32_kernel;
            name = "ocean_tensor_binary_int32";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_FLOAT32:
            slot = &ocean_tensor_opencl.scalar_kernel;
            name = "ocean_tensor_scalar";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_INT32:
            slot = &ocean_tensor_opencl.scalar_int32_kernel;
            name = "ocean_tensor_scalar_int32";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_FLOAT32:
            slot = &ocean_tensor_opencl.softmax_kernel;
            name = "ocean_tensor_softmax_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32:
            slot = &ocean_tensor_opencl.layer_norm_kernel;
            name = "ocean_tensor_layer_norm_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32:
            slot = &ocean_tensor_opencl.reduce_kernel;
            name = "ocean_tensor_reduce_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SGD_UPDATE_FLOAT32:
            slot = &ocean_tensor_opencl.sgd_update_kernel;
            name = "ocean_tensor_sgd_update";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_ADAMW_UPDATE_FLOAT32:
            slot = &ocean_tensor_opencl.adamw_update_kernel;
            name = "ocean_tensor_adamw_update";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_QUANTIZE_FLOAT32:
            slot = &ocean_tensor_opencl.ternary_quantize_kernel;
            name = "ocean_tensor_ternary_quantize";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_GELU_FLOAT32:
            slot = &ocean_tensor_opencl.gelu_kernel;
            name = "ocean_tensor_gelu";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32:
            slot = &ocean_tensor_opencl.gelu_backward_kernel;
            name = "ocean_tensor_gelu_backward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_FORWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.embedding_forward_kernel;
            name = "ocean_tensor_embedding_forward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_BACKWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.embedding_backward_kernel;
            name = "ocean_tensor_embedding_backward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_FORWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.cross_entropy_forward_kernel;
            name = "ocean_tensor_cross_entropy_forward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_BACKWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.cross_entropy_backward_kernel;
            name = "ocean_tensor_cross_entropy_backward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_BACKWARD_FLOAT32:
            slot = &ocean_tensor_opencl.softmax_backward_kernel;
            name = "ocean_tensor_softmax_backward_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_BACKWARD_FLOAT32:
            slot = &ocean_tensor_opencl.layer_norm_backward_kernel;
            name = "ocean_tensor_layer_norm_backward_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CACHE_WRITE_FLOAT32:
            slot = &ocean_tensor_opencl.cache_write_kernel;
            name = "ocean_tensor_cache_write";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CACHE_SLICE_FLOAT32:
            slot = &ocean_tensor_opencl.cache_slice_kernel;
            name = "ocean_tensor_cache_slice";
            break;
    }
    if (!slot || !name) ocean_tensor_fail("invalid OpenCL Tensor kernel key");
    if (!*slot) {
        cl_int status = CL_SUCCESS;
        *slot = clCreateKernel(ocean_tensor_opencl.program, name, &status);
        ocean_tensor_opencl_check(status, name);
    }
    return *slot;
}

static void ocean_tensor_gpu_alloc(ocean_tensor_handle_t tensor) {
    ocean_tensor_opencl_init();
    cl_int status = CL_SUCCESS;
    tensor->gpu_data = clCreateBuffer(
        ocean_tensor_opencl.context, CL_MEM_READ_WRITE,
        ocean_tensor_bytes(tensor) ? ocean_tensor_bytes(tensor) : 1,
        NULL, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
}

static void ocean_tensor_gpu_write(
    ocean_tensor_handle_t tensor,
    const void *data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_FALSE,
            0, bytes, data, 0, NULL, &event
        ),
        "clEnqueueWriteBuffer"
    );
    ocean_tensor_opencl_wait_event(event, "clWaitForEvents(write)");
}

static void ocean_tensor_gpu_zero(
    ocean_tensor_handle_t tensor
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;

#if defined(CL_VERSION_1_2)
    const unsigned char zero = 0;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueFillBuffer(
            ocean_tensor_opencl.queue,
            tensor->gpu_data,
            &zero,
            sizeof(zero),
            0,
            bytes,
            0,
            NULL,
            &event
        ),
        "clEnqueueFillBuffer"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue),
        "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
#else
    void *zeros = calloc(1, bytes);
    if (!zeros) ocean_tensor_fail("out of memory zeroing GPU Tensor");
    ocean_tensor_gpu_write(tensor, zeros);
    free(zeros);
#endif
}


static void ocean_tensor_gpu_read(
    ocean_tensor_handle_t tensor,
    void *data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueReadBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_FALSE,
            0, bytes, data, 0, NULL, &event
        ),
        "clEnqueueReadBuffer"
    );
    ocean_tensor_opencl_wait_event(event, "clWaitForEvents(read)");
}
#endif

static void ocean_tensor_cpu_allocate(ocean_tensor_handle_t tensor) {
    size_t bytes = ocean_tensor_bytes(tensor);
    tensor->cpu_data = bytes ? malloc(bytes) : NULL;
    if (bytes && !tensor->cpu_data) {
        ocean_tensor_fail("out of memory allocating CPU Tensor");
    }
}

static void ocean_tensor_cpu_zero(ocean_tensor_handle_t tensor) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes) memset(tensor->cpu_data, 0, bytes);
}

static void ocean_tensor_cpu_copy(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    size_t bytes = ocean_tensor_bytes(source);
    if (bytes) memcpy(destination->cpu_data, source->cpu_data, bytes);
}

static void ocean_tensor_cpu_read(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes) memcpy(host_data, tensor->cpu_data, bytes);
}

static void ocean_tensor_cpu_write(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes) memcpy(tensor->cpu_data, host_data, bytes);
}

static void ocean_tensor_cpu_release(ocean_tensor_handle_t tensor) {
    free(tensor->cpu_data);
    tensor->cpu_data = NULL;
}

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static void ocean_tensor_opencl_copy(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    size_t bytes = ocean_tensor_bytes(source);
    if (!bytes) return;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueCopyBuffer(
            ocean_tensor_opencl.queue, source->gpu_data, destination->gpu_data,
            0, 0, bytes, 0, NULL, &event
        ),
        "clEnqueueCopyBuffer"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_release(ocean_tensor_handle_t tensor) {
    if (tensor->gpu_data) {
        clReleaseMemObject(tensor->gpu_data);
        tensor->gpu_data = NULL;
    }
}
#else
static void ocean_tensor_gpu_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_copy_unavailable(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    (void)destination;
    (void)source;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_read_unavailable(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_write_unavailable(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_release_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
}
#endif

static const ocean_tensor_backend_ops ocean_tensor_cpu_backend = {
    .kind = OCEAN_TENSOR_BACKEND_CPU,
    .name = "cpu",
    .compiled = true,
    .allocate = ocean_tensor_cpu_allocate,
    .zero = ocean_tensor_cpu_zero,
    .copy = ocean_tensor_cpu_copy,
    .read = ocean_tensor_cpu_read,
    .write = ocean_tensor_cpu_write,
    .release = ocean_tensor_cpu_release,
    .matmul = ocean_tensor_matmul_cpu,
    .binary = ocean_tensor_binary_cpu,
    .scalar = ocean_tensor_scalar_cpu,
    .fill = ocean_tensor_fill_cpu,
};

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static const ocean_tensor_backend_ops ocean_tensor_opencl_backend = {
    .kind = OCEAN_TENSOR_BACKEND_OPENCL,
    .name = "opencl",
    .compiled = true,
    .allocate = ocean_tensor_gpu_alloc,
    .zero = ocean_tensor_gpu_zero,
    .copy = ocean_tensor_opencl_copy,
    .read = ocean_tensor_gpu_read,
    .write = ocean_tensor_gpu_write,
    .release = ocean_tensor_opencl_release,
    .matmul = ocean_tensor_matmul_opencl,
    .binary = ocean_tensor_binary_opencl,
    .scalar = ocean_tensor_scalar_opencl,
    .fill = ocean_tensor_fill_opencl,
};
#else
static const ocean_tensor_backend_ops ocean_tensor_opencl_backend = {
    .kind = OCEAN_TENSOR_BACKEND_OPENCL,
    .name = "opencl",
    .compiled = false,
    .allocate = ocean_tensor_gpu_unavailable,
    .zero = ocean_tensor_gpu_unavailable,
    .copy = ocean_tensor_gpu_copy_unavailable,
    .read = ocean_tensor_gpu_read_unavailable,
    .write = ocean_tensor_gpu_write_unavailable,
    .release = ocean_tensor_gpu_release_unavailable,
    .matmul = ocean_tensor_matmul_opencl,
    .binary = ocean_tensor_binary_opencl,
    .scalar = ocean_tensor_scalar_opencl,
    .fill = ocean_tensor_fill_opencl,
};
#endif

static const ocean_tensor_backend_ops *ocean_tensor_backend_for_device(
    ocean_tensor_backend_kind device
) {
    if (device == OCEAN_TENSOR_BACKEND_CPU) return &ocean_tensor_cpu_backend;
    if (device == OCEAN_TENSOR_BACKEND_OPENCL) return &ocean_tensor_opencl_backend;
    ocean_tensor_fail("invalid Tensor backend device");
    return &ocean_tensor_cpu_backend;
}

ocean_tensor_handle_t ocean_tensor_zeros_nd(
    const size_t *shape,
    size_t ndim,
    const char *dtype,
    const char *device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc_zeros(
        shape, ndim, ocean_tensor_parse_dtype(dtype),
        ocean_tensor_parse_device(device)
    );
    return tensor;
}

ocean_tensor_handle_t ocean_tensor_zeros(
    int rows,
    int cols,
    const char *device
) {
    if (rows < 0 || cols < 0) ocean_tensor_fail("Tensor dimensions must be non-negative");
    size_t shape[2] = {(size_t)rows, (size_t)cols};
    ocean_tensor_handle_t tensor =
        ocean_tensor_alloc_zeros(shape, 2, OCEAN_TENSOR_FLOAT32,
                                 ocean_tensor_parse_device(device));
    return tensor;
}

ocean_tensor_handle_t ocean_tensor_from_cpu_strided(
    const void *data,
    const size_t *shape,
    const size_t *strides,
    size_t ndim,
    const char *dtype,
    const char *device
) {
    if (!shape || !strides) {
        ocean_tensor_fail("Tensor metadata cannot be null");
    }

    ocean_tensor_dtype parsed_dtype = ocean_tensor_parse_dtype(dtype);
    ocean_tensor_handle_t host = ocean_tensor_alloc_uninitialized(
        shape, ndim, parsed_dtype, OCEAN_TENSOR_CPU
    );

    if (host->size && !data) {
        ocean_tensor_release(host);
        ocean_tensor_fail("Tensor data cannot be null for a non-empty Tensor");
    }

    unsigned char *destination = (unsigned char *)host->cpu_data;
    const unsigned char *source = (const unsigned char *)data;
    size_t item_size = host->item_size;

    bool contiguous = true;
    size_t expected = 1;

    for (size_t axis = ndim; axis-- > 0;) {
        if (strides[axis] != expected) {
            contiguous = false;
            break;
        }
        if (shape[axis] != 0 && expected > SIZE_MAX / shape[axis]) {
            contiguous = false;
            break;
        }
        expected *= shape[axis];
    }

    if (host->size && contiguous) {
        memcpy(destination, source, ocean_tensor_bytes(host));
    } else {
        for (size_t linear = 0; linear < host->size; ++linear) {
            size_t remaining = linear;
            size_t source_offset = 0;

            for (size_t axis = ndim; axis-- > 0;) {
                size_t coordinate = shape[axis]
                    ? remaining % shape[axis]
                    : 0;
                remaining = shape[axis]
                    ? remaining / shape[axis]
                    : 0;
                source_offset += coordinate * strides[axis];
            }

            memcpy(
                destination + linear * item_size,
                source + source_offset * item_size,
                item_size
            );
        }
    }

    int target = ocean_tensor_parse_device(device);
    if (target == OCEAN_TENSOR_CPU) {
        return host;
    }

    ocean_tensor_handle_t result = ocean_tensor_to(host, device);
    ocean_tensor_release(host);
    return result;
}


static void ocean_tensor_fill_cpu(
    ocean_tensor_handle_t tensor,
    double value
) {
    if (tensor->size == 0) return;

    if (value == 0.0) {
        memset(tensor->cpu_data, 0, ocean_tensor_bytes(tensor));
        return;
    }

    size_t size = tensor->size;

#define OCEAN_FILL(type, converted) \
    do { \
        type v = (converted); \
        type *restrict data = (type *)tensor->cpu_data; \
        for (size_t i = 0; i < size; ++i) data[i] = v; \
    } while (0)

    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: OCEAN_FILL(bool, value != 0.0); break;
        case OCEAN_TENSOR_INT8: OCEAN_FILL(int8_t, (int8_t)value); break;
        case OCEAN_TENSOR_INT16: OCEAN_FILL(int16_t, (int16_t)value); break;
        case OCEAN_TENSOR_INT32: OCEAN_FILL(int32_t, (int32_t)value); break;
        case OCEAN_TENSOR_INT64: OCEAN_FILL(int64_t, (int64_t)value); break;
        case OCEAN_TENSOR_UINT8: OCEAN_FILL(uint8_t, (uint8_t)value); break;
        case OCEAN_TENSOR_UINT16: OCEAN_FILL(uint16_t, (uint16_t)value); break;
        case OCEAN_TENSOR_UINT32: OCEAN_FILL(uint32_t, (uint32_t)value); break;
        case OCEAN_TENSOR_UINT64: OCEAN_FILL(uint64_t, (uint64_t)value); break;
        case OCEAN_TENSOR_FLOAT16:
            OCEAN_FILL(uint16_t, ocean_tensor_float_to_half((float)value));
            break;
        case OCEAN_TENSOR_FLOAT32: OCEAN_FILL(float, (float)value); break;
        case OCEAN_TENSOR_FLOAT64: OCEAN_FILL(double, value); break;
    }

#undef OCEAN_FILL
}


ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("cannot copy a null Tensor");
    ocean_tensor_handle_t result = ocean_tensor_alloc(
        tensor->shape, tensor->ndim, tensor->dtype, tensor->device
    );
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device(tensor->device);
    backend->allocate(result);
    backend->copy(result, tensor);
    return result;
}

ocean_tensor_handle_t ocean_tensor_ternary_quantize(
    ocean_tensor_handle_t tensor
) {
    if (!tensor) ocean_tensor_fail("Tensor.ternary_quantize on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.ternary_quantize currently requires float32");
    }

    ocean_tensor_handle_t contiguous = NULL;
    const ocean_tensor_handle_t source = !ocean_tensor_is_contiguous(tensor)
        ? (contiguous = ocean_tensor_contiguous(tensor))
        : tensor;

    if (source->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t result = ocean_tensor_ternary_quantize_cpu(source);
        if (contiguous) ocean_tensor_release(contiguous);
        return result;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        source->shape, source->ndim, source->dtype, OCEAN_TENSOR_GPU
    );
    if (source->size != 0) {
        ocean_tensor_opencl_ternary_quantize(source, result);
    }
    if (contiguous) ocean_tensor_release(contiguous);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_gelu(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor.gelu on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.gelu currently requires float32");
    }

    if (tensor->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_gelu_cpu(tensor);
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t source = ocean_tensor_is_contiguous(tensor)
        ? tensor : ocean_tensor_contiguous(tensor);
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        source->shape, source->ndim, source->dtype, OCEAN_TENSOR_GPU
    );
    if (source->size != 0) {
        ocean_tensor_opencl_gelu(
            source,
            NULL,
            result,
            OCEAN_TENSOR_OPENCL_KERNEL_GELU_FLOAT32
        );
    }
    if (source != tensor) ocean_tensor_release(source);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_gelu_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input
) {
    if (!upstream || !input) {
        ocean_tensor_fail("Tensor.gelu backward requires non-null tensors");
    }
    if (upstream->dtype != OCEAN_TENSOR_FLOAT32 ||
        input->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.gelu backward currently requires float32");
    }
    if (upstream->device != input->device ||
        upstream->size != input->size ||
        upstream->ndim != input->ndim) {
        ocean_tensor_fail("Tensor.gelu backward shape/device mismatch");
    }
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        if (upstream->shape[axis] != input->shape[axis]) {
            ocean_tensor_fail("Tensor.gelu backward shape mismatch");
        }
    }

    if (input->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_gelu_backward_cpu(upstream, input);
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t upstream_source = ocean_tensor_is_contiguous(upstream)
        ? upstream : ocean_tensor_contiguous(upstream);
    ocean_tensor_handle_t input_source = ocean_tensor_is_contiguous(input)
        ? input : ocean_tensor_contiguous(input);
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        input_source->shape, input_source->ndim,
        input_source->dtype, OCEAN_TENSOR_GPU
    );
    if (input_source->size != 0) {
        ocean_tensor_opencl_gelu(
            upstream_source,
            input_source,
            result,
            OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32
        );
    }
    if (upstream_source != upstream) ocean_tensor_release(upstream_source);
    if (input_source != input) ocean_tensor_release(input_source);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_embedding_forward(
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t indices
) {
    if (!weight || !indices) {
        ocean_tensor_fail("Embedding.forward requires non-null tensors");
    }
    if (weight->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Embedding weights must be Tensor[float32]");
    }
    if (indices->dtype != OCEAN_TENSOR_INT64) {
        ocean_tensor_fail("Embedding indices must be Tensor[int64]");
    }
    if (weight->ndim != 2 || indices->ndim < 1) {
        ocean_tensor_fail(
            "Embedding expects weight [V,D] and indices rank >= 1"
        );
    }

    size_t vocab = weight->shape[0];
    size_t dim = weight->shape[1];
    size_t count = indices->size;
    (void)vocab;
    if (dim != 0 && count > SIZE_MAX / dim) {
        ocean_tensor_fail("Embedding output is too large");
    }

    ocean_tensor_handle_t contiguous_weight =
        ocean_tensor_contiguous(weight);
    ocean_tensor_handle_t contiguous_indices =
        ocean_tensor_contiguous(indices);
    ocean_tensor_handle_t result = NULL;

    if (weight->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_indices = contiguous_indices;
        if (cpu_indices->device != OCEAN_TENSOR_CPU) {
            cpu_indices = ocean_tensor_to(cpu_indices, "cpu");
        }
        result = ocean_tensor_embedding_forward_cpu(
            contiguous_weight,
            cpu_indices
        );
        if (cpu_indices != contiguous_indices) ocean_tensor_release(cpu_indices);
    } else {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        ocean_tensor_handle_t gpu_indices = contiguous_indices;
        if (gpu_indices->device != OCEAN_TENSOR_GPU) {
            gpu_indices = ocean_tensor_to(gpu_indices, "gpu");
        }
        size_t output_shape[indices->ndim + 1];
        for (size_t axis = 0; axis < indices->ndim; ++axis) {
            output_shape[axis] = indices->shape[axis];
        }
        output_shape[indices->ndim] = dim;
        result = ocean_tensor_alloc_uninitialized(
            output_shape,
            indices->ndim + 1,
            OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_GPU
        );
        if (count && dim) {
            if (count > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
                dim > (size_t)INT32_MAX || count > SIZE_MAX / dim ||
                count * dim > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_weight);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail(
                    "Embedding is too large for OpenCL kernel indexing"
                );
            }
            size_t error_shape[1] = {1};
            ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
                error_shape, 1, "int32", "gpu"
            );
            ocean_tensor_opencl_embedding_forward(
                contiguous_weight,
                gpu_indices,
                result,
                error,
                (int)count,
                (int)vocab,
                (int)dim
            );
            int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
            ocean_tensor_release(error);
            if (error_value != 0) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_weight);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail("Embedding token id is out of range");
            }
        }
        if (gpu_indices != contiguous_indices) ocean_tensor_release(gpu_indices);
#else
        ocean_tensor_release(contiguous_weight);
        ocean_tensor_release(contiguous_indices);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    ocean_tensor_release(contiguous_weight);
    ocean_tensor_release(contiguous_indices);
    return result;
}

ocean_tensor_handle_t ocean_tensor_embedding_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
) {
    if (!upstream || !indices) {
        ocean_tensor_fail("Embedding.backward requires non-null tensors");
    }
    if (upstream->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Embedding gradient must be Tensor[float32]");
    }
    if (indices->dtype != OCEAN_TENSOR_INT64) {
        ocean_tensor_fail("Embedding indices must be Tensor[int64]");
    }
    size_t count = indices->size;
    if (dim != 0 && count > SIZE_MAX / dim) {
        ocean_tensor_fail("Embedding gradient metadata is too large");
    }
    if (upstream->ndim != indices->ndim + 1) {
        ocean_tensor_fail("Embedding gradient rank does not match indices");
    }
    for (size_t axis = 0; axis < indices->ndim; ++axis) {
        if (upstream->shape[axis] != indices->shape[axis]) {
            ocean_tensor_fail("Embedding gradient shape does not match indices");
        }
    }
    if (upstream->shape[indices->ndim] != dim) {
        ocean_tensor_fail("Embedding gradient width does not match weight");
    }
    if (upstream->size != count * dim) {
        ocean_tensor_fail("Embedding gradient shape does not match indices");
    }

    ocean_tensor_handle_t contiguous_upstream =
        ocean_tensor_contiguous(upstream);
    ocean_tensor_handle_t contiguous_indices =
        ocean_tensor_contiguous(indices);
    ocean_tensor_handle_t result = NULL;

    if (upstream->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_indices = contiguous_indices;
        if (cpu_indices->device != OCEAN_TENSOR_CPU) {
            cpu_indices = ocean_tensor_to(cpu_indices, "cpu");
        }
        result = ocean_tensor_embedding_backward_cpu(
            contiguous_upstream,
            cpu_indices,
            vocab,
            dim
        );
        if (cpu_indices != contiguous_indices) ocean_tensor_release(cpu_indices);
    } else {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        ocean_tensor_handle_t gpu_indices = contiguous_indices;
        if (gpu_indices->device != OCEAN_TENSOR_GPU) {
            gpu_indices = ocean_tensor_to(gpu_indices, "gpu");
        }
        size_t gradient_shape[2] = {vocab, dim};
        result = ocean_tensor_alloc_zeros(
            gradient_shape, 2, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
        );
        if (count && dim) {
            if (count > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
                dim > (size_t)INT32_MAX || count * dim > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_upstream);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail(
                    "Embedding is too large for OpenCL kernel indexing"
                );
            }
            size_t error_shape[1] = {1};
            ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
                error_shape, 1, "int32", "gpu"
            );
            ocean_tensor_opencl_embedding_backward(
                contiguous_upstream,
                gpu_indices,
                result,
                error,
                (int)count,
                (int)vocab,
                (int)dim
            );
            int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
            ocean_tensor_release(error);
            if (error_value != 0) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_upstream);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail("Embedding token id is out of range");
            }
        }
        if (gpu_indices != contiguous_indices) ocean_tensor_release(gpu_indices);
#else
        ocean_tensor_release(contiguous_upstream);
        ocean_tensor_release(contiguous_indices);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    ocean_tensor_release(contiguous_upstream);
    ocean_tensor_release(contiguous_indices);
    return result;
}

static void ocean_tensor_validate_cross_entropy_shapes(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets
) {
    if (!logits || !targets) {
        ocean_tensor_fail("CrossEntropyLoss requires non-null tensors");
    }
    if (logits->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("CrossEntropyLoss logits must be Tensor[float32]");
    }
    if (targets->dtype != OCEAN_TENSOR_INT64) {
        ocean_tensor_fail(
            "CrossEntropyLoss targets must be Tensor[int64]"
        );
    }
    if (logits->ndim < 2 || targets->ndim != logits->ndim - 1) {
        ocean_tensor_fail(
            "CrossEntropyLoss expects logits [...,V] and targets [...]"
        );
    }
    for (size_t axis = 0; axis < targets->ndim; ++axis) {
        if (logits->shape[axis] != targets->shape[axis]) {
            ocean_tensor_fail(
                "CrossEntropyLoss target shape must match logits prefix"
            );
        }
    }
    if (logits->shape[logits->ndim - 1] == 0 || targets->size == 0) {
        ocean_tensor_fail(
            "CrossEntropyLoss requires non-empty vocab and targets"
        );
    }
}

ocean_tensor_handle_t ocean_tensor_cross_entropy_forward(
    ocean_tensor_handle_t logits,
    ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
) {
    if (!probabilities_out) {
        ocean_tensor_fail(
            "CrossEntropyLoss forward requires a probabilities output"
        );
    }
    *probabilities_out = NULL;
    ocean_tensor_validate_cross_entropy_shapes(logits, targets);

    size_t rows = targets->size;
    size_t vocab = logits->shape[logits->ndim - 1];
    if (vocab != 0 && rows > SIZE_MAX / vocab) {
        ocean_tensor_fail("CrossEntropyLoss tensor is too large");
    }

    ocean_tensor_handle_t contiguous_logits =
        ocean_tensor_is_contiguous(logits)
        ? logits : ocean_tensor_contiguous(logits);
    ocean_tensor_handle_t contiguous_targets =
        ocean_tensor_is_contiguous(targets)
        ? targets : ocean_tensor_contiguous(targets);
    ocean_tensor_handle_t result = NULL;

    if (logits->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_targets = contiguous_targets;
        if (cpu_targets->device != OCEAN_TENSOR_CPU) {
            cpu_targets = ocean_tensor_to(cpu_targets, "cpu");
        }
        result = ocean_tensor_cross_entropy_forward_cpu(
            contiguous_logits,
            cpu_targets,
            probabilities_out
        );
        if (cpu_targets != contiguous_targets) ocean_tensor_release(cpu_targets);
    } else {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        if (rows > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
            rows * vocab > (size_t)INT32_MAX) {
            if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
            if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
            ocean_tensor_fail(
                "CrossEntropyLoss is too large for OpenCL kernel indexing"
            );
        }
        ocean_tensor_handle_t gpu_targets = contiguous_targets;
        if (gpu_targets->device != OCEAN_TENSOR_GPU) {
            gpu_targets = ocean_tensor_to(gpu_targets, "gpu");
        }
        ocean_tensor_handle_t probabilities = ocean_tensor_alloc_uninitialized(
            logits->shape,
            logits->ndim,
            OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_GPU
        );
        size_t row_shape[1] = {rows};
        ocean_tensor_handle_t row_losses = ocean_tensor_alloc_uninitialized(
            row_shape, 1, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
        );
        size_t error_shape[1] = {1};
        ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
            error_shape, 1, "int32", "gpu"
        );
        ocean_tensor_opencl_cross_entropy_forward(
            contiguous_logits,
            gpu_targets,
            probabilities,
            row_losses,
            error,
            (int)rows,
            (int)vocab
        );
        int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
        ocean_tensor_release(error);
        if (error_value != 0) {
            ocean_tensor_release(row_losses);
            ocean_tensor_release(probabilities);
            if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
            if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
            if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }

        ocean_tensor_handle_t total = ocean_tensor_sum_dim(
            row_losses, -1, false
        );
        ocean_tensor_handle_t mean = ocean_tensor_scalar(
            total, (double)rows, OCEAN_TENSOR_DIV
        );
        size_t loss_shape[2] = {1, 1};
        ocean_tensor_handle_t loss = ocean_tensor_reshape(
            mean, loss_shape, 2
        );
        ocean_tensor_release(mean);
        ocean_tensor_release(total);
        ocean_tensor_release(row_losses);
        *probabilities_out = probabilities;
        result = loss;
        if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
#else
        if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
        if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
    if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
    return result;
}

ocean_tensor_handle_t ocean_tensor_cross_entropy_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t targets
) {
    ocean_tensor_validate_cross_entropy_shapes(probabilities, targets);
    if (!upstream || upstream->dtype != OCEAN_TENSOR_FLOAT32 ||
        upstream->size != 1) {
        ocean_tensor_fail(
            "CrossEntropyLoss backward requires a float32 scalar upstream"
        );
    }
    if (upstream->device != probabilities->device) {
        ocean_tensor_fail(
            "CrossEntropyLoss backward requires matching Tensor devices"
        );
    }

    size_t rows = targets->size;
    size_t vocab = probabilities->shape[probabilities->ndim - 1];
    if (vocab != 0 && rows > SIZE_MAX / vocab) {
        ocean_tensor_fail("CrossEntropyLoss gradient is too large");
    }

    ocean_tensor_handle_t contiguous_upstream =
        ocean_tensor_is_contiguous(upstream)
        ? upstream : ocean_tensor_contiguous(upstream);
    ocean_tensor_handle_t contiguous_probabilities =
        ocean_tensor_is_contiguous(probabilities)
        ? probabilities : ocean_tensor_contiguous(probabilities);
    ocean_tensor_handle_t contiguous_targets =
        ocean_tensor_is_contiguous(targets)
        ? targets : ocean_tensor_contiguous(targets);
    ocean_tensor_handle_t result = NULL;

    if (probabilities->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_targets = contiguous_targets;
        if (cpu_targets->device != OCEAN_TENSOR_CPU) {
            cpu_targets = ocean_tensor_to(cpu_targets, "cpu");
        }
        result = ocean_tensor_cross_entropy_backward_cpu(
            contiguous_upstream,
            contiguous_probabilities,
            cpu_targets
        );
        if (cpu_targets != contiguous_targets) ocean_tensor_release(cpu_targets);
    } else {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        if (rows > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
            rows * vocab > (size_t)INT32_MAX) {
            ocean_tensor_fail(
                "CrossEntropyLoss gradient is too large for OpenCL indexing"
            );
        }
        ocean_tensor_handle_t gpu_targets = contiguous_targets;
        if (gpu_targets->device != OCEAN_TENSOR_GPU) {
            gpu_targets = ocean_tensor_to(gpu_targets, "gpu");
        }
        result = ocean_tensor_alloc_uninitialized(
            probabilities->shape,
            probabilities->ndim,
            OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_GPU
        );
        size_t error_shape[1] = {1};
        ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
            error_shape, 1, "int32", "gpu"
        );
        ocean_tensor_opencl_cross_entropy_backward(
            contiguous_upstream,
            contiguous_probabilities,
            gpu_targets,
            result,
            error,
            (int)rows,
            (int)vocab
        );
        int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
        ocean_tensor_release(error);
        if (error_value != 0) {
            ocean_tensor_release(result);
            if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }
        if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
#else
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    if (contiguous_upstream != upstream) ocean_tensor_release(contiguous_upstream);
    if (contiguous_probabilities != probabilities) ocean_tensor_release(contiguous_probabilities);
    if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
    return result;
}

void ocean_tensor_copy_into(
    ocean_tensor_handle_t destination,
    ocean_tensor_handle_t source
) {
    if (!destination || !source) {
        ocean_tensor_fail("Tensor.copy_into requires non-null tensors");
    }
    if (
        destination->dtype != source->dtype
        || destination->ndim != source->ndim
        || destination->size != source->size
    ) {
        ocean_tensor_fail("Tensor.copy_into metadata mismatch");
    }

    for (size_t axis = 0; axis < destination->ndim; ++axis) {
        if (destination->shape[axis] != source->shape[axis]) {
            ocean_tensor_fail("Tensor.copy_into shape mismatch");
        }
    }

    if (destination->device == source->device) {
        ocean_tensor_backend_for_device(destination->device)->copy(
            destination,
            source
        );
        return;
    }

    if (destination->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_backend_for_device(source->device)->read(
            source,
            destination->cpu_data
        );
        return;
    }

    if (source->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_backend_for_device(destination->device)->write(
            destination,
            source->cpu_data
        );
        return;
    }

    ocean_tensor_fail("Tensor.copy_into unsupported device pair");
}

ocean_tensor_handle_t ocean_tensor_to(
    ocean_tensor_handle_t tensor,
    const char *device
) {
    if (!tensor) ocean_tensor_fail("cannot move a null Tensor");
    int target = ocean_tensor_parse_device(device);
    if ((ocean_tensor_backend_kind)target == tensor->device) {
        return ocean_tensor_copy(tensor);
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc(
        tensor->shape, tensor->ndim, tensor->dtype, target
    );
    const ocean_tensor_backend_ops *source_backend =
        ocean_tensor_backend_for_device(tensor->device);
    const ocean_tensor_backend_ops *target_backend =
        ocean_tensor_backend_for_device((ocean_tensor_backend_kind)target);
    target_backend->allocate(result);
    if (target == OCEAN_TENSOR_CPU) {
        source_backend->read(tensor, result->cpu_data);
    } else {
        target_backend->write(result, tensor->cpu_data);
    }
    return result;
}

static float ocean_tensor_half_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | (exponent + 112u) << 23 | (mantissa << 13);
    }
    union { uint32_t bits; float value; } result = {bits};
    return result.value;
}

static uint16_t ocean_tensor_float_to_half(float value) {
    union { uint32_t bits; float value; } input = {0};
    input.value = value;
    uint32_t sign = (input.bits >> 16) & 0x8000u;
    uint32_t exponent = (input.bits >> 23) & 0xffu;
    uint32_t mantissa = input.bits & 0x7fffffu;
    if (exponent == 255) {
        return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x0200u : 0));
    }
    int unbiased = (int)exponent - 127;
    if (unbiased < -14) {
        if (unbiased < -24) return (uint16_t)sign;
        uint32_t shifted = (mantissa | 0x800000u) >> (uint32_t)(-unbiased - 14 + 13);
        return (uint16_t)(sign | shifted);
    }
    if (unbiased > 15) return (uint16_t)(sign | 0x7c00u);
    uint32_t half_exponent = (uint32_t)(unbiased + 15);
    uint32_t half_mantissa = mantissa >> 13;
    if (mantissa & 0x1000u) ++half_mantissa;
    if (half_mantissa == 0x0400u) {
        half_mantissa = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (half_exponent << 10) | half_mantissa);
}

static long double ocean_tensor_read_scalar(
    const ocean_tensor_handle_t tensor,
    size_t index
) {
    const unsigned char *data = (const unsigned char *)tensor->cpu_data;
    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: return ((const bool *)data)[index] ? 1.0L : 0.0L;
        case OCEAN_TENSOR_INT8: return ((const int8_t *)data)[index];
        case OCEAN_TENSOR_INT16: return ((const int16_t *)data)[index];
        case OCEAN_TENSOR_INT32: return ((const int32_t *)data)[index];
        case OCEAN_TENSOR_INT64: return (long double)((const int64_t *)data)[index];
        case OCEAN_TENSOR_UINT8: return ((const uint8_t *)data)[index];
        case OCEAN_TENSOR_UINT16: return ((const uint16_t *)data)[index];
        case OCEAN_TENSOR_UINT32: return ((const uint32_t *)data)[index];
        case OCEAN_TENSOR_UINT64: return (long double)((const uint64_t *)data)[index];
        case OCEAN_TENSOR_FLOAT16:
            return (long double)ocean_tensor_half_to_float(((const uint16_t *)data)[index]);
        case OCEAN_TENSOR_FLOAT32: return ((const float *)data)[index];
        case OCEAN_TENSOR_FLOAT64: return ((const double *)data)[index];
    }
    ocean_tensor_fail("invalid Tensor scalar type");
    return 0.0L;
}

static void ocean_tensor_write_scalar(
    const ocean_tensor_handle_t tensor,
    size_t index,
    long double value
) {
    unsigned char *data = (unsigned char *)tensor->cpu_data;
    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: ((bool *)data)[index] = value != 0.0L; return;
        case OCEAN_TENSOR_INT8: ((int8_t *)data)[index] = (int8_t)value; return;
        case OCEAN_TENSOR_INT16: ((int16_t *)data)[index] = (int16_t)value; return;
        case OCEAN_TENSOR_INT32: ((int32_t *)data)[index] = (int32_t)value; return;
        case OCEAN_TENSOR_INT64: ((int64_t *)data)[index] = (int64_t)value; return;
        case OCEAN_TENSOR_UINT8: ((uint8_t *)data)[index] = (uint8_t)value; return;
        case OCEAN_TENSOR_UINT16: ((uint16_t *)data)[index] = (uint16_t)value; return;
        case OCEAN_TENSOR_UINT32: ((uint32_t *)data)[index] = (uint32_t)value; return;
        case OCEAN_TENSOR_UINT64: ((uint64_t *)data)[index] = (uint64_t)value; return;
        case OCEAN_TENSOR_FLOAT16:
            ((uint16_t *)data)[index] = ocean_tensor_float_to_half((float)value);
            return;
        case OCEAN_TENSOR_FLOAT32: ((float *)data)[index] = (float)value; return;
        case OCEAN_TENSOR_FLOAT64: ((double *)data)[index] = (double)value; return;
    }
    ocean_tensor_fail("invalid Tensor scalar type");
}

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static bool ocean_tensor_contains_zero(const ocean_tensor_handle_t tensor) {
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    bool found = false;
    for (size_t index = 0; index < cpu->size; ++index) {
        if (ocean_tensor_read_scalar(cpu, index) == 0.0L) {
            found = true;
            break;
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return found;
}
#endif

static long double ocean_tensor_apply_binary(
    long double left,
    long double right,
    int operation
) {
    switch (operation) {
        case OCEAN_TENSOR_ADD: return left + right;
        case OCEAN_TENSOR_SUB: return left - right;
        case OCEAN_TENSOR_MUL: return left * right;
        case OCEAN_TENSOR_DIV:
            if (right == 0.0L) ocean_tensor_fail("Tensor division by zero");
            return left / right;
    }
    ocean_tensor_fail("invalid Tensor binary operation");
    return 0.0L;
}

static void ocean_tensor_broadcast_shape(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    size_t **shape_out,
    size_t *ndim_out
) {
    size_t ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating broadcast shape");
    for (size_t axis = 0; axis < ndim; ++axis) {
        size_t left_axis = axis + left->ndim >= ndim
            ? left->shape[axis + left->ndim - ndim] : 1;
        size_t right_axis = axis + right->ndim >= ndim
            ? right->shape[axis + right->ndim - ndim] : 1;
        if (left_axis != right_axis && left_axis != 1 && right_axis != 1) {
            free(shape);
            ocean_tensor_fail("Tensor shapes are not broadcast-compatible");
        }
        /* Broadcasting dimension 0 with dimension 1 produces dimension 0.
           Using max(left_axis, right_axis) incorrectly turns that case into
           a non-empty result and later dereferences a NULL data buffer. */
        if (left_axis == right_axis) shape[axis] = left_axis;
        else if (left_axis == 1) shape[axis] = right_axis;
        else if (right_axis == 1) shape[axis] = left_axis;
    }
    *shape_out = shape;
    *ndim_out = ndim;
}

static size_t ocean_tensor_broadcast_offset(
    const ocean_tensor_handle_t tensor,
    const size_t *result_shape,
    size_t result_ndim,
    size_t linear
) {
    size_t remaining = linear;
    size_t offset = 0;
    for (size_t axis = result_ndim; axis-- > 0;) {
        size_t coordinate = result_shape[axis]
            ? remaining % result_shape[axis] : 0;
        remaining = result_shape[axis]
            ? remaining / result_shape[axis] : 0;
        if (axis + tensor->ndim >= result_ndim) {
            size_t tensor_axis = axis + tensor->ndim - result_ndim;
            if (tensor->shape[tensor_axis] != 1) {
                offset += coordinate * tensor->strides[tensor_axis];
            }
        }
    }
    return offset;
}

static ocean_tensor_handle_t ocean_tensor_binary_cpu(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    int operation
) {
    size_t *shape = NULL;
    size_t ndim = 0;
    ocean_tensor_broadcast_shape(left, right, &shape, &ndim);
    if (shape == NULL || ndim == 0) {
        ocean_tensor_fail(
            "Tensor broadcast produced invalid metadata"
        );
    }


    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, ndim, left->dtype, OCEAN_TENSOR_CPU
    );

    bool same_shape = left->ndim == right->ndim && left->ndim == ndim;
    if (same_shape) {
        for (size_t axis = 0; axis < ndim; ++axis) {
            if (left->shape[axis] != right->shape[axis]) {
                same_shape = false;
                break;
            }
        }
    }

    if (same_shape && left->dtype == OCEAN_TENSOR_FLOAT32) {
        const float *restrict a = (const float *)left->cpu_data;
        const float *restrict b = (const float *)right->cpu_data;
        float *restrict out = (float *)result->cpu_data;
        size_t size = result->size;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
                break;
            case OCEAN_TENSOR_DIV:
                for (size_t i = 0; i < size; ++i) {
                    if (b[i] == 0.0f) {
                        free(shape);
                        ocean_tensor_release(result);
                        ocean_tensor_fail("Tensor division by zero");
                    }
                    out[i] = a[i] / b[i];
                }
                break;
            default:
                free(shape);
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }

        free(shape);
        return result;
    }

    if (same_shape && left->dtype == OCEAN_TENSOR_FLOAT64) {
        const double *restrict a = (const double *)left->cpu_data;
        const double *restrict b = (const double *)right->cpu_data;
        double *restrict out = (double *)result->cpu_data;
        size_t size = result->size;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
                break;
            case OCEAN_TENSOR_DIV:
                for (size_t i = 0; i < size; ++i) {
                    if (b[i] == 0.0) {
                        free(shape);
                        ocean_tensor_release(result);
                        ocean_tensor_fail("Tensor division by zero");
                    }
                    out[i] = a[i] / b[i];
                }
                break;
            default:
                free(shape);
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }

        free(shape);
        return result;
    }

    if (same_shape) {
        for (size_t linear = 0; linear < result->size; ++linear) {
            ocean_tensor_write_scalar(
                result,
                linear,
                ocean_tensor_apply_binary(
                    ocean_tensor_read_scalar(left, linear),
                    ocean_tensor_read_scalar(right, linear),
                    operation
                )
            );
        }
        free(shape);
        return result;
    }

    for (size_t linear = 0; linear < result->size; ++linear) {
        size_t left_index = ocean_tensor_broadcast_offset(
            left, shape, ndim, linear
        );
        size_t right_index = ocean_tensor_broadcast_offset(
            right, shape, ndim, linear
        );
        ocean_tensor_write_scalar(
            result,
            linear,
            ocean_tensor_apply_binary(
                ocean_tensor_read_scalar(left, left_index),
                ocean_tensor_read_scalar(right, right_index),
                operation
            )
        );
    }

    free(shape);
    return result;
}


static ocean_tensor_handle_t ocean_tensor_scalar_cpu(
    const ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    size_t size = tensor->size;

    if (tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        const float *restrict input = (const float *)tensor->cpu_data;
        float *restrict out = (float *)result->cpu_data;
        float s = (float)scalar;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] + s;
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] - s;
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] * s;
                break;
            case OCEAN_TENSOR_DIV:
                if (s == 0.0f) {
                    ocean_tensor_release(result);
                    ocean_tensor_fail("Tensor division by zero");
                }
                for (size_t i = 0; i < size; ++i) out[i] = input[i] / s;
                break;
            default:
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }
        return result;
    }

    if (tensor->dtype == OCEAN_TENSOR_FLOAT64) {
        const double *restrict input = (const double *)tensor->cpu_data;
        double *restrict out = (double *)result->cpu_data;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] + scalar;
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] - scalar;
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] * scalar;
                break;
            case OCEAN_TENSOR_DIV:
                if (scalar == 0.0) {
                    ocean_tensor_release(result);
                    ocean_tensor_fail("Tensor division by zero");
                }
                for (size_t i = 0; i < size; ++i) out[i] = input[i] / scalar;
                break;
            default:
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }
        return result;
    }

    for (size_t i = 0; i < size; ++i) {
        ocean_tensor_write_scalar(
            result,
            i,
            ocean_tensor_apply_binary(
                ocean_tensor_read_scalar(tensor, i),
                (long double)scalar,
                operation
            )
        );
    }

    return result;
}

static ocean_tensor_handle_t ocean_tensor_ternary_quantize_cpu(
    const ocean_tensor_handle_t tensor
) {
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.ternary_quantize currently requires float32");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    if (tensor->size == 0) return result;

    const float *input = (const float *)tensor->cpu_data;
    float *output = (float *)result->cpu_data;
    double sum_abs = 0.0;
    for (size_t index = 0; index < tensor->size; ++index) {
        sum_abs += fabs((double)input[index]);
    }

    float scale = (float)(sum_abs / (double)tensor->size);
    if (scale < 1.0e-8f) scale = 1.0e-8f;
    float threshold = 0.5f * scale;
    for (size_t index = 0; index < tensor->size; ++index) {
        float value = input[index];
        output[index] = value > threshold
            ? scale
            : (value < -threshold ? -scale : 0.0f);
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_gelu_cpu(
    const ocean_tensor_handle_t tensor
) {
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    const float *input = (const float *)tensor->cpu_data;
    float *output = (float *)result->cpu_data;
    const float coefficient = 0.7978845608028654f;
    const float cubic = 0.044715f;
    for (size_t index = 0; index < tensor->size; ++index) {
        float value = input[index];
        float value_squared = value * value;
        float argument = coefficient * (
            value + cubic * value * value_squared
        );
        output[index] = 0.5f * value * (1.0f + tanhf(argument));
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_gelu_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t input
) {
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        input->shape, input->ndim, input->dtype, OCEAN_TENSOR_CPU
    );
    const float *upstream_data = (const float *)upstream->cpu_data;
    const float *input_data = (const float *)input->cpu_data;
    float *output = (float *)result->cpu_data;
    const float coefficient = 0.7978845608028654f;
    const float cubic = 0.044715f;
    for (size_t index = 0; index < input->size; ++index) {
        float value = input_data[index];
        float argument = coefficient * (
            value + cubic * value * value * value
        );
        float tanh_argument = tanhf(argument);
        float derivative = 0.5f * (1.0f + tanh_argument);
        derivative += 0.5f * value
            * (1.0f - tanh_argument * tanh_argument)
            * coefficient
            * (1.0f + 3.0f * cubic * value * value);
        output[index] = upstream_data[index] * derivative;
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_embedding_forward_cpu(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices
) {
    size_t dim = weight->shape[1];
    size_t output_shape[indices->ndim + 1];
    for (size_t axis = 0; axis < indices->ndim; ++axis) {
        output_shape[axis] = indices->shape[axis];
    }
    output_shape[indices->ndim] = dim;
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape,
        indices->ndim + 1,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );

    size_t vocab = weight->shape[0];
    for (size_t position = 0; position < indices->size; ++position) {
        int64_t token = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)indices,
            position
        );
        if (token < 0 || (uint64_t)token >= (uint64_t)vocab) {
            ocean_tensor_release(result);
            ocean_tensor_fail("Embedding token id is out of range");
        }
        size_t row = (size_t)token;
        for (size_t feature = 0; feature < dim; ++feature) {
            ocean_tensor_write_scalar(
                result,
                position * dim + feature,
                (long double)ocean_tensor_get_flat_f32(
                    (ocean_tensor_handle_t)weight,
                    row * dim + feature
                )
            );
        }
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_embedding_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
) {
    size_t gradient_shape[2] = {vocab, dim};
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        gradient_shape,
        2,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );

    for (size_t position = 0; position < indices->size; ++position) {
        int64_t token = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)indices,
            position
        );
        if (token < 0 || (uint64_t)token >= (uint64_t)vocab) {
            ocean_tensor_release(result);
            ocean_tensor_fail("Embedding token id is out of range");
        }
        size_t row = (size_t)token;
        for (size_t feature = 0; feature < dim; ++feature) {
            size_t gradient_index = row * dim + feature;
            float value = ocean_tensor_get_flat_f32(
                (ocean_tensor_handle_t)upstream,
                position * dim + feature
            );
            float previous = ocean_tensor_get_flat_f32(result, gradient_index);
            ocean_tensor_write_scalar(
                result,
                gradient_index,
                (long double)(previous + value)
            );
        }
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cross_entropy_forward_cpu(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
) {
    size_t vocab = logits->shape[logits->ndim - 1];
    size_t rows = targets->size;
    ocean_tensor_handle_t probabilities = ocean_tensor_alloc_uninitialized(
        logits->shape,
        logits->ndim,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );
    double total_loss = 0.0;

    for (size_t row = 0; row < rows; ++row) {
        size_t offset = row * vocab;
        float maximum = -INFINITY;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float value = ocean_tensor_get_flat_f32(
                (ocean_tensor_handle_t)logits,
                offset + cls
            );
            if (value > maximum) maximum = value;
        }

        double denominator = 0.0;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float exponential = expf(
                ocean_tensor_get_flat_f32(
                    (ocean_tensor_handle_t)logits,
                    offset + cls
                ) - maximum
            );
            ocean_tensor_write_scalar(
                probabilities,
                offset + cls,
                (long double)exponential
            );
            denominator += (double)exponential;
        }

        int64_t target = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)targets,
            row
        );
        if (target < 0 || (uint64_t)target >= (uint64_t)vocab) {
            ocean_tensor_release(probabilities);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }
        float target_logit = ocean_tensor_get_flat_f32(
            (ocean_tensor_handle_t)logits,
            offset + (size_t)target
        );
        total_loss += log(denominator) + (double)maximum
            - (double)target_logit;

        for (size_t cls = 0; cls < vocab; ++cls) {
            float probability = ocean_tensor_get_flat_f32(
                probabilities,
                offset + cls
            );
            ocean_tensor_write_scalar(
                probabilities,
                offset + cls,
                (long double)((double)probability / denominator)
            );
        }
    }

    ocean_tensor_handle_t loss = ocean_tensor_zeros(1, 1, "cpu");
    ocean_tensor_fill(loss, total_loss / (double)rows);
    *probabilities_out = probabilities;
    return loss;
}

static ocean_tensor_handle_t ocean_tensor_cross_entropy_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets
) {
    size_t vocab = probabilities->shape[probabilities->ndim - 1];
    size_t rows = targets->size;
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        probabilities->shape,
        probabilities->ndim,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );
    float scale = ocean_tensor_get_flat_f32(
        (ocean_tensor_handle_t)upstream,
        0
    ) / (float)rows;

    for (size_t row = 0; row < rows; ++row) {
        int64_t target = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)targets,
            row
        );
        if (target < 0 || (uint64_t)target >= (uint64_t)vocab) {
            ocean_tensor_release(result);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }
        size_t offset = row * vocab;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float value = ocean_tensor_get_flat_f32(
                (ocean_tensor_handle_t)probabilities,
                offset + cls
            );
            if (cls == (size_t)target) value -= 1.0f;
            ocean_tensor_write_scalar(
                result,
                offset + cls,
                (long double)(value * scale)
            );
        }
    }
    return result;
}


#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static int ocean_tensor_same_shape(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right
) {
    if (left->ndim != right->ndim) return 0;
    for (size_t axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return 0;
    }
    return 1;
}

static void ocean_tensor_opencl_binary(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    ocean_tensor_handle_t result,
    int operation,
    cl_kernel kernel
) {
    int size = (int)left->size;
    if (left->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &left->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &right->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &result->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &operation),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size),
        "clSetKernelArg"
    );
    size_t global_size = left->size ? left->size : 1;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_scalar(
    const ocean_tensor_handle_t tensor,
    ocean_tensor_handle_t result,
    double scalar,
    int operation,
    cl_kernel kernel,
    int integer_kernel
) {
    if (tensor->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    float scalar_value = (float)scalar;
    int integer_value = (int)scalar;
    int size = (int)tensor->size;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &tensor->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &result->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel, 2,
            integer_kernel ? sizeof(int) : sizeof(float),
            integer_kernel ? (const void *)&integer_value : (const void *)&scalar_value
        ),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &operation),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size),
        "clSetKernelArg"
    );
    size_t global_size = tensor->size ? tensor->size : 1;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_gelu(
    const ocean_tensor_handle_t first,
    const ocean_tensor_handle_t second,
    ocean_tensor_handle_t output,
    int key
) {
    if (first->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(key);
    int size = (int)first->size;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &first->gpu_data),
        "clSetKernelArg"
    );
    if (key == OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &second->gpu_data),
            "clSetKernelArg"
        );
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->gpu_data),
            "clSetKernelArg"
        );
    } else {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
            "clSetKernelArg"
        );
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel,
            key == OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32 ? 3 : 2,
            sizeof(int),
            &size
        ),
        "clSetKernelArg"
    );
    size_t global_size = first->size ? first->size : 1;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_rowwise(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output,
    ocean_tensor_opencl_kernel_key key,
    int width,
    float epsilon,
    int mean
) {
    if (input->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(key);
    int rows = width > 0 ? (int)(input->size / (size_t)width) : 0;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &width), "clSetKernelArg"
    );
    if (key == OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 4, sizeof(float), &epsilon),
            "clSetKernelArg"
        );
    } else if (key == OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 4, sizeof(int), &mean), "clSetKernelArg"
        );
    }
    if (rows == 0) return;
    size_t global_size = (size_t)rows;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_ternary_quantize(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output
) {
    if (input->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }

    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_QUANTIZE_FLOAT32
    );
    int size = (int)input->size;
    const size_t local_size = 64u;

    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel, 2, local_size * sizeof(float), NULL
        ),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &size),
        "clSetKernelArg"
    );

    size_t global_size = local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_cache_write(
    const ocean_tensor_handle_t cache,
    const ocean_tensor_handle_t value,
    int position
) {
    if (value->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CACHE_WRITE_FLOAT32
    );
    int heads = (int)cache->shape[1];
    int sequence = (int)cache->shape[2];
    int width = (int)cache->shape[3];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &cache->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &value->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &heads), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &sequence), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &width), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &position), "clSetKernelArg"
    );
    size_t global_size = value->size ? value->size : 1;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_cache_slice(
    const ocean_tensor_handle_t cache,
    const ocean_tensor_handle_t output,
    int start
) {
    if (output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CACHE_SLICE_FLOAT32
    );
    int heads = (int)cache->shape[1];
    int source_sequence = (int)cache->shape[2];
    int output_sequence = (int)output->shape[2];
    int width = (int)cache->shape[3];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &cache->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &heads), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &source_sequence),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &output_sequence),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &width), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &start), "clSetKernelArg"
    );
    size_t global_size = output->size ? output->size : 1;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_embedding_forward(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t output,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_FORWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &weight->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &index_count),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &dim),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)index_count * (size_t)dim;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_embedding_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_BACKWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &index_count),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &dim),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)index_count * (size_t)dim;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_cross_entropy_forward(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t row_losses,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_FORWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &logits->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &targets->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &probabilities->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &row_losses->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &rows),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)rows;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_cross_entropy_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_BACKWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &probabilities->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &targets->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &rows),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)rows;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_softmax_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t output,
    ocean_tensor_handle_t gradient,
    int width
) {
    if (upstream->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_BACKWARD_FLOAT32
    );
    int rows = width > 0 ? (int)(upstream->size / (size_t)width) : 0;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &width), "clSetKernelArg"
    );
    if (rows == 0) return;
    size_t global_size = (size_t)rows;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_layer_norm_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t gradient,
    int width,
    double epsilon
) {
    if (upstream->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_BACKWARD_FLOAT32
    );
    int rows = width > 0 ? (int)(upstream->size / (size_t)width) : 0;
    float epsilon_value = (float)epsilon;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &width), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(float), &epsilon_value),
        "clSetKernelArg"
    );
    if (rows == 0) return;
    size_t global_size = (size_t)rows;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_sgd_update(
    ocean_tensor_handle_t parameter,
    const ocean_tensor_handle_t gradient,
    double learning_rate
) {
    if (parameter->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_SGD_UPDATE_FLOAT32
    );
    int size = (int)parameter->size;
    float rate = (float)learning_rate;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameter->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &size), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(float), &rate), "clSetKernelArg"
    );
    if (size == 0) return;
    size_t global_size = parameter->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_adamw_update(
    ocean_tensor_handle_t parameter,
    const ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t first_moment,
    ocean_tensor_handle_t second_moment,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay,
    double bias_correction1,
    double bias_correction2
) {
    if (parameter->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_ADAMW_UPDATE_FLOAT32
    );
    int size = (int)parameter->size;
    float values[7] = {
        (float)learning_rate,
        (float)beta1,
        (float)beta2,
        (float)epsilon,
        (float)weight_decay,
        (float)bias_correction1,
        (float)bias_correction2,
    };
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameter->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &first_moment->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &second_moment->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size), "clSetKernelArg"
    );
    for (int index = 0; index < 7; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(5 + index), sizeof(float), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    if (size == 0) return;
    size_t global_size = parameter->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}
#endif

void ocean_tensor_cache_write(
    ocean_tensor_handle_t cache,
    ocean_tensor_handle_t value,
    int position
) {
    if (!cache || !value) {
        ocean_tensor_fail("Tensor.cache_write requires non-null tensors");
    }
    if (cache->dtype != OCEAN_TENSOR_FLOAT32 ||
        value->dtype != OCEAN_TENSOR_FLOAT32 ||
        cache->ndim != 4 || value->ndim != 4 ||
        value->shape[0] != cache->shape[0] ||
        value->shape[1] != cache->shape[1] ||
        value->shape[2] != 1 ||
        value->shape[3] != cache->shape[3] ||
        cache->device != value->device ||
        !ocean_tensor_is_contiguous(cache) ||
        !ocean_tensor_is_contiguous(value)) {
        ocean_tensor_fail("Tensor.cache_write metadata mismatch");
    }
    if (position < 0 || (size_t)position >= cache->shape[2]) {
        ocean_tensor_fail("Tensor.cache_write position is out of bounds");
    }

    size_t batches = cache->shape[0];
    size_t heads = cache->shape[1];
    size_t sequence = cache->shape[2];
    size_t width = cache->shape[3];
    if (cache->device == OCEAN_TENSOR_CPU) {
        float *destination = (float *)cache->cpu_data;
        const float *source = (const float *)value->cpu_data;
        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t head = 0; head < heads; ++head) {
                size_t destination_offset =
                    ((batch * heads + head) * sequence + (size_t)position) * width;
                size_t source_offset = (batch * heads + head) * width;
                memcpy(
                    destination + destination_offset,
                    source + source_offset,
                    width * sizeof(float)
                );
            }
        }
        return;
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (heads > (size_t)INT32_MAX || sequence > (size_t)INT32_MAX ||
        width > (size_t)INT32_MAX || value->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor.cache_write dimensions are too large for OpenCL");
    }
    ocean_tensor_opencl_cache_write(cache, value, position);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
}

static ocean_tensor_handle_t ocean_tensor_binary_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (operation == OCEAN_TENSOR_DIV && ocean_tensor_contains_zero(right)) {
        ocean_tensor_fail("Tensor division by zero");
    }
    if ((left->dtype == OCEAN_TENSOR_FLOAT32 || left->dtype == OCEAN_TENSOR_INT32) &&
        ocean_tensor_same_shape(left, right)) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            left->shape, left->ndim, left->dtype, OCEAN_TENSOR_GPU
        );
        cl_kernel kernel = ocean_tensor_opencl_get_kernel(
            left->dtype == OCEAN_TENSOR_INT32
                ? OCEAN_TENSOR_OPENCL_KERNEL_BINARY_INT32
                : OCEAN_TENSOR_OPENCL_KERNEL_BINARY_FLOAT32
        );
        ocean_tensor_opencl_binary(left, right, result, operation, kernel);
        return result;
    }
    ocean_tensor_handle_t left_cpu = ocean_tensor_to(left, "cpu");
    ocean_tensor_handle_t right_cpu = ocean_tensor_to(right, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_binary_cpu(
        left_cpu, right_cpu, operation
    );
    ocean_tensor_handle_t gpu_result = ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(left_cpu);
    ocean_tensor_release(right_cpu);
    ocean_tensor_release(cpu_result);
    return gpu_result;
#else
    (void)left;
    (void)right;
    (void)operation;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_binary(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
    if (!left || !right) ocean_tensor_fail("Tensor operation on null handle");
    if (left->dtype != right->dtype) {
        ocean_tensor_fail("Tensor operation requires matching dtypes");
    }
    if (left->device != right->device) {
        ocean_tensor_fail("Tensor operation requires matching devices");
    }
    return ocean_tensor_backend_for_device(left->device)->binary(
        left, right, operation
    );
}

static ocean_tensor_handle_t ocean_tensor_scalar_opencl(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->dtype == OCEAN_TENSOR_FLOAT32 || tensor->dtype == OCEAN_TENSOR_INT32) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        cl_kernel kernel = ocean_tensor_opencl_get_kernel(
            tensor->dtype == OCEAN_TENSOR_INT32
                ? OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_INT32
                : OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_FLOAT32
        );
        ocean_tensor_opencl_scalar(
            tensor, result, scalar, operation, kernel,
            tensor->dtype == OCEAN_TENSOR_INT32
        );
        return result;
    }
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_scalar_cpu(cpu, scalar, operation);
    ocean_tensor_handle_t gpu_result = ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(cpu);
    ocean_tensor_release(cpu_result);
    return gpu_result;
#else
    (void)tensor;
    (void)scalar;
    (void)operation;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_scalar(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    if (!tensor) ocean_tensor_fail("Tensor scalar operation on null handle");
    if (operation == OCEAN_TENSOR_DIV && scalar == 0.0) {
        ocean_tensor_fail("Tensor division by zero");
    }
    return ocean_tensor_backend_for_device(tensor->device)->scalar(
        tensor, scalar, operation
    );
}

ocean_tensor_handle_t ocean_tensor_reshape(
    ocean_tensor_handle_t tensor,
    const size_t *shape,
    size_t ndim
) {
    if (!tensor || !shape) {
        ocean_tensor_fail("Tensor reshape received null metadata");
    }
    if (ocean_tensor_elements_from_shape(shape, ndim) != tensor->size) {
        ocean_tensor_fail("Tensor reshape must preserve the number of elements");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, ndim, tensor->dtype, tensor->device
    );
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device(tensor->device);
    backend->copy(result, tensor);
    return result;
}


ocean_tensor_handle_t ocean_tensor_reshape_2d(
    ocean_tensor_handle_t tensor,
    int rows,
    int cols
) {
    if (rows < 0 || cols < 0) {
        ocean_tensor_fail("Tensor reshape dimensions must be non-negative");
    }
    size_t shape[2] = {(size_t)rows, (size_t)cols};
    return ocean_tensor_reshape(tensor, shape, 2);
}

ocean_tensor_handle_t ocean_tensor_transpose(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor transpose on null handle");
    if (tensor->ndim != 2) ocean_tensor_fail("Tensor transpose() expects 2D; use transpose_dims() for ND Tensor");
    return ocean_tensor_transpose_dims(tensor, 0, 1);
}

static ocean_tensor_handle_t ocean_tensor_restore_device(
    const ocean_tensor_handle_t source,
    ocean_tensor_handle_t cpu_result
) {
    if (source->device == OCEAN_TENSOR_CPU) return cpu_result;
    ocean_tensor_handle_t result = ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(cpu_result);
    return result;
}

ocean_tensor_handle_t ocean_tensor_row(ocean_tensor_handle_t tensor, int row) {
    if (!tensor || tensor->ndim != 2) {
        ocean_tensor_fail("Tensor row() currently expects a 2D Tensor");
    }
    if (row < 0 || (size_t)row >= tensor->shape[0]) {
        ocean_tensor_fail("Tensor row index out of bounds");
    }
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t shape[1] = {tensor->shape[1]};
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t column = 0; column < shape[0]; ++column) {
        size_t source_index = (size_t)row * cpu->strides[0]
            + column * cpu->strides[1];
        memcpy(
            (unsigned char *)cpu_result->cpu_data + column * tensor->item_size,
            (unsigned char *)cpu->cpu_data + source_index * tensor->item_size,
            tensor->item_size
        );
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, cpu_result);
}

ocean_tensor_handle_t ocean_tensor_column(ocean_tensor_handle_t tensor, int column) {
    if (!tensor || tensor->ndim != 2) {
        ocean_tensor_fail("Tensor column() currently expects a 2D Tensor");
    }
    if (column < 0 || (size_t)column >= tensor->shape[1]) {
        ocean_tensor_fail("Tensor column index out of bounds");
    }
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t shape[1] = {tensor->shape[0]};
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t row = 0; row < shape[0]; ++row) {
        size_t source_index = row * cpu->strides[0]
            + (size_t)column * cpu->strides[1];
        memcpy(
            (unsigned char *)cpu_result->cpu_data + row * tensor->item_size,
            (unsigned char *)cpu->cpu_data + source_index * tensor->item_size,
            tensor->item_size
        );
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, cpu_result);
}

ocean_tensor_handle_t ocean_tensor_slice(
    ocean_tensor_handle_t tensor,
    int axis,
    int start,
    int stop,
    int step
) {
    if (!tensor) ocean_tensor_fail("Tensor slice() received a null Tensor");
    if (axis < 0 || (size_t)axis >= tensor->ndim) {
        ocean_tensor_fail("Tensor slice axis out of bounds");
    }
    if (start < 0 || stop < 0 || step <= 0 || start > stop
        || (size_t)stop > tensor->shape[axis]) {
        ocean_tensor_fail("Tensor slice bounds are invalid");
    }

    size_t *shape = (size_t *)malloc(tensor->ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating Tensor slice shape");
    memcpy(shape, tensor->shape, tensor->ndim * sizeof(size_t));
    shape[axis] = start == stop
        ? 0 : ((size_t)(stop - start) + (size_t)step - 1) / (size_t)step;

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32 &&
        tensor->ndim == 4 && axis == 2 && step == 1) {
        if (tensor->shape[0] > (size_t)INT32_MAX ||
            tensor->shape[1] > (size_t)INT32_MAX ||
            tensor->shape[2] > (size_t)INT32_MAX ||
            tensor->shape[3] > (size_t)INT32_MAX ||
            shape[2] > (size_t)INT32_MAX) {
            free(shape);
            ocean_tensor_fail("Tensor cache slice dimensions are too large for OpenCL");
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_cache_slice(tensor, result, start);
        free(shape);
        return result;
    }
#endif

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t linear = 0; linear < cpu_result->size; ++linear) {
        size_t remaining = linear;
        size_t source_offset = 0;
        for (size_t current_axis = tensor->ndim; current_axis-- > 0;) {
            size_t coordinate = cpu_result->shape[current_axis]
                ? remaining % cpu_result->shape[current_axis] : 0;
            remaining = cpu_result->shape[current_axis]
                ? remaining / cpu_result->shape[current_axis] : 0;
            if (current_axis == (size_t)axis) {
                coordinate = (size_t)start + coordinate * (size_t)step;
            }
            source_offset += coordinate * cpu->strides[current_axis];
        }
        memcpy(
            (unsigned char *)cpu_result->cpu_data + linear * tensor->item_size,
            (unsigned char *)cpu->cpu_data + source_offset * tensor->item_size,
            tensor->item_size
        );
    }
    free(shape);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, cpu_result);
}

double ocean_tensor_sum(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor sum on null handle");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double result = 0.0L;
    for (size_t index = 0; index < cpu->size; ++index) {
        result += ocean_tensor_read_scalar(cpu, index);
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return (double)result;
}

double ocean_tensor_mean(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor mean on null handle");
    if (tensor->size == 0) return 0.0;
    return ocean_tensor_sum(tensor) / (double)tensor->size;
}

double ocean_tensor_max(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor max on null handle");
    if (tensor->size == 0) ocean_tensor_fail("Tensor max on an empty Tensor");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double result = ocean_tensor_read_scalar(cpu, 0);
    for (size_t index = 1; index < cpu->size; ++index) {
        long double value = ocean_tensor_read_scalar(cpu, index);
        if (value > result) result = value;
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return (double)result;
}

double ocean_tensor_min(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor min on null handle");
    if (tensor->size == 0) ocean_tensor_fail("Tensor min on an empty Tensor");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double result = ocean_tensor_read_scalar(cpu, 0);
    for (size_t index = 1; index < cpu->size; ++index) {
        long double value = ocean_tensor_read_scalar(cpu, index);
        if (value < result) result = value;
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return (double)result;
}

double ocean_tensor_item(ocean_tensor_handle_t tensor) {
    if (!tensor) {
        ocean_tensor_fail("Tensor item received a null Tensor");
    }
    if (ocean_tensor_size(tensor) != 1) {
        ocean_tensor_fail("Tensor item requires exactly one element");
    }

    return ocean_tensor_get_flat(tensor, 0);
}

char *ocean_tensor_dtype_name(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor dtype on null handle");
    static const char *names[] = {
        "bool", "int8", "int16", "int32", "int64", "uint8",
        "uint16", "uint32", "uint64", "float16", "float32", "float64"
    };
    const char *name = names[tensor->dtype];
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying Tensor dtype");
    strcpy(result, name);
    return result;
}

bool ocean_tensor_is_contiguous(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor is_contiguous on null handle");
    size_t expected = 1;
    for (size_t axis = tensor->ndim; axis-- > 0;) {
        if (tensor->strides[axis] != expected) return false;
        if (tensor->shape[axis] != 0 && expected > SIZE_MAX / tensor->shape[axis]) {
            return false;
        }
        expected *= tensor->shape[axis];
    }
    return true;
}

ocean_tensor_handle_t ocean_tensor_contiguous(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor contiguous on null handle");
    if (ocean_tensor_is_contiguous(tensor)) return ocean_tensor_copy(tensor);
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t index = 0; index < cpu->size; ++index) {
        ocean_tensor_write_scalar(result, index, ocean_tensor_read_scalar(cpu, index));
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, result);
}

void ocean_tensor_fill(ocean_tensor_handle_t tensor, double value) {
    if (!tensor) ocean_tensor_fail("Tensor fill on null handle");
    ocean_tensor_backend_for_device(tensor->device)->fill(tensor, value);
}

static void ocean_tensor_fill_opencl(
    ocean_tensor_handle_t tensor,
    double value
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;

#if defined(CL_VERSION_1_2)
    unsigned char pattern[8] = {0};
    size_t pattern_size = tensor->item_size;

    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: {
            bool v = value != 0.0;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT8: {
            int8_t v = (int8_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT16: {
            int16_t v = (int16_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT32: {
            int32_t v = (int32_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT64: {
            int64_t v = (int64_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT8: {
            uint8_t v = (uint8_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT16: {
            uint16_t v = (uint16_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT32: {
            uint32_t v = (uint32_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT64: {
            uint64_t v = (uint64_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_FLOAT16: {
            uint16_t v = ocean_tensor_float_to_half((float)value);
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_FLOAT32: {
            float v = (float)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_FLOAT64: {
            double v = value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
    }

    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueFillBuffer(
            ocean_tensor_opencl.queue,
            tensor->gpu_data,
            pattern,
            pattern_size,
            0,
            bytes,
            0,
            NULL,
            &event
        ),
        "clEnqueueFillBuffer(fill)"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue),
        "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
#else
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    ocean_tensor_fill_cpu(cpu, value);
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
    ocean_tensor_release(cpu);
#endif
#else
    (void)tensor;
    (void)value;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
}


static size_t ocean_tensor_index_offset(
    const ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim
) {
    if (!tensor || !indices || ndim != tensor->ndim) {

        ocean_tensor_fail("Tensor index rank does not match Tensor rank");
    }
    size_t offset = 0;
    for (size_t axis = 0; axis < ndim; ++axis) {
        if (indices[axis] >= tensor->shape[axis]) {
            ocean_tensor_fail("Tensor index is out of bounds");
        }
        offset += indices[axis] * tensor->strides[axis];
    }
    return offset;
}

double ocean_tensor_get_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim
) {
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t offset = ocean_tensor_index_offset(cpu, indices, ndim);
    double result = (double)ocean_tensor_read_scalar(cpu, offset);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return result;
}

double ocean_tensor_get_flat(ocean_tensor_handle_t tensor, size_t index) {
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor");
    if (index >= tensor->size) ocean_tensor_fail("Tensor flat index is out of bounds");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    double result = (double)ocean_tensor_read_scalar(cpu, index);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return result;
}

static void ocean_tensor_set_flat_long_double(
    ocean_tensor_handle_t tensor, size_t index, long double value
) {
    if (!tensor) ocean_tensor_fail("Tensor set received a null Tensor");
    if (index >= tensor->size) ocean_tensor_fail("Tensor flat index is out of bounds");
    if (tensor->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_write_scalar(tensor, index, value);
        return;
    }
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    ocean_tensor_write_scalar(cpu, index, value);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
#else
    ocean_tensor_release(cpu);
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    ocean_tensor_release(cpu);
}

void ocean_tensor_set_flat(
    ocean_tensor_handle_t tensor, size_t index, double value
) {
    ocean_tensor_set_flat_long_double(tensor, index, (long double)value);
}

#define OCEAN_DEFINE_TYPED_SET_FLAT(name, c_type) \
void ocean_tensor_set_flat_##name( \
    ocean_tensor_handle_t tensor, size_t index, c_type value \
) { \
    ocean_tensor_set_flat_long_double(tensor, index, (long double)value); \
}

OCEAN_DEFINE_TYPED_SET_FLAT(bool, bool)
OCEAN_DEFINE_TYPED_SET_FLAT(i8, int8_t)
OCEAN_DEFINE_TYPED_SET_FLAT(i16, int16_t)
OCEAN_DEFINE_TYPED_SET_FLAT(i32, int32_t)
OCEAN_DEFINE_TYPED_SET_FLAT(i64, int64_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u8, uint8_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u16, uint16_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u32, uint32_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u64, uint64_t)
OCEAN_DEFINE_TYPED_SET_FLAT(f16, float)
OCEAN_DEFINE_TYPED_SET_FLAT(f32, float)
OCEAN_DEFINE_TYPED_SET_FLAT(f64, double)

#undef OCEAN_DEFINE_TYPED_SET_FLAT

void ocean_tensor_set_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim,
    double value
) {
    if (!tensor) ocean_tensor_fail("Tensor set received a null Tensor");
    if (tensor->device == OCEAN_TENSOR_CPU) {
        size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
        ocean_tensor_write_scalar(tensor, offset, (long double)value);
        return;
    }
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    size_t offset = ocean_tensor_index_offset(cpu, indices, ndim);
    ocean_tensor_write_scalar(cpu, offset, (long double)value);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    ocean_tensor_release(cpu);
}


/*
 * Typed scalar accessors required by tensor_runtime.h and autograd_runtime.c.
 */
#define OCEAN_DEFINE_TYPED_GET_FLAT(name, c_type) \
c_type ocean_tensor_get_flat_##name( \
    ocean_tensor_handle_t tensor, size_t index \
) { \
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor"); \
    if (index >= tensor->size) { \
        ocean_tensor_fail("Tensor flat index is out of bounds"); \
    } \
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU \
        ? tensor : ocean_tensor_to(tensor, "cpu"); \
    c_type result = (c_type)ocean_tensor_read_scalar(cpu, index); \
    if (cpu != tensor) ocean_tensor_release(cpu); \
    return result; \
}

OCEAN_DEFINE_TYPED_GET_FLAT(bool, bool)
OCEAN_DEFINE_TYPED_GET_FLAT(i8, int8_t)
OCEAN_DEFINE_TYPED_GET_FLAT(i16, int16_t)
OCEAN_DEFINE_TYPED_GET_FLAT(i32, int32_t)
OCEAN_DEFINE_TYPED_GET_FLAT(i64, int64_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u8, uint8_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u16, uint16_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u32, uint32_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u64, uint64_t)
OCEAN_DEFINE_TYPED_GET_FLAT(f16, float)
OCEAN_DEFINE_TYPED_GET_FLAT(f32, float)
OCEAN_DEFINE_TYPED_GET_FLAT(f64, double)

#undef OCEAN_DEFINE_TYPED_GET_FLAT

#define OCEAN_DEFINE_TYPED_GET_ND(name, c_type) \
c_type ocean_tensor_get_nd_##name( \
    ocean_tensor_handle_t tensor, \
    const size_t *indices, \
    size_t ndim \
) { \
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor"); \
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU \
        ? tensor : ocean_tensor_to(tensor, "cpu"); \
    size_t offset = ocean_tensor_index_offset(cpu, indices, ndim); \
    c_type result = (c_type)ocean_tensor_read_scalar(cpu, offset); \
    if (cpu != tensor) ocean_tensor_release(cpu); \
    return result; \
}

OCEAN_DEFINE_TYPED_GET_ND(bool, bool)
OCEAN_DEFINE_TYPED_GET_ND(i8, int8_t)
OCEAN_DEFINE_TYPED_GET_ND(i16, int16_t)
OCEAN_DEFINE_TYPED_GET_ND(i32, int32_t)
OCEAN_DEFINE_TYPED_GET_ND(i64, int64_t)
OCEAN_DEFINE_TYPED_GET_ND(u8, uint8_t)
OCEAN_DEFINE_TYPED_GET_ND(u16, uint16_t)
OCEAN_DEFINE_TYPED_GET_ND(u32, uint32_t)
OCEAN_DEFINE_TYPED_GET_ND(u64, uint64_t)
OCEAN_DEFINE_TYPED_GET_ND(f16, float)
OCEAN_DEFINE_TYPED_GET_ND(f32, float)
OCEAN_DEFINE_TYPED_GET_ND(f64, double)

#undef OCEAN_DEFINE_TYPED_GET_ND

static void ocean_tensor_set_nd_long_double(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim,
    long double value
) {
    if (!tensor) ocean_tensor_fail("Tensor set received a null Tensor");

    if (tensor->device == OCEAN_TENSOR_CPU) {
        size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
        ocean_tensor_write_scalar(tensor, offset, value);
        return;
    }

    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    size_t offset = ocean_tensor_index_offset(cpu, indices, ndim);
    ocean_tensor_write_scalar(cpu, offset, value);

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
#else
    ocean_tensor_release(cpu);
    ocean_tensor_fail(
        "GPU backend is unavailable: rebuild with OpenCL support"
    );
#endif

    ocean_tensor_release(cpu);
}

#define OCEAN_DEFINE_TYPED_SET_ND(name, c_type) \
void ocean_tensor_set_nd_##name( \
    ocean_tensor_handle_t tensor, \
    const size_t *indices, \
    size_t ndim, \
    c_type value \
) { \
    ocean_tensor_set_nd_long_double( \
        tensor, indices, ndim, (long double)value \
    ); \
}

OCEAN_DEFINE_TYPED_SET_ND(bool, bool)
OCEAN_DEFINE_TYPED_SET_ND(i8, int8_t)
OCEAN_DEFINE_TYPED_SET_ND(i16, int16_t)
OCEAN_DEFINE_TYPED_SET_ND(i32, int32_t)
OCEAN_DEFINE_TYPED_SET_ND(i64, int64_t)
OCEAN_DEFINE_TYPED_SET_ND(u8, uint8_t)
OCEAN_DEFINE_TYPED_SET_ND(u16, uint16_t)
OCEAN_DEFINE_TYPED_SET_ND(u32, uint32_t)
OCEAN_DEFINE_TYPED_SET_ND(u64, uint64_t)
OCEAN_DEFINE_TYPED_SET_ND(f16, float)
OCEAN_DEFINE_TYPED_SET_ND(f32, float)
OCEAN_DEFINE_TYPED_SET_ND(f64, double)

#undef OCEAN_DEFINE_TYPED_SET_ND

double ocean_tensor_get_2d(ocean_tensor_handle_t tensor, int row, int col) {
    if (row < 0 || col < 0) ocean_tensor_fail("Tensor get index is out of bounds");
    size_t indices[2] = {(size_t)row, (size_t)col};
    return ocean_tensor_get_nd(tensor, indices, 2);
}

void ocean_tensor_set_2d(
    ocean_tensor_handle_t tensor, int row, int col, double value
) {
    if (row < 0 || col < 0) ocean_tensor_fail("Tensor set index is out of bounds");
    size_t indices[2] = {(size_t)row, (size_t)col};
    ocean_tensor_set_nd(tensor, indices, 2, value);
}

static ocean_tensor_handle_t ocean_tensor_matmul_cpu(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    size_t shape[2] = {left->shape[0], right->shape[1]};
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, 2, left->dtype, OCEAN_TENSOR_CPU
    );

    bool left_contiguous =
        left->strides[1] == 1 && left->strides[0] == left->shape[1];
    bool right_contiguous =
        right->strides[1] == 1 && right->strides[0] == right->shape[1];

    if (left_contiguous && right_contiguous) {
        size_t rows = left->shape[0];
        size_t inner = left->shape[1];
        size_t cols = right->shape[1];

        if (rows == 0 || inner == 0 || cols == 0) return result;

        const size_t block_rows = 32;
        const size_t block_inner = 64;
        const size_t block_cols = 128;

        if (left->dtype == OCEAN_TENSOR_FLOAT32) {
            const float *restrict a = (const float *)left->cpu_data;
            const float *restrict b = (const float *)right->cpu_data;
            float *restrict c = (float *)result->cpu_data;

            if (rows < 16 || inner < 32 || cols < 64) {
                for (size_t row = 0; row < rows; ++row) {
                    float *restrict c_row = c + row * cols;
                    const float *restrict a_row = a + row * inner;

                    for (size_t k = 0; k < inner; ++k) {
                        float av = a_row[k];
                        const float *restrict b_row = b + k * cols;
                        for (size_t col = 0; col < cols; ++col) {
                            c_row[col] += av * b_row[col];
                        }
                    }
                }
                return result;
            }

            for (size_t row0 = 0; row0 < rows; row0 += block_rows) {
                size_t row_end = row0 + block_rows < rows
                    ? row0 + block_rows : rows;

                for (size_t k0 = 0; k0 < inner; k0 += block_inner) {
                    size_t k_end = k0 + block_inner < inner
                        ? k0 + block_inner : inner;

                    for (size_t col0 = 0; col0 < cols; col0 += block_cols) {
                        size_t col_end = col0 + block_cols < cols
                            ? col0 + block_cols : cols;

                        for (size_t row = row0; row < row_end; ++row) {
                            float *restrict c_row = c + row * cols;
                            const float *restrict a_row = a + row * inner;

                            for (size_t k = k0; k < k_end; ++k) {
                                float av = a_row[k];
                                const float *restrict b_row = b + k * cols;

                                for (size_t col = col0; col < col_end; ++col) {
                                    c_row[col] += av * b_row[col];
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }

        if (left->dtype == OCEAN_TENSOR_FLOAT64) {
            const double *restrict a = (const double *)left->cpu_data;
            const double *restrict b = (const double *)right->cpu_data;
            double *restrict c = (double *)result->cpu_data;

            if (rows < 16 || inner < 32 || cols < 64) {
                for (size_t row = 0; row < rows; ++row) {
                    double *restrict c_row = c + row * cols;
                    const double *restrict a_row = a + row * inner;

                    for (size_t k = 0; k < inner; ++k) {
                        double av = a_row[k];
                        const double *restrict b_row = b + k * cols;
                        for (size_t col = 0; col < cols; ++col) {
                            c_row[col] += av * b_row[col];
                        }
                    }
                }
                return result;
            }

            for (size_t row0 = 0; row0 < rows; row0 += block_rows) {
                size_t row_end = row0 + block_rows < rows
                    ? row0 + block_rows : rows;

                for (size_t k0 = 0; k0 < inner; k0 += block_inner) {
                    size_t k_end = k0 + block_inner < inner
                        ? k0 + block_inner : inner;

                    for (size_t col0 = 0; col0 < cols; col0 += block_cols) {
                        size_t col_end = col0 + block_cols < cols
                            ? col0 + block_cols : cols;

                        for (size_t row = row0; row < row_end; ++row) {
                            double *restrict c_row = c + row * cols;
                            const double *restrict a_row = a + row * inner;

                            for (size_t k = k0; k < k_end; ++k) {
                                double av = a_row[k];
                                const double *restrict b_row = b + k * cols;

                                for (size_t col = col0; col < col_end; ++col) {
                                    c_row[col] += av * b_row[col];
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }
    }

    for (size_t row = 0; row < left->shape[0]; ++row) {
        for (size_t col = 0; col < right->shape[1]; ++col) {
            long double sum = 0.0L;

            for (size_t k = 0; k < left->shape[1]; ++k) {
                size_t left_index =
                    row * left->strides[0] + k * left->strides[1];
                size_t right_index =
                    k * right->strides[0] + col * right->strides[1];

                sum += ocean_tensor_read_scalar(left, left_index)
                    * ocean_tensor_read_scalar(right, right_index);
            }

            ocean_tensor_write_scalar(
                result,
                row * result->strides[0] + col * result->strides[1],
                sum
            );
        }
    }

    return result;
}


#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static ocean_tensor_handle_t ocean_tensor_matmul_opencl_batched(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
) {
    if (left->dtype != OCEAN_TENSOR_FLOAT32 ||
        right->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail(
            "GPU batched matmul currently requires float32 tensors"
        );
    }

    size_t output_ndim = left->ndim > right->ndim
        ? left->ndim : right->ndim;
    size_t batch_ndim = output_ndim - 2;
    size_t left_batch_ndim = left->ndim - 2;
    size_t right_batch_ndim = right->ndim - 2;
    size_t left_leading = batch_ndim - left_batch_ndim;
    size_t right_leading = batch_ndim - right_batch_ndim;

    size_t rows = transpose_left
        ? left->shape[left->ndim - 1]
        : left->shape[left->ndim - 2];
    size_t inner = transpose_left
        ? left->shape[left->ndim - 2]
        : left->shape[left->ndim - 1];
    size_t right_inner = transpose_right
        ? right->shape[right->ndim - 1]
        : right->shape[right->ndim - 2];
    size_t cols = transpose_right
        ? right->shape[right->ndim - 2]
        : right->shape[right->ndim - 1];
    if (inner != right_inner) {
        ocean_tensor_fail("batched matmul shape mismatch");
    }

    size_t *output_shape = (size_t *)malloc(
        output_ndim * sizeof(size_t)
    );
    int *left_shape = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *right_shape = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *out_shape = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *left_strides = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *right_strides = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    if (!output_shape || !left_shape || !right_shape || !out_shape ||
        !left_strides || !right_strides) {
        free(output_shape);
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        ocean_tensor_fail("out of memory in batched matmul metadata");
    }

    for (size_t axis = 0; axis < batch_ndim; ++axis) {
        size_t left_axis = axis >= left_leading
            ? axis - left_leading : SIZE_MAX;
        size_t right_axis = axis >= right_leading
            ? axis - right_leading : SIZE_MAX;
        size_t left_dim = left_axis == SIZE_MAX
            ? 1 : left->shape[left_axis];
        size_t right_dim = right_axis == SIZE_MAX
            ? 1 : right->shape[right_axis];
        if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
            free(output_shape);
            free(left_shape);
            free(right_shape);
            free(out_shape);
            free(left_strides);
            free(right_strides);
            ocean_tensor_fail(
                "batched matmul batch dimensions are not broadcastable"
            );
        }
        size_t dimension = left_dim > right_dim ? left_dim : right_dim;
        output_shape[axis] = dimension;
        left_shape[axis] = (int)left_dim;
        right_shape[axis] = (int)right_dim;
        out_shape[axis] = (int)dimension;
        left_strides[axis] = left_axis == SIZE_MAX
            ? 0 : (int)left->strides[left_axis];
        right_strides[axis] = right_axis == SIZE_MAX
            ? 0 : (int)right->strides[right_axis];
        if (left_dim > (size_t)INT32_MAX || right_dim > (size_t)INT32_MAX ||
            dimension > (size_t)INT32_MAX ||
            (left_axis != SIZE_MAX && left->strides[left_axis] > (size_t)INT32_MAX) ||
            (right_axis != SIZE_MAX && right->strides[right_axis] > (size_t)INT32_MAX)) {
            free(output_shape);
            free(left_shape);
            free(right_shape);
            free(out_shape);
            free(left_strides);
            free(right_strides);
            ocean_tensor_fail(
                "batched matmul metadata is too large for OpenCL"
            );
        }
    }
    output_shape[output_ndim - 2] = rows;
    output_shape[output_ndim - 1] = cols;
    if (rows > (size_t)INT32_MAX || inner > (size_t)INT32_MAX ||
        cols > (size_t)INT32_MAX) {
        free(output_shape);
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        ocean_tensor_fail("batched matmul dimensions are too large for OpenCL");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape,
        output_ndim,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_GPU
    );
    free(output_shape);
    if (result->size == 0) {
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        return result;
    }
    if (result->size > (size_t)INT32_MAX) {
        ocean_tensor_release(result);
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        ocean_tensor_fail(
            "batched matmul output is too large for OpenCL indexing"
        );
    }

    size_t metadata_count = batch_ndim ? batch_ndim : 1;
    cl_int status = CL_SUCCESS;
    cl_mem left_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), left_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem right_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), right_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem out_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), out_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem left_strides_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), left_strides, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem right_strides_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), right_strides, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    free(left_shape);
    free(right_shape);
    free(out_shape);
    free(left_strides);
    free(right_strides);

    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_BATCHED_MATMUL_FLOAT32
    );
    int batch_count = (int)batch_ndim;
    int rows_value = (int)rows;
    int inner_value = (int)inner;
    int cols_value = (int)cols;
    int transpose_left_value = transpose_left ? 1 : 0;
    int transpose_right_value = transpose_right ? 1 : 0;
    int output_size = (int)result->size;
    cl_mem buffers[] = {
        left->gpu_data,
        right->gpu_data,
        result->gpu_data,
        left_shape_buffer,
        right_shape_buffer,
        out_shape_buffer,
        left_strides_buffer,
        right_strides_buffer,
    };
    for (int index = 0; index < 8; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, (cl_uint)index, sizeof(cl_mem), &buffers[index]),
            "clSetKernelArg"
        );
    }
    int values[] = {
        batch_count,
        rows_value,
        inner_value,
        cols_value,
        transpose_left_value,
        transpose_right_value,
        output_size,
    };
    for (int index = 0; index < 7; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(8 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    size_t global_size = result->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
    clReleaseMemObject(left_shape_buffer);
    clReleaseMemObject(right_shape_buffer);
    clReleaseMemObject(out_shape_buffer);
    clReleaseMemObject(left_strides_buffer);
    clReleaseMemObject(right_strides_buffer);
    return result;
}

static void ocean_tensor_opencl_permute(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output,
    const int *axes
) {
    if (input->size == 0) return;
    if (input->ndim > (size_t)INT32_MAX ||
        output->size > (size_t)INT32_MAX ||
        input->item_size > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor.permute is too large for OpenCL indexing");
    }

    int rank = (int)input->ndim;
    int item_size = (int)input->item_size;
    int output_size = (int)output->size;
    int *output_shape = (int *)malloc(input->ndim * sizeof(int));
    int *input_strides = (int *)malloc(input->ndim * sizeof(int));
    if (!output_shape || !input_strides) {
        free(output_shape);
        free(input_strides);
        ocean_tensor_fail("out of memory in Tensor.permute metadata");
    }
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        if (output->shape[axis] > (size_t)INT32_MAX ||
            input->strides[axis] > (size_t)INT32_MAX) {
            free(output_shape);
            free(input_strides);
            ocean_tensor_fail(
                "Tensor.permute metadata is too large for OpenCL"
            );
        }
        output_shape[axis] = (int)output->shape[axis];
        input_strides[axis] = (int)input->strides[axis];
    }

    cl_int status = CL_SUCCESS;
    cl_mem output_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input->ndim * sizeof(int), output_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem input_strides_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input->ndim * sizeof(int), input_strides, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem axes_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input->ndim * sizeof(int), (void *)axes, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    free(output_shape);
    free(input_strides);

    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_PERMUTE
    );
    cl_mem buffers[] = {
        input->gpu_data,
        output->gpu_data,
        output_shape_buffer,
        input_strides_buffer,
        axes_buffer,
    };
    for (int index = 0; index < 5; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, (cl_uint)index, sizeof(cl_mem), &buffers[index]),
            "clSetKernelArg"
        );
    }
    int values[] = {rank, item_size, output_size};
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(5 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    size_t global_size = output->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
    clReleaseMemObject(output_shape_buffer);
    clReleaseMemObject(input_strides_buffer);
    clReleaseMemObject(axes_buffer);
}
#endif


static ocean_tensor_handle_t ocean_tensor_matmul_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (left->dtype != OCEAN_TENSOR_FLOAT32 && left->dtype != OCEAN_TENSOR_INT32) {
        /* Preserve GPU semantics for dtypes without a specialized kernel
           through a correct CPU fallback and upload of the result. */
        ocean_tensor_handle_t left_cpu = ocean_tensor_to(left, "cpu");
        ocean_tensor_handle_t right_cpu = ocean_tensor_to(right, "cpu");
        ocean_tensor_handle_t cpu_result = ocean_tensor_matmul_cpu(left_cpu, right_cpu);
        ocean_tensor_handle_t gpu_result = ocean_tensor_to(cpu_result, "gpu");
        ocean_tensor_release(left_cpu);
        ocean_tensor_release(right_cpu);
        ocean_tensor_release(cpu_result);
        return gpu_result;
    }

    size_t shape[2] = {left->shape[0], right->shape[1]};
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, 2, left->dtype, OCEAN_TENSOR_GPU
    );
    if (left->shape[0] == 0 || right->shape[1] == 0) {
        return result;
    }
    if (left->shape[0] > (size_t)INT32_MAX ||
        left->shape[1] > (size_t)INT32_MAX ||
        right->shape[1] > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor dimensions are too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        left->dtype == OCEAN_TENSOR_INT32
            ? OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_INT32
            : OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_FLOAT32
    );
    int rows = (int)left->shape[0];
    int cols = (int)left->shape[1];
    int result_cols = (int)right->shape[1];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &left->gpu_data), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &right->gpu_data), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &result->gpu_data), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &cols), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &result_cols), "clSetKernelArg"
    );
    const size_t tile_size = 8u;
    size_t global_size[2] = {
        ((size_t)rows + tile_size - 1u) / tile_size * tile_size,
        ((size_t)result_cols + tile_size - 1u) / tile_size * tile_size,
    };
    size_t local_size[2] = {tile_size, tile_size};
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 2, NULL,
            global_size, local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    /* The queue is in-order.  Consumers (including a blocking download)
       provide the dependency, so avoid synchronizing the host here. */
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
    return result;
#else
    (void)left;
    (void)right;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}


/* ================= ND Tensor v0.2 ================= */
static size_t ocean_tensor_normalize_dim_v02(const ocean_tensor_handle_t tensor, int dim) {
    if (!tensor || tensor->ndim == 0) ocean_tensor_fail("Tensor dimension operation requires rank >= 1");
    long long rank = (long long)tensor->ndim;
    long long d = (long long)dim;
    if (d < 0) d += rank;
    if (d < 0 || d >= rank) ocean_tensor_fail("Tensor dimension is out of bounds");
    return (size_t)d;
}

ocean_tensor_handle_t ocean_tensor_reshape_3d(ocean_tensor_handle_t tensor, int d0, int d1, int d2) {
    if (d0 < 0 || d1 < 0 || d2 < 0) ocean_tensor_fail("Tensor reshape dimensions must be non-negative");
    size_t shape[3] = {(size_t)d0, (size_t)d1, (size_t)d2};
    return ocean_tensor_reshape(tensor, shape, 3);
}

ocean_tensor_handle_t ocean_tensor_reshape_4d(ocean_tensor_handle_t tensor, int d0, int d1, int d2, int d3) {
    if (d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0) ocean_tensor_fail("Tensor reshape dimensions must be non-negative");
    size_t shape[4] = {(size_t)d0, (size_t)d1, (size_t)d2, (size_t)d3};
    return ocean_tensor_reshape(tensor, shape, 4);
}

ocean_tensor_handle_t ocean_tensor_transpose_dims(ocean_tensor_handle_t tensor, int dim0, int dim1) {
    if (!tensor) ocean_tensor_fail("Tensor transpose_dims on null handle");
    size_t a = ocean_tensor_normalize_dim_v02(tensor, dim0);
    size_t b = ocean_tensor_normalize_dim_v02(tensor, dim1);
    if (a == b) return ocean_tensor_copy(tensor);
    int *axes = (int *)malloc(tensor->ndim * sizeof(int));
    if (!axes) ocean_tensor_fail("out of memory allocating transpose axes");
    for (size_t axis = 0; axis < tensor->ndim; ++axis) {
        axes[axis] = (int)axis;
    }
    axes[a] = (int)b;
    axes[b] = (int)a;
    ocean_tensor_handle_t result = ocean_tensor_permute(
        tensor, axes, tensor->ndim
    );
    free(axes);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_reduce_dim_v02(ocean_tensor_handle_t tensor, int dim, bool keepdim, bool mean) {
    if (!tensor) ocean_tensor_fail("Tensor reduction on null handle");
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t out_ndim = keepdim ? tensor->ndim : (tensor->ndim > 1 ? tensor->ndim - 1 : 1);
    size_t *shape = malloc(out_ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating reduction shape");
    if (keepdim) {
        for (size_t i=0;i<tensor->ndim;++i) shape[i] = i==axis ? 1 : tensor->shape[i];
    } else if (tensor->ndim == 1) {
        shape[0] = 1;
    } else {
        size_t j=0; for (size_t i=0;i<tensor->ndim;++i) if (i!=axis) shape[j++] = tensor->shape[i];
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32 &&
        axis == tensor->ndim - 1) {
        size_t width = tensor->shape[axis];
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, out_ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        if (width != 0 && tensor->size != 0) {
            if (width > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                free(shape);
                ocean_tensor_fail("GPU reduction dimension is too large for OpenCL");
            }
            ocean_tensor_opencl_rowwise(
                tensor,
                result,
                OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32,
                (int)width,
                0.0f,
                mean ? 1 : 0
            );
        } else if (tensor->size != 0) {
            ocean_tensor_backend_for_device(OCEAN_TENSOR_GPU)->zero(result);
        }
        free(shape);
        return result;
    }
#endif

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU ? tensor : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t out = ocean_tensor_alloc_zeros(shape, out_ndim, tensor->dtype, OCEAN_TENSOR_CPU);
    free(shape);
    size_t *coord = calloc(tensor->ndim, sizeof(size_t));
    if (!coord) ocean_tensor_fail("out of memory reducing Tensor");

    for (size_t linear=0; linear<cpu->size; ++linear) {
        size_t rem=linear;
        for (size_t i=cpu->ndim;i-- >0;) { size_t d=cpu->shape[i]; coord[i]=d?rem%d:0; rem=d?rem/d:0; }
        size_t out_linear=0;
        if (keepdim) {
            for (size_t i=0;i<cpu->ndim;++i) out_linear = out_linear*out->shape[i] + (i==axis?0:coord[i]);
        } else if (cpu->ndim > 1) {
            size_t j=0; for (size_t i=0;i<cpu->ndim;++i) if (i!=axis) out_linear = out_linear*out->shape[j++] + coord[i];
        }
        ocean_tensor_write_scalar(out, out_linear, ocean_tensor_read_scalar(out,out_linear)+ocean_tensor_read_scalar(cpu,linear));
    }
    if (mean && tensor->shape[axis]) {
        long double div=(long double)tensor->shape[axis];
        for (size_t i=0;i<out->size;++i) ocean_tensor_write_scalar(out,i,ocean_tensor_read_scalar(out,i)/div);
    }
    free(coord);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, out);
}

ocean_tensor_handle_t ocean_tensor_sum_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim) { return ocean_tensor_reduce_dim_v02(tensor,dim,keepdim,false); }
ocean_tensor_handle_t ocean_tensor_mean_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim) { return ocean_tensor_reduce_dim_v02(tensor,dim,keepdim,true); }

ocean_tensor_handle_t ocean_tensor_softmax(
    ocean_tensor_handle_t tensor,
    int dim
) {
    if (!tensor) ocean_tensor_fail("Tensor.softmax on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.softmax currently requires float32");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t axis_size = tensor->shape[axis];

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        if (axis_size != 0 && tensor->size != 0) {
            if (axis_size > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                ocean_tensor_fail("GPU softmax dimension is too large for OpenCL");
            }
            ocean_tensor_opencl_rowwise(
                tensor,
                result,
                OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_FLOAT32,
                (int)axis_size,
                0.0f,
                0
            );
        }
        return result;
    }
#endif

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor
        : ocean_tensor_to(tensor, "cpu");
    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= tensor->shape[i];
    for (size_t i = axis + 1; i < tensor->ndim; ++i) inner *= tensor->shape[i];
    if (axis_size == 0) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("Tensor.softmax cannot normalize an empty dimension");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float max_value = -INFINITY;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                float value = ocean_tensor_get_flat_f32(cpu, index);
                if (value > max_value) max_value = value;
            }
            double denominator = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                float value = expf(
                    ocean_tensor_get_flat_f32(cpu, index) - max_value
                );
                ocean_tensor_write_scalar(result, index, value);
                denominator += (double)value;
            }
            if (!(denominator > 0.0)) {
                ocean_tensor_release(result);
                if (cpu != tensor) ocean_tensor_release(cpu);
                free(device);
                ocean_tensor_fail("softmax denominator is not positive");
            }
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                ocean_tensor_write_scalar(
                    result, index,
                    ocean_tensor_read_scalar(result, index) / denominator
                );
            }
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return ocean_tensor_restore_device(tensor, result);
}

ocean_tensor_handle_t ocean_tensor_layer_norm(
    ocean_tensor_handle_t tensor,
    int dim,
    double epsilon
) {
    if (!tensor) ocean_tensor_fail("Tensor.layer_norm on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.layer_norm currently requires float32");
    }
    if (!(epsilon > 0.0)) {
        ocean_tensor_fail("LayerNorm epsilon must be positive");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t axis_size = tensor->shape[axis];

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        if (axis_size != 0 && tensor->size != 0) {
            if (axis_size > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                ocean_tensor_fail("GPU LayerNorm dimension is too large for OpenCL");
            }
            ocean_tensor_opencl_rowwise(
                tensor,
                result,
                OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32,
                (int)axis_size,
                (float)epsilon,
                0
            );
        }
        return result;
    }
#endif

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor
        : ocean_tensor_to(tensor, "cpu");
    if (axis_size == 0) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("LayerNorm cannot normalize an empty dimension");
    }
    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= tensor->shape[i];
    for (size_t i = axis + 1; i < tensor->ndim; ++i) inner *= tensor->shape[i];
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            double mean_value = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                mean_value += ocean_tensor_get_flat_f32(cpu, index);
            }
            mean_value /= (double)axis_size;
            double variance = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                double delta = ocean_tensor_get_flat_f32(cpu, index) - mean_value;
                variance += delta * delta;
            }
            double inverse_std = 1.0 / sqrt(variance / (double)axis_size + epsilon);
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                double value = ocean_tensor_get_flat_f32(cpu, index);
                ocean_tensor_write_scalar(
                    result, index, (value - mean_value) * inverse_std
                );
            }
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return ocean_tensor_restore_device(tensor, result);
}

static void ocean_tensor_validate_backward_inputs(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t reference,
    const char *operation
) {
    if (!upstream || !reference) {
        ocean_tensor_fail("Tensor backward operation received a null handle");
    }
    if (upstream->dtype != OCEAN_TENSOR_FLOAT32 ||
        reference->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor backward operation currently requires float32");
    }
    if (upstream->device != reference->device ||
        upstream->ndim != reference->ndim ||
        upstream->size != reference->size) {
        ocean_tensor_fail(operation);
    }
    for (size_t axis = 0; axis < reference->ndim; ++axis) {
        if (upstream->shape[axis] != reference->shape[axis]) {
            ocean_tensor_fail(operation);
        }
    }
}

ocean_tensor_handle_t ocean_tensor_softmax_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t output,
    int dim
) {
    ocean_tensor_validate_backward_inputs(
        upstream, output,
        "Tensor.softmax backward requires matching Tensor shapes and devices"
    );
    size_t axis = ocean_tensor_normalize_dim_v02(output, dim);
    if (output->device != OCEAN_TENSOR_GPU ||
        axis != output->ndim - 1) {
        ocean_tensor_fail(
            "GPU softmax backward currently requires the last Tensor dimension"
        );
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t width = output->shape[axis];
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output->shape, output->ndim, output->dtype, OCEAN_TENSOR_GPU
    );
    if (width != 0 && output->size != 0) {
        if (width > (size_t)INT32_MAX) {
            ocean_tensor_release(result);
            ocean_tensor_fail(
                "GPU softmax backward dimension is too large for OpenCL"
            );
        }
        ocean_tensor_opencl_softmax_backward(
            upstream, output, result, (int)width
        );
    }
    return result;
#else
    (void)upstream;
    (void)output;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_layer_norm_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input,
    int dim,
    double epsilon
) {
    ocean_tensor_validate_backward_inputs(
        upstream, input,
        "Tensor.LayerNorm backward requires matching Tensor shapes and devices"
    );
    if (!(epsilon > 0.0)) {
        ocean_tensor_fail("LayerNorm epsilon must be positive");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(input, dim);
    if (input->device != OCEAN_TENSOR_GPU ||
        axis != input->ndim - 1) {
        ocean_tensor_fail(
            "GPU LayerNorm backward currently requires the last Tensor dimension"
        );
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t width = input->shape[axis];
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        input->shape, input->ndim, input->dtype, OCEAN_TENSOR_GPU
    );
    if (width != 0 && input->size != 0) {
        if (width > (size_t)INT32_MAX) {
            ocean_tensor_release(result);
            ocean_tensor_fail(
                "GPU LayerNorm backward dimension is too large for OpenCL"
            );
        }
        ocean_tensor_opencl_layer_norm_backward(
            upstream, input, result, (int)width, epsilon
        );
    }
    return result;
#else
    (void)upstream;
    (void)input;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

static void ocean_tensor_validate_optimizer_tensors(
    ocean_tensor_handle_t parameter,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t first_moment,
    ocean_tensor_handle_t second_moment
) {
    if (!parameter || !gradient ||
        (first_moment == NULL) != (second_moment == NULL)) {
        ocean_tensor_fail("optimizer update received a null Tensor");
    }
    ocean_tensor_handle_t tensors[4] = {
        parameter, gradient, first_moment, second_moment
    };
    size_t tensor_count = first_moment ? 4 : 2;
    for (size_t i = 0; i < tensor_count; ++i) {
        if (tensors[i]->dtype != OCEAN_TENSOR_FLOAT32) {
            ocean_tensor_fail("optimizer updates currently require float32");
        }
        if (tensors[i]->device != parameter->device ||
            tensors[i]->size != parameter->size ||
            tensors[i]->ndim != parameter->ndim) {
            ocean_tensor_fail("optimizer tensors must have matching device and shape");
        }
        for (size_t axis = 0; axis < parameter->ndim; ++axis) {
            if (tensors[i]->shape[axis] != parameter->shape[axis]) {
                ocean_tensor_fail("optimizer tensors must have matching device and shape");
            }
        }
    }
}

void ocean_tensor_sgd_update(
    ocean_tensor_handle_t parameter,
    ocean_tensor_handle_t gradient,
    double learning_rate
) {
    ocean_tensor_validate_optimizer_tensors(parameter, gradient, NULL, NULL);
    if (learning_rate < 0.0) {
        ocean_tensor_fail("SGD learning rate must be non-negative");
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_opencl_sgd_update(parameter, gradient, learning_rate);
        return;
    }
#else
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    }
#endif
    float *values = (float *)parameter->cpu_data;
    const float *gradients = (const float *)gradient->cpu_data;
    float rate = (float)learning_rate;
    for (size_t i = 0; i < parameter->size; ++i) {
        values[i] -= rate * gradients[i];
    }
}

void ocean_tensor_adamw_update(
    ocean_tensor_handle_t parameter,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t first_moment,
    ocean_tensor_handle_t second_moment,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay,
    double bias_correction1,
    double bias_correction2
) {
    ocean_tensor_validate_optimizer_tensors(
        parameter, gradient, first_moment, second_moment
    );
    if (learning_rate < 0.0 || beta1 < 0.0 || beta1 >= 1.0 ||
        beta2 < 0.0 || beta2 >= 1.0 || epsilon <= 0.0 ||
        weight_decay < 0.0 || bias_correction1 <= 0.0 ||
        bias_correction2 <= 0.0) {
        ocean_tensor_fail("invalid AdamW optimizer arguments");
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_opencl_adamw_update(
            parameter, gradient, first_moment, second_moment,
            learning_rate, beta1, beta2, epsilon, weight_decay,
            bias_correction1, bias_correction2
        );
        return;
    }
#else
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    }
#endif
    float *values = (float *)parameter->cpu_data;
    const float *gradients = (const float *)gradient->cpu_data;
    float *first = (float *)first_moment->cpu_data;
    float *second = (float *)second_moment->cpu_data;
    for (size_t i = 0; i < parameter->size; ++i) {
        float gradient_value = gradients[i];
        float first_value = (float)(
            beta1 * (double)first[i]
            + (1.0 - beta1) * (double)gradient_value
        );
        float second_value = (float)(
            beta2 * (double)second[i]
            + (1.0 - beta2) * (double)gradient_value * (double)gradient_value
        );
        first[i] = first_value;
        second[i] = second_value;
        double adaptive =
            ((double)first_value / bias_correction1)
            / (sqrt((double)second_value / bias_correction2) + epsilon);
        values[i] = (float)(
            (double)values[i]
            - learning_rate * weight_decay * (double)values[i]
            - learning_rate * adaptive
        );
    }
}

static ocean_tensor_handle_t ocean_tensor_matmul_nd_cpu_v02(ocean_tensor_handle_t left, ocean_tensor_handle_t right) {
    size_t out_ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    size_t batch_ndim = out_ndim - 2;
    size_t lb = left->ndim - 2, rb = right->ndim - 2;
    size_t *shape = malloc(out_ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating batched matmul shape");

    for (size_t oa=0; oa<batch_ndim; ++oa) {
        long long la=(long long)oa-(long long)(batch_ndim-lb);
        long long ra=(long long)oa-(long long)(batch_ndim-rb);
        size_t ld=la>=0?left->shape[(size_t)la]:1;
        size_t rd=ra>=0?right->shape[(size_t)ra]:1;
        if (ld!=rd && ld!=1 && rd!=1) { free(shape); ocean_tensor_fail("batched matmul batch dimensions are not broadcastable"); }
        shape[oa]=ld>rd?ld:rd;
    }
    shape[out_ndim-2]=left->shape[left->ndim-2];
    shape[out_ndim-1]=right->shape[right->ndim-1];
    ocean_tensor_handle_t out=ocean_tensor_alloc_zeros(shape,out_ndim,left->dtype,OCEAN_TENSOR_CPU);
    free(shape);
    size_t *coord=calloc(out_ndim,sizeof(size_t));
    if (!coord) ocean_tensor_fail("out of memory in batched matmul");
    size_t inner=left->shape[left->ndim-1];

    for (size_t linear=0;linear<out->size;++linear) {
        size_t rem=linear;
        for (size_t i=out_ndim;i-- >0;) { size_t d=out->shape[i]; coord[i]=d?rem%d:0; rem=d?rem/d:0; }
        size_t row=coord[out_ndim-2], col=coord[out_ndim-1];
        size_t lbase=0, rbase=0;
        for (size_t i=0;i<lb;++i) { size_t oa=batch_ndim-lb+i; size_t c=left->shape[i]==1?0:coord[oa]; lbase+=c*left->strides[i]; }
        for (size_t i=0;i<rb;++i) { size_t oa=batch_ndim-rb+i; size_t c=right->shape[i]==1?0:coord[oa]; rbase+=c*right->strides[i]; }
        lbase += row*left->strides[left->ndim-2];
        rbase += col*right->strides[right->ndim-1];
        long double sum=0.0L;
        for (size_t k=0;k<inner;++k) sum += ocean_tensor_read_scalar(left,lbase+k*left->strides[left->ndim-1]) * ocean_tensor_read_scalar(right,rbase+k*right->strides[right->ndim-2]);
        ocean_tensor_write_scalar(out,linear,sum);
    }
    free(coord); return out;
}


ocean_tensor_handle_t ocean_tensor_matmul_transposed(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
) {
    if (!left || !right) {
        ocean_tensor_fail("matmul does not accept null Tensors");
    }
    if (left->ndim < 2 || right->ndim < 2) {
        ocean_tensor_fail("matmul expects Tensor rank >= 2");
    }
    if (left->dtype != right->dtype) {
        ocean_tensor_fail("matmul requires matching Tensor dtypes");
    }
    if (left->device != right->device) {
        ocean_tensor_fail("matmul requires Tensors on the same device");
    }

    size_t left_inner = transpose_left
        ? left->shape[left->ndim - 2]
        : left->shape[left->ndim - 1];
    size_t right_inner = transpose_right
        ? right->shape[right->ndim - 1]
        : right->shape[right->ndim - 2];
    if (left_inner != right_inner) {
        ocean_tensor_fail("matmul shape mismatch");
    }

    if (!transpose_left && !transpose_right) {
        return ocean_tensor_matmul(left, right);
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (left->device == OCEAN_TENSOR_GPU &&
        left->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_handle_t contiguous_left =
            ocean_tensor_is_contiguous(left)
            ? left : ocean_tensor_contiguous(left);
        ocean_tensor_handle_t contiguous_right =
            ocean_tensor_is_contiguous(right)
            ? right : ocean_tensor_contiguous(right);
        ocean_tensor_handle_t result = ocean_tensor_matmul_opencl_batched(
            contiguous_left,
            contiguous_right,
            transpose_left,
            transpose_right
        );
        if (contiguous_left != left) ocean_tensor_release(contiguous_left);
        if (contiguous_right != right) ocean_tensor_release(contiguous_right);
        return result;
    }
#endif

    ocean_tensor_handle_t transposed_left = transpose_left
        ? ocean_tensor_transpose_dims(left, -2, -1) : ocean_tensor_copy(left);
    ocean_tensor_handle_t transposed_right = transpose_right
        ? ocean_tensor_transpose_dims(right, -2, -1) : ocean_tensor_copy(right);
    ocean_tensor_handle_t result = ocean_tensor_matmul(
        transposed_left,
        transposed_right
    );
    ocean_tensor_release(transposed_left);
    ocean_tensor_release(transposed_right);
    return result;
}

ocean_tensor_handle_t ocean_tensor_matmul(ocean_tensor_handle_t left, ocean_tensor_handle_t right) {
    if (!left || !right) ocean_tensor_fail("matmul does not accept null Tensors");
    if (left->ndim < 2 || right->ndim < 2) ocean_tensor_fail("matmul expects Tensor rank >= 2");
    if (left->shape[left->ndim-1] != right->shape[right->ndim-2]) ocean_tensor_fail("matmul shape mismatch");
    if (left->dtype != right->dtype) ocean_tensor_fail("matmul requires matching Tensor dtypes");
    if (left->device != right->device) ocean_tensor_fail("matmul requires Tensors on the same device");
    if (left->ndim == 2 && right->ndim == 2) return ocean_tensor_backend_for_device(left->device)->matmul(left,right);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (left->device == OCEAN_TENSOR_GPU &&
        left->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_handle_t contiguous_left =
            ocean_tensor_is_contiguous(left)
            ? left : ocean_tensor_contiguous(left);
        ocean_tensor_handle_t contiguous_right =
            ocean_tensor_is_contiguous(right)
            ? right : ocean_tensor_contiguous(right);
        ocean_tensor_handle_t result = ocean_tensor_matmul_opencl_batched(
            contiguous_left,
            contiguous_right,
            false,
            false
        );
        if (contiguous_left != left) ocean_tensor_release(contiguous_left);
        if (contiguous_right != right) ocean_tensor_release(contiguous_right);
        return result;
    }
#endif
    ocean_tensor_handle_t lc = left->device==OCEAN_TENSOR_CPU ? left : ocean_tensor_to(left,"cpu");
    ocean_tensor_handle_t rc = right->device==OCEAN_TENSOR_CPU ? right : ocean_tensor_to(right,"cpu");
    ocean_tensor_handle_t cpu=ocean_tensor_matmul_nd_cpu_v02(lc,rc);
    if (lc != left) {
        ocean_tensor_release(lc);
    }
    if (rc != right) {
        ocean_tensor_release(rc);
    }
    if (left->device==OCEAN_TENSOR_CPU) return cpu;
    ocean_tensor_handle_t out=ocean_tensor_to(cpu,"gpu"); ocean_tensor_release(cpu); return out;
}

static bool ocean_tensor_host_is_little_endian(void) {
    const uint16_t value = 1u;
    return *((const unsigned char *)&value) == 1u;
}

static const char *ocean_tensor_npy_descr(ocean_tensor_dtype dtype) {
    switch (dtype) {
        case OCEAN_TENSOR_BOOL: return "|b1";
        case OCEAN_TENSOR_INT8: return "|i1";
        case OCEAN_TENSOR_INT16: return "<i2";
        case OCEAN_TENSOR_INT32: return "<i4";
        case OCEAN_TENSOR_INT64: return "<i8";
        case OCEAN_TENSOR_UINT8: return "|u1";
        case OCEAN_TENSOR_UINT16: return "<u2";
        case OCEAN_TENSOR_UINT32: return "<u4";
        case OCEAN_TENSOR_UINT64: return "<u8";
        case OCEAN_TENSOR_FLOAT16: return "<f2";
        case OCEAN_TENSOR_FLOAT32: return "<f4";
        case OCEAN_TENSOR_FLOAT64: return "<f8";
    }
    ocean_tensor_fail("unsupported Tensor dtype for .npy");
    return "";
}

static bool ocean_tensor_npy_write(
    FILE *stream,
    const void *data,
    size_t bytes,
    size_t item_size
) {
    if (ocean_tensor_host_is_little_endian() || item_size <= 1 || bytes == 0) {
        return fwrite(data, 1, bytes, stream) == bytes;
    }

    unsigned char *swapped = (unsigned char *)malloc(bytes);
    if (!swapped) ocean_tensor_fail("out of memory byte-swapping .npy data");
    const unsigned char *source = (const unsigned char *)data;
    for (size_t offset = 0; offset < bytes; offset += item_size) {
        for (size_t byte = 0; byte < item_size; ++byte) {
            swapped[offset + byte] = source[offset + item_size - byte - 1];
        }
    }
    bool written = fwrite(swapped, 1, bytes, stream) == bytes;
    free(swapped);
    return written;
}

static char *ocean_tensor_npy_header(
    const ocean_tensor_handle_t tensor,
    unsigned char major,
    size_t *header_size_out
) {
    if (tensor->ndim > (SIZE_MAX - 256u) / 64u) {
        ocean_tensor_fail("Tensor rank is too large for .npy header");
    }
    size_t capacity = 256u + tensor->ndim * 64u;
    char *header = (char *)malloc(capacity);
    if (!header) ocean_tensor_fail("out of memory allocating .npy header");

    int prefix = snprintf(
        header, capacity,
        "{'descr': '%s', 'fortran_order': False, 'shape': (",
        ocean_tensor_npy_descr(tensor->dtype)
    );
    if (prefix < 0 || (size_t)prefix >= capacity) {
        free(header);
        ocean_tensor_fail("failed to format .npy header");
    }
    size_t position = (size_t)prefix;
    for (size_t axis = 0; axis < tensor->ndim; ++axis) {
        const char *separator = axis + 1 < tensor->ndim
            ? ", " : (tensor->ndim == 1 ? "," : "");
        int written = snprintf(
            header + position, capacity - position, "%zu%s",
            tensor->shape[axis], separator
        );
        if (written < 0 || (size_t)written >= capacity - position) {
            free(header);
            ocean_tensor_fail("failed to format .npy shape");
        }
        position += (size_t)written;
    }
    int suffix = snprintf(header + position, capacity - position, "), }");
    if (suffix < 0 || (size_t)suffix >= capacity - position) {
        free(header);
        ocean_tensor_fail("failed to finish .npy header");
    }
    position += (size_t)suffix;

    size_t prefix_size = major == 1 ? 10u : 12u;
    size_t with_newline = prefix_size + position + 1u;
    size_t padding = (64u - (with_newline % 64u)) % 64u;
    if (position > capacity - padding - 2u) {
        free(header);
        ocean_tensor_fail(".npy header is too large");
    }
    memset(header + position, ' ', padding);
    position += padding;
    header[position++] = '\n';
    header[position] = '\0';
    *header_size_out = position;
    return header;
}

static void ocean_tensor_npy_write_u16(FILE *stream, uint16_t value) {
    unsigned char bytes[2] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
    };
    if (fwrite(bytes, 1, sizeof(bytes), stream) != sizeof(bytes)) {
        ocean_tensor_fail("failed writing .npy header length");
    }
}

static void ocean_tensor_npy_write_u32(FILE *stream, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)((value >> 24) & 0xffu),
    };
    if (fwrite(bytes, 1, sizeof(bytes), stream) != sizeof(bytes)) {
        ocean_tensor_fail("failed writing .npy header length");
    }
}

void ocean_tensor_save_npy(
    ocean_tensor_handle_t tensor,
    const char *path
) {
    if (!tensor || !path) ocean_tensor_fail("Tensor.save_npy received invalid arguments");
    if (tensor->ndim == 0) ocean_tensor_fail("cannot save a scalar Tensor as .npy");

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? NULL : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t source = cpu ? cpu : tensor;
    ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(source)
        ? NULL : ocean_tensor_contiguous(source);
    if (contiguous) source = contiguous;

    size_t header_size = 0;
    unsigned char version = 1;
    char *header = ocean_tensor_npy_header(source, version, &header_size);
    if (header_size > UINT16_MAX) {
        free(header);
        version = 2;
        header = ocean_tensor_npy_header(source, version, &header_size);
    }
    if (version == 1 && header_size > UINT16_MAX) {
        free(header);
        if (contiguous) ocean_tensor_release(contiguous);
        if (cpu) ocean_tensor_release(cpu);
        ocean_tensor_fail(".npy v2 header length is required but could not be encoded");
    }
    if (version == 2 && header_size > UINT32_MAX) {
        free(header);
        if (contiguous) ocean_tensor_release(contiguous);
        if (cpu) ocean_tensor_release(cpu);
        ocean_tensor_fail(".npy header is too large");
    }

    FILE *stream = fopen(path, "wb");
    if (!stream) {
        free(header);
        if (contiguous) ocean_tensor_release(contiguous);
        if (cpu) ocean_tensor_release(cpu);
        ocean_tensor_fail("could not open .npy file for writing");
    }
    const unsigned char magic[6] = {0x93u, 'N', 'U', 'M', 'P', 'Y'};
    const unsigned char version_bytes[2] = {version, 0};
    bool ok = fwrite(magic, 1, sizeof(magic), stream) == sizeof(magic) &&
        fwrite(version_bytes, 1, sizeof(version_bytes), stream) == sizeof(version_bytes);
    if (ok) {
        if (version == 1) ocean_tensor_npy_write_u16(stream, (uint16_t)header_size);
        else ocean_tensor_npy_write_u32(stream, (uint32_t)header_size);
        ok = fwrite(header, 1, header_size, stream) == header_size &&
            ocean_tensor_npy_write(
                stream, source->cpu_data, ocean_tensor_bytes(source), source->item_size
            );
    }
    free(header);
    bool close_ok = fclose(stream) == 0;
    if (contiguous) ocean_tensor_release(contiguous);
    if (cpu) ocean_tensor_release(cpu);
    if (!ok || !close_ok) ocean_tensor_fail("failed writing .npy file");
}

static uint16_t ocean_tensor_npy_read_u16(const unsigned char *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static uint32_t ocean_tensor_npy_read_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static const char *ocean_tensor_npy_value(
    const char *header,
    const char *name
) {
    char quoted_name[32];
    int written = snprintf(quoted_name, sizeof(quoted_name), "'%s'", name);
    if (written < 0 || (size_t)written >= sizeof(quoted_name)) {
        ocean_tensor_fail("invalid .npy header field name");
    }
    const char *field = strstr(header, quoted_name);
    if (!field) {
        written = snprintf(quoted_name, sizeof(quoted_name), "\"%s\"", name);
        if (written < 0 || (size_t)written >= sizeof(quoted_name)) {
            ocean_tensor_fail("invalid .npy header field name");
        }
        field = strstr(header, quoted_name);
    }
    if (!field) ocean_tensor_fail(".npy header is missing a required field");
    const char *colon = strchr(field, ':');
    if (!colon) ocean_tensor_fail("invalid .npy header field");
    colon += 1;
    while (isspace((unsigned char)*colon)) ++colon;
    return colon;
}

static void ocean_tensor_npy_read_descr(
    const char *header,
    ocean_tensor_dtype *dtype_out,
    bool *swap_out
) {
    const char *value = ocean_tensor_npy_value(header, "descr");
    if (*value != '\'' && *value != '"') ocean_tensor_fail("invalid .npy descr");
    char quote = *value++;
    char descr[32];
    size_t length = 0;
    while (value[length] && value[length] != quote) {
        if (length + 1 >= sizeof(descr)) ocean_tensor_fail(".npy descr is too long");
        descr[length] = value[length];
        ++length;
    }
    if (value[length] != quote || length < 3) ocean_tensor_fail("invalid .npy descr");
    descr[length] = '\0';

    char order = descr[0];
    char kind = descr[1];
    if (order != '<' && order != '>' && order != '|' && order != '=') {
        ocean_tensor_fail("unsupported .npy byte order");
    }
    errno = 0;
    char *end = NULL;
    unsigned long item_size = strtoul(descr + 2, &end, 10);
    if (errno == ERANGE || end == descr + 2 || *end != '\0' || item_size > SIZE_MAX) {
        ocean_tensor_fail("invalid .npy dtype size");
    }

    ocean_tensor_dtype dtype;
    switch (kind) {
        case 'b':
        case '?':
            if (item_size != 1) ocean_tensor_fail("invalid .npy bool dtype");
            dtype = OCEAN_TENSOR_BOOL;
            break;
        case 'i':
            if (item_size == 1) dtype = OCEAN_TENSOR_INT8;
            else if (item_size == 2) dtype = OCEAN_TENSOR_INT16;
            else if (item_size == 4) dtype = OCEAN_TENSOR_INT32;
            else if (item_size == 8) dtype = OCEAN_TENSOR_INT64;
            else ocean_tensor_fail("unsupported .npy signed integer dtype");
            break;
        case 'u':
            if (item_size == 1) dtype = OCEAN_TENSOR_UINT8;
            else if (item_size == 2) dtype = OCEAN_TENSOR_UINT16;
            else if (item_size == 4) dtype = OCEAN_TENSOR_UINT32;
            else if (item_size == 8) dtype = OCEAN_TENSOR_UINT64;
            else ocean_tensor_fail("unsupported .npy unsigned integer dtype");
            break;
        case 'f':
            if (item_size == 2) dtype = OCEAN_TENSOR_FLOAT16;
            else if (item_size == 4) dtype = OCEAN_TENSOR_FLOAT32;
            else if (item_size == 8) dtype = OCEAN_TENSOR_FLOAT64;
            else ocean_tensor_fail("unsupported .npy floating-point dtype");
            break;
        default:
            ocean_tensor_fail(".npy dtype is not a supported numeric type");
            return;
    }
    if (item_size > 1 && order == '|') {
        ocean_tensor_fail("invalid .npy byte order for multi-byte dtype");
    }
    bool host_little = ocean_tensor_host_is_little_endian();
    *swap_out = item_size > 1 &&
        ((order == '<' && !host_little) || (order == '>' && host_little));
    *dtype_out = dtype;
}

static size_t *ocean_tensor_npy_read_shape(
    const char *header,
    size_t *ndim_out
) {
    const char *cursor = ocean_tensor_npy_value(header, "shape");
    if (*cursor != '(') ocean_tensor_fail("invalid .npy shape");
    ++cursor;
    size_t capacity = 4;
    size_t ndim = 0;
    size_t *shape = (size_t *)malloc(capacity * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory reading .npy shape");
    for (;;) {
        while (isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == ')') break;
        errno = 0;
        char *end = NULL;
        unsigned long long value = strtoull(cursor, &end, 10);
        if (errno == ERANGE || end == cursor || value > SIZE_MAX) {
            free(shape);
            ocean_tensor_fail("invalid .npy shape dimension");
        }
        if (ndim == capacity) {
            if (capacity > SIZE_MAX / 2u) {
                free(shape);
                ocean_tensor_fail(".npy rank is too large");
            }
            capacity *= 2u;
            size_t *grown = (size_t *)realloc(shape, capacity * sizeof(size_t));
            if (!grown) {
                free(shape);
                ocean_tensor_fail("out of memory growing .npy shape");
            }
            shape = grown;
        }
        shape[ndim++] = (size_t)value;
        cursor = end;
        while (isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor != ')') {
            free(shape);
            ocean_tensor_fail("invalid .npy shape tuple");
        }
        break;
    }
    if (ndim == 0) {
        free(shape);
        ocean_tensor_fail("scalar .npy arrays are not supported as Tensor");
    }
    *ndim_out = ndim;
    return shape;
}

ocean_tensor_handle_t ocean_tensor_load_npy(
    const char *path,
    const char *device
) {
    if (!path) ocean_tensor_fail("Tensor.load_npy requires a path");
    FILE *stream = fopen(path, "rb");
    if (!stream) ocean_tensor_fail("could not open .npy file for reading");

    unsigned char magic[6];
    unsigned char version[2];
    if (fread(magic, 1, sizeof(magic), stream) != sizeof(magic) ||
        memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0 ||
        fread(version, 1, sizeof(version), stream) != sizeof(version)) {
        fclose(stream);
        ocean_tensor_fail("invalid .npy file header");
    }
    uint64_t header_size;
    if (version[0] == 1) {
        unsigned char length[2];
        if (fread(length, 1, sizeof(length), stream) != sizeof(length)) {
            fclose(stream);
            ocean_tensor_fail("truncated .npy header length");
        }
        header_size = ocean_tensor_npy_read_u16(length);
    } else if (version[0] == 2 || version[0] == 3) {
        unsigned char length[4];
        if (fread(length, 1, sizeof(length), stream) != sizeof(length)) {
            fclose(stream);
            ocean_tensor_fail("truncated .npy header length");
        }
        header_size = ocean_tensor_npy_read_u32(length);
    } else {
        fclose(stream);
        ocean_tensor_fail("unsupported .npy format version");
    }
    if (header_size == 0 || header_size > 64u * 1024u * 1024u ||
        header_size > SIZE_MAX - 1u) {
        fclose(stream);
        ocean_tensor_fail("invalid .npy header size");
    }
    char *header = (char *)malloc((size_t)header_size + 1u);
    if (!header) {
        fclose(stream);
        ocean_tensor_fail("out of memory reading .npy header");
    }
    if (fread(header, 1, (size_t)header_size, stream) != (size_t)header_size) {
        free(header);
        fclose(stream);
        ocean_tensor_fail("truncated .npy header");
    }
    header[header_size] = '\0';
    const char *fortran = ocean_tensor_npy_value(header, "fortran_order");
    if (strncmp(fortran, "False", 5) != 0) {
        free(header);
        fclose(stream);
        ocean_tensor_fail("Fortran-order .npy arrays are not supported yet");
    }
    ocean_tensor_dtype dtype;
    bool swap = false;
    ocean_tensor_npy_read_descr(header, &dtype, &swap);
    size_t ndim = 0;
    size_t *shape = ocean_tensor_npy_read_shape(header, &ndim);
    free(header);

    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, ndim, dtype, OCEAN_TENSOR_CPU
    );
    free(shape);
    size_t bytes = ocean_tensor_bytes(result);
    if (!swap) {
        if (fread(result->cpu_data, 1, bytes, stream) != bytes) {
            ocean_tensor_release(result);
            fclose(stream);
            ocean_tensor_fail("truncated .npy data");
        }
    } else {
        unsigned char *payload = (unsigned char *)malloc(bytes);
        if (!payload && bytes != 0) {
            ocean_tensor_release(result);
            fclose(stream);
            ocean_tensor_fail("out of memory byte-swapping .npy data");
        }
        if (fread(payload, 1, bytes, stream) != bytes) {
            free(payload);
            ocean_tensor_release(result);
            fclose(stream);
            ocean_tensor_fail("truncated .npy data");
        }
        unsigned char *destination = (unsigned char *)result->cpu_data;
        for (size_t offset = 0; offset < bytes; offset += result->item_size) {
            for (size_t byte = 0; byte < result->item_size; ++byte) {
                destination[offset + byte] =
                    payload[offset + result->item_size - byte - 1];
            }
        }
        free(payload);
    }
    fclose(stream);

    if (device && strcmp(device, "cpu") == 0) return result;
    ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
    ocean_tensor_release(result);
    return moved;
}

ocean_tensor_handle_t ocean_tensor_load_npy_typed(
    const char *path,
    const char *device,
    const char *expected_dtype
) {
    if (!expected_dtype) {
        ocean_tensor_fail("Tensor.load_npy requires an expected dtype");
    }
    ocean_tensor_dtype expected = ocean_tensor_parse_dtype(expected_dtype);
    ocean_tensor_handle_t result = ocean_tensor_load_npy(path, device);
    if (result->dtype != expected) {
        ocean_tensor_release(result);
        ocean_tensor_fail(".npy dtype does not match Tensor[T]");
    }
    return result;
}

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis) {
    if (!tensor) ocean_tensor_fail("shape() does not accept a null Tensor");
    if (axis < 0 || (size_t)axis >= tensor->ndim) {
        ocean_tensor_fail("Tensor shape axis is out of bounds");
    }
    return tensor->shape[axis] > (size_t)INT32_MAX
        ? INT32_MAX : (int)tensor->shape[axis];
}

int ocean_tensor_len(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("len() does not accept a null Tensor");
    return tensor->size > (size_t)INT32_MAX
        ? INT32_MAX : (int)tensor->size;
}

int ocean_tensor_ndim(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("ndim() does not accept a null Tensor");
    return tensor->ndim > (size_t)INT32_MAX ? INT32_MAX : (int)tensor->ndim;
}

size_t ocean_tensor_size(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("size() does not accept a null Tensor");
    return tensor->size;
}

char *ocean_tensor_device(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("device() does not accept a null Tensor");
    const char *name = tensor->device == OCEAN_TENSOR_GPU ? "gpu" : "cpu";
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying device name");
    strcpy(result, name);
    return result;
}

char *ocean_tensor_device_info(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("device_info() does not accept a null Tensor");
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU && ocean_tensor_opencl_initialized) {
        size_t size = 0;
        ocean_tensor_opencl_check(
            clGetDeviceInfo(
                ocean_tensor_opencl.device,
                CL_DEVICE_NAME,
                0,
                NULL,
                &size
            ),
            "clGetDeviceInfo"
        );
        char *result = (char *)calloc(size + 1, 1);
        if (!result) ocean_tensor_fail("out of memory copying OpenCL device name");
        ocean_tensor_opencl_check(
            clGetDeviceInfo(
                ocean_tensor_opencl.device,
                CL_DEVICE_NAME,
                size,
                result,
                NULL
            ),
            "clGetDeviceInfo"
        );
        return result;
    }
#endif
    const char *name = tensor->device == OCEAN_TENSOR_GPU
        ? "OpenCL GPU (not initialized)"
        : "CPU";
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying device info");
    strcpy(result, name);
    return result;
}

uint64_t ocean_tensor_identity(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor identity on null handle");
    return tensor->identity;
}

void ocean_tensor_release(ocean_tensor_handle_t tensor) {
    if (!tensor) return;
    ocean_tensor_backend_for_device(tensor->device)->release(tensor);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

ocean_tensor_handle_t ocean_tensor_permute(
    ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
) {
    if (!tensor) {
        ocean_tensor_fail("Tensor.permute requires a Tensor");
    }
    if (!axes) {
        ocean_tensor_fail("Tensor.permute requires axes");
    }

    int rank = ocean_tensor_ndim(tensor);
    if (rank <= 0 || (size_t)rank != ndim) {
        ocean_tensor_fail("Tensor.permute axes must match Tensor rank");
    }

    int *normalized = (int *)malloc(ndim * sizeof(int));
    bool *seen = (bool *)calloc(ndim, sizeof(bool));

    if (!normalized || !seen) {
        free(normalized);
        free(seen);
        ocean_tensor_fail("out of memory in Tensor.permute");
    }

    for (size_t i = 0; i < ndim; ++i) {
        long long axis = (long long)axes[i];
        if (axis < 0) axis += (long long)rank;
        if (axis < 0 || axis >= (long long)rank) {
            free(normalized);
            free(seen);
            ocean_tensor_fail("Tensor.permute axis is out of bounds");
        }
        if (seen[(size_t)axis]) {
            free(normalized);
            free(seen);
            ocean_tensor_fail("Tensor.permute axes must be unique");
        }

        normalized[i] = (int)axis;
        seen[(size_t)axis] = true;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU) {
        size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
        if (!shape) {
            free(normalized);
            free(seen);
            ocean_tensor_fail("out of memory in Tensor.permute shape");
        }
        for (size_t axis = 0; axis < ndim; ++axis) {
            shape[axis] = tensor->shape[(size_t)normalized[axis]];
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_permute(tensor, result, normalized);
        free(shape);
        free(normalized);
        free(seen);
        return result;
    }
#endif

    ocean_tensor_handle_t result = ocean_tensor_permute_cpu(
        tensor, normalized, ndim
    );
    free(normalized);
    free(seen);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_permute_cpu(
    const ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
) {
    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory in CPU Tensor.permute shape");
    for (size_t axis = 0; axis < ndim; ++axis) {
        shape[axis] = tensor->shape[(size_t)axes[axis]];
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    free(shape);

    size_t *coordinates = (size_t *)calloc(
        ndim == 0 ? 1 : ndim, sizeof(size_t)
    );
    if (!coordinates) {
        ocean_tensor_release(result);
        ocean_tensor_fail("out of memory in CPU Tensor.permute coordinates");
    }

    for (size_t output_index = 0; output_index < result->size; ++output_index) {
        size_t remaining = output_index;
        size_t input_index = 0;

        for (size_t axis = ndim; axis-- > 0;) {
            size_t extent = result->shape[axis];
            size_t coordinate = extent == 0 ? 0 : remaining % extent;
            remaining = extent == 0 ? 0 : remaining / extent;
            coordinates[axis] = coordinate;
            input_index += coordinate * tensor->strides[(size_t)axes[axis]];
        }

        ocean_tensor_write_scalar(
            result,
            output_index,
            ocean_tensor_read_scalar(tensor, input_index)
        );
    }

    free(coordinates);
    return result;
}
