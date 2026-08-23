#include "std/tensor/tensor_cuda_backend.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

extern "C" void ocean_tensor_fail(const char *message);

static void ocean_cuda_check(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return;
    char message[512];
    snprintf(
        message, sizeof(message), "CUDA %s failed: %s",
        operation, cudaGetErrorString(status)
    );
    ocean_tensor_fail(message);
}

static void ocean_cuda_check_launch(const char *operation) {
    ocean_cuda_check(cudaGetLastError(), operation);
}

void *ocean_cuda_malloc(size_t bytes) {
    if (bytes == 0) return NULL;
    void *device_data = NULL;
    ocean_cuda_check(cudaMalloc(&device_data, bytes), "cudaMalloc");
    return device_data;
}

void ocean_cuda_free(void *device_data) {
    if (device_data != NULL) {
        ocean_cuda_check(cudaFree(device_data), "cudaFree");
    }
}

void ocean_cuda_memcpy_h2d(
    void *device_data,
    const void *host_data,
    size_t bytes
) {
    if (bytes == 0) return;
    ocean_cuda_check(
        cudaMemcpy(device_data, host_data, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy host-to-device"
    );
}

void ocean_cuda_memcpy_d2h(
    void *host_data,
    const void *device_data,
    size_t bytes
) {
    if (bytes == 0) return;
    ocean_cuda_check(
        cudaMemcpy(host_data, device_data, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy device-to-host"
    );
}

void ocean_cuda_memcpy_d2d(
    void *destination,
    const void *source,
    size_t bytes
) {
    if (bytes == 0) return;
    ocean_cuda_check(
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice),
        "cudaMemcpy device-to-device"
    );
}

void ocean_cuda_zero(void *device_data, size_t bytes) {
    if (bytes == 0) return;
    ocean_cuda_check(cudaMemset(device_data, 0, bytes), "cudaMemset");
}

template <typename T>
__device__ T ocean_cuda_apply_binary(T left, T right, int operation) {
    if (operation == 0) return left + right;
    if (operation == 1) return left - right;
    if (operation == 2) return left * right;
    return left / right;
}

template <typename T>
__global__ void ocean_cuda_binary_kernel(
    const T *left,
    const T *right,
    T *output,
    size_t size,
    int operation
) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = ocean_cuda_apply_binary(left[index], right[index], operation);
    }
}

__global__ void ocean_cuda_binary_broadcast_f32_kernel(
    const float *left,
    const float *right,
    float *output,
    size_t size,
    int operation,
    ocean_cuda_broadcast_desc descriptor
) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    size_t remaining = index;
    size_t left_offset = 0;
    size_t right_offset = 0;
    for (int axis = descriptor.ndim - 1; axis >= 0; --axis) {
        size_t coordinate = descriptor.output_shape[axis] == 0
            ? 0 : remaining % descriptor.output_shape[axis];
        remaining = descriptor.output_shape[axis] == 0
            ? 0 : remaining / descriptor.output_shape[axis];
        if (descriptor.left_shape[axis] != 1) {
            left_offset += coordinate * descriptor.left_strides[axis];
        }
        if (descriptor.right_shape[axis] != 1) {
            right_offset += coordinate * descriptor.right_strides[axis];
        }
    }
    float a = left[left_offset];
    float b = right[right_offset];
    if (operation == 0) output[index] = a + b;
    else if (operation == 1) output[index] = a - b;
    else if (operation == 2) output[index] = a * b;
    else output[index] = a / b;
}

template <typename T>
__global__ void ocean_cuda_scalar_kernel(
    const T *input,
    T *output,
    size_t size,
    T scalar,
    int operation
) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = ocean_cuda_apply_binary(input[index], scalar, operation);
    }
}

template <typename T>
__global__ void ocean_cuda_fill_kernel(T *output, size_t size, T value) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) output[index] = value;
}

template <typename T>
__global__ void ocean_cuda_set_scalar_kernel(
    T *output, size_t index, T value
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) output[index] = value;
}

template <typename T, typename Accumulator>
__global__ void ocean_cuda_matmul_kernel(
    const T *left,
    const T *right,
    T *output,
    int rows,
    int inner,
    int columns
) {
    __shared__ T left_tile[16][16];
    __shared__ T right_tile[16][16];

    int row = (int)blockIdx.y * 16 + (int)threadIdx.y;
    int column = (int)blockIdx.x * 16 + (int)threadIdx.x;
    Accumulator sum = (Accumulator)0;

    for (int tile = 0; tile < inner; tile += 16) {
        int left_column = tile + (int)threadIdx.x;
        int right_row = tile + (int)threadIdx.y;

        left_tile[threadIdx.y][threadIdx.x] =
            row < rows && left_column < inner
                ? left[row * inner + left_column] : (T)0;
        right_tile[threadIdx.y][threadIdx.x] =
            right_row < inner && column < columns
                ? right[right_row * columns + column] : (T)0;
        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            sum += (Accumulator)left_tile[threadIdx.y][k] *
                   (Accumulator)right_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && column < columns) {
        output[row * columns + column] = (T)sum;
    }
}

static int ocean_cuda_blocks(size_t size) {
    const size_t threads = 256;
    size_t blocks = (size + threads - 1u) / threads;
    if (blocks > (size_t)2147483647) {
        ocean_tensor_fail("CUDA launch grid is too large");
    }
    return (int)blocks;
}

static int ocean_cuda_packed_blocks(int rows, int columns) {
    if (rows <= 0 || columns <= 0) return 0;
    size_t groups_per_row = ((size_t)columns + 127u) / 128u;
    size_t blocks = (size_t)rows * groups_per_row;
    if (blocks > (size_t)2147483647) {
        ocean_tensor_fail("CUDA packed kernel grid is too large");
    }
    return (int)blocks;
}

void ocean_cuda_fill_f32(void *device_data, float value, size_t size) {
    if (size == 0) return;
    ocean_cuda_fill_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (float *)device_data, size, value
    );
    ocean_cuda_check_launch("float32 fill kernel");
}

void ocean_cuda_fill_i32(void *device_data, int value, size_t size) {
    if (size == 0) return;
    ocean_cuda_fill_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (int32_t *)device_data, size, value
    );
    ocean_cuda_check_launch("int32 fill kernel");
}

void ocean_cuda_set_f32(void *device_data, size_t index, float value) {
    ocean_cuda_set_scalar_kernel<<<1, 1>>>(
        (float *)device_data, index, value
    );
    ocean_cuda_check_launch("float32 scalar assignment kernel");
}

void ocean_cuda_set_i32(void *device_data, size_t index, int value) {
    ocean_cuda_set_scalar_kernel<<<1, 1>>>(
        (int32_t *)device_data, index, value
    );
    ocean_cuda_check_launch("int32 scalar assignment kernel");
}

void ocean_cuda_set_i64(void *device_data, size_t index, int64_t value) {
    ocean_cuda_set_scalar_kernel<<<1, 1>>>(
        (int64_t *)device_data, index, value
    );
    ocean_cuda_check_launch("int64 scalar assignment kernel");
}

void ocean_cuda_binary_f32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation
) {
    if (size == 0) return;
    ocean_cuda_binary_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)left, (const float *)right, (float *)output,
        size, operation
    );
    ocean_cuda_check_launch("float32 binary kernel");
}

void ocean_cuda_binary_i32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation
) {
    if (size == 0) return;
    ocean_cuda_binary_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const int32_t *)left, (const int32_t *)right, (int32_t *)output,
        size, operation
    );
    ocean_cuda_check_launch("int32 binary kernel");
}

void ocean_cuda_binary_broadcast_f32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation,
    const ocean_cuda_broadcast_desc *descriptor
) {
    if (size == 0) return;
    ocean_cuda_binary_broadcast_f32_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)left, (const float *)right, (float *)output,
        size, operation, *descriptor
    );
    ocean_cuda_check_launch("float32 broadcast binary kernel");
}

void ocean_cuda_scalar_f32(
    const void *input,
    void *output,
    size_t size,
    float scalar,
    int operation
) {
    if (size == 0) return;
    ocean_cuda_scalar_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)input, (float *)output, size, scalar, operation
    );
    ocean_cuda_check_launch("float32 scalar kernel");
}

void ocean_cuda_scalar_i32(
    const void *input,
    void *output,
    size_t size,
    int scalar,
    int operation
) {
    if (size == 0) return;
    ocean_cuda_scalar_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const int32_t *)input, (int32_t *)output, size, scalar, operation
    );
    ocean_cuda_check_launch("int32 scalar kernel");
}

void ocean_cuda_matmul_f32(
    const void *left,
    const void *right,
    void *output,
    int rows,
    int inner,
    int columns
) {
    if (rows == 0 || columns == 0) return;
    if (inner == 0) {
        ocean_cuda_zero(output, (size_t)rows * (size_t)columns * sizeof(float));
        return;
    }
    dim3 threads(16, 16);
    dim3 blocks(
        ((unsigned int)columns + 15u) / 16u,
        ((unsigned int)rows + 15u) / 16u
    );
    ocean_cuda_matmul_kernel<float, float><<<blocks, threads>>>(
        (const float *)left, (const float *)right, (float *)output,
        rows, inner, columns
    );
    ocean_cuda_check_launch("float32 matmul kernel");
}

void ocean_cuda_matmul_i32(
    const void *left,
    const void *right,
    void *output,
    int rows,
    int inner,
    int columns
) {
    if (rows == 0 || columns == 0) return;
    if (inner == 0) {
        ocean_cuda_zero(
            output, (size_t)rows * (size_t)columns * sizeof(int32_t)
        );
        return;
    }
    dim3 threads(16, 16);
    dim3 blocks(
        ((unsigned int)columns + 15u) / 16u,
        ((unsigned int)rows + 15u) / 16u
    );
    ocean_cuda_matmul_kernel<int32_t, long long><<<blocks, threads>>>(
        (const int32_t *)left, (const int32_t *)right, (int32_t *)output,
        rows, inner, columns
    );
    ocean_cuda_check_launch("int32 matmul kernel");
}

static __device__ float ocean_cuda_ternary_value(
    const int32_t *packed,
    int row,
    int column,
    int packed_cols
) {
    uint32_t word = (uint32_t)packed[row * packed_cols + column / 16];
    uint32_t code = (word >> (2 * (column % 16))) & 3u;
    return code == 1u ? 1.0f : (code == 2u ? -1.0f : 0.0f);
}

__global__ void ocean_cuda_softmax_kernel(
    const float *input, float *output, int rows, int width
) {
    extern __shared__ float partial[];
    int row = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (row >= rows) return;
    int offset = row * width;
    float maximum = -INFINITY;
    for (int index = lane; index < width; index += blockDim.x) {
        maximum = fmaxf(maximum, input[offset + index]);
    }
    partial[lane] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        __syncthreads();
    }
    maximum = partial[0];
    float sum = 0.0f;
    for (int index = lane; index < width; index += blockDim.x) {
        sum += expf(input[offset + index] - maximum);
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    sum = partial[0];
    for (int index = lane; index < width; index += blockDim.x) {
        output[offset + index] = expf(input[offset + index] - maximum) / sum;
    }
}

__global__ void ocean_cuda_causal_softmax_kernel(
    const float *input, float *output, int rows, int width
) {
    extern __shared__ float partial[];
    int row = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (row >= rows) return;
    int query = row % width;
    int offset = row * width;
    float maximum = -INFINITY;
    for (int index = lane; index <= query; index += blockDim.x) {
        maximum = fmaxf(maximum, input[offset + index]);
    }
    partial[lane] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        __syncthreads();
    }
    maximum = partial[0];
    float sum = 0.0f;
    for (int index = lane; index <= query; index += blockDim.x) {
        sum += expf(input[offset + index] - maximum);
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    sum = partial[0];
    for (int index = lane; index < width; index += blockDim.x) {
        output[offset + index] = index <= query
            ? expf(input[offset + index] - maximum) / sum : 0.0f;
    }
}

__global__ void ocean_cuda_layer_norm_kernel(
    const float *input, float *output, int rows, int width, float epsilon
) {
    extern __shared__ float partial[];
    int row = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (row >= rows) return;
    int offset = row * width;
    float sum = 0.0f;
    for (int index = lane; index < width; index += blockDim.x) sum += input[offset + index];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float mean = partial[0] / (float)width;
    float variance = 0.0f;
    for (int index = lane; index < width; index += blockDim.x) {
        float delta = input[offset + index] - mean;
        variance += delta * delta;
    }
    partial[lane] = variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float inverse_std = rsqrtf(partial[0] / (float)width + epsilon);
    for (int index = lane; index < width; index += blockDim.x) {
        output[offset + index] = (input[offset + index] - mean) * inverse_std;
    }
}

__global__ void ocean_cuda_layer_norm_affine_kernel(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    int rows,
    int width,
    float epsilon
) {
    extern __shared__ float partial[];
    int row = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (row >= rows) return;
    int offset = row * width;
    float sum = 0.0f;
    for (int index = lane; index < width; index += blockDim.x) {
        sum += input[offset + index];
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float mean = partial[0] / (float)width;
    float variance = 0.0f;
    for (int index = lane; index < width; index += blockDim.x) {
        float delta = input[offset + index] - mean;
        variance += delta * delta;
    }
    partial[lane] = variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float inverse_std = rsqrtf(partial[0] / (float)width + epsilon);
    for (int index = lane; index < width; index += blockDim.x) {
        float normalized = (input[offset + index] - mean) * inverse_std;
        output[offset + index] = normalized * gamma[index] + beta[index];
    }
}

__global__ void ocean_cuda_gelu_kernel(
    const float *input, float *output, size_t size
) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const float coefficient = 0.7978845608028654f;
    const float cubic = 0.044715f;
    float value = input[index];
    float argument = coefficient * (value + cubic * value * value * value);
    output[index] = 0.5f * value * (1.0f + tanhf(argument));
}

__global__ void ocean_cuda_ternary_scale_kernel(
    const float *input, float *output, size_t size
) {
    extern __shared__ float partial[];
    int lane = (int)threadIdx.x;
    float sum = 0.0f;
    for (size_t index = (size_t)lane; index < size; index += blockDim.x) {
        sum += fabsf(input[index]);
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float scale = partial[0] / (float)size;
    if (scale < 1.0e-8f) scale = 1.0e-8f;
    float threshold = 0.5f * scale;
    for (size_t index = (size_t)lane; index < size; index += blockDim.x) {
        float value = input[index];
        output[index] = value > threshold ? scale : (value < -threshold ? -scale : 0.0f);
    }
}

__global__ void ocean_cuda_ternary_pack_kernel(
    const float *input, int32_t *output,
    int source_rows, int source_cols, int output_rows, int packed_cols,
    float scale, int transpose
) {
    int index = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    int output_size = output_rows * packed_cols;
    if (index >= output_size) return;
    int row = index / packed_cols;
    int group = index - row * packed_cols;
    uint32_t packed = 0u;
    float threshold = 0.5f * scale;
    for (int bit = 0; bit < 16; ++bit) {
        int source_row = transpose ? group * 16 + bit : row;
        int source_col = transpose ? row : group * 16 + bit;
        uint32_t code = 0u;
        if (source_row < source_rows && source_col < source_cols) {
            float value = input[source_row * source_cols + source_col];
            code = value > threshold ? 1u : (value < -threshold ? 2u : 0u);
        }
        packed |= code << (2 * bit);
    }
    output[index] = (int32_t)packed;
}

__global__ void ocean_cuda_packed_linear_kernel(
    const float *input, const int32_t *packed, const float *bias, float *output,
    int rows, int cols_a, int cols_b, int packed_cols, float scale
) {
    __shared__ float input_tile[128];
    int lane = (int)threadIdx.x;
    int groups_per_row = (cols_b + 127) / 128;
    int group = (int)blockIdx.x;
    int row = group / groups_per_row;
    int column = (group % groups_per_row) * 128 + lane;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int tile = 0; tile < cols_a; tile += 128) {
        int input_index = tile + lane;
        input_tile[lane] = input_index < cols_a
            ? input[row * cols_a + input_index] : 0.0f;
        __syncthreads();
        int tile_end = tile + 128;
        if (tile_end > cols_a) tile_end = cols_a;
        if (column < cols_b) {
            for (int k = tile; k < tile_end; ++k) {
                sum += input_tile[k - tile] * ocean_cuda_ternary_value(
                    packed, k, column, packed_cols
                );
            }
        }
        __syncthreads();
    }
    if (column < cols_b) {
        output[(size_t)row * (size_t)cols_b + (size_t)column] =
            sum * scale + (bias ? bias[column] : 0.0f);
    }
}

__global__ void ocean_cuda_packed_qkv_kernel(
    const float *input,
    const int32_t *q_packed, const float *q_bias,
    const int32_t *k_packed, const float *k_bias,
    const int32_t *v_packed, const float *v_bias,
    float *output, int rows, int cols_a, int cols_b, int packed_cols,
    float q_scale, float k_scale, float v_scale
) {
    __shared__ float input_tile[128];
    int lane = (int)threadIdx.x;
    int groups_per_row = (cols_b + 127) / 128;
    int group = (int)blockIdx.x;
    int row = group / groups_per_row;
    int column = (group % groups_per_row) * 128 + lane;
    if (row >= rows) return;
    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
    for (int tile = 0; tile < cols_a; tile += 128) {
        int input_index = tile + lane;
        input_tile[lane] = input_index < cols_a
            ? input[row * cols_a + input_index] : 0.0f;
        __syncthreads();
        int tile_end = tile + 128;
        if (tile_end > cols_a) tile_end = cols_a;
        if (column < cols_b) {
            for (int k = tile; k < tile_end; ++k) {
                float value = input_tile[k - tile];
                q_sum += value * ocean_cuda_ternary_value(q_packed, k, column, packed_cols);
                k_sum += value * ocean_cuda_ternary_value(k_packed, k, column, packed_cols);
                v_sum += value * ocean_cuda_ternary_value(v_packed, k, column, packed_cols);
            }
        }
        __syncthreads();
    }
    if (column < cols_b) {
        size_t offset = (size_t)row * (size_t)(3 * cols_b) + (size_t)column;
        output[offset] = q_sum * q_scale + q_bias[column];
        output[offset + cols_b] = k_sum * k_scale + k_bias[column];
        output[offset + 2 * cols_b] = v_sum * v_scale + v_bias[column];
    }
}

__global__ void ocean_cuda_packed_qkv_split_kernel(
    const float *input,
    const int32_t *q_packed, const float *q_bias,
    const int32_t *k_packed, const float *k_bias,
    const int32_t *v_packed, const float *v_bias,
    float *q_output, float *k_output, float *v_output,
    int rows, int cols_a, int cols_b, int packed_cols,
    float q_scale, float k_scale, float v_scale
) {
    __shared__ float input_tile[128];
    int lane = (int)threadIdx.x;
    int groups_per_row = (cols_b + 127) / 128;
    int group = (int)blockIdx.x;
    int row = group / groups_per_row;
    int column = (group % groups_per_row) * 128 + lane;
    if (row >= rows) return;
    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
    for (int tile = 0; tile < cols_a; tile += 128) {
        int input_index = tile + lane;
        input_tile[lane] = input_index < cols_a
            ? input[row * cols_a + input_index] : 0.0f;
        __syncthreads();
        int tile_end = tile + 128;
        if (tile_end > cols_a) tile_end = cols_a;
        if (column < cols_b) {
            for (int k = tile; k < tile_end; ++k) {
                float value = input_tile[k - tile];
                q_sum += value * ocean_cuda_ternary_value(q_packed, k, column, packed_cols);
                k_sum += value * ocean_cuda_ternary_value(k_packed, k, column, packed_cols);
                v_sum += value * ocean_cuda_ternary_value(v_packed, k, column, packed_cols);
            }
        }
        __syncthreads();
    }
    if (column < cols_b) {
        size_t linear = (size_t)row * (size_t)cols_b + (size_t)column;
        q_output[linear] = q_sum * q_scale + q_bias[column];
        k_output[linear] = k_sum * k_scale + k_bias[column];
        v_output[linear] = v_sum * v_scale + v_bias[column];
    }
}

__global__ void ocean_cuda_cache_write_kernel(
    float *cache, const float *value, int batches, int heads, int sequence,
    int value_sequence, int width, int position
) {
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batches * (size_t)heads * (size_t)value_sequence * (size_t)width;
    if (linear >= total) return;
    size_t row_width = (size_t)value_sequence * (size_t)width;
    size_t head_span = (size_t)heads * row_width;
    size_t batch = linear / head_span;
    size_t remainder = linear % head_span;
    size_t head = remainder / row_width;
    remainder %= row_width;
    size_t value_position = remainder / (size_t)width;
    size_t column = remainder % (size_t)width;
    size_t destination = (batch * (size_t)heads + head) * (size_t)sequence * (size_t)width
        + ((size_t)position + value_position) * (size_t)width + column;
    cache[destination] = value[linear];
}

__global__ void ocean_cuda_permute_swap12_f32_kernel(
    const float *input,
    float *output,
    int batches,
    int first_dim,
    int second_dim,
    int head_dim
) {
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batches * (size_t)first_dim *
        (size_t)second_dim * (size_t)head_dim;
    if (linear >= total) return;
    size_t feature = linear % (size_t)head_dim;
    size_t output_position = linear / (size_t)head_dim;
    size_t output_first = output_position % (size_t)second_dim;
    size_t output_batch_position = output_position / (size_t)second_dim;
    size_t output_second = output_batch_position % (size_t)first_dim;
    size_t batch = output_batch_position / (size_t)first_dim;
    size_t input_index = (((batch * (size_t)first_dim + output_second) *
        (size_t)second_dim + output_first) * (size_t)head_dim) + feature;
    output[linear] = input[input_index];
}

__global__ void ocean_cuda_cache_slice_kernel(
    const float *cache, float *output, int batches, int heads,
    int source_sequence, int output_sequence, int width, int start
) {
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batches * (size_t)heads * (size_t)output_sequence * (size_t)width;
    if (linear >= total) return;
    size_t row_width = (size_t)output_sequence * (size_t)width;
    size_t head_span = (size_t)heads * row_width;
    size_t batch = linear / head_span;
    size_t remainder = linear % head_span;
    size_t head = remainder / row_width;
    remainder %= row_width;
    size_t position = remainder / (size_t)width;
    size_t column = remainder % (size_t)width;
    size_t source = (batch * (size_t)heads + head) * (size_t)source_sequence * (size_t)width
        + ((size_t)start + position) * (size_t)width + column;
    output[linear] = cache[source];
}

__device__ static bool ocean_cuda_sparse_precedes(
    float score, int index, float other_score, int other_index
) {
    return score > other_score ||
        (score == other_score && index < other_index);
}

__global__ void ocean_cuda_sparse_build_summaries_kernel(
    const float *key,
    float *summaries,
    int batches,
    int heads,
    int key_length,
    int active_length,
    int head_dim,
    int block_size,
    int summary_blocks
) {
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batches * (size_t)heads *
        (size_t)summary_blocks * (size_t)head_dim;
    if (linear >= total) return;
    int dimension = (int)(linear % (size_t)head_dim);
    size_t block_linear = linear / (size_t)head_dim;
    int block = (int)(block_linear % (size_t)summary_blocks);
    size_t head_linear = block_linear / (size_t)summary_blocks;
    int head = (int)(head_linear % (size_t)heads);
    int batch = (int)(head_linear / (size_t)heads);
    int start = block * block_size;
    int end = start + block_size;
    if (end > active_length) end = active_length;
    if (start >= end) {
        summaries[linear] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (int token = start; token < end; ++token) {
        size_t offset = ((size_t)batch * (size_t)heads + (size_t)head) *
            (size_t)key_length * (size_t)head_dim +
            (size_t)token * (size_t)head_dim + (size_t)dimension;
        sum += key[offset];
    }
    summaries[linear] = sum / (float)(end - start);
}

__global__ void ocean_cuda_sparse_update_summary_kernel(
    const float *key,
    float *summaries,
    int batches,
    int heads,
    int key_length,
    int summary_blocks,
    int active_length,
    int head_dim,
    int block_size,
    int position
) {
    int lane = (int)threadIdx.x;
    int group = (int)blockIdx.x;
    if (group >= batches * heads || lane >= head_dim) return;
    int batch = group / heads;
    int head = group % heads;
    int block = position / block_size;
    if (block < 0 || block >= summary_blocks) return;
    int start = block * block_size;
    int end = start + block_size;
    if (end > active_length) end = active_length;
    if (start >= end) return;
    float sum = 0.0f;
    for (int token = start; token < end; ++token) {
        size_t offset = ((size_t)batch * (size_t)heads + (size_t)head) *
            (size_t)key_length * (size_t)head_dim +
            (size_t)token * (size_t)head_dim + (size_t)lane;
        sum += key[offset];
    }
    size_t summary_offset = ((size_t)batch * (size_t)heads + (size_t)head) *
        (size_t)summary_blocks * (size_t)head_dim +
        (size_t)block * (size_t)head_dim + (size_t)lane;
    summaries[summary_offset] = sum / (float)(end - start);
}

__global__ void ocean_cuda_sparse_attention_kernel(
    const float *query,
    const float *key,
    const float *value,
    const float *summaries,
    float *output,
    int batches,
    int heads,
    int query_length,
    int key_length,
    int active_length,
    int head_dim,
    int summary_blocks,
    int top_k,
    int top_blocks,
    int block_size,
    float scale,
    int query_start,
    int causal
) {
    extern __shared__ unsigned char storage[];
    float *q_vector = (float *)storage;
    int *selected_blocks = (int *)(q_vector + head_dim);
    float *selected_block_scores =
        (float *)(selected_blocks + top_blocks);
    int *selected_indices =
        (int *)(selected_block_scores + top_blocks);
    float *selected_scores = (float *)(selected_indices + top_k);
    int *selected_count = (int *)(selected_scores + top_k);

    size_t group = (size_t)blockIdx.x;
    size_t total = (size_t)batches * (size_t)heads * (size_t)query_length;
    if (group >= total) return;
    int query_index = (int)(group % (size_t)query_length);
    size_t head_group = group / (size_t)query_length;
    int head = (int)(head_group % (size_t)heads);
    int batch = (int)(head_group / (size_t)heads);
    int lane = (int)threadIdx.x;
    int absolute_query = query_start + query_index;
    size_t query_base = ((size_t)batch * (size_t)heads + (size_t)head) *
        (size_t)query_length * (size_t)head_dim +
        (size_t)query_index * (size_t)head_dim;
    if (lane < head_dim) q_vector[lane] = query[query_base + (size_t)lane];
    __syncthreads();

    if (lane == 0) {
        int block_count = (active_length + block_size - 1) / block_size;
        int block_limit = top_blocks < block_count ? top_blocks : block_count;
        int token_limit = top_k < active_length ? top_k : active_length;
        int chosen_blocks = 0;
        for (int block = 0; block < block_count; ++block) {
            int start = block * block_size;
            int end = start + block_size;
            if (end > active_length) end = active_length;
            int visible_end = end;
            if (causal && visible_end > absolute_query + 1) {
                visible_end = absolute_query + 1;
            }
            if (visible_end <= start) continue;
            int visible_count = visible_end - start;
            float block_score = 0.0f;
            if (visible_count == end - start && block < summary_blocks) {
                size_t summary_base = ((size_t)batch * (size_t)heads +
                    (size_t)head) * (size_t)summary_blocks * (size_t)head_dim +
                    (size_t)block * (size_t)head_dim;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    block_score += q_vector[dimension] *
                        summaries[summary_base + (size_t)dimension];
                }
            } else {
                for (int token = start; token < visible_end; ++token) {
                    size_t key_base = ((size_t)batch * (size_t)heads +
                        (size_t)head) * (size_t)key_length * (size_t)head_dim +
                        (size_t)token * (size_t)head_dim;
                    float score = 0.0f;
                    for (int dimension = 0; dimension < head_dim; ++dimension) {
                        score += q_vector[dimension] *
                            key[key_base + (size_t)dimension];
                    }
                    block_score += score / (float)visible_count;
                }
            }
            block_score *= scale;
            if (chosen_blocks < block_limit) {
                int insert = chosen_blocks;
                while (insert > 0 && ocean_cuda_sparse_precedes(
                    block_score, block,
                    selected_block_scores[insert - 1],
                    selected_blocks[insert - 1]
                )) {
                    selected_block_scores[insert] = selected_block_scores[insert - 1];
                    selected_blocks[insert] = selected_blocks[insert - 1];
                    --insert;
                }
                selected_block_scores[insert] = block_score;
                selected_blocks[insert] = block;
                ++chosen_blocks;
            } else if (ocean_cuda_sparse_precedes(
                block_score, block,
                selected_block_scores[block_limit - 1],
                selected_blocks[block_limit - 1]
            )) {
                int insert = block_limit - 1;
                while (insert > 0 && ocean_cuda_sparse_precedes(
                    block_score, block,
                    selected_block_scores[insert - 1],
                    selected_blocks[insert - 1]
                )) {
                    selected_block_scores[insert] = selected_block_scores[insert - 1];
                    selected_blocks[insert] = selected_blocks[insert - 1];
                    --insert;
                }
                selected_block_scores[insert] = block_score;
                selected_blocks[insert] = block;
            }
        }

        int chosen_tokens = 0;
        for (int selected_block = 0; selected_block < chosen_blocks; ++selected_block) {
            int block = selected_blocks[selected_block];
            int start = block * block_size;
            int end = start + block_size;
            if (end > active_length) end = active_length;
            if (causal && end > absolute_query + 1) end = absolute_query + 1;
            for (int token = start; token < end; ++token) {
                size_t key_base = ((size_t)batch * (size_t)heads +
                    (size_t)head) * (size_t)key_length * (size_t)head_dim +
                    (size_t)token * (size_t)head_dim;
                float token_score = 0.0f;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    token_score += q_vector[dimension] *
                        key[key_base + (size_t)dimension];
                }
                token_score *= scale;
                if (chosen_tokens < token_limit) {
                    int insert = chosen_tokens;
                    while (insert > 0 && ocean_cuda_sparse_precedes(
                        token_score, token,
                        selected_scores[insert - 1],
                        selected_indices[insert - 1]
                    )) {
                        selected_scores[insert] = selected_scores[insert - 1];
                        selected_indices[insert] = selected_indices[insert - 1];
                        --insert;
                    }
                    selected_scores[insert] = token_score;
                    selected_indices[insert] = token;
                    ++chosen_tokens;
                } else if (ocean_cuda_sparse_precedes(
                    token_score, token,
                    selected_scores[token_limit - 1],
                    selected_indices[token_limit - 1]
                )) {
                    int insert = token_limit - 1;
                    while (insert > 0 && ocean_cuda_sparse_precedes(
                        token_score, token,
                        selected_scores[insert - 1],
                        selected_indices[insert - 1]
                    )) {
                        selected_scores[insert] = selected_scores[insert - 1];
                        selected_indices[insert] = selected_indices[insert - 1];
                        --insert;
                    }
                    selected_scores[insert] = token_score;
                    selected_indices[insert] = token;
                }
            }
        }
        float maximum = -INFINITY;
        for (int selected = 0; selected < chosen_tokens; ++selected) {
            if (selected_scores[selected] > maximum) {
                maximum = selected_scores[selected];
            }
        }
        float denominator = 0.0f;
        for (int selected = 0; selected < chosen_tokens; ++selected) {
            selected_scores[selected] = expf(selected_scores[selected] - maximum);
            denominator += selected_scores[selected];
        }
        if (denominator > 0.0f) {
            for (int selected = 0; selected < chosen_tokens; ++selected) {
                selected_scores[selected] /= denominator;
            }
        }
        *selected_count = chosen_tokens;
    }
    __syncthreads();

    size_t output_base = ((size_t)batch * (size_t)heads + (size_t)head) *
        (size_t)query_length * (size_t)head_dim +
        (size_t)query_index * (size_t)head_dim;
    float context = 0.0f;
    if (lane < head_dim) {
        for (int selected = 0; selected < *selected_count; ++selected) {
            int token = selected_indices[selected];
            size_t value_base = ((size_t)batch * (size_t)heads +
                (size_t)head) * (size_t)key_length * (size_t)head_dim +
                (size_t)token * (size_t)head_dim;
            context += selected_scores[selected] * value[value_base + (size_t)lane];
        }
        output[output_base + (size_t)lane] = context;
    }
}

__global__ void ocean_cuda_embedding_kernel(
    const float *weight, const int64_t *indices, float *output,
    int index_count, int vocab, int dim
) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)index_count * (size_t)dim;
    if (index >= total) return;
    int token_position = (int)(index / (size_t)dim);
    int feature = (int)(index % (size_t)dim);
    int64_t token = indices[token_position];
    output[index] = token >= 0 && token < (int64_t)vocab
        ? weight[token * (int64_t)dim + feature] : 0.0f;
}

__global__ void ocean_cuda_argmax_kernel(
    const float *input, int *result, size_t size
) {
    extern __shared__ unsigned char storage[];
    float *values = (float *)storage;
    int *indices = (int *)(values + blockDim.x);
    int lane = (int)threadIdx.x;
    float best_value = -INFINITY;
    int best_index = -1;
    for (size_t index = (size_t)lane; index < size; index += blockDim.x) {
        float value = input[index];
        if (value > best_value || (value == best_value &&
            (best_index < 0 || index < (size_t)best_index))) {
            best_value = value;
            best_index = index > (size_t)INT_MAX ? -1 : (int)index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            float other_value = values[lane + stride];
            int other_index = indices[lane + stride];
            if (other_value > values[lane] || (other_value == values[lane] &&
                other_index >= 0 && (indices[lane] < 0 || other_index < indices[lane]))) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) *result = indices[0];
}

__global__ void ocean_cuda_packed_attention_kernel(
    const float *input,
    const int32_t *q_packed, const float *q_bias,
    const int32_t *k_packed, const float *k_bias,
    const int32_t *v_packed, const float *v_bias,
    float *cache_k, float *cache_v, float *output,
    int cols_a, int packed_cols, int max_seq, int position,
    int n_heads, int head_dim, float q_scale, float k_scale, float v_scale
) {
    extern __shared__ float shared[];
    float *q_vector = shared;
    float *k_vector = q_vector + head_dim;
    float *v_vector = k_vector + head_dim;
    float *scores = v_vector + head_dim;
    float *input_tile = scores + max_seq;
    int lane = (int)threadIdx.x;
    int head = (int)blockIdx.x;
    int channel = head * head_dim + lane;
    if (head >= n_heads) return;
    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
    for (int tile = 0; tile < cols_a; tile += 128) {
        int input_index = tile + lane;
        input_tile[lane] = input_index < cols_a ? input[input_index] : 0.0f;
        __syncthreads();
        int tile_end = tile + 128;
        if (tile_end > cols_a) tile_end = cols_a;
        if (lane < head_dim) {
            for (int input_column = tile; input_column < tile_end; ++input_column) {
                float value = input_tile[input_column - tile];
                q_sum += value * ocean_cuda_ternary_value(
                    q_packed, input_column, channel, packed_cols
                );
                k_sum += value * ocean_cuda_ternary_value(
                    k_packed, input_column, channel, packed_cols
                );
                v_sum += value * ocean_cuda_ternary_value(
                    v_packed, input_column, channel, packed_cols
                );
            }
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        q_vector[lane] = q_sum * q_scale + q_bias[channel];
        k_vector[lane] = k_sum * k_scale + k_bias[channel];
        v_vector[lane] = v_sum * v_scale + v_bias[channel];
    }
    __syncthreads();
    if (lane < head_dim) {
        size_t offset = ((size_t)head * (size_t)max_seq + (size_t)position) * (size_t)head_dim + (size_t)lane;
        cache_k[offset] = k_vector[lane];
        cache_v[offset] = v_vector[lane];
    }
    __syncthreads();
    for (int token = lane; token <= position; token += blockDim.x) {
        float score = 0.0f;
        for (int dimension = 0; dimension < head_dim; ++dimension) {
            size_t offset = ((size_t)head * (size_t)max_seq + (size_t)token) * (size_t)head_dim + (size_t)dimension;
            score += q_vector[dimension] * cache_k[offset];
        }
        scores[token] = score / sqrtf((float)head_dim);
    }
    __syncthreads();
    if (lane == 0) {
        float maximum = -INFINITY;
        for (int token = 0; token <= position; ++token) maximum = fmaxf(maximum, scores[token]);
        float denominator = 0.0f;
        for (int token = 0; token <= position; ++token) {
            scores[token] = expf(scores[token] - maximum);
            denominator += scores[token];
        }
        for (int token = 0; token <= position; ++token) scores[token] /= denominator;
    }
    __syncthreads();
    if (lane < head_dim) {
        float context = 0.0f;
        for (int token = 0; token <= position; ++token) {
            size_t offset = ((size_t)head * (size_t)max_seq + (size_t)token) * (size_t)head_dim + (size_t)lane;
            context += scores[token] * cache_v[offset];
        }
        output[channel] = context;
    }
}

void ocean_cuda_softmax_last_dim(const void *input, void *output, int rows, int width) {
    if (rows <= 0 || width <= 0) return;
    ocean_cuda_softmax_kernel<<<rows, 256, 256 * sizeof(float)>>>(
        (const float *)input, (float *)output, rows, width
    );
    ocean_cuda_check_launch("softmax kernel");
}

void ocean_cuda_causal_softmax(const void *input, void *output, int rows, int width) {
    if (rows <= 0 || width <= 0) return;
    ocean_cuda_causal_softmax_kernel<<<rows, 256, 256 * sizeof(float)>>>(
        (const float *)input, (float *)output, rows, width
    );
    ocean_cuda_check_launch("causal softmax kernel");
}

void ocean_cuda_layer_norm_last_dim(const void *input, void *output, int rows, int width, float epsilon) {
    if (rows <= 0 || width <= 0) return;
    ocean_cuda_layer_norm_kernel<<<rows, 256, 256 * sizeof(float)>>>(
        (const float *)input, (float *)output, rows, width, epsilon
    );
    ocean_cuda_check_launch("layer norm kernel");
}

void ocean_cuda_layer_norm_affine_last_dim(
    const void *input,
    const void *gamma,
    const void *beta,
    void *output,
    int rows,
    int width,
    float epsilon
) {
    if (rows <= 0 || width <= 0) return;
    ocean_cuda_layer_norm_affine_kernel<<<rows, 256, 256 * sizeof(float)>>>(
        (const float *)input,
        (const float *)gamma,
        (const float *)beta,
        (float *)output,
        rows,
        width,
        epsilon
    );
    ocean_cuda_check_launch("fused layer norm affine kernel");
}

void ocean_cuda_gelu(const void *input, void *output, size_t size) {
    if (size == 0) return;
    ocean_cuda_gelu_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)input, (float *)output, size
    );
    ocean_cuda_check_launch("GELU kernel");
}

void ocean_cuda_ternary_quantize(const void *input, void *output, size_t size) {
    if (size == 0) return;
    ocean_cuda_ternary_scale_kernel<<<1, 256, 256 * sizeof(float)>>>(
        (const float *)input, (float *)output, size
    );
    ocean_cuda_check_launch("ternary quantization kernel");
}

void ocean_cuda_ternary_pack(const void *input, void *output, int source_rows, int source_cols, int output_rows, int packed_cols, float scale, int transpose) {
    size_t size = (size_t)output_rows * (size_t)packed_cols;
    if (size == 0) return;
    ocean_cuda_ternary_pack_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)input, (int32_t *)output, source_rows, source_cols,
        output_rows, packed_cols, scale, transpose
    );
    ocean_cuda_check_launch("ternary packing kernel");
}

void ocean_cuda_packed_linear(const void *input, const void *packed, const void *bias, void *output, int rows, int cols_a, int cols_b, int packed_cols, float scale) {
    size_t size = (size_t)rows * (size_t)cols_b;
    if (size == 0) return;
    ocean_cuda_packed_linear_kernel<<<
        ocean_cuda_packed_blocks(rows, cols_b), 128
    >>>(
        (const float *)input, (const int32_t *)packed, (const float *)bias,
        (float *)output, rows, cols_a, cols_b, packed_cols, scale
    );
    ocean_cuda_check_launch("packed linear kernel");
}

void ocean_cuda_packed_qkv(const void *input, const void *q_packed, const void *q_bias, const void *k_packed, const void *k_bias, const void *v_packed, const void *v_bias, void *output, int rows, int cols_a, int cols_b, int packed_cols, float q_scale, float k_scale, float v_scale) {
    size_t size = (size_t)rows * (size_t)cols_b;
    if (size == 0) return;
    ocean_cuda_packed_qkv_kernel<<<
        ocean_cuda_packed_blocks(rows, cols_b), 128
    >>>(
        (const float *)input, (const int32_t *)q_packed, (const float *)q_bias,
        (const int32_t *)k_packed, (const float *)k_bias,
        (const int32_t *)v_packed, (const float *)v_bias, (float *)output,
        rows, cols_a, cols_b, packed_cols, q_scale, k_scale, v_scale
    );
    ocean_cuda_check_launch("packed QKV kernel");
}

void ocean_cuda_packed_qkv_split(const void *input, const void *q_packed, const void *q_bias, const void *k_packed, const void *k_bias, const void *v_packed, const void *v_bias, void *q_output, void *k_output, void *v_output, int rows, int cols_a, int cols_b, int packed_cols, float q_scale, float k_scale, float v_scale) {
    size_t size = (size_t)rows * (size_t)cols_b;
    if (size == 0) return;
    ocean_cuda_packed_qkv_split_kernel<<<
        ocean_cuda_packed_blocks(rows, cols_b), 128
    >>>(
        (const float *)input, (const int32_t *)q_packed, (const float *)q_bias,
        (const int32_t *)k_packed, (const float *)k_bias,
        (const int32_t *)v_packed, (const float *)v_bias,
        (float *)q_output, (float *)k_output, (float *)v_output,
        rows, cols_a, cols_b, packed_cols, q_scale, k_scale, v_scale
    );
    ocean_cuda_check_launch("split packed QKV kernel");
}

void ocean_cuda_packed_qkv_attention_decode(const void *input, const void *q_packed, const void *q_bias, const void *k_packed, const void *k_bias, const void *v_packed, const void *v_bias, void *cache_k, void *cache_v, void *output, int cols_a, int packed_cols, int max_seq, int position, int n_heads, int head_dim, float q_scale, float k_scale, float v_scale) {
    if (head_dim <= 0 || head_dim > 128 || max_seq <= 0 || position < 0 || position >= max_seq) {
        ocean_tensor_fail("invalid CUDA packed attention dimensions");
    }
    size_t shared_bytes = (size_t)(3 * head_dim + max_seq + 128) * sizeof(float);
    ocean_cuda_packed_attention_kernel<<<n_heads, 128, shared_bytes>>>(
        (const float *)input, (const int32_t *)q_packed, (const float *)q_bias,
        (const int32_t *)k_packed, (const float *)k_bias,
        (const int32_t *)v_packed, (const float *)v_bias,
        (float *)cache_k, (float *)cache_v, (float *)output,
        cols_a, packed_cols, max_seq, position, n_heads, head_dim,
        q_scale, k_scale, v_scale
    );
    ocean_cuda_check_launch("packed QKV attention decode kernel");
}

void ocean_cuda_cache_write(void *cache, const void *value, int batches, int heads, int sequence, int value_sequence, int width, int position) {
    size_t size = (size_t)batches * (size_t)heads * (size_t)value_sequence * (size_t)width;
    if (size == 0) return;
    ocean_cuda_cache_write_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (float *)cache, (const float *)value, batches, heads, sequence,
        value_sequence, width, position
    );
    ocean_cuda_check_launch("cache write kernel");
}

void ocean_cuda_permute_swap12_f32(
    const void *input,
    void *output,
    int batches,
    int first_dim,
    int second_dim,
    int head_dim
) {
    size_t total = (size_t)batches * (size_t)first_dim *
        (size_t)second_dim * (size_t)head_dim;
    if (total == 0) return;
    ocean_cuda_permute_swap12_f32_kernel<<<ocean_cuda_blocks(total), 256>>>(
        (const float *)input, (float *)output,
        batches, first_dim, second_dim, head_dim
    );
    ocean_cuda_check_launch("permute swap12 kernel");
}

void ocean_cuda_cache_slice(const void *cache, void *output, int batches, int heads, int source_sequence, int output_sequence, int width, int start) {
    size_t size = (size_t)batches * (size_t)heads * (size_t)output_sequence * (size_t)width;
    if (size == 0) return;
    ocean_cuda_cache_slice_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)cache, (float *)output, batches, heads, source_sequence,
        output_sequence, width, start
    );
    ocean_cuda_check_launch("cache slice kernel");
}

void ocean_cuda_sparse_build_summaries(
    const void *key,
    void *summaries,
    int batches,
    int heads,
    int key_length,
    int active_length,
    int head_dim,
    int block_size
) {
    int summary_blocks = (key_length + block_size - 1) / block_size;
    size_t total = (size_t)batches * (size_t)heads *
        (size_t)summary_blocks * (size_t)head_dim;
    if (total == 0) return;
    ocean_cuda_sparse_build_summaries_kernel<<<ocean_cuda_blocks(total), 256>>>(
        (const float *)key, (float *)summaries,
        batches, heads, key_length, active_length, head_dim,
        block_size, summary_blocks
    );
    ocean_cuda_check_launch("sparse summary build kernel");
}

void ocean_cuda_sparse_update_summary(
    const void *key,
    void *summaries,
    int batches,
    int heads,
    int key_length,
    int summary_blocks,
    int active_length,
    int head_dim,
    int block_size,
    int position
) {
    if (batches <= 0 || heads <= 0 || head_dim <= 0) return;
    ocean_cuda_sparse_update_summary_kernel<<<batches * heads, 128>>>(
        (const float *)key, (float *)summaries,
        batches, heads, key_length, summary_blocks, active_length,
        head_dim, block_size, position
    );
    ocean_cuda_check_launch("sparse summary update kernel");
}

void ocean_cuda_sparse_attention(
    const void *query,
    const void *key,
    const void *value,
    const void *summaries,
    void *output,
    int batches,
    int heads,
    int query_length,
    int key_length,
    int active_length,
    int head_dim,
    int summary_blocks,
    int top_k,
    int top_blocks,
    int block_size,
    float scale,
    int query_start,
    int causal
) {
    size_t total = (size_t)batches * (size_t)heads * (size_t)query_length;
    if (total == 0) return;
    size_t shared_bytes = (size_t)head_dim * sizeof(float) +
        (size_t)top_blocks * (sizeof(int) + sizeof(float)) +
        (size_t)top_k * (sizeof(int) + sizeof(float)) + sizeof(int);
    ocean_cuda_sparse_attention_kernel<<<ocean_cuda_blocks(total), 128, shared_bytes>>>(
        (const float *)query, (const float *)key, (const float *)value,
        (const float *)summaries, (float *)output,
        batches, heads, query_length, key_length, active_length,
        head_dim, summary_blocks,
        top_k, top_blocks, block_size, scale, query_start, causal
    );
    ocean_cuda_check_launch("sparse attention kernel");
}

void ocean_cuda_embedding_forward(const void *weight, const void *indices, void *output, int index_count, int vocab, int dim) {
    size_t size = (size_t)index_count * (size_t)dim;
    if (size == 0) return;
    ocean_cuda_embedding_kernel<<<ocean_cuda_blocks(size), 256>>>(
        (const float *)weight, (const int64_t *)indices, (float *)output,
        index_count, vocab, dim
    );
    ocean_cuda_check_launch("embedding kernel");
}

int ocean_cuda_argmax_f32(const void *input, size_t size) {
    if (size == 0 || size > (size_t)INT_MAX) ocean_tensor_fail("CUDA argmax size is invalid");
    int *device_result = (int *)ocean_cuda_malloc(sizeof(int));
    ocean_cuda_argmax_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int))>>>(
        (const float *)input, device_result, size
    );
    ocean_cuda_check_launch("argmax kernel");
    int result = -1;
    ocean_cuda_memcpy_d2h(&result, device_result, sizeof(result));
    ocean_cuda_free(device_result);
    if (result < 0) ocean_tensor_fail("CUDA argmax failed");
    return result;
}
