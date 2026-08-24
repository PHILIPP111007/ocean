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

__global__ void ocean_cuda_copy_strided_kernel(
    const unsigned char *source,
    unsigned char *destination,
    ocean_cuda_strided_copy_desc descriptor
) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= descriptor.total) return;
    size_t remaining = index;
    size_t source_offset = 0;
    size_t destination_offset = 0;
    for (int axis = descriptor.ndim - 1; axis >= 0; --axis) {
        size_t coordinate = descriptor.shape[axis] == 0
            ? 0 : remaining % descriptor.shape[axis];
        remaining = descriptor.shape[axis] == 0
            ? 0 : remaining / descriptor.shape[axis];
        source_offset += coordinate * descriptor.source_strides[axis];
        destination_offset += coordinate * descriptor.destination_strides[axis];
    }
    for (size_t byte = 0; byte < descriptor.item_size; ++byte) {
        destination[destination_offset * descriptor.item_size + byte] =
            source[source_offset * descriptor.item_size + byte];
    }
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

void ocean_cuda_copy_strided(
    const void *source,
    void *destination,
    const ocean_cuda_strided_copy_desc *descriptor
) {
    if (!descriptor || descriptor->total == 0) return;
    ocean_cuda_copy_strided_kernel<<<ocean_cuda_blocks(descriptor->total), 256>>>(
        (const unsigned char *)source, (unsigned char *)destination, *descriptor
    );
    ocean_cuda_check_launch("strided copy kernel");
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

__device__ static void ocean_cuda_sparse_tree_range(
    int node,
    int leaf_count,
    int *start,
    int *span
) {
    int first = 1;
    int current_span = leaf_count;
    while (node >= first * 2) {
        first *= 2;
        current_span /= 2;
    }
    *start = (node - first) * current_span;
    *span = current_span;
}

__global__ void ocean_cuda_sparse_hierarchy_leaves_kernel(
    const float *summaries,
    float *hierarchy,
    int batches,
    int heads,
    int summary_blocks,
    int head_dim,
    int leaf_count
) {
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batches * (size_t)heads *
        (size_t)leaf_count * (size_t)head_dim;
    if (linear >= total) return;
    int dimension = (int)(linear % (size_t)head_dim);
    size_t leaf_linear = linear / (size_t)head_dim;
    int leaf = (int)(leaf_linear % (size_t)leaf_count);
    size_t group = leaf_linear / (size_t)leaf_count;
    size_t destination = (group * (size_t)(leaf_count * 2) +
        (size_t)leaf_count + (size_t)leaf) * (size_t)head_dim +
        (size_t)dimension;
    if (leaf < summary_blocks) {
        size_t source = (group * (size_t)summary_blocks + (size_t)leaf) *
            (size_t)head_dim + (size_t)dimension;
        hierarchy[destination] = summaries[source];
    } else {
        hierarchy[destination] = 0.0f;
    }
}

__global__ void ocean_cuda_sparse_hierarchy_level_kernel(
    float *hierarchy,
    int batches,
    int heads,
    int valid_blocks,
    int head_dim,
    int leaf_count,
    int level
) {
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batches * (size_t)heads * (size_t)level *
        (size_t)head_dim;
    if (linear >= total) return;
    int dimension = (int)(linear % (size_t)head_dim);
    size_t node_linear = linear / (size_t)head_dim;
    int node = level + (int)(node_linear % (size_t)level);
    size_t group = node_linear / (size_t)level;
    int start = 0;
    int span = 0;
    ocean_cuda_sparse_tree_range(node, leaf_count, &start, &span);
    int left_start = 0;
    int left_span = 0;
    int right_start = 0;
    int right_span = 0;
    ocean_cuda_sparse_tree_range(node * 2, leaf_count, &left_start, &left_span);
    ocean_cuda_sparse_tree_range(node * 2 + 1, leaf_count, &right_start, &right_span);
    int left_count = valid_blocks - left_start;
    if (left_count < 0) left_count = 0;
    if (left_count > left_span) left_count = left_span;
    int right_count = valid_blocks - right_start;
    if (right_count < 0) right_count = 0;
    if (right_count > right_span) right_count = right_span;
    size_t tree_stride = (size_t)(leaf_count * 2) * (size_t)head_dim;
    size_t base = group * tree_stride;
    float value = 0.0f;
    int count = left_count + right_count;
    if (count > 0) {
        if (left_count > 0) {
            value += hierarchy[base + (size_t)(node * 2) * (size_t)head_dim +
                (size_t)dimension] * (float)left_count;
        }
        if (right_count > 0) {
            value += hierarchy[base + (size_t)(node * 2 + 1) * (size_t)head_dim +
                (size_t)dimension] * (float)right_count;
        }
        value /= (float)count;
    }
    hierarchy[base + (size_t)node * (size_t)head_dim + (size_t)dimension] = value;
}

__global__ void ocean_cuda_sparse_update_hierarchy_kernel(
    const float *summaries,
    float *hierarchy,
    int batches,
    int heads,
    int summary_blocks,
    int valid_blocks,
    int head_dim,
    int leaf_count,
    int block
) {
    int lane = (int)threadIdx.x;
    int group = (int)blockIdx.x;
    if (group >= batches * heads || lane >= head_dim || block < 0 ||
        block >= valid_blocks || block >= summary_blocks) return;
    size_t tree_stride = (size_t)(leaf_count * 2) * (size_t)head_dim;
    size_t group_base = (size_t)group * tree_stride;
    size_t summary_base = ((size_t)group * (size_t)summary_blocks +
        (size_t)block) * (size_t)head_dim;
    size_t leaf_base = group_base + (size_t)(leaf_count + block) *
        (size_t)head_dim;
    hierarchy[leaf_base + (size_t)lane] = summaries[summary_base + (size_t)lane];
    int node = (leaf_count + block) / 2;
    while (node > 0) {
        int left_start = 0;
        int left_span = 0;
        int right_start = 0;
        int right_span = 0;
        ocean_cuda_sparse_tree_range(node * 2, leaf_count,
            &left_start, &left_span);
        ocean_cuda_sparse_tree_range(node * 2 + 1, leaf_count,
            &right_start, &right_span);
        int left_count = valid_blocks - left_start;
        if (left_count < 0) left_count = 0;
        if (left_count > left_span) left_count = left_span;
        int right_count = valid_blocks - right_start;
        if (right_count < 0) right_count = 0;
        if (right_count > right_span) right_count = right_span;
        int count = left_count + right_count;
        float value = 0.0f;
        if (count > 0) {
            if (left_count > 0) {
                value += hierarchy[group_base + (size_t)(node * 2) *
                    (size_t)head_dim + (size_t)lane] * (float)left_count;
            }
            if (right_count > 0) {
                value += hierarchy[group_base + (size_t)(node * 2 + 1) *
                    (size_t)head_dim + (size_t)lane] * (float)right_count;
            }
            value /= (float)count;
        }
        hierarchy[group_base + (size_t)node * (size_t)head_dim +
            (size_t)lane] = value;
        node /= 2;
    }
}

__device__ static bool ocean_cuda_sparse_tree_precedes(
    float score, int index, float other_score, int other_index
) {
    return ocean_cuda_sparse_precedes(score, index, other_score, other_index);
}

__global__ void ocean_cuda_sparse_hierarchical_route_kernel(
    const float *key,
    const float *hierarchy,
    int32_t *route,
    int batches,
    int heads,
    int key_length,
    int tree_nodes,
    int leaf_count,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed,
    int beam_width
) {
    extern __shared__ unsigned char storage[];
    float *recent = (float *)storage;
    float *candidate_scores = recent + head_dim;
    int *candidate_nodes = (int *)(candidate_scores + beam_width * 2);
    float *selected_scores = (float *)(candidate_nodes + beam_width * 2);
    int *selected_blocks = (int *)(selected_scores + semantic_blocks);
    size_t group = (size_t)blockIdx.x;
    if (group >= (size_t)batches * (size_t)heads || threadIdx.x != 0) return;

    size_t route_width = (size_t)local_blocks + (size_t)semantic_blocks + 1u;
    size_t route_base = group * route_width;
    for (size_t index = 0; index < route_width; ++index) {
        route[route_base + index] = -1;
    }
    int end = active_length;
    if (end > key_length) end = key_length;
    int start = end - summary_window;
    if (start < 0) start = 0;
    int count = end - start;
    if (count <= 0) return;
    int batch = (int)(group / (size_t)heads);
    int head = (int)(group % (size_t)heads);
    const float *key_group = key +
        (((size_t)batch * (size_t)heads + (size_t)head) *
            (size_t)key_length * (size_t)head_dim);
    float query_norm = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
        float sum = 0.0f;
        for (int token = start; token < end; ++token) {
            sum += key_group[(size_t)token * (size_t)head_dim +
                (size_t)dimension];
        }
        recent[dimension] = sum / (float)count;
        query_norm += recent[dimension] * recent[dimension];
    }
    query_norm = sqrtf(query_norm);
    int block_count = (active_length + block_size - 1) / block_size;
    if (block_count > tree_nodes / 2) block_count = tree_nodes / 2;
    int local_count = local_blocks;
    if (local_count > block_count) local_count = block_count;
    int local_start = block_count - local_count;
    for (int index = 0; index < local_count; ++index) {
        route[route_base + (size_t)index] = local_start + index;
    }

    int depth = 0;
    for (int span = leaf_count; span > 1; span /= 2) ++depth;
    int candidate_count = 1;
    candidate_nodes[0] = 1;
    candidate_scores[0] = 0.0f;
    size_t tree_stride = (size_t)tree_nodes * (size_t)head_dim;
    for (int level = 0; level < depth; ++level) {
        int next_count = 0;
        for (int candidate = 0; candidate < candidate_count; ++candidate) {
            int parent = candidate_nodes[candidate];
            for (int branch = 0; branch < 2; ++branch) {
                int node = parent * 2 + branch;
                if (node <= 0 || node >= leaf_count * 2) continue;
                int node_start = 0;
                int node_span = 0;
                ocean_cuda_sparse_tree_range(node, leaf_count,
                    &node_start, &node_span);
                if (node_start >= block_count) continue;
                size_t summary_base = group * tree_stride +
                    (size_t)node * (size_t)head_dim;
                float dot = 0.0f;
                float node_norm = 0.0f;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    float value = hierarchy[summary_base + (size_t)dimension];
                    dot += recent[dimension] * value;
                    node_norm += value * value;
                }
                float score = dot / (query_norm * sqrtf(node_norm) + 1.0e-8f);
                int insert = next_count;
                if (insert > beam_width) insert = beam_width;
                if (next_count < beam_width) {
                    while (insert > 0 && ocean_cuda_sparse_tree_precedes(
                        score, node, candidate_scores[beam_width + insert - 1],
                        candidate_nodes[beam_width + insert - 1])) {
                        candidate_scores[beam_width + insert] =
                            candidate_scores[beam_width + insert - 1];
                        candidate_nodes[beam_width + insert] =
                            candidate_nodes[beam_width + insert - 1];
                        --insert;
                    }
                    candidate_scores[beam_width + insert] = score;
                    candidate_nodes[beam_width + insert] = node;
                    ++next_count;
                } else if (ocean_cuda_sparse_tree_precedes(
                    score, node, candidate_scores[beam_width + beam_width - 1],
                    candidate_nodes[beam_width + beam_width - 1])) {
                    insert = beam_width - 1;
                    while (insert > 0 && ocean_cuda_sparse_tree_precedes(
                        score, node, candidate_scores[beam_width + insert - 1],
                        candidate_nodes[beam_width + insert - 1])) {
                        candidate_scores[beam_width + insert] =
                            candidate_scores[beam_width + insert - 1];
                        candidate_nodes[beam_width + insert] =
                            candidate_nodes[beam_width + insert - 1];
                        --insert;
                    }
                    candidate_scores[beam_width + insert] = score;
                    candidate_nodes[beam_width + insert] = node;
                }
            }
        }
        candidate_count = next_count;
        for (int candidate = 0; candidate < candidate_count; ++candidate) {
            candidate_nodes[candidate] = candidate_nodes[beam_width + candidate];
            candidate_scores[candidate] = candidate_scores[beam_width + candidate];
        }
        if (candidate_count == 0) break;
    }
    int selected_count = 0;
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        int block = candidate_nodes[candidate] - leaf_count;
        if (block < 0 || block >= block_count || block >= local_start) continue;
        float score = candidate_scores[candidate];
        if (selected_count < semantic_blocks) {
            int insert = selected_count;
            while (insert > 0 && ocean_cuda_sparse_precedes(
                score, block, selected_scores[insert - 1], selected_blocks[insert - 1])) {
                selected_scores[insert] = selected_scores[insert - 1];
                selected_blocks[insert] = selected_blocks[insert - 1];
                --insert;
            }
            selected_scores[insert] = score;
            selected_blocks[insert] = block;
            ++selected_count;
        } else if (ocean_cuda_sparse_precedes(
            score, block, selected_scores[semantic_blocks - 1],
            selected_blocks[semantic_blocks - 1])) {
            int insert = semantic_blocks - 1;
            while (insert > 0 && ocean_cuda_sparse_precedes(
                score, block, selected_scores[insert - 1], selected_blocks[insert - 1])) {
                selected_scores[insert] = selected_scores[insert - 1];
                selected_blocks[insert] = selected_blocks[insert - 1];
                --insert;
            }
            selected_scores[insert] = score;
            selected_blocks[insert] = block;
        }
    }
    for (int index = 0; index < selected_count; ++index) {
        route[route_base + (size_t)local_blocks + (size_t)index] =
            selected_blocks[index];
    }
    if (block_count > 0) {
        unsigned int state = random_seed ^
            ((unsigned int)group * 747796405u + 2891336453u);
        state = state * 1664525u + 1013904223u;
        int random_block = (int)(state % (unsigned int)block_count);
        bool duplicate = false;
        for (int attempt = 0; attempt < block_count; ++attempt) {
            duplicate = false;
            for (int index = 0; index < local_blocks + selected_count; ++index) {
                if (route[route_base + (size_t)index] == random_block) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) break;
            state = state * 1664525u + 1013904223u;
            random_block = (int)(state % (unsigned int)block_count);
        }
        if (duplicate) random_block = -1;
        route[route_base + (size_t)local_blocks + (size_t)semantic_blocks] =
            random_block;
    }
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

/*
 * Chunked semantic routing.  One route is built for a group of query tokens
 * and then reused by the routed attention kernel.  The selector deliberately
 * stays simple and deterministic: it compares the mean of the recent key
 * window with the mean key summary of every visible block, keeps the best
 * semantic blocks, and appends one exploration block.
 */
__global__ void ocean_cuda_sparse_route_kernel(
    const float *key,
    const float *summaries,
    int32_t *route,
    int batches,
    int heads,
    int key_length,
    int summary_blocks,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed
) {
    extern __shared__ unsigned char storage[];
    float *recent = (float *)storage;
    float *selected_scores = recent + head_dim;
    int *selected_blocks = (int *)(selected_scores + semantic_blocks);
    size_t group = (size_t)blockIdx.x;
    size_t total = (size_t)batches * (size_t)heads;
    if (group >= total) return;
    int lane = (int)threadIdx.x;
    size_t route_width = (size_t)local_blocks +
        (size_t)semantic_blocks + 1u;
    size_t route_base = group * route_width;
    if (lane == 0) {
        for (size_t index = 0; index < route_width; ++index) {
            route[route_base + (size_t)index] = -1;
        }
        int end = active_length;
        if (end > key_length) end = key_length;
        int start = end - summary_window;
        if (start < 0) start = 0;
        int count = end - start;
        if (count <= 0) return;
        int head = (int)(group % (size_t)heads);
        int batch = (int)(group / (size_t)heads);
        const float *key_group = key +
            (((size_t)batch * (size_t)heads + (size_t)head) *
                (size_t)key_length * (size_t)head_dim);
        float query_norm = 0.0f;
        for (int dimension = 0; dimension < head_dim; ++dimension) {
            float sum = 0.0f;
            for (int token = start; token < end; ++token) {
                sum += key_group[(size_t)token * (size_t)head_dim +
                    (size_t)dimension];
            }
            recent[dimension] = sum / (float)count;
            query_norm += recent[dimension] * recent[dimension];
        }
        query_norm = sqrtf(query_norm);
        int block_count = (active_length + block_size - 1) / block_size;
        if (block_count > summary_blocks) block_count = summary_blocks;
        int local_count = local_blocks;
        if (local_count > block_count) local_count = block_count;
        int local_start = block_count - local_count;
        for (int index = 0; index < local_count; ++index) {
            route[route_base + (size_t)index] = local_start + index;
        }
        int selected_count = 0;
        for (int block = 0; block < block_count; ++block) {
            if (block >= local_start) continue;
            size_t summary_base =
                (group * (size_t)summary_blocks + (size_t)block) *
                (size_t)head_dim;
            float dot = 0.0f;
            float block_norm = 0.0f;
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                float value = summaries[summary_base + (size_t)dimension];
                dot += recent[dimension] * value;
                block_norm += value * value;
            }
            float denominator = query_norm * sqrtf(block_norm) + 1.0e-8f;
            float score = dot / denominator;
            if (selected_count < semantic_blocks) {
                int insert = selected_count;
                while (insert > 0 && ocean_cuda_sparse_precedes(
                    score, block, selected_scores[insert - 1],
                    selected_blocks[insert - 1]
                )) {
                    selected_scores[insert] = selected_scores[insert - 1];
                    selected_blocks[insert] = selected_blocks[insert - 1];
                    --insert;
                }
                selected_scores[insert] = score;
                selected_blocks[insert] = block;
                ++selected_count;
            } else if (ocean_cuda_sparse_precedes(
                score, block, selected_scores[semantic_blocks - 1],
                selected_blocks[semantic_blocks - 1]
            )) {
                int insert = semantic_blocks - 1;
                while (insert > 0 && ocean_cuda_sparse_precedes(
                    score, block, selected_scores[insert - 1],
                    selected_blocks[insert - 1]
                )) {
                    selected_scores[insert] = selected_scores[insert - 1];
                    selected_blocks[insert] = selected_blocks[insert - 1];
                    --insert;
                }
                selected_scores[insert] = score;
                selected_blocks[insert] = block;
            }
        }
        for (int index = 0; index < selected_count; ++index) {
            route[route_base + (size_t)local_blocks + (size_t)index] =
                selected_blocks[index];
        }
        if (block_count > 0) {
            unsigned int state = random_seed ^
                ((unsigned int)group * 747796405u + 2891336453u);
            state = state * 1664525u + 1013904223u;
            int random_block = (int)(state % (unsigned int)block_count);
            bool duplicate = false;
            for (int attempt = 0; attempt < block_count; ++attempt) {
                duplicate = false;
                for (int index = 0; index < local_blocks + selected_count;
                     ++index) {
                    if (route[route_base + (size_t)index] == random_block) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) break;
                state = state * 1664525u + 1013904223u;
                random_block = (int)(state % (unsigned int)block_count);
            }
            if (duplicate) random_block = -1;
            route[route_base + (size_t)local_blocks +
                (size_t)semantic_blocks] = random_block;
        }
    }
}

__global__ void ocean_cuda_sparse_routed_attention_kernel(
    const float *query,
    const float *key,
    const float *value,
    const int32_t *route,
    float *output,
    int batches,
    int heads,
    int query_length,
    int key_length,
    int active_length,
    int head_dim,
    int route_blocks,
    int block_size,
    float scale,
    int query_start,
    int causal
) {
    extern __shared__ unsigned char storage[];
    float *q_vector = (float *)storage;
    int max_tokens = route_blocks * block_size;
    float *scores = q_vector + head_dim;
    int *selected_count = (int *)(scores + max_tokens);
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
        int chosen = 0;
        size_t route_base = ((size_t)batch * (size_t)heads + (size_t)head) *
            (size_t)route_blocks;
        for (int route_index = 0; route_index < route_blocks; ++route_index) {
            int block = route[route_base + (size_t)route_index];
            if (block < 0) continue;
            int start = block * block_size;
            int end = start + block_size;
            if (end > active_length) end = active_length;
            if (end > key_length) end = key_length;
            if (causal && end > absolute_query + 1) end = absolute_query + 1;
            if (start < 0 || start >= end) continue;
            size_t head_base = ((size_t)batch * (size_t)heads +
                (size_t)head) * (size_t)key_length * (size_t)head_dim;
            for (int token = start; token < end && chosen < max_tokens; ++token) {
                size_t key_base = head_base + (size_t)token * (size_t)head_dim;
                float score = 0.0f;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    score += q_vector[dimension] *
                        key[key_base + (size_t)dimension];
                }
                scores[chosen] = score * scale;
                ++chosen;
            }
        }
        float maximum = -INFINITY;
        for (int index = 0; index < chosen; ++index) {
            if (scores[index] > maximum) maximum = scores[index];
        }
        float denominator = 0.0f;
        for (int index = 0; index < chosen; ++index) {
            scores[index] = expf(scores[index] - maximum);
            denominator += scores[index];
        }
        if (denominator > 0.0f) {
            for (int index = 0; index < chosen; ++index) {
                scores[index] /= denominator;
            }
        }
        *selected_count = chosen;
    }
    __syncthreads();
    size_t output_base = ((size_t)batch * (size_t)heads + (size_t)head) *
        (size_t)query_length * (size_t)head_dim +
        (size_t)query_index * (size_t)head_dim;
    if (lane < head_dim) {
        float context = 0.0f;
        int selected = 0;
        size_t route_base = ((size_t)batch * (size_t)heads +
            (size_t)head) * (size_t)route_blocks;
        for (int route_index = 0; route_index < route_blocks; ++route_index) {
            int block = route[route_base + (size_t)route_index];
            if (block < 0) continue;
            int start = block * block_size;
            int end = start + block_size;
            if (end > active_length) end = active_length;
            if (end > key_length) end = key_length;
            if (causal && end > absolute_query + 1) end = absolute_query + 1;
            if (start < 0 || start >= end) continue;
            size_t head_base = ((size_t)batch * (size_t)heads +
                (size_t)head) * (size_t)key_length * (size_t)head_dim;
            for (int token = start; token < end; ++token) {
                size_t value_base = head_base +
                    (size_t)token * (size_t)head_dim;
                if (selected < *selected_count) {
                    context += scores[selected] * value[value_base +
                        (size_t)lane];
                }
                ++selected;
            }
        }
        output[output_base + (size_t)lane] = context;
    }
}

__global__ void ocean_cuda_sparse_build_paged_summary_kernel(
    const float * const *key_pages,
    float *summaries,
    int batches,
    int heads,
    int page_count,
    int page_size,
    int active_length,
    int head_dim,
    int page_index
) {
    size_t group = (size_t)blockIdx.x;
    size_t total = (size_t)batches * (size_t)heads;
    int lane = (int)threadIdx.x;
    if (group >= total || page_index < 0 || page_index >= page_count ||
        lane >= head_dim) return;
    int start = page_index * page_size;
    int end = start + page_size;
    if (end > active_length) end = active_length;
    int count = end - start;
    size_t summary_base = (group * (size_t)page_count +
        (size_t)page_index) * (size_t)head_dim;
    if (count <= 0) {
        summaries[summary_base + (size_t)lane] = 0.0f;
        return;
    }
    const float *page = key_pages[page_index];
    size_t page_base = group * (size_t)page_size * (size_t)head_dim;
    float sum = 0.0f;
    for (int token = start; token < end; ++token) {
        int local = token - start;
        sum += page[page_base + (size_t)local * (size_t)head_dim +
            (size_t)lane];
    }
    summaries[summary_base + (size_t)lane] = sum / (float)count;
}

__global__ void ocean_cuda_sparse_routed_attention_paged_kernel(
    const float *query,
    const float * const *key_pages,
    const float * const *value_pages,
    const int32_t *route,
    float *output,
    int batches,
    int heads,
    int query_length,
    int active_length,
    int head_dim,
    int route_blocks,
    int page_size,
    float scale,
    int query_start,
    int causal
) {
    extern __shared__ unsigned char storage[];
    float *q_vector = (float *)storage;
    int max_tokens = route_blocks * page_size;
    float *scores = q_vector + head_dim;
    float *context = scores + max_tokens;
    int *selected_count = (int *)(context + head_dim);
    size_t group = (size_t)blockIdx.x;
    size_t total = (size_t)batches * (size_t)heads *
        (size_t)query_length;
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
        int chosen = 0;
        size_t route_base = ((size_t)batch * (size_t)heads + (size_t)head) *
            (size_t)route_blocks;
        for (int route_index = 0; route_index < route_blocks; ++route_index) {
            int page = route[route_base + (size_t)route_index];
            if (page < 0) continue;
            int start = page * page_size;
            int end = start + page_size;
            if (end > active_length) end = active_length;
            if (causal && end > absolute_query + 1) end = absolute_query + 1;
            if (start < 0 || start >= end) continue;
            const float *key_page = key_pages[page];
            for (int token = start; token < end && chosen < max_tokens; ++token) {
                int local = token - start;
                size_t key_base = ((size_t)batch * (size_t)heads +
                    (size_t)head) * (size_t)page_size * (size_t)head_dim +
                    (size_t)local * (size_t)head_dim;
                float score = 0.0f;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    score += q_vector[dimension] *
                        key_page[key_base + (size_t)dimension];
                }
                scores[chosen++] = score * scale;
            }
        }
        float maximum = -INFINITY;
        for (int index = 0; index < chosen; ++index) {
            if (scores[index] > maximum) maximum = scores[index];
        }
        float denominator = 0.0f;
        for (int index = 0; index < chosen; ++index) {
            scores[index] = expf(scores[index] - maximum);
            denominator += scores[index];
        }
        if (denominator > 0.0f) {
            for (int index = 0; index < chosen; ++index) {
                scores[index] /= denominator;
            }
        }
        *selected_count = chosen;
    }
    __syncthreads();
    if (lane < head_dim) context[lane] = 0.0f;
    __syncthreads();
    if (lane < head_dim) {
        int selected = 0;
        size_t route_base = ((size_t)batch * (size_t)heads + (size_t)head) *
            (size_t)route_blocks;
        for (int route_index = 0; route_index < route_blocks; ++route_index) {
            int page = route[route_base + (size_t)route_index];
            if (page < 0) continue;
            int start = page * page_size;
            int end = start + page_size;
            if (end > active_length) end = active_length;
            if (causal && end > absolute_query + 1) end = absolute_query + 1;
            if (start < 0 || start >= end) continue;
            const float *value_page = value_pages[page];
            for (int token = start; token < end && selected < *selected_count; ++token) {
                int local = token - start;
                size_t value_base = ((size_t)batch * (size_t)heads +
                    (size_t)head) * (size_t)page_size * (size_t)head_dim +
                    (size_t)local * (size_t)head_dim;
                context[lane] += scores[selected] *
                    value_page[value_base + (size_t)lane];
                ++selected;
            }
        }
    }
    if (lane < head_dim) {
        output[query_base + (size_t)lane] = context[lane];
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

void ocean_cuda_sparse_build_route(
    const void *key,
    const void *summaries,
    void *route,
    int batches,
    int heads,
    int key_length,
    int summary_blocks,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed
) {
    if (batches <= 0 || heads <= 0 || head_dim <= 0 ||
        semantic_blocks <= 0 || local_blocks < 0 || block_size <= 0) return;
    size_t shared_bytes = (size_t)head_dim * sizeof(float) +
        (size_t)semantic_blocks * (sizeof(float) + sizeof(int));
    ocean_cuda_sparse_route_kernel<<<batches * heads, 128, shared_bytes>>>(
        (const float *)key, (const float *)summaries, (int32_t *)route,
        batches, heads, key_length, summary_blocks, active_length, head_dim,
        summary_window, semantic_blocks, local_blocks, block_size, random_seed
    );
    ocean_cuda_check_launch("sparse route kernel");
}

void ocean_cuda_sparse_build_hierarchy(
    const void *summaries,
    void *hierarchy,
    int batches,
    int heads,
    int summary_blocks,
    int valid_blocks,
    int head_dim,
    int leaf_count
) {
    if (batches <= 0 || heads <= 0 || summary_blocks <= 0 ||
        head_dim <= 0 || leaf_count <= 0) return;
    size_t leaves_total = (size_t)batches * (size_t)heads *
        (size_t)leaf_count * (size_t)head_dim;
    ocean_cuda_sparse_hierarchy_leaves_kernel<<<
        ocean_cuda_blocks(leaves_total), 256
    >>>(
        (const float *)summaries, (float *)hierarchy, batches, heads,
        summary_blocks, head_dim, leaf_count
    );
    ocean_cuda_check_launch("sparse hierarchy leaves kernel");
    for (int level = leaf_count / 2; level >= 1; level /= 2) {
        size_t level_total = (size_t)batches * (size_t)heads *
            (size_t)level * (size_t)head_dim;
        ocean_cuda_sparse_hierarchy_level_kernel<<<
            ocean_cuda_blocks(level_total), 256
        >>>(
            (float *)hierarchy, batches, heads, valid_blocks, head_dim,
            leaf_count, level
        );
        ocean_cuda_check_launch("sparse hierarchy level kernel");
    }
}

void ocean_cuda_sparse_update_hierarchy(
    const void *summaries,
    void *hierarchy,
    int batches,
    int heads,
    int summary_blocks,
    int valid_blocks,
    int head_dim,
    int leaf_count,
    int block
) {
    if (batches <= 0 || heads <= 0 || summary_blocks <= 0 ||
        valid_blocks <= 0 || head_dim <= 0 || leaf_count <= 0 || block < 0) return;
    ocean_cuda_sparse_update_hierarchy_kernel<<<batches * heads, 128>>>(
        (const float *)summaries, (float *)hierarchy, batches, heads,
        summary_blocks, valid_blocks, head_dim, leaf_count, block
    );
    ocean_cuda_check_launch("sparse hierarchy update kernel");
}

void ocean_cuda_sparse_build_hierarchical_route(
    const void *key,
    const void *hierarchy,
    void *route,
    int batches,
    int heads,
    int key_length,
    int tree_nodes,
    int leaf_count,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed
) {
    if (batches <= 0 || heads <= 0 || key_length <= 0 || tree_nodes <= 0 ||
        leaf_count <= 0 || active_length <= 0 || head_dim <= 0 ||
        summary_window <= 0 || semantic_blocks <= 0 || local_blocks < 0 ||
        block_size <= 0) return;
    int beam_width = semantic_blocks * 4;
    if (beam_width < 8) beam_width = 8;
    if (beam_width > 32) beam_width = 32;
    size_t shared_bytes = (size_t)head_dim * sizeof(float) +
        (size_t)beam_width * 2u * (sizeof(float) + sizeof(int)) +
        (size_t)semantic_blocks * (sizeof(float) + sizeof(int));
    ocean_cuda_sparse_hierarchical_route_kernel<<<
        batches * heads, 128, shared_bytes
    >>>(
        (const float *)key, (const float *)hierarchy, (int32_t *)route,
        batches, heads, key_length, tree_nodes, leaf_count, active_length,
        head_dim, summary_window, semantic_blocks, local_blocks, block_size,
        random_seed, beam_width
    );
    ocean_cuda_check_launch("hierarchical sparse route kernel");
}

void ocean_cuda_sparse_attention_routed(
    const void *query,
    const void *key,
    const void *value,
    const void *route,
    void *output,
    int batches,
    int heads,
    int query_length,
    int key_length,
    int active_length,
    int head_dim,
    int route_blocks,
    int block_size,
    float scale,
    int query_start,
    int causal
) {
    size_t total = (size_t)batches * (size_t)heads *
        (size_t)query_length;
    if (total == 0) return;
    size_t max_tokens = (size_t)route_blocks * (size_t)block_size;
    size_t shared_bytes = (size_t)head_dim * sizeof(float) +
        max_tokens * sizeof(float) + sizeof(int);
    ocean_cuda_sparse_routed_attention_kernel<<<(int)total, 128, shared_bytes>>>(
        (const float *)query, (const float *)key, (const float *)value,
        (const int32_t *)route, (float *)output,
        batches, heads, query_length, key_length, active_length, head_dim,
        route_blocks, block_size, scale, query_start, causal
    );
    ocean_cuda_check_launch("routed sparse attention kernel");
}

void *ocean_cuda_page_table_create(int capacity) {
    if (capacity <= 0) return NULL;
    void *table = NULL;
    ocean_cuda_check(cudaMalloc(&table, (size_t)capacity * sizeof(void *)),
        "cudaMalloc page table");
    ocean_cuda_check(cudaMemset(table, 0, (size_t)capacity * sizeof(void *)),
        "cudaMemset page table");
    return table;
}

void ocean_cuda_page_table_update(
    void *table,
    int index,
    const void *page
) {
    if (!table || index < 0) return;
    ocean_cuda_check(
        cudaMemcpy(
            (unsigned char *)table + (size_t)index * sizeof(void *),
            &page, sizeof(void *), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy page table entry"
    );
}

void ocean_cuda_page_table_release(void *table) {
    if (table != NULL) ocean_cuda_check(cudaFree(table), "cudaFree page table");
}

void ocean_cuda_sparse_build_paged_summary(
    const void *key_pages,
    void *summaries,
    int batches,
    int heads,
    int page_count,
    int page_size,
    int active_length,
    int head_dim,
    int page_index
) {
    size_t total = (size_t)batches * (size_t)heads;
    if (total == 0) return;
    ocean_cuda_sparse_build_paged_summary_kernel<<<
        (int)total, 128
    >>>(
        (const float * const *)key_pages, (float *)summaries,
        batches, heads, page_count, page_size, active_length, head_dim,
        page_index
    );
    ocean_cuda_check_launch("paged sparse summary kernel");
}

void ocean_cuda_sparse_attention_routed_paged(
    const void *query,
    const void *key_pages,
    const void *value_pages,
    const void *route,
    void *output,
    int batches,
    int heads,
    int query_length,
    int active_length,
    int head_dim,
    int route_blocks,
    int page_size,
    float scale,
    int query_start,
    int causal
) {
    size_t total = (size_t)batches * (size_t)heads * (size_t)query_length;
    if (total == 0) return;
    size_t shared_bytes = (size_t)head_dim * sizeof(float) +
        (size_t)route_blocks * (size_t)page_size * sizeof(float) +
        (size_t)head_dim * sizeof(float) + sizeof(int);
    ocean_cuda_sparse_routed_attention_paged_kernel<<<
        (int)total, 128, shared_bytes
    >>>(
        (const float *)query,
        (const float * const *)key_pages,
        (const float * const *)value_pages,
        (const int32_t *)route, (float *)output,
        batches, heads, query_length, active_length, head_dim, route_blocks,
        page_size, scale, query_start, causal
    );
    ocean_cuda_check_launch("paged routed sparse attention kernel");
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
