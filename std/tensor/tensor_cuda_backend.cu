#include "std/tensor/tensor_cuda_backend.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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
