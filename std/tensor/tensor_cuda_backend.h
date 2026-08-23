#ifndef OCEAN_STD_TENSOR_CUDA_BACKEND_H
#define OCEAN_STD_TENSOR_CUDA_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *ocean_cuda_malloc(size_t bytes);
void ocean_cuda_free(void *device_data);
void ocean_cuda_memcpy_h2d(void *device_data, const void *host_data, size_t bytes);
void ocean_cuda_memcpy_d2h(void *host_data, const void *device_data, size_t bytes);
void ocean_cuda_memcpy_d2d(void *destination, const void *source, size_t bytes);
void ocean_cuda_zero(void *device_data, size_t bytes);
void ocean_cuda_fill_f32(void *device_data, float value, size_t size);
void ocean_cuda_fill_i32(void *device_data, int value, size_t size);

void ocean_cuda_binary_f32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation
);
void ocean_cuda_binary_i32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation
);
void ocean_cuda_scalar_f32(
    const void *input,
    void *output,
    size_t size,
    float scalar,
    int operation
);
void ocean_cuda_scalar_i32(
    const void *input,
    void *output,
    size_t size,
    int scalar,
    int operation
);
void ocean_cuda_matmul_f32(
    const void *left,
    const void *right,
    void *output,
    int rows,
    int inner,
    int columns
);
void ocean_cuda_matmul_i32(
    const void *left,
    const void *right,
    void *output,
    int rows,
    int inner,
    int columns
);

#ifdef __cplusplus
}
#endif

#endif
