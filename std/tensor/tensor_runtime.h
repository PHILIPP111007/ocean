#ifndef OCEAN_STD_TENSOR_RUNTIME_H
#define OCEAN_STD_TENSOR_RUNTIME_H

#include <stddef.h>

/*
 * Opaque storage handle used by the safe Ocean Tensor facade.  The concrete
 * representation is private to tensor_runtime.c: CPU storage owns float32
 * memory, while GPU storage owns an OpenCL buffer when that backend is built.
 */
typedef struct ocean_tensor_handle *ocean_tensor_handle_t;
typedef struct ocean_tensor_float32 ocean_tensor_float32;

ocean_tensor_handle_t ocean_tensor_zeros(int rows, int cols, const char *device);
ocean_tensor_handle_t ocean_tensor_from_cpu_strided(
    const float *data,
    const size_t *shape,
    const size_t *strides,
    size_t ndim,
    const char *device
);
ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_to(ocean_tensor_handle_t tensor, const char *device);
ocean_tensor_handle_t ocean_tensor_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis);
int ocean_tensor_ndim(ocean_tensor_handle_t tensor);
size_t ocean_tensor_size(ocean_tensor_handle_t tensor);
char *ocean_tensor_device(ocean_tensor_handle_t tensor);
ocean_tensor_float32 *ocean_tensor_to_cpu_tensor(ocean_tensor_handle_t tensor);
void ocean_tensor_release(ocean_tensor_handle_t tensor);

#endif
