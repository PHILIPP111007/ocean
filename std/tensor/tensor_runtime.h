#ifndef OCEAN_STD_TENSOR_RUNTIME_H
#define OCEAN_STD_TENSOR_RUNTIME_H

#include <stddef.h>

typedef struct ocean_tensor_handle *ocean_tensor_handle_t;

ocean_tensor_handle_t ocean_tensor_zeros(int rows, int cols, const char *device);
ocean_tensor_handle_t ocean_tensor_zeros_nd(
    const size_t *shape, size_t ndim, const char *dtype, const char *device
);
ocean_tensor_handle_t ocean_tensor_from_cpu_strided(
    const void *data,
    const size_t *shape,
    const size_t *strides,
    size_t ndim,
    const char *dtype,
    const char *device
);
ocean_tensor_handle_t ocean_tensor_from_cpu_native(
    const void *source, const char *dtype, const char *device
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
void *ocean_tensor_to_cpu_tensor(ocean_tensor_handle_t tensor);
void ocean_tensor_export_free(void *value);
void ocean_tensor_release(ocean_tensor_handle_t tensor);

#endif
