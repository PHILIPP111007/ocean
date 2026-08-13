#ifndef OCEAN_STD_TENSOR_RUNTIME_H
#define OCEAN_STD_TENSOR_RUNTIME_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct ocean_tensor_handle *ocean_tensor_handle_t;

_Noreturn void ocean_tensor_fail(const char *message);
void ocean_tensor_validate_list_length(size_t actual, size_t expected);

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
ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_to(ocean_tensor_handle_t tensor, const char *device);
ocean_tensor_handle_t ocean_tensor_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
ocean_tensor_handle_t ocean_tensor_binary(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
);
ocean_tensor_handle_t ocean_tensor_scalar(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
);
ocean_tensor_handle_t ocean_tensor_reshape(
    ocean_tensor_handle_t tensor,
    const size_t *shape,
    size_t ndim
);
ocean_tensor_handle_t ocean_tensor_reshape_2d(
    ocean_tensor_handle_t tensor,
    int rows,
    int cols
);
ocean_tensor_handle_t ocean_tensor_transpose(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_row(ocean_tensor_handle_t tensor, int row);
ocean_tensor_handle_t ocean_tensor_column(ocean_tensor_handle_t tensor, int column);
ocean_tensor_handle_t ocean_tensor_slice(
    ocean_tensor_handle_t tensor,
    int axis,
    int start,
    int stop,
    int step
);
double ocean_tensor_sum(ocean_tensor_handle_t tensor);
double ocean_tensor_mean(ocean_tensor_handle_t tensor);
double ocean_tensor_max(ocean_tensor_handle_t tensor);
double ocean_tensor_min(ocean_tensor_handle_t tensor);
double ocean_tensor_item(ocean_tensor_handle_t tensor);
char *ocean_tensor_dtype_name(ocean_tensor_handle_t tensor);
bool ocean_tensor_is_contiguous(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_contiguous(ocean_tensor_handle_t tensor);
void ocean_tensor_fill(ocean_tensor_handle_t tensor, double value);
double ocean_tensor_get_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim
);
void ocean_tensor_set_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim,
    double value
);
double ocean_tensor_get_2d(ocean_tensor_handle_t tensor, int row, int col);
void ocean_tensor_set_2d(ocean_tensor_handle_t tensor, int row, int col, double value);

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis);
int ocean_tensor_ndim(ocean_tensor_handle_t tensor);
size_t ocean_tensor_size(ocean_tensor_handle_t tensor);
char *ocean_tensor_device(ocean_tensor_handle_t tensor);
void ocean_tensor_release(ocean_tensor_handle_t tensor);

#endif
