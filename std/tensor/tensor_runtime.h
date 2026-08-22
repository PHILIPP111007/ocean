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
ocean_tensor_handle_t ocean_tensor_load_npy(
    const char *path,
    const char *device
);
ocean_tensor_handle_t ocean_tensor_load_npy_typed(
    const char *path,
    const char *device,
    const char *expected_dtype
);
void ocean_tensor_save_npy(
    ocean_tensor_handle_t tensor,
    const char *path
);
ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_ternary_quantize(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_gelu(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_tensor_gelu_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input
);
ocean_tensor_handle_t ocean_tensor_embedding_forward(
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t indices
);
ocean_tensor_handle_t ocean_tensor_embedding_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
);
ocean_tensor_handle_t ocean_tensor_cross_entropy_forward(
    ocean_tensor_handle_t logits,
    ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
);
ocean_tensor_handle_t ocean_tensor_cross_entropy_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t targets
);
void ocean_tensor_copy_into(ocean_tensor_handle_t destination, ocean_tensor_handle_t source);
ocean_tensor_handle_t ocean_tensor_to(ocean_tensor_handle_t tensor, const char *device);
ocean_tensor_handle_t ocean_tensor_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
ocean_tensor_handle_t ocean_tensor_matmul_transposed(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
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
ocean_tensor_handle_t ocean_tensor_permute(
    ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
);
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
double ocean_tensor_get_flat(ocean_tensor_handle_t tensor, size_t index);
void ocean_tensor_set_flat(ocean_tensor_handle_t tensor, size_t index, double value);
bool ocean_tensor_get_flat_bool(ocean_tensor_handle_t tensor, size_t index);
int8_t ocean_tensor_get_flat_i8(ocean_tensor_handle_t tensor, size_t index);
int16_t ocean_tensor_get_flat_i16(ocean_tensor_handle_t tensor, size_t index);
int32_t ocean_tensor_get_flat_i32(ocean_tensor_handle_t tensor, size_t index);
int64_t ocean_tensor_get_flat_i64(ocean_tensor_handle_t tensor, size_t index);
uint8_t ocean_tensor_get_flat_u8(ocean_tensor_handle_t tensor, size_t index);
uint16_t ocean_tensor_get_flat_u16(ocean_tensor_handle_t tensor, size_t index);
uint32_t ocean_tensor_get_flat_u32(ocean_tensor_handle_t tensor, size_t index);
uint64_t ocean_tensor_get_flat_u64(ocean_tensor_handle_t tensor, size_t index);
float ocean_tensor_get_flat_f16(ocean_tensor_handle_t tensor, size_t index);
float ocean_tensor_get_flat_f32(ocean_tensor_handle_t tensor, size_t index);
double ocean_tensor_get_flat_f64(ocean_tensor_handle_t tensor, size_t index);
void ocean_tensor_set_flat_bool(ocean_tensor_handle_t tensor, size_t index, bool value);
void ocean_tensor_set_flat_i8(ocean_tensor_handle_t tensor, size_t index, int8_t value);
void ocean_tensor_set_flat_i16(ocean_tensor_handle_t tensor, size_t index, int16_t value);
void ocean_tensor_set_flat_i32(ocean_tensor_handle_t tensor, size_t index, int32_t value);
void ocean_tensor_set_flat_i64(ocean_tensor_handle_t tensor, size_t index, int64_t value);
void ocean_tensor_set_flat_u8(ocean_tensor_handle_t tensor, size_t index, uint8_t value);
void ocean_tensor_set_flat_u16(ocean_tensor_handle_t tensor, size_t index, uint16_t value);
void ocean_tensor_set_flat_u32(ocean_tensor_handle_t tensor, size_t index, uint32_t value);
void ocean_tensor_set_flat_u64(ocean_tensor_handle_t tensor, size_t index, uint64_t value);
void ocean_tensor_set_flat_f16(ocean_tensor_handle_t tensor, size_t index, float value);
void ocean_tensor_set_flat_f32(ocean_tensor_handle_t tensor, size_t index, float value);
void ocean_tensor_set_flat_f64(ocean_tensor_handle_t tensor, size_t index, double value);
bool ocean_tensor_get_nd_bool(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
int8_t ocean_tensor_get_nd_i8(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
int16_t ocean_tensor_get_nd_i16(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
int32_t ocean_tensor_get_nd_i32(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
int64_t ocean_tensor_get_nd_i64(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
uint8_t ocean_tensor_get_nd_u8(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
uint16_t ocean_tensor_get_nd_u16(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
uint32_t ocean_tensor_get_nd_u32(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
uint64_t ocean_tensor_get_nd_u64(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
float ocean_tensor_get_nd_f16(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
float ocean_tensor_get_nd_f32(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
double ocean_tensor_get_nd_f64(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim
);
void ocean_tensor_set_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim,
    double value
);
void ocean_tensor_set_nd_bool(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, bool value
);
void ocean_tensor_set_nd_i8(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, int8_t value
);
void ocean_tensor_set_nd_i16(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, int16_t value
);
void ocean_tensor_set_nd_i32(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, int32_t value
);
void ocean_tensor_set_nd_i64(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, int64_t value
);
void ocean_tensor_set_nd_u8(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, uint8_t value
);
void ocean_tensor_set_nd_u16(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, uint16_t value
);
void ocean_tensor_set_nd_u32(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, uint32_t value
);
void ocean_tensor_set_nd_u64(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, uint64_t value
);
void ocean_tensor_set_nd_f16(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, float value
);
void ocean_tensor_set_nd_f32(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, float value
);
void ocean_tensor_set_nd_f64(
    ocean_tensor_handle_t tensor, const size_t *indices, size_t ndim, double value
);
double ocean_tensor_get_2d(ocean_tensor_handle_t tensor, int row, int col);
void ocean_tensor_set_2d(ocean_tensor_handle_t tensor, int row, int col, double value);

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis);
int ocean_tensor_len(ocean_tensor_handle_t tensor);
int ocean_tensor_ndim(ocean_tensor_handle_t tensor);
size_t ocean_tensor_size(ocean_tensor_handle_t tensor);
char *ocean_tensor_device(ocean_tensor_handle_t tensor);
char *ocean_tensor_device_info(ocean_tensor_handle_t tensor);
uint64_t ocean_tensor_identity(ocean_tensor_handle_t tensor);
void ocean_tensor_release(ocean_tensor_handle_t tensor);


/* ND Tensor v0.2 */
ocean_tensor_handle_t ocean_tensor_reshape_3d(ocean_tensor_handle_t tensor, int d0, int d1, int d2);
ocean_tensor_handle_t ocean_tensor_reshape_4d(ocean_tensor_handle_t tensor, int d0, int d1, int d2, int d3);
ocean_tensor_handle_t ocean_tensor_transpose_dims(ocean_tensor_handle_t tensor, int dim0, int dim1);
ocean_tensor_handle_t ocean_tensor_sum_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim);
ocean_tensor_handle_t ocean_tensor_mean_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim);
ocean_tensor_handle_t ocean_tensor_softmax(ocean_tensor_handle_t tensor, int dim);
ocean_tensor_handle_t ocean_tensor_layer_norm(
    ocean_tensor_handle_t tensor, int dim, double epsilon
);
ocean_tensor_handle_t ocean_tensor_softmax_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t output,
    int dim
);
ocean_tensor_handle_t ocean_tensor_layer_norm_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input,
    int dim,
    double epsilon
);

/* Device-aware optimizer update primitives. Moment tensors remain opaque
   Tensor handles, so GPU optimizers never need to expose OpenCL objects. */
void ocean_tensor_sgd_update(
    ocean_tensor_handle_t parameter,
    ocean_tensor_handle_t gradient,
    double learning_rate
);
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
);

#endif
