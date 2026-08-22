#ifndef OCEAN_STD_TENSOR_AUTOGRAD_RUNTIME_H
#define OCEAN_STD_TENSOR_AUTOGRAD_RUNTIME_H

#include <stdbool.h>
#include "std/tensor/tensor_runtime.h"

void ocean_autograd_set_requires_grad(
    ocean_tensor_handle_t tensor,
    bool value
);
bool ocean_autograd_requires_grad(
    ocean_tensor_handle_t tensor
);
bool ocean_autograd_has_grad(
    ocean_tensor_handle_t tensor
);
ocean_tensor_handle_t ocean_autograd_grad_copy(
    ocean_tensor_handle_t tensor
);
void ocean_autograd_zero_grad(
    ocean_tensor_handle_t tensor
);
void ocean_autograd_backward(
    ocean_tensor_handle_t tensor
);
void ocean_autograd_set_grad_enabled(bool enabled);
bool ocean_autograd_grad_enabled(void);

ocean_tensor_handle_t ocean_autograd_binary(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
);
ocean_tensor_handle_t ocean_autograd_scalar(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
);
ocean_tensor_handle_t ocean_autograd_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
ocean_tensor_handle_t ocean_autograd_transpose(
    ocean_tensor_handle_t tensor
);
ocean_tensor_handle_t ocean_autograd_relu(
    ocean_tensor_handle_t tensor
);
ocean_tensor_handle_t ocean_autograd_gelu(
    ocean_tensor_handle_t tensor
);
ocean_tensor_handle_t ocean_autograd_mse_loss(
    ocean_tensor_handle_t prediction,
    ocean_tensor_handle_t target
);
ocean_tensor_handle_t ocean_autograd_embedding(
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t indices
);
ocean_tensor_handle_t ocean_autograd_cross_entropy(
    ocean_tensor_handle_t logits,
    ocean_tensor_handle_t targets
);

ocean_tensor_handle_t ocean_autograd_parameter_uniform(
    int rows,
    int cols,
    double scale,
    const char *device
);
void ocean_autograd_sgd_step(
    ocean_tensor_handle_t tensor,
    double learning_rate
);


/* ND autograd v0.2 */
ocean_tensor_handle_t ocean_autograd_reshape_3d(ocean_tensor_handle_t tensor, int d0, int d1, int d2);
ocean_tensor_handle_t ocean_autograd_reshape_4d(ocean_tensor_handle_t tensor, int d0, int d1, int d2, int d3);
ocean_tensor_handle_t ocean_autograd_transpose_dims(ocean_tensor_handle_t tensor, int dim0, int dim1);
ocean_tensor_handle_t ocean_autograd_permute(
    ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
);
ocean_tensor_handle_t ocean_autograd_sum_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim);
ocean_tensor_handle_t ocean_autograd_mean_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim);

ocean_tensor_handle_t ocean_autograd_reshape(
    ocean_tensor_handle_t tensor,
    const size_t *shape,
    size_t ndim
);

/* Tensor/autograd v0.3 math */
ocean_tensor_handle_t ocean_autograd_exp(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_autograd_log(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_autograd_sqrt(ocean_tensor_handle_t tensor);
ocean_tensor_handle_t ocean_autograd_pow(
    ocean_tensor_handle_t tensor,
    double exponent
);
ocean_tensor_handle_t ocean_autograd_softmax(
    ocean_tensor_handle_t tensor,
    int dim
);
ocean_tensor_handle_t ocean_autograd_layer_norm(
    ocean_tensor_handle_t tensor,
    int dim,
    double epsilon
);

/* AdamW v0.1 */
int ocean_autograd_adamw_create(void);
int ocean_autograd_adamw_begin_step(int state_id);
void ocean_autograd_adamw_step(
    int state_id,
    int step,
    ocean_tensor_handle_t tensor,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay
);

#endif
