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
ocean_tensor_handle_t ocean_autograd_mse_loss(
    ocean_tensor_handle_t prediction,
    ocean_tensor_handle_t target
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

#endif
