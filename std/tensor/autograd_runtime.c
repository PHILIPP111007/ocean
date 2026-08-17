#include "std/tensor/autograd_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    OCEAN_AUTOGRAD_ADD = 0,
    OCEAN_AUTOGRAD_SUB = 1,
    OCEAN_AUTOGRAD_MUL = 2,
    OCEAN_AUTOGRAD_DIV = 3,
    OCEAN_AUTOGRAD_MATMUL = 10,
    OCEAN_AUTOGRAD_SCALAR = 11,
    OCEAN_AUTOGRAD_TRANSPOSE = 12,
    OCEAN_AUTOGRAD_RELU = 13,
    OCEAN_AUTOGRAD_MSE = 14,
};

typedef struct ocean_autograd_meta ocean_autograd_meta;

typedef struct ocean_autograd_node {
    int operation;
    ocean_autograd_meta *left;
    ocean_autograd_meta *right;
    ocean_tensor_handle_t saved_left;
    ocean_tensor_handle_t saved_right;
    double scalar;
    int scalar_operation;
} ocean_autograd_node;

struct ocean_autograd_meta {
    ocean_tensor_handle_t tensor;
    bool requires_grad;
    bool leaf;
    ocean_tensor_handle_t grad;
    ocean_autograd_node *grad_fn;
    size_t ndim;
    size_t *shape;
    char *device;
    ocean_autograd_meta *next;
};

static ocean_autograd_meta *ocean_autograd_metas = NULL;
static bool ocean_autograd_shutdown_registered = false;

static void ocean_autograd_require_float32(ocean_tensor_handle_t tensor) {
    char *dtype = ocean_tensor_dtype_name(tensor);
    bool valid = dtype && strcmp(dtype, "float32") == 0;
    free(dtype);
    if (!valid) {
        ocean_tensor_fail("ML v0.1 autograd supports Tensor[float32] only");
    }
}

static size_t *ocean_autograd_shape_copy(
    ocean_tensor_handle_t tensor,
    size_t *ndim_out
) {
    int rank = ocean_tensor_ndim(tensor);
    if (rank <= 0) ocean_tensor_fail("autograd requires Tensor rank >= 1");
    size_t ndim = (size_t)rank;
    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory copying Tensor shape");
    for (size_t axis = 0; axis < ndim; ++axis) {
        int dim = ocean_tensor_shape(tensor, (int)axis);
        if (dim < 0) {
            free(shape);
            ocean_tensor_fail("invalid Tensor dimension in autograd");
        }
        shape[axis] = (size_t)dim;
    }
    *ndim_out = ndim;
    return shape;
}

static bool ocean_autograd_same_shape_meta(
    ocean_tensor_handle_t tensor,
    const ocean_autograd_meta *meta
) {
    if ((size_t)ocean_tensor_ndim(tensor) != meta->ndim) return false;
    for (size_t axis = 0; axis < meta->ndim; ++axis) {
        if ((size_t)ocean_tensor_shape(tensor, (int)axis) != meta->shape[axis]) {
            return false;
        }
    }
    return true;
}

static ocean_autograd_meta *ocean_autograd_find(ocean_tensor_handle_t tensor) {
    for (ocean_autograd_meta *meta = ocean_autograd_metas; meta; meta = meta->next) {
        if (meta->tensor == tensor) return meta;
    }
    return NULL;
}

static void ocean_autograd_node_free(ocean_autograd_node *node) {
    if (!node) return;
    ocean_tensor_release(node->saved_left);
    ocean_tensor_release(node->saved_right);
    free(node);
}

static void ocean_autograd_meta_free(ocean_autograd_meta *meta) {
    if (!meta) return;
    ocean_tensor_release(meta->grad);
    ocean_autograd_node_free(meta->grad_fn);
    free(meta->shape);
    free(meta->device);
    free(meta);
}

static void ocean_autograd_shutdown(void) {
    ocean_autograd_meta *meta = ocean_autograd_metas;
    while (meta) {
        ocean_autograd_meta *next = meta->next;
        ocean_autograd_meta_free(meta);
        meta = next;
    }
    ocean_autograd_metas = NULL;
}

static ocean_autograd_meta *ocean_autograd_get(
    ocean_tensor_handle_t tensor,
    bool create
) {
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    if (meta || !create) return meta;

    ocean_autograd_require_float32(tensor);

    meta = (ocean_autograd_meta *)calloc(1, sizeof(*meta));
    if (!meta) ocean_tensor_fail("out of memory creating autograd metadata");

    meta->tensor = tensor;
    meta->shape = ocean_autograd_shape_copy(tensor, &meta->ndim);
    meta->device = ocean_tensor_device(tensor);
    if (!meta->device) {
        ocean_autograd_meta_free(meta);
        ocean_tensor_fail("could not read Tensor device for autograd");
    }
    meta->leaf = true;
    meta->next = ocean_autograd_metas;
    ocean_autograd_metas = meta;

    if (!ocean_autograd_shutdown_registered) {
        ocean_autograd_shutdown_registered = true;
        atexit(ocean_autograd_shutdown);
    }
    return meta;
}

static void ocean_autograd_remove_meta(ocean_autograd_meta *target) {
    ocean_autograd_meta **cursor = &ocean_autograd_metas;
    while (*cursor) {
        if (*cursor == target) {
            *cursor = target->next;
            target->next = NULL;
            ocean_autograd_meta_free(target);
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static ocean_tensor_handle_t ocean_autograd_zeros_meta(
    const ocean_autograd_meta *meta
) {
    return ocean_tensor_zeros_nd(
        meta->shape,
        meta->ndim,
        "float32",
        meta->device
    );
}

static void ocean_autograd_accumulate(
    ocean_autograd_meta *meta,
    ocean_tensor_handle_t contribution
) {
    if (!meta) {
        ocean_tensor_release(contribution);
        return;
    }
    if (!contribution) ocean_tensor_fail("null gradient contribution");
    if (!meta->grad) {
        meta->grad = contribution;
        return;
    }
    ocean_tensor_handle_t sum = ocean_tensor_binary(
        meta->grad,
        contribution,
        OCEAN_AUTOGRAD_ADD
    );
    ocean_tensor_release(meta->grad);
    ocean_tensor_release(contribution);
    meta->grad = sum;
}

static ocean_tensor_handle_t ocean_autograd_sum_to_meta(
    ocean_tensor_handle_t source,
    const ocean_autograd_meta *target
) {
    if (ocean_autograd_same_shape_meta(source, target)) {
        return ocean_tensor_copy(source);
    }

    size_t source_ndim = 0;
    size_t *source_shape = ocean_autograd_shape_copy(source, &source_ndim);

    if (target->ndim > source_ndim) {
        free(source_shape);
        ocean_tensor_fail("autograd cannot reduce gradient to a higher rank");
    }

    char *source_device = ocean_tensor_device(source);
    bool source_is_cpu = strcmp(source_device, "cpu") == 0;
    free(source_device);

    ocean_tensor_handle_t source_cpu = source_is_cpu
        ? source
        : ocean_tensor_to(source, "cpu");

    size_t target_size = 1;
    for (size_t axis = 0; axis < target->ndim; ++axis) {
        if (
            target->shape[axis] != 0
            && target_size > SIZE_MAX / target->shape[axis]
        ) {
            free(source_shape);
            if (source_cpu != source) ocean_tensor_release(source_cpu);
            ocean_tensor_fail("autograd reduction target shape is too large");
        }
        target_size *= target->shape[axis];
    }

    float *target_data = target_size
        ? (float *)calloc(target_size, sizeof(float))
        : NULL;

    size_t *coordinates = source_ndim
        ? (size_t *)calloc(source_ndim, sizeof(size_t))
        : NULL;

    if ((target_size && !target_data) || (source_ndim && !coordinates)) {
        free(source_shape);
        free(target_data);
        free(coordinates);
        if (source_cpu != source) ocean_tensor_release(source_cpu);
        ocean_tensor_fail("out of memory reducing broadcast gradient");
    }

    size_t source_size = ocean_tensor_size(source_cpu);
    size_t leading = source_ndim - target->ndim;

    for (size_t linear = 0; linear < source_size; ++linear) {
        size_t remaining = linear;

        for (size_t axis = source_ndim; axis-- > 0;) {
            size_t dim = source_shape[axis];
            coordinates[axis] = dim ? remaining % dim : 0;
            remaining = dim ? remaining / dim : 0;
        }

        size_t target_linear = 0;

        for (size_t axis = 0; axis < target->ndim; ++axis) {
            size_t coordinate = coordinates[leading + axis];

            if (target->shape[axis] == 1) {
                coordinate = 0;
            } else if (coordinate >= target->shape[axis]) {
                free(source_shape);
                free(target_data);
                free(coordinates);
                if (source_cpu != source) ocean_tensor_release(source_cpu);
                ocean_tensor_fail(
                    "autograd broadcast reduction shape mismatch"
                );
            }

            target_linear =
                target_linear * target->shape[axis] + coordinate;
        }

        target_data[target_linear] +=
            ocean_tensor_get_flat_f32(source_cpu, linear);
    }

    size_t *target_strides = target->ndim
        ? (size_t *)malloc(target->ndim * sizeof(size_t))
        : NULL;

    if (target->ndim && !target_strides) {
        free(source_shape);
        free(target_data);
        free(coordinates);
        if (source_cpu != source) ocean_tensor_release(source_cpu);
        ocean_tensor_fail("out of memory creating reduction strides");
    }

    if (target->ndim) {
        target_strides[target->ndim - 1] = 1;

        for (size_t axis = target->ndim - 1; axis > 0; --axis) {
            target_strides[axis - 1] =
                target_strides[axis] * target->shape[axis];
        }
    }

    ocean_tensor_handle_t target_cpu = ocean_tensor_from_cpu_strided(
        target_data,
        target->shape,
        target_strides,
        target->ndim,
        "float32",
        "cpu"
    );

    free(source_shape);
    free(target_data);
    free(target_strides);
    free(coordinates);

    if (source_cpu != source) {
        ocean_tensor_release(source_cpu);
    }

    if (strcmp(target->device, "cpu") == 0) {
        return target_cpu;
    }

    ocean_tensor_handle_t result =
        ocean_tensor_to(target_cpu, target->device);

    ocean_tensor_release(target_cpu);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_from_float_data_like(
    ocean_tensor_handle_t reference,
    const float *data
) {
    size_t ndim = 0;
    size_t *shape = ocean_autograd_shape_copy(reference, &ndim);
    size_t *strides = (size_t *)malloc(ndim * sizeof(size_t));
    if (!strides) {
        free(shape);
        ocean_tensor_fail("out of memory creating autograd strides");
    }

    strides[ndim - 1] = 1;
    for (size_t axis = ndim - 1; axis > 0; --axis) {
        strides[axis - 1] = strides[axis] * shape[axis];
    }

    ocean_tensor_handle_t cpu = ocean_tensor_from_cpu_strided(
        data,
        shape,
        strides,
        ndim,
        "float32",
        "cpu"
    );

    char *device = ocean_tensor_device(reference);
    ocean_tensor_handle_t result = cpu;
    if (strcmp(device, "cpu") != 0) {
        result = ocean_tensor_to(cpu, device);
        ocean_tensor_release(cpu);
    }

    free(device);
    free(shape);
    free(strides);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_relu_impl(
    ocean_tensor_handle_t tensor
) {
    ocean_autograd_require_float32(tensor);

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = strcmp(device, "cpu") == 0
        ? tensor
        : ocean_tensor_to(tensor, "cpu");

    size_t size = ocean_tensor_size(cpu);
    float *data = size ? (float *)malloc(size * sizeof(float)) : NULL;
    if (size && !data) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("out of memory in ReLU");
    }

    for (size_t index = 0; index < size; ++index) {
        float value = ocean_tensor_get_flat_f32(cpu, index);
        data[index] = value > 0.0f ? value : 0.0f;
    }

    ocean_tensor_handle_t result = ocean_autograd_from_float_data_like(cpu, data);
    if (strcmp(device, "cpu") != 0) {
        ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
        ocean_tensor_release(result);
        result = moved;
    }

    free(data);
    free(device);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_relu_backward_impl(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t saved_input
) {
    char *device = ocean_tensor_device(upstream);
    ocean_tensor_handle_t upstream_cpu = strcmp(device, "cpu") == 0
        ? upstream
        : ocean_tensor_to(upstream, "cpu");
    ocean_tensor_handle_t input_cpu = strcmp(device, "cpu") == 0
        ? saved_input
        : ocean_tensor_to(saved_input, "cpu");

    size_t size = ocean_tensor_size(upstream_cpu);
    float *data = size ? (float *)malloc(size * sizeof(float)) : NULL;
    if (size && !data) {
        if (upstream_cpu != upstream) ocean_tensor_release(upstream_cpu);
        if (input_cpu != saved_input) ocean_tensor_release(input_cpu);
        free(device);
        ocean_tensor_fail("out of memory in ReLU backward");
    }

    for (size_t index = 0; index < size; ++index) {
        float gradient = ocean_tensor_get_flat_f32(upstream_cpu, index);
        float input = ocean_tensor_get_flat_f32(input_cpu, index);
        data[index] = input > 0.0f ? gradient : 0.0f;
    }

    ocean_tensor_handle_t result = ocean_autograd_from_float_data_like(
        upstream_cpu,
        data
    );
    if (strcmp(device, "cpu") != 0) {
        ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
        ocean_tensor_release(result);
        result = moved;
    }

    free(data);
    free(device);
    if (upstream_cpu != upstream) ocean_tensor_release(upstream_cpu);
    if (input_cpu != saved_input) ocean_tensor_release(input_cpu);
    return result;
}

static ocean_autograd_node *ocean_autograd_node_new(int operation) {
    ocean_autograd_node *node =
        (ocean_autograd_node *)calloc(1, sizeof(*node));
    if (!node) ocean_tensor_fail("out of memory creating autograd node");
    node->operation = operation;
    return node;
}

static void ocean_autograd_attach(
    ocean_tensor_handle_t result,
    ocean_autograd_node *node
) {
    ocean_autograd_meta *meta = ocean_autograd_get(result, true);
    meta->requires_grad = true;
    meta->leaf = false;
    ocean_autograd_node_free(meta->grad_fn);
    meta->grad_fn = node;
}

void ocean_autograd_set_requires_grad(
    ocean_tensor_handle_t tensor,
    bool value
) {
    if (!tensor) ocean_tensor_fail("requires_grad on null Tensor");
    ocean_autograd_meta *meta = ocean_autograd_get(tensor, value);
    if (!meta) return;

    if (!value) {
        ocean_tensor_release(meta->grad);
        meta->grad = NULL;
        meta->requires_grad = false;
        return;
    }

    ocean_autograd_require_float32(tensor);
    meta->requires_grad = true;
    if (!meta->grad_fn) meta->leaf = true;
}

bool ocean_autograd_requires_grad(ocean_tensor_handle_t tensor) {
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    return meta && meta->requires_grad;
}

bool ocean_autograd_has_grad(ocean_tensor_handle_t tensor) {
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    return meta && meta->grad != NULL;
}

ocean_tensor_handle_t ocean_autograd_grad_copy(
    ocean_tensor_handle_t tensor
) {
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    if (!meta || !meta->grad) {
        ocean_tensor_fail("Tensor has no gradient");
    }
    return ocean_tensor_copy(meta->grad);
}

void ocean_autograd_zero_grad(ocean_tensor_handle_t tensor) {
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    if (!meta) return;
    ocean_tensor_release(meta->grad);
    meta->grad = NULL;
}

ocean_tensor_handle_t ocean_autograd_binary(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
    fprintf(stderr, "[MLDIAG] binary op=%d L ndim=%d R ndim=%d\n",
        operation, ocean_tensor_ndim(left), ocean_tensor_ndim(right));
    ocean_tensor_handle_t result = ocean_tensor_binary(left, right, operation);
    fprintf(stderr, "[MLDIAG] binary result ndim=%d\n", ocean_tensor_ndim(result));

    ocean_autograd_meta *left_meta = ocean_autograd_find(left);
    ocean_autograd_meta *right_meta = ocean_autograd_find(right);
    bool left_grad = left_meta && left_meta->requires_grad;
    bool right_grad = right_meta && right_meta->requires_grad;

    if (!left_grad && !right_grad) return result;
    ocean_autograd_require_float32(left);
    ocean_autograd_require_float32(right);

    ocean_autograd_node *node = ocean_autograd_node_new(operation);
    node->left = left_grad ? left_meta : NULL;
    node->right = right_grad ? right_meta : NULL;

    if (operation == OCEAN_AUTOGRAD_MUL || operation == OCEAN_AUTOGRAD_DIV) {
        node->saved_left = ocean_tensor_copy(left);
        node->saved_right = ocean_tensor_copy(right);
    }

    fprintf(stderr, "[MLDIAG] mse before attach\n");
    ocean_autograd_attach(result, node);
    fprintf(stderr, "[MLDIAG] mse after attach\n");
    return result;
}

ocean_tensor_handle_t ocean_autograd_scalar(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    ocean_tensor_handle_t result = ocean_tensor_scalar(tensor, scalar, operation);
    ocean_autograd_meta *parent = ocean_autograd_find(tensor);

    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node = ocean_autograd_node_new(OCEAN_AUTOGRAD_SCALAR);
    node->left = parent;
    node->scalar = scalar;
    node->scalar_operation = operation;
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    fprintf(stderr, "[MLDIAG] matmul enter L(ndim=%d,%d,%d) R(ndim=%d,%d,%d)\n",
        ocean_tensor_ndim(left), ocean_tensor_shape(left, 0), ocean_tensor_shape(left, 1),
        ocean_tensor_ndim(right), ocean_tensor_shape(right, 0), ocean_tensor_shape(right, 1));
    ocean_tensor_handle_t result = ocean_tensor_matmul(left, right);
    fprintf(stderr, "[MLDIAG] matmul result ndim=%d shape0=%d shape1=%d\n",
        ocean_tensor_ndim(result), ocean_tensor_shape(result, 0), ocean_tensor_shape(result, 1));
    ocean_autograd_meta *left_meta = ocean_autograd_find(left);
    ocean_autograd_meta *right_meta = ocean_autograd_find(right);
    bool left_grad = left_meta && left_meta->requires_grad;
    bool right_grad = right_meta && right_meta->requires_grad;

    if (!left_grad && !right_grad) return result;

    if (ocean_tensor_ndim(left) != 2 || ocean_tensor_ndim(right) != 2) {
        ocean_tensor_release(result);
        ocean_tensor_fail("ML v0.1 autograd matmul supports 2D Tensors only");
    }

    ocean_autograd_node *node = ocean_autograd_node_new(OCEAN_AUTOGRAD_MATMUL);
    node->left = left_grad ? left_meta : NULL;
    node->right = right_grad ? right_meta : NULL;
    node->saved_left = ocean_tensor_copy(left);
    node->saved_right = ocean_tensor_copy(right);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_transpose(
    ocean_tensor_handle_t tensor
) {
    ocean_tensor_handle_t result = ocean_tensor_transpose(tensor);
    ocean_autograd_meta *parent = ocean_autograd_find(tensor);

    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_TRANSPOSE);
    node->left = parent;
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_relu(
    ocean_tensor_handle_t tensor
) {
    ocean_tensor_handle_t result = ocean_autograd_relu_impl(tensor);
    ocean_autograd_meta *parent = ocean_autograd_find(tensor);

    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node = ocean_autograd_node_new(OCEAN_AUTOGRAD_RELU);
    node->left = parent;
    node->saved_left = ocean_tensor_copy(tensor);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_mse_loss(
    ocean_tensor_handle_t prediction,
    ocean_tensor_handle_t target
) {
    fprintf(stderr, "[MLDIAG] mse enter prediction ndim=%d target ndim=%d\n",
        ocean_tensor_ndim(prediction), ocean_tensor_ndim(target));
    ocean_autograd_require_float32(prediction);
    ocean_autograd_require_float32(target);

    if (ocean_tensor_size(prediction) != ocean_tensor_size(target)) {
        ocean_tensor_fail("mse_loss requires matching Tensor shapes");
    }

    ocean_tensor_handle_t difference = ocean_tensor_binary(
        prediction,
        target,
        OCEAN_AUTOGRAD_SUB
    );
    ocean_tensor_handle_t squared = ocean_tensor_binary(
        difference,
        difference,
        OCEAN_AUTOGRAD_MUL
    );
    double mean = ocean_tensor_mean(squared);

    char *device = ocean_tensor_device(prediction);
    fprintf(stderr, "[MLDIAG] mse before scalar zeros\n");
    ocean_tensor_handle_t scalar_cpu = ocean_tensor_zeros(1, 1, "cpu");
    fprintf(stderr, "[MLDIAG] mse scalar ndim=%d shape0=%d shape1=%d\n",
        ocean_tensor_ndim(scalar_cpu), ocean_tensor_shape(scalar_cpu, 0), ocean_tensor_shape(scalar_cpu, 1));
    ocean_tensor_fill(scalar_cpu, mean);
    fprintf(stderr, "[MLDIAG] mse after fill\n");

    ocean_tensor_handle_t result = scalar_cpu;
    if (strcmp(device, "cpu") != 0) {
        result = ocean_tensor_to(scalar_cpu, device);
        ocean_tensor_release(scalar_cpu);
    }

    free(device);
    ocean_tensor_release(difference);
    ocean_tensor_release(squared);
    fprintf(stderr, "[MLDIAG] mse after cleanup temporaries\n");

    fprintf(stderr, "[MLDIAG] mse before autograd_find\n");
    ocean_autograd_meta *prediction_meta = ocean_autograd_find(prediction);
    ocean_autograd_meta *target_meta = ocean_autograd_find(target);
    fprintf(stderr, "[MLDIAG] mse after autograd_find pred=%p target=%p\n", (void*)prediction_meta, (void*)target_meta);
    bool prediction_grad =
        prediction_meta && prediction_meta->requires_grad;
    bool target_grad = target_meta && target_meta->requires_grad;

    if (!prediction_grad && !target_grad) return result;

    fprintf(stderr, "[MLDIAG] mse before node_new\n");
    ocean_autograd_node *node = ocean_autograd_node_new(OCEAN_AUTOGRAD_MSE);
    fprintf(stderr, "[MLDIAG] mse after node_new\n");
    node->left = prediction_grad ? prediction_meta : NULL;
    node->right = target_grad ? target_meta : NULL;
    fprintf(stderr, "[MLDIAG] mse before copy prediction\n");
    node->saved_left = ocean_tensor_copy(prediction);
    fprintf(stderr, "[MLDIAG] mse after copy prediction\n");
    fprintf(stderr, "[MLDIAG] mse before copy target\n");
    node->saved_right = ocean_tensor_copy(target);
    fprintf(stderr, "[MLDIAG] mse after copy target\n");
    ocean_autograd_attach(result, node);
    return result;
}

typedef struct ocean_autograd_topology {
    ocean_autograd_meta **items;
    size_t count;
    size_t capacity;
} ocean_autograd_topology;

static bool ocean_autograd_topology_contains(
    const ocean_autograd_topology *topology,
    const ocean_autograd_meta *meta
) {
    for (size_t index = 0; index < topology->count; ++index) {
        if (topology->items[index] == meta) return true;
    }
    return false;
}

static void ocean_autograd_topology_push(
    ocean_autograd_topology *topology,
    ocean_autograd_meta *meta
) {
    if (ocean_autograd_topology_contains(topology, meta)) return;
    if (topology->count == topology->capacity) {
        size_t capacity = topology->capacity ? topology->capacity * 2 : 16;
        ocean_autograd_meta **grown = (ocean_autograd_meta **)realloc(
            topology->items,
            capacity * sizeof(*grown)
        );
        if (!grown) ocean_tensor_fail("out of memory building autograd graph");
        topology->items = grown;
        topology->capacity = capacity;
    }
    topology->items[topology->count++] = meta;
}

static void ocean_autograd_topology_visit(
    ocean_autograd_topology *topology,
    ocean_autograd_meta *meta
) {
    if (!meta || ocean_autograd_topology_contains(topology, meta)) return;

    ocean_autograd_node *node = meta->grad_fn;
    if (node) {
        ocean_autograd_topology_visit(topology, node->left);
        ocean_autograd_topology_visit(topology, node->right);
    }
    ocean_autograd_topology_push(topology, meta);
}

static void ocean_autograd_backward_node(ocean_autograd_meta *meta) {
    ocean_autograd_node *node = meta->grad_fn;
    ocean_tensor_handle_t upstream = meta->grad;
    if (!node || !upstream) return;

    if (node->operation == OCEAN_AUTOGRAD_SCALAR) {
        ocean_tensor_handle_t contribution = NULL;

        switch (node->scalar_operation) {
            case OCEAN_AUTOGRAD_ADD:
            case OCEAN_AUTOGRAD_SUB:
                contribution = ocean_tensor_copy(upstream);
                break;
            case OCEAN_AUTOGRAD_MUL:
                contribution = ocean_tensor_scalar(
                    upstream,
                    node->scalar,
                    OCEAN_AUTOGRAD_MUL
                );
                break;
            case OCEAN_AUTOGRAD_DIV:
                contribution = ocean_tensor_scalar(
                    upstream,
                    node->scalar,
                    OCEAN_AUTOGRAD_DIV
                );
                break;
            default:
                ocean_tensor_fail("invalid scalar autograd operation");
        }

        ocean_autograd_accumulate(node->left, contribution);
        return;
    }

    switch (node->operation) {
        case OCEAN_AUTOGRAD_ADD: {
            if (node->left) {
                ocean_autograd_accumulate(
                    node->left,
                    ocean_autograd_sum_to_meta(upstream, node->left)
                );
            }
            if (node->right) {
                ocean_autograd_accumulate(
                    node->right,
                    ocean_autograd_sum_to_meta(upstream, node->right)
                );
            }
            break;
        }

        case OCEAN_AUTOGRAD_SUB: {
            if (node->left) {
                ocean_autograd_accumulate(
                    node->left,
                    ocean_autograd_sum_to_meta(upstream, node->left)
                );
            }
            if (node->right) {
                ocean_tensor_handle_t negative = ocean_tensor_scalar(
                    upstream,
                    -1.0,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t reduced =
                    ocean_autograd_sum_to_meta(negative, node->right);
                ocean_tensor_release(negative);
                ocean_autograd_accumulate(node->right, reduced);
            }
            break;
        }

        case OCEAN_AUTOGRAD_MUL: {
            if (node->left) {
                ocean_tensor_handle_t raw = ocean_tensor_binary(
                    upstream,
                    node->saved_right,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t reduced =
                    ocean_autograd_sum_to_meta(raw, node->left);
                ocean_tensor_release(raw);
                ocean_autograd_accumulate(node->left, reduced);
            }
            if (node->right) {
                ocean_tensor_handle_t raw = ocean_tensor_binary(
                    upstream,
                    node->saved_left,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t reduced =
                    ocean_autograd_sum_to_meta(raw, node->right);
                ocean_tensor_release(raw);
                ocean_autograd_accumulate(node->right, reduced);
            }
            break;
        }

        case OCEAN_AUTOGRAD_DIV: {
            if (node->left) {
                ocean_tensor_handle_t raw = ocean_tensor_binary(
                    upstream,
                    node->saved_right,
                    OCEAN_AUTOGRAD_DIV
                );
                ocean_tensor_handle_t reduced =
                    ocean_autograd_sum_to_meta(raw, node->left);
                ocean_tensor_release(raw);
                ocean_autograd_accumulate(node->left, reduced);
            }
            if (node->right) {
                ocean_tensor_handle_t denominator_squared = ocean_tensor_binary(
                    node->saved_right,
                    node->saved_right,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t numerator = ocean_tensor_binary(
                    upstream,
                    node->saved_left,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t divided = ocean_tensor_binary(
                    numerator,
                    denominator_squared,
                    OCEAN_AUTOGRAD_DIV
                );
                ocean_tensor_handle_t negative = ocean_tensor_scalar(
                    divided,
                    -1.0,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t reduced =
                    ocean_autograd_sum_to_meta(negative, node->right);
                ocean_tensor_release(denominator_squared);
                ocean_tensor_release(numerator);
                ocean_tensor_release(divided);
                ocean_tensor_release(negative);
                ocean_autograd_accumulate(node->right, reduced);
            }
            break;
        }

        case OCEAN_AUTOGRAD_MATMUL: {
            if (node->left) {
                ocean_tensor_handle_t right_t =
                    ocean_tensor_transpose(node->saved_right);
                ocean_tensor_handle_t contribution =
                    ocean_tensor_matmul(upstream, right_t);
                ocean_tensor_release(right_t);
                ocean_autograd_accumulate(node->left, contribution);
            }
            if (node->right) {
                ocean_tensor_handle_t left_t =
                    ocean_tensor_transpose(node->saved_left);
                ocean_tensor_handle_t contribution =
                    ocean_tensor_matmul(left_t, upstream);
                ocean_tensor_release(left_t);
                ocean_autograd_accumulate(node->right, contribution);
            }
            break;
        }

        case OCEAN_AUTOGRAD_TRANSPOSE: {
            ocean_autograd_accumulate(
                node->left,
                ocean_tensor_transpose(upstream)
            );
            break;
        }

        case OCEAN_AUTOGRAD_RELU: {
            ocean_autograd_accumulate(
                node->left,
                ocean_autograd_relu_backward_impl(
                    upstream,
                    node->saved_left
                )
            );
            break;
        }

        case OCEAN_AUTOGRAD_MSE: {
            double scale =
                2.0 * ocean_tensor_item(upstream)
                / (double)ocean_tensor_size(node->saved_left);

            ocean_tensor_handle_t difference = ocean_tensor_binary(
                node->saved_left,
                node->saved_right,
                OCEAN_AUTOGRAD_SUB
            );

            if (node->left) {
                ocean_tensor_handle_t contribution = ocean_tensor_scalar(
                    difference,
                    scale,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_autograd_accumulate(node->left, contribution);
            }

            if (node->right) {
                ocean_tensor_handle_t contribution = ocean_tensor_scalar(
                    difference,
                    -scale,
                    OCEAN_AUTOGRAD_MUL
                );
                ocean_autograd_accumulate(node->right, contribution);
            }

            ocean_tensor_release(difference);
            break;
        }

        default:
            ocean_tensor_fail("unsupported autograd operation");
    }
}

void ocean_autograd_backward(ocean_tensor_handle_t tensor) {
    ocean_autograd_meta *output = ocean_autograd_find(tensor);
    if (!output || !output->requires_grad) {
        ocean_tensor_fail("backward() requires a Tensor with requires_grad");
    }
    if (ocean_tensor_size(tensor) != 1) {
        ocean_tensor_fail("ML v0.1 backward() requires a scalar Tensor");
    }

    ocean_tensor_release(output->grad);
    output->grad = ocean_autograd_zeros_meta(output);
    ocean_tensor_fill(output->grad, 1.0);

    ocean_autograd_topology topology = {0};
    ocean_autograd_topology_visit(&topology, output);

    for (size_t index = topology.count; index-- > 0;) {
        ocean_autograd_backward_node(topology.items[index]);
    }

    /*
     * Like PyTorch's default backward(), release the dynamic graph after use.
     * Leaf metadata and leaf gradients survive for optimizer.step().
     */
    for (size_t index = 0; index < topology.count; ++index) {
        ocean_autograd_meta *meta = topology.items[index];
        if (!meta->leaf) {
            ocean_autograd_remove_meta(meta);
        }
    }

    free(topology.items);
}

ocean_tensor_handle_t ocean_autograd_parameter_uniform(
    int rows,
    int cols,
    double scale,
    const char *device
) {
    if (rows < 0 || cols < 0) {
        ocean_tensor_fail("Parameter dimensions must be non-negative");
    }

    fprintf(stderr, "[MLDIAG] parameter_uniform rows=%d cols=%d\n", rows, cols);
    ocean_tensor_handle_t cpu = ocean_tensor_zeros(rows, cols, "cpu");
    fprintf(stderr, "[MLDIAG] parameter_uniform zeros ndim=%d shape0=%d shape1=%d\n",
        ocean_tensor_ndim(cpu), ocean_tensor_shape(cpu, 0), ocean_tensor_shape(cpu, 1));
    static uint32_t state = 0x9e3779b9u;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            state = state * 1664525u + 1013904223u;
            double unit = (double)(state >> 8) / 16777215.0;
            double value = (unit * 2.0 - 1.0) * scale;
            ocean_tensor_set_2d(cpu, row, col, value);
        }
    }

    if (!device || strcmp(device, "cpu") == 0) {
        return cpu;
    }

    ocean_tensor_handle_t result = ocean_tensor_to(cpu, device);
    ocean_tensor_release(cpu);
    return result;
}

void ocean_autograd_sgd_step(
    ocean_tensor_handle_t tensor,
    double learning_rate
) {
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    if (!meta || !meta->requires_grad || !meta->leaf) {
        ocean_tensor_fail("SGD expects a leaf Parameter");
    }
    if (!meta->grad) return;

    char *device = ocean_tensor_device(tensor);
    if (strcmp(device, "cpu") != 0) {
        free(device);
        ocean_tensor_fail("ML v0.1 SGD supports CPU Parameters only");
    }
    free(device);

    ocean_autograd_require_float32(tensor);

    size_t *indices = (size_t *)calloc(meta->ndim, sizeof(size_t));
    if (!indices) ocean_tensor_fail("out of memory in SGD");

    size_t size = ocean_tensor_size(tensor);
    for (size_t linear = 0; linear < size; ++linear) {
        size_t remaining = linear;
        for (size_t axis = meta->ndim; axis-- > 0;) {
            size_t dim = meta->shape[axis];
            indices[axis] = dim ? remaining % dim : 0;
            remaining = dim ? remaining / dim : 0;
        }

        float parameter = ocean_tensor_get_flat_f32(tensor, linear);
        float gradient = ocean_tensor_get_flat_f32(meta->grad, linear);
        ocean_tensor_set_nd_f32(
            tensor,
            indices,
            meta->ndim,
            parameter - (float)learning_rate * gradient
        );
    }

    free(indices);
}
