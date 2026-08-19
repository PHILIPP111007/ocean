#include "std/tensor/autograd_runtime.h"

#include <math.h>

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
    OCEAN_AUTOGRAD_RESHAPE = 15,
    OCEAN_AUTOGRAD_TRANSPOSE_DIMS = 16,
    OCEAN_AUTOGRAD_SUM_DIM = 17,
    OCEAN_AUTOGRAD_MEAN_DIM = 18,
    OCEAN_AUTOGRAD_EXP = 19,
    OCEAN_AUTOGRAD_LOG = 20,
    OCEAN_AUTOGRAD_SQRT = 21,
    OCEAN_AUTOGRAD_POW = 22,
    OCEAN_AUTOGRAD_SOFTMAX = 23,
    OCEAN_AUTOGRAD_LAYER_NORM = 24,
    OCEAN_AUTOGRAD_PERMUTE = 25,
    OCEAN_AUTOGRAD_EMBEDDING = 26,
    OCEAN_AUTOGRAD_CROSS_ENTROPY = 27,
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
    int dim0;
    int dim1;
    bool keepdim;
    int *axes;
    size_t axes_count;
} ocean_autograd_node;

struct ocean_autograd_meta {
    ocean_tensor_handle_t tensor;
    uint64_t tensor_identity;
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

static void ocean_autograd_remove_meta(ocean_autograd_meta *target);

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

static ocean_autograd_meta *ocean_autograd_find(
    ocean_tensor_handle_t tensor
) {
    if (!tensor) return NULL;

    uint64_t identity = ocean_tensor_identity(tensor);

    for (
        ocean_autograd_meta *meta = ocean_autograd_metas;
        meta;
        meta = meta->next
    ) {
        if (
            meta->tensor == tensor
            && meta->tensor_identity == identity
        ) {
            return meta;
        }
    }
    return NULL;
}

static void ocean_autograd_node_free(ocean_autograd_node *node) {
    if (!node) return;
    ocean_tensor_release(node->saved_left);
    ocean_tensor_release(node->saved_right);
    free(node->axes);
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
    meta->tensor_identity = ocean_tensor_identity(tensor);
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
        size_t padded_ndim = target->ndim;
        size_t leading = padded_ndim - source_ndim;
        size_t *padded_shape = (size_t *)malloc(
            padded_ndim * sizeof(size_t)
        );
        if (!padded_shape) {
            free(source_shape);
            ocean_tensor_fail(
                "out of memory padding broadcast gradient rank"
            );
        }

        for (size_t axis = 0; axis < leading; ++axis) {
            padded_shape[axis] = 1;
        }
        for (size_t axis = 0; axis < source_ndim; ++axis) {
            padded_shape[leading + axis] = source_shape[axis];
        }

        ocean_tensor_handle_t padded = ocean_tensor_reshape(
            source,
            padded_shape,
            padded_ndim
        );

        free(padded_shape);
        free(source_shape);

        ocean_tensor_handle_t result =
            ocean_autograd_sum_to_meta(padded, target);
        ocean_tensor_release(padded);
        return result;
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

    ocean_tensor_handle_t result = ocean_tensor_binary(left, right, operation);

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


    ocean_autograd_attach(result, node);

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

    ocean_tensor_handle_t result = ocean_tensor_matmul(left, right);

    ocean_autograd_meta *left_meta = ocean_autograd_find(left);
    ocean_autograd_meta *right_meta = ocean_autograd_find(right);
    bool left_grad = left_meta && left_meta->requires_grad;
    bool right_grad = right_meta && right_meta->requires_grad;

    if (!left_grad && !right_grad) return result;

    ocean_autograd_node *node = ocean_autograd_node_new(OCEAN_AUTOGRAD_MATMUL);
    node->left = left_grad ? left_meta : NULL;
    node->right = right_grad ? right_meta : NULL;
    node->saved_left = ocean_tensor_copy(left);
    node->saved_right = ocean_tensor_copy(right);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_transpose(ocean_tensor_handle_t tensor) { return ocean_autograd_transpose_dims(tensor,0,1); }

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

    ocean_tensor_handle_t scalar_cpu = ocean_tensor_zeros(1, 1, "cpu");

    ocean_tensor_fill(scalar_cpu, mean);

    ocean_tensor_handle_t result = scalar_cpu;
    if (strcmp(device, "cpu") != 0) {
        result = ocean_tensor_to(scalar_cpu, device);
        ocean_tensor_release(scalar_cpu);
    }

    free(device);
    ocean_tensor_release(difference);
    ocean_tensor_release(squared);


    ocean_autograd_meta *prediction_meta = ocean_autograd_find(prediction);
    ocean_autograd_meta *target_meta = ocean_autograd_find(target);

    bool prediction_grad =
        prediction_meta && prediction_meta->requires_grad;
    bool target_grad = target_meta && target_meta->requires_grad;

    if (!prediction_grad && !target_grad) return result;


    ocean_autograd_node *node = ocean_autograd_node_new(OCEAN_AUTOGRAD_MSE);

    node->left = prediction_grad ? prediction_meta : NULL;
    node->right = target_grad ? target_meta : NULL;

    node->saved_left = ocean_tensor_copy(prediction);


    node->saved_right = ocean_tensor_copy(target);

    ocean_autograd_attach(result, node);
    return result;
}


/* ================= ND autograd v0.2 ================= */
static int ocean_autograd_normalize_dim_v02(const ocean_autograd_meta *meta, int dim) {
    long long rank=(long long)meta->ndim, d=(long long)dim; if (d<0) d+=rank;
    if (d < 0 || d >= rank) {
        ocean_tensor_fail("autograd dimension is out of bounds");
    }
    return (int)d;
}

static ocean_tensor_handle_t ocean_autograd_expand_reduction_v02(ocean_tensor_handle_t upstream, const ocean_autograd_meta *target, int dim, bool keepdim, double scale) {
    size_t axis=(size_t)ocean_autograd_normalize_dim_v02(target,dim);
    char *device=ocean_tensor_device(upstream);
    ocean_tensor_handle_t uc=strcmp(device,"cpu")==0?upstream:ocean_tensor_to(upstream,"cpu");
    size_t n=1; for(size_t i=0;i<target->ndim;++i)n*=target->shape[i];
    float *data=n?malloc(n*sizeof(float)):NULL; size_t *coord=calloc(target->ndim,sizeof(size_t));
    if ((n&&!data)||!coord) ocean_tensor_fail("out of memory expanding reduction gradient");
    for(size_t linear=0;linear<n;++linear){
        size_t rem=linear; for(size_t i=target->ndim;i-- >0;){size_t d=target->shape[i];coord[i]=d?rem%d:0;rem=d?rem/d:0;}
        size_t ul=0;
        if(keepdim){for(size_t i=0;i<target->ndim;++i){size_t d=(size_t)ocean_tensor_shape(uc,(int)i);ul=ul*d+(i==axis?0:coord[i]);}}
        else if(target->ndim>1){size_t j=0;for(size_t i=0;i<target->ndim;++i)if(i!=axis){size_t d=(size_t)ocean_tensor_shape(uc,(int)j++);ul=ul*d+coord[i];}}
        data[linear]=ocean_tensor_get_flat_f32(uc,ul)*(float)scale;
    }
    size_t *strides=malloc(target->ndim*sizeof(size_t)); if(!strides)ocean_tensor_fail("out of memory creating reduction gradient strides");
    strides[target->ndim-1]=1; for(size_t i=target->ndim-1;i>0;--i)strides[i-1]=strides[i]*target->shape[i];
    ocean_tensor_handle_t cpu=ocean_tensor_from_cpu_strided(data,target->shape,strides,target->ndim,"float32","cpu");
    free(data);free(coord);free(strides);if(uc!=upstream)ocean_tensor_release(uc);
    if(strcmp(device,"cpu")==0){free(device);return cpu;} ocean_tensor_handle_t out=ocean_tensor_to(cpu,device);ocean_tensor_release(cpu);free(device);return out;
}


ocean_tensor_handle_t ocean_autograd_reshape(
    ocean_tensor_handle_t tensor,
    const size_t *shape,
    size_t ndim
) {
    ocean_tensor_handle_t result = ocean_tensor_reshape(tensor, shape, ndim);
    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_RESHAPE);
    node->left = parent;
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_reshape_3d(ocean_tensor_handle_t tensor,int d0,int d1,int d2){
    ocean_tensor_handle_t out=ocean_tensor_reshape_3d(tensor,d0,d1,d2); ocean_autograd_meta *p=ocean_autograd_find(tensor); if(!p||!p->requires_grad)return out;
    ocean_autograd_node *n=ocean_autograd_node_new(OCEAN_AUTOGRAD_RESHAPE);n->left=p;ocean_autograd_attach(out,n);return out;
}
ocean_tensor_handle_t ocean_autograd_reshape_4d(ocean_tensor_handle_t tensor,int d0,int d1,int d2,int d3){
    ocean_tensor_handle_t out=ocean_tensor_reshape_4d(tensor,d0,d1,d2,d3); ocean_autograd_meta *p=ocean_autograd_find(tensor); if(!p||!p->requires_grad)return out;
    ocean_autograd_node *n=ocean_autograd_node_new(OCEAN_AUTOGRAD_RESHAPE);n->left=p;ocean_autograd_attach(out,n);return out;
}
ocean_tensor_handle_t ocean_autograd_transpose_dims(ocean_tensor_handle_t tensor,int dim0,int dim1){
    ocean_tensor_handle_t out=ocean_tensor_transpose_dims(tensor,dim0,dim1); ocean_autograd_meta *p=ocean_autograd_find(tensor); if(!p||!p->requires_grad)return out;
    ocean_autograd_node *n=ocean_autograd_node_new(OCEAN_AUTOGRAD_TRANSPOSE_DIMS);n->left=p;n->dim0=dim0;n->dim1=dim1;ocean_autograd_attach(out,n);return out;
}
static ocean_tensor_handle_t ocean_autograd_reduce_dim_v02(ocean_tensor_handle_t tensor,int dim,bool keepdim,bool mean){
    ocean_tensor_handle_t out=mean?ocean_tensor_mean_dim(tensor,dim,keepdim):ocean_tensor_sum_dim(tensor,dim,keepdim); ocean_autograd_meta *p=ocean_autograd_find(tensor); if(!p||!p->requires_grad)return out;
    ocean_autograd_node *n=ocean_autograd_node_new(mean?OCEAN_AUTOGRAD_MEAN_DIM:OCEAN_AUTOGRAD_SUM_DIM);n->left=p;n->dim0=dim;n->keepdim=keepdim;ocean_autograd_attach(out,n);return out;
}
ocean_tensor_handle_t ocean_autograd_sum_dim(ocean_tensor_handle_t tensor,int dim,bool keepdim){return ocean_autograd_reduce_dim_v02(tensor,dim,keepdim,false);}
ocean_tensor_handle_t ocean_autograd_mean_dim(ocean_tensor_handle_t tensor,int dim,bool keepdim){return ocean_autograd_reduce_dim_v02(tensor,dim,keepdim,true);}


/* ================= Tensor/autograd v0.3 math ================= */

static int ocean_autograd_normalize_tensor_dim_v03(
    ocean_tensor_handle_t tensor,
    int dim
) {
    int rank = ocean_tensor_ndim(tensor);
    if (rank <= 0) {
        ocean_tensor_fail("Tensor dimension operation requires rank >= 1");
    }
    int normalized = dim < 0 ? dim + rank : dim;
    if (normalized < 0 || normalized >= rank) {
        ocean_tensor_fail("Tensor dimension is out of bounds");
    }
    return normalized;
}

static ocean_tensor_handle_t ocean_autograd_unary_cpu_v03(
    ocean_tensor_handle_t tensor,
    int operation,
    double scalar
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
        ocean_tensor_fail("out of memory in Tensor unary operation");
    }

    for (size_t index = 0; index < size; ++index) {
        float value = ocean_tensor_get_flat_f32(cpu, index);
        float result = 0.0f;

        switch (operation) {
            case OCEAN_AUTOGRAD_EXP:
                result = expf(value);
                break;
            case OCEAN_AUTOGRAD_LOG:
                if (!(value > 0.0f)) {
                    free(data);
                    if (cpu != tensor) ocean_tensor_release(cpu);
                    free(device);
                    ocean_tensor_fail("Tensor.log requires values > 0");
                }
                result = logf(value);
                break;
            case OCEAN_AUTOGRAD_SQRT:
                if (value < 0.0f) {
                    free(data);
                    if (cpu != tensor) ocean_tensor_release(cpu);
                    free(device);
                    ocean_tensor_fail("Tensor.sqrt requires values >= 0");
                }
                result = sqrtf(value);
                break;
            case OCEAN_AUTOGRAD_POW:
                result = powf(value, (float)scalar);
                break;
            default:
                free(data);
                if (cpu != tensor) ocean_tensor_release(cpu);
                free(device);
                ocean_tensor_fail("invalid Tensor unary operation");
        }
        data[index] = result;
    }

    ocean_tensor_handle_t result =
        ocean_autograd_from_float_data_like(cpu, data);

    if (strcmp(device, "cpu") != 0) {
        ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
        ocean_tensor_release(result);
        result = moved;
    }

    free(data);
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_softmax_impl_v03(
    ocean_tensor_handle_t tensor,
    int dim
) {
    ocean_autograd_require_float32(tensor);
    int axis = ocean_autograd_normalize_tensor_dim_v03(tensor, dim);
    int rank = ocean_tensor_ndim(tensor);

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = strcmp(device, "cpu") == 0
        ? tensor
        : ocean_tensor_to(tensor, "cpu");

    size_t outer = 1;
    size_t axis_size = (size_t)ocean_tensor_shape(cpu, axis);
    size_t inner = 1;

    for (int i = 0; i < axis; ++i) {
        outer *= (size_t)ocean_tensor_shape(cpu, i);
    }
    for (int i = axis + 1; i < rank; ++i) {
        inner *= (size_t)ocean_tensor_shape(cpu, i);
    }

    size_t size = ocean_tensor_size(cpu);
    float *data = size ? (float *)malloc(size * sizeof(float)) : NULL;
    if (size && !data) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("out of memory in softmax");
    }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float max_value = -INFINITY;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                float value = ocean_tensor_get_flat_f32(cpu, index);
                if (value > max_value) max_value = value;
            }

            double denominator = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                float value = ocean_tensor_get_flat_f32(cpu, index);
                float e = expf(value - max_value);
                data[index] = e;
                denominator += (double)e;
            }

            if (!(denominator > 0.0)) {
                free(data);
                if (cpu != tensor) ocean_tensor_release(cpu);
                free(device);
                ocean_tensor_fail("softmax denominator is not positive");
            }

            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                data[index] = (float)((double)data[index] / denominator);
            }
        }
    }

    ocean_tensor_handle_t result =
        ocean_autograd_from_float_data_like(cpu, data);

    if (strcmp(device, "cpu") != 0) {
        ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
        ocean_tensor_release(result);
        result = moved;
    }

    free(data);
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_softmax_backward_v03(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t output,
    int dim
) {
    int axis = ocean_autograd_normalize_tensor_dim_v03(output, dim);
    int rank = ocean_tensor_ndim(output);

    char *device = ocean_tensor_device(output);
    ocean_tensor_handle_t uc = strcmp(device, "cpu") == 0
        ? upstream
        : ocean_tensor_to(upstream, "cpu");
    ocean_tensor_handle_t yc = strcmp(device, "cpu") == 0
        ? output
        : ocean_tensor_to(output, "cpu");

    size_t outer = 1;
    size_t axis_size = (size_t)ocean_tensor_shape(yc, axis);
    size_t inner = 1;

    for (int i = 0; i < axis; ++i) {
        outer *= (size_t)ocean_tensor_shape(yc, i);
    }
    for (int i = axis + 1; i < rank; ++i) {
        inner *= (size_t)ocean_tensor_shape(yc, i);
    }

    size_t size = ocean_tensor_size(yc);
    float *data = size ? (float *)malloc(size * sizeof(float)) : NULL;
    if (size && !data) {
        if (uc != upstream) ocean_tensor_release(uc);
        if (yc != output) ocean_tensor_release(yc);
        free(device);
        ocean_tensor_fail("out of memory in softmax backward");
    }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            double dot = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                dot +=
                    (double)ocean_tensor_get_flat_f32(uc, index)
                    * (double)ocean_tensor_get_flat_f32(yc, index);
            }

            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                float g = ocean_tensor_get_flat_f32(uc, index);
                float y = ocean_tensor_get_flat_f32(yc, index);
                data[index] = y * (g - (float)dot);
            }
        }
    }

    ocean_tensor_handle_t cpu_result =
        ocean_autograd_from_float_data_like(yc, data);
    ocean_tensor_handle_t result = cpu_result;

    if (strcmp(device, "cpu") != 0) {
        result = ocean_tensor_to(cpu_result, device);
        ocean_tensor_release(cpu_result);
    }

    free(data);
    if (uc != upstream) ocean_tensor_release(uc);
    if (yc != output) ocean_tensor_release(yc);
    free(device);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_layer_norm_impl_v03(
    ocean_tensor_handle_t tensor,
    int dim,
    double epsilon
) {
    ocean_autograd_require_float32(tensor);
    if (!(epsilon > 0.0)) {
        ocean_tensor_fail("LayerNorm epsilon must be positive");
    }

    int axis = ocean_autograd_normalize_tensor_dim_v03(tensor, dim);
    int rank = ocean_tensor_ndim(tensor);

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = strcmp(device, "cpu") == 0
        ? tensor
        : ocean_tensor_to(tensor, "cpu");

    size_t outer = 1;
    size_t axis_size = (size_t)ocean_tensor_shape(cpu, axis);
    size_t inner = 1;

    for (int i = 0; i < axis; ++i) {
        outer *= (size_t)ocean_tensor_shape(cpu, i);
    }
    for (int i = axis + 1; i < rank; ++i) {
        inner *= (size_t)ocean_tensor_shape(cpu, i);
    }

    if (axis_size == 0) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("LayerNorm cannot normalize an empty dimension");
    }

    size_t size = ocean_tensor_size(cpu);
    float *data = size ? (float *)malloc(size * sizeof(float)) : NULL;
    if (size && !data) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("out of memory in LayerNorm");
    }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            double mean = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                mean += (double)ocean_tensor_get_flat_f32(cpu, index);
            }
            mean /= (double)axis_size;

            double variance = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                double delta =
                    (double)ocean_tensor_get_flat_f32(cpu, index) - mean;
                variance += delta * delta;
            }
            variance /= (double)axis_size;
            double inverse_std = 1.0 / sqrt(variance + epsilon);

            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                double value = (double)ocean_tensor_get_flat_f32(cpu, index);
                data[index] = (float)((value - mean) * inverse_std);
            }
        }
    }

    ocean_tensor_handle_t result =
        ocean_autograd_from_float_data_like(cpu, data);

    if (strcmp(device, "cpu") != 0) {
        ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
        ocean_tensor_release(result);
        result = moved;
    }

    free(data);
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_layer_norm_backward_v03(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t normalized,
    int dim,
    double epsilon
) {
    (void)normalized;

    int axis = ocean_autograd_normalize_tensor_dim_v03(input, dim);
    int rank = ocean_tensor_ndim(input);

    char *device = ocean_tensor_device(input);
    ocean_tensor_handle_t gc = strcmp(device, "cpu") == 0
        ? upstream
        : ocean_tensor_to(upstream, "cpu");
    ocean_tensor_handle_t xc = strcmp(device, "cpu") == 0
        ? input
        : ocean_tensor_to(input, "cpu");

    size_t outer = 1;
    size_t axis_size = (size_t)ocean_tensor_shape(xc, axis);
    size_t inner = 1;

    for (int i = 0; i < axis; ++i) {
        outer *= (size_t)ocean_tensor_shape(xc, i);
    }
    for (int i = axis + 1; i < rank; ++i) {
        inner *= (size_t)ocean_tensor_shape(xc, i);
    }

    if (axis_size == 0) {
        if (gc != upstream) ocean_tensor_release(gc);
        if (xc != input) ocean_tensor_release(xc);
        free(device);
        ocean_tensor_fail("LayerNorm backward cannot normalize an empty dimension");
    }

    size_t size = ocean_tensor_size(xc);
    float *data = size ? (float *)malloc(size * sizeof(float)) : NULL;
    if (size && !data) {
        if (gc != upstream) ocean_tensor_release(gc);
        if (xc != input) ocean_tensor_release(xc);
        free(device);
        ocean_tensor_fail("out of memory in LayerNorm backward");
    }

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            double mean = 0.0;

            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                mean += (double)ocean_tensor_get_flat_f32(xc, index);
            }
            mean /= (double)axis_size;

            double variance = 0.0;
            double sum_g = 0.0;
            double sum_g_centered = 0.0;

            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;

                double x =
                    (double)ocean_tensor_get_flat_f32(xc, index);
                double g =
                    (double)ocean_tensor_get_flat_f32(gc, index);
                double centered = x - mean;

                variance += centered * centered;
                sum_g += g;
                sum_g_centered += g * centered;
            }

            variance /= (double)axis_size;

            double inverse_std =
                1.0 / sqrt(variance + epsilon);
            double inverse_variance =
                inverse_std * inverse_std;

            double mean_g =
                sum_g / (double)axis_size;
            double mean_g_centered =
                sum_g_centered / (double)axis_size;

            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;

                double x =
                    (double)ocean_tensor_get_flat_f32(xc, index);
                double g =
                    (double)ocean_tensor_get_flat_f32(gc, index);
                double centered = x - mean;

                data[index] = (float)(
                    inverse_std
                    * (
                        g
                        - mean_g
                        - centered
                            * inverse_variance
                            * mean_g_centered
                    )
                );
            }
        }
    }

    ocean_tensor_handle_t cpu_result =
        ocean_autograd_from_float_data_like(xc, data);
    ocean_tensor_handle_t result = cpu_result;

    if (strcmp(device, "cpu") != 0) {
        result = ocean_tensor_to(cpu_result, device);
        ocean_tensor_release(cpu_result);
    }

    free(data);
    if (gc != upstream) ocean_tensor_release(gc);
    if (xc != input) ocean_tensor_release(xc);
    free(device);
    return result;
}

ocean_tensor_handle_t ocean_autograd_exp(ocean_tensor_handle_t tensor) {
    ocean_tensor_handle_t result =
        ocean_autograd_unary_cpu_v03(tensor, OCEAN_AUTOGRAD_EXP, 0.0);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_EXP);
    node->left = parent;
    node->saved_left = ocean_tensor_copy(result);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_log(ocean_tensor_handle_t tensor) {
    ocean_tensor_handle_t result =
        ocean_autograd_unary_cpu_v03(tensor, OCEAN_AUTOGRAD_LOG, 0.0);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_LOG);
    node->left = parent;
    node->saved_left = ocean_tensor_copy(tensor);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_sqrt(ocean_tensor_handle_t tensor) {
    ocean_tensor_handle_t result =
        ocean_autograd_unary_cpu_v03(tensor, OCEAN_AUTOGRAD_SQRT, 0.0);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_SQRT);
    node->left = parent;
    node->saved_left = ocean_tensor_copy(result);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_pow(
    ocean_tensor_handle_t tensor,
    double exponent
) {
    ocean_tensor_handle_t result =
        ocean_autograd_unary_cpu_v03(tensor, OCEAN_AUTOGRAD_POW, exponent);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_POW);
    node->left = parent;
    node->scalar = exponent;
    node->saved_left = ocean_tensor_copy(tensor);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_softmax(
    ocean_tensor_handle_t tensor,
    int dim
) {
    ocean_tensor_handle_t result =
        ocean_autograd_softmax_impl_v03(tensor, dim);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_SOFTMAX);
    node->left = parent;
    node->dim0 = dim;
    node->saved_left = ocean_tensor_copy(result);
    ocean_autograd_attach(result, node);
    return result;
}

ocean_tensor_handle_t ocean_autograd_layer_norm(
    ocean_tensor_handle_t tensor,
    int dim,
    double epsilon
) {
    ocean_tensor_handle_t result =
        ocean_autograd_layer_norm_impl_v03(tensor, dim, epsilon);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) return result;

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_LAYER_NORM);
    node->left = parent;
    node->dim0 = dim;
    node->scalar = epsilon;

    /*
     * LayerNorm backward needs the original input values.  The Ocean
     * language-level Tensor wrapper that produced this handle may be
     * destroyed before backward(), so parent->tensor is not a safe
     * lifetime anchor.  Keep an owned runtime copy on the grad node.
     */
    node->saved_left = ocean_tensor_copy(tensor);

    ocean_autograd_attach(result, node);
    return result;
}


ocean_tensor_handle_t ocean_autograd_permute(
    ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
) {
    ocean_tensor_handle_t result =
        ocean_tensor_permute(tensor, axes, ndim);

    ocean_autograd_meta *parent = ocean_autograd_find(tensor);
    if (!parent || !parent->requires_grad) {
        return result;
    }

    if (parent->ndim != ndim) {
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "autograd Tensor.permute axes must match Tensor rank"
        );
    }

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_PERMUTE);
    node->left = parent;
    node->axes_count = ndim;
    node->axes = ndim
        ? (int *)malloc(ndim * sizeof(int))
        : NULL;

    if (ndim && !node->axes) {
        ocean_autograd_node_free(node);
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "out of memory storing Tensor.permute autograd axes"
        );
    }

    bool *seen = ndim
        ? (bool *)calloc(ndim, sizeof(bool))
        : NULL;
    if (ndim && !seen) {
        ocean_autograd_node_free(node);
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "out of memory validating Tensor.permute autograd axes"
        );
    }

    for (size_t i = 0; i < ndim; ++i) {
        long long axis = (long long)axes[i];
        if (axis < 0) axis += (long long)ndim;

        if (
            axis < 0
            || axis >= (long long)ndim
            || seen[(size_t)axis]
        ) {
            free(seen);
            ocean_autograd_node_free(node);
            ocean_tensor_release(result);
            ocean_tensor_fail(
                "invalid Tensor.permute autograd permutation"
            );
        }

        node->axes[i] = (int)axis;
        seen[(size_t)axis] = true;
    }

    free(seen);
    ocean_autograd_attach(result, node);
    return result;
}


static void ocean_autograd_require_int64_v04(
    ocean_tensor_handle_t tensor,
    const char *message
) {
    char *dtype = ocean_tensor_dtype_name(tensor);
    bool valid = dtype && strcmp(dtype, "int64") == 0;
    free(dtype);
    if (!valid) ocean_tensor_fail(message);
}

static void ocean_autograd_contiguous_strides_v04(
    const size_t *shape,
    size_t ndim,
    size_t *strides
) {
    if (!ndim) return;
    strides[ndim - 1] = 1;
    for (size_t i = ndim - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
}

static ocean_tensor_handle_t ocean_autograd_embedding_forward_v04(
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t indices
) {
    ocean_autograd_require_float32(weight);
    ocean_autograd_require_int64_v04(
        indices,
        "Embedding indices must be Tensor[int64]"
    );

    int weight_rank = ocean_tensor_ndim(weight);
    int index_rank = ocean_tensor_ndim(indices);
    if (weight_rank != 2 || index_rank < 1) {
        ocean_tensor_fail(
            "Embedding expects weight [V,D] and indices rank >= 1"
        );
    }

    size_t vocab = (size_t)ocean_tensor_shape(weight, 0);
    size_t dim = (size_t)ocean_tensor_shape(weight, 1);
    size_t count = ocean_tensor_size(indices);
    size_t output_rank = (size_t)index_rank + 1;

    size_t *shape = (size_t *)malloc(output_rank * sizeof(size_t));
    size_t *strides = (size_t *)malloc(output_rank * sizeof(size_t));
    float *data = count && dim
        ? (float *)malloc(count * dim * sizeof(float))
        : NULL;

    if (!shape || !strides || (count && dim && !data)) {
        free(shape);
        free(strides);
        free(data);
        ocean_tensor_fail("out of memory in Embedding forward");
    }

    for (int axis = 0; axis < index_rank; ++axis) {
        shape[(size_t)axis] =
            (size_t)ocean_tensor_shape(indices, axis);
    }
    shape[output_rank - 1] = dim;
    ocean_autograd_contiguous_strides_v04(
        shape,
        output_rank,
        strides
    );

    char *device = ocean_tensor_device(weight);
    ocean_tensor_handle_t wc = strcmp(device, "cpu") == 0
        ? weight
        : ocean_tensor_to(weight, "cpu");

    char *indices_device = ocean_tensor_device(indices);
    ocean_tensor_handle_t ic = strcmp(indices_device, "cpu") == 0
        ? indices
        : ocean_tensor_to(indices, "cpu");

    for (size_t i = 0; i < count; ++i) {
        int64_t token = ocean_tensor_get_flat_i64(ic, i);
        if (token < 0 || (uint64_t)token >= (uint64_t)vocab) {
            free(data);
            free(shape);
            free(strides);
            if (wc != weight) ocean_tensor_release(wc);
            if (ic != indices) ocean_tensor_release(ic);
            free(device);
            free(indices_device);
            ocean_tensor_fail("Embedding token id is out of range");
        }

        size_t row = (size_t)token;
        for (size_t feature = 0; feature < dim; ++feature) {
            data[i * dim + feature] =
                ocean_tensor_get_flat_f32(
                    wc,
                    row * dim + feature
                );
        }
    }

    ocean_tensor_handle_t cpu = ocean_tensor_from_cpu_strided(
        data,
        shape,
        strides,
        output_rank,
        "float32",
        "cpu"
    );
    ocean_tensor_handle_t result = cpu;

    if (strcmp(device, "cpu") != 0) {
        result = ocean_tensor_to(cpu, device);
        ocean_tensor_release(cpu);
    }

    if (wc != weight) ocean_tensor_release(wc);
    if (ic != indices) ocean_tensor_release(ic);
    free(device);
    free(indices_device);
    free(data);
    free(shape);
    free(strides);
    return result;
}

ocean_tensor_handle_t ocean_autograd_embedding(
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t indices
) {
    ocean_tensor_handle_t result =
        ocean_autograd_embedding_forward_v04(weight, indices);

    ocean_autograd_meta *weight_meta =
        ocean_autograd_find(weight);
    if (!weight_meta || !weight_meta->requires_grad) {
        return result;
    }

    ocean_autograd_node *node =
        ocean_autograd_node_new(OCEAN_AUTOGRAD_EMBEDDING);
    node->left = weight_meta;
    node->saved_right = ocean_tensor_copy(indices);
    ocean_autograd_attach(result, node);
    return result;
}

static ocean_tensor_handle_t ocean_autograd_cross_entropy_forward_v04(
    ocean_tensor_handle_t logits,
    ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
) {
    ocean_autograd_require_float32(logits);
    ocean_autograd_require_int64_v04(
        targets,
        "CrossEntropyLoss targets must be Tensor[int64]"
    );

    int logits_rank = ocean_tensor_ndim(logits);
    int targets_rank = ocean_tensor_ndim(targets);

    if (logits_rank < 2 || targets_rank != logits_rank - 1) {
        ocean_tensor_fail(
            "CrossEntropyLoss expects logits [...,V] and targets [...]"
        );
    }

    for (int axis = 0; axis < targets_rank; ++axis) {
        if (
            ocean_tensor_shape(logits, axis)
            != ocean_tensor_shape(targets, axis)
        ) {
            ocean_tensor_fail(
                "CrossEntropyLoss target shape must match logits prefix"
            );
        }
    }

    size_t vocab =
        (size_t)ocean_tensor_shape(logits, logits_rank - 1);
    size_t examples = ocean_tensor_size(targets);

    if (!vocab || !examples) {
        ocean_tensor_fail(
            "CrossEntropyLoss requires non-empty vocab and targets"
        );
    }

    char *device = ocean_tensor_device(logits);
    ocean_tensor_handle_t lc = strcmp(device, "cpu") == 0
        ? logits
        : ocean_tensor_to(logits, "cpu");

    char *target_device = ocean_tensor_device(targets);
    ocean_tensor_handle_t tc = strcmp(target_device, "cpu") == 0
        ? targets
        : ocean_tensor_to(targets, "cpu");

    size_t total = ocean_tensor_size(lc);
    float *probabilities =
        (float *)malloc(total * sizeof(float));
    if (!probabilities) {
        if (lc != logits) ocean_tensor_release(lc);
        if (tc != targets) ocean_tensor_release(tc);
        free(device);
        free(target_device);
        ocean_tensor_fail(
            "out of memory in CrossEntropyLoss forward"
        );
    }

    double total_loss = 0.0;

    for (size_t example = 0; example < examples; ++example) {
        size_t base = example * vocab;
        float maximum = -INFINITY;

        for (size_t cls = 0; cls < vocab; ++cls) {
            float value =
                ocean_tensor_get_flat_f32(lc, base + cls);
            if (value > maximum) maximum = value;
        }

        double denominator = 0.0;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float value =
                ocean_tensor_get_flat_f32(lc, base + cls);
            float exponential = expf(value - maximum);
            probabilities[base + cls] = exponential;
            denominator += (double)exponential;
        }

        int64_t target =
            ocean_tensor_get_flat_i64(tc, example);
        if (target < 0 || (uint64_t)target >= (uint64_t)vocab) {
            free(probabilities);
            if (lc != logits) ocean_tensor_release(lc);
            if (tc != targets) ocean_tensor_release(tc);
            free(device);
            free(target_device);
            ocean_tensor_fail(
                "CrossEntropyLoss target is out of range"
            );
        }

        float target_logit =
            ocean_tensor_get_flat_f32(
                lc,
                base + (size_t)target
            );
        total_loss +=
            log(denominator)
            + (double)maximum
            - (double)target_logit;

        for (size_t cls = 0; cls < vocab; ++cls) {
            probabilities[base + cls] =
                (float)(
                    (double)probabilities[base + cls]
                    / denominator
                );
        }
    }

    size_t rank = (size_t)logits_rank;
    size_t *shape = (size_t *)malloc(rank * sizeof(size_t));
    size_t *strides = (size_t *)malloc(rank * sizeof(size_t));

    if (!shape || !strides) {
        free(shape);
        free(strides);
        free(probabilities);
        if (lc != logits) ocean_tensor_release(lc);
        if (tc != targets) ocean_tensor_release(tc);
        free(device);
        free(target_device);
        ocean_tensor_fail(
            "out of memory storing CrossEntropyLoss probabilities"
        );
    }

    for (size_t axis = 0; axis < rank; ++axis) {
        shape[axis] =
            (size_t)ocean_tensor_shape(lc, (int)axis);
    }
    ocean_autograd_contiguous_strides_v04(
        shape,
        rank,
        strides
    );

    ocean_tensor_handle_t probabilities_cpu =
        ocean_tensor_from_cpu_strided(
            probabilities,
            shape,
            strides,
            rank,
            "float32",
            "cpu"
        );
    ocean_tensor_handle_t probabilities_result =
        probabilities_cpu;

    if (strcmp(device, "cpu") != 0) {
        probabilities_result =
            ocean_tensor_to(probabilities_cpu, device);
        ocean_tensor_release(probabilities_cpu);
    }

    ocean_tensor_handle_t loss_cpu =
        ocean_tensor_zeros(1, 1, "cpu");
    ocean_tensor_fill(
        loss_cpu,
        total_loss / (double)examples
    );
    ocean_tensor_handle_t loss = loss_cpu;

    if (strcmp(device, "cpu") != 0) {
        loss = ocean_tensor_to(loss_cpu, device);
        ocean_tensor_release(loss_cpu);
    }

    *probabilities_out = probabilities_result;

    free(shape);
    free(strides);
    free(probabilities);
    if (lc != logits) ocean_tensor_release(lc);
    if (tc != targets) ocean_tensor_release(tc);
    free(device);
    free(target_device);
    return loss;
}

ocean_tensor_handle_t ocean_autograd_cross_entropy(
    ocean_tensor_handle_t logits,
    ocean_tensor_handle_t targets
) {
    ocean_tensor_handle_t probabilities = NULL;
    ocean_tensor_handle_t result =
        ocean_autograd_cross_entropy_forward_v04(
            logits,
            targets,
            &probabilities
        );

    ocean_autograd_meta *logits_meta =
        ocean_autograd_find(logits);

    if (!logits_meta || !logits_meta->requires_grad) {
        ocean_tensor_release(probabilities);
        return result;
    }

    ocean_autograd_node *node =
        ocean_autograd_node_new(
            OCEAN_AUTOGRAD_CROSS_ENTROPY
        );
    node->left = logits_meta;
    node->saved_left = probabilities;
    node->saved_right = ocean_tensor_copy(targets);
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
            if (node->left) { ocean_tensor_handle_t rt=ocean_tensor_transpose_dims(node->saved_right,-2,-1); ocean_tensor_handle_t raw=ocean_tensor_matmul(upstream,rt); ocean_tensor_handle_t red=ocean_autograd_sum_to_meta(raw,node->left); ocean_tensor_release(rt); ocean_tensor_release(raw); ocean_autograd_accumulate(node->left,red); }
            if (node->right) { ocean_tensor_handle_t lt=ocean_tensor_transpose_dims(node->saved_left,-2,-1); ocean_tensor_handle_t raw=ocean_tensor_matmul(lt,upstream); ocean_tensor_handle_t red=ocean_autograd_sum_to_meta(raw,node->right); ocean_tensor_release(lt); ocean_tensor_release(raw); ocean_autograd_accumulate(node->right,red); }
            break;
        }

        case OCEAN_AUTOGRAD_TRANSPOSE: { if(node->left)ocean_autograd_accumulate(node->left,ocean_tensor_transpose_dims(upstream,0,1)); break; }
        case OCEAN_AUTOGRAD_TRANSPOSE_DIMS: { if(node->left)ocean_autograd_accumulate(node->left,ocean_tensor_transpose_dims(upstream,node->dim0,node->dim1)); break; }
        case OCEAN_AUTOGRAD_EMBEDDING: {
            if (node->left) {
                size_t vocab = node->left->shape[0];
                size_t dim = node->left->shape[1];
                size_t index_count =
                    ocean_tensor_size(node->saved_right);

                char *device =
                    ocean_tensor_device(upstream);
                ocean_tensor_handle_t gc =
                    strcmp(device, "cpu") == 0
                    ? upstream
                    : ocean_tensor_to(upstream, "cpu");

                char *indices_device =
                    ocean_tensor_device(node->saved_right);
                ocean_tensor_handle_t ic =
                    strcmp(indices_device, "cpu") == 0
                    ? node->saved_right
                    : ocean_tensor_to(
                        node->saved_right,
                        "cpu"
                    );

                size_t total = vocab * dim;
                float *data = total
                    ? (float *)calloc(total, sizeof(float))
                    : NULL;
                size_t shape[2] = {vocab, dim};
                size_t strides[2] = {dim, 1};

                if (total && !data) {
                    if (gc != upstream) ocean_tensor_release(gc);
                    if (ic != node->saved_right) {
                        ocean_tensor_release(ic);
                    }
                    free(device);
                    free(indices_device);
                    ocean_tensor_fail(
                        "out of memory in Embedding backward"
                    );
                }

                for (size_t i = 0; i < index_count; ++i) {
                    int64_t token =
                        ocean_tensor_get_flat_i64(ic, i);
                    size_t row = (size_t)token;

                    for (size_t feature = 0; feature < dim; ++feature) {
                        data[row * dim + feature] +=
                            ocean_tensor_get_flat_f32(
                                gc,
                                i * dim + feature
                            );
                    }
                }

                ocean_tensor_handle_t cpu =
                    ocean_tensor_from_cpu_strided(
                        data,
                        shape,
                        strides,
                        2,
                        "float32",
                        "cpu"
                    );
                ocean_tensor_handle_t contribution = cpu;

                if (strcmp(device, "cpu") != 0) {
                    contribution =
                        ocean_tensor_to(cpu, device);
                    ocean_tensor_release(cpu);
                }

                free(data);
                if (gc != upstream) ocean_tensor_release(gc);
                if (ic != node->saved_right) {
                    ocean_tensor_release(ic);
                }
                free(device);
                free(indices_device);

                ocean_autograd_accumulate(
                    node->left,
                    contribution
                );
            }
            break;
        }

        case OCEAN_AUTOGRAD_CROSS_ENTROPY: {
            if (node->left) {
                ocean_tensor_handle_t probabilities =
                    node->saved_left;
                ocean_tensor_handle_t targets =
                    node->saved_right;

                int rank = ocean_tensor_ndim(probabilities);
                size_t vocab =
                    (size_t)ocean_tensor_shape(
                        probabilities,
                        rank - 1
                    );
                size_t examples =
                    ocean_tensor_size(targets);
                size_t total =
                    ocean_tensor_size(probabilities);

                char *device =
                    ocean_tensor_device(probabilities);
                ocean_tensor_handle_t pc =
                    strcmp(device, "cpu") == 0
                    ? probabilities
                    : ocean_tensor_to(
                        probabilities,
                        "cpu"
                    );

                char *target_device =
                    ocean_tensor_device(targets);
                ocean_tensor_handle_t tc =
                    strcmp(target_device, "cpu") == 0
                    ? targets
                    : ocean_tensor_to(targets, "cpu");

                float *data = total
                    ? (float *)malloc(
                        total * sizeof(float)
                    )
                    : NULL;
                size_t *shape = (size_t *)malloc(
                    (size_t)rank * sizeof(size_t)
                );
                size_t *strides = (size_t *)malloc(
                    (size_t)rank * sizeof(size_t)
                );

                if (
                    (total && !data)
                    || !shape
                    || !strides
                ) {
                    free(data);
                    free(shape);
                    free(strides);
                    if (pc != probabilities) {
                        ocean_tensor_release(pc);
                    }
                    if (tc != targets) {
                        ocean_tensor_release(tc);
                    }
                    free(device);
                    free(target_device);
                    ocean_tensor_fail(
                        "out of memory in CrossEntropyLoss backward"
                    );
                }

                for (int axis = 0; axis < rank; ++axis) {
                    shape[(size_t)axis] =
                        (size_t)ocean_tensor_shape(
                            pc,
                            axis
                        );
                }
                ocean_autograd_contiguous_strides_v04(
                    shape,
                    (size_t)rank,
                    strides
                );

                float upstream_scale =
                    ocean_tensor_get_flat_f32(upstream, 0)
                    / (float)examples;

                for (size_t example = 0; example < examples; ++example) {
                    int64_t target =
                        ocean_tensor_get_flat_i64(
                            tc,
                            example
                        );
                    size_t base = example * vocab;

                    for (size_t cls = 0; cls < vocab; ++cls) {
                        float gradient =
                            ocean_tensor_get_flat_f32(
                                pc,
                                base + cls
                            );
                        if (cls == (size_t)target) {
                            gradient -= 1.0f;
                        }
                        data[base + cls] =
                            gradient * upstream_scale;
                    }
                }

                ocean_tensor_handle_t cpu =
                    ocean_tensor_from_cpu_strided(
                        data,
                        shape,
                        strides,
                        (size_t)rank,
                        "float32",
                        "cpu"
                    );
                ocean_tensor_handle_t contribution = cpu;

                if (strcmp(device, "cpu") != 0) {
                    contribution =
                        ocean_tensor_to(cpu, device);
                    ocean_tensor_release(cpu);
                }

                free(data);
                free(shape);
                free(strides);
                if (pc != probabilities) {
                    ocean_tensor_release(pc);
                }
                if (tc != targets) {
                    ocean_tensor_release(tc);
                }
                free(device);
                free(target_device);

                ocean_autograd_accumulate(
                    node->left,
                    contribution
                );
            }
            break;
        }

        case OCEAN_AUTOGRAD_PERMUTE: {
            if (node->left) {
                int *inverse = node->axes_count
                    ? (int *)malloc(node->axes_count * sizeof(int))
                    : NULL;

                if (node->axes_count && !inverse) {
                    ocean_tensor_fail(
                        "out of memory in Tensor.permute backward"
                    );
                }

                for (size_t i = 0; i < node->axes_count; ++i) {
                    inverse[(size_t)node->axes[i]] = (int)i;
                }

                ocean_tensor_handle_t contribution =
                    ocean_tensor_permute(
                        upstream,
                        inverse,
                        node->axes_count
                    );
                free(inverse);
                ocean_autograd_accumulate(
                    node->left,
                    contribution
                );
            }
            break;
        }

        case OCEAN_AUTOGRAD_RESHAPE: { if(node->left){ocean_tensor_handle_t r=ocean_tensor_reshape(upstream,node->left->shape,node->left->ndim);ocean_autograd_accumulate(node->left,r);} break; }
        case OCEAN_AUTOGRAD_SUM_DIM:
        case OCEAN_AUTOGRAD_MEAN_DIM: { if(node->left){int ax=ocean_autograd_normalize_dim_v02(node->left,node->dim0);double sc=node->operation==OCEAN_AUTOGRAD_MEAN_DIM?1.0/(double)node->left->shape[(size_t)ax]:1.0;ocean_tensor_handle_t g=ocean_autograd_expand_reduction_v02(upstream,node->left,node->dim0,node->keepdim,sc);ocean_autograd_accumulate(node->left,g);} break; }


        case OCEAN_AUTOGRAD_EXP: {
            if (node->left) {
                ocean_tensor_handle_t contribution = ocean_tensor_binary(
                    upstream, node->saved_left, OCEAN_AUTOGRAD_MUL
                );
                ocean_autograd_accumulate(node->left, contribution);
            }
            break;
        }

        case OCEAN_AUTOGRAD_LOG: {
            if (node->left) {
                ocean_tensor_handle_t contribution = ocean_tensor_binary(
                    upstream, node->saved_left, OCEAN_AUTOGRAD_DIV
                );
                ocean_autograd_accumulate(node->left, contribution);
            }
            break;
        }

        case OCEAN_AUTOGRAD_SQRT: {
            if (node->left) {
                ocean_tensor_handle_t denominator = ocean_tensor_scalar(
                    node->saved_left, 2.0, OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t contribution = ocean_tensor_binary(
                    upstream, denominator, OCEAN_AUTOGRAD_DIV
                );
                ocean_tensor_release(denominator);
                ocean_autograd_accumulate(node->left, contribution);
            }
            break;
        }

        case OCEAN_AUTOGRAD_POW: {
            if (node->left) {
                ocean_tensor_handle_t power = ocean_autograd_unary_cpu_v03(
                    node->saved_left,
                    OCEAN_AUTOGRAD_POW,
                    node->scalar - 1.0
                );
                ocean_tensor_handle_t scaled = ocean_tensor_scalar(
                    power, node->scalar, OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_handle_t contribution = ocean_tensor_binary(
                    upstream, scaled, OCEAN_AUTOGRAD_MUL
                );
                ocean_tensor_release(power);
                ocean_tensor_release(scaled);
                ocean_autograd_accumulate(node->left, contribution);
            }
            break;
        }

        case OCEAN_AUTOGRAD_SOFTMAX: {
            if (node->left) {
                ocean_tensor_handle_t contribution =
                    ocean_autograd_softmax_backward_v03(
                        upstream, node->saved_left, node->dim0
                    );
                ocean_autograd_accumulate(node->left, contribution);
            }
            break;
        }

        case OCEAN_AUTOGRAD_LAYER_NORM: {
            if (node->left) {
                ocean_tensor_handle_t contribution =
                    ocean_autograd_layer_norm_backward_v03(
                        upstream,
                        node->saved_left,
                        NULL,
                        node->dim0,
                        node->scalar
                    );
                ocean_autograd_accumulate(node->left, contribution);
            }
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


    ocean_tensor_handle_t cpu = ocean_tensor_zeros(rows, cols, "cpu");

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
    /* SGD GPU/CPU device-aware v0.1 */
    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    if (!meta || !meta->requires_grad || !meta->leaf) {
        ocean_tensor_fail("SGD expects a leaf Parameter");
    }
    if (!meta->grad) return;

    ocean_autograd_require_float32(tensor);

    char *device = ocean_tensor_device(tensor);
    bool is_cpu = strcmp(device, "cpu") == 0;
    free(device);

    ocean_tensor_handle_t parameter_cpu = is_cpu
        ? tensor
        : ocean_tensor_to(tensor, "cpu");

    ocean_tensor_handle_t gradient_cpu = is_cpu
        ? meta->grad
        : ocean_tensor_to(meta->grad, "cpu");

    size_t *indices = (size_t *)calloc(meta->ndim, sizeof(size_t));
    if (!indices) {
        if (parameter_cpu != tensor) ocean_tensor_release(parameter_cpu);
        if (gradient_cpu != meta->grad) ocean_tensor_release(gradient_cpu);
        ocean_tensor_fail("out of memory in SGD");
    }

    size_t size = ocean_tensor_size(parameter_cpu);
    for (size_t linear = 0; linear < size; ++linear) {
        size_t remaining = linear;
        for (size_t axis = meta->ndim; axis-- > 0;) {
            size_t dim = meta->shape[axis];
            indices[axis] = dim ? remaining % dim : 0;
            remaining = dim ? remaining / dim : 0;
        }

        float parameter =
            ocean_tensor_get_flat_f32(parameter_cpu, linear);
        float gradient =
            ocean_tensor_get_flat_f32(gradient_cpu, linear);

        ocean_tensor_set_nd_f32(
            parameter_cpu,
            indices,
            meta->ndim,
            parameter - (float)learning_rate * gradient
        );
    }

    free(indices);

    if (!is_cpu) {
        ocean_tensor_copy_into(tensor, parameter_cpu);
        ocean_tensor_release(parameter_cpu);
        ocean_tensor_release(gradient_cpu);
    }
}

/* AdamW v0.1 */

typedef struct ocean_adamw_parameter_state {
    uint64_t tensor_identity;
    size_t size;
    float *first_moment;
    float *second_moment;
    struct ocean_adamw_parameter_state *next;
} ocean_adamw_parameter_state;

typedef struct ocean_adamw_optimizer_state {
    int id;
    int step;
    ocean_adamw_parameter_state *parameters;
    struct ocean_adamw_optimizer_state *next;
} ocean_adamw_optimizer_state;

static ocean_adamw_optimizer_state *ocean_adamw_states = NULL;
static int ocean_adamw_next_id = 1;
static bool ocean_adamw_shutdown_registered = false;

static void ocean_adamw_free_parameter_state(
    ocean_adamw_parameter_state *state
) {
    if (!state) return;
    free(state->first_moment);
    free(state->second_moment);
    free(state);
}

static void ocean_adamw_shutdown(void) {
    ocean_adamw_optimizer_state *state = ocean_adamw_states;

    while (state) {
        ocean_adamw_optimizer_state *next_state = state->next;
        ocean_adamw_parameter_state *parameter = state->parameters;

        while (parameter) {
            ocean_adamw_parameter_state *next_parameter = parameter->next;
            ocean_adamw_free_parameter_state(parameter);
            parameter = next_parameter;
        }

        free(state);
        state = next_state;
    }

    ocean_adamw_states = NULL;
}

static ocean_adamw_optimizer_state *ocean_adamw_find_state(int id) {
    for (
        ocean_adamw_optimizer_state *state = ocean_adamw_states;
        state;
        state = state->next
    ) {
        if (state->id == id) return state;
    }

    ocean_tensor_fail("AdamW optimizer state id is invalid");
    return NULL;
}

static ocean_adamw_parameter_state *ocean_adamw_get_parameter_state(
    ocean_adamw_optimizer_state *optimizer,
    ocean_tensor_handle_t tensor
) {
    uint64_t identity = ocean_tensor_identity(tensor);
    size_t size = ocean_tensor_size(tensor);

    for (
        ocean_adamw_parameter_state *state = optimizer->parameters;
        state;
        state = state->next
    ) {
        if (state->tensor_identity == identity) {
            if (state->size != size) {
                ocean_tensor_fail(
                    "AdamW Parameter size changed after optimizer creation"
                );
            }
            return state;
        }
    }

    ocean_adamw_parameter_state *state =
        (ocean_adamw_parameter_state *)calloc(1, sizeof(*state));

    if (!state) {
        ocean_tensor_fail("out of memory creating AdamW Parameter state");
    }

    state->tensor_identity = identity;
    state->size = size;
    state->first_moment = size
        ? (float *)calloc(size, sizeof(float))
        : NULL;
    state->second_moment = size
        ? (float *)calloc(size, sizeof(float))
        : NULL;

    if (
        size
        && (!state->first_moment || !state->second_moment)
    ) {
        ocean_adamw_free_parameter_state(state);
        ocean_tensor_fail("out of memory creating AdamW moments");
    }

    state->next = optimizer->parameters;
    optimizer->parameters = state;
    return state;
}

int ocean_autograd_adamw_create(void) {
    if (ocean_adamw_next_id <= 0) {
        ocean_tensor_fail("AdamW optimizer id space exhausted");
    }

    ocean_adamw_optimizer_state *state =
        (ocean_adamw_optimizer_state *)calloc(1, sizeof(*state));

    if (!state) {
        ocean_tensor_fail("out of memory creating AdamW optimizer");
    }

    state->id = ocean_adamw_next_id++;
    state->next = ocean_adamw_states;
    ocean_adamw_states = state;

    if (!ocean_adamw_shutdown_registered) {
        ocean_adamw_shutdown_registered = true;
        atexit(ocean_adamw_shutdown);
    }

    return state->id;
}

int ocean_autograd_adamw_begin_step(int state_id) {
    ocean_adamw_optimizer_state *state =
        ocean_adamw_find_state(state_id);

    if (state->step == INT32_MAX) {
        ocean_tensor_fail("AdamW step counter overflow");
    }

    state->step += 1;
    return state->step;
}

void ocean_autograd_adamw_step(
    int state_id,
    int step,
    ocean_tensor_handle_t tensor,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay
) {
    /* AdamW GPU/CPU device-aware v0.1 */
    if (step <= 0) ocean_tensor_fail("AdamW step must be positive");
    if (learning_rate < 0.0) ocean_tensor_fail("AdamW learning_rate must be non-negative");
    if (beta1 < 0.0 || beta1 >= 1.0) ocean_tensor_fail("AdamW beta1 must be in [0, 1)");
    if (beta2 < 0.0 || beta2 >= 1.0) ocean_tensor_fail("AdamW beta2 must be in [0, 1)");
    if (epsilon <= 0.0) ocean_tensor_fail("AdamW epsilon must be positive");
    if (weight_decay < 0.0) ocean_tensor_fail("AdamW weight_decay must be non-negative");

    ocean_adamw_optimizer_state *optimizer =
        ocean_adamw_find_state(state_id);

    if (optimizer->step != step) {
        ocean_tensor_fail("AdamW Parameter update used the wrong optimizer step");
    }

    ocean_autograd_meta *meta = ocean_autograd_find(tensor);
    if (!meta || !meta->requires_grad || !meta->leaf) {
        ocean_tensor_fail("AdamW expects a leaf Parameter");
    }
    if (!meta->grad) return;

    ocean_autograd_require_float32(tensor);

    ocean_adamw_parameter_state *parameter_state =
        ocean_adamw_get_parameter_state(optimizer, tensor);

    char *device = ocean_tensor_device(tensor);
    bool is_cpu = strcmp(device, "cpu") == 0;
    free(device);

    ocean_tensor_handle_t parameter_cpu = is_cpu
        ? tensor
        : ocean_tensor_to(tensor, "cpu");

    ocean_tensor_handle_t gradient_cpu = is_cpu
        ? meta->grad
        : ocean_tensor_to(meta->grad, "cpu");

    size_t size = ocean_tensor_size(parameter_cpu);
    size_t *indices = meta->ndim
        ? (size_t *)calloc(meta->ndim, sizeof(size_t))
        : NULL;

    if (meta->ndim && !indices) {
        if (parameter_cpu != tensor) ocean_tensor_release(parameter_cpu);
        if (gradient_cpu != meta->grad) ocean_tensor_release(gradient_cpu);
        ocean_tensor_fail("out of memory in AdamW");
    }

    double bias_correction1 = 1.0 - pow(beta1, (double)step);
    double bias_correction2 = 1.0 - pow(beta2, (double)step);

    if (bias_correction1 <= 0.0 || bias_correction2 <= 0.0) {
        free(indices);
        if (parameter_cpu != tensor) ocean_tensor_release(parameter_cpu);
        if (gradient_cpu != meta->grad) ocean_tensor_release(gradient_cpu);
        ocean_tensor_fail("AdamW bias correction became invalid");
    }

    for (size_t linear = 0; linear < size; ++linear) {
        float parameter =
            ocean_tensor_get_flat_f32(parameter_cpu, linear);
        float gradient =
            ocean_tensor_get_flat_f32(gradient_cpu, linear);

        float first_moment = (float)(
            beta1 * (double)parameter_state->first_moment[linear]
            + (1.0 - beta1) * (double)gradient
        );

        float second_moment = (float)(
            beta2 * (double)parameter_state->second_moment[linear]
            + (1.0 - beta2) * (double)gradient * (double)gradient
        );

        parameter_state->first_moment[linear] = first_moment;
        parameter_state->second_moment[linear] = second_moment;

        double first_unbiased =
            (double)first_moment / bias_correction1;
        double second_unbiased =
            (double)second_moment / bias_correction2;

        double adaptive_update =
            first_unbiased / (sqrt(second_unbiased) + epsilon);

        double updated =
            (double)parameter
            - learning_rate * weight_decay * (double)parameter
            - learning_rate * adaptive_update;

        size_t remaining = linear;
        for (size_t axis = meta->ndim; axis-- > 0;) {
            size_t dim = meta->shape[axis];
            indices[axis] = dim ? remaining % dim : 0;
            remaining = dim ? remaining / dim : 0;
        }

        ocean_tensor_set_nd_f32(
            parameter_cpu,
            indices,
            meta->ndim,
            (float)updated
        );
    }

    free(indices);

    if (!is_cpu) {
        ocean_tensor_copy_into(tensor, parameter_cpu);
        ocean_tensor_release(parameter_cpu);
        ocean_tensor_release(gradient_cpu);
    }
}
