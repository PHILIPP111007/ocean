#include "std/tensor/tensor_runtime.h"
#include "std/tensor/tensor_backend.h"

#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include <CL/cl.h>
#endif
#ifdef OCEAN_TENSOR_ENABLE_CUDA
#include "std/tensor/tensor_cuda_backend.h"
#endif

typedef enum ocean_tensor_dtype {
    OCEAN_TENSOR_BOOL,
    OCEAN_TENSOR_INT8,
    OCEAN_TENSOR_INT16,
    OCEAN_TENSOR_INT32,
    OCEAN_TENSOR_INT64,
    OCEAN_TENSOR_UINT8,
    OCEAN_TENSOR_UINT16,
    OCEAN_TENSOR_UINT32,
    OCEAN_TENSOR_UINT64,
    OCEAN_TENSOR_FLOAT16,
    OCEAN_TENSOR_FLOAT32,
    OCEAN_TENSOR_FLOAT64,
} ocean_tensor_dtype;

enum {
    OCEAN_TENSOR_ADD = 0,
    OCEAN_TENSOR_SUB = 1,
    OCEAN_TENSOR_MUL = 2,
    OCEAN_TENSOR_DIV = 3,
};

struct ocean_tensor_handle {
    uint64_t identity;
    ocean_tensor_backend_kind device;
    ocean_tensor_dtype dtype;
    size_t item_size;
    size_t ndim;
    size_t size;
    size_t *shape;
    size_t *strides;
    void *cpu_data;
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    cl_mem gpu_data;
#endif
#ifdef OCEAN_TENSOR_ENABLE_CUDA
    void *cuda_data;
#endif
};

#define OCEAN_TENSOR_CPU OCEAN_TENSOR_BACKEND_CPU
#define OCEAN_TENSOR_GPU OCEAN_TENSOR_BACKEND_OPENCL

static uint64_t ocean_tensor_next_identity = 1;



static const ocean_tensor_backend_ops *ocean_tensor_backend_for_device(
    ocean_tensor_backend_kind device
);
static ocean_tensor_backend_kind ocean_tensor_select_gpu_backend(void);
static ocean_tensor_handle_t ocean_tensor_matmul_cpu(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
static ocean_tensor_handle_t ocean_tensor_matmul_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static ocean_tensor_handle_t ocean_tensor_matmul_opencl_batched(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
);
#endif
static ocean_tensor_handle_t ocean_tensor_binary_cpu(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_binary_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
);
static void ocean_tensor_broadcast_shape(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    size_t **shape_out,
    size_t *ndim_out
);
static ocean_tensor_handle_t ocean_tensor_scalar_cpu(
    const ocean_tensor_handle_t tensor,
    double scalar,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_scalar_opencl(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
);
static ocean_tensor_handle_t ocean_tensor_ternary_quantize_cpu(
    const ocean_tensor_handle_t tensor
);
static ocean_tensor_handle_t ocean_tensor_gelu_cpu(
    const ocean_tensor_handle_t tensor
);
static ocean_tensor_handle_t ocean_tensor_gelu_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t input
);
static ocean_tensor_handle_t ocean_tensor_embedding_forward_cpu(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices
);
static ocean_tensor_handle_t ocean_tensor_embedding_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
);
static ocean_tensor_handle_t ocean_tensor_cross_entropy_forward_cpu(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
);
static ocean_tensor_handle_t ocean_tensor_cross_entropy_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets
);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static void ocean_tensor_opencl_ternary_quantize(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output
);
static void ocean_tensor_opencl_embedding_forward(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t output,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
);
static void ocean_tensor_opencl_embedding_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
);
static void ocean_tensor_opencl_cross_entropy_forward(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t row_losses,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
);
static void ocean_tensor_opencl_cross_entropy_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
);
static void ocean_tensor_opencl_permute(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output,
    const int *axes
);
static void ocean_tensor_opencl_gelu(
    const ocean_tensor_handle_t first,
    const ocean_tensor_handle_t second,
    ocean_tensor_handle_t output,
    int key
);
static void ocean_tensor_opencl_ternary_pack(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t output,
    double scale,
    bool transpose
);
#endif
static void ocean_tensor_write_scalar(
    const ocean_tensor_handle_t tensor,
    size_t index,
    long double value
);
static uint16_t ocean_tensor_float_to_half(float value);

static void ocean_tensor_fill_cpu(ocean_tensor_handle_t tensor, double value);
static void ocean_tensor_fill_opencl(ocean_tensor_handle_t tensor, double value);

static ocean_tensor_handle_t ocean_tensor_restore_device(
    const ocean_tensor_handle_t source,
    ocean_tensor_handle_t cpu_result
);

static ocean_tensor_handle_t ocean_tensor_permute_cpu(
    const ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
);

_Noreturn void ocean_tensor_fail(const char *message) {
    fprintf(stderr, "Ocean Tensor error: %s\n", message);
    exit(EXIT_FAILURE);
}

#ifdef OCEAN_TENSOR_ENABLE_CUDA
static bool ocean_tensor_cuda_set_scalar_fast(
    ocean_tensor_handle_t tensor,
    size_t index,
    long double value
) {
    switch (tensor->dtype) {
        case OCEAN_TENSOR_INT32:
            ocean_cuda_set_i32(tensor->cuda_data, index, (int)value);
            return true;
        case OCEAN_TENSOR_INT64:
            ocean_cuda_set_i64(tensor->cuda_data, index, (int64_t)value);
            return true;
        case OCEAN_TENSOR_FLOAT32:
            ocean_cuda_set_f32(tensor->cuda_data, index, (float)value);
            return true;
        default:
            return false;
    }
}
#endif

void ocean_tensor_validate_list_length(size_t actual, size_t expected) {
    if (actual != expected) {
        ocean_tensor_fail("Tensor.from_list requires rectangular lists");
    }
}

static int ocean_tensor_parse_device(const char *device) {
    if (device && strcmp(device, "cpu") == 0) return OCEAN_TENSOR_CPU;
    if (device && strcmp(device, "gpu") == 0) {
        return (int)ocean_tensor_select_gpu_backend();
    }
    /* Internal backend-preserving transfers use these names.  The public
       Ocean API continues to document only cpu/gpu. */
    if (device && strcmp(device, "opencl") == 0) {
        return OCEAN_TENSOR_BACKEND_OPENCL;
    }
    if (device && (strcmp(device, "cuda") == 0 ||
                   strcmp(device, "nvidia") == 0)) {
        return OCEAN_TENSOR_BACKEND_CUDA;
    }
    ocean_tensor_fail(
        "device must be \"cpu\" or \"gpu\" (internal: opencl/cuda)"
    );
    return OCEAN_TENSOR_CPU;
}

static ocean_tensor_dtype ocean_tensor_parse_dtype(const char *name) {
    if (!name || strcmp(name, "float32") == 0) {
        return OCEAN_TENSOR_FLOAT32;
    }
    if (strcmp(name, "bool") == 0) return OCEAN_TENSOR_BOOL;
    if (strcmp(name, "int8") == 0 || strcmp(name, "int8_t") == 0) {
        return OCEAN_TENSOR_INT8;
    }
    if (strcmp(name, "int16") == 0 || strcmp(name, "int16_t") == 0) {
        return OCEAN_TENSOR_INT16;
    }
    if (strcmp(name, "int") == 0 || strcmp(name, "int32") == 0) {
        return OCEAN_TENSOR_INT32;
    }
    if (strcmp(name, "int32_t") == 0) return OCEAN_TENSOR_INT32;
    if (strcmp(name, "int64") == 0 || strcmp(name, "int64_t") == 0) {
        return OCEAN_TENSOR_INT64;
    }
    if (strcmp(name, "uint8") == 0 || strcmp(name, "uint8_t") == 0) {
        return OCEAN_TENSOR_UINT8;
    }
    if (strcmp(name, "uint16") == 0 || strcmp(name, "uint16_t") == 0) {
        return OCEAN_TENSOR_UINT16;
    }
    if (strcmp(name, "uint") == 0 || strcmp(name, "uint32") == 0) {
        return OCEAN_TENSOR_UINT32;
    }
    if (strcmp(name, "uint32_t") == 0) return OCEAN_TENSOR_UINT32;
    if (strcmp(name, "uint64") == 0 || strcmp(name, "uint64_t") == 0) {
        return OCEAN_TENSOR_UINT64;
    }
    if (strcmp(name, "size_t") == 0 || strcmp(name, "uintptr_t") == 0) {
        return OCEAN_TENSOR_UINT64;
    }
    if (strcmp(name, "intptr_t") == 0) return OCEAN_TENSOR_INT64;
    if (strcmp(name, "float16") == 0) return OCEAN_TENSOR_FLOAT16;
    if (strcmp(name, "float") == 0 ||
        strcmp(name, "float64") == 0 ||
        strcmp(name, "double") == 0) {
        return OCEAN_TENSOR_FLOAT64;
    }
    ocean_tensor_fail("unsupported Tensor dtype");
    return OCEAN_TENSOR_FLOAT32;
}

static size_t ocean_tensor_dtype_size(ocean_tensor_dtype dtype) {
    switch (dtype) {
        case OCEAN_TENSOR_BOOL:
        case OCEAN_TENSOR_INT8:
        case OCEAN_TENSOR_UINT8:
            return 1;
        case OCEAN_TENSOR_INT16:
        case OCEAN_TENSOR_UINT16:
        case OCEAN_TENSOR_FLOAT16:
            return 2;
        case OCEAN_TENSOR_INT32:
        case OCEAN_TENSOR_UINT32:
        case OCEAN_TENSOR_FLOAT32:
            return 4;
        case OCEAN_TENSOR_INT64:
        case OCEAN_TENSOR_UINT64:
        case OCEAN_TENSOR_FLOAT64:
            return 8;
    }
    ocean_tensor_fail("invalid Tensor dtype");
    return 0;
}

static size_t ocean_tensor_elements_from_shape(const size_t *shape, size_t ndim) {
    if (!shape || ndim == 0) ocean_tensor_fail("Tensor must have at least one dimension");
    size_t elements = 1;
    for (size_t axis = 0; axis < ndim; ++axis) {
        if (shape[axis] != 0 && elements > SIZE_MAX / shape[axis]) {
            ocean_tensor_fail("Tensor shape is too large");
        }
        elements *= shape[axis];
    }
    return elements;
}

static size_t ocean_tensor_bytes(const ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("invalid null Tensor handle");
    if (tensor->size != 0 && tensor->item_size > SIZE_MAX / tensor->size) {
        ocean_tensor_fail("Tensor allocation is too large");
    }
    return tensor->size * tensor->item_size;
}

static ocean_tensor_handle_t ocean_tensor_alloc(
    const size_t *shape,
    size_t ndim,
    ocean_tensor_dtype dtype,
    int device
) {
    ocean_tensor_handle_t tensor =
        (ocean_tensor_handle_t)calloc(1, sizeof(*tensor));
    if (!tensor) ocean_tensor_fail("out of memory allocating Tensor handle");

    tensor->identity = ocean_tensor_next_identity++;
    if (ocean_tensor_next_identity == 0) {
        ocean_tensor_fail("Tensor identity counter overflow");
    }

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->item_size = ocean_tensor_dtype_size(dtype);
    tensor->size = ocean_tensor_elements_from_shape(shape, ndim);
    if (ndim > SIZE_MAX / sizeof(size_t)) {
        ocean_tensor_fail("Tensor metadata is too large");
    }
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        ocean_tensor_fail("out of memory allocating Tensor metadata");
    }
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->strides[ndim - 1] = 1;
    for (size_t axis = ndim - 1; axis > 0; --axis) {
        if (tensor->strides[axis] != 0 &&
            tensor->shape[axis] > SIZE_MAX / tensor->strides[axis]) {
            free(tensor->shape);
            free(tensor->strides);
            free(tensor);
            ocean_tensor_fail("Tensor strides are too large");
        }
        tensor->strides[axis - 1] =
            tensor->strides[axis] * tensor->shape[axis];
    }
    tensor->device = device;
    return tensor;
}

static ocean_tensor_handle_t ocean_tensor_alloc_zeros(
    const size_t *shape,
    size_t ndim,
    ocean_tensor_dtype dtype,
    int device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc(shape, ndim, dtype, device);
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device((ocean_tensor_backend_kind)device);
    backend->allocate(tensor);
    backend->zero(tensor);
    return tensor;
}

static ocean_tensor_handle_t ocean_tensor_alloc_uninitialized(
    const size_t *shape,
    size_t ndim,
    ocean_tensor_dtype dtype,
    int device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc(shape, ndim, dtype, device);
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device((ocean_tensor_backend_kind)device);
    backend->allocate(tensor);
    return tensor;
}


#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include "std/tensor/tensor_opencl_kernels.inc"
#include "std/tensor/tensor_opencl_runtime.inc"
#endif

static void ocean_tensor_cpu_allocate(ocean_tensor_handle_t tensor) {
    size_t bytes = ocean_tensor_bytes(tensor);
    tensor->cpu_data = bytes ? malloc(bytes) : NULL;
    if (bytes && !tensor->cpu_data) {
        ocean_tensor_fail("out of memory allocating CPU Tensor");
    }
}

static void ocean_tensor_cpu_zero(ocean_tensor_handle_t tensor) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes) memset(tensor->cpu_data, 0, bytes);
}

static void ocean_tensor_cpu_copy(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    size_t bytes = ocean_tensor_bytes(source);
    if (bytes) memcpy(destination->cpu_data, source->cpu_data, bytes);
}

static void ocean_tensor_cpu_read(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes) memcpy(host_data, tensor->cpu_data, bytes);
}

static void ocean_tensor_cpu_write(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes) memcpy(tensor->cpu_data, host_data, bytes);
}

static void ocean_tensor_cpu_release(ocean_tensor_handle_t tensor) {
    free(tensor->cpu_data);
    tensor->cpu_data = NULL;
#include "std/tensor/tensor_opencl_memory.inc"
    const size_t *shape,
    size_t ndim,
    const char *dtype,
    const char *device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc_zeros(
        shape, ndim, ocean_tensor_parse_dtype(dtype),
        ocean_tensor_parse_device(device)
    );
    return tensor;
}

ocean_tensor_handle_t ocean_tensor_empty_nd(
    const size_t *shape,
    size_t ndim,
    const char *dtype,
    const char *device
) {
    if (!shape || ndim == 0 || !dtype || !device) {
        ocean_tensor_fail("Tensor.empty requires non-empty shape and device");
    }
    return ocean_tensor_alloc_uninitialized(
        shape, ndim, ocean_tensor_parse_dtype(dtype),
        ocean_tensor_parse_device(device)
    );
}

ocean_tensor_handle_t ocean_tensor_zeros(
    int rows,
    int cols,
    const char *device
) {
    if (rows < 0 || cols < 0) ocean_tensor_fail("Tensor dimensions must be non-negative");
    size_t shape[2] = {(size_t)rows, (size_t)cols};
    ocean_tensor_handle_t tensor =
        ocean_tensor_alloc_zeros(shape, 2, OCEAN_TENSOR_FLOAT32,
                                 ocean_tensor_parse_device(device));
    return tensor;
}

ocean_tensor_handle_t ocean_tensor_from_cpu_strided(
    const void *data,
    const size_t *shape,
    const size_t *strides,
    size_t ndim,
    const char *dtype,
    const char *device
) {
    if (!shape || !strides) {
        ocean_tensor_fail("Tensor metadata cannot be null");
    }

    ocean_tensor_dtype parsed_dtype = ocean_tensor_parse_dtype(dtype);
    ocean_tensor_handle_t host = ocean_tensor_alloc_uninitialized(
        shape, ndim, parsed_dtype, OCEAN_TENSOR_CPU
    );

    if (host->size && !data) {
        ocean_tensor_release(host);
        ocean_tensor_fail("Tensor data cannot be null for a non-empty Tensor");
    }

    unsigned char *destination = (unsigned char *)host->cpu_data;
    const unsigned char *source = (const unsigned char *)data;
    size_t item_size = host->item_size;

    bool contiguous = true;
    size_t expected = 1;

    for (size_t axis = ndim; axis-- > 0;) {
        if (strides[axis] != expected) {
            contiguous = false;
            break;
        }
        if (shape[axis] != 0 && expected > SIZE_MAX / shape[axis]) {
            contiguous = false;
            break;
        }
        expected *= shape[axis];
    }

    if (host->size && contiguous) {
        memcpy(destination, source, ocean_tensor_bytes(host));
    } else {
        for (size_t linear = 0; linear < host->size; ++linear) {
            size_t remaining = linear;
            size_t source_offset = 0;

            for (size_t axis = ndim; axis-- > 0;) {
                size_t coordinate = shape[axis]
                    ? remaining % shape[axis]
                    : 0;
                remaining = shape[axis]
                    ? remaining / shape[axis]
                    : 0;
                source_offset += coordinate * strides[axis];
            }

            memcpy(
                destination + linear * item_size,
                source + source_offset * item_size,
                item_size
            );
        }
    }

    int target = ocean_tensor_parse_device(device);
    if (target == OCEAN_TENSOR_CPU) {
        return host;
    }

    ocean_tensor_handle_t result = ocean_tensor_to(host, device);
    ocean_tensor_release(host);
    return result;
}


static void ocean_tensor_fill_cpu(
    ocean_tensor_handle_t tensor,
    double value
) {
    if (tensor->size == 0) return;

    if (value == 0.0) {
        memset(tensor->cpu_data, 0, ocean_tensor_bytes(tensor));
        return;
    }

    size_t size = tensor->size;

#define OCEAN_FILL(type, converted) \
    do { \
        type v = (converted); \
        type *restrict data = (type *)tensor->cpu_data; \
        for (size_t i = 0; i < size; ++i) data[i] = v; \
    } while (0)

    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: OCEAN_FILL(bool, value != 0.0); break;
        case OCEAN_TENSOR_INT8: OCEAN_FILL(int8_t, (int8_t)value); break;
        case OCEAN_TENSOR_INT16: OCEAN_FILL(int16_t, (int16_t)value); break;
        case OCEAN_TENSOR_INT32: OCEAN_FILL(int32_t, (int32_t)value); break;
        case OCEAN_TENSOR_INT64: OCEAN_FILL(int64_t, (int64_t)value); break;
        case OCEAN_TENSOR_UINT8: OCEAN_FILL(uint8_t, (uint8_t)value); break;
        case OCEAN_TENSOR_UINT16: OCEAN_FILL(uint16_t, (uint16_t)value); break;
        case OCEAN_TENSOR_UINT32: OCEAN_FILL(uint32_t, (uint32_t)value); break;
        case OCEAN_TENSOR_UINT64: OCEAN_FILL(uint64_t, (uint64_t)value); break;
        case OCEAN_TENSOR_FLOAT16:
            OCEAN_FILL(uint16_t, ocean_tensor_float_to_half((float)value));
            break;
        case OCEAN_TENSOR_FLOAT32: OCEAN_FILL(float, (float)value); break;
        case OCEAN_TENSOR_FLOAT64: OCEAN_FILL(double, value); break;
    }

#undef OCEAN_FILL
}


ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("cannot copy a null Tensor");
    ocean_tensor_handle_t result = ocean_tensor_alloc(
        tensor->shape, tensor->ndim, tensor->dtype, tensor->device
    );
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device(tensor->device);
    backend->allocate(result);
    backend->copy(result, tensor);
    return result;
}

ocean_tensor_handle_t ocean_tensor_ternary_quantize(
    ocean_tensor_handle_t tensor
) {
    if (!tensor) ocean_tensor_fail("Tensor.ternary_quantize on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.ternary_quantize currently requires float32");
    }

    ocean_tensor_handle_t contiguous = NULL;
    const ocean_tensor_handle_t source = !ocean_tensor_is_contiguous(tensor)
        ? (contiguous = ocean_tensor_contiguous(tensor))
        : tensor;

    if (source->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t result = ocean_tensor_ternary_quantize_cpu(source);
        if (contiguous) ocean_tensor_release(contiguous);
        return result;
    }
    if (source->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        ocean_tensor_handle_t result = ocean_tensor_cuda_ternary_quantize(source);
        if (contiguous) ocean_tensor_release(contiguous);
        return result;
#else
        if (contiguous) ocean_tensor_release(contiguous);
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        source->shape, source->ndim, source->dtype, OCEAN_TENSOR_GPU
    );
    if (source->size != 0) {
        ocean_tensor_opencl_ternary_quantize(source, result);
    }
    if (contiguous) ocean_tensor_release(contiguous);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

double ocean_tensor_ternary_scale(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor.ternary_scale on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.ternary_scale currently requires float32");
    }
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double sum = 0.0L;
    const float *values = (const float *)cpu->cpu_data;
    for (size_t index = 0; index < cpu->size; ++index) {
        float value = values[index];
        sum += value < 0.0f ? -(long double)value : (long double)value;
    }
    double scale = cpu->size == 0
        ? 1.0e-8
        : (double)(sum / (long double)cpu->size);
    if (scale < 1.0e-8) scale = 1.0e-8;
    if (cpu != tensor) ocean_tensor_release(cpu);
    return scale;
}

ocean_tensor_handle_t ocean_tensor_ternary_pack(
    ocean_tensor_handle_t tensor,
    double scale,
    bool transpose
) {
    if (!tensor) ocean_tensor_fail("Tensor.ternary_pack on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32 || tensor->ndim != 2) {
        ocean_tensor_fail(
            "Tensor.ternary_pack requires a rank-2 float32 Tensor"
        );
    }
    if (!(scale > 0.0)) {
        ocean_tensor_fail("Tensor.ternary_pack scale must be positive");
    }
    ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(tensor)
        ? tensor : ocean_tensor_contiguous(tensor);
    size_t source_rows = contiguous->shape[0];
    size_t source_cols = contiguous->shape[1];
    size_t output_rows = transpose ? source_cols : source_rows;
    size_t packed_source = transpose ? source_rows : source_cols;
    size_t packed_cols = (packed_source + 15u) / 16u;
    size_t output_shape[2] = {output_rows, packed_cols};
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape, 2, OCEAN_TENSOR_INT32, contiguous->device
    );

    if (contiguous->device == OCEAN_TENSOR_CPU) {
        const float *values = (const float *)contiguous->cpu_data;
        int32_t *packed = (int32_t *)result->cpu_data;
        float threshold = (float)(0.5 * scale);
        for (size_t row = 0; row < output_rows; ++row) {
            for (size_t group = 0; group < packed_cols; ++group) {
                uint32_t word = 0u;
                for (size_t bit = 0; bit < 16; ++bit) {
                    size_t source_row = transpose
                        ? group * 16u + bit : row;
                    size_t source_col = transpose
                        ? row : group * 16u + bit;
                    uint32_t code = 0u;
                    if (source_row < source_rows && source_col < source_cols) {
                        float value = values[source_row * source_cols + source_col];
                        code = value > threshold
                            ? 1u : (value < -threshold ? 2u : 0u);
                    }
                    word |= code << (2u * (uint32_t)bit);
                }
                packed[row * packed_cols + group] = (int32_t)word;
            }
        }
    } else {
        if (contiguous->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
            ocean_tensor_cuda_ternary_pack(contiguous, result, scale, transpose);
            if (contiguous != tensor) ocean_tensor_release(contiguous);
            return result;
#else
            ocean_tensor_release(result);
            if (contiguous != tensor) ocean_tensor_release(contiguous);
            ocean_tensor_fail("CUDA backend was not compiled");
#endif
        }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        ocean_tensor_opencl_ternary_pack(
            contiguous, result, scale, transpose
        );
#else
        ocean_tensor_release(result);
        if (contiguous != tensor) ocean_tensor_release(contiguous);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }
    if (contiguous != tensor) ocean_tensor_release(contiguous);
    return result;
}

ocean_tensor_handle_t ocean_tensor_gelu(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor.gelu on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.gelu currently requires float32");
    }

    if (tensor->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_gelu_cpu(tensor);
    }
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        ocean_tensor_handle_t source = ocean_tensor_is_contiguous(tensor)
            ? tensor : ocean_tensor_contiguous(tensor);
        ocean_tensor_handle_t result = ocean_tensor_cuda_gelu(source);
        if (source != tensor) ocean_tensor_release(source);
        return result;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t source = ocean_tensor_is_contiguous(tensor)
        ? tensor : ocean_tensor_contiguous(tensor);
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        source->shape, source->ndim, source->dtype, OCEAN_TENSOR_GPU
    );
    if (source->size != 0) {
        ocean_tensor_opencl_gelu(
            source,
            NULL,
            result,
            OCEAN_TENSOR_OPENCL_KERNEL_GELU_FLOAT32
        );
    }
    if (source != tensor) ocean_tensor_release(source);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_gelu_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input
) {
    if (!upstream || !input) {
        ocean_tensor_fail("Tensor.gelu backward requires non-null tensors");
    }
    if (upstream->dtype != OCEAN_TENSOR_FLOAT32 ||
        input->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.gelu backward currently requires float32");
    }
    if (upstream->device != input->device ||
        upstream->size != input->size ||
        upstream->ndim != input->ndim) {
        ocean_tensor_fail("Tensor.gelu backward shape/device mismatch");
    }
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        if (upstream->shape[axis] != input->shape[axis]) {
            ocean_tensor_fail("Tensor.gelu backward shape mismatch");
        }
    }

    if (input->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_gelu_backward_cpu(upstream, input);
    }
    if (input->device == OCEAN_TENSOR_BACKEND_CUDA) {
        /* Backward is not part of inference. Keep the correctness-first CPU
           path until the CUDA autograd kernels are added. */
        ocean_tensor_handle_t cpu_upstream = ocean_tensor_to(upstream, "cpu");
        ocean_tensor_handle_t cpu_input = ocean_tensor_to(input, "cpu");
        ocean_tensor_handle_t cpu_result = ocean_tensor_gelu_backward_cpu(
            cpu_upstream, cpu_input
        );
        ocean_tensor_release(cpu_upstream);
        ocean_tensor_release(cpu_input);
        ocean_tensor_handle_t result = ocean_tensor_to(cpu_result, "cuda");
        ocean_tensor_release(cpu_result);
        return result;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t upstream_source = ocean_tensor_is_contiguous(upstream)
        ? upstream : ocean_tensor_contiguous(upstream);
    ocean_tensor_handle_t input_source = ocean_tensor_is_contiguous(input)
        ? input : ocean_tensor_contiguous(input);
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        input_source->shape, input_source->ndim,
        input_source->dtype, OCEAN_TENSOR_GPU
    );
    if (input_source->size != 0) {
        ocean_tensor_opencl_gelu(
            upstream_source,
            input_source,
            result,
            OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32
        );
    }
    if (upstream_source != upstream) ocean_tensor_release(upstream_source);
    if (input_source != input) ocean_tensor_release(input_source);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_embedding_forward(
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t indices
) {
    if (!weight || !indices) {
        ocean_tensor_fail("Embedding.forward requires non-null tensors");
    }
    if (weight->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Embedding weights must be Tensor[float32]");
    }
    if (indices->dtype != OCEAN_TENSOR_INT64) {
        ocean_tensor_fail("Embedding indices must be Tensor[int64]");
    }
    if (weight->ndim != 2 || indices->ndim < 1) {
        ocean_tensor_fail(
            "Embedding expects weight [V,D] and indices rank >= 1"
        );
    }

    size_t vocab = weight->shape[0];
    size_t dim = weight->shape[1];
    size_t count = indices->size;
    (void)vocab;
    if (dim != 0 && count > SIZE_MAX / dim) {
        ocean_tensor_fail("Embedding output is too large");
    }

    ocean_tensor_handle_t contiguous_weight =
        ocean_tensor_contiguous(weight);
    ocean_tensor_handle_t contiguous_indices =
        ocean_tensor_contiguous(indices);
    ocean_tensor_handle_t result = NULL;

    if (weight->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_indices = contiguous_indices;
        if (cpu_indices->device != OCEAN_TENSOR_CPU) {
            cpu_indices = ocean_tensor_to(cpu_indices, "cpu");
        }
        result = ocean_tensor_embedding_forward_cpu(
            contiguous_weight,
            cpu_indices
        );
        if (cpu_indices != contiguous_indices) ocean_tensor_release(cpu_indices);
    } else {
        if (weight->device == OCEAN_TENSOR_BACKEND_CUDA) {
            if (count > (size_t)INT_MAX || vocab > (size_t)INT_MAX ||
                dim > (size_t)INT_MAX || count * dim > (size_t)INT_MAX) {
                ocean_tensor_release(contiguous_weight);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail("CUDA Embedding dimensions exceed int32 indexing");
            }
            ocean_tensor_handle_t cpu_indices = ocean_tensor_to(
                contiguous_indices, "cpu"
            );
            const int64_t *index_values = (const int64_t *)cpu_indices->cpu_data;
            for (size_t index = 0; index < count; ++index) {
                if (index_values[index] < 0 || (uint64_t)index_values[index] >= vocab) {
                    ocean_tensor_release(cpu_indices);
                    ocean_tensor_release(contiguous_weight);
                    ocean_tensor_release(contiguous_indices);
                    ocean_tensor_fail("Embedding token id is out of range");
                }
            }
            ocean_tensor_release(cpu_indices);
            size_t output_shape[indices->ndim + 1];
            for (size_t axis = 0; axis < indices->ndim; ++axis) {
                output_shape[axis] = indices->shape[axis];
            }
            output_shape[indices->ndim] = dim;
            result = ocean_tensor_alloc_uninitialized(
                output_shape, indices->ndim + 1,
                OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_BACKEND_CUDA
            );
#ifdef OCEAN_TENSOR_ENABLE_CUDA
            ocean_cuda_embedding_forward(
                contiguous_weight->cuda_data,
                contiguous_indices->cuda_data,
                result->cuda_data,
                (int)count, (int)vocab, (int)dim
            );
#else
            ocean_tensor_release(result);
            ocean_tensor_release(contiguous_weight);
            ocean_tensor_release(contiguous_indices);
            ocean_tensor_fail("CUDA backend was not compiled");
#endif
            ocean_tensor_release(contiguous_weight);
            ocean_tensor_release(contiguous_indices);
            return result;
        }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        ocean_tensor_handle_t gpu_indices = contiguous_indices;
        if (gpu_indices->device != OCEAN_TENSOR_GPU) {
            gpu_indices = ocean_tensor_to(gpu_indices, "gpu");
        }
        size_t output_shape[indices->ndim + 1];
        for (size_t axis = 0; axis < indices->ndim; ++axis) {
            output_shape[axis] = indices->shape[axis];
        }
        output_shape[indices->ndim] = dim;
        result = ocean_tensor_alloc_uninitialized(
            output_shape,
            indices->ndim + 1,
            OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_GPU
        );
        if (count && dim) {
            if (count > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
                dim > (size_t)INT32_MAX || count > SIZE_MAX / dim ||
                count * dim > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_weight);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail(
                    "Embedding is too large for OpenCL kernel indexing"
                );
            }
            size_t error_shape[1] = {1};
            ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
                error_shape, 1, "int32", "gpu"
            );
            ocean_tensor_opencl_embedding_forward(
                contiguous_weight,
                gpu_indices,
                result,
                error,
                (int)count,
                (int)vocab,
                (int)dim
            );
            int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
            ocean_tensor_release(error);
            if (error_value != 0) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_weight);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail("Embedding token id is out of range");
            }
        }
        if (gpu_indices != contiguous_indices) ocean_tensor_release(gpu_indices);
#else
        ocean_tensor_release(contiguous_weight);
        ocean_tensor_release(contiguous_indices);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    ocean_tensor_release(contiguous_weight);
    ocean_tensor_release(contiguous_indices);
    return result;
}

ocean_tensor_handle_t ocean_tensor_embedding_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
) {
    if (!upstream || !indices) {
        ocean_tensor_fail("Embedding.backward requires non-null tensors");
    }
    if (upstream->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Embedding gradient must be Tensor[float32]");
    }
    if (indices->dtype != OCEAN_TENSOR_INT64) {
        ocean_tensor_fail("Embedding indices must be Tensor[int64]");
    }
    size_t count = indices->size;
    if (dim != 0 && count > SIZE_MAX / dim) {
        ocean_tensor_fail("Embedding gradient metadata is too large");
    }
    if (upstream->ndim != indices->ndim + 1) {
        ocean_tensor_fail("Embedding gradient rank does not match indices");
    }
    for (size_t axis = 0; axis < indices->ndim; ++axis) {
        if (upstream->shape[axis] != indices->shape[axis]) {
            ocean_tensor_fail("Embedding gradient shape does not match indices");
        }
    }
    if (upstream->shape[indices->ndim] != dim) {
        ocean_tensor_fail("Embedding gradient width does not match weight");
    }
    if (upstream->size != count * dim) {
        ocean_tensor_fail("Embedding gradient shape does not match indices");
    }

    ocean_tensor_handle_t contiguous_upstream =
        ocean_tensor_contiguous(upstream);
    ocean_tensor_handle_t contiguous_indices =
        ocean_tensor_contiguous(indices);
    ocean_tensor_handle_t result = NULL;

    if (upstream->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_indices = contiguous_indices;
        if (cpu_indices->device != OCEAN_TENSOR_CPU) {
            cpu_indices = ocean_tensor_to(cpu_indices, "cpu");
        }
        result = ocean_tensor_embedding_backward_cpu(
            contiguous_upstream,
            cpu_indices,
            vocab,
            dim
        );
        if (cpu_indices != contiguous_indices) ocean_tensor_release(cpu_indices);
    } else {
        if (upstream->device == OCEAN_TENSOR_BACKEND_CUDA) {
            ocean_tensor_release(contiguous_upstream);
            ocean_tensor_release(contiguous_indices);
            ocean_tensor_fail(
                "CUDA Embedding backward kernel is not implemented yet"
            );
        }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        ocean_tensor_handle_t gpu_indices = contiguous_indices;
        if (gpu_indices->device != OCEAN_TENSOR_GPU) {
            gpu_indices = ocean_tensor_to(gpu_indices, "gpu");
        }
        size_t gradient_shape[2] = {vocab, dim};
        result = ocean_tensor_alloc_zeros(
            gradient_shape, 2, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
        );
        if (count && dim) {
            if (count > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
                dim > (size_t)INT32_MAX || count * dim > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_upstream);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail(
                    "Embedding is too large for OpenCL kernel indexing"
                );
            }
            size_t error_shape[1] = {1};
            ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
                error_shape, 1, "int32", "gpu"
            );
            ocean_tensor_opencl_embedding_backward(
                contiguous_upstream,
                gpu_indices,
                result,
                error,
                (int)count,
                (int)vocab,
                (int)dim
            );
            int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
            ocean_tensor_release(error);
            if (error_value != 0) {
                ocean_tensor_release(result);
                if (gpu_indices != contiguous_indices) {
                    ocean_tensor_release(gpu_indices);
                }
                ocean_tensor_release(contiguous_upstream);
                ocean_tensor_release(contiguous_indices);
                ocean_tensor_fail("Embedding token id is out of range");
            }
        }
        if (gpu_indices != contiguous_indices) ocean_tensor_release(gpu_indices);
#else
        ocean_tensor_release(contiguous_upstream);
        ocean_tensor_release(contiguous_indices);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    ocean_tensor_release(contiguous_upstream);
    ocean_tensor_release(contiguous_indices);
    return result;
}

static void ocean_tensor_validate_cross_entropy_shapes(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets
) {
    if (!logits || !targets) {
        ocean_tensor_fail("CrossEntropyLoss requires non-null tensors");
    }
    if (logits->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("CrossEntropyLoss logits must be Tensor[float32]");
    }
    if (targets->dtype != OCEAN_TENSOR_INT64) {
        ocean_tensor_fail(
            "CrossEntropyLoss targets must be Tensor[int64]"
        );
    }
    if (logits->ndim < 2 || targets->ndim != logits->ndim - 1) {
        ocean_tensor_fail(
            "CrossEntropyLoss expects logits [...,V] and targets [...]"
        );
    }
    for (size_t axis = 0; axis < targets->ndim; ++axis) {
        if (logits->shape[axis] != targets->shape[axis]) {
            ocean_tensor_fail(
                "CrossEntropyLoss target shape must match logits prefix"
            );
        }
    }
    if (logits->shape[logits->ndim - 1] == 0 || targets->size == 0) {
        ocean_tensor_fail(
            "CrossEntropyLoss requires non-empty vocab and targets"
        );
    }
}

ocean_tensor_handle_t ocean_tensor_cross_entropy_forward(
    ocean_tensor_handle_t logits,
    ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
) {
    if (!probabilities_out) {
        ocean_tensor_fail(
            "CrossEntropyLoss forward requires a probabilities output"
        );
    }
    *probabilities_out = NULL;
    ocean_tensor_validate_cross_entropy_shapes(logits, targets);

    size_t rows = targets->size;
    size_t vocab = logits->shape[logits->ndim - 1];
    if (vocab != 0 && rows > SIZE_MAX / vocab) {
        ocean_tensor_fail("CrossEntropyLoss tensor is too large");
    }

    ocean_tensor_handle_t contiguous_logits =
        ocean_tensor_is_contiguous(logits)
        ? logits : ocean_tensor_contiguous(logits);
    ocean_tensor_handle_t contiguous_targets =
        ocean_tensor_is_contiguous(targets)
        ? targets : ocean_tensor_contiguous(targets);
    ocean_tensor_handle_t result = NULL;

    if (logits->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_targets = contiguous_targets;
        if (cpu_targets->device != OCEAN_TENSOR_CPU) {
            cpu_targets = ocean_tensor_to(cpu_targets, "cpu");
        }
        result = ocean_tensor_cross_entropy_forward_cpu(
            contiguous_logits,
            cpu_targets,
            probabilities_out
        );
        if (cpu_targets != contiguous_targets) ocean_tensor_release(cpu_targets);
    } else {
        if (logits->device == OCEAN_TENSOR_BACKEND_CUDA) {
            if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
            if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
            ocean_tensor_fail("CUDA CrossEntropyLoss kernel is not implemented yet");
        }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        if (rows > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
            rows * vocab > (size_t)INT32_MAX) {
            if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
            if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
            ocean_tensor_fail(
                "CrossEntropyLoss is too large for OpenCL kernel indexing"
            );
        }
        ocean_tensor_handle_t gpu_targets = contiguous_targets;
        if (gpu_targets->device != OCEAN_TENSOR_GPU) {
            gpu_targets = ocean_tensor_to(gpu_targets, "gpu");
        }
        ocean_tensor_handle_t probabilities = ocean_tensor_alloc_uninitialized(
            logits->shape,
            logits->ndim,
            OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_GPU
        );
        size_t row_shape[1] = {rows};
        ocean_tensor_handle_t row_losses = ocean_tensor_alloc_uninitialized(
            row_shape, 1, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
        );
        size_t error_shape[1] = {1};
        ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
            error_shape, 1, "int32", "gpu"
        );
        ocean_tensor_opencl_cross_entropy_forward(
            contiguous_logits,
            gpu_targets,
            probabilities,
            row_losses,
            error,
            (int)rows,
            (int)vocab
        );
        int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
        ocean_tensor_release(error);
        if (error_value != 0) {
            ocean_tensor_release(row_losses);
            ocean_tensor_release(probabilities);
            if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
            if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
            if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }

        ocean_tensor_handle_t total = ocean_tensor_sum_dim(
            row_losses, -1, false
        );
        ocean_tensor_handle_t mean = ocean_tensor_scalar(
            total, (double)rows, OCEAN_TENSOR_DIV
        );
        size_t loss_shape[2] = {1, 1};
        ocean_tensor_handle_t loss = ocean_tensor_reshape(
            mean, loss_shape, 2
        );
        ocean_tensor_release(mean);
        ocean_tensor_release(total);
        ocean_tensor_release(row_losses);
        *probabilities_out = probabilities;
        result = loss;
        if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
#else
        if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
        if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    if (contiguous_logits != logits) ocean_tensor_release(contiguous_logits);
    if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
    return result;
}

ocean_tensor_handle_t ocean_tensor_cross_entropy_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t targets
) {
    ocean_tensor_validate_cross_entropy_shapes(probabilities, targets);
    if (!upstream || upstream->dtype != OCEAN_TENSOR_FLOAT32 ||
        upstream->size != 1) {
        ocean_tensor_fail(
            "CrossEntropyLoss backward requires a float32 scalar upstream"
        );
    }
    if (upstream->device != probabilities->device) {
        ocean_tensor_fail(
            "CrossEntropyLoss backward requires matching Tensor devices"
        );
    }

    size_t rows = targets->size;
    size_t vocab = probabilities->shape[probabilities->ndim - 1];
    if (vocab != 0 && rows > SIZE_MAX / vocab) {
        ocean_tensor_fail("CrossEntropyLoss gradient is too large");
    }

    ocean_tensor_handle_t contiguous_upstream =
        ocean_tensor_is_contiguous(upstream)
        ? upstream : ocean_tensor_contiguous(upstream);
    ocean_tensor_handle_t contiguous_probabilities =
        ocean_tensor_is_contiguous(probabilities)
        ? probabilities : ocean_tensor_contiguous(probabilities);
    ocean_tensor_handle_t contiguous_targets =
        ocean_tensor_is_contiguous(targets)
        ? targets : ocean_tensor_contiguous(targets);
    ocean_tensor_handle_t result = NULL;

    if (probabilities->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t cpu_targets = contiguous_targets;
        if (cpu_targets->device != OCEAN_TENSOR_CPU) {
            cpu_targets = ocean_tensor_to(cpu_targets, "cpu");
        }
        result = ocean_tensor_cross_entropy_backward_cpu(
            contiguous_upstream,
            contiguous_probabilities,
            cpu_targets
        );
        if (cpu_targets != contiguous_targets) ocean_tensor_release(cpu_targets);
    } else {
        if (probabilities->device == OCEAN_TENSOR_BACKEND_CUDA) {
            ocean_tensor_release(contiguous_upstream);
            ocean_tensor_release(contiguous_probabilities);
            ocean_tensor_release(contiguous_targets);
            ocean_tensor_fail(
                "CUDA CrossEntropyLoss backward kernel is not implemented yet"
            );
        }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        if (rows > (size_t)INT32_MAX || vocab > (size_t)INT32_MAX ||
            rows * vocab > (size_t)INT32_MAX) {
            ocean_tensor_fail(
                "CrossEntropyLoss gradient is too large for OpenCL indexing"
            );
        }
        ocean_tensor_handle_t gpu_targets = contiguous_targets;
        if (gpu_targets->device != OCEAN_TENSOR_GPU) {
            gpu_targets = ocean_tensor_to(gpu_targets, "gpu");
        }
        result = ocean_tensor_alloc_uninitialized(
            probabilities->shape,
            probabilities->ndim,
            OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_GPU
        );
        size_t error_shape[1] = {1};
        ocean_tensor_handle_t error = ocean_tensor_zeros_nd(
            error_shape, 1, "int32", "gpu"
        );
        ocean_tensor_opencl_cross_entropy_backward(
            contiguous_upstream,
            contiguous_probabilities,
            gpu_targets,
            result,
            error,
            (int)rows,
            (int)vocab
        );
        int32_t error_value = ocean_tensor_get_flat_i32(error, 0);
        ocean_tensor_release(error);
        if (error_value != 0) {
            ocean_tensor_release(result);
            if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }
        if (gpu_targets != contiguous_targets) ocean_tensor_release(gpu_targets);
#else
        ocean_tensor_fail(
            "GPU backend is unavailable: rebuild with OpenCL support"
        );
#endif
    }

    if (contiguous_upstream != upstream) ocean_tensor_release(contiguous_upstream);
    if (contiguous_probabilities != probabilities) ocean_tensor_release(contiguous_probabilities);
    if (contiguous_targets != targets) ocean_tensor_release(contiguous_targets);
    return result;
}

void ocean_tensor_copy_into(
    ocean_tensor_handle_t destination,
    ocean_tensor_handle_t source
) {
    if (!destination || !source) {
        ocean_tensor_fail("Tensor.copy_into requires non-null tensors");
    }
    if (
        destination->dtype != source->dtype
        || destination->ndim != source->ndim
        || destination->size != source->size
    ) {
        ocean_tensor_fail("Tensor.copy_into metadata mismatch");
    }

    for (size_t axis = 0; axis < destination->ndim; ++axis) {
        if (destination->shape[axis] != source->shape[axis]) {
            ocean_tensor_fail("Tensor.copy_into shape mismatch");
        }
    }

    if (destination->device == source->device) {
        if (ocean_tensor_is_contiguous(destination) &&
            ocean_tensor_is_contiguous(source)) {
            ocean_tensor_backend_for_device(destination->device)->copy(
                destination,
                source
            );
            return;
        }
        if (destination->device == OCEAN_TENSOR_CPU) {
            unsigned char *destination_data =
                (unsigned char *)destination->cpu_data;
            const unsigned char *source_data =
                (const unsigned char *)source->cpu_data;
            for (size_t index = 0; index < destination->size; ++index) {
                size_t remaining = index;
                size_t destination_offset = 0;
                size_t source_offset = 0;
                for (size_t axis = destination->ndim; axis-- > 0;) {
                    size_t coordinate = destination->shape[axis] == 0
                        ? 0 : remaining % destination->shape[axis];
                    remaining = destination->shape[axis] == 0
                        ? 0 : remaining / destination->shape[axis];
                    destination_offset += coordinate * destination->strides[axis];
                    source_offset += coordinate * source->strides[axis];
                }
                memcpy(
                    destination_data + destination_offset * destination->item_size,
                    source_data + source_offset * source->item_size,
                    destination->item_size
                );
            }
            return;
        }
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (destination->device == OCEAN_TENSOR_BACKEND_CUDA) {
            if (destination->ndim > OCEAN_CUDA_MAX_BROADCAST_RANK) {
                ocean_tensor_fail("CUDA strided copy rank is too large");
            }
            ocean_cuda_strided_copy_desc descriptor = {0};
            descriptor.ndim = (int)destination->ndim;
            descriptor.item_size = destination->item_size;
            descriptor.total = destination->size;
            for (size_t axis = 0; axis < destination->ndim; ++axis) {
                descriptor.shape[axis] = destination->shape[axis];
                descriptor.source_strides[axis] = source->strides[axis];
                descriptor.destination_strides[axis] = destination->strides[axis];
            }
            ocean_cuda_copy_strided(
                source->cuda_data,
                destination->cuda_data,
                &descriptor
            );
            return;
        }
#endif
        ocean_tensor_backend_for_device(destination->device)->copy(
            destination,
            source
        );
        return;
    }

    if (destination->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_backend_for_device(source->device)->read(
            source,
            destination->cpu_data
        );
        return;
    }

    if (source->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_backend_for_device(destination->device)->write(
            destination,
            source->cpu_data
        );
        return;
    }

    ocean_tensor_fail("Tensor.copy_into unsupported device pair");
}

ocean_tensor_handle_t ocean_tensor_to(
    ocean_tensor_handle_t tensor,
    const char *device
) {
    if (!tensor) ocean_tensor_fail("cannot move a null Tensor");
    int target = ocean_tensor_parse_device(device);
    if ((ocean_tensor_backend_kind)target == tensor->device) {
        return ocean_tensor_copy(tensor);
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc(
        tensor->shape, tensor->ndim, tensor->dtype, target
    );
    const ocean_tensor_backend_ops *source_backend =
        ocean_tensor_backend_for_device(tensor->device);
    const ocean_tensor_backend_ops *target_backend =
        ocean_tensor_backend_for_device((ocean_tensor_backend_kind)target);
    target_backend->allocate(result);
    if (target == OCEAN_TENSOR_CPU) {
        source_backend->read(tensor, result->cpu_data);
    } else if (tensor->device == OCEAN_TENSOR_CPU) {
        target_backend->write(result, tensor->cpu_data);
    } else {
        /* Cross-GPU transfers are staged through host memory until the CUDA
           backend exposes a peer-to-peer transfer contract. */
        ocean_tensor_handle_t host = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
        );
        source_backend->read(tensor, host->cpu_data);
        target_backend->write(result, host->cpu_data);
        ocean_tensor_release(host);
    }
    return result;
}

static float ocean_tensor_half_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | (exponent + 112u) << 23 | (mantissa << 13);
    }
    union { uint32_t bits; float value; } result = {bits};
    return result.value;
}

static uint16_t ocean_tensor_float_to_half(float value) {
    union { uint32_t bits; float value; } input = {0};
    input.value = value;
    uint32_t sign = (input.bits >> 16) & 0x8000u;
    uint32_t exponent = (input.bits >> 23) & 0xffu;
    uint32_t mantissa = input.bits & 0x7fffffu;
    if (exponent == 255) {
        return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x0200u : 0));
    }
    int unbiased = (int)exponent - 127;
    if (unbiased < -14) {
        if (unbiased < -24) return (uint16_t)sign;
        uint32_t shifted = (mantissa | 0x800000u) >> (uint32_t)(-unbiased - 14 + 13);
        return (uint16_t)(sign | shifted);
    }
    if (unbiased > 15) return (uint16_t)(sign | 0x7c00u);
    uint32_t half_exponent = (uint32_t)(unbiased + 15);
    uint32_t half_mantissa = mantissa >> 13;
    if (mantissa & 0x1000u) ++half_mantissa;
    if (half_mantissa == 0x0400u) {
        half_mantissa = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (half_exponent << 10) | half_mantissa);
}

static long double ocean_tensor_read_scalar(
    const ocean_tensor_handle_t tensor,
    size_t index
) {
    const unsigned char *data = (const unsigned char *)tensor->cpu_data;
    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: return ((const bool *)data)[index] ? 1.0L : 0.0L;
        case OCEAN_TENSOR_INT8: return ((const int8_t *)data)[index];
        case OCEAN_TENSOR_INT16: return ((const int16_t *)data)[index];
        case OCEAN_TENSOR_INT32: return ((const int32_t *)data)[index];
        case OCEAN_TENSOR_INT64: return (long double)((const int64_t *)data)[index];
        case OCEAN_TENSOR_UINT8: return ((const uint8_t *)data)[index];
        case OCEAN_TENSOR_UINT16: return ((const uint16_t *)data)[index];
        case OCEAN_TENSOR_UINT32: return ((const uint32_t *)data)[index];
        case OCEAN_TENSOR_UINT64: return (long double)((const uint64_t *)data)[index];
        case OCEAN_TENSOR_FLOAT16:
            return (long double)ocean_tensor_half_to_float(((const uint16_t *)data)[index]);
        case OCEAN_TENSOR_FLOAT32: return ((const float *)data)[index];
        case OCEAN_TENSOR_FLOAT64: return ((const double *)data)[index];
    }
    ocean_tensor_fail("invalid Tensor scalar type");
    return 0.0L;
}

static void ocean_tensor_write_scalar(
    const ocean_tensor_handle_t tensor,
    size_t index,
    long double value
) {
    unsigned char *data = (unsigned char *)tensor->cpu_data;
    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: ((bool *)data)[index] = value != 0.0L; return;
        case OCEAN_TENSOR_INT8: ((int8_t *)data)[index] = (int8_t)value; return;
        case OCEAN_TENSOR_INT16: ((int16_t *)data)[index] = (int16_t)value; return;
        case OCEAN_TENSOR_INT32: ((int32_t *)data)[index] = (int32_t)value; return;
        case OCEAN_TENSOR_INT64: ((int64_t *)data)[index] = (int64_t)value; return;
        case OCEAN_TENSOR_UINT8: ((uint8_t *)data)[index] = (uint8_t)value; return;
        case OCEAN_TENSOR_UINT16: ((uint16_t *)data)[index] = (uint16_t)value; return;
        case OCEAN_TENSOR_UINT32: ((uint32_t *)data)[index] = (uint32_t)value; return;
        case OCEAN_TENSOR_UINT64: ((uint64_t *)data)[index] = (uint64_t)value; return;
        case OCEAN_TENSOR_FLOAT16:
            ((uint16_t *)data)[index] = ocean_tensor_float_to_half((float)value);
            return;
        case OCEAN_TENSOR_FLOAT32: ((float *)data)[index] = (float)value; return;
        case OCEAN_TENSOR_FLOAT64: ((double *)data)[index] = (double)value; return;
    }
    ocean_tensor_fail("invalid Tensor scalar type");
}

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static bool ocean_tensor_contains_zero(const ocean_tensor_handle_t tensor) {
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    bool found = false;
    for (size_t index = 0; index < cpu->size; ++index) {
        if (ocean_tensor_read_scalar(cpu, index) == 0.0L) {
            found = true;
            break;
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return found;
}
#endif

static long double ocean_tensor_apply_binary(
    long double left,
    long double right,
    int operation
) {
    switch (operation) {
        case OCEAN_TENSOR_ADD: return left + right;
        case OCEAN_TENSOR_SUB: return left - right;
        case OCEAN_TENSOR_MUL: return left * right;
        case OCEAN_TENSOR_DIV:
            if (right == 0.0L) ocean_tensor_fail("Tensor division by zero");
            return left / right;
    }
    ocean_tensor_fail("invalid Tensor binary operation");
    return 0.0L;
}

static void ocean_tensor_broadcast_shape(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    size_t **shape_out,
    size_t *ndim_out
) {
    size_t ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating broadcast shape");
    for (size_t axis = 0; axis < ndim; ++axis) {
        size_t left_axis = axis + left->ndim >= ndim
            ? left->shape[axis + left->ndim - ndim] : 1;
        size_t right_axis = axis + right->ndim >= ndim
            ? right->shape[axis + right->ndim - ndim] : 1;
        if (left_axis != right_axis && left_axis != 1 && right_axis != 1) {
            free(shape);
            ocean_tensor_fail("Tensor shapes are not broadcast-compatible");
        }
        /* Broadcasting dimension 0 with dimension 1 produces dimension 0.
           Using max(left_axis, right_axis) incorrectly turns that case into
           a non-empty result and later dereferences a NULL data buffer. */
        if (left_axis == right_axis) shape[axis] = left_axis;
        else if (left_axis == 1) shape[axis] = right_axis;
        else if (right_axis == 1) shape[axis] = left_axis;
    }
    *shape_out = shape;
    *ndim_out = ndim;
}

static size_t ocean_tensor_broadcast_offset(
    const ocean_tensor_handle_t tensor,
    const size_t *result_shape,
    size_t result_ndim,
    size_t linear
) {
    size_t remaining = linear;
    size_t offset = 0;
    for (size_t axis = result_ndim; axis-- > 0;) {
        size_t coordinate = result_shape[axis]
            ? remaining % result_shape[axis] : 0;
        remaining = result_shape[axis]
            ? remaining / result_shape[axis] : 0;
        if (axis + tensor->ndim >= result_ndim) {
            size_t tensor_axis = axis + tensor->ndim - result_ndim;
            if (tensor->shape[tensor_axis] != 1) {
                offset += coordinate * tensor->strides[tensor_axis];
            }
        }
    }
    return offset;
}

static ocean_tensor_handle_t ocean_tensor_binary_cpu(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    int operation
) {
    size_t *shape = NULL;
    size_t ndim = 0;
    ocean_tensor_broadcast_shape(left, right, &shape, &ndim);
    if (shape == NULL || ndim == 0) {
        ocean_tensor_fail(
            "Tensor broadcast produced invalid metadata"
        );
    }


    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, ndim, left->dtype, OCEAN_TENSOR_CPU
    );

    bool same_shape = left->ndim == right->ndim && left->ndim == ndim;
    if (same_shape) {
        for (size_t axis = 0; axis < ndim; ++axis) {
            if (left->shape[axis] != right->shape[axis]) {
                same_shape = false;
                break;
            }
        }
    }

    if (same_shape && left->dtype == OCEAN_TENSOR_FLOAT32) {
        const float *restrict a = (const float *)left->cpu_data;
        const float *restrict b = (const float *)right->cpu_data;
        float *restrict out = (float *)result->cpu_data;
        size_t size = result->size;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
                break;
            case OCEAN_TENSOR_DIV:
                for (size_t i = 0; i < size; ++i) {
                    if (b[i] == 0.0f) {
                        free(shape);
                        ocean_tensor_release(result);
                        ocean_tensor_fail("Tensor division by zero");
                    }
                    out[i] = a[i] / b[i];
                }
                break;
            default:
                free(shape);
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }

        free(shape);
        return result;
    }

    if (same_shape && left->dtype == OCEAN_TENSOR_FLOAT64) {
        const double *restrict a = (const double *)left->cpu_data;
        const double *restrict b = (const double *)right->cpu_data;
        double *restrict out = (double *)result->cpu_data;
        size_t size = result->size;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
                break;
            case OCEAN_TENSOR_DIV:
                for (size_t i = 0; i < size; ++i) {
                    if (b[i] == 0.0) {
                        free(shape);
                        ocean_tensor_release(result);
                        ocean_tensor_fail("Tensor division by zero");
                    }
                    out[i] = a[i] / b[i];
                }
                break;
            default:
                free(shape);
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }

        free(shape);
        return result;
    }

    if (same_shape) {
        for (size_t linear = 0; linear < result->size; ++linear) {
            ocean_tensor_write_scalar(
                result,
                linear,
                ocean_tensor_apply_binary(
                    ocean_tensor_read_scalar(left, linear),
                    ocean_tensor_read_scalar(right, linear),
                    operation
                )
            );
        }
        free(shape);
        return result;
    }

    for (size_t linear = 0; linear < result->size; ++linear) {
        size_t left_index = ocean_tensor_broadcast_offset(
            left, shape, ndim, linear
        );
        size_t right_index = ocean_tensor_broadcast_offset(
            right, shape, ndim, linear
        );
        ocean_tensor_write_scalar(
            result,
            linear,
            ocean_tensor_apply_binary(
                ocean_tensor_read_scalar(left, left_index),
                ocean_tensor_read_scalar(right, right_index),
                operation
            )
        );
    }

    free(shape);
    return result;
}


static ocean_tensor_handle_t ocean_tensor_scalar_cpu(
    const ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    size_t size = tensor->size;

    if (tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        const float *restrict input = (const float *)tensor->cpu_data;
        float *restrict out = (float *)result->cpu_data;
        float s = (float)scalar;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] + s;
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] - s;
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] * s;
                break;
            case OCEAN_TENSOR_DIV:
                if (s == 0.0f) {
                    ocean_tensor_release(result);
                    ocean_tensor_fail("Tensor division by zero");
                }
                for (size_t i = 0; i < size; ++i) out[i] = input[i] / s;
                break;
            default:
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }
        return result;
    }

    if (tensor->dtype == OCEAN_TENSOR_FLOAT64) {
        const double *restrict input = (const double *)tensor->cpu_data;
        double *restrict out = (double *)result->cpu_data;

        switch (operation) {
            case OCEAN_TENSOR_ADD:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] + scalar;
                break;
            case OCEAN_TENSOR_SUB:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] - scalar;
                break;
            case OCEAN_TENSOR_MUL:
                for (size_t i = 0; i < size; ++i) out[i] = input[i] * scalar;
                break;
            case OCEAN_TENSOR_DIV:
                if (scalar == 0.0) {
                    ocean_tensor_release(result);
                    ocean_tensor_fail("Tensor division by zero");
                }
                for (size_t i = 0; i < size; ++i) out[i] = input[i] / scalar;
                break;
            default:
                ocean_tensor_release(result);
                ocean_tensor_fail("invalid Tensor binary operation");
        }
        return result;
    }

    for (size_t i = 0; i < size; ++i) {
        ocean_tensor_write_scalar(
            result,
            i,
            ocean_tensor_apply_binary(
                ocean_tensor_read_scalar(tensor, i),
                (long double)scalar,
                operation
            )
        );
    }

    return result;
}

static ocean_tensor_handle_t ocean_tensor_ternary_quantize_cpu(
    const ocean_tensor_handle_t tensor
) {
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.ternary_quantize currently requires float32");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    if (tensor->size == 0) return result;

    const float *input = (const float *)tensor->cpu_data;
    float *output = (float *)result->cpu_data;
    double sum_abs = 0.0;
    for (size_t index = 0; index < tensor->size; ++index) {
        sum_abs += fabs((double)input[index]);
    }

    float scale = (float)(sum_abs / (double)tensor->size);
    if (scale < 1.0e-8f) scale = 1.0e-8f;
    float threshold = 0.5f * scale;
    for (size_t index = 0; index < tensor->size; ++index) {
        float value = input[index];
        output[index] = value > threshold
            ? scale
            : (value < -threshold ? -scale : 0.0f);
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_gelu_cpu(
    const ocean_tensor_handle_t tensor
) {
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    const float *input = (const float *)tensor->cpu_data;
    float *output = (float *)result->cpu_data;
    const float coefficient = 0.7978845608028654f;
    const float cubic = 0.044715f;
    for (size_t index = 0; index < tensor->size; ++index) {
        float value = input[index];
        float value_squared = value * value;
        float argument = coefficient * (
            value + cubic * value * value_squared
        );
        output[index] = 0.5f * value * (1.0f + tanhf(argument));
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_gelu_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t input
) {
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        input->shape, input->ndim, input->dtype, OCEAN_TENSOR_CPU
    );
    const float *upstream_data = (const float *)upstream->cpu_data;
    const float *input_data = (const float *)input->cpu_data;
    float *output = (float *)result->cpu_data;
    const float coefficient = 0.7978845608028654f;
    const float cubic = 0.044715f;
    for (size_t index = 0; index < input->size; ++index) {
        float value = input_data[index];
        float argument = coefficient * (
            value + cubic * value * value * value
        );
        float tanh_argument = tanhf(argument);
        float derivative = 0.5f * (1.0f + tanh_argument);
        derivative += 0.5f * value
            * (1.0f - tanh_argument * tanh_argument)
            * coefficient
            * (1.0f + 3.0f * cubic * value * value);
        output[index] = upstream_data[index] * derivative;
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_embedding_forward_cpu(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices
) {
    size_t dim = weight->shape[1];
    size_t output_shape[indices->ndim + 1];
    for (size_t axis = 0; axis < indices->ndim; ++axis) {
        output_shape[axis] = indices->shape[axis];
    }
    output_shape[indices->ndim] = dim;
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape,
        indices->ndim + 1,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );

    size_t vocab = weight->shape[0];
    for (size_t position = 0; position < indices->size; ++position) {
        int64_t token = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)indices,
            position
        );
        if (token < 0 || (uint64_t)token >= (uint64_t)vocab) {
            ocean_tensor_release(result);
            ocean_tensor_fail("Embedding token id is out of range");
        }
        size_t row = (size_t)token;
        for (size_t feature = 0; feature < dim; ++feature) {
            ocean_tensor_write_scalar(
                result,
                position * dim + feature,
                (long double)ocean_tensor_get_flat_f32(
                    (ocean_tensor_handle_t)weight,
                    row * dim + feature
                )
            );
        }
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_embedding_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    size_t vocab,
    size_t dim
) {
    size_t gradient_shape[2] = {vocab, dim};
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        gradient_shape,
        2,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );

    for (size_t position = 0; position < indices->size; ++position) {
        int64_t token = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)indices,
            position
        );
        if (token < 0 || (uint64_t)token >= (uint64_t)vocab) {
            ocean_tensor_release(result);
            ocean_tensor_fail("Embedding token id is out of range");
        }
        size_t row = (size_t)token;
        for (size_t feature = 0; feature < dim; ++feature) {
            size_t gradient_index = row * dim + feature;
            float value = ocean_tensor_get_flat_f32(
                (ocean_tensor_handle_t)upstream,
                position * dim + feature
            );
            float previous = ocean_tensor_get_flat_f32(result, gradient_index);
            ocean_tensor_write_scalar(
                result,
                gradient_index,
                (long double)(previous + value)
            );
        }
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cross_entropy_forward_cpu(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t *probabilities_out
) {
    size_t vocab = logits->shape[logits->ndim - 1];
    size_t rows = targets->size;
    ocean_tensor_handle_t probabilities = ocean_tensor_alloc_uninitialized(
        logits->shape,
        logits->ndim,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );
    double total_loss = 0.0;

    for (size_t row = 0; row < rows; ++row) {
        size_t offset = row * vocab;
        float maximum = -INFINITY;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float value = ocean_tensor_get_flat_f32(
                (ocean_tensor_handle_t)logits,
                offset + cls
            );
            if (value > maximum) maximum = value;
        }

        double denominator = 0.0;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float exponential = expf(
                ocean_tensor_get_flat_f32(
                    (ocean_tensor_handle_t)logits,
                    offset + cls
                ) - maximum
            );
            ocean_tensor_write_scalar(
                probabilities,
                offset + cls,
                (long double)exponential
            );
            denominator += (double)exponential;
        }

        int64_t target = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)targets,
            row
        );
        if (target < 0 || (uint64_t)target >= (uint64_t)vocab) {
            ocean_tensor_release(probabilities);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }
        float target_logit = ocean_tensor_get_flat_f32(
            (ocean_tensor_handle_t)logits,
            offset + (size_t)target
        );
        total_loss += log(denominator) + (double)maximum
            - (double)target_logit;

        for (size_t cls = 0; cls < vocab; ++cls) {
            float probability = ocean_tensor_get_flat_f32(
                probabilities,
                offset + cls
            );
            ocean_tensor_write_scalar(
                probabilities,
                offset + cls,
                (long double)((double)probability / denominator)
            );
        }
    }

    ocean_tensor_handle_t loss = ocean_tensor_zeros(1, 1, "cpu");
    ocean_tensor_fill(loss, total_loss / (double)rows);
    *probabilities_out = probabilities;
    return loss;
}

static ocean_tensor_handle_t ocean_tensor_cross_entropy_backward_cpu(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets
) {
    size_t vocab = probabilities->shape[probabilities->ndim - 1];
    size_t rows = targets->size;
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        probabilities->shape,
        probabilities->ndim,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_CPU
    );
    float scale = ocean_tensor_get_flat_f32(
        (ocean_tensor_handle_t)upstream,
        0
    ) / (float)rows;

    for (size_t row = 0; row < rows; ++row) {
        int64_t target = ocean_tensor_get_flat_i64(
            (ocean_tensor_handle_t)targets,
            row
        );
        if (target < 0 || (uint64_t)target >= (uint64_t)vocab) {
            ocean_tensor_release(result);
            ocean_tensor_fail("CrossEntropyLoss target is out of range");
        }
        size_t offset = row * vocab;
        for (size_t cls = 0; cls < vocab; ++cls) {
            float value = ocean_tensor_get_flat_f32(
                (ocean_tensor_handle_t)probabilities,
                offset + cls
            );
            if (cls == (size_t)target) value -= 1.0f;
            ocean_tensor_write_scalar(
                result,
                offset + cls,
                (long double)(value * scale)
            );
        }
    }
    return result;
}


#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include "std/tensor/tensor_opencl_elementwise.inc"
#endif

void ocean_tensor_cache_write(
    ocean_tensor_handle_t cache,
    ocean_tensor_handle_t value,
    int position
) {
    if (!cache || !value) {
        ocean_tensor_fail("Tensor.cache_write requires non-null tensors");
    }
    if (cache->dtype != OCEAN_TENSOR_FLOAT32 ||
        value->dtype != OCEAN_TENSOR_FLOAT32 ||
        cache->ndim != 4 || value->ndim != 4 ||
        value->shape[0] != cache->shape[0] ||
        value->shape[1] != cache->shape[1] ||
        value->shape[2] == 0 ||
        value->shape[3] != cache->shape[3] ||
        cache->device != value->device ||
        !ocean_tensor_is_contiguous(cache) ||
        !ocean_tensor_is_contiguous(value)) {
        ocean_tensor_fail("Tensor.cache_write metadata mismatch");
    }
    if (position < 0 || (size_t)position > cache->shape[2] ||
        value->shape[2] > cache->shape[2] - (size_t)position) {
        ocean_tensor_fail("Tensor.cache_write position is out of bounds");
    }

    size_t batches = cache->shape[0];
    size_t heads = cache->shape[1];
    size_t sequence = cache->shape[2];
    size_t value_sequence = value->shape[2];
    size_t width = cache->shape[3];
    if (cache->device == OCEAN_TENSOR_CPU) {
        float *destination = (float *)cache->cpu_data;
        const float *source = (const float *)value->cpu_data;
        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t head = 0; head < heads; ++head) {
                for (size_t value_position = 0;
                     value_position < value_sequence; ++value_position) {
                    size_t destination_offset =
                        ((batch * heads + head) * sequence +
                         (size_t)position + value_position) * width;
                    size_t source_offset =
                        ((batch * heads + head) * value_sequence + value_position) * width;
                    memcpy(
                        destination + destination_offset,
                        source + source_offset,
                        width * sizeof(float)
                    );
                }
            }
        }
        return;
    }
    if (cache->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (batches > (size_t)INT_MAX || heads > (size_t)INT_MAX ||
            sequence > (size_t)INT_MAX || value_sequence > (size_t)INT_MAX ||
            width > (size_t)INT_MAX || value->size > (size_t)INT_MAX) {
            ocean_tensor_fail("Tensor.cache_write dimensions are too large for CUDA");
        }
        ocean_cuda_cache_write(
            cache->cuda_data, value->cuda_data,
            (int)batches, (int)heads, (int)sequence, (int)value_sequence,
            (int)width, position
        );
        return;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (heads > (size_t)INT32_MAX || sequence > (size_t)INT32_MAX ||
        value_sequence > (size_t)INT32_MAX || width > (size_t)INT32_MAX ||
        value->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor.cache_write dimensions are too large for OpenCL");
    }
    ocean_tensor_opencl_cache_write(cache, value, position);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
}

int ocean_tensor_argmax(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor.argmax received a null Tensor");
    if (tensor->size == 0) ocean_tensor_fail("Tensor.argmax on an empty Tensor");
#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(tensor)
            ? tensor : ocean_tensor_contiguous(tensor);
        int result = ocean_cuda_argmax_f32(contiguous->cuda_data, contiguous->size);
        if (contiguous != tensor) ocean_tensor_release(contiguous);
        return result;
    }
#endif
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        return ocean_tensor_opencl_argmax(tensor);
    }
#endif

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t best_index = 0;
    long double best_value = ocean_tensor_read_scalar(cpu, 0);
    for (size_t index = 1; index < cpu->size; ++index) {
        long double value = ocean_tensor_read_scalar(cpu, index);
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    if (best_index > (size_t)INT_MAX) {
        ocean_tensor_fail("Tensor.argmax index does not fit in int");
    }
    return (int)best_index;
}

static ocean_tensor_handle_t ocean_tensor_binary_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (operation == OCEAN_TENSOR_DIV && ocean_tensor_contains_zero(right)) {
        ocean_tensor_fail("Tensor division by zero");
    }
    if ((left->dtype == OCEAN_TENSOR_FLOAT32 || left->dtype == OCEAN_TENSOR_INT32) &&
        ocean_tensor_same_shape(left, right)) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            left->shape, left->ndim, left->dtype, OCEAN_TENSOR_GPU
        );
        cl_kernel kernel = ocean_tensor_opencl_get_kernel(
            left->dtype == OCEAN_TENSOR_INT32
                ? OCEAN_TENSOR_OPENCL_KERNEL_BINARY_INT32
                : OCEAN_TENSOR_OPENCL_KERNEL_BINARY_FLOAT32
        );
        ocean_tensor_opencl_binary(left, right, result, operation, kernel);
        return result;
    }
    ocean_tensor_handle_t left_cpu = ocean_tensor_to(left, "cpu");
    ocean_tensor_handle_t right_cpu = ocean_tensor_to(right, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_binary_cpu(
        left_cpu, right_cpu, operation
    );
    ocean_tensor_handle_t gpu_result = ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(left_cpu);
    ocean_tensor_release(right_cpu);
    ocean_tensor_release(cpu_result);
    return gpu_result;
#else
    (void)left;
    (void)right;
    (void)operation;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_binary(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
    if (!left || !right) ocean_tensor_fail("Tensor operation on null handle");
    if (left->dtype != right->dtype) {
        ocean_tensor_fail("Tensor operation requires matching dtypes");
    }
    if (left->device != right->device) {
        ocean_tensor_fail("Tensor operation requires matching devices");
    }
    return ocean_tensor_backend_for_device(left->device)->binary(
        left, right, operation
    );
}

static ocean_tensor_handle_t ocean_tensor_scalar_opencl(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->dtype == OCEAN_TENSOR_FLOAT32 || tensor->dtype == OCEAN_TENSOR_INT32) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        cl_kernel kernel = ocean_tensor_opencl_get_kernel(
            tensor->dtype == OCEAN_TENSOR_INT32
                ? OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_INT32
                : OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_FLOAT32
        );
        ocean_tensor_opencl_scalar(
            tensor, result, scalar, operation, kernel,
            tensor->dtype == OCEAN_TENSOR_INT32
        );
        return result;
    }
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_scalar_cpu(cpu, scalar, operation);
    ocean_tensor_handle_t gpu_result = ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(cpu);
    ocean_tensor_release(cpu_result);
    return gpu_result;
#else
    (void)tensor;
    (void)scalar;
    (void)operation;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_scalar(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    if (!tensor) ocean_tensor_fail("Tensor scalar operation on null handle");
    if (operation == OCEAN_TENSOR_DIV && scalar == 0.0) {
        ocean_tensor_fail("Tensor division by zero");
    }
    return ocean_tensor_backend_for_device(tensor->device)->scalar(
        tensor, scalar, operation
    );
}

ocean_tensor_handle_t ocean_tensor_reshape(
    ocean_tensor_handle_t tensor,
    const size_t *shape,
    size_t ndim
) {
    if (!tensor || !shape) {
        ocean_tensor_fail("Tensor reshape received null metadata");
    }
    if (ocean_tensor_elements_from_shape(shape, ndim) != tensor->size) {
        ocean_tensor_fail("Tensor reshape must preserve the number of elements");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, ndim, tensor->dtype, tensor->device
    );
    const ocean_tensor_backend_ops *backend =
        ocean_tensor_backend_for_device(tensor->device);
    backend->copy(result, tensor);
    return result;
}


ocean_tensor_handle_t ocean_tensor_reshape_2d(
    ocean_tensor_handle_t tensor,
    int rows,
    int cols
) {
    if (rows < 0 || cols < 0) {
        ocean_tensor_fail("Tensor reshape dimensions must be non-negative");
    }
    size_t shape[2] = {(size_t)rows, (size_t)cols};
    return ocean_tensor_reshape(tensor, shape, 2);
}

ocean_tensor_handle_t ocean_tensor_transpose(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor transpose on null handle");
    if (tensor->ndim != 2) ocean_tensor_fail("Tensor transpose() expects 2D; use transpose_dims() for ND Tensor");
    return ocean_tensor_transpose_dims(tensor, 0, 1);
}

static ocean_tensor_handle_t ocean_tensor_restore_device(
    const ocean_tensor_handle_t source,
    ocean_tensor_handle_t cpu_result
) {
    if (source->device == OCEAN_TENSOR_CPU) return cpu_result;
    const char *device = source->device == OCEAN_TENSOR_BACKEND_CUDA
        ? "cuda" : "opencl";
    ocean_tensor_handle_t result = ocean_tensor_to(cpu_result, device);
    ocean_tensor_release(cpu_result);
    return result;
}

ocean_tensor_handle_t ocean_tensor_row(ocean_tensor_handle_t tensor, int row) {
    if (!tensor || tensor->ndim != 2) {
        ocean_tensor_fail("Tensor row() currently expects a 2D Tensor");
    }
    if (row < 0 || (size_t)row >= tensor->shape[0]) {
        ocean_tensor_fail("Tensor row index out of bounds");
    }
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t shape[1] = {tensor->shape[1]};
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t column = 0; column < shape[0]; ++column) {
        size_t source_index = (size_t)row * cpu->strides[0]
            + column * cpu->strides[1];
        memcpy(
            (unsigned char *)cpu_result->cpu_data + column * tensor->item_size,
            (unsigned char *)cpu->cpu_data + source_index * tensor->item_size,
            tensor->item_size
        );
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, cpu_result);
}

ocean_tensor_handle_t ocean_tensor_column(ocean_tensor_handle_t tensor, int column) {
    if (!tensor || tensor->ndim != 2) {
        ocean_tensor_fail("Tensor column() currently expects a 2D Tensor");
    }
    if (column < 0 || (size_t)column >= tensor->shape[1]) {
        ocean_tensor_fail("Tensor column index out of bounds");
    }
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t shape[1] = {tensor->shape[0]};
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t row = 0; row < shape[0]; ++row) {
        size_t source_index = row * cpu->strides[0]
            + (size_t)column * cpu->strides[1];
        memcpy(
            (unsigned char *)cpu_result->cpu_data + row * tensor->item_size,
            (unsigned char *)cpu->cpu_data + source_index * tensor->item_size,
            tensor->item_size
        );
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, cpu_result);
}

ocean_tensor_handle_t ocean_tensor_slice(
    ocean_tensor_handle_t tensor,
    int axis,
    int start,
    int stop,
    int step
) {
    if (!tensor) ocean_tensor_fail("Tensor slice() received a null Tensor");
    if (axis < 0 || (size_t)axis >= tensor->ndim) {
        ocean_tensor_fail("Tensor slice axis out of bounds");
    }
    if (start < 0 || stop < 0 || step <= 0 || start > stop
        || (size_t)stop > tensor->shape[axis]) {
        ocean_tensor_fail("Tensor slice bounds are invalid");
    }

    size_t *shape = (size_t *)malloc(tensor->ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating Tensor slice shape");
    memcpy(shape, tensor->shape, tensor->ndim * sizeof(size_t));
    shape[axis] = start == stop
        ? 0 : ((size_t)(stop - start) + (size_t)step - 1) / (size_t)step;

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32 &&
        tensor->ndim == 4 && axis == 2 && step == 1) {
        if (tensor->shape[0] > (size_t)INT32_MAX ||
            tensor->shape[1] > (size_t)INT32_MAX ||
            tensor->shape[2] > (size_t)INT32_MAX ||
            tensor->shape[3] > (size_t)INT32_MAX ||
            shape[2] > (size_t)INT32_MAX) {
            free(shape);
            ocean_tensor_fail("Tensor cache slice dimensions are too large for OpenCL");
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_cache_slice(tensor, result, start);
        free(shape);
        return result;
    }
#endif

#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32 && tensor->ndim == 4 &&
        axis == 2 && step == 1 && ocean_tensor_is_contiguous(tensor)) {
        if (tensor->shape[0] > (size_t)INT_MAX ||
            tensor->shape[1] > (size_t)INT_MAX ||
            tensor->shape[2] > (size_t)INT_MAX ||
            tensor->shape[3] > (size_t)INT_MAX ||
            shape[2] > (size_t)INT_MAX) {
            free(shape);
            ocean_tensor_fail("Tensor cache slice dimensions are too large for CUDA");
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_BACKEND_CUDA
        );
        ocean_cuda_cache_slice(
            tensor->cuda_data, result->cuda_data,
            (int)tensor->shape[0], (int)tensor->shape[1],
            (int)tensor->shape[2], (int)shape[2], (int)tensor->shape[3],
            start
        );
        free(shape);
        return result;
    }
#endif

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32 && step == 1 &&
        (size_t)axis == tensor->ndim - 1 &&
        ocean_tensor_is_contiguous(tensor)) {
        if (tensor->shape[tensor->ndim - 1] == 0 ||
            tensor->shape[tensor->ndim - 1] > (size_t)INT32_MAX ||
            shape[axis] > (size_t)INT32_MAX ||
            tensor->size / tensor->shape[tensor->ndim - 1] > (size_t)INT32_MAX) {
            free(shape);
            ocean_tensor_fail("Tensor slice dimensions are too large for OpenCL");
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_slice_last_dim(tensor, result, start);
        free(shape);
        return result;
    }
#endif

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t linear = 0; linear < cpu_result->size; ++linear) {
        size_t remaining = linear;
        size_t source_offset = 0;
        for (size_t current_axis = tensor->ndim; current_axis-- > 0;) {
            size_t coordinate = cpu_result->shape[current_axis]
                ? remaining % cpu_result->shape[current_axis] : 0;
            remaining = cpu_result->shape[current_axis]
                ? remaining / cpu_result->shape[current_axis] : 0;
            if (current_axis == (size_t)axis) {
                coordinate = (size_t)start + coordinate * (size_t)step;
            }
            source_offset += coordinate * cpu->strides[current_axis];
        }
        memcpy(
            (unsigned char *)cpu_result->cpu_data + linear * tensor->item_size,
            (unsigned char *)cpu->cpu_data + source_offset * tensor->item_size,
            tensor->item_size
        );
    }
    free(shape);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, cpu_result);
}

double ocean_tensor_sum(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor sum on null handle");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double result = 0.0L;
    for (size_t index = 0; index < cpu->size; ++index) {
        result += ocean_tensor_read_scalar(cpu, index);
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return (double)result;
}

double ocean_tensor_mean(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor mean on null handle");
    if (tensor->size == 0) return 0.0;
    return ocean_tensor_sum(tensor) / (double)tensor->size;
}

double ocean_tensor_max(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor max on null handle");
    if (tensor->size == 0) ocean_tensor_fail("Tensor max on an empty Tensor");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double result = ocean_tensor_read_scalar(cpu, 0);
    for (size_t index = 1; index < cpu->size; ++index) {
        long double value = ocean_tensor_read_scalar(cpu, index);
        if (value > result) result = value;
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return (double)result;
}

double ocean_tensor_min(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor min on null handle");
    if (tensor->size == 0) ocean_tensor_fail("Tensor min on an empty Tensor");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    long double result = ocean_tensor_read_scalar(cpu, 0);
    for (size_t index = 1; index < cpu->size; ++index) {
        long double value = ocean_tensor_read_scalar(cpu, index);
        if (value < result) result = value;
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return (double)result;
}

double ocean_tensor_item(ocean_tensor_handle_t tensor) {
    if (!tensor) {
        ocean_tensor_fail("Tensor item received a null Tensor");
    }
    if (ocean_tensor_size(tensor) != 1) {
        ocean_tensor_fail("Tensor item requires exactly one element");
    }

    return ocean_tensor_get_flat(tensor, 0);
}

char *ocean_tensor_dtype_name(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor dtype on null handle");
    static const char *names[] = {
        "bool", "int8", "int16", "int32", "int64", "uint8",
        "uint16", "uint32", "uint64", "float16", "float32", "float64"
    };
    const char *name = names[tensor->dtype];
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying Tensor dtype");
    strcpy(result, name);
    return result;
}

bool ocean_tensor_is_contiguous(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor is_contiguous on null handle");
    size_t expected = 1;
    for (size_t axis = tensor->ndim; axis-- > 0;) {
        if (tensor->strides[axis] != expected) return false;
        if (tensor->shape[axis] != 0 && expected > SIZE_MAX / tensor->shape[axis]) {
            return false;
        }
        expected *= tensor->shape[axis];
    }
    return true;
}

ocean_tensor_handle_t ocean_tensor_contiguous(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor contiguous on null handle");
    if (ocean_tensor_is_contiguous(tensor)) return ocean_tensor_copy(tensor);
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t index = 0; index < cpu->size; ++index) {
        ocean_tensor_write_scalar(result, index, ocean_tensor_read_scalar(cpu, index));
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, result);
}

void ocean_tensor_fill(ocean_tensor_handle_t tensor, double value) {
    if (!tensor) ocean_tensor_fail("Tensor fill on null handle");
    ocean_tensor_backend_for_device(tensor->device)->fill(tensor, value);
}

static void ocean_tensor_fill_opencl(
    ocean_tensor_handle_t tensor,
    double value
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;

#if defined(CL_VERSION_1_2)
    unsigned char pattern[8] = {0};
    size_t pattern_size = tensor->item_size;

    switch (tensor->dtype) {
        case OCEAN_TENSOR_BOOL: {
            bool v = value != 0.0;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT8: {
            int8_t v = (int8_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT16: {
            int16_t v = (int16_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT32: {
            int32_t v = (int32_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_INT64: {
            int64_t v = (int64_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT8: {
            uint8_t v = (uint8_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT16: {
            uint16_t v = (uint16_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT32: {
            uint32_t v = (uint32_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_UINT64: {
            uint64_t v = (uint64_t)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_FLOAT16: {
            uint16_t v = ocean_tensor_float_to_half((float)value);
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_FLOAT32: {
            float v = (float)value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
        case OCEAN_TENSOR_FLOAT64: {
            double v = value;
            memcpy(pattern, &v, sizeof(v));
            break;
        }
    }

    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueFillBuffer(
            ocean_tensor_opencl.queue,
            tensor->gpu_data,
            pattern,
            pattern_size,
            0,
            bytes,
            0,
            NULL,
            &event
        ),
        "clEnqueueFillBuffer(fill)"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue),
        "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
#else
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    ocean_tensor_fill_cpu(cpu, value);
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
    ocean_tensor_release(cpu);
#endif
#else
    (void)tensor;
    (void)value;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
}


static size_t ocean_tensor_index_offset(
    const ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim
) {
    if (!tensor || !indices || ndim != tensor->ndim) {

        ocean_tensor_fail("Tensor index rank does not match Tensor rank");
    }
    size_t offset = 0;
    for (size_t axis = 0; axis < ndim; ++axis) {
        if (indices[axis] >= tensor->shape[axis]) {
            ocean_tensor_fail("Tensor index is out of bounds");
        }
        offset += indices[axis] * tensor->strides[axis];
    }
    return offset;
}

double ocean_tensor_get_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim
) {
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t offset = ocean_tensor_index_offset(cpu, indices, ndim);
    double result = (double)ocean_tensor_read_scalar(cpu, offset);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return result;
}

double ocean_tensor_get_flat(ocean_tensor_handle_t tensor, size_t index) {
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor");
    if (index >= tensor->size) ocean_tensor_fail("Tensor flat index is out of bounds");
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    double result = (double)ocean_tensor_read_scalar(cpu, index);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return result;
}

static void ocean_tensor_set_flat_long_double(
    ocean_tensor_handle_t tensor, size_t index, long double value
) {
    if (!tensor) ocean_tensor_fail("Tensor set received a null Tensor");
    if (index >= tensor->size) ocean_tensor_fail("Tensor flat index is out of bounds");
    if (tensor->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_write_scalar(tensor, index, value);
        return;
    }
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (ocean_tensor_cuda_set_scalar_fast(tensor, index, value)) return;
#endif
        ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
        ocean_tensor_set_flat_long_double(cpu, index, value);
        ocean_tensor_copy_into(tensor, cpu);
        ocean_tensor_release(cpu);
        return;
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t scalar_shape[1] = {1};
    ocean_tensor_handle_t scalar = ocean_tensor_alloc_uninitialized(
        scalar_shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    ocean_tensor_write_scalar(scalar, 0, value);
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE,
            index * tensor->item_size, tensor->item_size,
            scalar->cpu_data, 0, NULL, NULL
        ),
        "clEnqueueWriteBuffer(scalar)"
    );
    ocean_tensor_release(scalar);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
}

void ocean_tensor_set_flat(
    ocean_tensor_handle_t tensor, size_t index, double value
) {
    ocean_tensor_set_flat_long_double(tensor, index, (long double)value);
}

#define OCEAN_DEFINE_TYPED_SET_FLAT(name, c_type) \
void ocean_tensor_set_flat_##name( \
    ocean_tensor_handle_t tensor, size_t index, c_type value \
) { \
    ocean_tensor_set_flat_long_double(tensor, index, (long double)value); \
}

OCEAN_DEFINE_TYPED_SET_FLAT(bool, bool)
OCEAN_DEFINE_TYPED_SET_FLAT(i8, int8_t)
OCEAN_DEFINE_TYPED_SET_FLAT(i16, int16_t)
OCEAN_DEFINE_TYPED_SET_FLAT(i32, int32_t)
OCEAN_DEFINE_TYPED_SET_FLAT(i64, int64_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u8, uint8_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u16, uint16_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u32, uint32_t)
OCEAN_DEFINE_TYPED_SET_FLAT(u64, uint64_t)
OCEAN_DEFINE_TYPED_SET_FLAT(f16, float)
OCEAN_DEFINE_TYPED_SET_FLAT(f32, float)
OCEAN_DEFINE_TYPED_SET_FLAT(f64, double)

#undef OCEAN_DEFINE_TYPED_SET_FLAT

void ocean_tensor_set_nd(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim,
    double value
) {
    if (!tensor) ocean_tensor_fail("Tensor set received a null Tensor");
    if (tensor->device == OCEAN_TENSOR_CPU) {
        size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
        ocean_tensor_write_scalar(tensor, offset, (long double)value);
        return;
    }
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
        if (ocean_tensor_cuda_set_scalar_fast(tensor, offset, (long double)value)) {
            return;
        }
#endif
        ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
        ocean_tensor_set_nd(cpu, indices, ndim, value);
        ocean_tensor_copy_into(tensor, cpu);
        ocean_tensor_release(cpu);
        return;
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
    size_t scalar_shape[1] = {1};
    ocean_tensor_handle_t scalar = ocean_tensor_alloc_uninitialized(
        scalar_shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    ocean_tensor_write_scalar(scalar, 0, (long double)value);
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE,
            offset * tensor->item_size, tensor->item_size,
            scalar->cpu_data, 0, NULL, NULL
        ),
        "clEnqueueWriteBuffer(scalar)"
    );
    ocean_tensor_release(scalar);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
}


/*
 * Typed scalar accessors required by tensor_runtime.h and autograd_runtime.c.
 */
#define OCEAN_DEFINE_TYPED_GET_FLAT(name, c_type) \
c_type ocean_tensor_get_flat_##name( \
    ocean_tensor_handle_t tensor, size_t index \
) { \
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor"); \
    if (index >= tensor->size) { \
        ocean_tensor_fail("Tensor flat index is out of bounds"); \
    } \
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU \
        ? tensor : ocean_tensor_to(tensor, "cpu"); \
    c_type result = (c_type)ocean_tensor_read_scalar(cpu, index); \
    if (cpu != tensor) ocean_tensor_release(cpu); \
    return result; \
}

OCEAN_DEFINE_TYPED_GET_FLAT(bool, bool)
OCEAN_DEFINE_TYPED_GET_FLAT(i8, int8_t)
OCEAN_DEFINE_TYPED_GET_FLAT(i16, int16_t)
OCEAN_DEFINE_TYPED_GET_FLAT(i32, int32_t)
OCEAN_DEFINE_TYPED_GET_FLAT(i64, int64_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u8, uint8_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u16, uint16_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u32, uint32_t)
OCEAN_DEFINE_TYPED_GET_FLAT(u64, uint64_t)
OCEAN_DEFINE_TYPED_GET_FLAT(f16, float)
OCEAN_DEFINE_TYPED_GET_FLAT(f32, float)
OCEAN_DEFINE_TYPED_GET_FLAT(f64, double)

#undef OCEAN_DEFINE_TYPED_GET_FLAT

#define OCEAN_DEFINE_TYPED_GET_ND(name, c_type) \
c_type ocean_tensor_get_nd_##name( \
    ocean_tensor_handle_t tensor, \
    const size_t *indices, \
    size_t ndim \
) { \
    if (!tensor) ocean_tensor_fail("Tensor get received a null Tensor"); \
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU \
        ? tensor : ocean_tensor_to(tensor, "cpu"); \
    size_t offset = ocean_tensor_index_offset(cpu, indices, ndim); \
    c_type result = (c_type)ocean_tensor_read_scalar(cpu, offset); \
    if (cpu != tensor) ocean_tensor_release(cpu); \
    return result; \
}

OCEAN_DEFINE_TYPED_GET_ND(bool, bool)
OCEAN_DEFINE_TYPED_GET_ND(i8, int8_t)
OCEAN_DEFINE_TYPED_GET_ND(i16, int16_t)
OCEAN_DEFINE_TYPED_GET_ND(i32, int32_t)
OCEAN_DEFINE_TYPED_GET_ND(i64, int64_t)
OCEAN_DEFINE_TYPED_GET_ND(u8, uint8_t)
OCEAN_DEFINE_TYPED_GET_ND(u16, uint16_t)
OCEAN_DEFINE_TYPED_GET_ND(u32, uint32_t)
OCEAN_DEFINE_TYPED_GET_ND(u64, uint64_t)
OCEAN_DEFINE_TYPED_GET_ND(f16, float)
OCEAN_DEFINE_TYPED_GET_ND(f32, float)
OCEAN_DEFINE_TYPED_GET_ND(f64, double)

#undef OCEAN_DEFINE_TYPED_GET_ND

static void ocean_tensor_set_nd_long_double(
    ocean_tensor_handle_t tensor,
    const size_t *indices,
    size_t ndim,
    long double value
) {
    if (!tensor) ocean_tensor_fail("Tensor set received a null Tensor");

    if (tensor->device == OCEAN_TENSOR_CPU) {
        size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
        ocean_tensor_write_scalar(tensor, offset, value);
        return;
    }

    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
        if (ocean_tensor_cuda_set_scalar_fast(tensor, offset, value)) return;
#endif
        ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
        ocean_tensor_set_nd_long_double(cpu, indices, ndim, value);
        ocean_tensor_copy_into(tensor, cpu);
        ocean_tensor_release(cpu);
        return;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t offset = ocean_tensor_index_offset(tensor, indices, ndim);
    size_t scalar_shape[1] = {1};
    ocean_tensor_handle_t scalar = ocean_tensor_alloc_uninitialized(
        scalar_shape, 1, tensor->dtype, OCEAN_TENSOR_CPU
    );
    ocean_tensor_write_scalar(scalar, 0, value);
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE,
            offset * tensor->item_size, tensor->item_size,
            scalar->cpu_data, 0, NULL, NULL
        ),
        "clEnqueueWriteBuffer(scalar)"
    );
    ocean_tensor_release(scalar);
#else
    ocean_tensor_fail(
        "GPU backend is unavailable: rebuild with OpenCL support"
    );
#endif
}

#define OCEAN_DEFINE_TYPED_SET_ND(name, c_type) \
void ocean_tensor_set_nd_##name( \
    ocean_tensor_handle_t tensor, \
    const size_t *indices, \
    size_t ndim, \
    c_type value \
) { \
    ocean_tensor_set_nd_long_double( \
        tensor, indices, ndim, (long double)value \
    ); \
}

OCEAN_DEFINE_TYPED_SET_ND(bool, bool)
OCEAN_DEFINE_TYPED_SET_ND(i8, int8_t)
OCEAN_DEFINE_TYPED_SET_ND(i16, int16_t)
OCEAN_DEFINE_TYPED_SET_ND(i32, int32_t)
OCEAN_DEFINE_TYPED_SET_ND(i64, int64_t)
OCEAN_DEFINE_TYPED_SET_ND(u8, uint8_t)
OCEAN_DEFINE_TYPED_SET_ND(u16, uint16_t)
OCEAN_DEFINE_TYPED_SET_ND(u32, uint32_t)
OCEAN_DEFINE_TYPED_SET_ND(u64, uint64_t)
OCEAN_DEFINE_TYPED_SET_ND(f16, float)
OCEAN_DEFINE_TYPED_SET_ND(f32, float)
OCEAN_DEFINE_TYPED_SET_ND(f64, double)

#undef OCEAN_DEFINE_TYPED_SET_ND

double ocean_tensor_get_2d(ocean_tensor_handle_t tensor, int row, int col) {
    if (row < 0 || col < 0) ocean_tensor_fail("Tensor get index is out of bounds");
    size_t indices[2] = {(size_t)row, (size_t)col};
    return ocean_tensor_get_nd(tensor, indices, 2);
}

void ocean_tensor_set_2d(
    ocean_tensor_handle_t tensor, int row, int col, double value
) {
    if (row < 0 || col < 0) ocean_tensor_fail("Tensor set index is out of bounds");
    size_t indices[2] = {(size_t)row, (size_t)col};
    ocean_tensor_set_nd(tensor, indices, 2, value);
}

static ocean_tensor_handle_t ocean_tensor_matmul_cpu(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    size_t shape[2] = {left->shape[0], right->shape[1]};
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, 2, left->dtype, OCEAN_TENSOR_CPU
    );

    bool left_contiguous =
        left->strides[1] == 1 && left->strides[0] == left->shape[1];
    bool right_contiguous =
        right->strides[1] == 1 && right->strides[0] == right->shape[1];

    if (left_contiguous && right_contiguous) {
        size_t rows = left->shape[0];
        size_t inner = left->shape[1];
        size_t cols = right->shape[1];

        if (rows == 0 || inner == 0 || cols == 0) return result;

        const size_t block_rows = 32;
        const size_t block_inner = 64;
        const size_t block_cols = 128;

        if (left->dtype == OCEAN_TENSOR_FLOAT32) {
            const float *restrict a = (const float *)left->cpu_data;
            const float *restrict b = (const float *)right->cpu_data;
            float *restrict c = (float *)result->cpu_data;

            if (rows < 16 || inner < 32 || cols < 64) {
                for (size_t row = 0; row < rows; ++row) {
                    float *restrict c_row = c + row * cols;
                    const float *restrict a_row = a + row * inner;

                    for (size_t k = 0; k < inner; ++k) {
                        float av = a_row[k];
                        const float *restrict b_row = b + k * cols;
                        for (size_t col = 0; col < cols; ++col) {
                            c_row[col] += av * b_row[col];
                        }
                    }
                }
                return result;
            }

            for (size_t row0 = 0; row0 < rows; row0 += block_rows) {
                size_t row_end = row0 + block_rows < rows
                    ? row0 + block_rows : rows;

                for (size_t k0 = 0; k0 < inner; k0 += block_inner) {
                    size_t k_end = k0 + block_inner < inner
                        ? k0 + block_inner : inner;

                    for (size_t col0 = 0; col0 < cols; col0 += block_cols) {
                        size_t col_end = col0 + block_cols < cols
                            ? col0 + block_cols : cols;

                        for (size_t row = row0; row < row_end; ++row) {
                            float *restrict c_row = c + row * cols;
                            const float *restrict a_row = a + row * inner;

                            for (size_t k = k0; k < k_end; ++k) {
                                float av = a_row[k];
                                const float *restrict b_row = b + k * cols;

                                for (size_t col = col0; col < col_end; ++col) {
                                    c_row[col] += av * b_row[col];
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }

        if (left->dtype == OCEAN_TENSOR_FLOAT64) {
            const double *restrict a = (const double *)left->cpu_data;
            const double *restrict b = (const double *)right->cpu_data;
            double *restrict c = (double *)result->cpu_data;

            if (rows < 16 || inner < 32 || cols < 64) {
                for (size_t row = 0; row < rows; ++row) {
                    double *restrict c_row = c + row * cols;
                    const double *restrict a_row = a + row * inner;

                    for (size_t k = 0; k < inner; ++k) {
                        double av = a_row[k];
                        const double *restrict b_row = b + k * cols;
                        for (size_t col = 0; col < cols; ++col) {
                            c_row[col] += av * b_row[col];
                        }
                    }
                }
                return result;
            }

            for (size_t row0 = 0; row0 < rows; row0 += block_rows) {
                size_t row_end = row0 + block_rows < rows
                    ? row0 + block_rows : rows;

                for (size_t k0 = 0; k0 < inner; k0 += block_inner) {
                    size_t k_end = k0 + block_inner < inner
                        ? k0 + block_inner : inner;

                    for (size_t col0 = 0; col0 < cols; col0 += block_cols) {
                        size_t col_end = col0 + block_cols < cols
                            ? col0 + block_cols : cols;

                        for (size_t row = row0; row < row_end; ++row) {
                            double *restrict c_row = c + row * cols;
                            const double *restrict a_row = a + row * inner;

                            for (size_t k = k0; k < k_end; ++k) {
                                double av = a_row[k];
                                const double *restrict b_row = b + k * cols;

                                for (size_t col = col0; col < col_end; ++col) {
                                    c_row[col] += av * b_row[col];
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }
    }

    for (size_t row = 0; row < left->shape[0]; ++row) {
        for (size_t col = 0; col < right->shape[1]; ++col) {
            long double sum = 0.0L;

            for (size_t k = 0; k < left->shape[1]; ++k) {
                size_t left_index =
                    row * left->strides[0] + k * left->strides[1];
                size_t right_index =
                    k * right->strides[0] + col * right->strides[1];

                sum += ocean_tensor_read_scalar(left, left_index)
                    * ocean_tensor_read_scalar(right, right_index);
            }

            ocean_tensor_write_scalar(
                result,
                row * result->strides[0] + col * result->strides[1],
                sum
            );
        }
    }

    return result;
}


#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include "std/tensor/tensor_opencl_matmul.inc"
#endif


static ocean_tensor_handle_t ocean_tensor_matmul_opencl(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (left->dtype != OCEAN_TENSOR_FLOAT32 && left->dtype != OCEAN_TENSOR_INT32) {
        /* Preserve GPU semantics for dtypes without a specialized kernel
           through a correct CPU fallback and upload of the result. */
        ocean_tensor_handle_t left_cpu = ocean_tensor_to(left, "cpu");
        ocean_tensor_handle_t right_cpu = ocean_tensor_to(right, "cpu");
        ocean_tensor_handle_t cpu_result = ocean_tensor_matmul_cpu(left_cpu, right_cpu);
        ocean_tensor_handle_t gpu_result = ocean_tensor_to(cpu_result, "gpu");
        ocean_tensor_release(left_cpu);
        ocean_tensor_release(right_cpu);
        ocean_tensor_release(cpu_result);
        return gpu_result;
    }

    size_t shape[2] = {left->shape[0], right->shape[1]};
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, 2, left->dtype, OCEAN_TENSOR_GPU
    );
    if (left->shape[0] == 0 || right->shape[1] == 0) {
        return result;
    }
    if (left->shape[0] > (size_t)INT32_MAX ||
        left->shape[1] > (size_t)INT32_MAX ||
        right->shape[1] > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor dimensions are too large for OpenCL kernel indexing");
    }
    if (left->dtype == OCEAN_TENSOR_FLOAT32 && left->shape[0] == 1) {
        ocean_tensor_opencl_matvec(left, right, result);
        return result;
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        left->dtype == OCEAN_TENSOR_INT32
            ? OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_INT32
            : OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_FLOAT32
    );
    int rows = (int)left->shape[0];
    int cols = (int)left->shape[1];
    int result_cols = (int)right->shape[1];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &left->gpu_data), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &right->gpu_data), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &result->gpu_data), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &cols), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &result_cols), "clSetKernelArg"
    );
    const size_t tile_size = 8u;
    size_t global_size[2] = {
        ((size_t)rows + tile_size - 1u) / tile_size * tile_size,
        ((size_t)result_cols + tile_size - 1u) / tile_size * tile_size,
    };
    size_t local_size[2] = {tile_size, tile_size};
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 2, NULL,
            global_size, local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    /* The queue is in-order.  Consumers (including a blocking download)
       provide the dependency, so avoid synchronizing the host here. */
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
    return result;
#else
    (void)left;
    (void)right;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}


/* ================= ND Tensor v0.2 ================= */
static size_t ocean_tensor_normalize_dim_v02(const ocean_tensor_handle_t tensor, int dim) {
    if (!tensor || tensor->ndim == 0) ocean_tensor_fail("Tensor dimension operation requires rank >= 1");
    long long rank = (long long)tensor->ndim;
    long long d = (long long)dim;
    if (d < 0) d += rank;
    if (d < 0 || d >= rank) ocean_tensor_fail("Tensor dimension is out of bounds");
    return (size_t)d;
}

ocean_tensor_handle_t ocean_tensor_reshape_3d(ocean_tensor_handle_t tensor, int d0, int d1, int d2) {
    if (d0 < 0 || d1 < 0 || d2 < 0) ocean_tensor_fail("Tensor reshape dimensions must be non-negative");
    size_t shape[3] = {(size_t)d0, (size_t)d1, (size_t)d2};
    return ocean_tensor_reshape(tensor, shape, 3);
}

ocean_tensor_handle_t ocean_tensor_reshape_4d(ocean_tensor_handle_t tensor, int d0, int d1, int d2, int d3) {
    if (d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0) ocean_tensor_fail("Tensor reshape dimensions must be non-negative");
    size_t shape[4] = {(size_t)d0, (size_t)d1, (size_t)d2, (size_t)d3};
    return ocean_tensor_reshape(tensor, shape, 4);
}

ocean_tensor_handle_t ocean_tensor_transpose_dims(ocean_tensor_handle_t tensor, int dim0, int dim1) {
    if (!tensor) ocean_tensor_fail("Tensor transpose_dims on null handle");
    size_t a = ocean_tensor_normalize_dim_v02(tensor, dim0);
    size_t b = ocean_tensor_normalize_dim_v02(tensor, dim1);
    if (a == b) return ocean_tensor_copy(tensor);
    int *axes = (int *)malloc(tensor->ndim * sizeof(int));
    if (!axes) ocean_tensor_fail("out of memory allocating transpose axes");
    for (size_t axis = 0; axis < tensor->ndim; ++axis) {
        axes[axis] = (int)axis;
    }
    axes[a] = (int)b;
    axes[b] = (int)a;
    ocean_tensor_handle_t result = ocean_tensor_permute(
        tensor, axes, tensor->ndim
    );
    free(axes);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_reduce_dim_v02(ocean_tensor_handle_t tensor, int dim, bool keepdim, bool mean) {
    if (!tensor) ocean_tensor_fail("Tensor reduction on null handle");
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t out_ndim = keepdim ? tensor->ndim : (tensor->ndim > 1 ? tensor->ndim - 1 : 1);
    size_t *shape = malloc(out_ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating reduction shape");
    if (keepdim) {
        for (size_t i=0;i<tensor->ndim;++i) shape[i] = i==axis ? 1 : tensor->shape[i];
    } else if (tensor->ndim == 1) {
        shape[0] = 1;
    } else {
        size_t j=0; for (size_t i=0;i<tensor->ndim;++i) if (i!=axis) shape[j++] = tensor->shape[i];
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32 &&
        axis == tensor->ndim - 1) {
        size_t width = tensor->shape[axis];
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, out_ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        if (width != 0 && tensor->size != 0) {
            if (width > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                free(shape);
                ocean_tensor_fail("GPU reduction dimension is too large for OpenCL");
            }
            ocean_tensor_opencl_rowwise(
                tensor,
                result,
                OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32,
                (int)width,
                0.0f,
                mean ? 1 : 0
            );
        } else if (tensor->size != 0) {
            ocean_tensor_backend_for_device(OCEAN_TENSOR_GPU)->zero(result);
        }
        free(shape);
        return result;
    }
#endif

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU ? tensor : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t out = ocean_tensor_alloc_zeros(shape, out_ndim, tensor->dtype, OCEAN_TENSOR_CPU);
    free(shape);
    size_t *coord = calloc(tensor->ndim, sizeof(size_t));
    if (!coord) ocean_tensor_fail("out of memory reducing Tensor");

    for (size_t linear=0; linear<cpu->size; ++linear) {
        size_t rem=linear;
        for (size_t i=cpu->ndim;i-- >0;) { size_t d=cpu->shape[i]; coord[i]=d?rem%d:0; rem=d?rem/d:0; }
        size_t out_linear=0;
        if (keepdim) {
            for (size_t i=0;i<cpu->ndim;++i) out_linear = out_linear*out->shape[i] + (i==axis?0:coord[i]);
        } else if (cpu->ndim > 1) {
            size_t j=0; for (size_t i=0;i<cpu->ndim;++i) if (i!=axis) out_linear = out_linear*out->shape[j++] + coord[i];
        }
        ocean_tensor_write_scalar(out, out_linear, ocean_tensor_read_scalar(out,out_linear)+ocean_tensor_read_scalar(cpu,linear));
    }
    if (mean && tensor->shape[axis]) {
        long double div=(long double)tensor->shape[axis];
        for (size_t i=0;i<out->size;++i) ocean_tensor_write_scalar(out,i,ocean_tensor_read_scalar(out,i)/div);
    }
    free(coord);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return ocean_tensor_restore_device(tensor, out);
}

ocean_tensor_handle_t ocean_tensor_sum_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim) { return ocean_tensor_reduce_dim_v02(tensor,dim,keepdim,false); }
ocean_tensor_handle_t ocean_tensor_mean_dim(ocean_tensor_handle_t tensor, int dim, bool keepdim) { return ocean_tensor_reduce_dim_v02(tensor,dim,keepdim,true); }

ocean_tensor_handle_t ocean_tensor_softmax(
    ocean_tensor_handle_t tensor,
    int dim
) {
    if (!tensor) ocean_tensor_fail("Tensor.softmax on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.softmax currently requires float32");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t axis_size = tensor->shape[axis];

#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t source = ocean_tensor_is_contiguous(tensor)
            ? tensor : ocean_tensor_contiguous(tensor);
        ocean_tensor_handle_t result = ocean_tensor_cuda_softmax(source);
        if (source != tensor) ocean_tensor_release(source);
        return result;
    }
#endif
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        if (axis_size != 0 && tensor->size != 0) {
            if (axis_size > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                ocean_tensor_fail("GPU softmax dimension is too large for OpenCL");
            }
            ocean_tensor_opencl_rowwise(
                tensor,
                result,
                OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_FLOAT32,
                (int)axis_size,
                0.0f,
                0
            );
        }
        return result;
    }
#endif

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor
        : ocean_tensor_to(tensor, "cpu");
    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= tensor->shape[i];
    for (size_t i = axis + 1; i < tensor->ndim; ++i) inner *= tensor->shape[i];
    if (axis_size == 0) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("Tensor.softmax cannot normalize an empty dimension");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
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
                float value = expf(
                    ocean_tensor_get_flat_f32(cpu, index) - max_value
                );
                ocean_tensor_write_scalar(result, index, value);
                denominator += (double)value;
            }
            if (!(denominator > 0.0)) {
                ocean_tensor_release(result);
                if (cpu != tensor) ocean_tensor_release(cpu);
                free(device);
                ocean_tensor_fail("softmax denominator is not positive");
            }
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                ocean_tensor_write_scalar(
                    result, index,
                    ocean_tensor_read_scalar(result, index) / denominator
                );
            }
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return ocean_tensor_restore_device(tensor, result);
}

ocean_tensor_handle_t ocean_tensor_causal_softmax(
    ocean_tensor_handle_t tensor
) {
    if (!tensor) ocean_tensor_fail("Tensor.causal_softmax on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.causal_softmax currently requires float32");
    }
    if (tensor->ndim != 4 || tensor->shape[2] != tensor->shape[3]) {
        ocean_tensor_fail(
            "Tensor.causal_softmax requires a [batch, heads, sequence, sequence] Tensor"
        );
    }
    size_t width = tensor->shape[3];
    if (width == 0) {
        ocean_tensor_fail("Tensor.causal_softmax cannot normalize an empty dimension");
    }
    size_t rows = tensor->size / width;
    if (rows > (size_t)INT32_MAX || width > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor.causal_softmax dimensions are too large");
    }

    ocean_tensor_handle_t contiguous = NULL;
    const ocean_tensor_handle_t source = !ocean_tensor_is_contiguous(tensor)
        ? (contiguous = ocean_tensor_contiguous(tensor))
        : tensor;

#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (source->device == OCEAN_TENSOR_BACKEND_CUDA) {
        ocean_tensor_handle_t result = ocean_tensor_cuda_causal_softmax(source);
        if (contiguous) ocean_tensor_release(contiguous);
        return result;
    }
#endif
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (source->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            source->shape, source->ndim, source->dtype, OCEAN_TENSOR_GPU
        );
        if (source->size != 0) {
            ocean_tensor_opencl_causal_softmax(
                source, result, (int)rows, (int)width
            );
        }
        if (contiguous) ocean_tensor_release(contiguous);
        return result;
    }
#endif

    ocean_tensor_handle_t cpu = source->device == OCEAN_TENSOR_CPU
        ? source : ocean_tensor_to(source, "cpu");
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t row = 0; row < rows; ++row) {
        size_t query = row % width;
        size_t offset = row * width;
        float max_value = -INFINITY;
        for (size_t column = 0; column <= query; ++column) {
            float value = ocean_tensor_get_flat_f32(cpu, offset + column);
            if (value > max_value) max_value = value;
        }
        double denominator = 0.0;
        for (size_t column = 0; column <= query; ++column) {
            denominator += expf(
                ocean_tensor_get_flat_f32(cpu, offset + column) - max_value
            );
        }
        if (!(denominator > 0.0)) {
            ocean_tensor_release(result);
            if (cpu != source) ocean_tensor_release(cpu);
            if (contiguous) ocean_tensor_release(contiguous);
            ocean_tensor_fail("causal softmax denominator is not positive");
        }
        for (size_t column = 0; column < width; ++column) {
            float value = column <= query
                ? (float)(expf(
                    ocean_tensor_get_flat_f32(cpu, offset + column)
                    - max_value
                ) / denominator)
                : 0.0f;
            ocean_tensor_write_scalar(result, offset + column, value);
        }
    }
    if (cpu != source) ocean_tensor_release(cpu);
    if (contiguous) ocean_tensor_release(contiguous);
    return ocean_tensor_restore_device(tensor, result);
}

ocean_tensor_handle_t ocean_tensor_layer_norm(
    ocean_tensor_handle_t tensor,
    int dim,
    double epsilon
) {
    if (!tensor) ocean_tensor_fail("Tensor.layer_norm on null handle");
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.layer_norm currently requires float32");
    }
    if (!(epsilon > 0.0)) {
        ocean_tensor_fail("LayerNorm epsilon must be positive");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t axis_size = tensor->shape[axis];

#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t source = ocean_tensor_is_contiguous(tensor)
            ? tensor : ocean_tensor_contiguous(tensor);
        ocean_tensor_handle_t result = ocean_tensor_cuda_layer_norm(
            source, epsilon
        );
        if (source != tensor) ocean_tensor_release(source);
        return result;
    }
#endif
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        if (axis_size != 0 && tensor->size != 0) {
            if (axis_size > (size_t)INT32_MAX) {
                ocean_tensor_release(result);
                ocean_tensor_fail("GPU LayerNorm dimension is too large for OpenCL");
            }
            ocean_tensor_opencl_rowwise(
                tensor,
                result,
                OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32,
                (int)axis_size,
                (float)epsilon,
                0
            );
        }
        return result;
    }
#endif

    char *device = ocean_tensor_device(tensor);
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor
        : ocean_tensor_to(tensor, "cpu");
    if (axis_size == 0) {
        if (cpu != tensor) ocean_tensor_release(cpu);
        free(device);
        ocean_tensor_fail("LayerNorm cannot normalize an empty dimension");
    }
    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= tensor->shape[i];
    for (size_t i = axis + 1; i < tensor->ndim; ++i) inner *= tensor->shape[i];
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        cpu->shape, cpu->ndim, cpu->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            double mean_value = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                mean_value += ocean_tensor_get_flat_f32(cpu, index);
            }
            mean_value /= (double)axis_size;
            double variance = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                double delta = ocean_tensor_get_flat_f32(cpu, index) - mean_value;
                variance += delta * delta;
            }
            double inverse_std = 1.0 / sqrt(variance / (double)axis_size + epsilon);
            for (size_t a = 0; a < axis_size; ++a) {
                size_t index = (o * axis_size + a) * inner + in;
                double value = ocean_tensor_get_flat_f32(cpu, index);
                long double normalized =
                    ((long double)value - (long double)mean_value) *
                    (long double)inverse_std;
                ocean_tensor_write_scalar(
                    result, index, normalized
                );
            }
        }
    }
    if (cpu != tensor) ocean_tensor_release(cpu);
    free(device);
    return ocean_tensor_restore_device(tensor, result);
}

ocean_tensor_handle_t ocean_tensor_layer_norm_affine(
    ocean_tensor_handle_t tensor,
    ocean_tensor_handle_t gamma,
    ocean_tensor_handle_t beta,
    int dim,
    double epsilon
) {
    if (!tensor || !gamma || !beta) {
        ocean_tensor_fail("Tensor.layer_norm_affine received a null Tensor");
    }
    if (tensor->dtype != OCEAN_TENSOR_FLOAT32 ||
        gamma->dtype != OCEAN_TENSOR_FLOAT32 ||
        beta->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor.layer_norm_affine currently requires float32");
    }
    if (tensor->device != gamma->device || tensor->device != beta->device) {
        ocean_tensor_fail(
            "Tensor.layer_norm_affine requires matching Tensor devices"
        );
    }
    if (!(epsilon > 0.0)) {
        ocean_tensor_fail("LayerNorm epsilon must be positive");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(tensor, dim);
    size_t axis_size = tensor->shape[axis];
    if (gamma->size != axis_size || beta->size != axis_size) {
        ocean_tensor_fail(
            "Tensor.layer_norm_affine gamma/beta size must match the normalized dimension"
        );
    }

#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA &&
        axis == tensor->ndim - 1) {
        ocean_tensor_handle_t source = ocean_tensor_is_contiguous(tensor)
            ? tensor : ocean_tensor_contiguous(tensor);
        ocean_tensor_handle_t contiguous_gamma = ocean_tensor_is_contiguous(gamma)
            ? gamma : ocean_tensor_contiguous(gamma);
        ocean_tensor_handle_t contiguous_beta = ocean_tensor_is_contiguous(beta)
            ? beta : ocean_tensor_contiguous(beta);
        ocean_tensor_handle_t result = ocean_tensor_cuda_layer_norm_affine(
            source, contiguous_gamma, contiguous_beta, epsilon
        );
        if (source != tensor) ocean_tensor_release(source);
        if (contiguous_gamma != gamma) ocean_tensor_release(contiguous_gamma);
        if (contiguous_beta != beta) ocean_tensor_release(contiguous_beta);
        return result;
    }
#endif

    ocean_tensor_handle_t normalized = ocean_tensor_layer_norm(
        tensor, dim, epsilon
    );
    ocean_tensor_handle_t scaled = ocean_tensor_binary(
        normalized, gamma, OCEAN_TENSOR_MUL
    );
    ocean_tensor_handle_t result = ocean_tensor_binary(
        scaled, beta, OCEAN_TENSOR_ADD
    );
    ocean_tensor_release(normalized);
    ocean_tensor_release(scaled);
    return result;
}

static bool ocean_tensor_sparse_score_precedes(
    float left_score,
    size_t left_index,
    float right_score,
    size_t right_index
) {
    return left_score > right_score ||
        (left_score == right_score && left_index < right_index);
}

ocean_tensor_handle_t ocean_tensor_sparse_attention(
    ocean_tensor_handle_t query,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t value,
    int top_k,
    double scale,
    int query_start,
    bool causal
) {
    if (!query || !key || !value) {
        ocean_tensor_fail("SparseAttention received a null Tensor");
    }
    if (query->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 ||
        value->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("SparseAttention currently requires float32 Tensors");
    }
    if (query->device != OCEAN_TENSOR_CPU ||
        key->device != OCEAN_TENSOR_CPU ||
        value->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail(
            "SparseAttention v0.1 is CPU-only; CUDA routing is not implemented yet"
        );
    }
    if (query->ndim != 4 || key->ndim != 4 || value->ndim != 4) {
        ocean_tensor_fail(
            "SparseAttention expects query, key, value with shape [batch, heads, sequence, head_dim]"
        );
    }
    if (top_k <= 0) {
        ocean_tensor_fail("SparseAttention top_k must be positive");
    }
    if (query_start < 0) {
        ocean_tensor_fail("SparseAttention query_start must be non-negative");
    }

    size_t batch = query->shape[0];
    size_t heads = query->shape[1];
    size_t query_length = query->shape[2];
    size_t head_dim = query->shape[3];
    size_t key_length = key->shape[2];
    if (key->shape[0] != batch || key->shape[1] != heads ||
        value->shape[0] != batch || value->shape[1] != heads ||
        key->shape[3] != head_dim || value->shape[2] != key_length ||
        value->shape[3] != head_dim) {
        ocean_tensor_fail(
            "SparseAttention query/key/value shapes are incompatible"
        );
    }
    if (head_dim == 0 || key_length == 0) {
        ocean_tensor_fail("SparseAttention cannot use an empty key dimension");
    }
    if ((size_t)query_start > SIZE_MAX - query_length) {
        ocean_tensor_fail("SparseAttention query position overflows size_t");
    }

    ocean_tensor_handle_t query_contiguous = ocean_tensor_is_contiguous(query)
        ? query : ocean_tensor_contiguous(query);
    ocean_tensor_handle_t key_contiguous = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    ocean_tensor_handle_t value_contiguous = ocean_tensor_is_contiguous(value)
        ? value : ocean_tensor_contiguous(value);
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        query->shape, query->ndim, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_CPU
    );

    size_t requested = (size_t)top_k;
    size_t *selected_indices = (size_t *)malloc(requested * sizeof(size_t));
    float *selected_scores = (float *)malloc(requested * sizeof(float));
    if (!selected_indices || !selected_scores) {
        free(selected_indices);
        free(selected_scores);
        if (query_contiguous != query) ocean_tensor_release(query_contiguous);
        if (key_contiguous != key) ocean_tensor_release(key_contiguous);
        if (value_contiguous != value) ocean_tensor_release(value_contiguous);
        ocean_tensor_release(result);
        ocean_tensor_fail("out of memory in SparseAttention top-k selection");
    }

    const float *query_data = (const float *)query_contiguous->cpu_data;
    const float *key_data = (const float *)key_contiguous->cpu_data;
    const float *value_data = (const float *)value_contiguous->cpu_data;
    float *result_data = (float *)result->cpu_data;
    float score_scale = scale == 0.0
        ? 1.0f / sqrtf((float)head_dim) : (float)scale;
    if (!(score_scale > 0.0f) || !isfinite(score_scale)) {
        free(selected_indices);
        free(selected_scores);
        if (query_contiguous != query) ocean_tensor_release(query_contiguous);
        if (key_contiguous != key) ocean_tensor_release(key_contiguous);
        if (value_contiguous != value) ocean_tensor_release(value_contiguous);
        ocean_tensor_release(result);
        ocean_tensor_fail("SparseAttention scale must be finite and positive");
    }

    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            for (size_t query_index = 0; query_index < query_length; ++query_index) {
                size_t selected_count = 0;
                size_t absolute_query = (size_t)query_start + query_index;
                size_t query_base =
                    ((batch_index * heads + head) * query_length + query_index)
                    * head_dim;
                for (size_t key_index = 0; key_index < key_length; ++key_index) {
                    if (causal && key_index > absolute_query) continue;
                    size_t key_base =
                        ((batch_index * heads + head) * key_length + key_index)
                        * head_dim;
                    float score = 0.0f;
                    for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                        score += query_data[query_base + dimension]
                            * key_data[key_base + dimension];
                    }
                    score *= score_scale;

                    size_t limit = requested < key_length ? requested : key_length;
                    if (selected_count < limit) {
                        size_t insert = selected_count;
                        while (insert > 0 && ocean_tensor_sparse_score_precedes(
                            score, key_index,
                            selected_scores[insert - 1], selected_indices[insert - 1]
                        )) {
                            selected_scores[insert] = selected_scores[insert - 1];
                            selected_indices[insert] = selected_indices[insert - 1];
                            --insert;
                        }
                        selected_scores[insert] = score;
                        selected_indices[insert] = key_index;
                        ++selected_count;
                    } else if (ocean_tensor_sparse_score_precedes(
                        score, key_index,
                        selected_scores[limit - 1], selected_indices[limit - 1]
                    )) {
                        size_t insert = limit - 1;
                        while (insert > 0 && ocean_tensor_sparse_score_precedes(
                            score, key_index,
                            selected_scores[insert - 1], selected_indices[insert - 1]
                        )) {
                            selected_scores[insert] = selected_scores[insert - 1];
                            selected_indices[insert] = selected_indices[insert - 1];
                            --insert;
                        }
                        selected_scores[insert] = score;
                        selected_indices[insert] = key_index;
                    }
                }

                float maximum = -INFINITY;
                for (size_t selected = 0; selected < selected_count; ++selected) {
                    if (selected_scores[selected] > maximum) {
                        maximum = selected_scores[selected];
                    }
                }
                float denominator = 0.0f;
                for (size_t selected = 0; selected < selected_count; ++selected) {
                    selected_scores[selected] = expf(
                        selected_scores[selected] - maximum
                    );
                    denominator += selected_scores[selected];
                }
                size_t result_base = query_base;
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    float output = 0.0f;
                    for (size_t selected = 0; selected < selected_count; ++selected) {
                        size_t value_base =
                            ((batch_index * heads + head) * key_length
                                + selected_indices[selected]) * head_dim;
                        output += (selected_scores[selected] / denominator)
                            * value_data[value_base + dimension];
                    }
                    result_data[result_base + dimension] = output;
                }
            }
        }
    }

    free(selected_indices);
    free(selected_scores);
    if (query_contiguous != query) ocean_tensor_release(query_contiguous);
    if (key_contiguous != key) ocean_tensor_release(key_contiguous);
    if (value_contiguous != value) ocean_tensor_release(value_contiguous);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_sparse_attention_blocked_impl(
    ocean_tensor_handle_t query,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t value,
    int top_k,
    int top_blocks,
    int block_size,
    double scale,
    int query_start,
    bool causal,
    const float *precomputed_summaries
) {
    if (!query || !key || !value) {
        ocean_tensor_fail("Blocked SparseAttention received a null Tensor");
    }
    if (query->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 ||
        value->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Blocked SparseAttention requires float32 Tensors");
    }
    if (query->device != OCEAN_TENSOR_CPU ||
        key->device != OCEAN_TENSOR_CPU ||
        value->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail(
            "Blocked SparseAttention v0.2 is CPU-only; CUDA routing is not implemented yet"
        );
    }
    if (query->ndim != 4 || key->ndim != 4 || value->ndim != 4) {
        ocean_tensor_fail(
            "Blocked SparseAttention expects [batch, heads, sequence, head_dim]"
        );
    }
    if (top_k <= 0 || top_blocks <= 0 || block_size <= 0) {
        ocean_tensor_fail(
            "Blocked SparseAttention top_k, top_blocks, and block_size must be positive"
        );
    }
    if (query_start < 0) {
        ocean_tensor_fail("Blocked SparseAttention query_start must be non-negative");
    }

    size_t batch = query->shape[0];
    size_t heads = query->shape[1];
    size_t query_length = query->shape[2];
    size_t head_dim = query->shape[3];
    size_t key_length = key->shape[2];
    if (key->shape[0] != batch || key->shape[1] != heads ||
        value->shape[0] != batch || value->shape[1] != heads ||
        key->shape[3] != head_dim || value->shape[2] != key_length ||
        value->shape[3] != head_dim) {
        ocean_tensor_fail("Blocked SparseAttention shapes are incompatible");
    }
    if (head_dim == 0 || key_length == 0) {
        ocean_tensor_fail("Blocked SparseAttention cannot use an empty key dimension");
    }
    if ((size_t)query_start > SIZE_MAX - query_length) {
        ocean_tensor_fail("Blocked SparseAttention query position overflows size_t");
    }

    size_t block_width = (size_t)block_size;
    size_t block_count = (key_length + block_width - 1) / block_width;
    size_t block_limit = (size_t)top_blocks < block_count
        ? (size_t)top_blocks : block_count;
    size_t token_limit = (size_t)top_k < key_length
        ? (size_t)top_k : key_length;
    if (block_count == 0 || block_limit == 0 || token_limit == 0) {
        ocean_tensor_fail("Blocked SparseAttention produced an empty routing domain");
    }
    if (batch != 0 && heads > SIZE_MAX / batch) {
        ocean_tensor_fail("Blocked SparseAttention summary shape overflows size_t");
    }
    size_t summary_count = batch * heads;
    if (block_count != 0 && summary_count > SIZE_MAX / block_count) {
        ocean_tensor_fail("Blocked SparseAttention summary shape overflows size_t");
    }
    summary_count *= block_count;
    if (head_dim != 0 && summary_count > SIZE_MAX / head_dim) {
        ocean_tensor_fail("Blocked SparseAttention summary shape overflows size_t");
    }
    summary_count *= head_dim;

    ocean_tensor_handle_t query_contiguous = ocean_tensor_is_contiguous(query)
        ? query : ocean_tensor_contiguous(query);
    ocean_tensor_handle_t key_contiguous = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    ocean_tensor_handle_t value_contiguous = ocean_tensor_is_contiguous(value)
        ? value : ocean_tensor_contiguous(value);
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        query->shape, query->ndim, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_CPU
    );
    bool owns_summaries = precomputed_summaries == NULL;
    float *summaries = (float *)precomputed_summaries;
    if (owns_summaries) {
        summaries = (float *)calloc(summary_count, sizeof(float));
    }
    size_t *selected_blocks = (size_t *)malloc(block_limit * sizeof(size_t));
    float *selected_block_scores =
        (float *)malloc(block_limit * sizeof(float));
    size_t *selected_indices = (size_t *)malloc(token_limit * sizeof(size_t));
    float *selected_scores = (float *)malloc(token_limit * sizeof(float));
    if (!summaries || !selected_blocks || !selected_block_scores ||
        !selected_indices || !selected_scores) {
        if (owns_summaries) free(summaries);
        free(selected_blocks);
        free(selected_block_scores);
        free(selected_indices);
        free(selected_scores);
        if (query_contiguous != query) ocean_tensor_release(query_contiguous);
        if (key_contiguous != key) ocean_tensor_release(key_contiguous);
        if (value_contiguous != value) ocean_tensor_release(value_contiguous);
        ocean_tensor_release(result);
        ocean_tensor_fail("out of memory in blocked SparseAttention");
    }

    const float *query_data = (const float *)query_contiguous->cpu_data;
    const float *key_data = (const float *)key_contiguous->cpu_data;
    const float *value_data = (const float *)value_contiguous->cpu_data;
    float *result_data = (float *)result->cpu_data;
    if (owns_summaries) {
        for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
            for (size_t head = 0; head < heads; ++head) {
                for (size_t block = 0; block < block_count; ++block) {
                    size_t start = block * block_width;
                    size_t end = start + block_width;
                    if (end > key_length) end = key_length;
                    size_t count = end - start;
                    size_t summary_base =
                        ((batch_index * heads + head) * block_count + block)
                        * head_dim;
                    for (size_t key_index = start; key_index < end; ++key_index) {
                        size_t key_base =
                            ((batch_index * heads + head) * key_length + key_index)
                            * head_dim;
                        for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                            summaries[summary_base + dimension] +=
                                key_data[key_base + dimension] / (float)count;
                        }
                    }
                }
            }
        }
    }

    float score_scale = scale == 0.0
        ? 1.0f / sqrtf((float)head_dim) : (float)scale;
    if (!(score_scale > 0.0f) || !isfinite(score_scale)) {
        if (owns_summaries) free(summaries);
        free(selected_blocks);
        free(selected_block_scores);
        free(selected_indices);
        free(selected_scores);
        if (query_contiguous != query) ocean_tensor_release(query_contiguous);
        if (key_contiguous != key) ocean_tensor_release(key_contiguous);
        if (value_contiguous != value) ocean_tensor_release(value_contiguous);
        ocean_tensor_release(result);
        ocean_tensor_fail("Blocked SparseAttention scale must be finite and positive");
    }

    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            for (size_t query_index = 0; query_index < query_length; ++query_index) {
                size_t absolute_query = (size_t)query_start + query_index;
                size_t query_base =
                    ((batch_index * heads + head) * query_length + query_index)
                    * head_dim;
                size_t selected_block_count = 0;
                for (size_t block = 0; block < block_count; ++block) {
                    size_t start = block * block_width;
                    size_t end = start + block_width;
                    if (end > key_length) end = key_length;
                    size_t visible_end = end;
                    if (causal && visible_end > absolute_query + 1) {
                        visible_end = absolute_query + 1;
                    }
                    if (visible_end <= start) continue;

                    size_t visible_count = visible_end - start;
                    float block_score = 0.0f;
                    if (visible_count == end - start) {
                        size_t summary_base =
                            ((batch_index * heads + head) * block_count + block)
                            * head_dim;
                        for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                            block_score += query_data[query_base + dimension]
                                * summaries[summary_base + dimension];
                        }
                    } else {
                        for (size_t key_index = start; key_index < visible_end; ++key_index) {
                            size_t key_base =
                                ((batch_index * heads + head) * key_length + key_index)
                                * head_dim;
                            float score = 0.0f;
                            for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                                score += query_data[query_base + dimension]
                                    * key_data[key_base + dimension];
                            }
                            block_score += score / (float)visible_count;
                        }
                    }
                    block_score *= score_scale;

                    if (selected_block_count < block_limit) {
                        size_t insert = selected_block_count;
                        while (insert > 0 && ocean_tensor_sparse_score_precedes(
                            block_score, block,
                            selected_block_scores[insert - 1],
                            selected_blocks[insert - 1]
                        )) {
                            selected_block_scores[insert] = selected_block_scores[insert - 1];
                            selected_blocks[insert] = selected_blocks[insert - 1];
                            --insert;
                        }
                        selected_block_scores[insert] = block_score;
                        selected_blocks[insert] = block;
                        ++selected_block_count;
                    } else if (ocean_tensor_sparse_score_precedes(
                        block_score, block,
                        selected_block_scores[block_limit - 1],
                        selected_blocks[block_limit - 1]
                    )) {
                        size_t insert = block_limit - 1;
                        while (insert > 0 && ocean_tensor_sparse_score_precedes(
                            block_score, block,
                            selected_block_scores[insert - 1],
                            selected_blocks[insert - 1]
                        )) {
                            selected_block_scores[insert] = selected_block_scores[insert - 1];
                            selected_blocks[insert] = selected_blocks[insert - 1];
                            --insert;
                        }
                        selected_block_scores[insert] = block_score;
                        selected_blocks[insert] = block;
                    }
                }

                size_t selected_count = 0;
                for (size_t selected_block = 0;
                     selected_block < selected_block_count;
                     ++selected_block) {
                    size_t block = selected_blocks[selected_block];
                    size_t start = block * block_width;
                    size_t end = start + block_width;
                    if (end > key_length) end = key_length;
                    if (causal && end > absolute_query + 1) {
                        end = absolute_query + 1;
                    }
                    for (size_t key_index = start; key_index < end; ++key_index) {
                        size_t key_base =
                            ((batch_index * heads + head) * key_length + key_index)
                            * head_dim;
                        float token_score = 0.0f;
                        for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                            token_score += query_data[query_base + dimension]
                                * key_data[key_base + dimension];
                        }
                        token_score *= score_scale;
                        if (selected_count < token_limit) {
                            size_t insert = selected_count;
                            while (insert > 0 && ocean_tensor_sparse_score_precedes(
                                token_score, key_index,
                                selected_scores[insert - 1],
                                selected_indices[insert - 1]
                            )) {
                                selected_scores[insert] = selected_scores[insert - 1];
                                selected_indices[insert] = selected_indices[insert - 1];
                                --insert;
                            }
                            selected_scores[insert] = token_score;
                            selected_indices[insert] = key_index;
                            ++selected_count;
                        } else if (ocean_tensor_sparse_score_precedes(
                            token_score, key_index,
                            selected_scores[token_limit - 1],
                            selected_indices[token_limit - 1]
                        )) {
                            size_t insert = token_limit - 1;
                            while (insert > 0 && ocean_tensor_sparse_score_precedes(
                                token_score, key_index,
                                selected_scores[insert - 1],
                                selected_indices[insert - 1]
                            )) {
                                selected_scores[insert] = selected_scores[insert - 1];
                                selected_indices[insert] = selected_indices[insert - 1];
                                --insert;
                            }
                            selected_scores[insert] = token_score;
                            selected_indices[insert] = key_index;
                        }
                    }
                }

                float maximum = -INFINITY;
                for (size_t selected = 0; selected < selected_count; ++selected) {
                    if (selected_scores[selected] > maximum) {
                        maximum = selected_scores[selected];
                    }
                }
                float denominator = 0.0f;
                for (size_t selected = 0; selected < selected_count; ++selected) {
                    selected_scores[selected] = expf(
                        selected_scores[selected] - maximum
                    );
                    denominator += selected_scores[selected];
                }
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    float output = 0.0f;
                    for (size_t selected = 0; selected < selected_count; ++selected) {
                        size_t value_base =
                            ((batch_index * heads + head) * key_length
                                + selected_indices[selected]) * head_dim;
                        output += (selected_scores[selected] / denominator)
                            * value_data[value_base + dimension];
                    }
                    result_data[query_base + dimension] = output;
                }
            }
        }
    }

    if (owns_summaries) free(summaries);
    free(selected_blocks);
    free(selected_block_scores);
    free(selected_indices);
    free(selected_scores);
    if (query_contiguous != query) ocean_tensor_release(query_contiguous);
    if (key_contiguous != key) ocean_tensor_release(key_contiguous);
    if (value_contiguous != value) ocean_tensor_release(value_contiguous);
    return result;
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_blocked(
    ocean_tensor_handle_t query,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t value,
    int top_k,
    int top_blocks,
    int block_size,
    double scale,
    int query_start,
    bool causal
) {
    return ocean_tensor_sparse_attention_blocked_impl(
        query, key, value, top_k, top_blocks, block_size,
        scale, query_start, causal, NULL
    );
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_build_summaries(
    ocean_tensor_handle_t key,
    int block_size
) {
    if (!key || key->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->device != OCEAN_TENSOR_CPU || key->ndim != 4 || block_size <= 0) {
        ocean_tensor_fail(
            "SparseAttention summaries require a CPU float32 key Tensor [B,H,K,D]"
        );
    }
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    if (key_length == 0 || head_dim == 0) {
        ocean_tensor_fail("SparseAttention summaries cannot use empty key dimensions");
    }
    size_t width = (size_t)block_size;
    size_t block_count = (key_length + width - 1) / width;
    size_t shape[4] = {batch, heads, block_count, head_dim};
    ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, 4, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_CPU
    );
    const float *source = (const float *)contiguous->cpu_data;
    float *destination = (float *)result->cpu_data;
    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            for (size_t block = 0; block < block_count; ++block) {
                size_t start = block * width;
                size_t end = start + width;
                if (end > key_length) end = key_length;
                size_t count = end - start;
                size_t summary_base =
                    ((batch_index * heads + head) * block_count + block)
                    * head_dim;
                for (size_t key_index = start; key_index < end; ++key_index) {
                    size_t source_base =
                        ((batch_index * heads + head) * key_length + key_index)
                        * head_dim;
                    for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                        destination[summary_base + dimension] +=
                            source[source_base + dimension] / (float)count;
                    }
                }
            }
        }
    }
    if (contiguous != key) ocean_tensor_release(contiguous);
    return result;
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_build_summaries_active(
    ocean_tensor_handle_t key,
    int active_length,
    int block_size
) {
    if (!key || key->dtype != OCEAN_TENSOR_FLOAT32 || key->ndim != 4 ||
        active_length <= 0 || block_size <= 0 ||
        (size_t)active_length > key->shape[2]) {
        ocean_tensor_fail(
            "Active SparseAttention summaries require float32 key [B,H,K,D]"
        );
    }
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    if (key_length == 0 || head_dim == 0) {
        ocean_tensor_fail("Active SparseAttention summaries cannot be empty");
    }
    size_t width = (size_t)block_size;
    size_t block_count = (key_length + width - 1) / width;
    size_t shape[4] = {batch, heads, block_count, head_dim};
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, 4, OCEAN_TENSOR_FLOAT32, key->device
    );
    if (key->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (!ocean_tensor_is_contiguous(key) ||
            batch > (size_t)INT_MAX || heads > (size_t)INT_MAX ||
            key_length > (size_t)INT_MAX ||
            head_dim > 128 || block_count > (size_t)INT_MAX) {
            ocean_tensor_release(result);
            ocean_tensor_fail(
                "CUDA active SparseAttention summary metadata is unsupported"
            );
        }
        ocean_cuda_sparse_build_summaries(
            key->cuda_data, result->cuda_data,
            (int)batch, (int)heads, (int)key_length, active_length,
            (int)head_dim, block_size
        );
        return result;
#else
        ocean_tensor_release(result);
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (key->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "Active SparseAttention summaries currently require CPU or CUDA"
        );
    }
    ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    const float *source = (const float *)contiguous->cpu_data;
    float *destination = (float *)result->cpu_data;
    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            for (size_t block = 0; block < block_count; ++block) {
                size_t start = block * width;
                size_t end = start + width;
                if (end > (size_t)active_length) end = (size_t)active_length;
                if (start >= end) continue;
                size_t count = end - start;
                size_t summary_base =
                    ((batch_index * heads + head) * block_count + block)
                    * head_dim;
                for (size_t key_index = start; key_index < end; ++key_index) {
                    size_t source_base =
                        ((batch_index * heads + head) * key_length + key_index)
                        * head_dim;
                    for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                        destination[summary_base + dimension] +=
                            source[source_base + dimension] / (float)count;
                    }
                }
            }
        }
    }
    if (contiguous != key) ocean_tensor_release(contiguous);
    return result;
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_blocked_cached(
    ocean_tensor_handle_t query,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t value,
    ocean_tensor_handle_t summaries,
    int top_k,
    int top_blocks,
    int block_size,
    double scale,
    int query_start,
    bool causal
) {
    if (!query || !key || !value || !summaries) {
        ocean_tensor_fail("Cached SparseAttention received a null Tensor");
    }
    if (summaries->dtype != OCEAN_TENSOR_FLOAT32 || summaries->ndim != 4) {
        ocean_tensor_fail(
            "Cached SparseAttention requires float32 summaries [B,H,blocks,D]"
        );
    }
    if (key->ndim != 4 || block_size <= 0) {
        ocean_tensor_fail("Cached SparseAttention received invalid key metadata");
    }
    size_t expected_blocks =
        (key->shape[2] + (size_t)block_size - 1) / (size_t)block_size;
    if (summaries->shape[0] != key->shape[0] ||
        summaries->shape[1] != key->shape[1] ||
        summaries->shape[2] != expected_blocks ||
        summaries->shape[3] != key->shape[3]) {
        ocean_tensor_fail("Cached SparseAttention summaries do not match key metadata");
    }
    if (query->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        size_t head_dim = key->shape[3];
        if (key->device != query->device || value->device != query->device ||
            summaries->device != query->device ||
            query->dtype != OCEAN_TENSOR_FLOAT32 ||
            key->dtype != OCEAN_TENSOR_FLOAT32 ||
            value->dtype != OCEAN_TENSOR_FLOAT32 ||
            query->ndim != 4 || value->ndim != 4 ||
            query->shape[0] != key->shape[0] ||
            query->shape[1] != key->shape[1] ||
            value->shape[0] != key->shape[0] ||
            value->shape[1] != key->shape[1] ||
            value->shape[2] != key->shape[2] ||
            query->shape[3] != key->shape[3] ||
            value->shape[3] != key->shape[3] ||
            query->shape[2] == 0 || key->shape[2] == 0 ||
            head_dim > 128 ||
            query->shape[0] > (size_t)INT_MAX ||
            query->shape[1] > (size_t)INT_MAX ||
            query->shape[2] > (size_t)INT_MAX ||
            key->shape[2] > (size_t)INT_MAX ||
            expected_blocks > (size_t)INT_MAX ||
            top_k <= 0 || top_blocks <= 0 || query_start < 0 ||
            !ocean_tensor_is_contiguous(query) ||
            !ocean_tensor_is_contiguous(key) ||
            !ocean_tensor_is_contiguous(value) ||
            !ocean_tensor_is_contiguous(summaries)) {
            ocean_tensor_fail("CUDA cached SparseAttention metadata mismatch");
        }
        float score_scale = scale == 0.0
            ? 1.0f / sqrtf((float)head_dim) : (float)scale;
        if (!(score_scale > 0.0f) || !isfinite(score_scale)) {
            ocean_tensor_fail("Cached SparseAttention scale must be positive");
        }
        if ((size_t)query_start > key->shape[2] ||
            query->shape[2] > key->shape[2] - (size_t)query_start) {
            ocean_tensor_fail("CUDA cached SparseAttention query range is invalid");
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            query->shape, query->ndim, OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_BACKEND_CUDA
        );
        ocean_cuda_sparse_attention(
            query->cuda_data, key->cuda_data, value->cuda_data,
            summaries->cuda_data, result->cuda_data,
            (int)query->shape[0], (int)query->shape[1],
            (int)query->shape[2], (int)key->shape[2], (int)key->shape[2],
            (int)head_dim,
            (int)expected_blocks, top_k, top_blocks, block_size,
            score_scale, query_start, causal ? 1 : 0
        );
        return result;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (summaries->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail(
            "Cached SparseAttention currently supports CPU or CUDA only"
        );
    }
    ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(summaries)
        ? summaries : ocean_tensor_contiguous(summaries);
    ocean_tensor_handle_t result = ocean_tensor_sparse_attention_blocked_impl(
        query, key, value, top_k, top_blocks, block_size,
        scale, query_start, causal, (const float *)contiguous->cpu_data
    );
    if (contiguous != summaries) ocean_tensor_release(contiguous);
    return result;
}

void ocean_tensor_sparse_attention_update_summary(
    ocean_tensor_handle_t summaries,
    ocean_tensor_handle_t key,
    int position,
    int block_size
) {
    if (!summaries || !key ||
        summaries->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 ||
        summaries->ndim != 4 || key->ndim != 4 ||
        block_size <= 0 || position < 0) {
        ocean_tensor_fail(
            "SparseAttention summary update requires CPU float32 [B,H,K,D] tensors"
        );
    }
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    if (key_length == 0 || head_dim == 0 || (size_t)position >= key_length) {
        ocean_tensor_fail("SparseAttention summary update position is out of bounds");
    }
    size_t width = (size_t)block_size;
    size_t block_count = (key_length + width - 1) / width;
    if (summaries->shape[0] != batch || summaries->shape[1] != heads ||
        summaries->shape[2] != block_count || summaries->shape[3] != head_dim ||
        !ocean_tensor_is_contiguous(summaries)) {
        ocean_tensor_fail(
            "SparseAttention summary update metadata mismatch"
        );
    }

    if (key->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (summaries->device != key->device ||
            !ocean_tensor_is_contiguous(key) ||
            batch > (size_t)INT_MAX || heads > (size_t)INT_MAX ||
            key_length > (size_t)INT_MAX || block_count > (size_t)INT_MAX ||
            head_dim > 128) {
            ocean_tensor_fail("CUDA SparseAttention summary update metadata mismatch");
        }
        ocean_cuda_sparse_update_summary(
            key->cuda_data, summaries->cuda_data,
            (int)batch, (int)heads, (int)key_length, (int)block_count,
            (int)key_length, (int)head_dim, block_size, position
        );
        return;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (key->device != OCEAN_TENSOR_CPU ||
        summaries->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail(
            "SparseAttention summary update currently supports CPU or CUDA only"
        );
    }

    ocean_tensor_handle_t contiguous_key = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    const float *key_data = (const float *)contiguous_key->cpu_data;
    float *summary_data = (float *)summaries->cpu_data;
    size_t block = (size_t)position / width;
    size_t start = block * width;
    size_t end = start + width;
    if (end > key_length) end = key_length;
    size_t count = end - start;
    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            size_t summary_base =
                ((batch_index * heads + head) * block_count + block)
                * head_dim;
            memset(
                summary_data + summary_base,
                0,
                head_dim * sizeof(float)
            );
            for (size_t key_index = start; key_index < end; ++key_index) {
                size_t key_base =
                    ((batch_index * heads + head) * key_length + key_index)
                    * head_dim;
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    summary_data[summary_base + dimension] +=
                        key_data[key_base + dimension] / (float)count;
                }
            }
        }
    }
    if (contiguous_key != key) ocean_tensor_release(contiguous_key);
}

void ocean_tensor_sparse_attention_update_summary_active(
    ocean_tensor_handle_t summaries,
    ocean_tensor_handle_t key,
    int active_length,
    int position,
    int block_size
) {
    if (!summaries || !key ||
        summaries->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 ||
        summaries->ndim != 4 || key->ndim != 4 ||
        active_length <= 0 || position < 0 || block_size <= 0) {
        ocean_tensor_fail(
            "Active SparseAttention summary update requires float32 [B,H,K,D] tensors"
        );
    }
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    if (key_length == 0 || head_dim == 0 ||
        (size_t)active_length > key_length ||
        (size_t)position >= (size_t)active_length) {
        ocean_tensor_fail(
            "Active SparseAttention summary update position is out of bounds"
        );
    }
    size_t width = (size_t)block_size;
    size_t block_count = (key_length + width - 1) / width;
    if (summaries->shape[0] != batch || summaries->shape[1] != heads ||
        summaries->shape[2] != block_count || summaries->shape[3] != head_dim ||
        summaries->device != key->device ||
        !ocean_tensor_is_contiguous(summaries)) {
        ocean_tensor_fail(
            "Active SparseAttention summary update metadata mismatch"
        );
    }

    if (key->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (!ocean_tensor_is_contiguous(key) ||
            batch > (size_t)INT_MAX || heads > (size_t)INT_MAX ||
            key_length > (size_t)INT_MAX || block_count > (size_t)INT_MAX ||
            head_dim > 128) {
            ocean_tensor_fail(
                "CUDA active SparseAttention summary update metadata mismatch"
            );
        }
        ocean_cuda_sparse_update_summary(
            key->cuda_data, summaries->cuda_data,
            (int)batch, (int)heads, (int)key_length, (int)block_count,
            active_length, (int)head_dim, block_size, position
        );
        return;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (key->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail(
            "Active SparseAttention summary update supports CPU or CUDA only"
        );
    }

    ocean_tensor_handle_t contiguous_key = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    const float *key_data = (const float *)contiguous_key->cpu_data;
    float *summary_data = (float *)summaries->cpu_data;
    size_t block = (size_t)position / width;
    size_t start = block * width;
    size_t end = start + width;
    if (end > (size_t)active_length) end = (size_t)active_length;
    size_t count = end - start;
    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            size_t summary_base =
                ((batch_index * heads + head) * block_count + block)
                * head_dim;
            memset(summary_data + summary_base, 0, head_dim * sizeof(float));
            for (size_t key_index = start; key_index < end; ++key_index) {
                size_t key_base =
                    ((batch_index * heads + head) * key_length + key_index)
                    * head_dim;
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    summary_data[summary_base + dimension] +=
                        key_data[key_base + dimension] / (float)count;
                }
            }
        }
    }
    if (contiguous_key != key) ocean_tensor_release(contiguous_key);
}

static bool ocean_tensor_sparse_route_contains(
    const int32_t *route,
    size_t count,
    int32_t block
) {
    for (size_t index = 0; index < count; ++index) {
        if (route[index] == block) return true;
    }
    return false;
}

static void ocean_tensor_sparse_route_cpu(
    ocean_tensor_handle_t route,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t summaries,
    int active_length,
    int summary_window,
    int semantic_blocks,
    int block_size,
    int random_seed
) {
    ocean_tensor_handle_t contiguous_key = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    ocean_tensor_handle_t contiguous_summaries =
        ocean_tensor_is_contiguous(summaries)
        ? summaries : ocean_tensor_contiguous(summaries);
    const float *key_data = (const float *)contiguous_key->cpu_data;
    const float *summary_data = (const float *)contiguous_summaries->cpu_data;
    int32_t *route_data = (int32_t *)route->cpu_data;
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    size_t summary_blocks = summaries->shape[2];
    size_t route_width = (size_t)semantic_blocks + 1u;
    int end = active_length;
    if (end > (int)key_length) end = (int)key_length;
    int start = end - summary_window;
    if (start < 0) start = 0;
    int recent_count = end - start;
    size_t block_count = ((size_t)active_length + (size_t)block_size - 1u) /
        (size_t)block_size;
    if (block_count > summary_blocks) block_count = summary_blocks;

    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            size_t route_base =
                (batch_index * heads + head) * route_width;
            for (size_t index = 0; index < route_width; ++index) {
                route_data[route_base + index] = -1;
            }
            if (recent_count <= 0 || block_count == 0) continue;
            float *recent = (float *)calloc(head_dim, sizeof(float));
            float *selected_scores = (float *)malloc(
                (size_t)semantic_blocks * sizeof(float)
            );
            int32_t *selected_blocks = (int32_t *)malloc(
                (size_t)semantic_blocks * sizeof(int32_t)
            );
            if (!recent || !selected_scores || !selected_blocks) {
                free(recent);
                free(selected_scores);
                free(selected_blocks);
                if (contiguous_summaries != summaries) {
                    ocean_tensor_release(contiguous_summaries);
                }
                if (contiguous_key != key) ocean_tensor_release(contiguous_key);
                ocean_tensor_fail("out of memory in sparse route selection");
            }
            size_t key_group = (batch_index * heads + head) * key_length * head_dim;
            float recent_norm = 0.0f;
            for (int token = start; token < end; ++token) {
                size_t key_base = key_group + (size_t)token * head_dim;
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    recent[dimension] += key_data[key_base + dimension]
                        / (float)recent_count;
                }
            }
            for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                recent_norm += recent[dimension] * recent[dimension];
            }
            recent_norm = sqrtf(recent_norm);
            size_t selected_count = 0;
            for (size_t block = 0; block < block_count; ++block) {
                size_t summary_base =
                    ((batch_index * heads + head) * summary_blocks + block)
                    * head_dim;
                float dot = 0.0f;
                float block_norm = 0.0f;
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    float value = summary_data[summary_base + dimension];
                    dot += recent[dimension] * value;
                    block_norm += value * value;
                }
                float score = dot / (recent_norm * sqrtf(block_norm) + 1.0e-8f);
                if (selected_count < (size_t)semantic_blocks) {
                    size_t insert = selected_count;
                    while (insert > 0 && ocean_tensor_sparse_score_precedes(
                        score, block, selected_scores[insert - 1],
                        (size_t)selected_blocks[insert - 1]
                    )) {
                        selected_scores[insert] = selected_scores[insert - 1];
                        selected_blocks[insert] = selected_blocks[insert - 1];
                        --insert;
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = (int32_t)block;
                    ++selected_count;
                } else if (ocean_tensor_sparse_score_precedes(
                    score, block, selected_scores[semantic_blocks - 1],
                    (size_t)selected_blocks[semantic_blocks - 1]
                )) {
                    size_t insert = (size_t)semantic_blocks - 1u;
                    while (insert > 0 && ocean_tensor_sparse_score_precedes(
                        score, block, selected_scores[insert - 1],
                        (size_t)selected_blocks[insert - 1]
                    )) {
                        selected_scores[insert] = selected_scores[insert - 1];
                        selected_blocks[insert] = selected_blocks[insert - 1];
                        --insert;
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = (int32_t)block;
                }
            }
            for (size_t index = 0; index < selected_count; ++index) {
                route_data[route_base + index] = selected_blocks[index];
            }
            uint32_t state = (uint32_t)random_seed ^
                (uint32_t)((batch_index * heads + head) * 747796405u +
                2891336453u);
            state = state * 1664525u + 1013904223u;
            int32_t random_block = (int32_t)(state % (uint32_t)block_count);
            for (size_t attempt = 0; attempt < block_count; ++attempt) {
                if (!ocean_tensor_sparse_route_contains(
                    selected_blocks, selected_count, random_block
                )) break;
                state = state * 1664525u + 1013904223u;
                random_block = (int32_t)(state % (uint32_t)block_count);
            }
            route_data[route_base + (size_t)semantic_blocks] = random_block;
            free(recent);
            free(selected_scores);
            free(selected_blocks);
        }
    }
    if (contiguous_summaries != summaries) ocean_tensor_release(contiguous_summaries);
    if (contiguous_key != key) ocean_tensor_release(contiguous_key);
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_build_route_active(
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t summaries,
    int active_length,
    int summary_window,
    int semantic_blocks,
    int block_size,
    int random_seed
) {
    if (!key || !summaries || key->dtype != OCEAN_TENSOR_FLOAT32 ||
        summaries->dtype != OCEAN_TENSOR_FLOAT32 || key->ndim != 4 ||
        summaries->ndim != 4 || active_length <= 0 || summary_window <= 0 ||
        semantic_blocks <= 0 || block_size <= 0) {
        ocean_tensor_fail("Sparse route requires valid float32 key and summaries");
    }
    size_t block_count = ((size_t)active_length + (size_t)block_size - 1u) /
        (size_t)block_size;
    if (active_length > (int)key->shape[2] ||
        summaries->shape[0] != key->shape[0] ||
        summaries->shape[1] != key->shape[1] ||
        summaries->shape[2] < block_count ||
        summaries->shape[3] != key->shape[3]) {
        ocean_tensor_fail("Sparse route metadata mismatch");
    }
    size_t route_shape[3] = {
        key->shape[0], key->shape[1], (size_t)semantic_blocks + 1u
    };
    ocean_tensor_handle_t route = ocean_tensor_alloc_uninitialized(
        route_shape, 3, OCEAN_TENSOR_INT32, key->device
    );
    ocean_tensor_sparse_attention_update_route_active(
        route, key, summaries, active_length, summary_window,
        semantic_blocks, block_size, random_seed
    );
    return route;
}

void ocean_tensor_sparse_attention_update_route_active(
    ocean_tensor_handle_t route,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t summaries,
    int active_length,
    int summary_window,
    int semantic_blocks,
    int block_size,
    int random_seed
) {
    if (!route || !key || !summaries || route->dtype != OCEAN_TENSOR_INT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 || summaries->dtype != OCEAN_TENSOR_FLOAT32 ||
        route->ndim != 3 || key->ndim != 4 || summaries->ndim != 4 ||
        active_length <= 0 || summary_window <= 0 || semantic_blocks <= 0 ||
        block_size <= 0) {
        ocean_tensor_fail("Sparse route update requires valid Tensor metadata");
    }
    size_t block_count = ((size_t)active_length + (size_t)block_size - 1u) /
        (size_t)block_size;
    if (active_length > (int)key->shape[2] ||
        route->shape[0] != key->shape[0] || route->shape[1] != key->shape[1] ||
        route->shape[2] != (size_t)semantic_blocks + 1u ||
        summaries->shape[0] != key->shape[0] ||
        summaries->shape[1] != key->shape[1] ||
        summaries->shape[2] < block_count ||
        summaries->shape[3] != key->shape[3] ||
        route->device != key->device || summaries->device != key->device ||
        !ocean_tensor_is_contiguous(route)) {
        ocean_tensor_fail("Sparse route update metadata mismatch");
    }
    if (key->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (!ocean_tensor_is_contiguous(key) || !ocean_tensor_is_contiguous(summaries) ||
            key->shape[0] > (size_t)INT_MAX || key->shape[1] > (size_t)INT_MAX ||
            key->shape[2] > (size_t)INT_MAX || key->shape[3] > 128 ||
            summaries->shape[2] > (size_t)INT_MAX) {
            ocean_tensor_fail("CUDA sparse route metadata mismatch");
        }
        ocean_cuda_sparse_build_route(
            key->cuda_data, summaries->cuda_data, route->cuda_data,
            (int)key->shape[0], (int)key->shape[1], (int)key->shape[2],
            (int)summaries->shape[2], active_length, (int)key->shape[3],
            summary_window, semantic_blocks, block_size, (unsigned int)random_seed
        );
        return;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (key->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail("Sparse route currently supports CPU or CUDA only");
    }
    ocean_tensor_sparse_route_cpu(
        route, key, summaries, active_length, summary_window,
        semantic_blocks, block_size, random_seed
    );
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_blocked_cached_routed(
    ocean_tensor_handle_t query,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t value,
    ocean_tensor_handle_t route,
    int active_length,
    int block_size,
    double scale,
    int query_start,
    bool causal
) {
    if (!query || !key || !value || !route ||
        query->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 ||
        value->dtype != OCEAN_TENSOR_FLOAT32 ||
        route->dtype != OCEAN_TENSOR_INT32 || query->ndim != 4 ||
        key->ndim != 4 || value->ndim != 4 || route->ndim != 3 ||
        block_size <= 0 || active_length <= 0 || query_start < 0) {
        ocean_tensor_fail("Routed SparseAttention received invalid metadata");
    }
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t query_length = query->shape[2];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    int route_blocks = (int)route->shape[2];
    if (route->shape[0] != batch || route->shape[1] != heads ||
        query->shape[0] != batch || query->shape[1] != heads ||
        value->shape[0] != batch || value->shape[1] != heads ||
        value->shape[2] != key_length || query->shape[3] != head_dim ||
        value->shape[3] != head_dim || (size_t)active_length > key_length ||
        (size_t)query_start > (size_t)active_length ||
        query_length > (size_t)active_length - (size_t)query_start ||
        route_blocks <= 0 || head_dim == 0) {
        ocean_tensor_fail("Routed SparseAttention metadata mismatch");
    }
    float score_scale = scale == 0.0
        ? 1.0f / sqrtf((float)head_dim) : (float)scale;
    if (!(score_scale > 0.0f) || !isfinite(score_scale)) {
        ocean_tensor_fail("Routed SparseAttention scale must be positive");
    }
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        query->shape, query->ndim, OCEAN_TENSOR_FLOAT32, query->device
    );
    if (query->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (!ocean_tensor_is_contiguous(query) || !ocean_tensor_is_contiguous(key) ||
            !ocean_tensor_is_contiguous(value) || !ocean_tensor_is_contiguous(route) ||
            batch > (size_t)INT_MAX || heads > (size_t)INT_MAX ||
            query_length > (size_t)INT_MAX || key_length > (size_t)INT_MAX ||
            head_dim > 128 || route_blocks > 64) {
            ocean_tensor_release(result);
            ocean_tensor_fail("CUDA routed SparseAttention metadata mismatch");
        }
        ocean_cuda_sparse_attention_routed(
            query->cuda_data, key->cuda_data, value->cuda_data, route->cuda_data,
            result->cuda_data, (int)batch, (int)heads, (int)query_length,
            (int)key_length, active_length, (int)head_dim, route_blocks,
            block_size, score_scale, query_start, causal ? 1 : 0
        );
        return result;
#else
        ocean_tensor_release(result);
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (query->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_release(result);
        ocean_tensor_fail("Routed SparseAttention currently supports CPU or CUDA only");
    }
    ocean_tensor_handle_t contiguous_query = ocean_tensor_is_contiguous(query)
        ? query : ocean_tensor_contiguous(query);
    ocean_tensor_handle_t contiguous_key = ocean_tensor_is_contiguous(key)
        ? key : ocean_tensor_contiguous(key);
    ocean_tensor_handle_t contiguous_value = ocean_tensor_is_contiguous(value)
        ? value : ocean_tensor_contiguous(value);
    ocean_tensor_handle_t contiguous_route = ocean_tensor_is_contiguous(route)
        ? route : ocean_tensor_contiguous(route);
    const float *query_data = (const float *)contiguous_query->cpu_data;
    const float *key_data = (const float *)contiguous_key->cpu_data;
    const float *value_data = (const float *)contiguous_value->cpu_data;
    const int32_t *route_data = (const int32_t *)contiguous_route->cpu_data;
    float *result_data = (float *)result->cpu_data;
    size_t route_width = (size_t)route_blocks;
    size_t max_tokens = route_width * (size_t)block_size;
    float *scores = (float *)malloc(max_tokens * sizeof(float));
    if (!scores) ocean_tensor_fail("out of memory in routed SparseAttention");
    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head = 0; head < heads; ++head) {
            for (size_t query_index = 0; query_index < query_length; ++query_index) {
                size_t route_base = (batch_index * heads + head) * route_width;
                size_t query_base = ((batch_index * heads + head) * query_length +
                    query_index) * head_dim;
                size_t selected = 0;
                int absolute_query = query_start + (int)query_index;
                for (int route_index = 0; route_index < route_blocks; ++route_index) {
                    int block = route_data[route_base + (size_t)route_index];
                    if (block < 0) continue;
                    int start = block * block_size;
                    int end = start + block_size;
                    if (end > active_length) end = active_length;
                    if (end > (int)key_length) end = (int)key_length;
                    if (causal && end > absolute_query + 1) end = absolute_query + 1;
                    if (start < 0 || start >= end) continue;
                    for (int token = start; token < end; ++token) {
                        size_t key_base = ((batch_index * heads + head) * key_length +
                            (size_t)token) * head_dim;
                        float score = 0.0f;
                        for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                            score += query_data[query_base + dimension] *
                                key_data[key_base + dimension];
                        }
                        scores[selected++] = score * score_scale;
                    }
                }
                if (selected == 0) {
                    for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                        result_data[query_base + dimension] = 0.0f;
                    }
                    continue;
                }
                float maximum = -INFINITY;
                for (size_t index = 0; index < selected; ++index) {
                    if (scores[index] > maximum) maximum = scores[index];
                }
                float denominator = 0.0f;
                for (size_t index = 0; index < selected; ++index) {
                    scores[index] = expf(scores[index] - maximum);
                    denominator += scores[index];
                }
                size_t output_base = query_base;
                for (size_t dimension = 0; dimension < head_dim; ++dimension) {
                    float output = 0.0f;
                    size_t selected_index = 0;
                    for (int route_index = 0; route_index < route_blocks; ++route_index) {
                        int block = route_data[route_base + (size_t)route_index];
                        if (block < 0) continue;
                        int start = block * block_size;
                        int end = start + block_size;
                        if (end > active_length) end = active_length;
                        if (end > (int)key_length) end = (int)key_length;
                        if (causal && end > absolute_query + 1) end = absolute_query + 1;
                        if (start < 0 || start >= end) continue;
                        for (int token = start; token < end; ++token) {
                            size_t value_base = ((batch_index * heads + head) * key_length +
                                (size_t)token) * head_dim;
                            output += (scores[selected_index++] / denominator) *
                                value_data[value_base + dimension];
                        }
                    }
                    result_data[output_base + dimension] = output;
                }
            }
        }
    }
    free(scores);
    if (contiguous_route != route) ocean_tensor_release(contiguous_route);
    if (contiguous_value != value) ocean_tensor_release(contiguous_value);
    if (contiguous_key != key) ocean_tensor_release(contiguous_key);
    if (contiguous_query != query) ocean_tensor_release(contiguous_query);
    return result;
}

ocean_tensor_handle_t ocean_tensor_sparse_attention_blocked_cached_active(
    ocean_tensor_handle_t query,
    ocean_tensor_handle_t key,
    ocean_tensor_handle_t value,
    ocean_tensor_handle_t summaries,
    int active_length,
    int top_k,
    int top_blocks,
    int block_size,
    double scale,
    int query_start,
    bool causal
) {
    if (!query || !key || !value || !summaries) {
        ocean_tensor_fail("Active cached SparseAttention received a null Tensor");
    }
    if (query->dtype != OCEAN_TENSOR_FLOAT32 ||
        key->dtype != OCEAN_TENSOR_FLOAT32 ||
        value->dtype != OCEAN_TENSOR_FLOAT32 ||
        summaries->dtype != OCEAN_TENSOR_FLOAT32 ||
        query->ndim != 4 || key->ndim != 4 || value->ndim != 4 ||
        summaries->ndim != 4 || block_size <= 0 || top_k <= 0 ||
        top_blocks <= 0 || active_length <= 0 || query_start < 0) {
        ocean_tensor_fail(
            "Active cached SparseAttention requires float32 rank-4 tensors"
        );
    }
    size_t batch = key->shape[0];
    size_t heads = key->shape[1];
    size_t key_length = key->shape[2];
    size_t head_dim = key->shape[3];
    size_t expected_blocks =
        (key_length + (size_t)block_size - 1) / (size_t)block_size;
    if (key_length == 0 || head_dim == 0 ||
        (size_t)active_length > key_length ||
        query->device != key->device || value->device != key->device ||
        summaries->device != key->device ||
        query->shape[0] != batch || query->shape[1] != heads ||
        value->shape[0] != batch || value->shape[1] != heads ||
        value->shape[2] != key_length || query->shape[3] != head_dim ||
        value->shape[3] != head_dim ||
        summaries->shape[0] != batch || summaries->shape[1] != heads ||
        summaries->shape[2] != expected_blocks ||
        summaries->shape[3] != head_dim ||
        (size_t)query_start > (size_t)active_length ||
        query->shape[2] > (size_t)active_length - (size_t)query_start) {
        ocean_tensor_fail("Active cached SparseAttention metadata mismatch");
    }
    float score_scale = scale == 0.0
        ? 1.0f / sqrtf((float)head_dim) : (float)scale;
    if (!(score_scale > 0.0f) || !isfinite(score_scale)) {
        ocean_tensor_fail("Active cached SparseAttention scale must be positive");
    }

    if (query->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (head_dim > 128 || batch > (size_t)INT_MAX ||
            heads > (size_t)INT_MAX || query->shape[2] > (size_t)INT_MAX ||
            key_length > (size_t)INT_MAX ||
            expected_blocks > (size_t)INT_MAX ||
            !ocean_tensor_is_contiguous(query) ||
            !ocean_tensor_is_contiguous(key) ||
            !ocean_tensor_is_contiguous(value) ||
            !ocean_tensor_is_contiguous(summaries)) {
            ocean_tensor_fail("CUDA active cached SparseAttention metadata mismatch");
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            query->shape, query->ndim, OCEAN_TENSOR_FLOAT32,
            OCEAN_TENSOR_BACKEND_CUDA
        );
        ocean_cuda_sparse_attention(
            query->cuda_data, key->cuda_data, value->cuda_data,
            summaries->cuda_data, result->cuda_data,
            (int)batch, (int)heads, (int)query->shape[2],
            (int)key_length, active_length, (int)head_dim,
            (int)expected_blocks, top_k, top_blocks, block_size,
            score_scale, query_start, causal ? 1 : 0
        );
        return result;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }
    if (query->device != OCEAN_TENSOR_CPU) {
        ocean_tensor_fail(
            "Active cached SparseAttention currently supports CPU or CUDA only"
        );
    }
    if (active_length == (int)key_length) {
        return ocean_tensor_sparse_attention_blocked_cached(
            query, key, value, summaries, top_k, top_blocks, block_size,
            scale, query_start, causal
        );
    }
    ocean_tensor_handle_t active_key = ocean_tensor_slice(
        key, 2, 0, active_length, 1
    );
    ocean_tensor_handle_t active_value = ocean_tensor_slice(
        value, 2, 0, active_length, 1
    );
    size_t active_blocks =
        ((size_t)active_length + (size_t)block_size - 1) /
        (size_t)block_size;
    ocean_tensor_handle_t active_summaries = ocean_tensor_slice(
        summaries, 2, 0, (int)active_blocks, 1
    );
    ocean_tensor_handle_t result = ocean_tensor_sparse_attention_blocked_cached(
        query, active_key, active_value, active_summaries,
        top_k, top_blocks, block_size, scale, query_start, causal
    );
    ocean_tensor_release(active_summaries);
    ocean_tensor_release(active_value);
    ocean_tensor_release(active_key);
    return result;
}

static void ocean_tensor_validate_backward_inputs(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t reference,
    const char *operation
) {
    if (!upstream || !reference) {
        ocean_tensor_fail("Tensor backward operation received a null handle");
    }
    if (upstream->dtype != OCEAN_TENSOR_FLOAT32 ||
        reference->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail("Tensor backward operation currently requires float32");
    }
    if (upstream->device != reference->device ||
        upstream->ndim != reference->ndim ||
        upstream->size != reference->size) {
        ocean_tensor_fail(operation);
    }
    for (size_t axis = 0; axis < reference->ndim; ++axis) {
        if (upstream->shape[axis] != reference->shape[axis]) {
            ocean_tensor_fail(operation);
        }
    }
}

ocean_tensor_handle_t ocean_tensor_softmax_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t output,
    int dim
) {
    ocean_tensor_validate_backward_inputs(
        upstream, output,
        "Tensor.softmax backward requires matching Tensor shapes and devices"
    );
    size_t axis = ocean_tensor_normalize_dim_v02(output, dim);
    if (output->device != OCEAN_TENSOR_GPU ||
        axis != output->ndim - 1) {
        ocean_tensor_fail(
            "GPU softmax backward currently requires the last Tensor dimension"
        );
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t width = output->shape[axis];
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output->shape, output->ndim, output->dtype, OCEAN_TENSOR_GPU
    );
    if (width != 0 && output->size != 0) {
        if (width > (size_t)INT32_MAX) {
            ocean_tensor_release(result);
            ocean_tensor_fail(
                "GPU softmax backward dimension is too large for OpenCL"
            );
        }
        ocean_tensor_opencl_softmax_backward(
            upstream, output, result, (int)width
        );
    }
    return result;
#else
    (void)upstream;
    (void)output;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_layer_norm_backward(
    ocean_tensor_handle_t upstream,
    ocean_tensor_handle_t input,
    int dim,
    double epsilon
) {
    ocean_tensor_validate_backward_inputs(
        upstream, input,
        "Tensor.LayerNorm backward requires matching Tensor shapes and devices"
    );
    if (!(epsilon > 0.0)) {
        ocean_tensor_fail("LayerNorm epsilon must be positive");
    }
    size_t axis = ocean_tensor_normalize_dim_v02(input, dim);
    if (input->device != OCEAN_TENSOR_GPU ||
        axis != input->ndim - 1) {
        ocean_tensor_fail(
            "GPU LayerNorm backward currently requires the last Tensor dimension"
        );
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t width = input->shape[axis];
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        input->shape, input->ndim, input->dtype, OCEAN_TENSOR_GPU
    );
    if (width != 0 && input->size != 0) {
        if (width > (size_t)INT32_MAX) {
            ocean_tensor_release(result);
            ocean_tensor_fail(
                "GPU LayerNorm backward dimension is too large for OpenCL"
            );
        }
        ocean_tensor_opencl_layer_norm_backward(
            upstream, input, result, (int)width, epsilon
        );
    }
    return result;
#else
    (void)upstream;
    (void)input;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

static void ocean_tensor_validate_optimizer_tensors(
    ocean_tensor_handle_t parameter,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t first_moment,
    ocean_tensor_handle_t second_moment
) {
    if (!parameter || !gradient ||
        (first_moment == NULL) != (second_moment == NULL)) {
        ocean_tensor_fail("optimizer update received a null Tensor");
    }
    ocean_tensor_handle_t tensors[4] = {
        parameter, gradient, first_moment, second_moment
    };
    size_t tensor_count = first_moment ? 4 : 2;
    for (size_t i = 0; i < tensor_count; ++i) {
        if (tensors[i]->dtype != OCEAN_TENSOR_FLOAT32) {
            ocean_tensor_fail("optimizer updates currently require float32");
        }
        if (tensors[i]->device != parameter->device ||
            tensors[i]->size != parameter->size ||
            tensors[i]->ndim != parameter->ndim) {
            ocean_tensor_fail("optimizer tensors must have matching device and shape");
        }
        for (size_t axis = 0; axis < parameter->ndim; ++axis) {
            if (tensors[i]->shape[axis] != parameter->shape[axis]) {
                ocean_tensor_fail("optimizer tensors must have matching device and shape");
            }
        }
    }
}

void ocean_tensor_sgd_update(
    ocean_tensor_handle_t parameter,
    ocean_tensor_handle_t gradient,
    double learning_rate
) {
    ocean_tensor_validate_optimizer_tensors(parameter, gradient, NULL, NULL);
    if (learning_rate < 0.0) {
        ocean_tensor_fail("SGD learning rate must be non-negative");
    }
    if (parameter->device == OCEAN_TENSOR_BACKEND_CUDA) {
        ocean_tensor_fail("CUDA SGD kernel is not implemented yet");
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_opencl_sgd_update(parameter, gradient, learning_rate);
        return;
    }
#else
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    }
#endif
    float *values = (float *)parameter->cpu_data;
    const float *gradients = (const float *)gradient->cpu_data;
    float rate = (float)learning_rate;
    for (size_t i = 0; i < parameter->size; ++i) {
        values[i] -= rate * gradients[i];
    }
}

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
) {
    ocean_tensor_validate_optimizer_tensors(
        parameter, gradient, first_moment, second_moment
    );
    if (learning_rate < 0.0 || beta1 < 0.0 || beta1 >= 1.0 ||
        beta2 < 0.0 || beta2 >= 1.0 || epsilon <= 0.0 ||
        weight_decay < 0.0 || bias_correction1 <= 0.0 ||
        bias_correction2 <= 0.0) {
        ocean_tensor_fail("invalid AdamW optimizer arguments");
    }
    if (parameter->device == OCEAN_TENSOR_BACKEND_CUDA) {
        ocean_tensor_fail("CUDA AdamW kernel is not implemented yet");
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_opencl_adamw_update(
            parameter, gradient, first_moment, second_moment,
            learning_rate, beta1, beta2, epsilon, weight_decay,
            bias_correction1, bias_correction2
        );
        return;
    }
#else
    if (parameter->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    }
#endif
    float *values = (float *)parameter->cpu_data;
    const float *gradients = (const float *)gradient->cpu_data;
    float *first = (float *)first_moment->cpu_data;
    float *second = (float *)second_moment->cpu_data;
    for (size_t i = 0; i < parameter->size; ++i) {
        float gradient_value = gradients[i];
        float first_value = (float)(
            beta1 * (double)first[i]
            + (1.0 - beta1) * (double)gradient_value
        );
        float second_value = (float)(
            beta2 * (double)second[i]
            + (1.0 - beta2) * (double)gradient_value * (double)gradient_value
        );
        first[i] = first_value;
        second[i] = second_value;
        double adaptive =
            ((double)first_value / bias_correction1)
            / (sqrt((double)second_value / bias_correction2) + epsilon);
        values[i] = (float)(
            (double)values[i]
            - learning_rate * weight_decay * (double)values[i]
            - learning_rate * adaptive
        );
    }
}

static ocean_tensor_handle_t ocean_tensor_matmul_nd_cpu_v02(ocean_tensor_handle_t left, ocean_tensor_handle_t right) {
    size_t out_ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    size_t batch_ndim = out_ndim - 2;
    size_t lb = left->ndim - 2, rb = right->ndim - 2;
    size_t *shape = malloc(out_ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory allocating batched matmul shape");

    for (size_t oa=0; oa<batch_ndim; ++oa) {
        long long la=(long long)oa-(long long)(batch_ndim-lb);
        long long ra=(long long)oa-(long long)(batch_ndim-rb);
        size_t ld=la>=0?left->shape[(size_t)la]:1;
        size_t rd=ra>=0?right->shape[(size_t)ra]:1;
        if (ld!=rd && ld!=1 && rd!=1) { free(shape); ocean_tensor_fail("batched matmul batch dimensions are not broadcastable"); }
        shape[oa]=ld>rd?ld:rd;
    }
    shape[out_ndim-2]=left->shape[left->ndim-2];
    shape[out_ndim-1]=right->shape[right->ndim-1];
    ocean_tensor_handle_t out=ocean_tensor_alloc_zeros(shape,out_ndim,left->dtype,OCEAN_TENSOR_CPU);
    free(shape);
    size_t *coord=calloc(out_ndim,sizeof(size_t));
    if (!coord) ocean_tensor_fail("out of memory in batched matmul");
    size_t inner=left->shape[left->ndim-1];

    for (size_t linear=0;linear<out->size;++linear) {
        size_t rem=linear;
        for (size_t i=out_ndim;i-- >0;) { size_t d=out->shape[i]; coord[i]=d?rem%d:0; rem=d?rem/d:0; }
        size_t row=coord[out_ndim-2], col=coord[out_ndim-1];
        size_t lbase=0, rbase=0;
        for (size_t i=0;i<lb;++i) { size_t oa=batch_ndim-lb+i; size_t c=left->shape[i]==1?0:coord[oa]; lbase+=c*left->strides[i]; }
        for (size_t i=0;i<rb;++i) { size_t oa=batch_ndim-rb+i; size_t c=right->shape[i]==1?0:coord[oa]; rbase+=c*right->strides[i]; }
        lbase += row*left->strides[left->ndim-2];
        rbase += col*right->strides[right->ndim-1];
        long double sum=0.0L;
        for (size_t k=0;k<inner;++k) sum += ocean_tensor_read_scalar(left,lbase+k*left->strides[left->ndim-1]) * ocean_tensor_read_scalar(right,rbase+k*right->strides[right->ndim-2]);
        ocean_tensor_write_scalar(out,linear,sum);
    }
    free(coord); return out;
}


ocean_tensor_handle_t ocean_tensor_matmul_transposed(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
) {
    if (!left || !right) {
        ocean_tensor_fail("matmul does not accept null Tensors");
    }
    if (left->ndim < 2 || right->ndim < 2) {
        ocean_tensor_fail("matmul expects Tensor rank >= 2");
    }
    if (left->dtype != right->dtype) {
        ocean_tensor_fail("matmul requires matching Tensor dtypes");
    }
    if (left->device != right->device) {
        ocean_tensor_fail("matmul requires Tensors on the same device");
    }

    size_t left_inner = transpose_left
        ? left->shape[left->ndim - 2]
        : left->shape[left->ndim - 1];
    size_t right_inner = transpose_right
        ? right->shape[right->ndim - 1]
        : right->shape[right->ndim - 2];
    if (left_inner != right_inner) {
        ocean_tensor_fail("matmul shape mismatch");
    }

    if (!transpose_left && !transpose_right) {
        return ocean_tensor_matmul(left, right);
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (left->device == OCEAN_TENSOR_GPU &&
        left->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_handle_t contiguous_left =
            ocean_tensor_is_contiguous(left)
            ? left : ocean_tensor_contiguous(left);
        ocean_tensor_handle_t contiguous_right =
            ocean_tensor_is_contiguous(right)
            ? right : ocean_tensor_contiguous(right);
        ocean_tensor_handle_t result = ocean_tensor_matmul_opencl_batched(
            contiguous_left,
            contiguous_right,
            transpose_left,
            transpose_right
        );
        if (contiguous_left != left) ocean_tensor_release(contiguous_left);
        if (contiguous_right != right) ocean_tensor_release(contiguous_right);
        return result;
    }
#endif

    ocean_tensor_handle_t transposed_left = transpose_left
        ? ocean_tensor_transpose_dims(left, -2, -1) : ocean_tensor_copy(left);
    ocean_tensor_handle_t transposed_right = transpose_right
        ? ocean_tensor_transpose_dims(right, -2, -1) : ocean_tensor_copy(right);
    ocean_tensor_handle_t result = ocean_tensor_matmul(
        transposed_left,
        transposed_right
    );
    ocean_tensor_release(transposed_left);
    ocean_tensor_release(transposed_right);
    return result;
}

ocean_tensor_handle_t ocean_tensor_matmul(ocean_tensor_handle_t left, ocean_tensor_handle_t right) {
    if (!left || !right) ocean_tensor_fail("matmul does not accept null Tensors");
    if (left->ndim < 2 || right->ndim < 2) ocean_tensor_fail("matmul expects Tensor rank >= 2");
    if (left->shape[left->ndim-1] != right->shape[right->ndim-2]) ocean_tensor_fail("matmul shape mismatch");
    if (left->dtype != right->dtype) ocean_tensor_fail("matmul requires matching Tensor dtypes");
    if (left->device != right->device) ocean_tensor_fail("matmul requires Tensors on the same device");
    if (left->ndim == 2 && right->ndim == 2) return ocean_tensor_backend_for_device(left->device)->matmul(left,right);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (left->device == OCEAN_TENSOR_GPU &&
        left->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_handle_t contiguous_left =
            ocean_tensor_is_contiguous(left)
            ? left : ocean_tensor_contiguous(left);
        ocean_tensor_handle_t contiguous_right =
            ocean_tensor_is_contiguous(right)
            ? right : ocean_tensor_contiguous(right);
        ocean_tensor_handle_t result = ocean_tensor_matmul_opencl_batched(
            contiguous_left,
            contiguous_right,
            false,
            false
        );
        if (contiguous_left != left) ocean_tensor_release(contiguous_left);
        if (contiguous_right != right) ocean_tensor_release(contiguous_right);
        return result;
    }
#endif
    ocean_tensor_handle_t lc = left->device==OCEAN_TENSOR_CPU ? left : ocean_tensor_to(left,"cpu");
    ocean_tensor_handle_t rc = right->device==OCEAN_TENSOR_CPU ? right : ocean_tensor_to(right,"cpu");
    ocean_tensor_handle_t cpu=ocean_tensor_matmul_nd_cpu_v02(lc,rc);
    if (lc != left) {
        ocean_tensor_release(lc);
    }
    if (rc != right) {
        ocean_tensor_release(rc);
    }
    if (left->device==OCEAN_TENSOR_CPU) return cpu;
    ocean_tensor_handle_t out=ocean_tensor_to(cpu,"gpu"); ocean_tensor_release(cpu); return out;
}

ocean_tensor_handle_t ocean_tensor_linear_inference(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t weight,
    ocean_tensor_handle_t bias
) {
    if (!input || !weight || !bias) {
        ocean_tensor_fail("linear_inference does not accept null Tensors");
    }
    if (input->dtype != OCEAN_TENSOR_FLOAT32 ||
        weight->dtype != OCEAN_TENSOR_FLOAT32 ||
        bias->dtype != OCEAN_TENSOR_FLOAT32 ||
        input->device != weight->device || input->device != bias->device) {
        ocean_tensor_fail(
            "linear_inference requires matching float32 Tensors on one device"
        );
    }
    if (weight->ndim != 2 || bias->ndim != 2 || bias->shape[0] != 1 ||
        weight->shape[1] != bias->shape[1] ||
        input->shape[input->ndim - 1] != weight->shape[0]) {
        ocean_tensor_fail("linear_inference shape mismatch");
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (input->device == OCEAN_TENSOR_GPU && input->ndim == 3 &&
        input->shape[0] == 1 && input->shape[1] == 1) {
        size_t output_shape[3] = {1, 1, weight->shape[1]};
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            output_shape, 3, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_matvec_bias(input, weight, bias, result);
        return result;
    }
#endif

    ocean_tensor_handle_t product = ocean_tensor_matmul(input, weight);
    ocean_tensor_handle_t result = ocean_tensor_binary(product, bias, 0);
    ocean_tensor_release(product);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_packed_linear_inference_impl(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t packed_weight,
    double scale,
    ocean_tensor_handle_t bias,
    int out_features
) {
    if (!input || !packed_weight) {
        ocean_tensor_fail("packed inference does not accept null Tensors");
    }
    if (input->dtype != OCEAN_TENSOR_FLOAT32 ||
        packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        input->device != packed_weight->device) {
        ocean_tensor_fail(
            "packed inference requires float32 input and int32 packed weights on one device"
        );
    }
    if (input->ndim < 2 || packed_weight->ndim != 2 || out_features <= 0) {
        ocean_tensor_fail("packed inference shape mismatch");
    }
    size_t cols_a = input->shape[input->ndim - 1];
    size_t expected_packed_cols = ((size_t)out_features + 15u) / 16u;
    if (packed_weight->shape[0] != cols_a ||
        packed_weight->shape[1] != expected_packed_cols) {
        ocean_tensor_fail("packed inference weight metadata mismatch");
    }
    if (bias && (bias->dtype != OCEAN_TENSOR_FLOAT32 ||
        bias->device != input->device || bias->ndim != 2 ||
        bias->shape[0] != 1 || bias->shape[1] != (size_t)out_features)) {
        ocean_tensor_fail("packed inference bias metadata mismatch");
    }
    if (!(scale > 0.0)) {
        ocean_tensor_fail("packed inference scale must be positive");
    }

    size_t *output_shape = (size_t *)malloc(input->ndim * sizeof(size_t));
    if (!output_shape) ocean_tensor_fail("out of memory in packed inference");
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        output_shape[axis] = input->shape[axis];
    }
    output_shape[input->ndim - 1] = (size_t)out_features;
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape, input->ndim, OCEAN_TENSOR_FLOAT32, input->device
    );
    free(output_shape);

    if (input->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
            ? input : ocean_tensor_contiguous(input);
        ocean_tensor_handle_t packed_contiguous = ocean_tensor_is_contiguous(packed_weight)
            ? packed_weight : ocean_tensor_contiguous(packed_weight);
        ocean_tensor_handle_t bias_contiguous = bias && !ocean_tensor_is_contiguous(bias)
            ? ocean_tensor_contiguous(bias) : bias;
        ocean_tensor_cuda_packed_linear(
            input_contiguous, packed_contiguous, bias_contiguous, result,
            out_features, scale
        );
        if (input_contiguous != input) ocean_tensor_release(input_contiguous);
        if (packed_contiguous != packed_weight) ocean_tensor_release(packed_contiguous);
        if (bias_contiguous != bias) ocean_tensor_release(bias_contiguous);
        return result;
#else
        ocean_tensor_release(result);
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }

    if (input->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
            ? input : ocean_tensor_contiguous(input);
        ocean_tensor_handle_t packed_contiguous = ocean_tensor_is_contiguous(packed_weight)
            ? packed_weight : ocean_tensor_contiguous(packed_weight);
        ocean_tensor_handle_t bias_contiguous = bias && !ocean_tensor_is_contiguous(bias)
            ? ocean_tensor_contiguous(bias) : bias;
        const float *input_values = (const float *)input_contiguous->cpu_data;
        const int32_t *packed_values = (const int32_t *)packed_contiguous->cpu_data;
        const float *bias_values = bias_contiguous
            ? (const float *)bias_contiguous->cpu_data : NULL;
        size_t rows = input_contiguous->size / cols_a;
        for (size_t row = 0; row < rows; ++row) {
            for (int col = 0; col < out_features; ++col) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols_a; ++k) {
                    uint32_t word = (uint32_t)packed_values[
                        k * expected_packed_cols + (size_t)col / 16u
                    ];
                    uint32_t code = (word >> (2u * ((uint32_t)col % 16u))) & 3u;
                    float sign = code == 1u
                        ? 1.0f : (code == 2u ? -1.0f : 0.0f);
                    sum += input_values[row * cols_a + k] * sign;
                }
                float output_value = sum * (float)scale;
                if (bias_values) output_value += bias_values[col];
                ((float *)result->cpu_data)[row * (size_t)out_features + (size_t)col] =
                    output_value;
            }
        }
        if (input_contiguous != input) ocean_tensor_release(input_contiguous);
        if (packed_contiguous != packed_weight) ocean_tensor_release(packed_contiguous);
        if (bias_contiguous != bias) ocean_tensor_release(bias_contiguous);
        return result;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
        ? input : ocean_tensor_contiguous(input);
    ocean_tensor_handle_t packed_contiguous = ocean_tensor_is_contiguous(packed_weight)
        ? packed_weight : ocean_tensor_contiguous(packed_weight);
    ocean_tensor_handle_t bias_contiguous = bias && !ocean_tensor_is_contiguous(bias)
        ? ocean_tensor_contiguous(bias) : bias;
    ocean_tensor_opencl_packed_linear(
        input_contiguous, packed_contiguous, bias_contiguous, result,
        out_features, scale
    );
    if (input_contiguous != input) ocean_tensor_release(input_contiguous);
    if (packed_contiguous != packed_weight) ocean_tensor_release(packed_contiguous);
    if (bias_contiguous != bias) ocean_tensor_release(bias_contiguous);
    return result;
#else
    ocean_tensor_release(result);
    ocean_tensor_fail(
        "GPU backend is unavailable: rebuild with OpenCL support"
    );
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_packed_matmul_inference(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t packed_weight,
    double scale,
    int out_features
) {
    return ocean_tensor_packed_linear_inference_impl(
        input, packed_weight, scale, NULL, out_features
    );
}

ocean_tensor_handle_t ocean_tensor_packed_linear_inference(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t packed_weight,
    double scale,
    ocean_tensor_handle_t bias,
    int out_features
) {
    return ocean_tensor_packed_linear_inference_impl(
        input, packed_weight, scale, bias, out_features
    );
}

static ocean_tensor_handle_t ocean_tensor_packed_qkv_inference_impl(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t q_packed_weight,
    double q_scale,
    ocean_tensor_handle_t q_bias,
    ocean_tensor_handle_t k_packed_weight,
    double k_scale,
    ocean_tensor_handle_t k_bias,
    ocean_tensor_handle_t v_packed_weight,
    double v_scale,
    ocean_tensor_handle_t v_bias,
    int out_features
) {
    if (!input || !q_packed_weight || !q_bias || !k_packed_weight ||
        !k_bias || !v_packed_weight || !v_bias) {
        ocean_tensor_fail("packed QKV inference does not accept null Tensors");
    }
    if (input->dtype != OCEAN_TENSOR_FLOAT32 ||
        q_packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        k_packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        v_packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        q_bias->dtype != OCEAN_TENSOR_FLOAT32 ||
        k_bias->dtype != OCEAN_TENSOR_FLOAT32 ||
        v_bias->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail(
            "packed QKV inference requires float32 input/biases and int32 weights"
        );
    }
    ocean_tensor_handle_t tensors[] = {
        q_packed_weight, q_bias, k_packed_weight, k_bias,
        v_packed_weight, v_bias,
    };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        if (tensors[index]->device != input->device) {
            ocean_tensor_fail(
                "packed QKV inference requires tensors on one device"
            );
        }
    }
    if (input->ndim < 2 || out_features <= 0 ||
        q_packed_weight->ndim != 2 || k_packed_weight->ndim != 2 ||
        v_packed_weight->ndim != 2 || q_bias->ndim != 2 ||
        k_bias->ndim != 2 || v_bias->ndim != 2 ||
        q_bias->shape[0] != 1 || k_bias->shape[0] != 1 ||
        v_bias->shape[0] != 1 ||
        q_bias->shape[1] != (size_t)out_features ||
        k_bias->shape[1] != (size_t)out_features ||
        v_bias->shape[1] != (size_t)out_features) {
        ocean_tensor_fail("packed QKV inference shape mismatch");
    }
    size_t cols_a = input->shape[input->ndim - 1];
    size_t expected_packed_cols = ((size_t)out_features + 15u) / 16u;
    ocean_tensor_handle_t packed_weights[] = {
        q_packed_weight, k_packed_weight, v_packed_weight,
    };
    for (size_t index = 0; index < 3; ++index) {
        if (packed_weights[index]->shape[0] != cols_a ||
            packed_weights[index]->shape[1] != expected_packed_cols) {
            ocean_tensor_fail("packed QKV weight metadata mismatch");
        }
    }
    if (!(q_scale > 0.0) || !(k_scale > 0.0) || !(v_scale > 0.0)) {
        ocean_tensor_fail("packed QKV scales must be positive");
    }
    if ((size_t)out_features > SIZE_MAX / 3u) {
        ocean_tensor_fail("packed QKV output shape is too large");
    }

    size_t *output_shape = (size_t *)malloc(input->ndim * sizeof(size_t));
    if (!output_shape) ocean_tensor_fail("out of memory in packed QKV inference");
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        output_shape[axis] = input->shape[axis];
    }
    output_shape[input->ndim - 1] = (size_t)out_features * 3u;
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape, input->ndim, OCEAN_TENSOR_FLOAT32, input->device
    );
    free(output_shape);

    if (input->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
            ? input : ocean_tensor_contiguous(input);
        ocean_tensor_handle_t packed_contiguous[3];
        ocean_tensor_handle_t bias_contiguous[3];
        ocean_tensor_handle_t packed_weights[] = {
            q_packed_weight, k_packed_weight, v_packed_weight,
        };
        ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
        for (int index = 0; index < 3; ++index) {
            packed_contiguous[index] = ocean_tensor_is_contiguous(packed_weights[index])
                ? packed_weights[index] : ocean_tensor_contiguous(packed_weights[index]);
            bias_contiguous[index] = ocean_tensor_is_contiguous(biases[index])
                ? biases[index] : ocean_tensor_contiguous(biases[index]);
        }
        ocean_tensor_cuda_packed_qkv(
            input_contiguous,
            packed_contiguous[0], bias_contiguous[0],
            packed_contiguous[1], bias_contiguous[1],
            packed_contiguous[2], bias_contiguous[2],
            result, out_features, q_scale, k_scale, v_scale
        );
        if (input_contiguous != input) ocean_tensor_release(input_contiguous);
        for (int index = 0; index < 3; ++index) {
            if (packed_contiguous[index] != packed_weights[index]) {
                ocean_tensor_release(packed_contiguous[index]);
            }
            if (bias_contiguous[index] != biases[index]) {
                ocean_tensor_release(bias_contiguous[index]);
            }
        }
        return result;
#else
        ocean_tensor_release(result);
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }

    if (input->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
            ? input : ocean_tensor_contiguous(input);
        ocean_tensor_handle_t packed_contiguous[3];
        ocean_tensor_handle_t bias_contiguous[3];
        for (int index = 0; index < 3; ++index) {
            packed_contiguous[index] = ocean_tensor_is_contiguous(packed_weights[index])
                ? packed_weights[index]
                : ocean_tensor_contiguous(packed_weights[index]);
        }
        ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
        for (int index = 0; index < 3; ++index) {
            bias_contiguous[index] = ocean_tensor_is_contiguous(biases[index])
                ? biases[index] : ocean_tensor_contiguous(biases[index]);
        }
        const float *input_values = (const float *)input_contiguous->cpu_data;
        const int32_t *packed_values[3];
        const float *bias_values[3];
        for (int index = 0; index < 3; ++index) {
            packed_values[index] = (const int32_t *)packed_contiguous[index]->cpu_data;
            bias_values[index] = (const float *)bias_contiguous[index]->cpu_data;
        }
        float *output_values = (float *)result->cpu_data;
        size_t rows = input_contiguous->size / cols_a;
        double scales[] = {q_scale, k_scale, v_scale};
        for (size_t row = 0; row < rows; ++row) {
            for (int col = 0; col < out_features; ++col) {
                float sums[] = {0.0f, 0.0f, 0.0f};
                for (size_t k = 0; k < cols_a; ++k) {
                    for (int projection = 0; projection < 3; ++projection) {
                        uint32_t word = (uint32_t)packed_values[projection][
                            k * expected_packed_cols + (size_t)col / 16u
                        ];
                        uint32_t code =
                            (word >> (2u * ((uint32_t)col % 16u))) & 3u;
                        float sign = code == 1u
                            ? 1.0f : (code == 2u ? -1.0f : 0.0f);
                        sums[projection] += input_values[row * cols_a + k] * sign;
                    }
                }
                size_t output_offset = row * (size_t)out_features * 3u + (size_t)col;
                output_values[output_offset] =
                    sums[0] * (float)scales[0] + bias_values[0][col];
                output_values[output_offset + (size_t)out_features] =
                    sums[1] * (float)scales[1] + bias_values[1][col];
                output_values[output_offset + (size_t)out_features * 2u] =
                    sums[2] * (float)scales[2] + bias_values[2][col];
            }
        }
        if (input_contiguous != input) ocean_tensor_release(input_contiguous);
        for (int index = 0; index < 3; ++index) {
            if (packed_contiguous[index] != packed_weights[index]) {
                ocean_tensor_release(packed_contiguous[index]);
            }
            if (bias_contiguous[index] != biases[index]) {
                ocean_tensor_release(bias_contiguous[index]);
            }
        }
        return result;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
        ? input : ocean_tensor_contiguous(input);
    ocean_tensor_handle_t packed_contiguous[3];
    ocean_tensor_handle_t bias_contiguous[3];
    ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
    for (int index = 0; index < 3; ++index) {
        packed_contiguous[index] = ocean_tensor_is_contiguous(packed_weights[index])
            ? packed_weights[index]
            : ocean_tensor_contiguous(packed_weights[index]);
        bias_contiguous[index] = ocean_tensor_is_contiguous(biases[index])
            ? biases[index] : ocean_tensor_contiguous(biases[index]);
    }
    ocean_tensor_opencl_packed_qkv(
        input_contiguous,
        packed_contiguous[0], bias_contiguous[0],
        packed_contiguous[1], bias_contiguous[1],
        packed_contiguous[2], bias_contiguous[2],
        result, out_features, q_scale, k_scale, v_scale
    );
    if (input_contiguous != input) ocean_tensor_release(input_contiguous);
    for (int index = 0; index < 3; ++index) {
        if (packed_contiguous[index] != packed_weights[index]) {
            ocean_tensor_release(packed_contiguous[index]);
        }
        if (bias_contiguous[index] != biases[index]) {
            ocean_tensor_release(bias_contiguous[index]);
        }
    }
    return result;
#else
    ocean_tensor_release(result);
    ocean_tensor_fail(
        "GPU backend is unavailable: rebuild with OpenCL support"
    );
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_packed_qkv_inference(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t q_packed_weight,
    double q_scale,
    ocean_tensor_handle_t q_bias,
    ocean_tensor_handle_t k_packed_weight,
    double k_scale,
    ocean_tensor_handle_t k_bias,
    ocean_tensor_handle_t v_packed_weight,
    double v_scale,
    ocean_tensor_handle_t v_bias,
    int out_features
) {
    return ocean_tensor_packed_qkv_inference_impl(
        input, q_packed_weight, q_scale, q_bias,
        k_packed_weight, k_scale, k_bias,
        v_packed_weight, v_scale, v_bias, out_features
    );
}

void ocean_tensor_packed_qkv_inference_into(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t q_packed_weight,
    double q_scale,
    ocean_tensor_handle_t q_bias,
    ocean_tensor_handle_t k_packed_weight,
    double k_scale,
    ocean_tensor_handle_t k_bias,
    ocean_tensor_handle_t v_packed_weight,
    double v_scale,
    ocean_tensor_handle_t v_bias,
    ocean_tensor_handle_t q_output,
    ocean_tensor_handle_t k_output,
    ocean_tensor_handle_t v_output,
    int out_features
) {
    if (!input || !q_output || !k_output || !v_output) {
        ocean_tensor_fail("packed QKV output Tensor is null");
    }
    ocean_tensor_handle_t outputs[] = {q_output, k_output, v_output};
    for (int index = 0; index < 3; ++index) {
        if (outputs[index]->dtype != OCEAN_TENSOR_FLOAT32 ||
            outputs[index]->device != input->device ||
            outputs[index]->ndim != input->ndim ||
            !ocean_tensor_is_contiguous(outputs[index])) {
            ocean_tensor_fail(
                "packed QKV outputs must be contiguous float32 Tensors on the input device"
            );
        }
        for (size_t axis = 0; axis + 1 < input->ndim; ++axis) {
            if (outputs[index]->shape[axis] != input->shape[axis]) {
                ocean_tensor_fail("packed QKV output shape mismatch");
            }
        }
        if (outputs[index]->shape[input->ndim - 1] != (size_t)out_features) {
            ocean_tensor_fail("packed QKV output width mismatch");
        }
    }

    if (input->device == OCEAN_TENSOR_CPU) {
        ocean_tensor_handle_t combined = ocean_tensor_packed_qkv_inference_impl(
            input, q_packed_weight, q_scale, q_bias,
            k_packed_weight, k_scale, k_bias,
            v_packed_weight, v_scale, v_bias, out_features
        );
        int last_axis = (int)input->ndim - 1;
        for (int projection = 0; projection < 3; ++projection) {
            int start = projection * out_features;
            ocean_tensor_handle_t slice = ocean_tensor_slice(
                combined, last_axis, start, start + out_features, 1
            );
            ocean_tensor_copy_into(outputs[projection], slice);
            ocean_tensor_release(slice);
        }
        ocean_tensor_release(combined);
        return;
    }

    if (input->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
            ? input : ocean_tensor_contiguous(input);
        ocean_tensor_handle_t packed_weights[] = {
            q_packed_weight, k_packed_weight, v_packed_weight,
        };
        ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
        ocean_tensor_handle_t packed_contiguous[3];
        ocean_tensor_handle_t bias_contiguous[3];
        for (int index = 0; index < 3; ++index) {
            packed_contiguous[index] = ocean_tensor_is_contiguous(packed_weights[index])
                ? packed_weights[index] : ocean_tensor_contiguous(packed_weights[index]);
            bias_contiguous[index] = ocean_tensor_is_contiguous(biases[index])
                ? biases[index] : ocean_tensor_contiguous(biases[index]);
        }
        ocean_tensor_cuda_packed_qkv_split(
            input_contiguous,
            packed_contiguous[0], bias_contiguous[0],
            packed_contiguous[1], bias_contiguous[1],
            packed_contiguous[2], bias_contiguous[2],
            q_output, k_output, v_output,
            out_features, q_scale, k_scale, v_scale
        );
        if (input_contiguous != input) ocean_tensor_release(input_contiguous);
        for (int index = 0; index < 3; ++index) {
            if (packed_contiguous[index] != packed_weights[index]) {
                ocean_tensor_release(packed_contiguous[index]);
            }
            if (bias_contiguous[index] != biases[index]) {
                ocean_tensor_release(bias_contiguous[index]);
            }
        }
        return;
#else
        ocean_tensor_fail("CUDA backend was not compiled");
#endif
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t input_contiguous = ocean_tensor_is_contiguous(input)
        ? input : ocean_tensor_contiguous(input);
    ocean_tensor_handle_t packed_weights[] = {
        q_packed_weight, k_packed_weight, v_packed_weight,
    };
    ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
    ocean_tensor_handle_t packed_contiguous[3];
    ocean_tensor_handle_t bias_contiguous[3];
    for (int index = 0; index < 3; ++index) {
        packed_contiguous[index] = ocean_tensor_is_contiguous(packed_weights[index])
            ? packed_weights[index]
            : ocean_tensor_contiguous(packed_weights[index]);
        bias_contiguous[index] = ocean_tensor_is_contiguous(biases[index])
            ? biases[index] : ocean_tensor_contiguous(biases[index]);
    }
    ocean_tensor_opencl_packed_qkv_split(
        input_contiguous,
        packed_contiguous[0], bias_contiguous[0],
        packed_contiguous[1], bias_contiguous[1],
        packed_contiguous[2], bias_contiguous[2],
        q_output, k_output, v_output,
        out_features, q_scale, k_scale, v_scale
    );
    if (input_contiguous != input) ocean_tensor_release(input_contiguous);
    for (int index = 0; index < 3; ++index) {
        if (packed_contiguous[index] != packed_weights[index]) {
            ocean_tensor_release(packed_contiguous[index]);
        }
        if (bias_contiguous[index] != biases[index]) {
            ocean_tensor_release(bias_contiguous[index]);
        }
    }
    return;
#else
    ocean_tensor_fail(
        "GPU backend is unavailable: rebuild with OpenCL support"
    );
#endif
}

ocean_tensor_handle_t ocean_tensor_packed_qkv_attention_decode(
    ocean_tensor_handle_t input,
    ocean_tensor_handle_t q_packed_weight,
    double q_scale,
    ocean_tensor_handle_t q_bias,
    ocean_tensor_handle_t k_packed_weight,
    double k_scale,
    ocean_tensor_handle_t k_bias,
    ocean_tensor_handle_t v_packed_weight,
    double v_scale,
    ocean_tensor_handle_t v_bias,
    ocean_tensor_handle_t cache_k,
    ocean_tensor_handle_t cache_v,
    int position,
    int n_heads,
    int head_dim
) {
    ocean_tensor_handle_t tensors[] = {
        input, q_packed_weight, q_bias, k_packed_weight, k_bias,
        v_packed_weight, v_bias, cache_k, cache_v,
    };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        if (!tensors[index]) {
            ocean_tensor_fail("fused attention received a null Tensor");
        }
    }
    if (input->device == OCEAN_TENSOR_BACKEND_CPU ||
        input->dtype != OCEAN_TENSOR_FLOAT32 || input->ndim != 3 ||
        input->shape[0] != 1 || input->shape[1] != 1 ||
        n_heads <= 0 || head_dim <= 0 || head_dim > 128 ||
        position < 0 || cache_k->device != input->device ||
        cache_v->device != input->device ||
        cache_k->dtype != OCEAN_TENSOR_FLOAT32 ||
        cache_v->dtype != OCEAN_TENSOR_FLOAT32 || cache_k->ndim != 4 ||
        cache_v->ndim != 4 || cache_k->shape[0] != 1 ||
        cache_v->shape[0] != 1 || cache_k->shape[1] != (size_t)n_heads ||
        cache_v->shape[1] != (size_t)n_heads ||
        cache_k->shape[3] != (size_t)head_dim ||
        cache_v->shape[3] != (size_t)head_dim ||
        position >= (int)cache_k->shape[2] ||
        cache_v->shape[2] != cache_k->shape[2]) {
        ocean_tensor_fail("fused attention decode shape/device mismatch");
    }
    size_t d_model = (size_t)n_heads * (size_t)head_dim;
    if (input->shape[2] != d_model || q_packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        k_packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        v_packed_weight->dtype != OCEAN_TENSOR_INT32 ||
        q_bias->dtype != OCEAN_TENSOR_FLOAT32 ||
        k_bias->dtype != OCEAN_TENSOR_FLOAT32 ||
        v_bias->dtype != OCEAN_TENSOR_FLOAT32 || q_bias->ndim != 2 ||
        k_bias->ndim != 2 || v_bias->ndim != 2 || q_bias->shape[0] != 1 ||
        k_bias->shape[0] != 1 || v_bias->shape[0] != 1 ||
        q_bias->shape[1] != d_model || k_bias->shape[1] != d_model ||
        v_bias->shape[1] != d_model ||
        !(q_scale > 0.0) || !(k_scale > 0.0) || !(v_scale > 0.0)) {
        ocean_tensor_fail("fused attention projection metadata mismatch");
    }
    if (q_packed_weight->device != input->device ||
        k_packed_weight->device != input->device ||
        v_packed_weight->device != input->device ||
        q_bias->device != input->device || k_bias->device != input->device ||
        v_bias->device != input->device) {
        ocean_tensor_fail("fused attention requires GPU projection tensors");
    }
    size_t packed_cols = (d_model + 15u) / 16u;
    ocean_tensor_handle_t packed[] = {
        q_packed_weight, k_packed_weight, v_packed_weight,
    };
    for (int index = 0; index < 3; ++index) {
        if (packed[index]->ndim != 2 || packed[index]->shape[0] != d_model ||
            packed[index]->shape[1] != packed_cols ||
            !ocean_tensor_is_contiguous(packed[index])) {
            ocean_tensor_fail("fused attention packed weight mismatch");
        }
    }
    if (!ocean_tensor_is_contiguous(input) ||
        !ocean_tensor_is_contiguous(q_bias) ||
        !ocean_tensor_is_contiguous(k_bias) ||
        !ocean_tensor_is_contiguous(v_bias) ||
        !ocean_tensor_is_contiguous(cache_k) ||
        !ocean_tensor_is_contiguous(cache_v)) {
        ocean_tensor_fail("fused attention requires contiguous GPU tensors");
    }
    size_t output_shape[3] = {1, 1, d_model};
    ocean_tensor_handle_t output = ocean_tensor_alloc_uninitialized(
        output_shape, 3, OCEAN_TENSOR_FLOAT32, input->device
    );
#ifdef OCEAN_TENSOR_ENABLE_CUDA
    if (input->device == OCEAN_TENSOR_BACKEND_CUDA) {
        ocean_tensor_cuda_packed_attention_decode(
            input, q_packed_weight, q_bias,
            k_packed_weight, k_bias,
            v_packed_weight, v_bias,
            cache_k, cache_v, output,
            position, n_heads, head_dim,
            q_scale, k_scale, v_scale
        );
        return output;
    }
#endif
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_opencl_packed_qkv_attention_decode(
        input, q_packed_weight, q_bias,
        k_packed_weight, k_bias,
        v_packed_weight, v_bias,
        cache_k, cache_v, output,
        position, n_heads, head_dim,
        q_scale, k_scale, v_scale
    );
    return output;
#else
    ocean_tensor_release(output);
    ocean_tensor_fail(
        "GPU backend is unavailable: rebuild with OpenCL support"
    );
    return NULL;
#endif
}

static bool ocean_tensor_host_is_little_endian(void) {
    const uint16_t value = 1u;
    return *((const unsigned char *)&value) == 1u;
}

static const char *ocean_tensor_npy_descr(ocean_tensor_dtype dtype) {
    switch (dtype) {
        case OCEAN_TENSOR_BOOL: return "|b1";
        case OCEAN_TENSOR_INT8: return "|i1";
        case OCEAN_TENSOR_INT16: return "<i2";
        case OCEAN_TENSOR_INT32: return "<i4";
        case OCEAN_TENSOR_INT64: return "<i8";
        case OCEAN_TENSOR_UINT8: return "|u1";
        case OCEAN_TENSOR_UINT16: return "<u2";
        case OCEAN_TENSOR_UINT32: return "<u4";
        case OCEAN_TENSOR_UINT64: return "<u8";
        case OCEAN_TENSOR_FLOAT16: return "<f2";
        case OCEAN_TENSOR_FLOAT32: return "<f4";
        case OCEAN_TENSOR_FLOAT64: return "<f8";
    }
    ocean_tensor_fail("unsupported Tensor dtype for .npy");
    return "";
}

static bool ocean_tensor_npy_write(
    FILE *stream,
    const void *data,
    size_t bytes,
    size_t item_size
) {
    if (ocean_tensor_host_is_little_endian() || item_size <= 1 || bytes == 0) {
        return fwrite(data, 1, bytes, stream) == bytes;
    }

    unsigned char *swapped = (unsigned char *)malloc(bytes);
    if (!swapped) ocean_tensor_fail("out of memory byte-swapping .npy data");
    const unsigned char *source = (const unsigned char *)data;
    for (size_t offset = 0; offset < bytes; offset += item_size) {
        for (size_t byte = 0; byte < item_size; ++byte) {
            swapped[offset + byte] = source[offset + item_size - byte - 1];
        }
    }
    bool written = fwrite(swapped, 1, bytes, stream) == bytes;
    free(swapped);
    return written;
}

static char *ocean_tensor_npy_header(
    const ocean_tensor_handle_t tensor,
    unsigned char major,
    size_t *header_size_out
) {
    if (tensor->ndim > (SIZE_MAX - 256u) / 64u) {
        ocean_tensor_fail("Tensor rank is too large for .npy header");
    }
    size_t capacity = 256u + tensor->ndim * 64u;
    char *header = (char *)malloc(capacity);
    if (!header) ocean_tensor_fail("out of memory allocating .npy header");

    int prefix = snprintf(
        header, capacity,
        "{'descr': '%s', 'fortran_order': False, 'shape': (",
        ocean_tensor_npy_descr(tensor->dtype)
    );
    if (prefix < 0 || (size_t)prefix >= capacity) {
        free(header);
        ocean_tensor_fail("failed to format .npy header");
    }
    size_t position = (size_t)prefix;
    for (size_t axis = 0; axis < tensor->ndim; ++axis) {
        const char *separator = axis + 1 < tensor->ndim
            ? ", " : (tensor->ndim == 1 ? "," : "");
        int written = snprintf(
            header + position, capacity - position, "%zu%s",
            tensor->shape[axis], separator
        );
        if (written < 0 || (size_t)written >= capacity - position) {
            free(header);
            ocean_tensor_fail("failed to format .npy shape");
        }
        position += (size_t)written;
    }
    int suffix = snprintf(header + position, capacity - position, "), }");
    if (suffix < 0 || (size_t)suffix >= capacity - position) {
        free(header);
        ocean_tensor_fail("failed to finish .npy header");
    }
    position += (size_t)suffix;

    size_t prefix_size = major == 1 ? 10u : 12u;
    size_t with_newline = prefix_size + position + 1u;
    size_t padding = (64u - (with_newline % 64u)) % 64u;
    if (position > capacity - padding - 2u) {
        free(header);
        ocean_tensor_fail(".npy header is too large");
    }
    memset(header + position, ' ', padding);
    position += padding;
    header[position++] = '\n';
    header[position] = '\0';
    *header_size_out = position;
    return header;
}

static void ocean_tensor_npy_write_u16(FILE *stream, uint16_t value) {
    unsigned char bytes[2] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
    };
    if (fwrite(bytes, 1, sizeof(bytes), stream) != sizeof(bytes)) {
        ocean_tensor_fail("failed writing .npy header length");
    }
}

static void ocean_tensor_npy_write_u32(FILE *stream, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)((value >> 24) & 0xffu),
    };
    if (fwrite(bytes, 1, sizeof(bytes), stream) != sizeof(bytes)) {
        ocean_tensor_fail("failed writing .npy header length");
    }
}

void ocean_tensor_save_npy(
    ocean_tensor_handle_t tensor,
    const char *path
) {
    if (!tensor || !path) ocean_tensor_fail("Tensor.save_npy received invalid arguments");
    if (tensor->ndim == 0) ocean_tensor_fail("cannot save a scalar Tensor as .npy");

    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? NULL : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t source = cpu ? cpu : tensor;
    ocean_tensor_handle_t contiguous = ocean_tensor_is_contiguous(source)
        ? NULL : ocean_tensor_contiguous(source);
    if (contiguous) source = contiguous;

    size_t header_size = 0;
    unsigned char version = 1;
    char *header = ocean_tensor_npy_header(source, version, &header_size);
    if (header_size > UINT16_MAX) {
        free(header);
        version = 2;
        header = ocean_tensor_npy_header(source, version, &header_size);
    }
    if (version == 1 && header_size > UINT16_MAX) {
        free(header);
        if (contiguous) ocean_tensor_release(contiguous);
        if (cpu) ocean_tensor_release(cpu);
        ocean_tensor_fail(".npy v2 header length is required but could not be encoded");
    }
    if (version == 2 && header_size > UINT32_MAX) {
        free(header);
        if (contiguous) ocean_tensor_release(contiguous);
        if (cpu) ocean_tensor_release(cpu);
        ocean_tensor_fail(".npy header is too large");
    }

    FILE *stream = fopen(path, "wb");
    if (!stream) {
        free(header);
        if (contiguous) ocean_tensor_release(contiguous);
        if (cpu) ocean_tensor_release(cpu);
        ocean_tensor_fail("could not open .npy file for writing");
    }
    const unsigned char magic[6] = {0x93u, 'N', 'U', 'M', 'P', 'Y'};
    const unsigned char version_bytes[2] = {version, 0};
    bool ok = fwrite(magic, 1, sizeof(magic), stream) == sizeof(magic) &&
        fwrite(version_bytes, 1, sizeof(version_bytes), stream) == sizeof(version_bytes);
    if (ok) {
        if (version == 1) ocean_tensor_npy_write_u16(stream, (uint16_t)header_size);
        else ocean_tensor_npy_write_u32(stream, (uint32_t)header_size);
        ok = fwrite(header, 1, header_size, stream) == header_size &&
            ocean_tensor_npy_write(
                stream, source->cpu_data, ocean_tensor_bytes(source), source->item_size
            );
    }
    free(header);
    bool close_ok = fclose(stream) == 0;
    if (contiguous) ocean_tensor_release(contiguous);
    if (cpu) ocean_tensor_release(cpu);
    if (!ok || !close_ok) ocean_tensor_fail("failed writing .npy file");
}

static uint16_t ocean_tensor_npy_read_u16(const unsigned char *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static uint32_t ocean_tensor_npy_read_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static const char *ocean_tensor_npy_value(
    const char *header,
    const char *name
) {
    char quoted_name[32];
    int written = snprintf(quoted_name, sizeof(quoted_name), "'%s'", name);
    if (written < 0 || (size_t)written >= sizeof(quoted_name)) {
        ocean_tensor_fail("invalid .npy header field name");
    }
    const char *field = strstr(header, quoted_name);
    if (!field) {
        written = snprintf(quoted_name, sizeof(quoted_name), "\"%s\"", name);
        if (written < 0 || (size_t)written >= sizeof(quoted_name)) {
            ocean_tensor_fail("invalid .npy header field name");
        }
        field = strstr(header, quoted_name);
    }
    if (!field) ocean_tensor_fail(".npy header is missing a required field");
    const char *colon = strchr(field, ':');
    if (!colon) ocean_tensor_fail("invalid .npy header field");
    colon += 1;
    while (isspace((unsigned char)*colon)) ++colon;
    return colon;
}

static void ocean_tensor_npy_read_descr(
    const char *header,
    ocean_tensor_dtype *dtype_out,
    bool *swap_out
) {
    const char *value = ocean_tensor_npy_value(header, "descr");
    if (*value != '\'' && *value != '"') ocean_tensor_fail("invalid .npy descr");
    char quote = *value++;
    char descr[32];
    size_t length = 0;
    while (value[length] && value[length] != quote) {
        if (length + 1 >= sizeof(descr)) ocean_tensor_fail(".npy descr is too long");
        descr[length] = value[length];
        ++length;
    }
    if (value[length] != quote || length < 3) ocean_tensor_fail("invalid .npy descr");
    descr[length] = '\0';

    char order = descr[0];
    char kind = descr[1];
    if (order != '<' && order != '>' && order != '|' && order != '=') {
        ocean_tensor_fail("unsupported .npy byte order");
    }
    errno = 0;
    char *end = NULL;
    unsigned long item_size = strtoul(descr + 2, &end, 10);
    if (errno == ERANGE || end == descr + 2 || *end != '\0' || item_size > SIZE_MAX) {
        ocean_tensor_fail("invalid .npy dtype size");
    }

    ocean_tensor_dtype dtype;
    switch (kind) {
        case 'b':
        case '?':
            if (item_size != 1) ocean_tensor_fail("invalid .npy bool dtype");
            dtype = OCEAN_TENSOR_BOOL;
            break;
        case 'i':
            if (item_size == 1) dtype = OCEAN_TENSOR_INT8;
            else if (item_size == 2) dtype = OCEAN_TENSOR_INT16;
            else if (item_size == 4) dtype = OCEAN_TENSOR_INT32;
            else if (item_size == 8) dtype = OCEAN_TENSOR_INT64;
            else ocean_tensor_fail("unsupported .npy signed integer dtype");
            break;
        case 'u':
            if (item_size == 1) dtype = OCEAN_TENSOR_UINT8;
            else if (item_size == 2) dtype = OCEAN_TENSOR_UINT16;
            else if (item_size == 4) dtype = OCEAN_TENSOR_UINT32;
            else if (item_size == 8) dtype = OCEAN_TENSOR_UINT64;
            else ocean_tensor_fail("unsupported .npy unsigned integer dtype");
            break;
        case 'f':
            if (item_size == 2) dtype = OCEAN_TENSOR_FLOAT16;
            else if (item_size == 4) dtype = OCEAN_TENSOR_FLOAT32;
            else if (item_size == 8) dtype = OCEAN_TENSOR_FLOAT64;
            else ocean_tensor_fail("unsupported .npy floating-point dtype");
            break;
        default:
            ocean_tensor_fail(".npy dtype is not a supported numeric type");
            return;
    }
    if (item_size > 1 && order == '|') {
        ocean_tensor_fail("invalid .npy byte order for multi-byte dtype");
    }
    bool host_little = ocean_tensor_host_is_little_endian();
    *swap_out = item_size > 1 &&
        ((order == '<' && !host_little) || (order == '>' && host_little));
    *dtype_out = dtype;
}

static size_t *ocean_tensor_npy_read_shape(
    const char *header,
    size_t *ndim_out
) {
    const char *cursor = ocean_tensor_npy_value(header, "shape");
    if (*cursor != '(') ocean_tensor_fail("invalid .npy shape");
    ++cursor;
    size_t capacity = 4;
    size_t ndim = 0;
    size_t *shape = (size_t *)malloc(capacity * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory reading .npy shape");
    for (;;) {
        while (isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == ')') break;
        errno = 0;
        char *end = NULL;
        unsigned long long value = strtoull(cursor, &end, 10);
        if (errno == ERANGE || end == cursor || value > SIZE_MAX) {
            free(shape);
            ocean_tensor_fail("invalid .npy shape dimension");
        }
        if (ndim == capacity) {
            if (capacity > SIZE_MAX / 2u) {
                free(shape);
                ocean_tensor_fail(".npy rank is too large");
            }
            capacity *= 2u;
            size_t *grown = (size_t *)realloc(shape, capacity * sizeof(size_t));
            if (!grown) {
                free(shape);
                ocean_tensor_fail("out of memory growing .npy shape");
            }
            shape = grown;
        }
        shape[ndim++] = (size_t)value;
        cursor = end;
        while (isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor != ')') {
            free(shape);
            ocean_tensor_fail("invalid .npy shape tuple");
        }
        break;
    }
    if (ndim == 0) {
        free(shape);
        ocean_tensor_fail("scalar .npy arrays are not supported as Tensor");
    }
    *ndim_out = ndim;
    return shape;
}

ocean_tensor_handle_t ocean_tensor_load_npy(
    const char *path,
    const char *device
) {
    if (!path) ocean_tensor_fail("Tensor.load_npy requires a path");
    FILE *stream = fopen(path, "rb");
    if (!stream) ocean_tensor_fail("could not open .npy file for reading");

    unsigned char magic[6];
    unsigned char version[2];
    if (fread(magic, 1, sizeof(magic), stream) != sizeof(magic) ||
        memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0 ||
        fread(version, 1, sizeof(version), stream) != sizeof(version)) {
        fclose(stream);
        ocean_tensor_fail("invalid .npy file header");
    }
    uint64_t header_size;
    if (version[0] == 1) {
        unsigned char length[2];
        if (fread(length, 1, sizeof(length), stream) != sizeof(length)) {
            fclose(stream);
            ocean_tensor_fail("truncated .npy header length");
        }
        header_size = ocean_tensor_npy_read_u16(length);
    } else if (version[0] == 2 || version[0] == 3) {
        unsigned char length[4];
        if (fread(length, 1, sizeof(length), stream) != sizeof(length)) {
            fclose(stream);
            ocean_tensor_fail("truncated .npy header length");
        }
        header_size = ocean_tensor_npy_read_u32(length);
    } else {
        fclose(stream);
        ocean_tensor_fail("unsupported .npy format version");
    }
    if (header_size == 0 || header_size > 64u * 1024u * 1024u ||
        header_size > SIZE_MAX - 1u) {
        fclose(stream);
        ocean_tensor_fail("invalid .npy header size");
    }
    char *header = (char *)malloc((size_t)header_size + 1u);
    if (!header) {
        fclose(stream);
        ocean_tensor_fail("out of memory reading .npy header");
    }
    if (fread(header, 1, (size_t)header_size, stream) != (size_t)header_size) {
        free(header);
        fclose(stream);
        ocean_tensor_fail("truncated .npy header");
    }
    header[header_size] = '\0';
    const char *fortran = ocean_tensor_npy_value(header, "fortran_order");
    if (strncmp(fortran, "False", 5) != 0) {
        free(header);
        fclose(stream);
        ocean_tensor_fail("Fortran-order .npy arrays are not supported yet");
    }
    ocean_tensor_dtype dtype;
    bool swap = false;
    ocean_tensor_npy_read_descr(header, &dtype, &swap);
    size_t ndim = 0;
    size_t *shape = ocean_tensor_npy_read_shape(header, &ndim);
    free(header);

    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, ndim, dtype, OCEAN_TENSOR_CPU
    );
    free(shape);
    size_t bytes = ocean_tensor_bytes(result);
    if (!swap) {
        if (fread(result->cpu_data, 1, bytes, stream) != bytes) {
            ocean_tensor_release(result);
            fclose(stream);
            ocean_tensor_fail("truncated .npy data");
        }
    } else {
        unsigned char *payload = (unsigned char *)malloc(bytes);
        if (!payload && bytes != 0) {
            ocean_tensor_release(result);
            fclose(stream);
            ocean_tensor_fail("out of memory byte-swapping .npy data");
        }
        if (fread(payload, 1, bytes, stream) != bytes) {
            free(payload);
            ocean_tensor_release(result);
            fclose(stream);
            ocean_tensor_fail("truncated .npy data");
        }
        unsigned char *destination = (unsigned char *)result->cpu_data;
        for (size_t offset = 0; offset < bytes; offset += result->item_size) {
            for (size_t byte = 0; byte < result->item_size; ++byte) {
                destination[offset + byte] =
                    payload[offset + result->item_size - byte - 1];
            }
        }
        free(payload);
    }
    fclose(stream);

    if (device && strcmp(device, "cpu") == 0) return result;
    ocean_tensor_handle_t moved = ocean_tensor_to(result, device);
    ocean_tensor_release(result);
    return moved;
}

ocean_tensor_handle_t ocean_tensor_load_npy_typed(
    const char *path,
    const char *device,
    const char *expected_dtype
) {
    if (!expected_dtype) {
        ocean_tensor_fail("Tensor.load_npy requires an expected dtype");
    }
    ocean_tensor_dtype expected = ocean_tensor_parse_dtype(expected_dtype);
    ocean_tensor_handle_t result = ocean_tensor_load_npy(path, device);
    if (result->dtype != expected) {
        ocean_tensor_release(result);
        ocean_tensor_fail(".npy dtype does not match Tensor[T]");
    }
    return result;
}

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis) {
    if (!tensor) ocean_tensor_fail("shape() does not accept a null Tensor");
    if (axis < 0 || (size_t)axis >= tensor->ndim) {
        ocean_tensor_fail("Tensor shape axis is out of bounds");
    }
    return tensor->shape[axis] > (size_t)INT32_MAX
        ? INT32_MAX : (int)tensor->shape[axis];
}

int ocean_tensor_len(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("len() does not accept a null Tensor");
    return tensor->size > (size_t)INT32_MAX
        ? INT32_MAX : (int)tensor->size;
}

int ocean_tensor_ndim(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("ndim() does not accept a null Tensor");
    return tensor->ndim > (size_t)INT32_MAX ? INT32_MAX : (int)tensor->ndim;
}

size_t ocean_tensor_size(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("size() does not accept a null Tensor");
    return tensor->size;
}

char *ocean_tensor_device(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("device() does not accept a null Tensor");
    const char *name = tensor->device == OCEAN_TENSOR_CPU ? "cpu" : "gpu";
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying device name");
    strcpy(result, name);
    return result;
}

char *ocean_tensor_device_info(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("device_info() does not accept a null Tensor");
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_BACKEND_OPENCL &&
        ocean_tensor_opencl_initialized) {
        size_t size = 0;
        ocean_tensor_opencl_check(
            clGetDeviceInfo(
                ocean_tensor_opencl.device,
                CL_DEVICE_NAME,
                0,
                NULL,
                &size
            ),
            "clGetDeviceInfo"
        );
        char *result = (char *)calloc(size + 1, 1);
        if (!result) ocean_tensor_fail("out of memory copying OpenCL device name");
        ocean_tensor_opencl_check(
            clGetDeviceInfo(
                ocean_tensor_opencl.device,
                CL_DEVICE_NAME,
                size,
                result,
                NULL
            ),
            "clGetDeviceInfo"
        );
        return result;
    }
#endif
    const char *name = tensor->device == OCEAN_TENSOR_BACKEND_CUDA
        ? "CUDA GPU (native kernels)"
        : (tensor->device == OCEAN_TENSOR_BACKEND_OPENCL
            ? "OpenCL GPU (not initialized)" : "CPU");
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying device info");
    strcpy(result, name);
    return result;
}

bool ocean_tensor_sparse_attention_cuda_available(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("SparseAttention capability query on null Tensor");
#ifdef OCEAN_TENSOR_ENABLE_CUDA
    return tensor->device == OCEAN_TENSOR_BACKEND_CUDA &&
        tensor->dtype == OCEAN_TENSOR_FLOAT32;
#else
    return false;
#endif
}

uint64_t ocean_tensor_identity(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("Tensor identity on null handle");
    return tensor->identity;
}

void ocean_tensor_release(ocean_tensor_handle_t tensor) {
    if (!tensor) return;
    ocean_tensor_backend_for_device(tensor->device)->release(tensor);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

ocean_tensor_handle_t ocean_tensor_permute(
    ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
) {
    if (!tensor) {
        ocean_tensor_fail("Tensor.permute requires a Tensor");
    }
    if (!axes) {
        ocean_tensor_fail("Tensor.permute requires axes");
    }

    int rank = ocean_tensor_ndim(tensor);
    if (rank <= 0 || (size_t)rank != ndim) {
        ocean_tensor_fail("Tensor.permute axes must match Tensor rank");
    }

    int *normalized = (int *)malloc(ndim * sizeof(int));
    bool *seen = (bool *)calloc(ndim, sizeof(bool));

    if (!normalized || !seen) {
        free(normalized);
        free(seen);
        ocean_tensor_fail("out of memory in Tensor.permute");
    }

    for (size_t i = 0; i < ndim; ++i) {
        long long axis = (long long)axes[i];
        if (axis < 0) axis += (long long)rank;
        if (axis < 0 || axis >= (long long)rank) {
            free(normalized);
            free(seen);
            ocean_tensor_fail("Tensor.permute axis is out of bounds");
        }
        if (seen[(size_t)axis]) {
            free(normalized);
            free(seen);
            ocean_tensor_fail("Tensor.permute axes must be unique");
        }

        normalized[i] = (int)axis;
        seen[(size_t)axis] = true;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU) {
        size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
        if (!shape) {
            free(normalized);
            free(seen);
            ocean_tensor_fail("out of memory in Tensor.permute shape");
        }
        for (size_t axis = 0; axis < ndim; ++axis) {
            shape[axis] = tensor->shape[(size_t)normalized[axis]];
        }
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            shape, ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_permute(tensor, result, normalized);
        free(shape);
        free(normalized);
        free(seen);
        return result;
    }
#endif

    if (tensor->device == OCEAN_TENSOR_BACKEND_CUDA) {
#ifdef OCEAN_TENSOR_ENABLE_CUDA
        if (tensor->dtype == OCEAN_TENSOR_FLOAT32 && ndim == 4 &&
            normalized[0] == 0 && normalized[1] == 2 &&
            normalized[2] == 1 && normalized[3] == 3 &&
            ocean_tensor_is_contiguous(tensor) &&
            tensor->shape[0] <= (size_t)INT_MAX &&
            tensor->shape[1] <= (size_t)INT_MAX &&
            tensor->shape[2] <= (size_t)INT_MAX &&
            tensor->shape[3] <= (size_t)INT_MAX) {
            size_t shape[4] = {
                tensor->shape[0], tensor->shape[2],
                tensor->shape[1], tensor->shape[3]
            };
            ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
                shape, 4, OCEAN_TENSOR_FLOAT32,
                OCEAN_TENSOR_BACKEND_CUDA
            );
            ocean_cuda_permute_swap12_f32(
                tensor->cuda_data, result->cuda_data,
                (int)tensor->shape[0], (int)tensor->shape[1],
                (int)tensor->shape[2], (int)tensor->shape[3]
            );
            free(normalized);
            free(seen);
            return result;
        }
#endif
        ocean_tensor_handle_t cpu_tensor = ocean_tensor_to(tensor, "cpu");
        ocean_tensor_handle_t cpu_result = ocean_tensor_permute_cpu(
            cpu_tensor, normalized, ndim
        );
        ocean_tensor_handle_t result = ocean_tensor_to(cpu_result, "cuda");
        ocean_tensor_release(cpu_result);
        ocean_tensor_release(cpu_tensor);
        free(normalized);
        free(seen);
        return result;
    }

    ocean_tensor_handle_t result = ocean_tensor_permute_cpu(
        tensor, normalized, ndim
    );
    free(normalized);
    free(seen);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_permute_cpu(
    const ocean_tensor_handle_t tensor,
    const int *axes,
    size_t ndim
) {
    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!shape) ocean_tensor_fail("out of memory in CPU Tensor.permute shape");
    for (size_t axis = 0; axis < ndim; ++axis) {
        shape[axis] = tensor->shape[(size_t)axes[axis]];
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    free(shape);

    size_t *coordinates = (size_t *)calloc(
        ndim == 0 ? 1 : ndim, sizeof(size_t)
    );
    if (!coordinates) {
        ocean_tensor_release(result);
        ocean_tensor_fail("out of memory in CPU Tensor.permute coordinates");
    }

    for (size_t output_index = 0; output_index < result->size; ++output_index) {
        size_t remaining = output_index;
        size_t input_index = 0;

        for (size_t axis = ndim; axis-- > 0;) {
            size_t extent = result->shape[axis];
            size_t coordinate = extent == 0 ? 0 : remaining % extent;
            remaining = extent == 0 ? 0 : remaining / extent;
            coordinates[axis] = coordinate;
            input_index += coordinate * tensor->strides[(size_t)axes[axis]];
        }

        ocean_tensor_write_scalar(
            result,
            output_index,
            ocean_tensor_read_scalar(tensor, input_index)
        );
    }

    free(coordinates);
    return result;
}
