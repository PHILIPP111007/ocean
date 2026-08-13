#include "std/tensor/tensor_runtime.h"

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include <CL/cl.h>
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

struct ocean_tensor_handle {
    int device;
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
};

enum {
    OCEAN_TENSOR_CPU = 0,
    OCEAN_TENSOR_GPU = 1,
};

static void ocean_tensor_fail(const char *message) {
    fprintf(stderr, "Ocean Tensor error: %s\n", message);
    exit(EXIT_FAILURE);
}

static int ocean_tensor_parse_device(const char *device) {
    if (device && strcmp(device, "cpu") == 0) return OCEAN_TENSOR_CPU;
    if (device && strcmp(device, "gpu") == 0) return OCEAN_TENSOR_GPU;
    ocean_tensor_fail("device must be \"cpu\" or \"gpu\"");
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

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->item_size = ocean_tensor_dtype_size(dtype);
    tensor->size = ocean_tensor_elements_from_shape(shape, ndim);
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        ocean_tensor_fail("out of memory allocating Tensor metadata");
    }
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->strides[ndim - 1] = 1;
    for (size_t axis = ndim - 1; axis > 0; --axis) {
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
    size_t bytes = ocean_tensor_bytes(tensor);
    if (device == OCEAN_TENSOR_CPU) {
        tensor->cpu_data = bytes ? calloc(1, bytes) : NULL;
        if (bytes && !tensor->cpu_data) {
            ocean_tensor_fail("out of memory allocating CPU Tensor");
        }
    }
    return tensor;
}

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static const char *ocean_tensor_matmul_kernel_source =
    "__kernel void ocean_tensor_matmul("
    "__global const float *a, __global const float *b, __global float *c, "
    "const int rows_a, const int cols_a, const int cols_b) {"
    "int row = (int)get_global_id(0);"
    "int col = (int)get_global_id(1);"
    "if (row < rows_a && col < cols_b) {"
    "float sum = 0.0f;"
    "for (int k = 0; k < cols_a; ++k)"
    "sum += a[row * cols_a + k] * b[k * cols_b + col];"
    "c[row * cols_b + col] = sum;"
    "}"
    "}"
    "__kernel void ocean_tensor_binary("
    "__global const float *a, __global const float *b, __global float *out, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "float left = a[index]; float right = b[index];"
    "if (operation == 0) out[index] = left + right;"
    "else if (operation == 1) out[index] = left - right;"
    "else if (operation == 2) out[index] = left * right;"
    "else out[index] = left / right;"
    "}"
    "}"
    "__kernel void ocean_tensor_scalar("
    "__global const float *input, __global float *out, const float scalar, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "float value = input[index];"
    "if (operation == 0) out[index] = value + scalar;"
    "else if (operation == 1) out[index] = value - scalar;"
    "else if (operation == 2) out[index] = value * scalar;"
    "else out[index] = value / scalar;"
    "}"
    "}"
    "__kernel void ocean_tensor_matmul_int32("
    "__global const int *a, __global const int *b, __global int *c, "
    "const int rows_a, const int cols_a, const int cols_b) {"
    "int row = (int)get_global_id(0);"
    "int col = (int)get_global_id(1);"
    "if (row < rows_a && col < cols_b) {"
    "int sum = 0;"
    "for (int k = 0; k < cols_a; ++k)"
    "sum += a[row * cols_a + k] * b[k * cols_b + col];"
    "c[row * cols_b + col] = sum;"
    "}"
    "}"
    "__kernel void ocean_tensor_binary_int32("
    "__global const int *a, __global const int *b, __global int *out, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "int left = a[index]; int right = b[index];"
    "if (operation == 0) out[index] = left + right;"
    "else if (operation == 1) out[index] = left - right;"
    "else if (operation == 2) out[index] = left * right;"
    "else out[index] = right == 0 ? 0 : left / right;"
    "}"
    "}"
    "__kernel void ocean_tensor_scalar_int32("
    "__global const int *input, __global int *out, const int scalar, "
    "const int operation, const int size) {"
    "int index = (int)get_global_id(0);"
    "if (index < size) {"
    "int value = input[index];"
    "if (operation == 0) out[index] = value + scalar;"
    "else if (operation == 1) out[index] = value - scalar;"
    "else if (operation == 2) out[index] = value * scalar;"
    "else out[index] = scalar == 0 ? 0 : value / scalar;"
    "}"
    "}";

typedef struct ocean_tensor_opencl_runtime {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_kernel;
    cl_kernel matmul_int32_kernel;
    cl_kernel binary_kernel;
    cl_kernel binary_int32_kernel;
    cl_kernel scalar_kernel;
    cl_kernel scalar_int32_kernel;
} ocean_tensor_opencl_runtime;

static ocean_tensor_opencl_runtime ocean_tensor_opencl;
static int ocean_tensor_opencl_initialized = 0;

static void ocean_tensor_opencl_shutdown(void) {
    if (!ocean_tensor_opencl_initialized) return;
    if (ocean_tensor_opencl.scalar_kernel) {
        clReleaseKernel(ocean_tensor_opencl.scalar_kernel);
        ocean_tensor_opencl.scalar_kernel = NULL;
    }
    if (ocean_tensor_opencl.binary_kernel) {
        clReleaseKernel(ocean_tensor_opencl.binary_kernel);
        ocean_tensor_opencl.binary_kernel = NULL;
    }
    if (ocean_tensor_opencl.matmul_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matmul_kernel);
        ocean_tensor_opencl.matmul_kernel = NULL;
    }
    if (ocean_tensor_opencl.matmul_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matmul_int32_kernel);
        ocean_tensor_opencl.matmul_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.scalar_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.scalar_int32_kernel);
        ocean_tensor_opencl.scalar_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.binary_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.binary_int32_kernel);
        ocean_tensor_opencl.binary_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.program) {
        clReleaseProgram(ocean_tensor_opencl.program);
        ocean_tensor_opencl.program = NULL;
    }
    if (ocean_tensor_opencl.queue) {
        clReleaseCommandQueue(ocean_tensor_opencl.queue);
        ocean_tensor_opencl.queue = NULL;
    }
    if (ocean_tensor_opencl.context) {
        clReleaseContext(ocean_tensor_opencl.context);
        ocean_tensor_opencl.context = NULL;
    }
    ocean_tensor_opencl_initialized = 0;
}

static void ocean_tensor_opencl_check(cl_int status, const char *operation) {
    if (status != CL_SUCCESS) {
        char message[256];
        snprintf(message, sizeof(message),
                 "%s failed (OpenCL error %d)", operation, status);
        ocean_tensor_fail(message);
    }
}

static void ocean_tensor_opencl_init(void) {
    if (ocean_tensor_opencl_initialized) return;
    ocean_tensor_opencl_initialized = 1;
    atexit(ocean_tensor_opencl_shutdown);

    cl_uint platform_count = 0;
    ocean_tensor_opencl_check(
        clGetPlatformIDs(0, NULL, &platform_count), "clGetPlatformIDs"
    );
    if (platform_count == 0) ocean_tensor_fail("no OpenCL platform is available");

    cl_platform_id platform = NULL;
    ocean_tensor_opencl_check(
        clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs"
    );
    cl_device_id device = NULL;
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (status != CL_SUCCESS) {
        ocean_tensor_opencl_check(
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL),
            "clGetDeviceIDs"
        );
    }

    ocean_tensor_opencl.context =
        clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    ocean_tensor_opencl_check(status, "clCreateContext");
    ocean_tensor_opencl.queue =
        clCreateCommandQueue(ocean_tensor_opencl.context, device, 0, &status);
    ocean_tensor_opencl_check(status, "clCreateCommandQueue");

    const char *sources[] = {ocean_tensor_matmul_kernel_source};
    ocean_tensor_opencl.program = clCreateProgramWithSource(
        ocean_tensor_opencl.context, 1, sources, NULL, &status
    );
    ocean_tensor_opencl_check(status, "clCreateProgramWithSource");
    status = clBuildProgram(
        ocean_tensor_opencl.program, 1, &device, NULL, NULL, NULL
    );
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(
            ocean_tensor_opencl.program, device, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size
        );
        char *log = (char *)calloc(log_size + 1, 1);
        if (log) {
            clGetProgramBuildInfo(
                ocean_tensor_opencl.program, device, CL_PROGRAM_BUILD_LOG,
                log_size, log, NULL
            );
            fprintf(stderr, "Ocean Tensor OpenCL build log:\n%s\n", log);
            free(log);
        }
        ocean_tensor_opencl_check(status, "clBuildProgram");
    }
    ocean_tensor_opencl.matmul_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_matmul", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
    ocean_tensor_opencl.matmul_int32_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_matmul_int32", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
    ocean_tensor_opencl.binary_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_binary", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
    ocean_tensor_opencl.binary_int32_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_binary_int32", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
    ocean_tensor_opencl.scalar_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_scalar", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
    ocean_tensor_opencl.scalar_int32_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_scalar_int32", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
}

static void ocean_tensor_gpu_alloc(ocean_tensor_handle_t tensor) {
    ocean_tensor_opencl_init();
    cl_int status = CL_SUCCESS;
    tensor->gpu_data = clCreateBuffer(
        ocean_tensor_opencl.context, CL_MEM_READ_WRITE,
        ocean_tensor_bytes(tensor) ? ocean_tensor_bytes(tensor) : 1,
        NULL, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
}

static void ocean_tensor_gpu_write(
    ocean_tensor_handle_t tensor,
    const void *data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE,
            0, bytes, data, 0, NULL, NULL
        ),
        "clEnqueueWriteBuffer"
    );
}

static void ocean_tensor_gpu_zero(ocean_tensor_handle_t tensor) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;
    void *zeros = calloc(1, bytes);
    if (!zeros) ocean_tensor_fail("out of memory zeroing GPU Tensor");
    ocean_tensor_gpu_write(tensor, zeros);
    free(zeros);
}

static void ocean_tensor_gpu_read(
    ocean_tensor_handle_t tensor,
    void *data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;
    ocean_tensor_opencl_check(
        clEnqueueReadBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE,
            0, bytes, data, 0, NULL, NULL
        ),
        "clEnqueueReadBuffer"
    );
}
#endif

ocean_tensor_handle_t ocean_tensor_zeros_nd(
    const size_t *shape,
    size_t ndim,
    const char *dtype,
    const char *device
) {
    ocean_tensor_handle_t tensor = ocean_tensor_alloc_zeros(
        shape, ndim, ocean_tensor_parse_dtype(dtype),
        ocean_tensor_parse_device(device)
    );
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_gpu_alloc(tensor);
        ocean_tensor_gpu_zero(tensor);
    }
#else
    if (tensor->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    }
#endif
    return tensor;
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
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_gpu_alloc(tensor);
        ocean_tensor_gpu_zero(tensor);
    }
#else
    if (tensor->device == OCEAN_TENSOR_GPU) {
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    }
#endif
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
    if (!shape || !strides) ocean_tensor_fail("Tensor metadata cannot be null");
    ocean_tensor_dtype parsed_dtype = ocean_tensor_parse_dtype(dtype);
    ocean_tensor_handle_t host = ocean_tensor_alloc_zeros(
        shape, ndim, parsed_dtype, OCEAN_TENSOR_CPU
    );
    size_t item_size = host->item_size;
    unsigned char *destination = (unsigned char *)host->cpu_data;
    const unsigned char *source = (const unsigned char *)data;

    for (size_t linear = 0; linear < host->size; ++linear) {
        size_t remaining = linear;
        size_t source_offset = 0;
        for (size_t axis = ndim; axis-- > 0;) {
            size_t coordinate = shape[axis] ? remaining % shape[axis] : 0;
            remaining = shape[axis] ? remaining / shape[axis] : 0;
            source_offset += coordinate * strides[axis];
        }
        if (destination && source) {
            memcpy(destination + linear * item_size,
                   source + source_offset * item_size, item_size);
        }
    }

    int target = ocean_tensor_parse_device(device);
    if (target == OCEAN_TENSOR_CPU) return host;
    ocean_tensor_handle_t result = ocean_tensor_to(host, device);
    ocean_tensor_release(host);
    return result;
}

typedef struct ocean_tensor_native_layout {
    void *data;
    size_t *shape;
    size_t *strides;
    size_t ndim;
} ocean_tensor_native_layout;

ocean_tensor_handle_t ocean_tensor_from_cpu_native(
    const void *source,
    const char *dtype,
    const char *device
) {
    if (!source) ocean_tensor_fail("cannot import a null native tensor");
    const ocean_tensor_native_layout *native =
        (const ocean_tensor_native_layout *)source;
    return ocean_tensor_from_cpu_strided(
        native->data, native->shape, native->strides, native->ndim,
        dtype, device
    );
}

ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("cannot copy a null Tensor");
    ocean_tensor_handle_t result = ocean_tensor_alloc(
        tensor->shape, tensor->ndim, tensor->dtype, tensor->device
    );
    size_t bytes = ocean_tensor_bytes(tensor);
    if (tensor->device == OCEAN_TENSOR_CPU) {
        result->cpu_data = bytes ? malloc(bytes) : NULL;
        if (bytes && !result->cpu_data) ocean_tensor_fail("out of memory copying Tensor");
        if (bytes) memcpy(result->cpu_data, tensor->cpu_data, bytes);
        return result;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_alloc(result);
    ocean_tensor_opencl_check(
        clEnqueueCopyBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, result->gpu_data,
            0, 0, bytes, 0, NULL, NULL
        ),
        "clEnqueueCopyBuffer"
    );
    ocean_tensor_opencl_check(
        clFinish(ocean_tensor_opencl.queue), "clFinish"
    );
    return result;
#else
    free(result->shape);
    free(result->strides);
    free(result);
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_to(
    ocean_tensor_handle_t tensor,
    const char *device
) {
    if (!tensor) ocean_tensor_fail("cannot move a null Tensor");
    int target = ocean_tensor_parse_device(device);
    if (target == tensor->device) return ocean_tensor_copy(tensor);

    if (target == OCEAN_TENSOR_CPU) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        ocean_tensor_handle_t result = ocean_tensor_alloc(
            tensor->shape, tensor->ndim, tensor->dtype, target
        );
        size_t bytes = ocean_tensor_bytes(tensor);
        result->cpu_data = bytes ? malloc(bytes) : NULL;
        if (bytes && !result->cpu_data) ocean_tensor_fail("out of memory downloading Tensor");
        ocean_tensor_gpu_read(tensor, result->cpu_data);
        return result;
#else
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_handle_t result = ocean_tensor_alloc(
        tensor->shape, tensor->ndim, tensor->dtype, target
    );
    ocean_tensor_gpu_alloc(result);
    ocean_tensor_gpu_write(result, tensor->cpu_data);
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    return NULL;
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

enum {
    OCEAN_TENSOR_ADD = 0,
    OCEAN_TENSOR_SUB = 1,
    OCEAN_TENSOR_MUL = 2,
    OCEAN_TENSOR_DIV = 3,
};

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
        shape[axis] = left_axis > right_axis ? left_axis : right_axis;
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
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, ndim, left->dtype, OCEAN_TENSOR_CPU
    );
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
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t linear = 0; linear < tensor->size; ++linear) {
        ocean_tensor_write_scalar(
            result,
            linear,
            ocean_tensor_apply_binary(
                ocean_tensor_read_scalar(tensor, linear),
                (long double)scalar,
                operation
            )
        );
    }
    return result;
}

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static int ocean_tensor_same_shape(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right
) {
    if (left->ndim != right->ndim) return 0;
    for (size_t axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return 0;
    }
    return 1;
}

static void ocean_tensor_opencl_binary(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    ocean_tensor_handle_t result,
    int operation,
    cl_kernel kernel
) {
    int size = (int)left->size;
    if (left->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &left->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &right->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &result->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &operation),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size),
        "clSetKernelArg"
    );
    size_t global_size = left->size ? left->size : 1;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, NULL
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFinish(ocean_tensor_opencl.queue), "clFinish"
    );
}

static void ocean_tensor_opencl_scalar(
    const ocean_tensor_handle_t tensor,
    ocean_tensor_handle_t result,
    double scalar,
    int operation,
    cl_kernel kernel,
    int integer_kernel
) {
    if (tensor->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    float scalar_value = (float)scalar;
    int integer_value = (int)scalar;
    int size = (int)tensor->size;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &tensor->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &result->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel, 2,
            integer_kernel ? sizeof(int) : sizeof(float),
            integer_kernel ? (const void *)&integer_value : (const void *)&scalar_value
        ),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &operation),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size),
        "clSetKernelArg"
    );
    size_t global_size = tensor->size ? tensor->size : 1;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, NULL
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFinish(ocean_tensor_opencl.queue), "clFinish"
    );
}
#endif

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
    if (left->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_binary_cpu(left, right, operation);
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if ((left->dtype == OCEAN_TENSOR_FLOAT32 || left->dtype == OCEAN_TENSOR_INT32) &&
        ocean_tensor_same_shape(left, right)) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
            left->shape, left->ndim, left->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_gpu_alloc(result);
        cl_kernel kernel = left->dtype == OCEAN_TENSOR_INT32
            ? ocean_tensor_opencl.binary_int32_kernel
            : ocean_tensor_opencl.binary_kernel;
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
    if (tensor->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_scalar_cpu(tensor, scalar, operation);
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->dtype == OCEAN_TENSOR_FLOAT32 || tensor->dtype == OCEAN_TENSOR_INT32) {
        ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
            tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_GPU
        );
        ocean_tensor_gpu_alloc(result);
        cl_kernel kernel = tensor->dtype == OCEAN_TENSOR_INT32
            ? ocean_tensor_opencl.scalar_int32_kernel
            : ocean_tensor_opencl.scalar_kernel;
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
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

ocean_tensor_handle_t ocean_tensor_reshape(
    ocean_tensor_handle_t tensor,
    const size_t *shape,
    size_t ndim
) {
    if (!tensor || !shape) ocean_tensor_fail("Tensor reshape received null metadata");
    if (ocean_tensor_elements_from_shape(shape, ndim) != tensor->size) {
        ocean_tensor_fail("Tensor reshape must preserve the number of elements");
    }
    ocean_tensor_handle_t cpu_source = tensor->device == OCEAN_TENSOR_CPU
        ? ocean_tensor_copy(tensor) : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, ndim, tensor->dtype, OCEAN_TENSOR_CPU
    );
    if (tensor->size) memcpy(cpu_result->cpu_data, cpu_source->cpu_data, ocean_tensor_bytes(tensor));
    ocean_tensor_handle_t result = tensor->device == OCEAN_TENSOR_CPU
        ? cpu_result : ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(cpu_source);
    if (tensor->device != OCEAN_TENSOR_CPU) ocean_tensor_release(cpu_result);
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
    if (tensor->ndim != 2) ocean_tensor_fail("Tensor transpose currently expects 2D Tensor");
    size_t shape[2] = {tensor->shape[1], tensor->shape[0]};
    ocean_tensor_handle_t cpu_source = tensor->device == OCEAN_TENSOR_CPU
        ? ocean_tensor_copy(tensor) : ocean_tensor_to(tensor, "cpu");
    ocean_tensor_handle_t cpu_result = ocean_tensor_alloc_zeros(
        shape, 2, tensor->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t row = 0; row < tensor->shape[0]; ++row) {
        for (size_t col = 0; col < tensor->shape[1]; ++col) {
            size_t source_index = row * cpu_source->strides[0] + col * cpu_source->strides[1];
            size_t result_index = col * cpu_result->strides[0] + row * cpu_result->strides[1];
            memcpy(
                (unsigned char *)cpu_result->cpu_data + result_index * tensor->item_size,
                (unsigned char *)cpu_source->cpu_data + source_index * tensor->item_size,
                tensor->item_size
            );
        }
    }
    ocean_tensor_handle_t result = tensor->device == OCEAN_TENSOR_CPU
        ? cpu_result : ocean_tensor_to(cpu_result, "gpu");
    ocean_tensor_release(cpu_source);
    if (tensor->device != OCEAN_TENSOR_CPU) ocean_tensor_release(cpu_result);
    return result;
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

void ocean_tensor_fill(ocean_tensor_handle_t tensor, double value) {
    if (!tensor) ocean_tensor_fail("Tensor fill on null handle");
    if (tensor->device == OCEAN_TENSOR_CPU) {
        for (size_t index = 0; index < tensor->size; ++index) {
            ocean_tensor_write_scalar(tensor, index, (long double)value);
        }
        return;
    }
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    ocean_tensor_fill(cpu, value);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    ocean_tensor_release(cpu);
}

double ocean_tensor_get_2d(ocean_tensor_handle_t tensor, int row, int col) {
    if (!tensor || tensor->ndim != 2) {
        ocean_tensor_fail("Tensor get expects a 2D Tensor");
    }
    if (row < 0 || col < 0 || (size_t)row >= tensor->shape[0]
        || (size_t)col >= tensor->shape[1]) {
        ocean_tensor_fail("Tensor get index is out of bounds");
    }
    ocean_tensor_handle_t cpu = tensor->device == OCEAN_TENSOR_CPU
        ? tensor : ocean_tensor_to(tensor, "cpu");
    size_t offset = (size_t)row * cpu->strides[0]
        + (size_t)col * cpu->strides[1];
    double result = (double)ocean_tensor_read_scalar(cpu, offset);
    if (cpu != tensor) ocean_tensor_release(cpu);
    return result;
}

void ocean_tensor_set_2d(
    ocean_tensor_handle_t tensor, int row, int col, double value
) {
    if (!tensor || tensor->ndim != 2) {
        ocean_tensor_fail("Tensor set expects a 2D Tensor");
    }
    if (row < 0 || col < 0 || (size_t)row >= tensor->shape[0]
        || (size_t)col >= tensor->shape[1]) {
        ocean_tensor_fail("Tensor set index is out of bounds");
    }
    if (tensor->device == OCEAN_TENSOR_CPU) {
        size_t offset = (size_t)row * tensor->strides[0]
            + (size_t)col * tensor->strides[1];
        ocean_tensor_write_scalar(tensor, offset, (long double)value);
        return;
    }
    ocean_tensor_handle_t cpu = ocean_tensor_to(tensor, "cpu");
    size_t offset = (size_t)row * cpu->strides[0]
        + (size_t)col * cpu->strides[1];
    ocean_tensor_write_scalar(cpu, offset, (long double)value);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_write(tensor, cpu->cpu_data);
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    ocean_tensor_release(cpu);
}

static ocean_tensor_handle_t ocean_tensor_matmul_cpu(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    size_t shape[2] = {left->shape[0], right->shape[1]};
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, 2, left->dtype, OCEAN_TENSOR_CPU
    );
    for (size_t row = 0; row < left->shape[0]; ++row) {
        for (size_t col = 0; col < right->shape[1]; ++col) {
            long double sum = 0.0L;
            for (size_t k = 0; k < left->shape[1]; ++k) {
                size_t left_index = row * left->strides[0] + k * left->strides[1];
                size_t right_index = k * right->strides[0] + col * right->strides[1];
                sum += ocean_tensor_read_scalar(left, left_index)
                    * ocean_tensor_read_scalar(right, right_index);
            }
            ocean_tensor_write_scalar(
                result, row * result->strides[0] + col * result->strides[1], sum
            );
        }
    }
    return result;
}

ocean_tensor_handle_t ocean_tensor_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    if (!left || !right) ocean_tensor_fail("matmul does not accept null Tensors");
    if (left->ndim != 2 || right->ndim != 2) {
        ocean_tensor_fail("matmul currently expects 2D Tensors");
    }
    if (left->shape[1] != right->shape[0]) {
        ocean_tensor_fail("matmul shape mismatch");
    }
    if (left->dtype != right->dtype) {
        ocean_tensor_fail("matmul requires matching Tensor dtypes");
    }
    if (left->device != right->device) {
        ocean_tensor_fail("matmul requires Tensors on the same device");
    }
    if (left->device == OCEAN_TENSOR_CPU) {
        return ocean_tensor_matmul_cpu(left, right);
    }

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
    ocean_tensor_handle_t result = ocean_tensor_alloc_zeros(
        shape, 2, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
    );
    ocean_tensor_gpu_alloc(result);
    if (left->shape[0] == 0 || right->shape[1] == 0) {
        return result;
    }
    if (left->shape[0] > (size_t)INT32_MAX ||
        left->shape[1] > (size_t)INT32_MAX ||
        right->shape[1] > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor dimensions are too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = left->dtype == OCEAN_TENSOR_INT32
        ? ocean_tensor_opencl.matmul_int32_kernel
        : ocean_tensor_opencl.matmul_kernel;
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
    size_t global_size[2] = {
        ((size_t)rows + 7u) / 8u * 8u,
        ((size_t)result_cols + 7u) / 8u * 8u,
    };
    size_t local_size[2] = {8u, 8u};
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 2, NULL,
            global_size, local_size, 0, NULL, NULL
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFinish(ocean_tensor_opencl.queue), "clFinish"
    );
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
    return NULL;
#endif
}

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis) {
    if (!tensor) ocean_tensor_fail("shape() does not accept a null Tensor");
    if (axis < 0 || (size_t)axis >= tensor->ndim) {
        ocean_tensor_fail("Tensor shape axis is out of bounds");
    }
    return tensor->shape[axis] > (size_t)INT32_MAX
        ? INT32_MAX : (int)tensor->shape[axis];
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
    const char *name = tensor->device == OCEAN_TENSOR_GPU ? "gpu" : "cpu";
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying device name");
    strcpy(result, name);
    return result;
}

typedef struct ocean_tensor_export_layout {
    void *data;
    size_t *shape;
    size_t *strides;
    size_t ndim;
    size_t size;
    size_t refcount;
    bool is_view;
    struct ocean_tensor_export_layout *owner;
} ocean_tensor_export_layout;

void *ocean_tensor_to_cpu_tensor(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("cannot export a null Tensor");
    ocean_tensor_export_layout *result =
        (ocean_tensor_export_layout *)calloc(1, sizeof(*result));
    if (!result) ocean_tensor_fail("out of memory exporting Tensor");
    result->ndim = tensor->ndim;
    result->size = tensor->size;
    result->refcount = 1;
    result->is_view = false;
    result->shape = (size_t *)malloc(result->ndim * sizeof(size_t));
    result->strides = (size_t *)malloc(result->ndim * sizeof(size_t));
    size_t bytes = ocean_tensor_bytes(tensor);
    result->data = bytes ? malloc(bytes) : NULL;
    if (!result->shape || !result->strides || (bytes && !result->data)) {
        ocean_tensor_fail("out of memory exporting Tensor metadata");
    }
    memcpy(result->shape, tensor->shape, result->ndim * sizeof(size_t));
    memcpy(result->strides, tensor->strides, result->ndim * sizeof(size_t));
    if (tensor->device == OCEAN_TENSOR_CPU) {
        if (bytes) memcpy(result->data, tensor->cpu_data, bytes);
    }
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    else {
        ocean_tensor_gpu_read(tensor, result->data);
    }
#endif
    return result;
}

void ocean_tensor_export_free(void *value) {
    ocean_tensor_export_layout *tensor = (ocean_tensor_export_layout *)value;
    if (!tensor) return;
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

void ocean_tensor_release(ocean_tensor_handle_t tensor) {
    if (!tensor) return;
    free(tensor->cpu_data);
    free(tensor->shape);
    free(tensor->strides);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->gpu_data) clReleaseMemObject(tensor->gpu_data);
#endif
    free(tensor);
}
