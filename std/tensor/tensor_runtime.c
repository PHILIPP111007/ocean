#include "std/tensor/tensor_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
#include <CL/cl.h>
#endif

struct ocean_tensor_handle {
    int rows;
    int cols;
    int device;
    float *cpu_data;
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

static size_t ocean_tensor_elements(const ocean_tensor_handle_t tensor) {
    if (!tensor || tensor->rows < 0 || tensor->cols < 0) {
        ocean_tensor_fail("invalid tensor handle or shape");
    }
    return (size_t)tensor->rows * (size_t)tensor->cols;
}

static ocean_tensor_handle_t ocean_tensor_alloc(int rows, int cols, int device) {
    if (rows < 0 || cols < 0) ocean_tensor_fail("shape dimensions must be non-negative");
    ocean_tensor_handle_t tensor = (ocean_tensor_handle_t)calloc(1, sizeof(*tensor));
    if (!tensor) ocean_tensor_fail("out of memory allocating tensor handle");
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->device = device;
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
    "}";

typedef struct ocean_tensor_opencl_runtime {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_kernel;
} ocean_tensor_opencl_runtime;

static ocean_tensor_opencl_runtime ocean_tensor_opencl;
static int ocean_tensor_opencl_initialized = 0;

static void ocean_tensor_opencl_check(cl_int status, const char *operation) {
    if (status != CL_SUCCESS) {
        char message[256];
        snprintf(message, sizeof(message), "%s failed (OpenCL error %d)", operation, status);
        ocean_tensor_fail(message);
    }
}

static void ocean_tensor_opencl_init(void) {
    if (ocean_tensor_opencl_initialized) return;

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

    ocean_tensor_opencl.context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    ocean_tensor_opencl_check(status, "clCreateContext");
    ocean_tensor_opencl.queue = clCreateCommandQueue(
        ocean_tensor_opencl.context, device, 0, &status
    );
    ocean_tensor_opencl_check(status, "clCreateCommandQueue");

    const char *sources[] = {ocean_tensor_matmul_kernel_source};
    ocean_tensor_opencl.program = clCreateProgramWithSource(
        ocean_tensor_opencl.context, 1, sources, NULL, &status
    );
    ocean_tensor_opencl_check(status, "clCreateProgramWithSource");
    status = clBuildProgram(ocean_tensor_opencl.program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(
            ocean_tensor_opencl.program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size
        );
        char *log = (char *)calloc(log_size + 1, 1);
        if (log) {
            clGetProgramBuildInfo(
                ocean_tensor_opencl.program, device, CL_PROGRAM_BUILD_LOG,
                log_size, log, NULL
            );
            fprintf(stderr, "Ocean Tensor OpenCL build log:\\n%s\\n", log);
            free(log);
        }
        ocean_tensor_opencl_check(status, "clBuildProgram");
    }
    ocean_tensor_opencl.matmul_kernel = clCreateKernel(
        ocean_tensor_opencl.program, "ocean_tensor_matmul", &status
    );
    ocean_tensor_opencl_check(status, "clCreateKernel");
    ocean_tensor_opencl_initialized = 1;
}

static void ocean_tensor_opencl_release(void) {
    if (!ocean_tensor_opencl_initialized) return;
    clReleaseKernel(ocean_tensor_opencl.matmul_kernel);
    clReleaseProgram(ocean_tensor_opencl.program);
    clReleaseCommandQueue(ocean_tensor_opencl.queue);
    clReleaseContext(ocean_tensor_opencl.context);
    memset(&ocean_tensor_opencl, 0, sizeof(ocean_tensor_opencl));
    ocean_tensor_opencl_initialized = 0;
}

static void ocean_tensor_gpu_alloc(ocean_tensor_handle_t tensor) {
    ocean_tensor_opencl_init();
    cl_int status = CL_SUCCESS;
    tensor->gpu_data = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_WRITE,
        ocean_tensor_elements(tensor) * sizeof(float),
        NULL,
        &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
}

static void ocean_tensor_gpu_write(ocean_tensor_handle_t tensor, const float *data) {
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE, 0,
            ocean_tensor_elements(tensor) * sizeof(float), data, 0, NULL, NULL
        ),
        "clEnqueueWriteBuffer"
    );
}

static void ocean_tensor_gpu_read(ocean_tensor_handle_t tensor, float *data) {
    ocean_tensor_opencl_check(
        clEnqueueReadBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_TRUE, 0,
            ocean_tensor_elements(tensor) * sizeof(float), data, 0, NULL, NULL
        ),
        "clEnqueueReadBuffer"
    );
}
#endif

ocean_tensor_handle_t ocean_tensor_zeros(int rows, int cols, const char *device) {
    int device_kind = ocean_tensor_parse_device(device);
    ocean_tensor_handle_t tensor = ocean_tensor_alloc(rows, cols, device_kind);

    if (device_kind == OCEAN_TENSOR_CPU) {
        tensor->cpu_data = (float *)calloc(ocean_tensor_elements(tensor), sizeof(float));
        if (!tensor->cpu_data) ocean_tensor_fail("out of memory allocating CPU tensor");
        return tensor;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    ocean_tensor_gpu_alloc(tensor);
#else
    free(tensor);
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    return tensor;
}

ocean_tensor_handle_t ocean_tensor_copy(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("cannot copy a null tensor");
    ocean_tensor_handle_t result = ocean_tensor_alloc(tensor->rows, tensor->cols, tensor->device);
    size_t bytes = ocean_tensor_elements(tensor) * sizeof(float);

    if (tensor->device == OCEAN_TENSOR_CPU) {
        result->cpu_data = (float *)malloc(bytes);
        if (!result->cpu_data) ocean_tensor_fail("out of memory copying CPU tensor");
        memcpy(result->cpu_data, tensor->cpu_data, bytes);
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
    ocean_tensor_opencl_check(clFinish(ocean_tensor_opencl.queue), "clFinish");
    return result;
#else
    free(result);
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    return NULL;
}

ocean_tensor_handle_t ocean_tensor_to(ocean_tensor_handle_t tensor, const char *device) {
    if (!tensor) ocean_tensor_fail("cannot move a null tensor");
    int target = ocean_tensor_parse_device(device);
    if (target == tensor->device) return ocean_tensor_copy(tensor);

    ocean_tensor_handle_t result = ocean_tensor_alloc(tensor->rows, tensor->cols, target);

    if (target == OCEAN_TENSOR_CPU) {
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
        size_t bytes = ocean_tensor_elements(tensor) * sizeof(float);
        result->cpu_data = (float *)malloc(bytes);
        if (!result->cpu_data) ocean_tensor_fail("out of memory downloading tensor");
        ocean_tensor_gpu_read(tensor, result->cpu_data);
        return result;
#else
        free(result);
        ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    size_t bytes = ocean_tensor_elements(tensor) * sizeof(float);
    result->gpu_data = NULL;
    ocean_tensor_gpu_alloc(result);
    ocean_tensor_gpu_write(result, tensor->cpu_data);
    return result;
#else
    free(result);
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    return NULL;
}

ocean_tensor_handle_t ocean_tensor_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    if (!left || !right) ocean_tensor_fail("matmul does not accept null tensors");
    if (left->cols != right->rows) ocean_tensor_fail("matmul shape mismatch");
    if (left->device != right->device) ocean_tensor_fail("matmul requires tensors on the same device");

    ocean_tensor_handle_t result = ocean_tensor_zeros(
        left->rows, right->cols,
        left->device == OCEAN_TENSOR_GPU ? "gpu" : "cpu"
    );
    if (left->device == OCEAN_TENSOR_CPU) {
        for (int row = 0; row < left->rows; ++row) {
            for (int col = 0; col < right->cols; ++col) {
                float sum = 0.0f;
                for (int k = 0; k < left->cols; ++k) {
                    sum += left->cpu_data[row * left->cols + k]
                        * right->cpu_data[k * right->cols + col];
                }
                result->cpu_data[row * result->cols + col] = sum;
            }
        }
        return result;
    }

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    cl_kernel kernel = ocean_tensor_opencl.matmul_kernel;
    ocean_tensor_opencl_check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &left->gpu_data), "clSetKernelArg");
    ocean_tensor_opencl_check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &right->gpu_data), "clSetKernelArg");
    ocean_tensor_opencl_check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &result->gpu_data), "clSetKernelArg");
    ocean_tensor_opencl_check(clSetKernelArg(kernel, 3, sizeof(int), &left->rows), "clSetKernelArg");
    ocean_tensor_opencl_check(clSetKernelArg(kernel, 4, sizeof(int), &left->cols), "clSetKernelArg");
    ocean_tensor_opencl_check(clSetKernelArg(kernel, 5, sizeof(int), &right->cols), "clSetKernelArg");

    size_t global_size[2] = {
        ((size_t)left->rows + 7u) / 8u * 8u,
        ((size_t)right->cols + 7u) / 8u * 8u,
    };
    size_t local_size[2] = {8u, 8u};
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 2, NULL,
            global_size, local_size, 0, NULL, NULL
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(clFinish(ocean_tensor_opencl.queue), "clFinish");
    return result;
#else
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
#endif
    return NULL;
}

int ocean_tensor_shape(ocean_tensor_handle_t tensor, int axis) {
    if (!tensor) ocean_tensor_fail("shape() does not accept a null tensor");
    if (axis == 0) return tensor->rows;
    if (axis == 1) return tensor->cols;
    ocean_tensor_fail("shape axis must be 0 or 1");
    return 0;
}

int ocean_tensor_ndim(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("ndim() does not accept a null tensor");
    return 2;
}

size_t ocean_tensor_size(ocean_tensor_handle_t tensor) {
    return ocean_tensor_elements(tensor);
}

char *ocean_tensor_device(ocean_tensor_handle_t tensor) {
    if (!tensor) ocean_tensor_fail("device() does not accept a null tensor");
    const char *name = tensor->device == OCEAN_TENSOR_GPU ? "gpu" : "cpu";
    char *result = (char *)malloc(strlen(name) + 1);
    if (!result) ocean_tensor_fail("out of memory copying device name");
    strcpy(result, name);
    return result;
}

void ocean_tensor_release(ocean_tensor_handle_t tensor) {
    if (!tensor) return;
    free(tensor->cpu_data);
#ifdef OCEAN_TENSOR_ENABLE_OPENCL
    if (tensor->gpu_data) clReleaseMemObject(tensor->gpu_data);
#endif
    free(tensor);
}
