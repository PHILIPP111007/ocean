/* OpenCL context, program and kernel-dispatch runtime internals. */
typedef struct ocean_tensor_opencl_runtime {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_kernel;
    cl_kernel matvec_kernel;
    cl_kernel matvec_bias_kernel;
    cl_kernel matmul_int32_kernel;
    cl_kernel batched_matmul_kernel;
    cl_kernel permute_kernel;
    cl_kernel binary_kernel;
    cl_kernel binary_int32_kernel;
    cl_kernel scalar_kernel;
    cl_kernel scalar_int32_kernel;
    cl_kernel softmax_kernel;
    cl_kernel layer_norm_kernel;
    cl_kernel reduce_kernel;
    cl_kernel sgd_update_kernel;
    cl_kernel adamw_update_kernel;
    cl_kernel ternary_quantize_kernel;
    cl_kernel gelu_kernel;
    cl_kernel gelu_backward_kernel;
    cl_kernel embedding_forward_kernel;
    cl_kernel embedding_backward_kernel;
    cl_kernel cross_entropy_forward_kernel;
    cl_kernel cross_entropy_backward_kernel;
    cl_kernel softmax_backward_kernel;
    cl_kernel layer_norm_backward_kernel;
    cl_kernel cache_write_kernel;
    cl_kernel cache_slice_kernel;
    cl_kernel argmax_kernel;
    cl_kernel causal_softmax_kernel;
    cl_kernel ternary_pack_kernel;
    cl_kernel packed_matmul_kernel;
    cl_kernel packed_linear_kernel;
    cl_kernel packed_qkv_kernel;
    cl_kernel slice_last_dim_kernel;
    cl_kernel packed_qkv_split_kernel;
    cl_kernel packed_qkv_attention_kernel;
    cl_mem argmax_result;
} ocean_tensor_opencl_runtime;

typedef enum ocean_tensor_opencl_kernel_key {
    OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_MATVEC_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_MATVEC_BIAS_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_INT32,
    OCEAN_TENSOR_OPENCL_KERNEL_BATCHED_MATMUL_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PERMUTE,
    OCEAN_TENSOR_OPENCL_KERNEL_BINARY_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_BINARY_INT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_INT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SGD_UPDATE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_ADAMW_UPDATE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_QUANTIZE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_GELU_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_FORWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_BACKWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_FORWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_BACKWARD_FLOAT32_INT64,
    OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_BACKWARD_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_BACKWARD_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_CACHE_WRITE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_CACHE_SLICE_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_ARGMAX_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_CAUSAL_SOFTMAX_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_PACK_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PACKED_MATMUL_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PACKED_LINEAR_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_SLICE_LAST_DIM_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_SPLIT_FLOAT32,
    OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_ATTENTION_FLOAT32,
} ocean_tensor_opencl_kernel_key;

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
    if (ocean_tensor_opencl.matvec_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matvec_kernel);
        ocean_tensor_opencl.matvec_kernel = NULL;
    }
    if (ocean_tensor_opencl.matvec_bias_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matvec_bias_kernel);
        ocean_tensor_opencl.matvec_bias_kernel = NULL;
    }
    if (ocean_tensor_opencl.matmul_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.matmul_int32_kernel);
        ocean_tensor_opencl.matmul_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.batched_matmul_kernel) {
        clReleaseKernel(ocean_tensor_opencl.batched_matmul_kernel);
        ocean_tensor_opencl.batched_matmul_kernel = NULL;
    }
    if (ocean_tensor_opencl.permute_kernel) {
        clReleaseKernel(ocean_tensor_opencl.permute_kernel);
        ocean_tensor_opencl.permute_kernel = NULL;
    }
    if (ocean_tensor_opencl.scalar_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.scalar_int32_kernel);
        ocean_tensor_opencl.scalar_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.binary_int32_kernel) {
        clReleaseKernel(ocean_tensor_opencl.binary_int32_kernel);
        ocean_tensor_opencl.binary_int32_kernel = NULL;
    }
    if (ocean_tensor_opencl.softmax_kernel) {
        clReleaseKernel(ocean_tensor_opencl.softmax_kernel);
        ocean_tensor_opencl.softmax_kernel = NULL;
    }
    if (ocean_tensor_opencl.layer_norm_kernel) {
        clReleaseKernel(ocean_tensor_opencl.layer_norm_kernel);
        ocean_tensor_opencl.layer_norm_kernel = NULL;
    }
    if (ocean_tensor_opencl.reduce_kernel) {
        clReleaseKernel(ocean_tensor_opencl.reduce_kernel);
        ocean_tensor_opencl.reduce_kernel = NULL;
    }
    if (ocean_tensor_opencl.sgd_update_kernel) {
        clReleaseKernel(ocean_tensor_opencl.sgd_update_kernel);
        ocean_tensor_opencl.sgd_update_kernel = NULL;
    }
    if (ocean_tensor_opencl.adamw_update_kernel) {
        clReleaseKernel(ocean_tensor_opencl.adamw_update_kernel);
        ocean_tensor_opencl.adamw_update_kernel = NULL;
    }
    if (ocean_tensor_opencl.ternary_quantize_kernel) {
        clReleaseKernel(ocean_tensor_opencl.ternary_quantize_kernel);
        ocean_tensor_opencl.ternary_quantize_kernel = NULL;
    }
    if (ocean_tensor_opencl.embedding_forward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.embedding_forward_kernel);
        ocean_tensor_opencl.embedding_forward_kernel = NULL;
    }
    if (ocean_tensor_opencl.embedding_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.embedding_backward_kernel);
        ocean_tensor_opencl.embedding_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.cross_entropy_forward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cross_entropy_forward_kernel);
        ocean_tensor_opencl.cross_entropy_forward_kernel = NULL;
    }
    if (ocean_tensor_opencl.cross_entropy_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cross_entropy_backward_kernel);
        ocean_tensor_opencl.cross_entropy_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.softmax_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.softmax_backward_kernel);
        ocean_tensor_opencl.softmax_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.layer_norm_backward_kernel) {
        clReleaseKernel(ocean_tensor_opencl.layer_norm_backward_kernel);
        ocean_tensor_opencl.layer_norm_backward_kernel = NULL;
    }
    if (ocean_tensor_opencl.cache_write_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cache_write_kernel);
        ocean_tensor_opencl.cache_write_kernel = NULL;
    }
    if (ocean_tensor_opencl.cache_slice_kernel) {
        clReleaseKernel(ocean_tensor_opencl.cache_slice_kernel);
        ocean_tensor_opencl.cache_slice_kernel = NULL;
    }
    if (ocean_tensor_opencl.argmax_kernel) {
        clReleaseKernel(ocean_tensor_opencl.argmax_kernel);
        ocean_tensor_opencl.argmax_kernel = NULL;
    }
    if (ocean_tensor_opencl.causal_softmax_kernel) {
        clReleaseKernel(ocean_tensor_opencl.causal_softmax_kernel);
        ocean_tensor_opencl.causal_softmax_kernel = NULL;
    }
    if (ocean_tensor_opencl.ternary_pack_kernel) {
        clReleaseKernel(ocean_tensor_opencl.ternary_pack_kernel);
        ocean_tensor_opencl.ternary_pack_kernel = NULL;
    }
    if (ocean_tensor_opencl.packed_matmul_kernel) {
        clReleaseKernel(ocean_tensor_opencl.packed_matmul_kernel);
        ocean_tensor_opencl.packed_matmul_kernel = NULL;
    }
    if (ocean_tensor_opencl.packed_linear_kernel) {
        clReleaseKernel(ocean_tensor_opencl.packed_linear_kernel);
        ocean_tensor_opencl.packed_linear_kernel = NULL;
    }
    if (ocean_tensor_opencl.packed_qkv_kernel) {
        clReleaseKernel(ocean_tensor_opencl.packed_qkv_kernel);
        ocean_tensor_opencl.packed_qkv_kernel = NULL;
    }
    if (ocean_tensor_opencl.slice_last_dim_kernel) {
        clReleaseKernel(ocean_tensor_opencl.slice_last_dim_kernel);
        ocean_tensor_opencl.slice_last_dim_kernel = NULL;
    }
    if (ocean_tensor_opencl.packed_qkv_split_kernel) {
        clReleaseKernel(ocean_tensor_opencl.packed_qkv_split_kernel);
        ocean_tensor_opencl.packed_qkv_split_kernel = NULL;
    }
    if (ocean_tensor_opencl.packed_qkv_attention_kernel) {
        clReleaseKernel(ocean_tensor_opencl.packed_qkv_attention_kernel);
        ocean_tensor_opencl.packed_qkv_attention_kernel = NULL;
    }
    if (ocean_tensor_opencl.argmax_result) {
        clReleaseMemObject(ocean_tensor_opencl.argmax_result);
        ocean_tensor_opencl.argmax_result = NULL;
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
    ocean_tensor_opencl.device = NULL;
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

static void ocean_tensor_opencl_wait_event(
    cl_event event,
    const char *operation
) {
    if (!event) return;
    ocean_tensor_opencl_check(clWaitForEvents(1, &event), operation);
    ocean_tensor_opencl_check(clReleaseEvent(event), "clReleaseEvent");
}

static void ocean_tensor_opencl_release_event(cl_event event) {
    if (!event) return;
    ocean_tensor_opencl_check(clReleaseEvent(event), "clReleaseEvent");
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

    cl_platform_id *platforms = (cl_platform_id *)calloc(
        platform_count, sizeof(*platforms)
    );
    if (!platforms) ocean_tensor_fail("out of memory enumerating OpenCL platforms");
    ocean_tensor_opencl_check(
        clGetPlatformIDs(platform_count, platforms, NULL), "clGetPlatformIDs"
    );

    /* A Tensor selected as "gpu" must never silently bind to a CPU OpenCL
       device through CL_DEVICE_TYPE_DEFAULT. Search every platform and fail
       explicitly when no real GPU device is exposed by the ICD. */
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    for (cl_uint index = 0; index < platform_count; ++index) {
        cl_device_id candidate = NULL;
        cl_int device_status = clGetDeviceIDs(
            platforms[index], CL_DEVICE_TYPE_GPU, 1, &candidate, NULL
        );
        if (device_status == CL_SUCCESS) {
            platform = platforms[index];
            device = candidate;
            break;
        }
    }
    free(platforms);
    if (!device || !platform) {
        ocean_tensor_fail(
            "no OpenCL GPU device is available; refusing CPU fallback for device= gpu"
        );
    }

    cl_int status = CL_SUCCESS;
    ocean_tensor_opencl.device = device;

    ocean_tensor_opencl.context =
        clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    ocean_tensor_opencl_check(status, "clCreateContext");
#ifdef CL_VERSION_2_0
    const cl_queue_properties queue_properties[] = {0};
    ocean_tensor_opencl.queue = clCreateCommandQueueWithProperties(
        ocean_tensor_opencl.context, device, queue_properties, &status
    );
#else
    ocean_tensor_opencl.queue =
        clCreateCommandQueue(ocean_tensor_opencl.context, device, 0, &status);
#endif
    ocean_tensor_opencl_check(status, "clCreateCommandQueue");

    const char *sources[] = {
        ocean_tensor_matmul_kernel_source,
        ocean_tensor_matvec_kernel_source,
        ocean_tensor_batched_matmul_kernel_source,
        ocean_tensor_permute_kernel_source,
        ocean_tensor_hotpath_kernel_source,
        ocean_tensor_gelu_kernel_source,
        ocean_tensor_embedding_kernel_source,
        ocean_tensor_cross_entropy_kernel_source,
        ocean_tensor_backward_kernel_source,
        ocean_tensor_cache_kernel_source,
        ocean_tensor_argmax_kernel_source,
        ocean_tensor_causal_softmax_kernel_source,
        ocean_tensor_packed_ternary_kernel_source,
        ocean_tensor_packed_qkv_kernel_source,
        ocean_tensor_slice_last_dim_kernel_source,
        ocean_tensor_packed_qkv_split_kernel_source,
        ocean_tensor_packed_qkv_attention_kernel_source,
    };
    ocean_tensor_opencl.program = clCreateProgramWithSource(
        ocean_tensor_opencl.context, 17, sources, NULL, &status
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
}

static cl_kernel ocean_tensor_opencl_get_kernel(
    ocean_tensor_opencl_kernel_key key
) {
    ocean_tensor_opencl_init();
    cl_kernel *slot = NULL;
    const char *name = NULL;
    switch (key) {
        case OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_FLOAT32:
            slot = &ocean_tensor_opencl.matmul_kernel;
            name = "ocean_tensor_matmul";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_MATVEC_FLOAT32:
            slot = &ocean_tensor_opencl.matvec_kernel;
            name = "ocean_tensor_matvec";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_MATVEC_BIAS_FLOAT32:
            slot = &ocean_tensor_opencl.matvec_bias_kernel;
            name = "ocean_tensor_matvec_bias";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_MATMUL_INT32:
            slot = &ocean_tensor_opencl.matmul_int32_kernel;
            name = "ocean_tensor_matmul_int32";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_BATCHED_MATMUL_FLOAT32:
            slot = &ocean_tensor_opencl.batched_matmul_kernel;
            name = "ocean_tensor_batched_matmul";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PERMUTE:
            slot = &ocean_tensor_opencl.permute_kernel;
            name = "ocean_tensor_permute";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_BINARY_FLOAT32:
            slot = &ocean_tensor_opencl.binary_kernel;
            name = "ocean_tensor_binary";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_BINARY_INT32:
            slot = &ocean_tensor_opencl.binary_int32_kernel;
            name = "ocean_tensor_binary_int32";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_FLOAT32:
            slot = &ocean_tensor_opencl.scalar_kernel;
            name = "ocean_tensor_scalar";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SCALAR_INT32:
            slot = &ocean_tensor_opencl.scalar_int32_kernel;
            name = "ocean_tensor_scalar_int32";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_FLOAT32:
            slot = &ocean_tensor_opencl.softmax_kernel;
            name = "ocean_tensor_softmax_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32:
            slot = &ocean_tensor_opencl.layer_norm_kernel;
            name = "ocean_tensor_layer_norm_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32:
            slot = &ocean_tensor_opencl.reduce_kernel;
            name = "ocean_tensor_reduce_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SGD_UPDATE_FLOAT32:
            slot = &ocean_tensor_opencl.sgd_update_kernel;
            name = "ocean_tensor_sgd_update";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_ADAMW_UPDATE_FLOAT32:
            slot = &ocean_tensor_opencl.adamw_update_kernel;
            name = "ocean_tensor_adamw_update";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_QUANTIZE_FLOAT32:
            slot = &ocean_tensor_opencl.ternary_quantize_kernel;
            name = "ocean_tensor_ternary_quantize";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_GELU_FLOAT32:
            slot = &ocean_tensor_opencl.gelu_kernel;
            name = "ocean_tensor_gelu";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32:
            slot = &ocean_tensor_opencl.gelu_backward_kernel;
            name = "ocean_tensor_gelu_backward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_FORWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.embedding_forward_kernel;
            name = "ocean_tensor_embedding_forward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_BACKWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.embedding_backward_kernel;
            name = "ocean_tensor_embedding_backward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_FORWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.cross_entropy_forward_kernel;
            name = "ocean_tensor_cross_entropy_forward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_BACKWARD_FLOAT32_INT64:
            slot = &ocean_tensor_opencl.cross_entropy_backward_kernel;
            name = "ocean_tensor_cross_entropy_backward";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_BACKWARD_FLOAT32:
            slot = &ocean_tensor_opencl.softmax_backward_kernel;
            name = "ocean_tensor_softmax_backward_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_BACKWARD_FLOAT32:
            slot = &ocean_tensor_opencl.layer_norm_backward_kernel;
            name = "ocean_tensor_layer_norm_backward_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CACHE_WRITE_FLOAT32:
            slot = &ocean_tensor_opencl.cache_write_kernel;
            name = "ocean_tensor_cache_write";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CACHE_SLICE_FLOAT32:
            slot = &ocean_tensor_opencl.cache_slice_kernel;
            name = "ocean_tensor_cache_slice";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_ARGMAX_FLOAT32:
            slot = &ocean_tensor_opencl.argmax_kernel;
            name = "ocean_tensor_argmax";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_CAUSAL_SOFTMAX_FLOAT32:
            slot = &ocean_tensor_opencl.causal_softmax_kernel;
            name = "ocean_tensor_causal_softmax";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_PACK_FLOAT32:
            slot = &ocean_tensor_opencl.ternary_pack_kernel;
            name = "ocean_tensor_ternary_pack";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PACKED_MATMUL_FLOAT32:
            slot = &ocean_tensor_opencl.packed_matmul_kernel;
            name = "ocean_tensor_packed_matmul";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PACKED_LINEAR_FLOAT32:
            slot = &ocean_tensor_opencl.packed_linear_kernel;
            name = "ocean_tensor_packed_linear";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_FLOAT32:
            slot = &ocean_tensor_opencl.packed_qkv_kernel;
            name = "ocean_tensor_packed_qkv";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_SLICE_LAST_DIM_FLOAT32:
            slot = &ocean_tensor_opencl.slice_last_dim_kernel;
            name = "ocean_tensor_slice_last_dim";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_SPLIT_FLOAT32:
            slot = &ocean_tensor_opencl.packed_qkv_split_kernel;
            name = "ocean_tensor_packed_qkv_split";
            break;
        case OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_ATTENTION_FLOAT32:
            slot = &ocean_tensor_opencl.packed_qkv_attention_kernel;
            name = "ocean_tensor_packed_qkv_attention_decode";
            break;
    }
    if (!slot || !name) ocean_tensor_fail("invalid OpenCL Tensor kernel key");
    if (!*slot) {
        cl_int status = CL_SUCCESS;
        *slot = clCreateKernel(ocean_tensor_opencl.program, name, &status);
        ocean_tensor_opencl_check(status, name);
    }
    return *slot;
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
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueWriteBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_FALSE,
            0, bytes, data, 0, NULL, &event
        ),
        "clEnqueueWriteBuffer"
    );
    ocean_tensor_opencl_wait_event(event, "clWaitForEvents(write)");
}

static void ocean_tensor_gpu_zero(
    ocean_tensor_handle_t tensor
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;

#if defined(CL_VERSION_1_2)
    const unsigned char zero = 0;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueFillBuffer(
            ocean_tensor_opencl.queue,
            tensor->gpu_data,
            &zero,
            sizeof(zero),
            0,
            bytes,
            0,
            NULL,
            &event
        ),
        "clEnqueueFillBuffer"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue),
        "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
#else
    void *zeros = calloc(1, bytes);
    if (!zeros) ocean_tensor_fail("out of memory zeroing GPU Tensor");
    ocean_tensor_gpu_write(tensor, zeros);
    free(zeros);
#endif
}


static void ocean_tensor_gpu_read(
    ocean_tensor_handle_t tensor,
    void *data
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (!bytes) return;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueReadBuffer(
            ocean_tensor_opencl.queue, tensor->gpu_data, CL_FALSE,
            0, bytes, data, 0, NULL, &event
        ),
        "clEnqueueReadBuffer"
    );
    ocean_tensor_opencl_wait_event(event, "clWaitForEvents(read)");
}

