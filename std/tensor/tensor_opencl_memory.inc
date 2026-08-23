/* OpenCL device memory operations and no-OpenCL diagnostics. */
}

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static void ocean_tensor_opencl_copy(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    size_t bytes = ocean_tensor_bytes(source);
    if (!bytes) return;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueCopyBuffer(
            ocean_tensor_opencl.queue, source->gpu_data, destination->gpu_data,
            0, 0, bytes, 0, NULL, &event
        ),
        "clEnqueueCopyBuffer"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_release(ocean_tensor_handle_t tensor) {
    if (tensor->gpu_data) {
        clReleaseMemObject(tensor->gpu_data);
        tensor->gpu_data = NULL;
    }
}
#else
static void ocean_tensor_gpu_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_copy_unavailable(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    (void)destination;
    (void)source;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_read_unavailable(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_write_unavailable(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail("GPU backend is unavailable: rebuild with OpenCL support");
}

static void ocean_tensor_gpu_release_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
}
#endif

/* Backend registry, OpenCL implementation and CUDA extension point. */
#include "std/tensor/tensor_backend_dispatch.inc"

ocean_tensor_handle_t ocean_tensor_zeros_nd(
