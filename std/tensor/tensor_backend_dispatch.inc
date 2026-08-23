/* Backend registry and device selection. */
static const ocean_tensor_backend_ops ocean_tensor_cpu_backend = {
    .kind = OCEAN_TENSOR_BACKEND_CPU,
    .name = "cpu",
    .compiled = true,
    .allocate = ocean_tensor_cpu_allocate,
    .zero = ocean_tensor_cpu_zero,
    .copy = ocean_tensor_cpu_copy,
    .read = ocean_tensor_cpu_read,
    .write = ocean_tensor_cpu_write,
    .release = ocean_tensor_cpu_release,
    .matmul = ocean_tensor_matmul_cpu,
    .binary = ocean_tensor_binary_cpu,
    .scalar = ocean_tensor_scalar_cpu,
    .fill = ocean_tensor_fill_cpu,
};

#ifdef OCEAN_TENSOR_ENABLE_OPENCL
static const ocean_tensor_backend_ops ocean_tensor_opencl_backend = {
    .kind = OCEAN_TENSOR_BACKEND_OPENCL,
    .name = "opencl",
    .compiled = true,
    .allocate = ocean_tensor_gpu_alloc,
    .zero = ocean_tensor_gpu_zero,
    .copy = ocean_tensor_opencl_copy,
    .read = ocean_tensor_gpu_read,
    .write = ocean_tensor_gpu_write,
    .release = ocean_tensor_opencl_release,
    .matmul = ocean_tensor_matmul_opencl,
    .binary = ocean_tensor_binary_opencl,
    .scalar = ocean_tensor_scalar_opencl,
    .fill = ocean_tensor_fill_opencl,
};
#else
static const ocean_tensor_backend_ops ocean_tensor_opencl_backend = {
    .kind = OCEAN_TENSOR_BACKEND_OPENCL,
    .name = "opencl",
    .compiled = false,
    .allocate = ocean_tensor_gpu_unavailable,
    .zero = ocean_tensor_gpu_unavailable,
    .copy = ocean_tensor_gpu_copy_unavailable,
    .read = ocean_tensor_gpu_read_unavailable,
    .write = ocean_tensor_gpu_write_unavailable,
    .release = ocean_tensor_gpu_release_unavailable,
    .matmul = ocean_tensor_matmul_opencl,
    .binary = ocean_tensor_binary_opencl,
    .scalar = ocean_tensor_scalar_opencl,
    .fill = ocean_tensor_fill_opencl,
};
#endif

#include "std/tensor/tensor_cuda_backend.inc"

static ocean_tensor_backend_kind ocean_tensor_select_gpu_backend(void) {
    const char *requested = getenv("OCEAN_TENSOR_GPU_BACKEND");
    if (!requested || strcmp(requested, "") == 0 ||
        strcmp(requested, "auto") == 0) {
        if (ocean_tensor_opencl_backend.compiled) {
            return OCEAN_TENSOR_BACKEND_OPENCL;
        }
        ocean_tensor_fail(
            "no compiled GPU backend; rebuild with OpenCL or native CUDA support"
        );
    }
    if (strcmp(requested, "opencl") == 0) {
        if (!ocean_tensor_opencl_backend.compiled) {
            ocean_tensor_fail(
                "OpenCL backend was requested but this build has no OpenCL support"
            );
        }
        return OCEAN_TENSOR_BACKEND_OPENCL;
    }
    if (strcmp(requested, "cuda") == 0 || strcmp(requested, "nvidia") == 0) {
        if (!ocean_tensor_cuda_backend.compiled) {
            ocean_tensor_fail(
                "CUDA backend was requested but native NVIDIA support is not built"
            );
        }
        return OCEAN_TENSOR_BACKEND_CUDA;
    }
    ocean_tensor_fail(
        "OCEAN_TENSOR_GPU_BACKEND must be auto, opencl, cuda, or nvidia"
    );
    return OCEAN_TENSOR_BACKEND_OPENCL;
}

static const ocean_tensor_backend_ops *ocean_tensor_backend_for_device(
    ocean_tensor_backend_kind device
) {
    if (device == OCEAN_TENSOR_BACKEND_CPU) return &ocean_tensor_cpu_backend;
    if (device == OCEAN_TENSOR_BACKEND_OPENCL) return &ocean_tensor_opencl_backend;
    if (device == OCEAN_TENSOR_BACKEND_CUDA) return &ocean_tensor_cuda_backend;
    ocean_tensor_fail("invalid Tensor backend device");
    return &ocean_tensor_cpu_backend;
}
