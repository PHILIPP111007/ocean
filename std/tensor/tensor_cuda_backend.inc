/* NVIDIA/CUDA backend ABI placeholder.
 *
 * The runtime deliberately keeps this backend in the dispatch table before
 * native CUDA kernels are implemented.  A request for CUDA therefore fails
 * explicitly instead of silently executing on the CPU or OpenCL.
 */
static void ocean_tensor_cuda_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
}

static void ocean_tensor_cuda_copy_unavailable(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    (void)destination;
    (void)source;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
}

static void ocean_tensor_cuda_read_unavailable(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
}

static void ocean_tensor_cuda_write_unavailable(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
}

static void ocean_tensor_cuda_release_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
}

static ocean_tensor_handle_t ocean_tensor_cuda_matmul_unavailable(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    (void)left;
    (void)right;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
    return NULL;
}

static ocean_tensor_handle_t ocean_tensor_cuda_binary_unavailable(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
    (void)left;
    (void)right;
    (void)operation;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
    return NULL;
}

static ocean_tensor_handle_t ocean_tensor_cuda_scalar_unavailable(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    (void)tensor;
    (void)scalar;
    (void)operation;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
    return NULL;
}

static void ocean_tensor_cuda_fill_unavailable(
    ocean_tensor_handle_t tensor,
    double value
) {
    (void)tensor;
    (void)value;
    ocean_tensor_fail(
        "CUDA backend is not implemented; use OpenCL or build the native CUDA backend"
    );
}

static const ocean_tensor_backend_ops ocean_tensor_cuda_backend = {
    .kind = OCEAN_TENSOR_BACKEND_CUDA,
    .name = "cuda",
    .compiled = false,
    .allocate = ocean_tensor_cuda_unavailable,
    .zero = ocean_tensor_cuda_unavailable,
    .copy = ocean_tensor_cuda_copy_unavailable,
    .read = ocean_tensor_cuda_read_unavailable,
    .write = ocean_tensor_cuda_write_unavailable,
    .release = ocean_tensor_cuda_release_unavailable,
    .matmul = ocean_tensor_cuda_matmul_unavailable,
    .binary = ocean_tensor_cuda_binary_unavailable,
    .scalar = ocean_tensor_cuda_scalar_unavailable,
    .fill = ocean_tensor_cuda_fill_unavailable,
};
