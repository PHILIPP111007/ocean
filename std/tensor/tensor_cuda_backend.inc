/* CUDA backend operations.  The kernels themselves live in tensor_cuda_backend.cu. */
#ifdef OCEAN_TENSOR_ENABLE_CUDA

static void ocean_tensor_cuda_allocate(ocean_tensor_handle_t tensor) {
    size_t bytes = ocean_tensor_bytes(tensor);
    tensor->cuda_data = bytes == 0 ? NULL : ocean_cuda_malloc(bytes);
}

static void ocean_tensor_cuda_zero(ocean_tensor_handle_t tensor) {
    ocean_cuda_zero(tensor->cuda_data, ocean_tensor_bytes(tensor));
}

static void ocean_tensor_cuda_copy(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    ocean_cuda_memcpy_d2d(
        destination->cuda_data,
        source->cuda_data,
        ocean_tensor_bytes(source)
    );
}

static void ocean_tensor_cuda_read(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    ocean_cuda_memcpy_d2h(
        host_data,
        tensor->cuda_data,
        ocean_tensor_bytes(tensor)
    );
}

static void ocean_tensor_cuda_write(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    ocean_cuda_memcpy_h2d(
        tensor->cuda_data,
        host_data,
        ocean_tensor_bytes(tensor)
    );
}

static void ocean_tensor_cuda_release(ocean_tensor_handle_t tensor) {
    ocean_cuda_free(tensor->cuda_data);
    tensor->cuda_data = NULL;
}

static bool ocean_tensor_cuda_same_shape(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right
) {
    if (left->ndim != right->ndim) return false;
    for (size_t axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return false;
    }
    return true;
}

static void ocean_tensor_cuda_require_contiguous(
    const ocean_tensor_handle_t tensor,
    const char *operation
) {
    if (!ocean_tensor_is_contiguous(tensor)) {
        ocean_tensor_fail(operation);
    }
}

static bool ocean_tensor_cuda_contains_zero(
    const ocean_tensor_handle_t tensor
) {
    size_t bytes = ocean_tensor_bytes(tensor);
    if (bytes == 0) return false;
    void *host_data = malloc(bytes);
    if (!host_data) ocean_tensor_fail("out of memory checking CUDA Tensor");
    ocean_cuda_memcpy_d2h(host_data, tensor->cuda_data, bytes);
    bool found = false;
    if (tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        const float *values = (const float *)host_data;
        for (size_t index = 0; index < tensor->size; ++index) {
            if (values[index] == 0.0f) {
                found = true;
                break;
            }
        }
    } else if (tensor->dtype == OCEAN_TENSOR_INT32) {
        const int32_t *values = (const int32_t *)host_data;
        for (size_t index = 0; index < tensor->size; ++index) {
            if (values[index] == 0) {
                found = true;
                break;
            }
        }
    }
    free(host_data);
    return found;
}

static ocean_tensor_handle_t ocean_tensor_cuda_matmul(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right
) {
    if (left->ndim != 2 || right->ndim != 2) {
        ocean_tensor_fail(
            "CUDA matmul currently supports contiguous 2D Tensors"
        );
    }
    if (left->shape[1] != right->shape[0]) {
        ocean_tensor_fail("CUDA matmul shape mismatch");
    }
    ocean_tensor_cuda_require_contiguous(
        left, "CUDA matmul requires contiguous left Tensor"
    );
    ocean_tensor_cuda_require_contiguous(
        right, "CUDA matmul requires contiguous right Tensor"
    );
    if (left->shape[0] > (size_t)INT_MAX ||
        left->shape[1] > (size_t)INT_MAX ||
        right->shape[1] > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA matmul dimensions exceed int32 indexing");
    }

    size_t shape[2] = {left->shape[0], right->shape[1]};
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        shape, 2, left->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    if (result->size == 0) return result;

    int rows = (int)left->shape[0];
    int inner = (int)left->shape[1];
    int columns = (int)right->shape[1];
    if (left->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_cuda_matmul_f32(
            left->cuda_data, right->cuda_data, result->cuda_data,
            rows, inner, columns
        );
    } else if (left->dtype == OCEAN_TENSOR_INT32) {
        ocean_cuda_matmul_i32(
            left->cuda_data, right->cuda_data, result->cuda_data,
            rows, inner, columns
        );
    } else {
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "CUDA matmul currently supports float32 and int32 Tensors"
        );
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cuda_binary(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    int operation
) {
    if (!ocean_tensor_cuda_same_shape(left, right)) {
        ocean_tensor_fail(
            "CUDA binary operations currently require matching shapes"
        );
    }
    ocean_tensor_cuda_require_contiguous(
        left, "CUDA binary operation requires contiguous left Tensor"
    );
    ocean_tensor_cuda_require_contiguous(
        right, "CUDA binary operation requires contiguous right Tensor"
    );
    if (operation == OCEAN_TENSOR_DIV &&
        ocean_tensor_cuda_contains_zero(right)) {
        ocean_tensor_fail("Tensor division by zero");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        left->shape, left->ndim, left->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    if (left->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_cuda_binary_f32(
            left->cuda_data, right->cuda_data, result->cuda_data,
            left->size, operation
        );
    } else if (left->dtype == OCEAN_TENSOR_INT32) {
        ocean_cuda_binary_i32(
            left->cuda_data, right->cuda_data, result->cuda_data,
            left->size, operation
        );
    } else {
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "CUDA binary operations currently support float32 and int32 Tensors"
        );
    }
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cuda_scalar(
    ocean_tensor_handle_t tensor,
    double scalar,
    int operation
) {
    if (operation == OCEAN_TENSOR_DIV && scalar == 0.0) {
        ocean_tensor_fail("Tensor division by zero");
    }
    ocean_tensor_cuda_require_contiguous(
        tensor, "CUDA scalar operation requires a contiguous Tensor"
    );
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype,
        OCEAN_TENSOR_BACKEND_CUDA
    );
    if (tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_cuda_scalar_f32(
            tensor->cuda_data, result->cuda_data, tensor->size,
            (float)scalar, operation
        );
    } else if (tensor->dtype == OCEAN_TENSOR_INT32) {
        ocean_cuda_scalar_i32(
            tensor->cuda_data, result->cuda_data, tensor->size,
            (int)scalar, operation
        );
    } else {
        ocean_tensor_release(result);
        ocean_tensor_fail(
            "CUDA scalar operations currently support float32 and int32 Tensors"
        );
    }
    return result;
}

static void ocean_tensor_cuda_fill(
    ocean_tensor_handle_t tensor,
    double value
) {
    if (tensor->size == 0) return;
    if (value == 0.0) {
        ocean_cuda_zero(tensor->cuda_data, ocean_tensor_bytes(tensor));
    } else if (tensor->dtype == OCEAN_TENSOR_FLOAT32) {
        ocean_cuda_fill_f32(tensor->cuda_data, (float)value, tensor->size);
    } else if (tensor->dtype == OCEAN_TENSOR_INT32) {
        ocean_cuda_fill_i32(tensor->cuda_data, (int)value, tensor->size);
    } else {
        ocean_tensor_fail(
            "CUDA fill currently supports zero for all dtypes and non-zero "
            "values for float32/int32"
        );
    }
}

static const ocean_tensor_backend_ops ocean_tensor_cuda_backend = {
    .kind = OCEAN_TENSOR_BACKEND_CUDA,
    .name = "cuda",
    .compiled = true,
    .allocate = ocean_tensor_cuda_allocate,
    .zero = ocean_tensor_cuda_zero,
    .copy = ocean_tensor_cuda_copy,
    .read = ocean_tensor_cuda_read,
    .write = ocean_tensor_cuda_write,
    .release = ocean_tensor_cuda_release,
    .matmul = ocean_tensor_cuda_matmul,
    .binary = ocean_tensor_cuda_binary,
    .scalar = ocean_tensor_cuda_scalar,
    .fill = ocean_tensor_cuda_fill,
};

#else

/* Keep a real dispatch entry even in non-CUDA builds so an explicit request
   produces a diagnostic instead of an unresolved symbol or CPU fallback. */
static void ocean_tensor_cuda_unavailable(ocean_tensor_handle_t tensor) {
    (void)tensor;
    ocean_tensor_fail(
        "CUDA backend was requested but this build has no CUDA support"
    );
}

static void ocean_tensor_cuda_copy_unavailable(
    ocean_tensor_handle_t destination,
    const ocean_tensor_handle_t source
) {
    (void)destination;
    (void)source;
    ocean_tensor_fail(
        "CUDA backend was requested but this build has no CUDA support"
    );
}

static void ocean_tensor_cuda_read_unavailable(
    const ocean_tensor_handle_t tensor,
    void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail(
        "CUDA backend was requested but this build has no CUDA support"
    );
}

static void ocean_tensor_cuda_write_unavailable(
    ocean_tensor_handle_t tensor,
    const void *host_data
) {
    (void)tensor;
    (void)host_data;
    ocean_tensor_fail(
        "CUDA backend was requested but this build has no CUDA support"
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
        "CUDA backend was requested but this build has no CUDA support"
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
        "CUDA backend was requested but this build has no CUDA support"
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
        "CUDA backend was requested but this build has no CUDA support"
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
        "CUDA backend was requested but this build has no CUDA support"
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

#endif
