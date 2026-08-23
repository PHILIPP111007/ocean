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
        /* Preserve the public Tensor broadcasting contract. The native CUDA
           elementwise kernel currently handles contiguous equal-shape data;
           use the correctness-first CPU broadcast implementation for the
           uncommon metadata/broadcast case and restore the CUDA device. */
        ocean_tensor_handle_t left_cpu = ocean_tensor_to(left, "cpu");
        ocean_tensor_handle_t right_cpu = ocean_tensor_to(right, "cpu");
        ocean_tensor_handle_t cpu_result = ocean_tensor_binary_cpu(
            left_cpu, right_cpu, operation
        );
        ocean_tensor_handle_t result = ocean_tensor_to(cpu_result, "cuda");
        ocean_tensor_release(left_cpu);
        ocean_tensor_release(right_cpu);
        ocean_tensor_release(cpu_result);
        return result;
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

static void ocean_tensor_cuda_require_f32_contiguous(
    const ocean_tensor_handle_t tensor,
    const char *operation
) {
    if (!tensor || tensor->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail(operation);
    }
    ocean_tensor_cuda_require_contiguous(tensor, operation);
}

static ocean_tensor_handle_t ocean_tensor_cuda_gelu(
    const ocean_tensor_handle_t tensor
) {
    ocean_tensor_cuda_require_f32_contiguous(
        tensor, "CUDA GELU requires a contiguous float32 Tensor"
    );
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    ocean_cuda_gelu(tensor->cuda_data, result->cuda_data, tensor->size);
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cuda_softmax(
    const ocean_tensor_handle_t tensor
) {
    ocean_tensor_cuda_require_f32_contiguous(
        tensor, "CUDA softmax requires a contiguous float32 Tensor"
    );
    size_t width_size = tensor->shape[tensor->ndim - 1];
    size_t rows_size = width_size == 0 ? 0 : tensor->size / width_size;
    if (width_size > (size_t)INT_MAX || rows_size > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA softmax dimensions exceed int32 indexing");
    }
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    ocean_cuda_softmax_last_dim(
        tensor->cuda_data, result->cuda_data,
        (int)rows_size, (int)width_size
    );
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cuda_causal_softmax(
    const ocean_tensor_handle_t tensor
) {
    ocean_tensor_cuda_require_f32_contiguous(
        tensor, "CUDA causal softmax requires a contiguous float32 Tensor"
    );
    size_t width_size = tensor->shape[tensor->ndim - 1];
    size_t rows_size = width_size == 0 ? 0 : tensor->size / width_size;
    if (width_size > (size_t)INT_MAX || rows_size > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA causal softmax dimensions exceed int32 indexing");
    }
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    ocean_cuda_causal_softmax(
        tensor->cuda_data, result->cuda_data,
        (int)rows_size, (int)width_size
    );
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cuda_layer_norm(
    const ocean_tensor_handle_t tensor,
    double epsilon
) {
    ocean_tensor_cuda_require_f32_contiguous(
        tensor, "CUDA LayerNorm requires a contiguous float32 Tensor"
    );
    size_t width_size = tensor->shape[tensor->ndim - 1];
    size_t rows_size = width_size == 0 ? 0 : tensor->size / width_size;
    if (width_size > (size_t)INT_MAX || rows_size > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA LayerNorm dimensions exceed int32 indexing");
    }
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    ocean_cuda_layer_norm_last_dim(
        tensor->cuda_data, result->cuda_data,
        (int)rows_size, (int)width_size, (float)epsilon
    );
    return result;
}

static ocean_tensor_handle_t ocean_tensor_cuda_ternary_quantize(
    const ocean_tensor_handle_t tensor
) {
    ocean_tensor_cuda_require_f32_contiguous(
        tensor, "CUDA ternary quantization requires a contiguous float32 Tensor"
    );
    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        tensor->shape, tensor->ndim, tensor->dtype, OCEAN_TENSOR_BACKEND_CUDA
    );
    ocean_cuda_ternary_quantize(
        tensor->cuda_data, result->cuda_data, tensor->size
    );
    return result;
}

static void ocean_tensor_cuda_ternary_pack(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t output,
    double scale,
    bool transpose
) {
    ocean_tensor_cuda_require_f32_contiguous(
        input, "CUDA ternary packing requires a contiguous float32 input"
    );
    if (!output || output->dtype != OCEAN_TENSOR_INT32 ||
        output->device != OCEAN_TENSOR_BACKEND_CUDA ||
        !ocean_tensor_is_contiguous(output)) {
        ocean_tensor_fail("CUDA ternary packing requires contiguous int32 output");
    }
    if (input->ndim != 2 || output->ndim != 2 ||
        input->shape[0] > (size_t)INT_MAX || input->shape[1] > (size_t)INT_MAX ||
        output->shape[0] > (size_t)INT_MAX || output->shape[1] > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA ternary packing dimensions exceed int32 indexing");
    }
    ocean_cuda_ternary_pack(
        input->cuda_data, output->cuda_data,
        (int)input->shape[0], (int)input->shape[1],
        (int)output->shape[0], (int)output->shape[1],
        (float)scale, transpose ? 1 : 0
    );
}

static void ocean_tensor_cuda_packed_linear(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t packed,
    const ocean_tensor_handle_t bias,
    const ocean_tensor_handle_t output,
    int out_features,
    double scale
) {
    ocean_tensor_cuda_require_f32_contiguous(
        input, "CUDA packed Linear requires contiguous float32 input"
    );
    if (!packed || packed->dtype != OCEAN_TENSOR_INT32 ||
        !ocean_tensor_is_contiguous(packed) ||
        (bias && (bias->dtype != OCEAN_TENSOR_FLOAT32 ||
                  !ocean_tensor_is_contiguous(bias))) ||
        !output || output->dtype != OCEAN_TENSOR_FLOAT32 ||
        !ocean_tensor_is_contiguous(output)) {
        ocean_tensor_fail("CUDA packed Linear tensor metadata mismatch");
    }
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t rows_size = cols_a_size == 0 ? 0 : input->size / cols_a_size;
    if (cols_a_size > (size_t)INT_MAX || rows_size > (size_t)INT_MAX ||
        (size_t)out_features > (size_t)INT_MAX || packed->shape[1] > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA packed Linear dimensions exceed int32 indexing");
    }
    ocean_cuda_packed_linear(
        input->cuda_data, packed->cuda_data, bias ? bias->cuda_data : NULL,
        output->cuda_data, (int)rows_size, (int)cols_a_size, out_features,
        (int)packed->shape[1], (float)scale
    );
}

static void ocean_tensor_cuda_packed_qkv(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t q_packed,
    const ocean_tensor_handle_t q_bias,
    const ocean_tensor_handle_t k_packed,
    const ocean_tensor_handle_t k_bias,
    const ocean_tensor_handle_t v_packed,
    const ocean_tensor_handle_t v_bias,
    const ocean_tensor_handle_t output,
    int out_features,
    double q_scale,
    double k_scale,
    double v_scale
) {
    ocean_tensor_cuda_require_f32_contiguous(
        input, "CUDA packed QKV requires contiguous float32 input"
    );
    const ocean_tensor_handle_t packed[] = {q_packed, k_packed, v_packed};
    const ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
    for (int index = 0; index < 3; ++index) {
        if (!packed[index] || packed[index]->dtype != OCEAN_TENSOR_INT32 ||
            !ocean_tensor_is_contiguous(packed[index]) ||
            !biases[index] || biases[index]->dtype != OCEAN_TENSOR_FLOAT32 ||
            !ocean_tensor_is_contiguous(biases[index])) {
            ocean_tensor_fail("CUDA packed QKV tensor metadata mismatch");
        }
    }
    if (!output || output->dtype != OCEAN_TENSOR_FLOAT32 ||
        !ocean_tensor_is_contiguous(output)) {
        ocean_tensor_fail("CUDA packed QKV output metadata mismatch");
    }
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t rows_size = cols_a_size == 0 ? 0 : input->size / cols_a_size;
    if (cols_a_size > (size_t)INT_MAX || rows_size > (size_t)INT_MAX ||
        (size_t)out_features > (size_t)INT_MAX || q_packed->shape[1] > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA packed QKV dimensions exceed int32 indexing");
    }
    ocean_cuda_packed_qkv(
        input->cuda_data, q_packed->cuda_data, q_bias->cuda_data,
        k_packed->cuda_data, k_bias->cuda_data,
        v_packed->cuda_data, v_bias->cuda_data, output->cuda_data,
        (int)rows_size, (int)cols_a_size, out_features, (int)q_packed->shape[1],
        (float)q_scale, (float)k_scale, (float)v_scale
    );
}

static void ocean_tensor_cuda_packed_qkv_split(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t q_packed,
    const ocean_tensor_handle_t q_bias,
    const ocean_tensor_handle_t k_packed,
    const ocean_tensor_handle_t k_bias,
    const ocean_tensor_handle_t v_packed,
    const ocean_tensor_handle_t v_bias,
    const ocean_tensor_handle_t q_output,
    const ocean_tensor_handle_t k_output,
    const ocean_tensor_handle_t v_output,
    int out_features,
    double q_scale,
    double k_scale,
    double v_scale
) {
    ocean_tensor_cuda_require_f32_contiguous(
        input, "CUDA packed QKV requires contiguous float32 input"
    );
    const ocean_tensor_handle_t packed[] = {q_packed, k_packed, v_packed};
    const ocean_tensor_handle_t biases[] = {q_bias, k_bias, v_bias};
    const ocean_tensor_handle_t outputs[] = {q_output, k_output, v_output};
    for (int index = 0; index < 3; ++index) {
        if (!packed[index] || packed[index]->dtype != OCEAN_TENSOR_INT32 ||
            !ocean_tensor_is_contiguous(packed[index]) ||
            !biases[index] || biases[index]->dtype != OCEAN_TENSOR_FLOAT32 ||
            !ocean_tensor_is_contiguous(biases[index]) ||
            !outputs[index] || outputs[index]->dtype != OCEAN_TENSOR_FLOAT32 ||
            !ocean_tensor_is_contiguous(outputs[index])) {
            ocean_tensor_fail("CUDA split packed QKV tensor metadata mismatch");
        }
    }
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t rows_size = cols_a_size == 0 ? 0 : input->size / cols_a_size;
    if (cols_a_size > (size_t)INT_MAX || rows_size > (size_t)INT_MAX ||
        (size_t)out_features > (size_t)INT_MAX || q_packed->shape[1] > (size_t)INT_MAX) {
        ocean_tensor_fail("CUDA split packed QKV dimensions exceed int32 indexing");
    }
    ocean_cuda_packed_qkv_split(
        input->cuda_data, q_packed->cuda_data, q_bias->cuda_data,
        k_packed->cuda_data, k_bias->cuda_data,
        v_packed->cuda_data, v_bias->cuda_data,
        q_output->cuda_data, k_output->cuda_data, v_output->cuda_data,
        (int)rows_size, (int)cols_a_size, out_features, (int)q_packed->shape[1],
        (float)q_scale, (float)k_scale, (float)v_scale
    );
}

static void ocean_tensor_cuda_packed_attention_decode(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t q_packed,
    const ocean_tensor_handle_t q_bias,
    const ocean_tensor_handle_t k_packed,
    const ocean_tensor_handle_t k_bias,
    const ocean_tensor_handle_t v_packed,
    const ocean_tensor_handle_t v_bias,
    const ocean_tensor_handle_t cache_k,
    const ocean_tensor_handle_t cache_v,
    const ocean_tensor_handle_t output,
    int position,
    int n_heads,
    int head_dim,
    double q_scale,
    double k_scale,
    double v_scale
) {
    const ocean_tensor_handle_t tensors[] = {
        input, q_packed, q_bias, k_packed, k_bias, v_packed, v_bias,
        cache_k, cache_v, output,
    };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        if (!tensors[index]) ocean_tensor_fail("CUDA packed attention received null Tensor");
    }
    ocean_tensor_cuda_require_f32_contiguous(
        input, "CUDA packed attention requires contiguous float32 input"
    );
    ocean_tensor_cuda_require_f32_contiguous(
        q_bias, "CUDA packed attention requires float32 q bias"
    );
    ocean_tensor_cuda_require_f32_contiguous(
        k_bias, "CUDA packed attention requires float32 k bias"
    );
    ocean_tensor_cuda_require_f32_contiguous(
        v_bias, "CUDA packed attention requires float32 v bias"
    );
    ocean_tensor_cuda_require_f32_contiguous(
        cache_k, "CUDA packed attention requires contiguous cache_k"
    );
    ocean_tensor_cuda_require_f32_contiguous(
        cache_v, "CUDA packed attention requires contiguous cache_v"
    );
    ocean_tensor_cuda_require_f32_contiguous(
        output, "CUDA packed attention requires contiguous output"
    );
    const ocean_tensor_handle_t packed[] = {q_packed, k_packed, v_packed};
    for (int index = 0; index < 3; ++index) {
        if (packed[index]->dtype != OCEAN_TENSOR_INT32 ||
            !ocean_tensor_is_contiguous(packed[index])) {
            ocean_tensor_fail("CUDA packed attention weight metadata mismatch");
        }
    }
    size_t cols_a = input->shape[input->ndim - 1];
    size_t max_seq = cache_k->shape[2];
    if (cols_a > (size_t)INT_MAX || q_packed->shape[1] > (size_t)INT_MAX ||
        max_seq > (size_t)INT_MAX || position < 0 || (size_t)position >= max_seq ||
        n_heads <= 0 || head_dim <= 0 || head_dim > 128) {
        ocean_tensor_fail("CUDA packed attention dimensions exceed supported range");
    }
    ocean_cuda_packed_qkv_attention_decode(
        input->cuda_data, q_packed->cuda_data, q_bias->cuda_data,
        k_packed->cuda_data, k_bias->cuda_data,
        v_packed->cuda_data, v_bias->cuda_data,
        cache_k->cuda_data, cache_v->cuda_data, output->cuda_data,
        (int)cols_a, (int)q_packed->shape[1], (int)max_seq, position,
        n_heads, head_dim, (float)q_scale, (float)k_scale, (float)v_scale
    );
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
