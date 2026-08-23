/* OpenCL matvec, packed-linear, batched-matmul and permute launchers. */
static void ocean_tensor_opencl_matvec(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    const ocean_tensor_handle_t result
) {
    if (result->size == 0) return;
    if (left->shape[left->ndim - 1] > (size_t)INT32_MAX ||
        right->shape[right->ndim - 1] > (size_t)INT32_MAX) {
        ocean_tensor_fail(
            "GPU matvec dimensions are too large for OpenCL kernel indexing"
        );
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_MATVEC_FLOAT32
    );
    int cols_a = (int)left->shape[left->ndim - 1];
    int cols_b = (int)right->shape[right->ndim - 1];
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
        clSetKernelArg(kernel, 3, sizeof(int), &cols_a), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &cols_b), "clSetKernelArg"
    );
    const size_t local_size = 128u;
    size_t global_size =
        ((size_t)cols_b + local_size - 1u) / local_size * local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_matvec_bias(
    const ocean_tensor_handle_t left,
    const ocean_tensor_handle_t right,
    const ocean_tensor_handle_t bias,
    const ocean_tensor_handle_t result
) {
    if (result->size == 0) return;
    if (left->shape[left->ndim - 1] > (size_t)INT32_MAX ||
        right->shape[right->ndim - 1] > (size_t)INT32_MAX) {
        ocean_tensor_fail(
            "GPU matvec dimensions are too large for OpenCL kernel indexing"
        );
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_MATVEC_BIAS_FLOAT32
    );
    int cols_a = (int)left->shape[left->ndim - 1];
    int cols_b = (int)right->shape[right->ndim - 1];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &left->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &right->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &result->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &cols_a), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &cols_b), "clSetKernelArg"
    );
    const size_t local_size = 128u;
    size_t global_size =
        ((size_t)cols_b + local_size - 1u) / local_size * local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_ternary_pack(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t output,
    double scale,
    bool transpose
) {
    if (output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_PACK_FLOAT32
    );
    size_t source_rows_size = input->shape[0];
    size_t source_cols_size = input->shape[1];
    size_t output_rows_size = output->shape[0];
    size_t packed_cols_size = output->shape[1];
    if (source_rows_size > (size_t)INT32_MAX ||
        source_cols_size > (size_t)INT32_MAX ||
        output_rows_size > (size_t)INT32_MAX ||
        packed_cols_size > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor.ternary_pack dimensions are too large");
    }
    int source_rows = (int)source_rows_size;
    int source_cols = (int)source_cols_size;
    int output_rows = (int)output_rows_size;
    int packed_cols = (int)packed_cols_size;
    float scale_value = (float)scale;
    int transpose_value = transpose ? 1 : 0;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    int values[] = {
        source_rows, source_cols, output_rows, packed_cols,
    };
    for (int index = 0; index < 4; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(2 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(float), &scale_value),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 7, sizeof(int), &transpose_value),
        "clSetKernelArg"
    );
    size_t global_size = output->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_packed_linear(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t packed,
    const ocean_tensor_handle_t bias,
    const ocean_tensor_handle_t output,
    int out_features,
    double scale
) {
    if (output->size == 0) return;
    bool with_bias = bias != NULL;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        with_bias
            ? OCEAN_TENSOR_OPENCL_KERNEL_PACKED_LINEAR_FLOAT32
            : OCEAN_TENSOR_OPENCL_KERNEL_PACKED_MATMUL_FLOAT32
    );
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t rows_size = input->size / cols_a_size;
    size_t packed_cols_size = packed->shape[1];
    if (cols_a_size > (size_t)INT32_MAX || rows_size > (size_t)INT32_MAX ||
        packed_cols_size > (size_t)INT32_MAX) {
        ocean_tensor_fail(
            "packed Tensor matmul dimensions are too large for OpenCL"
        );
    }
    int rows = (int)rows_size;
    int cols_a = (int)cols_a_size;
    int cols_b = out_features;
    int packed_cols = (int)packed_cols_size;
    float scale_value = (float)scale;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &packed->gpu_data),
        "clSetKernelArg"
    );
    int next_argument = 2;
    if (with_bias) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, next_argument, sizeof(cl_mem), &bias->gpu_data),
            "clSetKernelArg"
        );
        next_argument += 1;
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, next_argument, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    next_argument += 1;
    int values[] = {rows, cols_a, cols_b, packed_cols};
    for (int index = 0; index < 4; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(next_argument + index), sizeof(int),
                &values[index]
            ),
            "clSetKernelArg"
        );
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel, (cl_uint)(next_argument + 4), sizeof(float), &scale_value
        ),
        "clSetKernelArg"
    );
    const size_t local_size = 128u;
    size_t groups_per_row =
        ((size_t)cols_b + local_size - 1u) / local_size;
    size_t global_size = (size_t)rows * groups_per_row * local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_packed_qkv(
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
    if (output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_FLOAT32
    );
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t rows_size = input->size / cols_a_size;
    size_t packed_cols_size = q_packed->shape[1];
    if (cols_a_size > (size_t)INT32_MAX || rows_size > (size_t)INT32_MAX ||
        packed_cols_size > (size_t)INT32_MAX) {
        ocean_tensor_fail(
            "packed QKV dimensions are too large for OpenCL"
        );
    }
    int rows = (int)rows_size;
    int cols_a = (int)cols_a_size;
    int cols_b = out_features;
    int packed_cols = (int)packed_cols_size;
    float scales[] = {
        (float)q_scale, (float)k_scale, (float)v_scale,
    };
    ocean_tensor_handle_t packed[] = {
        q_packed, k_packed, v_packed,
    };
    ocean_tensor_handle_t biases[] = {
        q_bias, k_bias, v_bias,
    };
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(1 + index), sizeof(cl_mem),
                &packed[index]->gpu_data
            ),
            "clSetKernelArg"
        );
    }
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(4 + index), sizeof(cl_mem),
                &biases[index]->gpu_data
            ),
            "clSetKernelArg"
        );
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    int values[] = {rows, cols_a, cols_b, packed_cols};
    for (int index = 0; index < 4; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(8 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(12 + index), sizeof(float), &scales[index]
            ),
            "clSetKernelArg"
        );
    }
    const size_t local_size = 128u;
    size_t groups_per_row =
        ((size_t)cols_b + local_size - 1u) / local_size;
    size_t global_size = (size_t)rows * groups_per_row * local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_packed_qkv_split(
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
    if (q_output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_SPLIT_FLOAT32
    );
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t rows_size = input->size / cols_a_size;
    size_t packed_cols_size = q_packed->shape[1];
    if (cols_a_size > (size_t)INT32_MAX || rows_size > (size_t)INT32_MAX ||
        packed_cols_size > (size_t)INT32_MAX) {
        ocean_tensor_fail(
            "split packed QKV dimensions are too large for OpenCL"
        );
    }
    int rows = (int)rows_size;
    int cols_a = (int)cols_a_size;
    int cols_b = out_features;
    int packed_cols = (int)packed_cols_size;
    float scales[] = {
        (float)q_scale, (float)k_scale, (float)v_scale,
    };
    ocean_tensor_handle_t packed[] = {
        q_packed, k_packed, v_packed,
    };
    ocean_tensor_handle_t biases[] = {
        q_bias, k_bias, v_bias,
    };
    ocean_tensor_handle_t outputs[] = {
        q_output, k_output, v_output,
    };
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(1 + index), sizeof(cl_mem),
                &packed[index]->gpu_data
            ),
            "clSetKernelArg"
        );
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(4 + index), sizeof(cl_mem),
                &biases[index]->gpu_data
            ),
            "clSetKernelArg"
        );
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(7 + index), sizeof(cl_mem),
                &outputs[index]->gpu_data
            ),
            "clSetKernelArg"
        );
    }
    int values[] = {rows, cols_a, cols_b, packed_cols};
    for (int index = 0; index < 4; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(10 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(14 + index), sizeof(float), &scales[index]
            ),
            "clSetKernelArg"
        );
    }
    const size_t local_size = 128u;
    size_t groups_per_row =
        ((size_t)cols_b + local_size - 1u) / local_size;
    size_t global_size = (size_t)rows * groups_per_row * local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static void ocean_tensor_opencl_packed_qkv_attention_decode(
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
    if (output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_PACKED_QKV_ATTENTION_FLOAT32
    );
    size_t cols_a_size = input->shape[input->ndim - 1];
    size_t packed_cols_size = q_packed->shape[1];
    size_t max_seq_size = cache_k->shape[2];
    if (cols_a_size > (size_t)INT32_MAX ||
        packed_cols_size > (size_t)INT32_MAX ||
        max_seq_size > 2048u) {
        ocean_tensor_fail(
            "fused attention dimensions are too large for OpenCL"
        );
    }
    int cols_a = (int)cols_a_size;
    int packed_cols = (int)packed_cols_size;
    int max_seq = (int)max_seq_size;
    float scales[] = {
        (float)q_scale, (float)k_scale, (float)v_scale,
    };
    ocean_tensor_handle_t packed[] = {
        q_packed, k_packed, v_packed,
    };
    ocean_tensor_handle_t biases[] = {
        q_bias, k_bias, v_bias,
    };
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(1 + index), sizeof(cl_mem),
                &packed[index]->gpu_data
            ),
            "clSetKernelArg"
        );
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(4 + index), sizeof(cl_mem),
                &biases[index]->gpu_data
            ),
            "clSetKernelArg"
        );
    }
    ocean_tensor_handle_t buffers[] = {cache_k, cache_v, output};
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(7 + index), sizeof(cl_mem),
                &buffers[index]->gpu_data
            ),
            "clSetKernelArg"
        );
    }
    int values[] = {
        cols_a, packed_cols, max_seq, position, n_heads, head_dim,
    };
    for (int index = 0; index < 6; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(10 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(16 + index), sizeof(float), &scales[index]
            ),
            "clSetKernelArg"
        );
    }
    const size_t local_size = 128u;
    size_t global_size = (size_t)n_heads * local_size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
}

static ocean_tensor_handle_t ocean_tensor_matmul_opencl_batched(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    bool transpose_left,
    bool transpose_right
) {
    if (left->dtype != OCEAN_TENSOR_FLOAT32 ||
        right->dtype != OCEAN_TENSOR_FLOAT32) {
        ocean_tensor_fail(
            "GPU batched matmul currently requires float32 tensors"
        );
    }

    size_t output_ndim = left->ndim > right->ndim
        ? left->ndim : right->ndim;
    size_t batch_ndim = output_ndim - 2;
    size_t left_batch_ndim = left->ndim - 2;
    size_t right_batch_ndim = right->ndim - 2;
    size_t left_leading = batch_ndim - left_batch_ndim;
    size_t right_leading = batch_ndim - right_batch_ndim;

    size_t rows = transpose_left
        ? left->shape[left->ndim - 1]
        : left->shape[left->ndim - 2];
    size_t inner = transpose_left
        ? left->shape[left->ndim - 2]
        : left->shape[left->ndim - 1];
    size_t right_inner = transpose_right
        ? right->shape[right->ndim - 1]
        : right->shape[right->ndim - 2];
    size_t cols = transpose_right
        ? right->shape[right->ndim - 2]
        : right->shape[right->ndim - 1];
    if (inner != right_inner) {
        ocean_tensor_fail("batched matmul shape mismatch");
    }

    if (!transpose_left && !transpose_right &&
        left->ndim == 3 && right->ndim == 2 &&
        left->shape[0] == 1 && left->shape[1] == 1) {
        size_t output_shape[3] = {1, 1, cols};
        ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
            output_shape, 3, OCEAN_TENSOR_FLOAT32, OCEAN_TENSOR_GPU
        );
        ocean_tensor_opencl_matvec(left, right, result);
        return result;
    }

    size_t *output_shape = (size_t *)malloc(
        output_ndim * sizeof(size_t)
    );
    int *left_shape = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *right_shape = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *out_shape = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *left_strides = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    int *right_strides = (int *)calloc(batch_ndim ? batch_ndim : 1, sizeof(int));
    if (!output_shape || !left_shape || !right_shape || !out_shape ||
        !left_strides || !right_strides) {
        free(output_shape);
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        ocean_tensor_fail("out of memory in batched matmul metadata");
    }

    for (size_t axis = 0; axis < batch_ndim; ++axis) {
        size_t left_axis = axis >= left_leading
            ? axis - left_leading : SIZE_MAX;
        size_t right_axis = axis >= right_leading
            ? axis - right_leading : SIZE_MAX;
        size_t left_dim = left_axis == SIZE_MAX
            ? 1 : left->shape[left_axis];
        size_t right_dim = right_axis == SIZE_MAX
            ? 1 : right->shape[right_axis];
        if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
            free(output_shape);
            free(left_shape);
            free(right_shape);
            free(out_shape);
            free(left_strides);
            free(right_strides);
            ocean_tensor_fail(
                "batched matmul batch dimensions are not broadcastable"
            );
        }
        size_t dimension = left_dim > right_dim ? left_dim : right_dim;
        output_shape[axis] = dimension;
        left_shape[axis] = (int)left_dim;
        right_shape[axis] = (int)right_dim;
        out_shape[axis] = (int)dimension;
        left_strides[axis] = left_axis == SIZE_MAX
            ? 0 : (int)left->strides[left_axis];
        right_strides[axis] = right_axis == SIZE_MAX
            ? 0 : (int)right->strides[right_axis];
        if (left_dim > (size_t)INT32_MAX || right_dim > (size_t)INT32_MAX ||
            dimension > (size_t)INT32_MAX ||
            (left_axis != SIZE_MAX && left->strides[left_axis] > (size_t)INT32_MAX) ||
            (right_axis != SIZE_MAX && right->strides[right_axis] > (size_t)INT32_MAX)) {
            free(output_shape);
            free(left_shape);
            free(right_shape);
            free(out_shape);
            free(left_strides);
            free(right_strides);
            ocean_tensor_fail(
                "batched matmul metadata is too large for OpenCL"
            );
        }
    }
    output_shape[output_ndim - 2] = rows;
    output_shape[output_ndim - 1] = cols;
    if (rows > (size_t)INT32_MAX || inner > (size_t)INT32_MAX ||
        cols > (size_t)INT32_MAX) {
        free(output_shape);
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        ocean_tensor_fail("batched matmul dimensions are too large for OpenCL");
    }

    ocean_tensor_handle_t result = ocean_tensor_alloc_uninitialized(
        output_shape,
        output_ndim,
        OCEAN_TENSOR_FLOAT32,
        OCEAN_TENSOR_GPU
    );
    free(output_shape);
    if (result->size == 0) {
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        return result;
    }
    if (result->size > (size_t)INT32_MAX) {
        ocean_tensor_release(result);
        free(left_shape);
        free(right_shape);
        free(out_shape);
        free(left_strides);
        free(right_strides);
        ocean_tensor_fail(
            "batched matmul output is too large for OpenCL indexing"
        );
    }

    size_t metadata_count = batch_ndim ? batch_ndim : 1;
    cl_int status = CL_SUCCESS;
    cl_mem left_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), left_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem right_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), right_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem out_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), out_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem left_strides_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), left_strides, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem right_strides_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        metadata_count * sizeof(int), right_strides, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    free(left_shape);
    free(right_shape);
    free(out_shape);
    free(left_strides);
    free(right_strides);

    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_BATCHED_MATMUL_FLOAT32
    );
    int batch_count = (int)batch_ndim;
    int rows_value = (int)rows;
    int inner_value = (int)inner;
    int cols_value = (int)cols;
    int transpose_left_value = transpose_left ? 1 : 0;
    int transpose_right_value = transpose_right ? 1 : 0;
    int output_size = (int)result->size;
    cl_mem buffers[] = {
        left->gpu_data,
        right->gpu_data,
        result->gpu_data,
        left_shape_buffer,
        right_shape_buffer,
        out_shape_buffer,
        left_strides_buffer,
        right_strides_buffer,
    };
    for (int index = 0; index < 8; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, (cl_uint)index, sizeof(cl_mem), &buffers[index]),
            "clSetKernelArg"
        );
    }
    int values[] = {
        batch_count,
        rows_value,
        inner_value,
        cols_value,
        transpose_left_value,
        transpose_right_value,
        output_size,
    };
    for (int index = 0; index < 7; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(8 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    size_t global_size = result->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
    clReleaseMemObject(left_shape_buffer);
    clReleaseMemObject(right_shape_buffer);
    clReleaseMemObject(out_shape_buffer);
    clReleaseMemObject(left_strides_buffer);
    clReleaseMemObject(right_strides_buffer);
    return result;
}

static void ocean_tensor_opencl_permute(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output,
    const int *axes
) {
    if (input->size == 0) return;
    if (input->ndim > (size_t)INT32_MAX ||
        output->size > (size_t)INT32_MAX ||
        input->item_size > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor.permute is too large for OpenCL indexing");
    }

    int rank = (int)input->ndim;
    int item_size = (int)input->item_size;
    int output_size = (int)output->size;
    int *output_shape = (int *)malloc(input->ndim * sizeof(int));
    int *input_strides = (int *)malloc(input->ndim * sizeof(int));
    if (!output_shape || !input_strides) {
        free(output_shape);
        free(input_strides);
        ocean_tensor_fail("out of memory in Tensor.permute metadata");
    }
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        if (output->shape[axis] > (size_t)INT32_MAX ||
            input->strides[axis] > (size_t)INT32_MAX) {
            free(output_shape);
            free(input_strides);
            ocean_tensor_fail(
                "Tensor.permute metadata is too large for OpenCL"
            );
        }
        output_shape[axis] = (int)output->shape[axis];
        input_strides[axis] = (int)input->strides[axis];
    }

    cl_int status = CL_SUCCESS;
    cl_mem output_shape_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input->ndim * sizeof(int), output_shape, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem input_strides_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input->ndim * sizeof(int), input_strides, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    cl_mem axes_buffer = clCreateBuffer(
        ocean_tensor_opencl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input->ndim * sizeof(int), (void *)axes, &status
    );
    ocean_tensor_opencl_check(status, "clCreateBuffer");
    free(output_shape);
    free(input_strides);

    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_PERMUTE
    );
    cl_mem buffers[] = {
        input->gpu_data,
        output->gpu_data,
        output_shape_buffer,
        input_strides_buffer,
        axes_buffer,
    };
    for (int index = 0; index < 5; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, (cl_uint)index, sizeof(cl_mem), &buffers[index]),
            "clSetKernelArg"
        );
    }
    int values[] = {rank, item_size, output_size};
    for (int index = 0; index < 3; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(5 + index), sizeof(int), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    size_t global_size = output->size;
    cl_event event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, NULL, 0, NULL, &event
        ),
        "clEnqueueNDRangeKernel"
    );
    ocean_tensor_opencl_check(
        clFlush(ocean_tensor_opencl.queue), "clFlush"
    );
    ocean_tensor_opencl_release_event(event);
    clReleaseMemObject(output_shape_buffer);
    clReleaseMemObject(input_strides_buffer);
    clReleaseMemObject(axes_buffer);
}

