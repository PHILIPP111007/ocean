/* OpenCL elementwise, reduction, loss, cache and optimizer launchers. */
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

static void ocean_tensor_opencl_gelu(
    const ocean_tensor_handle_t first,
    const ocean_tensor_handle_t second,
    ocean_tensor_handle_t output,
    int key
) {
    if (first->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(key);
    int size = (int)first->size;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &first->gpu_data),
        "clSetKernelArg"
    );
    if (key == OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &second->gpu_data),
            "clSetKernelArg"
        );
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->gpu_data),
            "clSetKernelArg"
        );
    } else {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
            "clSetKernelArg"
        );
    }
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel,
            key == OCEAN_TENSOR_OPENCL_KERNEL_GELU_BACKWARD_FLOAT32 ? 3 : 2,
            sizeof(int),
            &size
        ),
        "clSetKernelArg"
    );
    size_t global_size = first->size ? first->size : 1;
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

static void ocean_tensor_opencl_rowwise(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output,
    ocean_tensor_opencl_kernel_key key,
    int width,
    float epsilon,
    int mean
) {
    if (input->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(key);
    int rows = width > 0 ? (int)(input->size / (size_t)width) : 0;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &width), "clSetKernelArg"
    );
    if (key == OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_FLOAT32) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 4, sizeof(float), &epsilon),
            "clSetKernelArg"
        );
    } else if (key == OCEAN_TENSOR_OPENCL_KERNEL_REDUCE_FLOAT32) {
        ocean_tensor_opencl_check(
            clSetKernelArg(kernel, 4, sizeof(int), &mean), "clSetKernelArg"
        );
    }
    if (rows == 0) return;
    size_t global_size = (size_t)rows;
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

static void ocean_tensor_opencl_ternary_quantize(
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t output
) {
    if (input->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }

    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_TERNARY_QUANTIZE_FLOAT32
    );
    int size = (int)input->size;
    const size_t local_size = 64u;

    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel, 2, local_size * sizeof(float), NULL
        ),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &size),
        "clSetKernelArg"
    );

    size_t global_size = local_size;
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

static void ocean_tensor_opencl_cache_write(
    const ocean_tensor_handle_t cache,
    const ocean_tensor_handle_t value,
    int position
) {
    if (value->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CACHE_WRITE_FLOAT32
    );
    int heads = (int)cache->shape[1];
    int sequence = (int)cache->shape[2];
    int value_sequence = (int)value->shape[2];
    int width = (int)cache->shape[3];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &cache->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &value->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &heads), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &sequence), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &value_sequence),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &width), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &position), "clSetKernelArg"
    );
    size_t global_size = value->size ? value->size : 1;
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

static void ocean_tensor_opencl_cache_slice(
    const ocean_tensor_handle_t cache,
    const ocean_tensor_handle_t output,
    int start
) {
    if (output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CACHE_SLICE_FLOAT32
    );
    int heads = (int)cache->shape[1];
    int source_sequence = (int)cache->shape[2];
    int output_sequence = (int)output->shape[2];
    int width = (int)cache->shape[3];
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &cache->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &heads), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &source_sequence),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &output_sequence),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &width), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &start), "clSetKernelArg"
    );
    size_t global_size = output->size ? output->size : 1;
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

static void ocean_tensor_opencl_slice_last_dim(
    const ocean_tensor_handle_t input,
    const ocean_tensor_handle_t output,
    int start
) {
    if (output->size == 0) return;
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_SLICE_LAST_DIM_FLOAT32
    );
    size_t input_width_size = input->shape[input->ndim - 1];
    size_t output_width_size = output->shape[output->ndim - 1];
    size_t rows_size = input->size / input_width_size;
    if (input_width_size > (size_t)INT32_MAX ||
        output_width_size > (size_t)INT32_MAX ||
        rows_size > (size_t)INT32_MAX) {
        ocean_tensor_fail("Tensor slice dimensions are too large for OpenCL");
    }
    int rows = (int)rows_size;
    int input_width = (int)input_width_size;
    int output_width = (int)output_width_size;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    int values[] = {rows, input_width, start, output_width};
    for (int index = 0; index < 4; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(2 + index), sizeof(int), &values[index]
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
}

static int ocean_tensor_opencl_argmax(
    const ocean_tensor_handle_t tensor
) {
    if (tensor->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL argmax");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_ARGMAX_FLOAT32
    );
    if (!ocean_tensor_opencl.argmax_result) {
        cl_int status = CL_SUCCESS;
        ocean_tensor_opencl.argmax_result = clCreateBuffer(
            ocean_tensor_opencl.context, CL_MEM_READ_WRITE,
            sizeof(cl_int), NULL, &status
        );
        ocean_tensor_opencl_check(status, "clCreateBuffer(argmax)");
    }

    const size_t local_size = 64u;
    int size = (int)tensor->size;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &tensor->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(
            kernel, 1, sizeof(cl_mem), &ocean_tensor_opencl.argmax_result
        ),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, local_size * sizeof(float), NULL),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, local_size * sizeof(int), NULL),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size), "clSetKernelArg"
    );

    size_t global_size = local_size;
    cl_event kernel_event = NULL;
    ocean_tensor_opencl_check(
        clEnqueueNDRangeKernel(
            ocean_tensor_opencl.queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &kernel_event
        ),
        "clEnqueueNDRangeKernel"
    );
    int result = 0;
    ocean_tensor_opencl_check(
        clEnqueueReadBuffer(
            ocean_tensor_opencl.queue, ocean_tensor_opencl.argmax_result,
            CL_TRUE, 0, sizeof(result), &result, 1, &kernel_event, NULL
        ),
        "clEnqueueReadBuffer(argmax)"
    );
    ocean_tensor_opencl_release_event(kernel_event);
    return result;
}

static void ocean_tensor_opencl_causal_softmax(
    const ocean_tensor_handle_t tensor,
    const ocean_tensor_handle_t result,
    int rows,
    int width
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CAUSAL_SOFTMAX_FLOAT32
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &tensor->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &result->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &width), "clSetKernelArg"
    );
    size_t global_size = (size_t)rows;
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

static void ocean_tensor_opencl_embedding_forward(
    const ocean_tensor_handle_t weight,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t output,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_FORWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &weight->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &index_count),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &dim),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)index_count * (size_t)dim;
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

static void ocean_tensor_opencl_embedding_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t indices,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int index_count,
    int vocab,
    int dim
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_EMBEDDING_BACKWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &index_count),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &dim),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)index_count * (size_t)dim;
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

static void ocean_tensor_opencl_cross_entropy_forward(
    const ocean_tensor_handle_t logits,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t probabilities,
    ocean_tensor_handle_t row_losses,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_FORWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &logits->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &targets->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &probabilities->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &row_losses->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &rows),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)rows;
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

static void ocean_tensor_opencl_cross_entropy_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t probabilities,
    const ocean_tensor_handle_t targets,
    ocean_tensor_handle_t gradient,
    ocean_tensor_handle_t error,
    int rows,
    int vocab
) {
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_CROSS_ENTROPY_BACKWARD_FLOAT32_INT64
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &probabilities->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &targets->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(int), &rows),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 6, sizeof(int), &vocab),
        "clSetKernelArg"
    );
    size_t global_size = (size_t)rows;
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

static void ocean_tensor_opencl_softmax_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t output,
    ocean_tensor_handle_t gradient,
    int width
) {
    if (upstream->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_SOFTMAX_BACKWARD_FLOAT32
    );
    int rows = width > 0 ? (int)(upstream->size / (size_t)width) : 0;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &width), "clSetKernelArg"
    );
    if (rows == 0) return;
    size_t global_size = (size_t)rows;
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

static void ocean_tensor_opencl_layer_norm_backward(
    const ocean_tensor_handle_t upstream,
    const ocean_tensor_handle_t input,
    ocean_tensor_handle_t gradient,
    int width,
    double epsilon
) {
    if (upstream->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_LAYER_NORM_BACKWARD_FLOAT32
    );
    int rows = width > 0 ? (int)(upstream->size / (size_t)width) : 0;
    float epsilon_value = (float)epsilon;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &upstream->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(int), &rows), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &width), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 5, sizeof(float), &epsilon_value),
        "clSetKernelArg"
    );
    if (rows == 0) return;
    size_t global_size = (size_t)rows;
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

static void ocean_tensor_opencl_sgd_update(
    ocean_tensor_handle_t parameter,
    const ocean_tensor_handle_t gradient,
    double learning_rate
) {
    if (parameter->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_SGD_UPDATE_FLOAT32
    );
    int size = (int)parameter->size;
    float rate = (float)learning_rate;
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameter->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(int), &size), "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(float), &rate), "clSetKernelArg"
    );
    if (size == 0) return;
    size_t global_size = parameter->size;
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

static void ocean_tensor_opencl_adamw_update(
    ocean_tensor_handle_t parameter,
    const ocean_tensor_handle_t gradient,
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
    if (parameter->size > (size_t)INT32_MAX) {
        ocean_tensor_fail("GPU Tensor is too large for OpenCL kernel indexing");
    }
    cl_kernel kernel = ocean_tensor_opencl_get_kernel(
        OCEAN_TENSOR_OPENCL_KERNEL_ADAMW_UPDATE_FLOAT32
    );
    int size = (int)parameter->size;
    float values[7] = {
        (float)learning_rate,
        (float)beta1,
        (float)beta2,
        (float)epsilon,
        (float)weight_decay,
        (float)bias_correction1,
        (float)bias_correction2,
    };
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameter->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &first_moment->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &second_moment->gpu_data),
        "clSetKernelArg"
    );
    ocean_tensor_opencl_check(
        clSetKernelArg(kernel, 4, sizeof(int), &size), "clSetKernelArg"
    );
    for (int index = 0; index < 7; ++index) {
        ocean_tensor_opencl_check(
            clSetKernelArg(
                kernel, (cl_uint)(5 + index), sizeof(float), &values[index]
            ),
            "clSetKernelArg"
        );
    }
    if (size == 0) return;
    size_t global_size = parameter->size;
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

