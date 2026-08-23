#ifndef OCEAN_STD_TENSOR_BACKEND_H
#define OCEAN_STD_TENSOR_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

/* Internal C backend contract. Ocean code only sees Tensor and never uses
   this header directly; it is shared by the runtime dispatch layer. */
typedef struct ocean_tensor_handle *ocean_tensor_handle_t;

typedef enum ocean_tensor_backend_kind {
    OCEAN_TENSOR_BACKEND_CPU = 0,
    OCEAN_TENSOR_BACKEND_OPENCL = 1,
    OCEAN_TENSOR_BACKEND_CUDA = 2,
} ocean_tensor_backend_kind;

/* Selection policy for the backend behind the public device="gpu" API.
   AUTO is the default; explicit values are useful for diagnostics and CI. */
typedef enum ocean_tensor_gpu_backend_preference {
    OCEAN_TENSOR_GPU_BACKEND_AUTO = 0,
    OCEAN_TENSOR_GPU_BACKEND_OPENCL = 1,
    OCEAN_TENSOR_GPU_BACKEND_CUDA = 2,
} ocean_tensor_gpu_backend_preference;

typedef struct ocean_tensor_backend_ops {
    ocean_tensor_backend_kind kind;
    const char *name;
    bool compiled;
    void (*allocate)(ocean_tensor_handle_t tensor);
    void (*zero)(ocean_tensor_handle_t tensor);
    void (*copy)(ocean_tensor_handle_t destination,
                 const ocean_tensor_handle_t source);
    void (*read)(const ocean_tensor_handle_t tensor, void *host_data);
    void (*write)(ocean_tensor_handle_t tensor, const void *host_data);
    void (*release)(ocean_tensor_handle_t tensor);
    ocean_tensor_handle_t (*matmul)(ocean_tensor_handle_t left,
                                    ocean_tensor_handle_t right);
    ocean_tensor_handle_t (*binary)(ocean_tensor_handle_t left,
                                    ocean_tensor_handle_t right,
                                    int operation);
    ocean_tensor_handle_t (*scalar)(ocean_tensor_handle_t tensor,
                                    double scalar,
                                    int operation);
    void (*fill)(ocean_tensor_handle_t tensor, double value);
} ocean_tensor_backend_ops;

#endif
