# OpenCL tensor backend

The first GPU backend uses the existing OpenCL design. The concrete runtime is
`std/tensor/tensor_runtime.c`; the compiler links it automatically when the
generated C includes `std/tensor/tensor_runtime.h`.

- `tensor_runtime.c` contains the tiled matmul design: each 8x8 work-group
  cooperatively loads input tiles into `__local` memory;
- the host wrapper lazily creates and reuses `cl_kernel` objects, uploads `A`,
  accepts a device buffer for `B`, launches a 2D NDRange, and downloads `C`;
- local workgroup size is `8 x 8`; partial tiles are zero-padded, so matrix
  dimensions do not need to be multiples of eight.
- float32 softmax and LayerNorm use one OpenCL work-item per row when the
  normalized dimension is the last axis;
- float32 `sum_dim` and `mean_dim` use a row-wise reduction kernel for the
  last axis;
- float32 SGD and AdamW update parameter and moment buffers in place on the
  device, without a host round-trip.

## Runtime objects

The runtime needs opaque objects with these ownership rules:

```text
  OpenCLContext  owns cl_context, queue, program, and a lazy kernel cache
GpuStorage     owns cl_mem, shape, strides, and a reference to OpenCLContext
```

`Tensor.to("gpu")` creates `GpuStorage`; `Tensor.to("cpu")` destroys only the
destination temporary after copying its data. OpenCL objects must be released
deterministically when the owning Tensor/backend object leaves scope.

## Kernel contract

The kernel contract should use ordinary row-major matrix multiplication:

```text
A[row, k] * B[k, col] -> C[row, col]
```

Batch/layer offsets must be represented explicitly as a byte or element stride.
The old `k * COLS_B * layer_index + col` expression is not a general batch
layout and must not be reused as the public Tensor ABI.

The current kernels reuse an 8x8 tile from both inputs through local memory.
This reduces global-memory traffic and keeps the inner traversal coalesced
while preserving the ordinary row-major result for arbitrary matrix sizes.

Before production use, the wrapper must also:

- validate dimensions and dtype;
- round global sizes up to the local size;
- keep bounds checks in the kernel;
- check every OpenCL return code;
- cache kernels by operation and dtype instead of creating one for every call;
- use nonblocking transfers with an event wait only when host data is requested;
- submit device copies and kernels with events, then use `clFlush` without a
  global queue `clFinish`;
- release buffers, kernels, programs, queues, and contexts on every failure path.

## Backend selection

The compiler should lower the public method to a runtime dispatch:

```text
Tensor.matmul -> cpu_tensor_matmul | opencl_tensor_matmul
Tensor.to     -> cpu_to_gpu       | gpu_to_cpu
```

This dispatch should be implemented in the standard runtime, not by emitting
raw OpenCL code into every user function. The Ocean compiler only needs to
recognize the standard `Tensor` methods and link the required backend runtime.

The runtime uses `tensor_backend.h` as the internal dispatch contract. CPU and
OpenCL provide the same storage lifecycle operations and operation entry points
for `matmul`, binary arithmetic, scalar arithmetic, and `fill`. OpenCL remains
optional at compile time: without
`OCEAN_TENSOR_ENABLE_OPENCL`, selecting `"gpu"` produces the explicit runtime
error instead of silently falling back to CPU.

The additional hot paths are runtime primitives rather than public OpenCL ABI:
`softmax`, `layer_norm`, `sum_dim`, `mean_dim`, SGD, and AdamW receive opaque
Tensor handles. The current native kernels target contiguous float32 tensors
and the last axis. Other axes, dtypes, and unsupported layouts retain the
correctness-first CPU round-trip behavior. Autograd backward for softmax and
LayerNorm is still CPU-based for now.

## Safety boundary

OpenCL FFI remains `unsafe` inside the runtime implementation. User code only
sees `Tensor`, `to`, `matmul`, and metadata methods. A GPU tensor cannot be
indexed as a CPU tensor until it is explicitly transferred to `"cpu"`.
