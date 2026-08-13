# OpenCL tensor backend

The first GPU backend uses the existing OpenCL design. The concrete runtime is
`std/tensor/tensor_runtime.c`; the compiler links it automatically when the
generated C includes `std/tensor/tensor_runtime.h`.

- `tensor_runtime.c` contains the tiled matmul design: each 8x8 work-group
  cooperatively loads input tiles into `__local` memory;
- the host wrapper creates or reuses `cl_kernel`, uploads `A`, accepts a device
  buffer for `B`, launches a 2D NDRange, and downloads `C`;
- local workgroup size is `8 x 8`; partial tiles are zero-padded, so matrix
  dimensions do not need to be multiples of eight.

## Runtime objects

The runtime needs opaque objects with these ownership rules:

```text
OpenCLContext  owns cl_context, queue, program, and cached kernels
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
- cache kernels instead of creating one for every matmul;
- use blocking transfers where host data is requested and an in-order queue
  with `clFlush` for asynchronous matmul dispatch;
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
OpenCL provide the same storage lifecycle operations and a `matmul` entry
point. OpenCL remains optional at compile time: without
`OCEAN_TENSOR_ENABLE_OPENCL`, selecting `"gpu"` produces the explicit runtime
error instead of silently falling back to CPU.

## Safety boundary

OpenCL FFI remains `unsafe` inside the runtime implementation. User code only
sees `Tensor`, `to`, `matmul`, and metadata methods. A GPU tensor cannot be
indexed as a CPU tensor until it is explicitly transferred to `"cpu"`.
