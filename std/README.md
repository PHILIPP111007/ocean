# Ocean standard library

`std/` contains the language-level contracts for standard-library objects and
their runtime backends. Backend implementations may use C/OpenCL through the
explicit unsafe boundary, but the public Ocean API must remain safe and
backend-independent.

## Current priorities

1. `Tensor` — a high-level tensor object with device-aware operations.
2. CPU tensor backend — the existing `tensor[T]` storage and code generator.
3. OpenCL GPU backend — persistent context, device buffers, kernels, and
   transfers.

The lower-case `tensor[T]` remains the typed dense storage primitive. The
upper-case `Tensor` is the public facade that owns metadata and one backend
storage variant. User code should not access `cl_mem`, OpenCL contexts, queues,
or backend-specific pointers directly.

See:

- [Tensor API](tensor/README.md)
- [OpenCL backend](gpu/opencl.md)
