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

The upper-case `Tensor[T]` is the public facade that owns metadata and one
backend storage variant. The lower-case `tensor[T]` remains an internal typed
dense storage primitive for compiler-generated CPU layouts and C ABI
interoperability. User code should not access `cl_mem`, OpenCL contexts,
queues, or backend-specific pointers directly.

The two types interoperate explicitly through `Tensor.from_tensor(...)` and
`Tensor.to_tensor()`. New user code should prefer `Tensor.from_list(...)`; the
native bridge remains available for low-level code. This avoids making the
compiler-native CPU layout part of the public GPU ABI.

See:

- [Tensor API](tensor/README.md)
- [OpenCL backend](gpu/opencl.md)

## Imports

Quoted imports are resolved relative to the importing source file.

    import "./examples/matmul.oc"

Angle-bracket imports beginning with std/ are resolved from the standard
library.

    import <std/tensor/tensor.oc>
