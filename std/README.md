# Ocean standard library

`std/` contains the language-level contracts for standard-library objects and
their runtime backends. Backend implementations may use C/OpenCL through the
explicit unsafe boundary, but the public Ocean API must remain safe and
backend-independent.

## Current priorities

1. `Tensor` — the public high-level tensor object with device-aware operations.
2. CPU tensor backend — currently retained only for legacy compiler compatibility.
3. OpenCL GPU backend — persistent context, device buffers, kernels, and
   transfers.

The upper-case `Tensor[T]` is the only public tensor type. The lower-case
`tensor[T]` remains temporarily as an internal typed dense storage primitive for
legacy compiler layouts and C ABI interoperability; it is deprecated and will
be removed after the compatibility bridges are retired. User code should not
access `cl_mem`, OpenCL contexts, queues, or backend-specific pointers directly.

The old types interoperate explicitly through `Tensor.from_tensor(...)` and
`Tensor.to_tensor()`. These are deprecated low-level bridges retained during
the migration; new user code must use `Tensor.from_list(...)` and public
metadata/accessor methods. This keeps the compiler-native CPU layout out of
the public GPU ABI.

See:

- [Tensor API](tensor/README.md)
- [OpenCL backend](gpu/opencl.md)

## Imports

Quoted imports are resolved relative to the importing source file.

    import "./examples/matmul.oc"

Angle-bracket imports beginning with std/ are resolved from the standard
library.

    import <std/tensor/tensor.oc>
