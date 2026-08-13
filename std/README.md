# Ocean standard library

`std/` contains the language-level contracts for standard-library objects and
their runtime backends. Backend implementations may use C/OpenCL through the
explicit unsafe boundary, but the public Ocean API must remain safe and
backend-independent.

## Current priorities

1. `File` / `BinaryFile` — managed text and byte-stream access.
2. `Tensor` — the public high-level tensor object with device-aware operations.
3. OpenCL GPU backend — persistent context, device buffers, kernels, and
   transfers.

`Tensor[T]` is the only tensor type. Its storage is opaque and managed by the
standard runtime; user code should not access `cl_mem`, OpenCL contexts,
queues, or backend-specific pointers directly.

Tensor data enters through `Tensor.from_list(...)` and public
metadata/accessor methods; the CPU/GPU layout stays behind the opaque runtime
handle.

See:

- [Tensor API](tensor/README.md)
- [OpenCL backend](tensor/OpenCL.md)
- [File and BinaryFile API](io/README.md)

## Imports

Quoted imports are resolved relative to the importing source file.

    import "./examples/matmul.oc"

Angle-bracket imports beginning with std/ are resolved from the standard
library.

    import <std/tensor/tensor.oc>
