# `Tensor[T]`

## Purpose

`Tensor` is the public, device-aware tensor object. It is intentionally
separate from the existing lower-case `tensor[T]` primitive:

```text
tensor[float32]   # typed CPU storage primitive
Tensor[float32]   # standard-library facade with a device
```

The facade accepts numeric `T`: `bool`, signed/unsigned integer types,
`size_t`/`intptr_t`/`uintptr_t`, `float16`, `float32`, and `float64`.
`str` and other reference types are rejected. `Tensor` without a type
argument is an alias for `Tensor[float32]`.

## Public API

The initial API contract is:

```text
class Tensor:
    @staticmethod
    def zeros(*shape: int, device: str) -> Tensor[T]
    @staticmethod
    def from_tensor(source: &tensor[T], device: str) -> Tensor[T]

    def to(self, device: str) -> Tensor[T]
    def to_tensor(self) -> pointer
    def matmul(self, other: &Tensor[T]) -> Tensor[T]
    def add(self, other: &Tensor[T]) -> Tensor[T]
    def sub(self, other: &Tensor[T]) -> Tensor[T]
    def mul(self, other: &Tensor[T]) -> Tensor[T]
    def div(self, other: &Tensor[T]) -> Tensor[T]
    def add_scalar(self, value: float64) -> Tensor[T]
    def sub_scalar(self, value: float64) -> Tensor[T]
    def mul_scalar(self, value: float64) -> Tensor[T]
    def div_scalar(self, value: float64) -> Tensor[T]
    def reshape(self, rows: int, cols: int) -> Tensor[T]
    def transpose(self) -> Tensor[T]
    def sum(self) -> float64
    def fill(self, value: float64) -> None
    def copy(self) -> Tensor[T]
    def shape(self, axis: int) -> int
    def ndim() -> int
    def size() -> size_t
    def device() -> str
```

Valid device strings in v1 are exactly `"cpu"` and `"gpu"`.

Example:

```text
var A: Tensor[float32] = Tensor[float32].zeros(100, 100, "cpu")
var B: Tensor[float32] = Tensor[float32].zeros(100, 100, "cpu")

var A_gpu: Tensor[float32] = A.to("gpu")
var B_gpu: Tensor[float32] = B.to("gpu")
var C_gpu: Tensor[float32] = A_gpu.matmul(B_gpu)
var C: Tensor[float32] = C_gpu.to("cpu")
```

## Semantics of `.to(device)`

`.to()` is a non-mutating operation, matching the useful part of the PyTorch
model:

- `cpu -> gpu`: allocate a GPU buffer and upload the CPU values;
- `gpu -> cpu`: allocate CPU storage and perform a blocking download;
- same device: return an equivalent owned tensor while preserving non-mutating
  `.to()` semantics;
- any other string: raise a runtime error before touching backend state.

The original tensor remains valid after a transfer. A future `to_inplace()` may
be added, but it must be explicit because it changes the owned backend storage.

## Interoperability with tensor[T]

The compiler-native tensor[T] remains the CPU storage type. The standard
facade provides explicit conversions for the first backend version:

var cpu: tensor[int32] = [[1, 2], [3, 4]]
var device_tensor: Tensor[int32] = Tensor[int32].from_tensor(cpu, "cpu")
var gpu_tensor: Tensor[int32] = device_tensor.to("gpu")
var restored: tensor[int32] = gpu_tensor.to_tensor()

from_tensor() copies source data into backend-owned storage for any rank and
numeric `T`, preserving tensor views through their shape and strides.
to_tensor() always returns a new CPU tensor; a GPU source is
downloaded first. This copy boundary keeps native tensor ownership separate
from the opaque device handle.

## Internal representation

The backend payload is opaque to Ocean code. Conceptually it is a tagged union:

```text
Tensor {
    device: "cpu" | "gpu"
    dtype: numeric T
    shape: metadata
    strides: metadata
    storage: CpuStorage | GpuStorage
}
```

The implementation must not expose both a public CPU tensor field and a public
GPU handle. That would allow `device` and storage to disagree. `GpuStorage`
owns an opaque runtime handle and its OpenCL context association.

## `matmul`

`A.matmul(B)` computes ordinary row-major `C = A x B`:

- `A.shape[1] == B.shape[0]` is required;
- both tensors must be on the same device in v1;
- the result is allocated on that device;
- CPU dispatches through a dtype-generic implementation;
- GPU uses OpenCL kernels for `float32` and `int32`, with a correct CPU
  fallback for other numeric dtypes until specialized kernels are added;
- the operation must not transpose `B` implicitly.

Mixed-device matmul is rejected in v1. An explicit `.to("cpu")` or
`.to("gpu")` makes the transfer visible and predictable.

## Elementwise operations and layout transforms

The facade provides explicit elementwise methods instead of relying on operator overloading:

```text
var sum: Tensor[int32] = left.add(right)
var scaled: Tensor[int32] = sum.mul_scalar(2.0)
var transposed: Tensor[int32] = scaled.transpose()
var flattened: Tensor[int32] = transposed.reshape(1, 4)
var total: float64 = flattened.sum()
```

`add`, `sub`, `mul`, and `div` support trailing-axis broadcasting on CPU. GPU `float32` and
`int32` tensors use OpenCL kernels for equal-shape elementwise operations; broadcasting and other
numeric dtypes use the CPU implementation and are transferred back to the original device.
`fill` updates the existing tensor in place. `reshape` and `transpose` currently return independent
contiguous storage, so they do not share an ownership link with the source tensor.

The OpenCL runtime caches its context, queue, program, and kernels for the process and releases
them through an exit handler. Individual GPU buffers are released with their owning `Tensor`
object. OpenCL failures are converted to a descriptive process error with the operation and error
code.
