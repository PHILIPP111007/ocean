# `Tensor`

## Purpose

`Tensor` is the public, device-aware tensor object. It is intentionally
separate from the existing lower-case `tensor[T]` primitive:

```text
tensor[float32]  # typed CPU storage primitive
Tensor           # standard-library facade with a device
```

The first implementation targets `float32`. The facade should become
monomorphized as `Tensor[T]` when generic classes are available in the type
system; the backend ABI must not be redesigned at that point.

## Public API

The initial API contract is:

```text
class Tensor:
    @staticmethod
    def zeros(rows: int, cols: int, device: str) -> Tensor
    @staticmethod
    def from_tensor(source: &tensor[float32], device: str) -> Tensor

    def to(self, device: str) -> Tensor
    def to_tensor(self) -> tensor[float32]
    def matmul(self, other: &Tensor) -> Tensor
    def copy(self) -> Tensor
    def shape(self, axis: int) -> int
    def ndim() -> int
    def size() -> size_t
    def device() -> str
```

Valid device strings in v1 are exactly `"cpu"` and `"gpu"`.

Example:

```text
var A: Tensor = Tensor.zeros(100, 100, "cpu")
var B: Tensor = Tensor.zeros(100, 100, "cpu")

var A_gpu: Tensor = A.to("gpu")
var B_gpu: Tensor = B.to("gpu")
var C_gpu: Tensor = A_gpu.matmul(B_gpu)
var C: Tensor = C_gpu.to("cpu")
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

    var cpu: tensor[float32] = [[1.0, 2.0], [3.0, 4.0]]
    var device_tensor: Tensor = Tensor.from_tensor(cpu, "cpu")
    var gpu_tensor: Tensor = device_tensor.to("gpu")
    var restored: tensor[float32] = gpu_tensor.to_tensor()

from_tensor() copies source data into backend-owned storage. It accepts 2D
tensor[float32] values and preserves tensor views by using their shape and
strides. to_tensor() always returns a new CPU tensor; a GPU source is
downloaded first. This copy boundary keeps native tensor ownership separate
from the opaque device handle.

## Internal representation

The backend payload is opaque to Ocean code. Conceptually it is a tagged union:

```text
Tensor {
    device: "cpu" | "gpu"
    dtype: "float32"
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
- CPU dispatches to the existing tensor implementation;
- GPU dispatches to the OpenCL `matmul_gpu` kernel;
- the operation must not transpose `B` implicitly.

Mixed-device matmul is rejected in v1. An explicit `.to("cpu")` or
`.to("gpu")` makes the transfer visible and predictable.
