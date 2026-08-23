# `Tensor[T]`

## Purpose

`Tensor` is the device-aware tensor object and the only tensor type:

```text
Tensor[float32]        # public device-aware tensor
```

The facade accepts numeric `T`: `bool`, signed/unsigned integer types,
`size_t`/`intptr_t`/`uintptr_t`, `float16`, `float32`, and `float64`.
`str` and other reference types are rejected. `Tensor` without a type
argument is an alias for `Tensor[float32]`.

Static constructors use the class itself as the receiver, in the same style as
Python. The dtype is taken from the surrounding `Tensor[T]` annotation; if no
dtype context is available, it defaults to `float32`:

```text
var weights: Tensor[float32] = Tensor.zeros(128, 128, "cpu")
var labels: Tensor[int32] = Tensor.from_list([[1, 2, 3]], "cpu")
```

The legacy spelling `Tensor[T].zeros(...)` remains accepted for compatibility,
but new code should prefer the bare `Tensor.method(...)` form.

## Public API

The initial API contract is:

```text
class Tensor:
    @staticmethod
    def zeros(*shape: int, device: str) -> Tensor[T]
    def from_list(source: list, device: str) -> Tensor[T]
    @staticmethod
    def load_npy(path: str, device: str) -> Tensor[T]
    def save_npy(self, path: str) -> None

    def to(self, device: str) -> Tensor[T]
    def matmul(self, other: &Tensor[T]) -> Tensor[T]
    def add(self, other: &Tensor[T]) -> Tensor[T]
    def sub(self, other: &Tensor[T]) -> Tensor[T]
    def mul(self, other: &Tensor[T]) -> Tensor[T]
    def div(self, other: &Tensor[T]) -> Tensor[T]
    def add_scalar(self, value: float64) -> Tensor[T]
    def sub_scalar(self, value: float64) -> Tensor[T]
    def mul_scalar(self, value: float64) -> Tensor[T]
    def div_scalar(self, value: float64) -> Tensor[T]
    def gelu(self) -> Tensor[float32]
    def get(self, row: int, col: int) -> float64
    def set(self, row: int, col: int, value: float64) -> None
    def reshape(self, rows: int, cols: int) -> Tensor[T]
    def transpose(self) -> Tensor[T]
    def row(self, row: int) -> Tensor[T]
    def column(self, column: int) -> Tensor[T]
    def slice(self, axis: int, start: int, stop: int, step: int) -> Tensor[T]
    def sum(self) -> float64
    def mean(self) -> float64
    def max(self) -> float64
    def min(self) -> float64
    def item(self) -> float64
    def dtype(self) -> str
    def is_contiguous(self) -> bool
    def contiguous(self) -> Tensor[T]
    def fill(self, value: float64) -> None
    def copy(self) -> Tensor[T]
    def ternary_quantize(self) -> Tensor[float32]
    def ternary_scale(self) -> float64
    def ternary_pack(self, scale: float64, transpose: bool) -> Tensor[int32]
    def packed_qkv_inference(self, q_weight: &Tensor, q_scale: float64, q_bias: &Tensor, k_weight: &Tensor, k_scale: float64, k_bias: &Tensor, v_weight: &Tensor, v_scale: float64, v_bias: &Tensor, out_features: int) -> Tensor[float32]
    def shape(self, axis: int) -> int
    def ndim() -> int
    def size() -> size_t
    def device() -> str
```

Valid device strings in v1 are exactly `"cpu"` and `"gpu"`.

When OpenCL is enabled, `"gpu"` requires an actual device reported as
`CL_DEVICE_TYPE_GPU`. The runtime searches all OpenCL platforms and refuses to
fall back to `CL_DEVICE_TYPE_DEFAULT`, because that default may be a CPU OpenCL
implementation and would make a GPU benchmark misleading.

Example:

```text
var A: Tensor[float32] = Tensor.zeros(100, 100, "cpu")
var B: Tensor[float32] = Tensor.zeros(100, 100, "cpu")

var A_gpu: Tensor[float32] = A.to("gpu")
var B_gpu: Tensor[float32] = B.to("gpu")
var C_gpu: Tensor[float32] = A_gpu.matmul(B_gpu)
var C: Tensor[float32] = C_gpu.to("cpu")
```

For user code, literals are converted directly into backend-owned storage:

```text
var A: Tensor[float32] = Tensor.from_list([[1.0, 2.0], [3.0, 4.0]], "cpu")
```

## NumPy `.npy` files

`Tensor.load_npy(path, device)` and `tensor.save_npy(path)` provide a
NumPy-compatible interchange path for dense numeric tensors:

```text
var weights: Tensor[float32] = Tensor.load_npy("weights.npy", "cpu")
weights.save_npy("weights_copy.npy")
```

The runtime reads `.npy` versions 1.0, 2.0, and 3.0, including little- and
big-endian numeric descriptors for `bool`, signed/unsigned 8/16/32/64-bit
integers, and `float16`/`float32`/`float64`. v1 output is emitted whenever the
header fits its 16-bit length field; larger headers use v2.0. Data is stored
in row-major (`fortran_order: False`) form. Fortran-order arrays and object,
string, and structured dtypes are rejected because they do not map to the
numeric `Tensor[T]` model yet.

Saving a GPU tensor first downloads a CPU copy. Non-contiguous views are
materialized as contiguous row-major data before writing, so the file can be
loaded by NumPy and other `.npy` implementations without Ocean-specific
metadata.

Indexing is rank-generic and supports read, write, and augmented assignment:

```text
var cube: Tensor[int32] = Tensor.from_list([[[1, 2], [3, 4]]], "cpu")
cube[0, 1, 1] *= 2
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

The C runtime implements this boundary through the internal
`std/tensor/tensor_backend.h` contract. Each backend supplies storage
`allocate`, `zero`, `copy`, `read`, `write`, `release`, and operation entry
points. The runtime selects the table from the Tensor device, so public methods
do not duplicate CPU/OpenCL storage transitions.

## `matmul`

`A.matmul(B)` computes ordinary row-major `C = A x B`:

- `A.shape[1] == B.shape[0]` is required;
- both tensors must be on the same device in v1;
- the result is allocated on that device;
- CPU dispatches through a dtype-generic implementation;
- GPU uses OpenCL kernels for `float32` and `int32`, with a correct CPU
  fallback for other numeric dtypes until specialized kernels are added;

For autoregressive GPU inference, the OpenCL backend selects a specialized
128-thread matrix-vector kernel for `[1, K] × [K, N]` and `[1, 1, K] × [K, N]`.
It tiles the input vector in local memory and avoids the wasted rows of the
general 8×8 tiled kernel. Inference-only Linear can fuse the bias addition into
the same kernel; training/autograd keeps the ordinary matmul plus add path.
- the operation must not transpose `B` implicitly.

`matmul` is dispatched through the selected backend's operation table. CPU and
OpenCL therefore share the same shape/dtype/device checks, while
backend-specific allocation and transfer code stays behind the runtime
contract.

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
`fill` updates the existing tensor in place. `reshape`, `transpose`, `row`, `column`, and `slice`
currently return independent contiguous storage, so they do not share an ownership link with the
source tensor. `row` and `column` currently require a 2D tensor. `slice` uses a positive step and
requires `0 <= start <= stop <= shape[axis]`.

`add`, `sub`, `mul`, `div`, scalar operations, and `fill` now also enter through the backend
operation table. The OpenCL backend keeps its fast equal-shape kernels and performs the existing
CPU round-trip fallback for broadcasting and unsupported dtypes.

For Transformer hot paths, OpenCL has native float32 kernels for softmax and LayerNorm forward
and backward over the last axis, plus `sum_dim`/`mean_dim` reductions over the last axis. SGD and AdamW update GPU
resident parameters, gradients, and AdamW moment tensors in place. Unsupported axes or dtypes
continue to use the documented correctness-first fallback; the public API remains device-neutral.

`Embedding` is also GPU-native for contiguous float32 weights and contiguous int64 indices:
forward gathers rows directly on the device, while backward accumulates duplicate token IDs with
an atomic float update. Invalid token IDs are reported through a device-side error flag. The
public `Embedding.forward()` API remains unchanged.

`CrossEntropyLoss` is GPU-native for contiguous float32 logits and int64 targets. Forward computes
stable softmax probabilities and the mean negative log-likelihood on the device; backward computes
the normalized `(softmax - one_hot)` gradient on the device. Invalid targets are reported through
a device-side error flag.

`gelu()` uses the GPT-2 tanh approximation and has an autograd backward path.
For contiguous float32 GPU tensors both forward and backward use native OpenCL
kernels.

Inference code can disable graph construction around a forward/generation loop:

```ocean
var previous: bool = Tensor.grad_enabled()
Tensor.set_grad_enabled(False)
model.eval()
var logits: Tensor[float32] = model.forward(tokens, positions, bias)
Tensor.set_grad_enabled(previous)
```

The GPT-2 generation benchmark is
`examples/ML/gpt2_native_ternary_inference.oc`. It reports steady-state
tokens/second after one warmup token. The decoder pre-fills a per-layer
GPU-resident KV-cache and reuses it during autoregressive generation. Cache row
writes and prefix reads use dedicated OpenCL kernels when the backend is
enabled.

GPU float32 `matmul` also supports arbitrary-rank batched operands with leading-dimension
broadcasting. The same kernel supports logical transposes for autograd `dA`/`dB` computation,
while unsupported dtypes retain the existing CPU fallback.

`permute()` and `transpose()` return independent contiguous tensors. On GPU they use an OpenCL
gather kernel driven by the input strides and output shape, so arbitrary-rank layout transforms
for any Tensor dtype stay on the device. CPU tensors use the same stride-aware mapping without
the former recursive transpose path.

`ternary_quantize()` is currently defined for float32 tensors. It computes the mean absolute
weight scale, the `0.5 * scale` threshold, and the ternary values `{-scale, 0, +scale}` inside
the selected backend. The OpenCL implementation uses one work-group reduction and quantization
kernel, so GPU weights are not downloaded for host-side element access.

`ternary_pack()` stores 16 ternary values per 32-bit word using two-bit codes
(`00 = 0`, `01 = +1`, `10 = -1`). The packed inference kernels accept the
separate scale and decode weights on the GPU. `TernaryLinear` and the tied GPT-2
LM head use this path only when gradients are disabled; full-precision master
weights remain untouched for training.

GPT-2 inference additionally uses `packed_qkv_inference()`. It evaluates the
three packed Q/K/V projections in one 128-thread OpenCL launch while loading
each input tile once, and returns `[... , 3 * d_model]` laid out as Q, K, V.
The model then performs device-local last-dimension slices, so this fused path
does not introduce a GPU-to-CPU transfer between projection and attention.

The OpenCL runtime caches its context, queue, program, and kernels for the process and releases
them through an exit handler. Individual GPU buffers are released with their owning `Tensor`
object. OpenCL failures are converted to a descriptive process error with the operation and error
code.

Kernel objects are created lazily on first use and cached by operation and dtype. A program using
only CPU tensors does not initialize OpenCL, and a GPU program does not create unused kernels.
Device operations use an in-order queue with command events; host reads wait for their own read
event, while device-to-device copies and kernels are flushed without a global queue barrier.

For scalar iteration, `len(tensor)` returns the total number of elements and
`tensor[i]` reads the row-major flattened element. Explicit multidimensional
access such as `tensor[i, j]` remains rank-checked. This makes weight-loading
loops concise while keeping shape-aware access available.
