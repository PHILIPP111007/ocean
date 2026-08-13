# Ocean 🌊

> Python-like syntax. Native C11 output. Explicit ownership for systems and ML-oriented code.

<img src="images/ocean.jpg" alt="Ocean project illustration" width="560" />

Ocean is an experimental programming language and compiler for people who want the readability of
Python with a direct path to C. Ocean lowers source code to readable C11, adds ownership and borrow
checks before code generation, and provides contiguous `array[T]` and N-dimensional `tensor[T]`
types for numerical workloads.

## Why Ocean?

- **Familiar syntax** — functions, loops, classes, and type annotations stay lightweight.
- **Predictable memory** — values, ARC-managed objects, owned buffers, and borrows have distinct
  rules.
- **C when it matters** — generated symbols use the `ocean_` namespace and can interoperate with
  C/POSIX APIs through an explicit `unsafe:` boundary.
- **Numerical foundations** — dense arrays and row-major tensors with shape metadata, strides, and
  bounds-checked indexing.
- **Standard-library backends** — a device-aware `Tensor[T]` facade can dispatch between CPU and
  OpenCL-backed GPU storage.
- **Parallel loops** — selected `range` loops lower to OpenMP, including nested loops with
  `collapse(n)`.
- **Inspectable output** — the compiler emits C you can read, compile, profile, and debug.

## A small example

```python
def scale(values: &mut array[float32], factor: float32) -> None:
    for i in range(len(values)):
        values[i] = values[i] * factor
    return None


def main() -> int:
    var values: array[float32] = [1.0, 2.0, 3.0]
    scale(values, 2.0)
    print(values[0])
    return 0
```

`&mut` is an exclusive mutable borrow. It does not add reference-count operations to the generated
code and prevents conflicting access during validation.

## Arrays and tensors

Use `array[T]` for a one-dimensional numeric buffer and `tensor[T]` for dense N-dimensional data:

```python
var matrix: tensor[float32] = tensor.zeros(100, 100)
matrix[0, 1] = 3.5

var rows: int = matrix.shape[0]
var elements: int = len(matrix)
matrix.fill(0.0)
var total: float32 = matrix.sum()
var row: tensor[float32] = matrix.row(0)
var transposed: tensor[float32] = matrix.transpose_view()
var result: tensor[float32] = matrix + transposed
```

Tensor storage is contiguous and row-major. Indexing is bounds-checked by the generated runtime;
provably safe hot loops can use a checked-once fast path. `sum`, `fill`, `copy`, `row`, `column`,
`slice`, `transpose_view`, and 2D `matmul` are lowered to stride-aware C helpers. Views share the
source tensor's storage and retain its owner until the view is released. Arithmetic supports
NumPy-style trailing-axis broadcasting for compatible shapes; `transpose()` remains an explicit
copying operation. The current benchmark multiplies two `100 × 100` matrices 1,000 times.

### Device-aware `Tensor[T]`

The standard library adds an uppercase `Tensor[T]` object with an explicit device, while the
lowercase `tensor[T]` remains the compiler-native CPU storage type:

```text
tensor[float32]   # dense CPU tensor
Tensor[float32]   # device-aware standard-library facade
```

`Tensor[T]` supports numeric element types such as `bool`, signed and unsigned integers,
`size_t`, `intptr_t`, `uintptr_t`, `float16`, `float32`, and `float64`. Reference types such as
`str` are rejected. Without a type argument, `Tensor` uses `float32`.

The initial API supports arbitrary tensor rank for allocation and transfer:

```python
import <std/tensor/tensor.oc>

var native: tensor[float32] = [[1.0, 2.0], [3.0, 4.0]]
var cpu: Tensor[float32] = Tensor[float32].from_tensor(native, "cpu")
var gpu: Tensor[float32] = cpu.to("gpu")
var result_gpu: Tensor[float32] = gpu.matmul(gpu)
var result_cpu: Tensor[float32] = result_gpu.to("cpu")
var restored: tensor[float32] = result_cpu.to_tensor()
```

The supported devices are exactly `"cpu"` and `"gpu"`. `.to(device)` is non-mutating: it returns
an owned tensor on the requested device and leaves the source tensor valid. `Tensor` also provides
`zeros`, `from_tensor`, `copy`, `matmul`, `add`, `sub`, `mul`, `div`, scalar arithmetic,
`reshape`, `transpose`, `sum`, `fill`, `shape`, `ndim`, `size`, `device`, `to_tensor`, and
`release`.

`Tensor.matmul` requires compatible 2D shapes and matching devices. CPU matmul is dtype-generic.
The GPU path currently has OpenCL kernels for `float32` and `int32`; other numeric dtypes use a
correct CPU fallback until specialized GPU kernels are added. The operation does not transpose the
right-hand matrix implicitly.

Elementwise methods support trailing-axis broadcasting on CPU. Equal-shape `float32` GPU operations
use OpenCL kernels; broadcasting and other dtypes use the CPU implementation and transfer the
result back to the original device. `reshape` and `transpose` currently return independent
contiguous tensors, while `fill` mutates the existing tensor.

The public facade owns an opaque runtime handle. Ocean code does not access `cl_mem`, OpenCL
contexts, queues, or backend-specific pointers directly. See [std/tensor/README.md](std/tensor/README.md)
and [std/gpu/opencl.md](std/gpu/opencl.md) for the API and backend design.

## Imports and the standard library

Imports have two explicit forms:

```python
import "./examples/matmul.oc"       # relative to the importing source file
import <std/tensor/tensor.oc>        # from the repository standard library
```

Quoted imports are resolved relative to the importing file. Angle-bracket imports beginning with
`std/` are resolved from `./std/`. Standard-library C runtimes referenced by generated code are
discovered and compiled automatically.

## OpenMP parallel loops

Ocean can lower a restricted, ownership-safe subset of `range` loops to OpenMP:

```python
#pragma omp parallel for collapse(2) schedule(static)
for i in range(rows):
    for j in range(columns):
        output[i, j] = left[i, j] + right[i, j]
```

Nested loops must be perfectly nested when using `collapse(n)`. Supported clauses include
`schedule`, `collapse`, `reduction`, `private`, `firstprivate`, `lastprivate`, `shared`, `default`,
`nowait`, and `ordered`. The compiler adds `-fopenmp` automatically when generated C contains an
OpenMP pragma; it can also be supplied explicitly:

```bash
python main.py examples/matmul.oc --cflag=-fopenmp --run
```

The current parallel-loop subset requires constant non-zero integer steps and `range(...)` loops.
Managed objects such as lists, dictionaries, strings, and classes, as well as function calls,
`break`, `continue`, and unsupported nested-loop forms are rejected because the current ownership
runtime is thread-confined. More examples and validation rules are documented in
[docs/OpenMP.md](docs/OpenMP.md).

## OpenCL GPU build

The Tensor runtime uses OpenCL through an unsafe C implementation in
`std/tensor/tensor_runtime.c`. When the generated C references this runtime, the compiler discovers
and compiles the standard-library C source automatically. If `pkg-config` can find an `OpenCL`
package, its include, library, and linker flags are added automatically as well.

For a custom OpenCL installation, pass the include directory containing `CL/cl.h`, the library
directory containing `libOpenCL.so`, and the feature macro explicitly:

```bash
python main.py run examples/your_tensor_program.oc \
    --cflags "-I/opt/opencl/include -L/opt/opencl/lib -lOpenCL -DOCEAN_TENSOR_ENABLE_OPENCL"
```

For example, if the header is `/opt/opencl/include/CL/cl.h`, the correct include flag is
`-I/opt/opencl/include`, not the `CL` directory itself. The equivalent package configuration is:

```toml
[build]
compiler = "gcc"
cflags = [
    "-std=c11",
    "-I/opt/opencl/include",
    "-L/opt/opencl/lib",
    "-lOpenCL",
    "-DOCEAN_TENSOR_ENABLE_OPENCL",
]
```

The GPU backend requires an OpenCL platform and device at runtime. A CPU-only machine can still
use `Tensor[T]` on `"cpu"`; requesting `"gpu"` fails with a runtime error when no usable OpenCL
backend is available.

## Memory model

| Ocean type | Runtime model |
| --- | --- |
| `int`, `float`, `bool` | plain value |
| `list[T]`, `dict[K,V]`, tuples, classes | non-atomic reference counting |
| `str` | owned C string with copied aliases |
| `array[T]`, `tensor[T]` | unique-owned buffer |
| `Tensor[T]` | ARC-managed facade over CPU or GPU tensor storage |
| `&T` / `&mut T` | lexical immutable / exclusive borrow |
| raw pointers and direct C calls | explicit `unsafe:` boundary |

Owned objects are released at scope exit. `del` can release a value early, while borrows do not
retain or free their source. Read the full model in [docs/MEMORY_MODEL.md](docs/MEMORY_MODEL.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

pytest -v
```

The minimal package model is:

```toml
[package]
name = "my_app"
version = "0.1.0"
entry = "src/main.oc"
source = "src"
build = "build"

[build]
compiler = "gcc"
cflags = ["-std=c11"]

[build.profiles.release]
cflags = ["-O2"]
```

`check` validates and generates C; `build` also invokes the C compiler; `run` builds and executes;
`test` runs pytest; and `clean` removes the package's `build/` directory. Package artifacts are
isolated under `build/<profile>/`.

## Install as a Python package

Build the distributable artifacts locally with the standard Python packaging tools:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

The editable install exposes the `ocean` command:

```bash
ocean --help
ocean run examples/neural_network.oc
```

The package metadata is defined in `pyproject.toml`. The compiler is currently distributed as an
experimental package; uploading to PyPI should be done only after choosing the final project name
and configuring a PyPI token.

## Object-oriented ML example

Classes, constructors, fields, methods, and single-inheritance-compatible object layouts are
lowered to ARC-managed C objects. The repository contains equivalent small decoder-only Transformer
language-model examples:

- [examples/transformer_pytorch.py](examples/transformer_pytorch.py) — PyTorch implementation.
- [examples/transformer_ocean.oc](examples/transformer_ocean.oc) — the same model structure using
  Ocean classes and methods.

The Ocean version uses ordinary OOP syntax and can call C math/POSIX functionality only through an
explicit `unsafe:` boundary. Generated C is available for inspection with `--c-output` or the
legacy `examples/*.generated.c` workflow.

For strict manual C checks:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    examples/generated_code.c -o /tmp/ocean_generated
```

## Performance snapshot

On the development machine, the current `examples/matmul.oc` benchmark runs in roughly **3 seconds
without compiler optimization** and **under 1 second with `-O3`**. These numbers are hardware- and
compiler-dependent; the benchmark is the source of truth for comparisons.

## Compiler pipeline

```text
Ocean source → Parser → typed JSON graph → JSONValidator → CCodeGenerator → C11
```

- `src/parser.py` — syntax, types, expressions, indexing, arrays, and tensors.
- `src/typed_ir.py` — typed scopes, expression result types, data dependencies, and ownership effects.
- `src/debug.py` — diagnostics, source locations, ownership, and borrow validation.
- `src/codegen/` — C lowering for scopes, types, ownership, containers, tensors, and OOP.
- `src/compiler.py` — public `CCodeGenerator` compatibility API.
- `tests/` — pytest regression and generated-C tests.
- `docs/` — memory model and backend design notes.
- `graphify-out/` — generated code-navigation graph, reports, and analysis cache. Refresh it with
  `graphify update .` after source changes.

## Honest project status

Ocean is an active prototype, not a finished Rust replacement. The compiler now has a typed IR and
explicit move checks for unique arrays/tensors, while lexical borrows remain the safe zero-cost
interface for shared access. Current boundaries include non-atomic ARC, thread-confined managed
objects, possible leaks from reference cycles, incomplete integer-overflow semantics, broadcasting
shape errors being runtime checks, and an unsafe FFI escape hatch.

## Contributing

Keep Python code readable with four-space indentation and focused backend modules. Add a regression
test for every compiler or memory-model change, then run:

```bash
pytest -v
git diff --check
```

The project roadmap and deeper backend rationale are available in `docs/`.
