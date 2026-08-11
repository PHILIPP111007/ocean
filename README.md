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

## Memory model

| Ocean type | Runtime model |
| --- | --- |
| `int`, `float`, `bool` | plain value |
| `list[T]`, `dict[K,V]`, tuples, classes | non-atomic reference counting |
| `str` | owned C string with copied aliases |
| `array[T]`, `tensor[T]` | unique-owned buffer |
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
python main.py build
```

`python main.py build` parses `examples/main.oc`, validates it, generates package artifacts under
`build/debug/`, and builds the executable with GCC. Running `python main.py` or the installed
`ocean` command without arguments only displays help. For a reproducible baseline without
optimization flags:

```bash
python benchmarks/benchmark_main.py --runs 3
```

The benchmark reports parsing, validation, generation, compilation, and runtime separately. Use
`--json` for machine-readable output and `--keep` to preserve generated artifacts.

The compiler pipeline also accepts a custom source file and build paths:

```bash
python main.py examples/threads.oc \
    --json-output /tmp/threads.json \
    --c-output /tmp/threads.c \
    --output /tmp/threads \
    --cflag=-pthread --cflag=-Wall
```

Use `--compiler clang` to select another C compiler, `--cflags "-O2 -g"` for a shell-style flag
group, `--no-compile` to stop after C generation, or `--run --run-arg VALUE` to execute the result.
Run `python main.py --help` for the complete option list.

This repository also contains an `ocean.toml` package manifest. Use `python main.py build` to use
the package layout and write artifacts to `build/debug/`. The legacy single-file workflow remains
available when a source path is passed explicitly, for example `python main.py examples/main.oc`.

## Packages and CLI

For repeatable builds, create a package instead of passing every path by hand:

```bash
python main.py init my_app
cd my_app
python ../main.py check
python ../main.py build --profile release
python ../main.py run --run-arg hello
python ../main.py test
```

`init` creates `ocean.toml` and `src/main.oc`. The CLI searches for the manifest in the current
directory and its parents, so commands also work from subdirectories. A manifest can be selected
explicitly with `--manifest`.

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
ocean check
ocean build --profile release
```

The package metadata is defined in `pyproject.toml`. The compiler is currently distributed as an
experimental package; uploading to PyPI should be done only after choosing the final project name
and configuring a PyPI token.

For strict manual C checks:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    examples/generated_code.c -o /tmp/ocean_generated
```

## Performance snapshot

On the development machine, the current `examples/main.oc` benchmark runs in roughly **3 seconds
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
