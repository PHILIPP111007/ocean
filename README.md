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
```

Tensor storage is contiguous and row-major. Indexing is bounds-checked by the generated runtime;
provably safe hot loops can use a checked-once fast path. The current benchmark multiplies two
`100 × 100` matrices 1,000 times.

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
python main.py
```

`python main.py` parses `examples/main.oc`, validates it, generates
`examples/generated_code.c`, and builds the executable with GCC. For a reproducible baseline
without optimization flags:

```bash
python benchmarks/benchmark_main.py --runs 3
```

The benchmark reports parsing, validation, generation, compilation, and runtime separately. Use
`--json` for machine-readable output and `--keep` to preserve generated artifacts.

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
- `src/debug.py` — diagnostics, source locations, ownership, and borrow validation.
- `src/codegen/` — C lowering for scopes, types, ownership, containers, tensors, and OOP.
- `src/compiler.py` — public `CCodeGenerator` compatibility API.
- `tests/` — pytest regression and generated-C tests.
- `docs/` — memory model and backend design notes.

## Honest project status

Ocean is an active prototype, not a finished Rust replacement. The safe core is strongest for
intra-function ownership and lexical borrows. The current boundaries include non-atomic ARC,
thread-confined managed objects, possible leaks from reference cycles, incomplete integer-overflow
semantics, and an unsafe FFI escape hatch. These limitations are documented rather than hidden.

## Contributing

Keep Python code readable with four-space indentation and focused backend modules. Add a regression
test for every compiler or memory-model change, then run:

```bash
pytest -v
git diff --check
```

The project roadmap and deeper backend rationale are available in `docs/`.
