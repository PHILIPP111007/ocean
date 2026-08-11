# Ocean 🌊

<img src="images/ocean.jpg" alt="Ocean project illustration" width="500"/>

Ocean is a Python-like language that compiles to C11. The project targets systems programming and ML-oriented computation, with explicit borrow parameters and ownership checks during code generation.

## Architecture

The main pipeline is:

`Parser → JSON AST → JSONValidator → CCodeGenerator → C11`

- `main.py` — demonstration pipeline: reads `examples/main.oc` and writes parsed JSON and generated C.
- `src/parser.py` — parses the language, types, functions, indexing, and `array`/`tensor` literals.
- `src/debug.py` — validates the JSON AST.
- `src/compiler.py` — compatibility import for the public `CCodeGenerator` API.
- `src/codegen/` — backend mixins for types, expressions, statements, scopes, ownership, containers, and OOP.
- `src/codegen/array_codegen.py` — uniquely owned contiguous `array[T]`.
- `src/codegen/tensor_codegen.py` — contiguous row-major `tensor[T]` with shape, strides, and bounds checks.
- `tests/` — pytest tests for code generation; `docs/` — handoff notes and the memory model description.

## Quick Start

Create or activate a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if requirements.txt is present
```

Run the full test suite:

```bash
pytest -v
```

Run the demonstration compiler pipeline:

```bash
python main.py
```

The command updates `examples/parsed_code.json` and `examples/generated_code.c`, then builds the C program as `examples/generated_code`. For manual generated-C checks, use C11 and strict warnings:

```bash
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined \
    examples/generated_code.c -o /tmp/ocean_generated
```

## Arrays and Tensors

`array[T]` is a one-dimensional, owned contiguous buffer:

```python
def scale(values: &mut array[float32], factor: float32) -> None:
    for i in range(len(values)):
        values[i] = values[i] * factor
    return None
```

`tensor[T]` provides N-dimensional row-major storage:

```python
var A: tensor[float32] = [[1.0, 2.0], [3.0, 4.0]]
var value: float32 = A[0, 1]
A[1, 0] = value
var rows: int = A.shape[0]
var elements: int = len(A)
```

For a dynamic shape, use the zero-filled constructor:

```python
var rows: int = 100
var cols: int = 100
var A: tensor[float32] = tensor.zeros(rows, cols)
```

`tensor.zeros(d0, d1, ...)` evaluates the shape at runtime, allocates contiguous storage, and initializes every element to zero.

A larger example with dot products, matrix multiplication, bias, and a 3D tensor is available in [examples/arrays_tensors.oc](examples/arrays_tensors.oc). Run its backend test with:

```bash
./.venv/bin/pytest -v tests/test_array_tensor.py
```

## Borrowing and Ownership

`array[T]` and `tensor[T]` own their allocated storage and are released automatically at the end of the scope. `&T` parameters are immutable borrows; `&mut T` parameters are exclusive mutable borrows. Borrow paths do not add ARC retain/release operations to generated C.

Direct C calls use `@` and remain an unsafe FFI boundary:

```python
cimport <math.h>

def main() -> float:
    return @sqrt(16.0)
```

## Legacy Examples Using Current Patterns

### Matrix: `list[list[int]]` → `tensor[float32]`

The old `Matrix` class from `examples/main.oc` is no longer needed for numerical code. Storage and shape now belong to the tensor, while computational functions receive borrowed parameters:

```python
def matmul(A: &tensor[float32], B: &tensor[float32], C: &mut tensor[float32]) -> None:
    var rows: int = A.shape[0]
    var shared: int = A.shape[1]
    var cols: int = B.shape[1]

    for i in range(rows):
        for j in range(cols):
            var total: float32 = 0.0
            for k in range(shared):
                total = total + A[i, k] * B[k, j]
            C[i, j] = total

    return None


def main() -> int:
    var A: tensor[float32] = [[1.0, 2.0], [3.0, 4.0]]
    var B: tensor[float32] = [[5.0, 6.0], [7.0, 8.0]]
    var C: tensor[float32] = [[0.0, 0.0], [0.0, 0.0]]
    matmul(A, B, C)
    print(C[0, 0], C[1, 1])
    return 0
```

### One-Dimensional List → Owned `array`

For a numeric vector, use `array[T]` instead of the general list runtime. The function mutates the buffer through an exclusive borrow:

```python
def scale(values: &mut array[float32], factor: float32) -> None:
    for i in range(len(values)):
        values[i] = values[i] * factor
    return None


def dot(left: &array[float32], right: &array[float32]) -> float32:
    var result: float32 = 0.0
    for i in range(len(left)):
        result = result + left[i] * right[i]
    return result


def main() -> int:
    var values: array[float32] = [1.0, 2.0, 3.0]
    var weights: array[float32] = [0.5, 1.0, 1.5]
    scale(values, 2.0)
    print(dot(values, weights))
    return 0
```

### Legacy Basic Examples

Input and loops remain simple, while computational data is best passed through typed borrows:

```python
def main() -> int:
    var name: str = input("Enter your name: ")
    var values: array[float32] = [1.0, 2.0, 3.0]
    scale(values, 2.0)
    print("Hello, ", name, values[0])
    return 0
```

C/POSIX calls remain explicitly separated from safe code with `@`:

```python
cimport <math.h>

def main() -> float:
    var result: float32 = @sqrt(16.0)
    return result
```

## Development

Use four spaces in Python code, `snake_case` for functions, and `PascalCase` for classes. Add new backend passes to the appropriate `src/codegen/` module, and add a focused pytest regression test for ownership or generated-C changes. Generated Ocean symbols use the `ocean_` prefix.
