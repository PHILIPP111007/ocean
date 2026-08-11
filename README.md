# Ocean 🌊

<img src="images/ocean.jpg" alt="Ocean project illustration" width="500"/>

Ocean is a Python-like language that compiles to C11. The project targets systems programming and ML-oriented computation, with explicit borrow parameters and ownership checks during code generation.

## Memory Model Overview

Ocean uses a hybrid model: automatic ownership, reference counting, and static borrow checks.

### Data Types

- `int`, `float`, `bool` — regular value types without memory management.
- `list[T]`, `dict[K, V]`, `tuple[T]`, class instances — ARC objects with reference counting.
- `str` — a dedicated C string, copied when creating an alias.
- `array[T]`, `tensor[T]` — unique-owned buffers with separate `free`.
- `&T` — immutable borrow.
- `&mut T` — exclusive mutable borrow.
- raw C pointers — completely unsafe ownership.

### Reference Counting

An ARC object has a header:

```c
typedef struct ocean_object_header {
    size_t refcount;
    void (*destroy)(void*);
} ocean_object_header;
```

When creating an object:

```text
refcount = 1
```

When creating a new alias, the following is called:

```c
ocean_retain(object);
```

When the owner leaves the scope:

```c
ocean_release(object);
```

When the counter reaches zero, the object's destructor is called.

```python
var a: list[int] = [1, 2, 3]
var b: list[int] = a
```

`a` and `b` point to the same object, but each owner maintains its own reference.

### Arrays and Tensors

`array[T]` and `tensor[T]` do not use ARC. They have a single owner:

```python
var matrix: tensor[float32] = tensor.zeros(100, 100)
```

When the scope ends, the generator calls:

```c
ocean_tensor_float32_free(matrix);
```

Assigning an owned value usually transfers ownership:

```python
var first: tensor[float32] = tensor.zeros(10, 10)
var second: tensor[float32] = first
```

After moving, `first` can no longer be used.

### Borrowing

Borrow does not increase the reference count or free memory:

```python
def read(matrix: &tensor[float32]) -> float32:
    return matrix[0, 0]
```

`&mut` allows modification:

```python
def clear(matrix: &mut tensor[float32]) -> None:
    matrix[0, 0] = 0.0
    return None
```

The compiler prohibits deleting or modifying the object while borrowing is active.

### Manual Management

For explicit deletion, use:

```python
del value
```

This is usually not necessary: owned objects are automatically freed at the end of the scope. Direct C calls via `@` are only permitted inside an explicit `unsafe:` block and may violate ownership rules.

Reference counting in the current version is non-atomic, so ARC objects are intended primarily for thread-confined use.

Details: [docs/MEMORY_MODEL.md](docs/MEMORY_MODEL.md).

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

Direct C calls use `@` and must be placed in an explicit `unsafe:` block:

```python
cimport <math.h>

def main() -> float:
    unsafe:
        var result: float = @sqrt(16.0)
    return result
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

C/POSIX calls remain explicitly separated from safe code with `unsafe:` and `@`:

```python
cimport <math.h>

def main() -> float:
    unsafe:
        var result: float32 = @sqrt(16.0)
    return result
```

## Development

Use four spaces in Python code, `snake_case` for functions, and `PascalCase` for classes. Add new backend passes to the appropriate `src/codegen/` module, and add a focused pytest regression test for ownership or generated-C changes. Generated Ocean symbols use the `ocean_` prefix.
