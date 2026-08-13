# Phils / Ocean C backend v0.2

This iteration turns the modular C backend into the first ownership-aware backend for Phils.
The public Python API remains:

```python
from src.codegen import CCodeGenerator
from src.parser import Parser

typed_module = Parser().parse_typed(source)
c_code = CCodeGenerator().generate_from_typed_ir(typed_module)
```

`TypedModule` is the canonical compiler-pass API. The backend no longer accepts
serialized parser data.

The C backend receives `TypedScope` and `TypedNode` mapping views from
`TypedModule.backend_scopes()`. Their legacy `.get(...)` read interface is kept
inside the lowering mixins while the parser's mutable scope list is no longer
passed through the main backend pipeline.

## What changed

### 1. `ocean_` C namespace

Every **Phils-generated C type and function** is namespaced with `ocean_` after lowering.
External C/POSIX symbols are deliberately not renamed, so C ABI interop keeps working.

Examples:

```text
list_int                 -> ocean_list_int
create_list_int          -> ocean_create_list_int
append_list_int          -> ocean_append_list_int
Box                      -> ocean_Box
create_Box               -> ocean_create_Box
Box_get                  -> ocean_Box_get
helper_function          -> ocean_helper_function
main                     -> main
pthread_create / malloc  -> unchanged external C symbols
```

User/local C variable names are not globally prefixed; only emitted Phils symbols are namespaced.

### 2. Automatic ownership management

Managed heap objects use a common runtime header:

```c
typedef struct ocean_object_header {
    size_t refcount;
    void (*destroy)(void*);
} ocean_object_header;
```

`list`, `dict`, homogeneous `tuple`, and class instances are ARC-managed. The backend emits
`ocean_retain()` and `ocean_release()` according to ownership, not ad-hoc `free()` calls.

Strings are currently uniquely-owned C buffers: aliases are duplicated with `ocean_strdup()`.
This deliberately avoids mixing static string literals with owned heap strings.

### 3. Hybrid borrow checker v1

The backend understands these type spellings in the AST:

```text
&T
&mut T
```

The C backend enforces the rules lexically and with zero runtime overhead. The validator adds a
conservative intra-function data-flow pass, including branch/loop state merging:

- `&T` allows multiple immutable borrows;
- `&mut T` is exclusive;
- an owner cannot be mutated or deleted while borrowed;
- while `&mut T` exists, the owner cannot be accessed directly;
- a borrow ends at lexical scope exit;
- deletion/move states are checked after control-flow joins;
- borrows cannot escape through returns, unsafe C calls, or incompatible function parameters;
- borrow variables do not retain/release the object;
- rebinding a borrow is rejected in v1;
- reborrowing a borrow is rejected in v1.

The backend currently enables borrows first for managed/reference objects (`list`, `dict`, class,
etc.) and immutable strings. Borrowing inline scalar/value types is intentionally deferred until
an addressable value/struct representation is added.

The parser graph is lowered into `src/typed_ir.py` before validation and code generation. Typed IR
nodes retain the legacy graph for compatibility while exposing result types, reads/writes, and
effects to semantic passes.

### 4. Deterministic scope cleanup

Owned managed objects and owned strings are automatically released at scope exit. `del` is now an
early-release/unbind operation, not a requirement for correct cleanup.

Return paths, `break`, and `continue` emit the cleanup required for scopes they leave.
Owned values returned from a function transfer ownership to the caller; borrowed managed values
are retained before return.

### 5. Ownership-aware containers

Container helpers now implement reference ownership consistently:

- `append` / `insert` retain ARC elements and duplicate strings;
- `set` retains/copies the replacement before releasing the previous value;
- `remove` / `clear` release removed elements;
- `pop` transfers the removed element's ownership to the caller;
- list destruction releases every stored managed element;
- dictionary values use the same retain/release rules;
- `keys()` / `values()` produce owned Ocean lists;
- `extend` snapshots its source first so `a.extend(a)` stays valid across reallocations.

Bounds checks in generated list/tuple access are not removed by `NDEBUG`.

### 6. Safer class lowering

Root class objects contain the Ocean object header. Derived objects use single inheritance with the
base at offset zero, so the ownership header remains at offset zero. Class destructors release
managed fields before freeing the object.

Per-instance vtable allocation was removed. Vtables remain a future dispatch feature; the current
backend keeps the field null where needed for compatibility.

Unsafe multiple inheritance from the old backend is explicitly rejected in this version instead of
emitting invalid casts/layout.

## Module layout

```text
src/codegen/
├── generator.py        public CCodeGenerator façade
├── core.py             compilation state/output buffer
├── naming.py           ocean_ symbol namespace pass
├── ownership.py        ARC + lexical borrow checker
├── scope.py            lexical scopes and ownership metadata
├── types.py            type mapping/discovery
├── orchestrator.py     compilation pipeline
├── statements.py       assignment/del/control flow/return
├── calls.py            function/method/C/builtin calls
├── expressions.py      expression lowering
├── indexing.py         indexing/slicing
├── io.py               input/string concatenation
├── list_codegen.py     list runtime generation
├── tuple_codegen.py    tuple runtime generation
├── dict_codegen.py     dict runtime generation
├── oop.py              class layout/constructors/methods
├── helpers.py          string/sort/conversion helpers
└── imports.py          includes and forward declarations
```

## Important safety boundary

This is **ownership/borrow checker v1**, not a claim that every possible Phils program is already
fully Rust-level memory-safe.

Current deliberate boundaries:

- direct C/POSIX calls are an unsafe FFI boundary; the backend cannot infer arbitrary C lifetime contracts;
- ARC is currently non-atomic and intended for thread-confined managed objects;
- cyclic ARC object graphs can leak (they do not create use-after-free);
- managed global variables are rejected in v1 until module initialization/destruction is formalized;
- heterogeneous tuples are rejected in ownership v1;
- multiple class inheritance is rejected in safe v1;
- interprocedural lifetime parameters and proven non-lexical lifetimes are not implemented yet;
- raw C pointers are not automatically freed;
- scalar/value borrows are not implemented yet;
- full integer-overflow semantics, `Shared[T]`, `Send`/`Sync`, arenas and owned `array` are future work.

For OS/ML, `array[T]` is a unique-owned buffer plus `&T` / `&mut T`, so hot loops do not perform
reference-count operations. Numerical code should use the managed opaque `Tensor[T]` facade,
which owns backend storage and exposes device-aware operations.

## Recommended compiler flags during development

```bash
gcc generated_code.c \
    -std=c11 \
    -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer \
    -o generated_code
```

For pthread code add `-pthread`.
