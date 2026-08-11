# Ocean automatic ownership model v1

## Categories

| Phils type | v1 memory model | Runtime RC |
|---|---|---:|
| `int`, `float`, `bool`, C scalar | value | no |
| `str` | unique owned buffer / copied alias | no |
| `list[T]` | managed shared object | yes |
| `dict[K,V]` | managed shared object | yes |
| `tuple[T]` | managed shared immutable object | yes |
| class instance | managed shared object | yes |
| `&T` | immutable lexical borrow | no |
| `&mut T` | exclusive mutable lexical borrow | no |
| raw C pointer | unsafe external ownership | no |

## Reference alias

Phils:

```python
var a: list[int] = [1, 2, 3]
var b: list[int] = a
```

Conceptual C:

```c
ocean_list_int* a = ocean_create_list_int(...); // refcount 1
ocean_list_int* b = a;
ocean_retain(b);                                // refcount 2
```

At scope exit each owning binding releases its reference.

## Borrow

Phils AST type spelling:

```text
&list[int]
&mut list[int]
```

Conceptually:

```c
ocean_list_int* borrowed = owner;
```

No retain/release is emitted. The validator tracks borrow state through straight-line code and
conservatively merges states across branches and loops, preventing the owner from being invalidated
while the borrow is alive.

## Return ABI

A managed return value is always an **owned reference for the caller**.

- returning a local owner transfers that owner;
- returning a borrowed parameter/index/field emits a retain first;
- returning an owned temporary transfers it without an extra retain/release pair.

## Containers

Containers own one reference to every managed element they store. Therefore the same object may be
stored multiple times without double-free:

```python
var d: list[int] = [1]
var m: list[list[int]] = [d, d]
del d
```

The two container entries each hold a reference. Destruction releases them independently.

## FFI

`@function(...)` / direct C calls do not become safe merely because surrounding Phils code is safe.
They must be placed inside an explicit lexical `unsafe:` block:

```python
def log_value(value: int) -> None:
    unsafe:
        @printf("%d\n", value)
    return None
```

Raw pointer declarations (`*T`) follow the same rule. The C ABI can retain pointers, free pointers,
access them asynchronously, or violate aliasing rules, so the validator and backend reject these
operations outside `unsafe:`. The block is an explicit escape hatch, not a proof that the enclosed C
code is memory-safe.
