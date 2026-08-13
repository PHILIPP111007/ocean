# Handoff — Phils Language / Ocean backend

## 1. Цель проекта

Разрабатывается язык **Phils** — компилируемый системный язык с Python-подобным синтаксисом и C-like производительностью.

Основные цели:

- простой Python-like синтаксис;
- компиляция в C;
- высокая производительность;
- memory safety;
- хороший C ABI / FFI;
- пригодность для:
  - обычного прикладного программирования;
  - OS/system programming;
  - HPC;
  - ML / нейронных сетей.

Целевая идея:

> Python-like systems language with automatic ownership management and zero-cost abstractions for performance-critical code.

---

# 2. Текущая архитектура

Изначально `CCodeGenerator` был монолитным классом примерно на 8000 строк.

Он был разбит на модули:

```text
src/codegen/
├── __init__.py
├── generator.py
├── core.py
├── scope.py
├── types.py
├── orchestrator.py
├── statements.py
├── calls.py
├── indexing.py
├── io.py
├── expressions.py
├── list_codegen.py
├── tuple_codegen.py
├── dict_codegen.py
├── helpers.py
├── imports.py
├── oop.py
└── ownership.py
```

Публичный API сохранён:

```python
from src.compiler import CCodeGenerator

generator = CCodeGenerator()
output = generator.generate_from_typed_ir(typed_module)
```

Добавлен compatibility-wrapper `src/compiler.py`.

---

# 3. Namespace generated C

Принято решение использовать единый префикс:

```text
ocean_
```

для всех внутренних типов и функций runtime Phils.

Например:

```c
ocean_object_header

ocean_list_int
ocean_dict_str_int
Tensor runtime handle

ocean_retain()
ocean_release()

ocean_create_list_int()
ocean_append_list_int()
```

При этом C/POSIX ABI не переименовывается:

```c
malloc
free
printf
sqrt
pthread_create
memcpy
```

остаются стандартными именами.

`main` также остаётся:

```c
int main(...)
```

---

# 4. Memory model

Принята гибридная модель:

```text
VALUE
    int
    float
    bool
    struct

OWNED
    array[T]

SHARED
    list[T]
    dict[K, V]
    str
    class
    Tensor[T]

BORROWED
    &T
    &mut T

RAW
    *T
    void*
    C FFI
```

---

## VALUE

Типы хранятся inline / stack:

```python
var x: int = 10
var y: float = 2.0
```

Никакого refcount.

---

## SHARED

Высокоуровневые Python-like объекты используют ARC:

```python
var a: list[int] = [1, 2, 3]
var b: list[int] = a
```

`a` и `b` ссылаются на один объект.

Backend должен генерировать:

```c
ocean_retain(a);
b = a;
```

При завершении lifetime:

```c
ocean_release(a);
```

---

## OWNED

Для performance-oriented объектов:

```python
array[T]
```

предполагается unique ownership.

Для них **не должен использоваться refcount в обычном hot path**.

Это особенно важно для OS / HPC / ML.

---

## BORROWED

Безопасные временные ссылки:

```python
&T
&mut T
```

### Immutable borrow

```python
def read(x: &array[float32]) -> float:
    return x[0]
```

Не делает:

```text
retain
release
copy
allocation
```

### Mutable borrow

```python
def scale(
    x: &mut array[float32],
    factor: float32
) -> None:
    ...
```

`&mut` должен быть exclusive.

Пока он активен, нельзя использовать владельца напрямую.

---

# 5. ARC runtime

Введена концепция общего header:

```c
typedef struct ocean_object_header {
    size_t refcount;
    void (*destroy)(void*);
} ocean_object_header;
```

Общие операции:

```c
ocean_retain(obj);
ocean_release(obj);
```

Shared-типы должны встраивать этот header.

Например:

```c
typedef struct ocean_list_int {
    ocean_object_header header;

    int* data;
    size_t size;
    size_t capacity;
} ocean_list_int;
```

---

# 6. Ownership контейнеров

Для reference элементов:

```python
var d: list[int] = [1, 2]

var f: list[list[int]] = []

f.append(d)
f.append(d)
```

контейнер обязан делать retain элемента.

Поэтому две ссылки на `d` внутри `f` больше не должны приводить к double-free.

Правило:

```text
append reference      -> retain
insert reference      -> retain
set reference         -> retain(new), release(old)

remove                -> release
clear                 -> release all
destroy               -> release all
```

`pop()` должен **передавать ownership** результата вызывающему.

---

# 7. `del`

`del` больше не должен быть обязательным для освобождения памяти.

Например:

```python
def foo() -> None:
    var a: list[int] = [1, 2, 3]
```

compiler автоматически освобождает `a` на scope exit.

```python
del a
```

становится explicit early-release.

---

# 8. Borrow checker

Реализуется гибридный lexical borrow checker.

Поддерживаемая модель:

```python
var a: list[int] = [1, 2, 3]

var x: &list[int] = a
```

`x` — immutable borrow.

Пока он существует:

```text
read a       OK
modify a     forbidden
del a        forbidden
```

Для:

```python
var x: &mut list[int] = a
```

borrow exclusive.

Пока `x` существует:

```text
x read       OK
x modify     OK

a read       forbidden
a modify     forbidden
a delete     forbidden
```

Это пока **lexical checker v1**, не полноценный Rust NLL/data-flow checker.

---

# 9. Threading

Пока обычный ARC предполагается non-atomic.

В будущем:

```python
shared[T]
```

или отдельный `Shared[T]` должен использовать atomic reference counting.

Дальше можно добавить аналоги:

```text
Send
Sync
```

для compile-time проверки передачи объектов между pthreads.

---

# 10. C interop

Текущий синтаксис:

```python
@sqrt(16)
@pthread_create(...)
```

остаётся прямым C ABI.

В перспективе raw C operations должны считаться unsafe boundary.

Например:

```python
unsafe:
    @malloc(...)
```

или через FFI contracts.

---

# 11. Parser

Исходный parser — крупный `Parser`, порядка 6500 строк / 87 методов.

Он поддерживает:

- variables;
- constants;
- list;
- dict;
- tuple;
- set;
- classes;
- inheritance;
- methods;
- loops;
- if/elif/else;
- C calls;
- raw pointers;
- address-of;
- dereference;
- indexing;
- nested indexing;
- slicing;
- function calls;
- C imports.

Исходный parser был прочитан и использован как основа.

---

# 12. Новый parser v0.2

Был подготовлен обновлённый parser:

```text
phils_parser_ocean_v02.zip
```

Он добавляет отдельный type parser:

```text
src/parsing/type_system.py
```

Типы больше не должны полностью анализироваться вручную через:

```python
startswith("list[")
```

---

## Новый type syntax

Parser теперь должен понимать:

```python
list[int]

dict[str, int]

tuple[int]

array[float32]

Tensor[float32]

shared[list[int]]

&list[int]

&mut list[int]

*int

str?
```

При этом старое поле сохраняется:

```text
"var_type": "&mut list[int]"
```

но добавляется structured metadata:

```text
"type_info": {
    "kind": "mut_borrow",
    "memory_kind": "mut_borrow",
    ...
}
```

---

# 13. `&x` vs borrow

В старом языке:

```python
var p: *int = &x
```

означает C address-of.

Это поведение сохраняется.

То есть:

```python
*int
```

+ expression:

```python
&x
```

остаются RAW pointer semantics.

Safe borrow записывается через тип:

```python
var p: &list[int] = values
```

или:

```python
var p: &mut list[int] = values
```

---

# 14. Struct

В parser добавлена концепция value struct:

```python
struct Point:
    x: float32
    y: float32
```

Целевой lowering:

```c
typedef struct ocean_point {
    float x;
    float y;
} ocean_point;
```

Без:

```text
malloc
ARC
vtable
```

Это особенно важно для OS/system programming.

---

# 15. Classes

Классы остаются reference objects.

Идея:

```c
typedef struct ocean_user {
    ocean_object_header header;
    const ocean_user_vtable* vtable;

    ...
} ocean_user;
```

Per-object malloc для vtable должен быть убран.

Vtable должна быть static per class.

---

## Multiple inheritance

Текущая старая реализация multiple inheritance небезопасна.

Она физически embedding делает только первого base class, но может кастовать объект ко второму.

Поэтому принято решение:

> временно запретить multiple inheritance.

Для v0.x оставить:

```text
single inheritance
```

Позже добавить:

```text
traits/interfaces
```

или корректный ABI для multiple bases.

---

# 16. Strings

Текущие строки всё ещё требуют дальнейшей унификации.

Старая реализация смешивает:

```text
static literal
malloc string
input string
string helper result
```

под одним `char*`.

Это опасно.

Долгосрочная цель:

```c
ocean_string
```

с чёткой ownership semantics.

Для C FFI выдавать borrowed:

```c
const char*
```

---

# 17. Bounds safety

Memory-safe язык не должен отключать bounds checks через:

```c
#ifndef NDEBUG
```

Правило:

```text
bounds check присутствует всегда
```

если compiler **не доказал**, что индекс безопасен.

Позже:

```text
Bounds Check Elimination
```

может убрать проверки из hot loops.

---

# 18. SIMD

Старый generic SIMD-copy был удалён/должен быть удалён.

Причина: он предполагал 4-byte element size и был некорректен для:

```text
double
64-bit pointers
nested lists
class references
```

План:

```text
generic containers -> memcpy/compiler auto-vectorization

array and Tensor numeric types ->
type-specific SIMD
```

---

# 19. Demand-driven runtime

Была найдена проблема: даже простая программа:

```python
@sqrt(16)
```

генерировала десятки ненужных runtime helpers:

```text
list[str]
string helpers
sorting helpers
ARC
...
```

Это было исправлено концептуально в backend v0.2.2.

Runtime должен генерироваться только когда реально нужен:

```text
math-only program
    -> zero Ocean heap runtime

list
    -> list runtime

dict
    -> dict runtime

str methods
    -> string runtime

ARC type
    -> ocean_retain/release
```

---

# 20. Тесты

Существующие тесты в основном golden:

```python
assert generated_c == expected_c
```

После перехода:

```text
list_int
```

→

```text
ocean_list_int
```

и появления automatic cleanup многие старые golden expected-C автоматически устарели.

Поэтому предложено разделить тестирование.

---

## Level 1 — AST / parser

Проверять структуру AST.

---

## Level 2 — C generation

Проверять ключевые конструкции:

```python
assert "ocean_list_int" in output
```

---

## Level 3 — compile

```bash
gcc \
    -std=c11 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Werror
```

---

## Level 4 — memory safety

```bash
-fsanitize=address,undefined
-fno-omit-frame-pointer
```

и реальное выполнение generated binary.

---

## Обязательные memory tests

Добавить:

```text
alias list
nested alias
double append same child
del original while aliases exist
list[str]
self-assignment
return ownership transfer
pop ownership transfer
borrow + delete
borrow + mutation
&mut exclusivity
scope cleanup
nested scope cleanup
```

---

# 21. Array — принятое устройство

`array[T]` — performance/system container.

Он отличается от `list[T]`.

### list

```text
dynamic
Python-like
append/pop
shared
ARC
```

### array

```text
contiguous
fixed-size или controlled resize
unique ownership
NO ARC
SIMD-friendly
OS-friendly
ML-friendly
```

Целевая структура:

```c
typedef struct ocean_array_float32 {
    float* data;
    size_t size;
} ocean_array_float32;
```

или при необходимости:

```c
typedef struct ocean_array_float32 {
    float* data;
    size_t size;
    size_t capacity;
} ocean_array_float32;
```

---

# 22. Tensor — принятое устройство

`Tensor[T]` — публичный N-dimensional dense row-major объект с managed runtime handle.

Он **не должен быть `list[list[T]]`**.

Физически данные должны храниться одним contiguous buffer.

Например:

```python
var A: Tensor[float32] = [
    [1.0, 2.0],
    [3.0, 4.0]
]
```

должен соответствовать:

```text
data:
[1.0, 2.0, 3.0, 4.0]

shape:
[2, 2]

strides:
[2, 1]

ndim:
2

size:
4
```

Целевая структура:

```c
Tensor хранит opaque runtime handle; layout и backend details остаются внутри
`std/tensor/tensor_runtime`.
```

---

# 23. Array vs Tensor

```text
array[T]
    1D contiguous buffer

Tensor[T]
    N-D abstraction over contiguous storage
```

Tensor добавляет:

```text
shape
strides
ndim
multi-dimensional indexing
reshape
transpose/view semantics
ML operations
```

---

# 24. Tensor indexing

Принято решение поддержать естественный синтаксис:

```python
A[i, j]
```

а не использовать:

```python
A[i][j]
```

как для nested lists.

Для row-major Tensor:

```text
offset =
    i * strides[0] +
    j * strides[1]
```

и затем:

```c
A->data[offset]
```

---

# 25. Tensor metadata

Планируемый интерфейс:

```python
A.shape
A.size
A.ndim
```

Также:

```python
A.shape[0]
A.shape[1]
```

Metadata lookup не должен копировать tensor data.

---

# 26. Tensor ownership

Tensor должен быть managed публичным объектом:

```text
SHARED
```

а не ARC-managed по умолчанию.

Например:

```python
var A: Tensor[float32] = ...
```

владеет storage.

Функция:

```python
def forward(
    A: &Tensor[float32]
) -> None:
```

получает borrow:

```text
retain = 0
release = 0
copy = 0
allocation = 0
```

Функция:

```python
def normalize(
    A: &mut Tensor[float32]
) -> None:
```

получает exclusive mutable borrow.

---

# 27. ML target

Целевой API:

```python
def matmul(
    A: &Tensor[float32],
    B: &Tensor[float32],
    C: &mut Tensor[float32]
) -> None:

    var M: int = A.shape[0]
    var K: int = A.shape[1]
    var N: int = B.shape[1]

    for i in range(M):
        for j in range(N):

            var total: float32 = 0.0

            for k in range(K):
                total += A[i, k] * B[k, j]

            C[i, j] = total
```

Целевой lowering:

```c
void ocean_matmul(
    const Tensor* restrict A,
    const Tensor* restrict B,
    Tensor* restrict C
)
```

Внутри compute loop не должно быть ARC.

---

# 28. Статус array/tensor на момент handoff

### Parser

Уже подготовлена поддержка AST для:

```text
array[T]
Tensor[T]
```

в parser v0.2.

Parser умеет концептуально сохранять:

```text
element_type
ownership
shape
rank
is_rectangular
```

для Tensor construction и array literals.

### Backend

Lowering `array` и публичного `Tensor` поддерживается; удалённый native
старый native tensor-тип больше не является частью языка.

Последний запрос перед handoff был:

> реализовать `array` и публичный `Tensor`.

Начат план следующей backend-итерации:

```text
array[T]
    unique-owned contiguous storage
    creation
    indexing
    mutation
    len
    cleanup
    borrowing

Tensor[T]
    contiguous row-major storage
    shape
    strides
    ndim
    size
    A[i, j]
    mutation
    automatic cleanup
    &T / &mut T
```

Именно с этого надо продолжить следующую сессию.

---

# 29. Что НЕ делать дальше

Не следует:

1. делать Tensor через `list[list[T]]`;
2. добавлять лишние копирования в Tensor hot path;
3. делать generic handwritten SIMD для любых T;
4. возвращаться к одному 8000-line `CCodeGenerator`;
5. добавлять новые type semantics непосредственно во время C emission;
6. продолжать unsafe multiple inheritance;
7. считать raw C FFI memory-safe;
8. отключать bounds checks через release/NDEBUG.

---

# 30. Следующий рекомендуемый этап

Продолжить с реализации:

```text
src/codegen/array_codegen.py
src/codegen/tensor_codegen.py
```

и добавить их в:

```python
class CCodeGenerator(...)
```

Нужно реализовать в таком порядке:

### Array

```text
1. type mapping
2. runtime struct
3. literal creation
4. get/set
5. bounds checks
6. len
7. scope cleanup
8. assignment/move semantics
9. &array[T]
10. &mut array[T]
```

### Tensor

```text
1. Tensor runtime handle
2. contiguous literal flattening
3. shape generation
4. stride generation
5. ndim
6. size
7. A[i, j]
8. multidimensional checked offset
9. set
10. shape access
11. borrow
12. cleanup
13. reshape/view
```

После этого:

```text
matmul benchmark
↓
bounds-check elimination
↓
restrict/noalias analysis
↓
SIMD
↓
BLAS
↓
CUDA backend/interoperability
```

---

# 31. Созданные артефакты

Compiler modular refactor:

`phils_codegen_refactor.zip`

Ocean ownership backend:

`phils_codegen_ocean_v02.zip`

Logging fix:

`phils_codegen_ocean_v021.zip`

Demand-driven helpers iteration:

`phils_codegen_ocean_v022.zip`

Parser v0.2:

`phils_parser_ocean_v02.zip`

---

# 32. Главная архитектурная цель

Итоговая модель Phils:

```text
Python-like syntax
        ↓
Parser
        ↓
Typed AST / HIR
        ↓
Semantic analyzer
        ↓
Automatic ownership management
        ↓
Borrow checking
        ↓
Safety optimizations
        ↓
C backend
        ↓
clang/gcc
```

Семантика памяти:

```text
Value
    ↓
zero-cost

Owned array / managed Tensor
    ↓
unique ownership
    ↓
zero refcount

Shared list/dict/class
    ↓
ARC

&T / &mut T
    ↓
zero-cost borrow

*T / C
    ↓
unsafe/raw
```

Главная цель производительности:

> В OS/ML hot paths Phils должен уметь генерировать C без hidden allocation, ARC и копирования, чтобы итоговый код мог оптимизироваться clang/gcc примерно на уровне C/Rust.
