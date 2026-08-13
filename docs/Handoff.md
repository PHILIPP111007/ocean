# Handoff — Phils Language / Ocean backend

> Обновлено: 2026-08-13. Этот документ описывает фактическое состояние
> репозитория после перехода на Typed IR, удаления legacy tensor/JSON-пути,
> добавления OpenMP/OpenCL, File IO и NumPy `.npy`. Исторические разделы ниже
> сохранены для контекста; актуальный статус и следующий план находятся в
> разделах 33–39.

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

# 28. Исторический статус array/tensor до завершения backend-перехода

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

# 29. Устойчивые архитектурные ограничения

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

# 30. Устаревший план до завершения Tensor backend

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

---

# 33. Актуальная архитектура компилятора

Текущий основной pipeline:

```text
Ocean source → Parser → TypedModule / Typed IR → validation
    → ownership/borrow analysis → structured diagnostics
    → CCodeGenerator → generated C11 → gcc/clang
```

`TypedModule` является главным API backend’а:

```python
typed_module = Parser().parse_typed(source)
c_code = CCodeGenerator().generate_from_typed_ir(typed_module)
```

Основные компоненты:

- `src/parser.py` — синтаксис и построение типизированного модуля;
- `src/typed_ir.py` — типы, зависимости, reads/writes и ownership effects;
- `src/debug.py` — validation, borrow/move checks и внешние C symbols;
- `src/diagnostics.py` — typed diagnostics с location и стабильными кодами;
- `src/codegen/` — lowering в C11;
- `src/compiler.py` — сохранённый public compatibility API;
- `std/` — стандартная библиотека и C runtime.

Внутреннее представление compiler pipeline больше не сериализуется в JSON.
Совместимость сохранена только для legacy dictionary projection diagnostics и
mapping-представлений Typed IR. JSON-файлы Graphify являются артефактами
инструмента анализа кода, а не частью compiler/runtime IR.

Импорты разделены по назначению:

```text
import "./examples/matmul.oc"       # относительно импортирующего файла
import <std/tensor/tensor.oc>        # из ./std/
```

Package CLI поддерживает `init`, `check`, `build`, `run`, `test`, `clean`.
Артефакты package build находятся в `build/<profile>/`; legacy single-file
workflow остаётся доступным при передаче исходного `.oc` файла.

# 34. Актуальная модель памяти

```text
scalar value       → plain C value
list/dict/class    → non-atomic ARC
str                → owned C string semantics
array[T]           → unique-owned numeric buffer
Tensor[T]          → ARC-managed facade over opaque runtime handle
File/BinaryFile    → ARC-managed facade over opaque FILE handle
&T / &mut T        → lexical zero-cost borrow
*T / C calls       → explicit unsafe boundary
```

Реализованы детерминированный cleanup managed объектов при выходе из scope,
`del` как раннее освобождение, ownership effects в Typed IR, move/use-after-move
checks, immutable/mutable lexical borrows, retain/release элементов list/dict,
ownership transfer через `pop` и return paths, а также bounds checks,
не зависящие от `NDEBUG`.

Ограничения v1:

- ARC неатомарен и рассчитан на thread-confined managed objects;
- полноценные NLL/lifetime parameters ещё не реализованы;
- `Send`/`Sync`, `Shared[T]`, arenas и allocator API отсутствуют;
- циклы ARC могут приводить к leak;
- raw C pointers и произвольные C ownership contracts не анализируются;
- multiple inheritance запрещён, безопасная модель пока single inheritance.

# 35. OpenMP и параллельные циклы

Поддерживается ограниченный безопасный subset:

```text
#pragma omp parallel for collapse(2) schedule(static)
for i in range(rows):
    for j in range(cols):
        output[i, j] = left[i, j] + right[i, j]
```

Допускается совместимое написание `#pragma opm`; в C всегда генерируется
правильный `#pragma omp`. Поддерживаются clauses `schedule`, `collapse`,
`reduction`, `private`, `firstprivate`, `lastprivate`, `shared`, `default`,
`nowait`, `ordered`.

Для `collapse(n)` циклы должны быть идеально вложенными, с постоянным
ненулевым integer step. Managed objects, вызовы функций, `break`, `continue`
и небезопасные формы вложенности отклоняются validator’ом. Если generated C
содержит OpenMP pragma, CLI автоматически добавляет `-fopenmp`.

# 36. Tensor backend: фактическое состояние

`Tensor[T]` — единственный публичный tensor type. Старый отдельный lowercase
`tensor` больше не является частью публичного API. Пользовательский код не
видит `cl_mem`, OpenCL context, queue или CPU pointers.

Основной API:

```text
Tensor.zeros(..., device)
Tensor.from_list(..., device)
Tensor.load_npy(path, device)
tensor.save_npy(path)
tensor.to(device), tensor.copy(), tensor.matmul(other)
tensor.add/sub/mul/div(...), tensor.reshape/transpose/row/column/slice(...)
tensor.sum/mean/min/max/item(), tensor.shape(axis), tensor.ndim()
tensor.size(), tensor.device(), tensor.get/set/fill()
```

Поддерживаются numeric dtypes: `bool`, signed/unsigned integers,
`float16`, `float32`, `float64`. `Tensor[str]` отклоняется. Устройства:
`"cpu"` и `"gpu"` (OpenCL).

Runtime использует backend operation table. OpenCL context, queue, program и
kernels создаются лениво и кэшируются по процессу; kernel cache разделён по
операции и dtype. Очередь in-order, операции flush’ятся без лишних глобальных
barrier, host reads ждут собственные read events. OpenCL events освобождаются
после постановки команды.

CPU `float32`/`float64` 2D matmul имеет contiguous row-major `i-k-j` fast path.
OpenCL kernels есть для `float32` и `int32`; остальные numeric dtypes используют
корректный CPU fallback с переносом результата на исходное устройство.

Для одноиндексного scalar iteration:

```text
len(tensor)  → число скалярных элементов
tensor[i]    → плоский row-major элемент
tensor[i, j] → строгий многомерный доступ с проверкой rank
```

Это позволяет загружать веса и проходить их линейным циклом. `Tensor` пока не
имеет zero-copy view semantics: `reshape`, `transpose`, `row`, `column` и
`slice` материализуют независимое contiguous storage.

# 37. File и BinaryFile

Добавлен стандартный импорт:

```text
import <std/io/file.oc>
```

`File` предоставляет `read`, `readline`, `readlines`, `write`, `writelines`,
`flush`, `eof`, `close`.

`open_binary(path, mode)` возвращает `BinaryFile` с методами `read_byte`,
`read_bytes`, `write_byte`, `write_bytes`, `flush`, `eof`, `close`.

Runtime находится в `std/io/file_runtime.h` и `std/io/file_runtime.c`.
В safe Ocean code проходят opaque handles; `FILE*` напрямую не экспортируется.
File handles закрываются явно или при уничтожении владеющего Ocean объекта.

# 38. NumPy `.npy`

Добавлены методы `Tensor.load_npy(path, device)` и `tensor.save_npy(path)`.
Reader/writer реализован напрямую в C runtime без зависимости от NumPy.

Поддерживаются `.npy` v1.0, v2.0 и v3.0, little/big endian и numeric
descriptors `bool`, `int8/int16/int32/int64`, `uint8/uint16/uint32/uint64`,
`float16/float32/float64`.

Writer использует v1, пока header помещается в 16-bit length field, иначе v2.
Данные сохраняются в C-order row-major виде. При сохранении GPU Tensor сначала
скачивается на CPU; non-contiguous storage материализуется перед записью.

Reader проверяет magic/version/header, извлекает `descr`, `fortran_order` и
`shape`, читает raw payload через `fread`, при необходимости делает endian
byte-swap и затем переносит Tensor на запрошенное устройство.

Пока отклоняются Fortran-order, object/string/structured dtypes и scalar arrays.
`.npy` не сжимается и не хранит quantization metadata. `Tensor[int8]` уже
поддерживается как компактный dtype, но настоящие `scale`/`zero_point`
quantization API ещё не реализованы.

# 39. Примеры и проверки

ML/OOP примеры:

- `examples/transformer_pytorch.py` — PyTorch reference implementation;
- `examples/transformer_ocean.oc` — OOP Transformer-like implementation;
- `examples/matmul.oc` — CPU Tensor/matmul;
- `examples/matmul_gpu.oc` — OpenCL Tensor path;
- `examples/load_npy.oc` — загрузка и линейная итерация `.npy` weights;
- `examples/openmp.oc` — OpenMP loops;
- `examples/neural_network.oc` — ML-oriented language example.

Последняя проверенная база:

```text
pytest -q                 → 109 passed
python main.py check --quiet
python main.py build --quiet
git diff --check
```

Runtime C проверяется также строгой компиляцией:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic -I. \
    -c std/tensor/tensor_runtime.c
gcc -std=c11 -Wall -Wextra -Wpedantic -I. \
    -c std/io/file_runtime.c
```

Для OpenCL необходимо передавать include directory с `CL/cl.h`, library
directory с `libOpenCL.so`, `-lOpenCL` и
`-DOCEAN_TENSOR_ENABLE_OPENCL`, например:

```bash
ocean run ./examples/matmul_gpu.oc \
  --cflags "-I${CONDA_PREFIX}/include \
-L/usr/local/cuda/targets/x86_64-linux/lib \
-lOpenCL -DOCEAN_TENSOR_ENABLE_OPENCL"
```

Graphify после последнего изменения кода:

```text
1468 nodes
3159 edges
89 communities
```

Обновлять graph artifacts:

```bash
./.venv/bin/graphify update .
```

# 40. Текущие ограничения и ближайший roadmap

Приоритет P0 — завершить compiler foundation:

1. Убрать оставшиеся прямые AST обходы из backend в пользу Typed IR.
2. Добавить `Result[T, E]`, `Option[T]` и `defer` вместо process-exit для обычных ошибок.
3. Улучшить diagnostics для rank/type/ownership ошибок, включая точные source spans.
4. Довести interprocedural borrow/data-flow checks и явные move diagnostics.

Приоритет P1 — Tensor performance:

1. Ввести zero-copy Tensor views через `offset`, `shape`, `strides`, `owns_data`.
2. Добавить memory pool для повторного использования CPU/GPU buffers.
3. Реализовать kernel fusion для цепочек elementwise операций.
4. Добавить SIMD CPU backend и benchmark suite с `-O2`/`-O3`.
5. Сделать async Tensor events публичным безопасным API без ручного OpenCL доступа.

Приоритет P2 — compact ML weights:

1. Реализовать affine `int8` quantization: `scale` и `zero_point`.
2. Добавить per-channel quantization для Linear/Conv weights.
3. Использовать стандартный `.npz` для `data`, `scale`, `zero_point`, не вводя
   собственный checkpoint format.
4. Позже добавить packed `int4`/`int2` storage.

Приоритет P3 — язык и backend:

1. Traits/interfaces: `Numeric`, `Readable`, `Writable`, `Backend`, `Allocator`.
2. Value generics/comptime для dtype, rank, layout и tile sizes.
3. Явные allocator’ы и arena lifetime для системного кода.
4. `Send`/`Sync`-подобные ограничения для pthread/OpenMP объектов.
5. Дополнительные backend’ы: SIMD, CUDA/внешний BLAS, затем возможно LLVM/MLIR.
6. `ocean fmt`, `ocean lint`, LSP, profiler и incremental compilation cache.

Целевая ниша Ocean — не полная замена Rust, Zig или Mojo, а их практический
пересекающийся слой:

```text
Python-like syntax
+ lexical ownership/borrows
+ explicit C ABI and unsafe boundary
+ comptime specialization
+ Tensor/OpenCL/ML standard library
```

Не следует возвращаться к Tensor через `list[list[T]]`, отключать bounds checks
через `NDEBUG`, добавлять handwritten generic SIMD без dtype/layout доказательств
или снова смешивать backend semantics с C emission.
