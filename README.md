# Ocean 🌊

<img src="images/ocean.jpg" alt="alt text" width="500"/>

Ocean — Python-подобный язык, который компилируется в C11. Проект ориентирован на системное программирование и вычисления для ML, с явными borrow-параметрами и проверками владения на этапе генерации.

## Архитектура

Основной pipeline выглядит так:

`Parser → JSON AST → JSONValidator → CCodeGenerator → C11`

- `main.py` — демонстрационный pipeline: читает `examples/main.oc`, сохраняет parsed JSON и generated C.
- `src/parser.py` — разбор языка, типов, функций, индексации и `array/tensor` literals.
- `src/debug.py` — проверка JSON AST.
- `src/compiler.py` — совместимый публичный импорт `CCodeGenerator`.
- `src/codegen/` — backend, разделённый на mixins: типы, выражения, statements, scopes, ownership, containers и OOP.
- `src/codegen/array_codegen.py` — unique-owned contiguous `array[T]`.
- `src/codegen/tensor_codegen.py` — contiguous row-major `tensor[T]` с shape, strides и bounds checks.
- `tests/` — pytest-тесты генератора; `docs/` — handoff и описание memory model.

## Быстрый запуск

Создайте или используйте локальное окружение:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt  # если requirements.txt присутствует
```

Запустить весь тестовый набор:

```bash
./.venv/bin/pytest -v
```

Запустить демонстрационный compiler pipeline:

```bash
./.venv/bin/python main.py
```

Команда обновит `examples/parsed_code.json` и `examples/generated_code.c`, затем соберёт C-программу в `examples/generated_code`. Для ручной проверки generated C используйте C11 и строгие предупреждения:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined \
    examples/generated_code.c -o /tmp/ocean_generated
```

## Array и tensor

`array[T]` — одномерный owned contiguous buffer:

```python
def scale(values: &mut array[float32], factor: float32) -> None:
    for i in range(len(values)):
        values[i] = values[i] * factor
    return None
```

`tensor[T]` — N-dimensional row-major storage:

```python
var A: tensor[float32] = [[1.0, 2.0], [3.0, 4.0]]
var value: float32 = A[0, 1]
A[1, 0] = value
var rows: int = A.shape[0]
var elements: int = len(A)
```

Для динамической формы используйте zero-filled constructor:

```python
var rows: int = 100
var cols: int = 100
var A: tensor[float32] = tensor.zeros(rows, cols)
```

`tensor.zeros(d0, d1, ...)` вычисляет форму во время выполнения, выделяет contiguous storage и инициализирует все элементы нулями.

Большой пример с dot product, matmul, bias и 3D tensor находится в [examples/arrays_tensors.oc](examples/arrays_tensors.oc). Его backend-проверка:

```bash
./.venv/bin/pytest -v tests/test_array_tensor.py
```

## Borrow и ownership

`array[T]` и `tensor[T]` владеют выделенным storage и автоматически освобождаются в конце scope. Параметры `&T` — immutable borrow, `&mut T` — exclusive mutable borrow. Borrow-пути не добавляют ARC retain/release в generated C.

Прямые вызовы C обозначаются через `@` и остаются unsafe FFI-границей:

```python
cimport <math.h>

def main() -> float:
    return @sqrt(16.0)
```

## Старые примеры в новых паттернах

### Matrix: `list[list[int]]` → `tensor[float32]`

Старый класс `Matrix` из `examples/main.oc` больше не нужен для математики. Storage и shape теперь принадлежат tensor, а вычислительные функции получают borrow-параметры:

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

### Одномерный список → owned `array`

Для вектора чисел используйте `array[T]`, а не общий list runtime. Функция изменяет буфер через exclusive borrow:

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

### Старые базовые примеры

Input и циклы остаются простыми, но вычислительные данные лучше передавать через typed borrow:

```python
def main() -> int:
    var name: str = input("Enter your name: ")
    var values: array[float32] = [1.0, 2.0, 3.0]
    scale(values, 2.0)
    print("Hello, ", name, values[0])
    return 0
```

C/POSIX-вызовы по-прежнему явно отделены от безопасного кода через `@`:

```python
cimport <math.h>

def main() -> float:
    var result: float32 = @sqrt(16.0)
    return result
```

## Разработка

Соблюдайте четыре пробела в Python-коде, snake_case для функций и PascalCase для классов. Новые backend-проходы добавляйте в соответствующий модуль `src/codegen/`, а для изменений ownership или generated C — отдельный pytest-регрессионный тест. Generated Ocean symbols используют префикс `ocean_`.
