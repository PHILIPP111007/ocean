# OpenMP

Ocean поддерживает ограниченный безопасный вариант OpenMP для циклов по `range`:

```text
def main() -> int:
    var total: int = 0

    #pragma omp parallel for reduction(+:total) schedule(static)
    for i in range(0, 1000):
        total += i

    return 0
```

Вложенные циклы можно объединить через `collapse(n)`:

```text
#pragma omp parallel for collapse(2) reduction(+:total)
for i in range(0, rows):
    for j in range(0, columns):
        total += values[i, j]
```

Для `collapse(n)` циклы должны быть идеально вложены: между ними не должно
быть операторов, каждый цикл должен использовать `range(...)`, а шаг должен быть
постоянным ненулевым целым числом. Все переменные collapsed-цикла должны входить
в индекс изменяемого `array`/`tensor`.

Генерируется стандартная C-директива `#pragma omp parallel for`. Допустимо также
написание `#pragma opm ...` для совместимости, но в C всегда выводится `omp`.

Поддерживаемые clauses: `schedule`, `collapse`, `reduction`, `private`,
`firstprivate`, `lastprivate`, `shared`, `default`, `nowait` и `ordered`.

Для `parallel for` сейчас разрешены только циклы `range(...)` с постоянным
ненулевым целочисленным шагом. Тело может работать со scalar-переменными и
записывать в `array`/`tensor`, если индекс содержит переменную цикла. Управляемые
объекты (`list`, `dict`, строки и классы), вызовы функций, `break`, `continue` и
вложенные циклы пока запрещены: текущий ARC/ownership runtime рассчитан на
thread-confined объекты и не является потокобезопасным.

При сборке `-fopenmp` добавляется автоматически, если в сгенерированном C есть
OpenMP pragma. Его также можно указать явно:

```bash
python main.py run examples/openmp.oc --cflag=-fopenmp --run
```
