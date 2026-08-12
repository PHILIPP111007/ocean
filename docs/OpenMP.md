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

Генерируется стандартная C-директива `#pragma omp parallel for`. Допустимо также
написание `#pragma opm ...` для совместимости, но в C всегда выводится `omp`.

Поддерживаемые clauses: `schedule`, `reduction`, `private`, `firstprivate`,
`lastprivate`, `shared`, `default`, `nowait` и `ordered`. `collapse` будет
добавлен вместе с безопасной поддержкой вложенных циклов.

Для `parallel for` сейчас разрешены только циклы `range(...)` с постоянным
ненулевым целочисленным шагом. Тело может работать со scalar-переменными и
записывать в `array`/`tensor`, если индекс содержит переменную цикла. Управляемые
объекты (`list`, `dict`, строки и классы), вызовы функций, `break`, `continue` и
вложенные циклы пока запрещены: текущий ARC/ownership runtime рассчитан на
thread-confined объекты и не является потокобезопасным.

При сборке `-fopenmp` добавляется автоматически, если в сгенерированном C есть
OpenMP pragma. Его также можно указать явно:

```bash
python main.py examples/openmp.oc --cflag=-fopenmp --run
```
