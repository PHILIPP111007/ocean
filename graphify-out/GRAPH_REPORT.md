# Graph Report - phils_language  (2026-08-11)

## Corpus Check
- 55 files · ~74,851 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 892 nodes · 1560 edges · 55 communities (48 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 39 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `40e2acaa`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- RuntimeError
- .add_warning
- TypesMixin
- .add_error
- OwnershipMixin
- SymbolTable
- Parser
- .parse_expression_to_ast
- JSONValidator
- generator.py
- .get_type_from_ast
- .parse_line
- .validate_assignment
- .get_symbol_info
- .extract_dependencies_from_ast
- ListCodegenMixin
- IndexingMixin
- IoMixin
- .parse_function_call
- StatementsMixin
- ScopeMixin
- .parse_function_arguments_to_ast
- HelpersMixin
- .parse_object_method_call_node
- .parse_type_annotation
- CoreMixin
- Handoff.md
- ._parse_with_priorities
- ExpressionsMixin
- CCodeGenerator
- ImportsMixin
- OrchestratorMixin
- ColoredFormatter
- NamingMixin
- .parse_class_attribute_initialization
- .validate_function_return_type
- Examples
- What changed
- parser.py
- CImportProcessor
- TypeSpec
- Repository Guidelines
- Ocean automatic ownership model v1
- 4. Memory model
- 20. Тесты
- ImportProcessor
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend

## God Nodes (most connected - your core abstractions)
1. `Parser` - 104 edges
2. `JSONValidator` - 90 edges
3. `run()` - 59 edges
4. `OopMixin` - 28 edges
5. `OwnershipMixin` - 27 edges
6. `CCodeGenerator` - 23 edges
7. `TypesMixin` - 22 edges
8. `SymbolTable` - 18 edges
9. `CallsMixin` - 17 edges
10. `StatementsMixin` - 14 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/base.py → src/codegen/generator.py
- `main()` --calls--> `CCodeGenerator`  [INFERRED]
  main.py → src/codegen/generator.py
- `main()` --calls--> `JSONValidator`  [EXTRACTED]
  main.py → src/debug.py
- `main()` --calls--> `Parser`  [EXTRACTED]
  main.py → src/parser.py
- `run()` --calls--> `Parser`  [EXTRACTED]
  tests/base.py → src/parser.py

## Import Cycles
- None detected.

## Communities (55 total, 7 thin omitted)

### Community 0 - "run"
Cohesion: 0.07
Nodes (40): run(), test_c_code_math(), test_c_code_pthread(), test_del(), test_dict(), test_dict_get(), test_for_loop_1(), test_for_loop_2() (+32 more)

### Community 1 - "OopMixin"
Cohesion: 0.05
Nodes (23): OopMixin, Генерирует структуру для класса C динамически, Анализирует метод для ссылок на атрибуты, Определяет тип поля по значению, Generate an ARC-owned zero-initialized class instance., Генерирует метод класса, Генерирует конструкторы для всех классов, Generate a method with borrowed parameters and automatic owner cleanup. (+15 more)

### Community 2 - "RuntimeError"
Cohesion: 0.07
Nodes (21): RuntimeError, CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Генерирует вызов встроенной функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует присваивание результата встроенной функции (+13 more)

### Community 3 - ".add_warning"
Cohesion: 0.06
Nodes (22): Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Проверяет объявленные, но неиспользуемые переменные, Собирает переменные из AST, Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль, Проверяет условия циклов на потенциальные проблемы, Проверяет потенциальные утечки памяти с указателями (+14 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (20): Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V], Очищает имя типа для использования в C идентификаторах, Извлекает информацию о вложенном типе списка с рекурсивным анализом (+12 more)

### Community 5 - ".add_error"
Cohesion: 0.07
Nodes (20): Находит родительский scope для заданного уровня, Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов, Валидирует один аргумент (может быть строкой или AST), Извлекает зависимости (имена переменных) из AST (+12 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.11
Nodes (8): OwnershipError, OwnershipMixin, Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Register the common ``ocean_`` ARC runtime in generated helpers., Raised when Phils ownership/borrow rules are violated during lowering.

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.07
Nodes (13): Parser, Парсит оператор break, Парсит оператор continue, Парсит оператор del (полное удаление), Определяет текущий scope на основе отступа, Парсит итерируемое выражение для for цикла, Проверяет, является ли имя именем класса, Извлекает содержимое внутри скобок, учитывая вложенность (+5 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.11
Nodes (11): Парсит литерал кортежа, Парсит оператор return, Parse an expression into the transitional Phils AST., Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке), Парсит значение опции и определяет его тип, Парсит условие для циклов и if, Парсит литерал списка: [1, 2, 3] или [[1, 2], [3, 4]] (+3 more)

### Community 10 - "JSONValidator"
Cohesion: 0.13
Nodes (11): JSONValidator, Строит карту соответствия узлов исходным строкам, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Основной метод валидации, Возвращает отчет о проверке, Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST (+3 more)

### Community 12 - ".get_type_from_ast"
Cohesion: 0.13
Nodes (11): Валидирует типы в условии if/elif, Валидирует типы в операциях, Определяет тип значения из AST, Валидирует операции с типами, Рекурсивно валидирует типы в AST, Проверяет, можно ли выполнить операцию между двумя типами, Находит scope по уровню, Валидирует типы в узле (+3 more)

### Community 13 - ".parse_line"
Cohesion: 0.14
Nodes (11): Парсит присваивание значения указателя переменной: x = *p, Parse a free function with fully nested type annotations., Возвращает область видимости для заданного уровня отступа, Находит конец блока с отступом, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Основной метод парсинга строки с поддержкой всех конструкций, Парсит вложенные if внутри других блоков (while, for, других if), Parse a value-semantic struct. Structs intentionally contain fields only in… (+3 more)

### Community 14 - ".validate_assignment"
Cohesion: 0.11
Nodes (12): Валидирует объявление переменной, Валидирует выражение (правая часть присваивания или инициализации), Валидирует присваивание, Валидирует оператор return, Валидирует запись через указатель (*p = значение), Валидирует присваивание, Проверяет совместимость типов при присваивании, Получает текущее состояние переменной (+4 more)

### Community 15 - ".get_symbol_info"
Cohesion: 0.12
Nodes (7): Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет выход за границы массивов/списков, Проверяет операции со строками, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST, Находит родительский узел (если есть)

### Community 16 - ".extract_dependencies_from_ast"
Cohesion: 0.12
Nodes (8): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит присваивание результата вызова функции: var x: type = func(args), Извлекает зависимости (используемые переменные) из AST, Парсит многомерное присваивание по индексу: A_data[0][0] = 10

### Community 17 - "ListCodegenMixin"
Cohesion: 0.16
Nodes (9): ListCodegenMixin, Генерирует структуру C для списка любой вложенности, Генерирует все функции для всех зарегистрированных структур списков, Рекурсивно генерирует элементы вложенного списка, Корректно генерирует элементы вложенного списка, Генерирует имя структуры для списка любой вложенности, Генерирует код для повторного объявления списка, Генерирует функции для работы со списком (без дублирования) (+1 more)

### Community 18 - "IndexingMixin"
Cohesion: 0.12
Nodes (9): IndexingMixin, Генерирует присваивание по индексу: list[index] = value или dict[key] = value, Генерирует код для многомерного индексного присваивания: A_data[0][0] = 10, Генерирует присваивание для вложенной индексации любой глубины, Генерирует присваивание среза: list[start:stop] = values, Генерирует составное присваивание по индексу: list[index] += value, Генерирует код для доступа по индексу, Генерирует выражение для вложенной индексации (для использования в выражениях) (+1 more)

### Community 19 - "IoMixin"
Cohesion: 0.16
Nodes (6): IoMixin, Генерирует выражение с input() и возвращает имя переменной с результатом, Генерирует код для чтения ввода с клавиатуры прямо в целевую переменную, Генерирует правильную конкатенацию строк, Генерирует вызов input() как отдельный statement (без присваивания), Генерирует код для чтения ввода с клавиатуры

### Community 20 - ".parse_function_call"
Cohesion: 0.20
Nodes (5): Парсит вызов встроенной функции, Разбирает аргументы функции с учетом строк и вложенных вызовов, Универсальный парсер любого вызова функции с поддержкой опций, Парсит прямой вызов C-функции, Определяет тип возвращаемого значения для встроенной функции

### Community 21 - "StatementsMixin"
Cohesion: 0.15
Nodes (7): Release loop-local owners before transferring control., Release current iteration owners before continuing., Ownership-safe class field assignment., Ownership-aware return: evaluate, establish return ownership, cleanup., Генерирует while loop с правильной обработкой структуры JSON, Генерирует if statement, StatementsMixin

### Community 22 - "ScopeMixin"
Cohesion: 0.23
Nodes (4): Enter a lexical ownership scope., Leave a lexical scope and deterministically release owned values., Generate a function with borrowed parameters and automatic cleanup., ScopeMixin

### Community 23 - ".parse_function_arguments_to_ast"
Cohesion: 0.17
Nodes (6): Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит аргументы функции в список AST, Парсит создание объекта: ClassName(arg1, arg2, ...), Парсит вызов статического метода: ClassName.method(args), Парсит вызов статического метода: Class.method(args)

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Генерирует вспомогательные функции для работы со строками, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - ".parse_object_method_call_node"
Cohesion: 0.18
Nodes (5): Парсит составные операции присваивания, Парсит вызов метода объекта с учетом наследования, Строит операции из AST выражения, Рекурсивно ищет символ в текущем и родительских scope'ах, Разрешает информацию о методе с учетом наследования

### Community 26 - ".parse_type_annotation"
Cohesion: 0.14
Nodes (9): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Парсит сложные выражения с несколькими операциями, Parse ``var self.attr: Type [= value]`` with nested types., Return canonical type text and structured metadata., Очищает значение от лишних пробелов, но для сложных выражений возвращает AST, Parse ``name: Type = default`` with nested generic/borrow types., find_top_level() (+1 more)

### Community 27 - "CoreMixin"
Cohesion: 0.22
Nodes (5): CoreMixin, Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку, Reset all per-compilation mutable state.

### Community 28 - "Handoff.md"
Cohesion: 0.08
Nodes (24): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+16 more)

### Community 29 - "._parse_with_priorities"
Cohesion: 0.10
Nodes (11): Парсит выражение с учетом приоритетов операторов Python, Парсит выражение на текущем уровне приоритета операторов, Парсит унарные операторы, Проверяет, что оператор в данной позиции является валидным оператором, Парсит цепочки индексации типа a[0][1][2], Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок (+3 more)

### Community 30 - "ExpressionsMixin"
Cohesion: 0.25
Nodes (5): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST для конструктора с подстановкой параметров, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 31 - "CCodeGenerator"
Cohesion: 0.20
Nodes (7): main(), DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря, CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…

### Community 32 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из JSON

### Community 33 - "OrchestratorMixin"
Cohesion: 0.33
Nodes (4): OrchestratorMixin, Run semantic prepasses, instantiate runtime types, then emit C., Генерирует объявление глобальной переменной, Генерирует имя временной переменной

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 36 - ".parse_class_attribute_initialization"
Cohesion: 0.40
Nodes (3): Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Выводит тип из AST выражения

### Community 39 - "Examples"
Cohesion: 0.15
Nodes (12): Benchmarks, C code -> function should starts with @, Cycles, Examples, Imports, Info, Input, Matmul (+4 more)

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Lexical hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 41 - "parser.py"
Cohesion: 0.30
Nodes (6): infer_literal_shape(), Split text only when not nested in (), [], {}, <> or strings., Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, split_top_level(), TypeParser

### Community 42 - "CImportProcessor"
Cohesion: 0.20
Nodes (4): CImportProcessor, Просто регистрирует C импорт без парсинга, Reset all per-compilation parser state. A Parser instance can safely be reused…, Parse one Phils compilation unit into the legacy graph + typed metadata. The…

### Community 44 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Safety & Configuration Notes, Testing Guidelines

### Community 45 - "Ocean automatic ownership model v1"
Cohesion: 0.25
Nodes (7): Borrow, Categories, Containers, FFI, Ocean automatic ownership model v1, Reference alias, Return ABI

### Community 46 - "4. Memory model"
Cohesion: 0.29
Nodes (7): 4. Memory model, BORROWED, Immutable borrow, Mutable borrow, OWNED, SHARED, VALUE

### Community 47 - "20. Тесты"
Cohesion: 0.33
Nodes (6): 20. Тесты, Level 1 — AST / parser, Level 2 — C generation, Level 3 — compile, Level 4 — memory safety, Обязательные memory tests

### Community 48 - "ImportProcessor"
Cohesion: 0.40
Nodes (3): ImportProcessor, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Статус array/tensor на момент handoff"
Cohesion: 0.67
Nodes (3): 28. Статус array/tensor на момент handoff, Backend, Parser

### Community 51 - "30. Следующий рекомендуемый этап"
Cohesion: 0.67
Nodes (3): 30. Следующий рекомендуемый этап, Array, Tensor

## Knowledge Gaps
- **73 isolated node(s):** `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines`, `Commit & Pull Request Guidelines` (+68 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `ImportsMixin`, `OopMixin`, `RuntimeError`, `NamingMixin`, `OrchestratorMixin`, `TypesMixin`, `OwnershipMixin`, `run`, `generator.py`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `CoreMixin`, `ExpressionsMixin`?**
  _High betweenness centrality (0.371) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `.parse_class_attribute_initialization`, `SymbolTable`, `parser.py`, `CImportProcessor`, `.parse_expression_to_ast`, `.parse_line`, `.extract_dependencies_from_ast`, `.parse_function_call`, `.parse_function_arguments_to_ast`, `.parse_object_method_call_node`, `.parse_type_annotation`, `._parse_with_priorities`, `CCodeGenerator`?**
  _High betweenness centrality (0.345) - this node is a cross-community bridge._
- **Why does `JSONValidator` connect `JSONValidator` to `.add_warning`, `.add_error`, `.validate_function_return_type`, `generator.py`, `.get_type_from_ast`, `.validate_assignment`, `.get_symbol_info`, `CCodeGenerator`?**
  _High betweenness centrality (0.302) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions` to the rest of the system?**
  _73 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run` be split into smaller, more focused modules?**
  _Cohesion score 0.06662770309760374 - nodes in this community are weakly interconnected._
- **Should `OopMixin` be split into smaller, more focused modules?**
  _Cohesion score 0.0549645390070922 - nodes in this community are weakly interconnected._