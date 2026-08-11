# Graph Report - phils_language  (2026-08-12)

## Corpus Check
- 68 files · ~84,838 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1125 nodes · 2163 edges · 63 communities (57 shown, 6 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 77 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `86d36176`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- RuntimeError
- .validate_scope
- TypesMixin
- .is_variable_deleted
- OwnershipMixin
- SymbolTable
- Parser
- .parse_expression_to_ast
- .check_undefined_methods
- generator.py
- .get_type_from_ast
- .calculate_indent_level
- main.py
- .add_error
- ._parse_line_impl
- ListCodegenMixin
- IndexingMixin
- IoMixin
- .parse_function_call
- StatementsMixin
- ScopeMixin
- JSONValidator
- HelpersMixin
- .parse_function_arguments_to_ast
- .parse_type_annotation
- CoreMixin
- Handoff.md
- .parse_complex_expression
- ExpressionsMixin
- test_array_tensor.py
- CCodeGenerator
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- NamingMixin
- parser.py
- compiler.py
- What changed
- split_top_level
- TypeParser
- test_memory_safety.py
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- TypedIRBuilder
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- IRType
- .validate
- test_debug_validator.py
- ocean-lang
- .validate_graph
- benchmark_main.py
- .validate_type_compatibility

## God Nodes (most connected - your core abstractions)
1. `JSONValidator` - 124 edges
2. `Parser` - 123 edges
3. `run()` - 60 edges
4. `CCodeGenerator` - 31 edges
5. `OwnershipMixin` - 29 edges
6. `OopMixin` - 28 edges
7. `TensorCodegenMixin` - 27 edges
8. `TypesMixin` - 23 edges
9. `ArrayCodegenMixin` - 22 edges
10. `TypeParser` - 20 edges

## Surprising Connections (you probably didn't know these)
- `run_benchmark()` --calls--> `CCodeGenerator`  [INFERRED]
  benchmarks/benchmark_main.py → src/codegen/generator.py
- `compile_pipeline()` --calls--> `CCodeGenerator`  [INFERRED]
  main.py → src/codegen/generator.py
- `run()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/base.py → src/codegen/generator.py
- `generate()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/test_array_tensor.py → src/codegen/generator.py
- `test_tensor_views_keep_owner_and_broadcast_shapes()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/test_array_tensor.py → src/codegen/generator.py

## Import Cycles
- None detected.

## Communities (63 total, 6 thin omitted)

### Community 0 - "run"
Cohesion: 0.07
Nodes (42): run(), test_c_code_math(), test_c_code_pthread(), test_del(), test_dict(), test_dict_get(), test_for_loop_1(), test_for_loop_2() (+34 more)

### Community 1 - "OopMixin"
Cohesion: 0.05
Nodes (23): OopMixin, Генерирует структуру для класса C динамически, Анализирует метод для ссылок на атрибуты, Определяет тип поля по значению, Generate an ARC-owned zero-initialized class instance., Генерирует метод класса, Генерирует конструкторы для всех классов, Generate a method with borrowed parameters and automatic owner cleanup. (+15 more)

### Community 2 - "RuntimeError"
Cohesion: 0.06
Nodes (21): RuntimeError, ArrayCodegenMixin, Lower uniquely-owned one-dimensional ``array[T]`` values to C., CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Генерирует вызов встроенной функции (+13 more)

### Community 3 - ".validate_scope"
Cohesion: 0.06
Nodes (23): Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение, Добавляет предупреждение с информацией о строке, Проверяет деление на ноль, Проверяет условия циклов на потенциальные проблемы, Проверяет потенциальные утечки памяти с указателями (+15 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (21): Resolve an object receiver and its C expression. Besides local variables and…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V], Очищает имя типа для использования в C идентификаторах (+13 more)

### Community 5 - ".is_variable_deleted"
Cohesion: 0.10
Nodes (13): Валидирует выражение (правая часть присваивания или инициализации), Валидирует присваивание, Валидирует унарную операцию, Валидирует вызов встроенной функции, Валидирует вызов функции print, Валидирует оператор return, Валидирует узел цикла, Валидирует присваивание (+5 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters., Register the common ``ocean_`` ARC runtime in generated helpers. (+1 more)

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.08
Nodes (12): Parser, Remove standalone triple-quoted blocks while preserving line count., Парсит оператор break, Возвращает область видимости для заданного уровня отступа, Определяет текущий scope на основе отступа, Reset all per-compilation parser state. A Parser instance can safely be reused…, Parse one Phils compilation unit into the legacy graph + typed metadata. The…, Проверяет, является ли имя именем класса (+4 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.10
Nodes (12): Парсит литерал кортежа, Парсит оператор return, Parse an expression into the transitional Phils AST., Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке), Парсит значение опции и определяет его тип, Парсит присваивание результата вызова функции: var x: type = func(args), Парсит условие для циклов и if (+4 more)

### Community 10 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 11 - "generator.py"
Cohesion: 0.16
Nodes (5): DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря, # TODO: рекурсивный анализ для определения типа атрибута, # TODO: анализировать возвращаемый тип функции

### Community 12 - ".get_type_from_ast"
Cohesion: 0.09
Nodes (16): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+8 more)

### Community 13 - ".calculate_indent_level"
Cohesion: 0.12
Nodes (12): Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in…, Парсит объявление класса, Parse a class method using the same typed parameter parser as functions. (+4 more)

### Community 14 - "main.py"
Cohesion: 0.09
Nodes (52): ArgumentParser, build_argument_parser(), cli(), _command(), compile_c(), compile_pipeline(), _compiler_settings(), default_output_paths() (+44 more)

### Community 15 - ".add_error"
Cohesion: 0.11
Nodes (9): Валидирует удаление переменной, Валидирует составное присваивание, Валидирует объявление указателя, Добавляет ошибку с информацией о строке, Получает информацию о символе из текущего или родительских scope'ов, Проверяет корректное использование указателей, Проверяет выход за границы массивов/списков, Проверяет вызовы C-функций (начинающиеся с @) (+1 more)

### Community 16 - "._parse_line_impl"
Cohesion: 0.08
Nodes (15): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит присваивание значения указателя переменной: x = *p (+7 more)

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
Cohesion: 0.25
Nodes (4): Парсит вызов встроенной функции, Разбирает аргументы функции с учетом строк и вложенных вызовов, Универсальный парсер любого вызова функции с поддержкой опций, Определяет тип возвращаемого значения для встроенной функции

### Community 21 - "StatementsMixin"
Cohesion: 0.12
Nodes (9): Генерирует if statement, Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Ownership-safe class field assignment., Release loop-local owners before transferring control., Release current iteration owners before continuing., Ownership-aware return: evaluate, establish return ownership, cleanup., Генерирует while loop с правильной обработкой структуры JSON (+1 more)

### Community 22 - "ScopeMixin"
Cohesion: 0.23
Nodes (4): Enter a lexical ownership scope., Leave a lexical scope and deterministically release owned values., Generate a function with borrowed parameters and automatic cleanup., ScopeMixin

### Community 23 - "JSONValidator"
Cohesion: 0.17
Nodes (6): JSONValidator, Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Собирает переменные из AST, Collect variable references from any expression AST variant., Collect variable references from expression ASTs only.

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует вспомогательные функции для работы со строками, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - ".parse_function_arguments_to_ast"
Cohesion: 0.12
Nodes (8): Парсит составные операции присваивания, Парсит аргументы функции в список AST, Парсит создание объекта: ClassName(arg1, arg2, ...), Парсит вызов статического метода: ClassName.method(args), Парсит вызов метода объекта с учетом наследования, Строит операции из AST выражения, Рекурсивно ищет символ в текущем и родительских scope'ах, Разрешает информацию о методе с учетом наследования

### Community 26 - ".parse_type_annotation"
Cohesion: 0.11
Nodes (12): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Парсит сложные выражения с несколькими операциями, Parse ``var self.attr: Type [= value]`` with nested types., Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Очищает значение от лишних пробелов, но для сложных выражений возвращает AST, Parse ``name: Type = default`` with nested generic/borrow types. (+4 more)

### Community 27 - "CoreMixin"
Cohesion: 0.22
Nodes (5): CoreMixin, Reset all per-compilation mutable state., Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку

### Community 28 - "Handoff.md"
Cohesion: 0.08
Nodes (24): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+16 more)

### Community 29 - ".parse_complex_expression"
Cohesion: 0.13
Nodes (8): Парсит выражение на текущем уровне приоритета операторов, Проверяет, что оператор в данной позиции является валидным оператором, Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок, Проверяет, является ли символ частью идентификатора, Находит позицию оператора вне скобок, строк и комментариев, Проверяет, содержит ли выражение какой-либо оператор

### Community 30 - "ExpressionsMixin"
Cohesion: 0.25
Nodes (5): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST для конструктора с подстановкой параметров, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 31 - "test_array_tensor.py"
Cohesion: 0.33
Nodes (8): generate(), test_array_lowering_and_index_mutation(), test_tensor_lowering_shape_and_index_mutation(), test_tensor_numeric_methods_use_shape_and_stride_runtime(), test_tensor_rank_specialization_is_unbounded(), test_tensor_the_big_code(), test_tensor_views_keep_owner_and_broadcast_shapes(), test_tensor_zeros_dynamic_shape()

### Community 32 - "CCodeGenerator"
Cohesion: 0.22
Nodes (6): CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из JSON

### Community 33 - "OrchestratorMixin"
Cohesion: 0.19
Nodes (7): OrchestratorMixin, Generate C from the semantic IR while preserving the legacy backend., Генерирует имя временной переменной, Генерирует код для узла графа, Run semantic prepasses, instantiate runtime types, then emit C., Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 35 - "TensorCodegenMixin"
Cohesion: 0.14
Nodes (5): Emit view-aware reductions, matrix operations, and broadcasting., Lower tensor arithmetic to shape-checked broadcasting helpers., Prepare checked-once direct access for provably bounded 2D loops., Lower dense row-major ``tensor[T]`` values with owned storage., TensorCodegenMixin

### Community 37 - "parser.py"
Cohesion: 0.21
Nodes (5): CImportProcessor, ImportProcessor, Просто регистрирует C импорт без парсинга, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов

### Community 39 - "compiler.py"
Cohesion: 0.18
Nodes (8): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, build_typed_ir(), Convenience entry point used by the compiler pipeline and tests., Typed compilation unit with a lossless legacy representation., Return a deep copy accepted by the existing validator and C backend., TypedModule, test_typed_ir_keeps_codegen_compatibility_format(), test_typed_ir_tracks_types_reads_writes_and_effects()

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 41 - "split_top_level"
Cohesion: 0.22
Nodes (6): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит выражение с учетом приоритетов операторов Python, Парсит унарные операторы, Парсит цепочки индексации типа a[0][1][2], Split text only when not nested in (), [], {}, <> or strings., split_top_level()

### Community 42 - "TypeParser"
Cohesion: 0.11
Nodes (11): infer_literal_shape(), Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, Structured representation of a Phils type. The parser still emits the canonical…, TypeParser, TypeSpec, Typed intermediate representation for the Ocean compiler. The parser's…, Semantic metadata for one legacy graph node. (+3 more)

### Community 43 - "test_memory_safety.py"
Cohesion: 0.27
Nodes (12): compile_ocean(), test_borrow_cannot_escape_through_return(), test_borrow_cannot_escape_to_non_borrowing_parameter(), test_borrow_is_released_at_block_exit(), test_direct_c_call_requires_unsafe_block(), test_immutable_borrow_cannot_be_passed_to_mutable_parameter(), test_mutable_and_immutable_borrows_are_exclusive(), test_owned_array_is_moved_into_by_value_parameter() (+4 more)

### Community 44 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Safety & Configuration Notes, Testing Guidelines

### Community 45 - "Ocean 🌊"
Cohesion: 0.09
Nodes (21): Borrow, Categories, Containers, FFI, Move, Ocean automatic ownership model v1, Reference alias, Return ABI (+13 more)

### Community 46 - "4. Memory model"
Cohesion: 0.29
Nodes (7): 4. Memory model, BORROWED, Immutable borrow, Mutable borrow, OWNED, SHARED, VALUE

### Community 47 - "20. Тесты"
Cohesion: 0.33
Nodes (6): 20. Тесты, Level 1 — AST / parser, Level 2 — C generation, Level 3 — compile, Level 4 — memory safety, Обязательные memory tests

### Community 48 - "TypedIRBuilder"
Cohesion: 0.45
Nodes (3): Any, Lower parser dictionaries into typed scopes and effect-annotated nodes., TypedIRBuilder

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Статус array/tensor на момент handoff"
Cohesion: 0.67
Nodes (3): 28. Статус array/tensor на момент handoff, Backend, Parser

### Community 51 - "30. Следующий рекомендуемый этап"
Cohesion: 0.67
Nodes (3): 30. Следующий рекомендуемый этап, Array, Tensor

### Community 56 - ".validate"
Cohesion: 0.25
Nodes (4): Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Основной метод валидации, Собирает информацию о всех символах в системе

### Community 57 - "test_debug_validator.py"
Cohesion: 0.48
Nodes (6): test_validator_accepts_typed_borrow_and_generic_symbols(), test_validator_checks_container_and_index_types_from_ast(), test_validator_does_not_leak_symbols_between_functions(), test_validator_reports_real_source_line_for_type_error(), test_validator_returns_report_for_malformed_input(), validate()

### Community 60 - ".validate_graph"
Cohesion: 0.12
Nodes (9): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов, Валидирует один аргумент (может быть строкой или AST), Извлекает зависимости (имена переменных) из AST, Валидирует вызов статического метода, Находит символ класса в таблице символов (+1 more)

### Community 61 - "benchmark_main.py"
Cohesion: 0.31
Nodes (8): main(), measure(), Path, Benchmark the generated C program for examples/main.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, ValueError

### Community 62 - ".validate_type_compatibility"
Cohesion: 0.33
Nodes (3): Валидирует объявление переменной, Проверяет совместимость типов при присваивании, Пытается определить тип по значению

## Knowledge Gaps
- **78 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+73 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Parser` connect `Parser` to `run`, `parser.py`, `SymbolTable`, `compiler.py`, `.parse_expression_to_ast`, `split_top_level`, `TypeParser`, `test_memory_safety.py`, `.calculate_indent_level`, `main.py`, `._parse_line_impl`, `.parse_complex_expression`, `.parse_function_call`, `.parse_function_arguments_to_ast`, `.parse_type_annotation`, `test_debug_validator.py`, `benchmark_main.py`, `test_array_tensor.py`?**
  _High betweenness centrality (0.338) - this node is a cross-community bridge._
- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `RuntimeError`, `TypesMixin`, `OwnershipMixin`, `generator.py`, `main.py`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `CoreMixin`, `ExpressionsMixin`, `test_array_tensor.py`, `OrchestratorMixin`, `TensorCodegenMixin`, `NamingMixin`, `compiler.py`, `test_memory_safety.py`, `benchmark_main.py`?**
  _High betweenness centrality (0.337) - this node is a cross-community bridge._
- **Why does `JSONValidator` connect `JSONValidator` to `.validate_scope`, `.is_variable_deleted`, `compiler.py`, `.check_undefined_methods`, `test_memory_safety.py`, `.get_type_from_ast`, `main.py`, `.add_error`, `.validate`, `test_debug_validator.py`, `.validate_graph`, `benchmark_main.py`, `.validate_type_compatibility`, `test_array_tensor.py`?**
  _High betweenness centrality (0.336) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `RuntimeError` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`RuntimeError` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `CCodeGenerator` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`CCodeGenerator` has 27 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _78 weakly-connected nodes found - possible documentation gaps or missing edges._