# Graph Report - phils_language  (2026-08-13)

## Corpus Check
- 73 files · ~87,295 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1200 nodes · 2345 edges · 71 communities (61 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 90 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `11bc83e4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- RuntimeError
- .validate_scope
- TypesMixin
- .find_symbol_in_scope
- OwnershipMixin
- SymbolTable
- Parser
- .parse_expression_to_ast
- .check_undefined_methods
- generator.py
- .add_error
- .calculate_indent_level
- main.py
- .get_symbol_info
- ._parse_line_impl
- ListCodegenMixin
- IndexingMixin
- IoMixin
- .parse_function_call
- StatementsMixin
- ScopeMixin
- JSONValidator
- HelpersMixin
- .extract_dependencies_from_ast
- .parse_type_annotation
- CoreMixin
- Handoff.md
- .parse_complex_expression
- ExpressionsMixin
- test_array_tensor.py
- ImportsMixin
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- NamingMixin
- ArrayCodegenMixin
- .parse_function_arguments_to_ast
- What changed
- TupleCodegenMixin
- parser.py
- test_memory_safety.py
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- .generate_for_loop
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- .generate_assignment
- .validate
- test_debug_validator.py
- ocean-lang
- .validate_graph
- benchmark_main.py
- .generate_builtin_function_call
- class_model.py
- ClassRegistry
- ._prepare_tensor_fast_path
- ClassModel
- CCodeGenerator
- .generate_all_methods
- DictCodegenMixin
- OpenMP.md

## God Nodes (most connected - your core abstractions)
1. `JSONValidator` - 136 edges
2. `Parser` - 132 edges
3. `run()` - 60 edges
4. `CCodeGenerator` - 37 edges
5. `OwnershipMixin` - 29 edges
6. `TensorCodegenMixin` - 27 edges
7. `TypesMixin` - 24 edges
8. `ArrayCodegenMixin` - 22 edges
9. `ClassRegistry` - 20 edges
10. `OopMixin` - 20 edges

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

## Communities (71 total, 10 thin omitted)

### Community 0 - "run"
Cohesion: 0.07
Nodes (42): run(), test_c_code_math(), test_c_code_pthread(), test_del(), test_dict(), test_dict_get(), test_for_loop_1(), test_for_loop_2() (+34 more)

### Community 1 - "OopMixin"
Cohesion: 0.16
Nodes (8): OopMixin, Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Initialize a zeroed field, retaining only borrowed incoming references., Generate an ARC-owned zero-initialized class instance., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models., Generate an ARC-compatible class layout with safe single inheritance.

### Community 2 - "RuntimeError"
Cohesion: 0.12
Nodes (11): RuntimeError, CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции (+3 more)

### Community 3 - ".validate_scope"
Cohesion: 0.06
Nodes (21): Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Добавляет предупреждение с информацией о строке, Warn about locals unused across the complete nested graph., Получает номер строки исходного кода для узла, Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль, Проверяет условия циклов на потенциальные проблемы (+13 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".find_symbol_in_scope"
Cohesion: 0.08
Nodes (14): Валидирует объявление переменной, Валидирует выражение (правая часть присваивания или инициализации), Валидирует присваивание, Валидирует унарную операцию, Валидирует вызов встроенной функции, Валидирует вызов функции print, Валидирует оператор return, Получает текущее состояние переменной (+6 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters., Register the common ``ocean_`` ARC runtime in generated helpers. (+1 more)

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.09
Nodes (10): Parser, Remove standalone triple-quoted blocks while preserving line count., Определяет текущий scope на основе отступа, Reset all per-compilation parser state. A Parser instance can safely be reused…, Parse one Phils compilation unit into the legacy graph + typed metadata. The…, Проверяет, является ли имя именем класса, Извлекает содержимое внутри скобок, учитывая вложенность, Удаляет дублирующиеся методы из классов (+2 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.09
Nodes (16): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит литерал кортежа, Parse an expression into the transitional Phils AST., Парсит выражение с учетом приоритетов операторов Python, Парсит унарные операторы, Парсит цепочки индексации типа a[0][1][2], Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке) (+8 more)

### Community 10 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 12 - ".add_error"
Cohesion: 0.10
Nodes (18): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+10 more)

### Community 13 - ".calculate_indent_level"
Cohesion: 0.12
Nodes (12): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in…, Парсит объявление класса (+4 more)

### Community 14 - "main.py"
Cohesion: 0.07
Nodes (64): ArgumentParser, build_argument_parser(), cli(), _command(), compile_c(), compile_pipeline(), _compiler_settings(), default_output_paths() (+56 more)

### Community 15 - ".get_symbol_info"
Cohesion: 0.12
Nodes (7): Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет выход за границы массивов/списков, Проверяет операции со строками, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST, Находит родительский узел (если есть)

### Community 16 - "._parse_line_impl"
Cohesion: 0.10
Nodes (9): Парсит присваивание значения указателя переменной: x = *p, Парсит оператор break, Парсит оператор continue, Парсит оператор del (полное удаление), Парсит оператор return, Возвращает область видимости для заданного уровня отступа, Parse the supported OpenMP loop directive into structured metadata., Парсит вложенные if внутри других блоков (while, for, других if) (+1 more)

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
Nodes (7): Генерирует if statement, Release loop-local owners before transferring control., Release current iteration owners before continuing., Ownership-safe class field assignment., Ownership-aware return: evaluate, establish return ownership, cleanup., Генерирует while loop с правильной обработкой структуры JSON, StatementsMixin

### Community 22 - "ScopeMixin"
Cohesion: 0.23
Nodes (4): Enter a lexical ownership scope., Leave a lexical scope and deterministically release owned values., Generate a function with borrowed parameters and automatic cleanup., ScopeMixin

### Community 23 - "JSONValidator"
Cohesion: 0.11
Nodes (11): JSONValidator, Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Валидирует узел цикла, Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate structured OpenMP metadata before C code generation. (+3 more)

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует вспомогательные функции для работы со строками, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - ".extract_dependencies_from_ast"
Cohesion: 0.09
Nodes (12): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит составные операции присваивания, Парсит присваивание результата вызова функции: var x: type = func(args), Парсит вызов метода объекта с учетом наследования (+4 more)

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
Cohesion: 0.33
Nodes (4): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 31 - "test_array_tensor.py"
Cohesion: 0.33
Nodes (8): generate(), test_array_lowering_and_index_mutation(), test_tensor_lowering_shape_and_index_mutation(), test_tensor_numeric_methods_use_shape_and_stride_runtime(), test_tensor_rank_specialization_is_unbounded(), test_tensor_the_big_code(), test_tensor_views_keep_owner_and_broadcast_shapes(), test_tensor_zeros_dynamic_shape()

### Community 32 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из JSON

### Community 33 - "OrchestratorMixin"
Cohesion: 0.19
Nodes (7): OrchestratorMixin, Generate C from the semantic IR while preserving the legacy backend., Генерирует имя временной переменной, Генерирует код для узла графа, Run semantic prepasses, instantiate runtime types, then emit C., Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 35 - "TensorCodegenMixin"
Cohesion: 0.17
Nodes (4): Emit view-aware reductions, matrix operations, and broadcasting., Lower tensor arithmetic to shape-checked broadcasting helpers., Lower dense row-major ``tensor[T]`` values with owned storage., TensorCodegenMixin

### Community 39 - ".parse_function_arguments_to_ast"
Cohesion: 0.17
Nodes (6): Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит аргументы функции в список AST, Парсит создание объекта: ClassName(arg1, arg2, ...), Парсит вызов статического метода: ClassName.method(args), Парсит вызов статического метода: Class.method(args)

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 41 - "TupleCodegenMixin"
Cohesion: 0.36
Nodes (5): Генерирует имя структуры для tuple, Генерирует код для повторного объявления кортежа, Create a homogeneous immutable tuple with owned element references., Generate an ARC-owned homogeneous tuple[T]., TupleCodegenMixin

### Community 42 - "parser.py"
Cohesion: 0.05
Nodes (26): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, CImportProcessor, ImportProcessor, Просто регистрирует C импорт без парсинга, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, infer_literal_shape(), Recursive parser for Phils type expressions. (+18 more)

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

### Community 48 - ".generate_for_loop"
Cohesion: 0.33
Nodes (3): Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Render validated structured OpenMP metadata as one C pragma.

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Статус array/tensor на момент handoff"
Cohesion: 0.67
Nodes (3): 28. Статус array/tensor на момент handoff, Backend, Parser

### Community 51 - "30. Следующий рекомендуемый этап"
Cohesion: 0.67
Nodes (3): 30. Следующий рекомендуемый этап, Array, Tensor

### Community 55 - ".generate_assignment"
Cohesion: 0.33
Nodes (3): Generate assignment with ARC/string ownership and borrow checks., Declare a value and establish its ownership state., Redeclaration is a destruction boundary followed by a new owner.

### Community 56 - ".validate"
Cohesion: 0.25
Nodes (4): Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Основной метод валидации, Собирает информацию о всех символах в системе

### Community 57 - "test_debug_validator.py"
Cohesion: 0.48
Nodes (6): test_validator_accepts_typed_borrow_and_generic_symbols(), test_validator_checks_container_and_index_types_from_ast(), test_validator_does_not_leak_symbols_between_functions(), test_validator_reports_real_source_line_for_type_error(), test_validator_returns_report_for_malformed_input(), validate()

### Community 60 - ".validate_graph"
Cohesion: 0.09
Nodes (12): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует удаление переменной, Валидирует составное присваивание, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов, Валидирует один аргумент (может быть строкой или AST), Извлекает зависимости (имена переменных) из AST (+4 more)

### Community 61 - "benchmark_main.py"
Cohesion: 0.10
Nodes (20): main(), measure(), Path, Benchmark the generated C program for examples/main.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, device (+12 more)

### Community 63 - "class_model.py"
Cohesion: 0.23
Nodes (11): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+3 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (4): ClassRegistry, Canonical class metadata and lookup service for the C backend., Resolve methods without rebuilding parser-shaped dictionaries., Build the canonical OOP metadata directly from parser output.

### Community 66 - "ClassModel"
Cohesion: 0.24
Nodes (6): ClassModel, FieldModel, A field declared directly by one class., Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "CCodeGenerator"
Cohesion: 0.31
Nodes (10): CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, compile_and_run(), test_oop_constructor_method_mutation_and_composition(), test_oop_default_constructor_without_init(), test_oop_inherited_field_access_uses_embedded_base_layout(), test_oop_metadata_has_one_canonical_class_model(), test_oop_rejects_inheritance_cycles() (+2 more)

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 69 - "DictCodegenMixin"
Cohesion: 0.50
Nodes (3): DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря

## Knowledge Gaps
- **79 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `RuntimeError`, `TypesMixin`, `OwnershipMixin`, `generator.py`, `main.py`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `CoreMixin`, `ExpressionsMixin`, `test_array_tensor.py`, `ImportsMixin`, `OrchestratorMixin`, `TensorCodegenMixin`, `NamingMixin`, `ArrayCodegenMixin`, `TupleCodegenMixin`, `parser.py`, `test_memory_safety.py`, `benchmark_main.py`, `DictCodegenMixin`?**
  _High betweenness centrality (0.323) - this node is a cross-community bridge._
- **Why does `JSONValidator` connect `JSONValidator` to `.validate_scope`, `.find_symbol_in_scope`, `.check_undefined_methods`, `generator.py`, `.add_error`, `parser.py`, `main.py`, `.get_symbol_info`, `test_memory_safety.py`, `.validate`, `test_debug_validator.py`, `.validate_graph`, `benchmark_main.py`, `test_array_tensor.py`?**
  _High betweenness centrality (0.295) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `CCodeGenerator`, `SymbolTable`, `.parse_function_arguments_to_ast`, `.parse_expression_to_ast`, `parser.py`, `test_memory_safety.py`, `.calculate_indent_level`, `main.py`, `._parse_line_impl`, `.parse_complex_expression`, `.parse_function_call`, `.extract_dependencies_from_ast`, `.parse_type_annotation`, `test_debug_validator.py`, `benchmark_main.py`, `test_array_tensor.py`?**
  _High betweenness centrality (0.293) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `RuntimeError` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`RuntimeError` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 33 inferred relationships involving `CCodeGenerator` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`CCodeGenerator` has 33 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _79 weakly-connected nodes found - possible documentation gaps or missing edges._