# Graph Report - phils_language  (2026-08-13)

## Corpus Check
- 79 files · ~95,036 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1320 nodes · 2730 edges · 74 communities (65 shown, 9 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 95 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c11834a8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- CallsMixin
- .add_error
- TypesMixin
- .get_type_from_ast
- OwnershipMixin
- SymbolTable
- Parser
- .parse_expression_to_ast
- .validate_scope
- generator.py
- Validator
- ._parse_line_impl
- main.py
- ArrayCodegenMixin
- .extract_dependencies_from_ast
- ListCodegenMixin
- IndexingMixin
- IoMixin
- .parse_function_call
- StatementsMixin
- ScopeMixin
- .get_symbol_info
- HelpersMixin
- TupleCodegenMixin
- .parse_type_annotation
- CoreMixin
- Handoff.md
- TypedScope
- ._validate_scopes
- .check_undefined_methods
- .parse_object_method_call_node
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- CCodeGenerator
- tensor_runtime.c
- .parse_complex_expression
- What changed
- modules/imports.py
- Any
- test_memory_safety.py
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- TypeSpec
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- .validate_type_compatibility
- TypedExpression
- ocean-lang
- RuntimeError
- benchmark_main.py
- parser.py
- class_model.py
- ClassRegistry
- test_debug_validator.py
- ClassModel
- TypedModule
- .generate_all_methods
- compiler.py
- ExpressionsMixin
- ImportsMixin
- TypedNode
- .resolved_methods

## God Nodes (most connected - your core abstractions)
1. `Validator` - 142 edges
2. `Parser` - 135 edges
3. `run()` - 60 edges
4. `ocean_tensor_fail()` - 50 edges
5. `CCodeGenerator` - 38 edges
6. `OwnershipMixin` - 29 edges
7. `ocean_tensor_to()` - 27 edges
8. `compile_c()` - 26 edges
9. `compile_pipeline()` - 24 edges
10. `TypesMixin` - 24 edges

## Surprising Connections (you probably didn't know these)
- `run_benchmark()` --calls--> `CCodeGenerator`  [INFERRED]
  benchmarks/benchmark_main.py → src/codegen/generator.py
- `compile_pipeline()` --calls--> `CCodeGenerator`  [INFERRED]
  main.py → src/codegen/generator.py
- `run()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/base.py → src/codegen/generator.py
- `generate()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/test_array_tensor.py → src/codegen/generator.py
- `compile_ocean()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/test_memory_safety.py → src/codegen/generator.py

## Import Cycles
- None detected.

## Communities (74 total, 9 thin omitted)

### Community 0 - "run"
Cohesion: 0.05
Nodes (47): ImportProcessor, Path, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield repository std directories from the active source context., run(), test_c_code_math(), test_c_code_pthread() (+39 more)

### Community 1 - "OopMixin"
Cohesion: 0.14
Nodes (9): OopMixin, Build the canonical OOP metadata directly from parser output., Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models. (+1 more)

### Community 2 - "CallsMixin"
Cohesion: 0.14
Nodes (8): CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует вызов встроенной функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует присваивание результата встроенной функции, Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции

### Community 3 - ".add_error"
Cohesion: 0.07
Nodes (22): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует выражение (правая часть присваивания или инициализации), Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов (+14 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".get_type_from_ast"
Cohesion: 0.08
Nodes (18): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+10 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Raised when Phils ownership/borrow rules are violated during lowering., Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters. (+1 more)

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.06
Nodes (22): Parser, Remove standalone triple-quoted blocks while preserving line count., Парсит присваивание значения указателя переменной: x = *p, Парсит оператор break, Парсит оператор continue, Возвращает область видимости для заданного уровня отступа, Определяет текущий scope на основе отступа, Парсит итерируемое выражение для for цикла (+14 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.08
Nodes (16): Парсит литерал кортежа, Парсит оператор return, Parse an expression into the transitional Phils AST., Парсит выражение с учетом приоритетов операторов Python, Парсит выражение на текущем уровне приоритета операторов, Парсит унарные операторы, Проверяет, что оператор в данной позиции является валидным оператором, Парсит цепочки индексации типа a[0][1][2] (+8 more)

### Community 10 - ".validate_scope"
Cohesion: 0.06
Nodes (21): Проверяет дублирование переменных в local_variables, Валидирует таблицу символов scope'а, Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль (+13 more)

### Community 12 - "Validator"
Cohesion: 0.11
Nodes (10): Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation., Collect variable references from any expression AST variant. (+2 more)

### Community 13 - "._parse_line_impl"
Cohesion: 0.12
Nodes (13): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Парсит оператор del (полное удаление), Находит конец блока с отступом, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Парсит вложенные if внутри других блоков (while, for, других if), Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in… (+5 more)

### Community 14 - "main.py"
Cohesion: 0.06
Nodes (81): ArgumentParser, build_argument_parser(), cli(), _command(), compile_c(), compile_pipeline(), _compiler_settings(), default_output_paths() (+73 more)

### Community 16 - ".extract_dependencies_from_ast"
Cohesion: 0.09
Nodes (12): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value (+4 more)

### Community 17 - "ListCodegenMixin"
Cohesion: 0.16
Nodes (9): ListCodegenMixin, Генерирует структуру C для списка любой вложенности, Генерирует все функции для всех зарегистрированных структур списков, Рекурсивно генерирует элементы вложенного списка, Корректно генерирует элементы вложенного списка, Генерирует имя структуры для списка любой вложенности, Генерирует код для повторного объявления списка, Генерирует функции для работы со списком (без дублирования) (+1 more)

### Community 18 - "IndexingMixin"
Cohesion: 0.14
Nodes (8): IndexingMixin, Генерирует присваивание по индексу: list[index] = value или dict[key] = value, Генерирует код для многомерного индексного присваивания: A_data[0][0] = 10, Генерирует присваивание для вложенной индексации любой глубины, Генерирует присваивание среза: list[start:stop] = values, Генерирует составное присваивание по индексу: list[index] += value, Генерирует код для доступа по индексу, Генерирует выражение для вложенной индексации (для использования в выражениях)

### Community 19 - "IoMixin"
Cohesion: 0.16
Nodes (6): IoMixin, Генерирует код для чтения ввода с клавиатуры прямо в целевую переменную, Генерирует выражение с input() и возвращает имя переменной с результатом, Генерирует правильную конкатенацию строк, Генерирует вызов input() как отдельный statement (без присваивания), Генерирует код для чтения ввода с клавиатуры

### Community 20 - ".parse_function_call"
Cohesion: 0.20
Nodes (5): Парсит вызов встроенной функции, Разбирает аргументы функции с учетом строк и вложенных вызовов, Универсальный парсер любого вызова функции с поддержкой опций, Парсит прямой вызов C-функции, Определяет тип возвращаемого значения для встроенной функции

### Community 21 - "StatementsMixin"
Cohesion: 0.12
Nodes (10): Генерирует if statement, Release loop-local owners before transferring control., Release current iteration owners before continuing., Ownership-safe class field assignment., Ownership-aware return: evaluate, establish return ownership, cleanup., Generate assignment with ARC/string ownership and borrow checks., Declare a value and establish its ownership state., Redeclaration is a destruction boundary followed by a new owner. (+2 more)

### Community 22 - "ScopeMixin"
Cohesion: 0.23
Nodes (4): Enter a lexical ownership scope., Leave a lexical scope and deterministically release owned values., Generate a function with borrowed parameters and automatic cleanup., ScopeMixin

### Community 23 - ".get_symbol_info"
Cohesion: 0.12
Nodes (7): Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет выход за границы массивов/списков, Проверяет операции со строками, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST, Находит родительский узел (если есть)

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует вспомогательные функции для работы со строками, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - "TupleCodegenMixin"
Cohesion: 0.36
Nodes (5): Генерирует имя структуры для tuple, Генерирует код для повторного объявления кортежа, Create a homogeneous immutable tuple with owned element references., Generate an ARC-owned homogeneous tuple[T]., TupleCodegenMixin

### Community 26 - ".parse_type_annotation"
Cohesion: 0.13
Nodes (10): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Parse ``var self.attr: Type [= value]`` with nested types., Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Parse ``name: Type = default`` with nested generic/borrow types., Выводит тип из AST выражения, Return canonical type text and structured metadata. (+2 more)

### Community 27 - "CoreMixin"
Cohesion: 0.33
Nodes (4): CoreMixin, Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку

### Community 28 - "Handoff.md"
Cohesion: 0.08
Nodes (24): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+16 more)

### Community 29 - "TypedScope"
Cohesion: 0.20
Nodes (4): Typed view of one parser scope., Return the typed, read-only lowering view consumed by the C backend., Find a typed scope by its parser level., TypedScope

### Community 30 - "._validate_scopes"
Cohesion: 0.20
Nodes (5): Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Validate the canonical semantic module before C lowering., Run the existing validation passes over a typed lowering view., Собирает информацию о всех символах в системе

### Community 31 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 32 - ".parse_object_method_call_node"
Cohesion: 0.18
Nodes (5): Парсит составные операции присваивания, Парсит вызов метода объекта с учетом наследования, Строит операции из AST выражения, Рекурсивно ищет символ в текущем и родительских scope'ах, Разрешает информацию о методе с учетом наследования

### Community 33 - "OrchestratorMixin"
Cohesion: 0.19
Nodes (7): OrchestratorMixin, Генерирует имя временной переменной, Генерирует код для узла графа, Generate C from the canonical semantic IR., Lower typed scope views after semantic IR construction., Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 36 - "CCodeGenerator"
Cohesion: 0.19
Nodes (7): DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря, CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, NamingMixin, Namespace every Phils-generated C type/function with ``ocean_``. The pass first…

### Community 37 - "tensor_runtime.c"
Cohesion: 0.13
Nodes (66): cl_int, cl_kernel, ocean_tensor_dtype, ocean_tensor_handle_t, ocean_tensor_alloc(), ocean_tensor_alloc_zeros(), ocean_tensor_apply_binary(), ocean_tensor_binary() (+58 more)

### Community 39 - ".parse_complex_expression"
Cohesion: 0.14
Nodes (7): Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок, Проверяет, является ли символ частью идентификатора, Находит позицию оператора вне скобок, строк и комментариев, Проверяет, содержит ли выражение какой-либо оператор, Очищает значение от лишних пробелов, но для сложных выражений возвращает AST

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 41 - "modules/imports.py"
Cohesion: 0.17
Nodes (5): CImportProcessor, Просто регистрирует C импорт без парсинга, Reset all per-compilation parser state. A Parser instance can safely be reused…, Build the parser's private graph before typed lowering., Parse source into the canonical ``TypedModule`` API.

### Community 42 - "Any"
Cohesion: 0.31
Nodes (6): _build_typed_module(), Any, Lower the parser's private graph into typed scopes and nodes., Wrap expression payloads recursively while preserving AST keys., Build the typed module from the parser's internal graph., TypedIRBuilder

### Community 43 - "test_memory_safety.py"
Cohesion: 0.27
Nodes (12): compile_ocean(), test_borrow_cannot_escape_through_return(), test_borrow_cannot_escape_to_non_borrowing_parameter(), test_borrow_is_released_at_block_exit(), test_direct_c_call_requires_unsafe_block(), test_immutable_borrow_cannot_be_passed_to_mutable_parameter(), test_mutable_and_immutable_borrows_are_exclusive(), test_owned_array_is_moved_into_by_value_parameter() (+4 more)

### Community 44 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Safety & Configuration Notes, Testing Guidelines

### Community 45 - "Ocean 🌊"
Cohesion: 0.04
Nodes (40): Borrow, Categories, Containers, FFI, Move, Ocean automatic ownership model v1, Reference alias, Return ABI (+32 more)

### Community 46 - "4. Memory model"
Cohesion: 0.29
Nodes (7): 4. Memory model, BORROWED, Immutable borrow, Mutable borrow, OWNED, SHARED, VALUE

### Community 47 - "20. Тесты"
Cohesion: 0.33
Nodes (6): 20. Тесты, Level 1 — AST / parser, Level 2 — C generation, Level 3 — compile, Level 4 — memory safety, Обязательные memory tests

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Статус array/tensor на момент handoff"
Cohesion: 0.67
Nodes (3): 28. Статус array/tensor на момент handoff, Backend, Parser

### Community 51 - "30. Следующий рекомендуемый этап"
Cohesion: 0.67
Nodes (3): 30. Следующий рекомендуемый этап, Array, Tensor

### Community 55 - ".validate_type_compatibility"
Cohesion: 0.33
Nodes (3): Валидирует объявление переменной, Проверяет совместимость типов при присваивании, Пытается определить тип по значению

### Community 60 - "RuntimeError"
Cohesion: 0.10
Nodes (10): RuntimeError, Генерирует прямой вызов C-функции, Генерирует выражение из AST для конструктора с подстановкой параметров, Генерирует доступ к элементу сложного атрибута (self.data[index]), Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Render validated structured OpenMP metadata as one C pragma., Return the validated collapse count for an OpenMP directive. (+2 more)

### Community 61 - "benchmark_main.py"
Cohesion: 0.10
Nodes (20): main(), measure(), Path, Benchmark the generated C program for examples/matmul.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, device (+12 more)

### Community 62 - "parser.py"
Cohesion: 0.25
Nodes (6): infer_literal_shape(), Split text only when not nested in (), [], {}, <> or strings., Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, split_top_level(), TypeParser

### Community 63 - "class_model.py"
Cohesion: 0.29
Nodes (9): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+1 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (5): ClassRegistry, FieldModel, Canonical class metadata and lookup service for the C backend., A field declared directly by one class., Reset all per-compilation mutable state.

### Community 65 - "test_debug_validator.py"
Cohesion: 0.39
Nodes (7): test_validator_accepts_typed_borrow_and_generic_symbols(), test_validator_checks_container_and_index_types_from_ast(), test_validator_does_not_leak_symbols_between_functions(), test_validator_reports_file_line_and_column(), test_validator_reports_real_source_line_for_type_error(), test_validator_returns_report_for_malformed_input(), validate()

### Community 66 - "ClassModel"
Cohesion: 0.28
Nodes (4): ClassModel, Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "TypedModule"
Cohesion: 0.13
Nodes (12): Typed intermediate representation for the Ocean compiler. The parser's…, Typed compilation unit exchanged by compiler passes., Iterate semantic nodes in source/scope order., TypedModule, test_parser_typed_entrypoint_is_ready_for_compiler_callers(), test_removed_native_tensor_is_rejected(), test_typed_ir_exposes_source_location_metadata(), test_typed_ir_exposes_typed_backend_views() (+4 more)

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 69 - "compiler.py"
Cohesion: 0.13
Nodes (5): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, IRType, A compiler-facing type backed by the parser's canonical ``TypeSpec``., generate(), test_array_lowering_and_index_mutation()

### Community 70 - "ExpressionsMixin"
Cohesion: 0.33
Nodes (4): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 74 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из typed AST.

### Community 76 - "TypedNode"
Cohesion: 0.20
Nodes (3): Structured OpenMP metadata attached to a loop, if present., Semantic metadata and read-only mapping view for one graph node., TypedNode

### Community 77 - ".resolved_methods"
Cohesion: 0.50
Nodes (3): Resolve methods without rebuilding parser-shaped dictionaries., A method together with the class that provides its implementation., ResolvedMethod

## Knowledge Gaps
- **93 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Validator` connect `Validator` to `test_debug_validator.py`, `.add_error`, `TypedModule`, `.get_type_from_ast`, `.validate_scope`, `test_memory_safety.py`, `main.py`, `.validate_type_compatibility`, `.get_symbol_info`, `benchmark_main.py`, `._validate_scopes`, `.check_undefined_methods`?**
  _High betweenness centrality (0.285) - this node is a cross-community bridge._
- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `CallsMixin`, `TypesMixin`, `OwnershipMixin`, `Parser`, `generator.py`, `main.py`, `ArrayCodegenMixin`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `OrchestratorMixin`, `TensorCodegenMixin`, `test_memory_safety.py`, `benchmark_main.py`, `TypedModule`, `compiler.py`, `ExpressionsMixin`, `ImportsMixin`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `.parse_object_method_call_node`, `run`, `test_debug_validator.py`, `TypedModule`, `compiler.py`, `SymbolTable`, `.parse_complex_expression`, `modules/imports.py`, `.parse_expression_to_ast`, `test_memory_safety.py`, `._parse_line_impl`, `main.py`, `.extract_dependencies_from_ast`, `.parse_function_call`, `.parse_type_annotation`, `benchmark_main.py`, `parser.py`?**
  _High betweenness centrality (0.249) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _93 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run` be split into smaller, more focused modules?**
  _Cohesion score 0.0528169014084507 - nodes in this community are weakly interconnected._
- **Should `OopMixin` be split into smaller, more focused modules?**
  _Cohesion score 0.1368421052631579 - nodes in this community are weakly interconnected._