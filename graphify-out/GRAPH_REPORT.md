# Graph Report - phils_language  (2026-08-13)

## Corpus Check
- 79 files · ~96,322 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1307 nodes · 2668 edges · 76 communities (66 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 95 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `75c24bbe`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- RuntimeError
- .validate_scope
- TypesMixin
- .add_error
- OwnershipMixin
- SymbolTable
- Parser
- .parse_expression_to_ast
- .check_undefined_methods
- generator.py
- .get_type_from_ast
- .calculate_indent_level
- main.py
- ArrayCodegenMixin
- ._parse_line_impl
- ListCodegenMixin
- IndexingMixin
- IoMixin
- .parse_function_call
- StatementsMixin
- ScopeMixin
- JSONValidator
- HelpersMixin
- TupleCodegenMixin
- .parse_type_annotation
- CoreMixin
- Handoff.md
- test_openmp.py
- .get_symbol_info
- test_array_tensor.py
- .generate_builtin_function_call
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- NamingMixin
- tensor_runtime.c
- .parse_function_arguments_to_ast
- What changed
- .generate_for_loop
- TypedIRBuilder
- test_memory_safety.py
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- parser.py
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- .validate_openmp_loop
- .validate
- ImportProcessor
- ocean-lang
- .validate_static_method_call
- benchmark_main.py
- TypeSpec
- class_model.py
- ClassRegistry
- modules/imports.py
- ClassModel
- CCodeGenerator
- .generate_all_methods
- IRType
- ExpressionsMixin
- TypedNode
- .resolved_methods
- ImportsMixin
- DictCodegenMixin

## God Nodes (most connected - your core abstractions)
1. `JSONValidator` - 142 edges
2. `Parser` - 132 edges
3. `run()` - 60 edges
4. `CCodeGenerator` - 38 edges
5. `ocean_tensor_fail()` - 36 edges
6. `OwnershipMixin` - 29 edges
7. `TensorCodegenMixin` - 28 edges
8. `ArrayCodegenMixin` - 25 edges
9. `TypesMixin` - 24 edges
10. `ClassRegistry` - 20 edges

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

## Communities (76 total, 10 thin omitted)

### Community 0 - "run"
Cohesion: 0.06
Nodes (43): run(), test_c_code_math(), test_c_code_pthread(), test_del(), test_dict(), test_dict_get(), test_for_loop_1(), test_for_loop_2() (+35 more)

### Community 1 - "OopMixin"
Cohesion: 0.14
Nodes (9): OopMixin, Build the canonical OOP metadata directly from parser output., Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models. (+1 more)

### Community 2 - "RuntimeError"
Cohesion: 0.10
Nodes (12): RuntimeError, CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции (+4 more)

### Community 3 - ".validate_scope"
Cohesion: 0.06
Nodes (23): Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Добавляет предупреждение с информацией о строке, Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль, Проверяет условия циклов на потенциальные проблемы (+15 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Определяет, является ли тип классом, Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".add_error"
Cohesion: 0.07
Nodes (24): Валидирует граф операций, Валидирует объявление переменной, Валидирует выражение (правая часть присваивания или инициализации), Валидирует присваивание, Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции (+16 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters., Register the common ``ocean_`` ARC runtime in generated helpers. (+1 more)

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.07
Nodes (16): Parser, Remove standalone triple-quoted blocks while preserving line count., Парсит выражение на текущем уровне приоритета операторов, Проверяет, что оператор в данной позиции является валидным оператором, Определяет текущий scope на основе отступа, Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, является ли имя именем класса, Извлекает содержимое внутри скобок, учитывая вложенность (+8 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.10
Nodes (13): Парсит литерал кортежа, Parse an expression into the transitional Phils AST., Парсит выражение с учетом приоритетов операторов Python, Парсит унарные операторы, Парсит цепочки индексации типа a[0][1][2], Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке), Парсит значение опции и определяет его тип (+5 more)

### Community 10 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 12 - ".get_type_from_ast"
Cohesion: 0.09
Nodes (16): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+8 more)

### Community 13 - ".calculate_indent_level"
Cohesion: 0.12
Nodes (12): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in…, Парсит объявление класса (+4 more)

### Community 14 - "main.py"
Cohesion: 0.07
Nodes (62): ArgumentParser, build_argument_parser(), cli(), _command(), compile_c(), compile_pipeline(), _compiler_settings(), default_output_paths() (+54 more)

### Community 16 - "._parse_line_impl"
Cohesion: 0.06
Nodes (20): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит присваивание значения указателя переменной: x = *p, Парсит оператор break (+12 more)

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
Cohesion: 0.12
Nodes (10): Генерирует if statement, Release loop-local owners before transferring control., Release current iteration owners before continuing., Ownership-safe class field assignment., Ownership-aware return: evaluate, establish return ownership, cleanup., Generate assignment with ARC/string ownership and borrow checks., Declare a value and establish its ownership state., Redeclaration is a destruction boundary followed by a new owner. (+2 more)

### Community 22 - "ScopeMixin"
Cohesion: 0.23
Nodes (4): Enter a lexical ownership scope., Leave a lexical scope and deterministically release owned values., Generate a function with borrowed parameters and automatic cleanup., ScopeMixin

### Community 23 - "JSONValidator"
Cohesion: 0.13
Nodes (8): JSONValidator, Validate the parser's typed graph before C code generation. The validator is…, Находит родительский scope для заданного уровня, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Извлекает зависимости (имена переменных) из AST, Собирает переменные из AST, Collect variable references from any expression AST variant., Collect variable references from expression ASTs only.

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует вспомогательные функции для работы со строками, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - "TupleCodegenMixin"
Cohesion: 0.36
Nodes (5): Генерирует имя структуры для tuple, Генерирует код для повторного объявления кортежа, Create a homogeneous immutable tuple with owned element references., Generate an ARC-owned homogeneous tuple[T]., TupleCodegenMixin

### Community 26 - ".parse_type_annotation"
Cohesion: 0.11
Nodes (12): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Парсит сложные выражения с несколькими операциями, Parse ``var self.attr: Type [= value]`` with nested types., Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Очищает значение от лишних пробелов, но для сложных выражений возвращает AST, Parse ``name: Type = default`` with nested generic/borrow types. (+4 more)

### Community 27 - "CoreMixin"
Cohesion: 0.33
Nodes (4): CoreMixin, Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку

### Community 28 - "Handoff.md"
Cohesion: 0.08
Nodes (24): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+16 more)

### Community 29 - "test_openmp.py"
Cohesion: 0.13
Nodes (23): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, build_typed_ir(), Typed intermediate representation for the Ocean compiler. The parser's…, Convenience entry point used by the compiler pipeline and tests., Typed compilation unit with a lossless legacy representation., TypedModule, test_validator_accepts_typed_borrow_and_generic_symbols(), test_validator_checks_container_and_index_types_from_ast() (+15 more)

### Community 30 - ".get_symbol_info"
Cohesion: 0.13
Nodes (6): Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет корректное использование указателей, Проверяет выход за границы массивов/списков, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST

### Community 31 - "test_array_tensor.py"
Cohesion: 0.33
Nodes (8): generate(), test_array_lowering_and_index_mutation(), test_tensor_lowering_shape_and_index_mutation(), test_tensor_numeric_methods_use_shape_and_stride_runtime(), test_tensor_rank_specialization_is_unbounded(), test_tensor_the_big_code(), test_tensor_views_keep_owner_and_broadcast_shapes(), test_tensor_zeros_dynamic_shape()

### Community 33 - "OrchestratorMixin"
Cohesion: 0.19
Nodes (7): OrchestratorMixin, Generate C from the semantic IR while preserving the legacy backend., Генерирует имя временной переменной, Генерирует код для узла графа, Run semantic prepasses, instantiate runtime types, then emit C., Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 35 - "TensorCodegenMixin"
Cohesion: 0.14
Nodes (5): Emit view-aware reductions, matrix operations, and broadcasting., Prepare checked-once direct access for provably bounded 2D loops., Lower dense row-major ``tensor[T]`` values with owned storage., Lower tensor arithmetic to shape-checked broadcasting helpers., TensorCodegenMixin

### Community 37 - "tensor_runtime.c"
Cohesion: 0.16
Nodes (51): cl_int, cl_kernel, ocean_tensor_dtype, ocean_tensor_handle_t, ocean_tensor_alloc(), ocean_tensor_alloc_zeros(), ocean_tensor_apply_binary(), ocean_tensor_binary() (+43 more)

### Community 39 - ".parse_function_arguments_to_ast"
Cohesion: 0.12
Nodes (8): Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит аргументы функции в список AST, Парсит создание объекта: ClassName(arg1, arg2, ...), Парсит вызов статического метода: ClassName.method(args), Парсит вызов метода объекта с учетом наследования, Парсит вызов статического метода: Class.method(args), Разрешает информацию о методе с учетом наследования

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 41 - ".generate_for_loop"
Cohesion: 0.25
Nodes (4): Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Render validated structured OpenMP metadata as one C pragma., Return the validated collapse count for an OpenMP directive.

### Community 42 - "TypedIRBuilder"
Cohesion: 0.35
Nodes (4): Any, Lower parser dictionaries into typed scopes and effect-annotated nodes., Return a deep copy accepted by the existing validator and C backend., TypedIRBuilder

### Community 43 - "test_memory_safety.py"
Cohesion: 0.27
Nodes (12): compile_ocean(), test_borrow_cannot_escape_through_return(), test_borrow_cannot_escape_to_non_borrowing_parameter(), test_borrow_is_released_at_block_exit(), test_direct_c_call_requires_unsafe_block(), test_immutable_borrow_cannot_be_passed_to_mutable_parameter(), test_mutable_and_immutable_borrows_are_exclusive(), test_owned_array_is_moved_into_by_value_parameter() (+4 more)

### Community 44 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Safety & Configuration Notes, Testing Guidelines

### Community 45 - "Ocean 🌊"
Cohesion: 0.04
Nodes (42): Borrow, Categories, Containers, FFI, Move, Ocean automatic ownership model v1, Reference alias, Return ABI (+34 more)

### Community 46 - "4. Memory model"
Cohesion: 0.29
Nodes (7): 4. Memory model, BORROWED, Immutable borrow, Mutable borrow, OWNED, SHARED, VALUE

### Community 47 - "20. Тесты"
Cohesion: 0.33
Nodes (6): 20. Тесты, Level 1 — AST / parser, Level 2 — C generation, Level 3 — compile, Level 4 — memory safety, Обязательные memory tests

### Community 48 - "parser.py"
Cohesion: 0.27
Nodes (6): infer_literal_shape(), Split text only when not nested in (), [], {}, <> or strings., Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, split_top_level(), TypeParser

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Статус array/tensor на момент handoff"
Cohesion: 0.67
Nodes (3): 28. Статус array/tensor на момент handoff, Backend, Parser

### Community 51 - "30. Следующий рекомендуемый этап"
Cohesion: 0.67
Nodes (3): 30. Следующий рекомендуемый этап, Array, Tensor

### Community 55 - ".validate_openmp_loop"
Cohesion: 0.19
Nodes (5): Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation.

### Community 56 - ".validate"
Cohesion: 0.25
Nodes (4): Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Основной метод валидации, Собирает информацию о всех символах в системе

### Community 57 - "ImportProcessor"
Cohesion: 0.26
Nodes (5): ImportProcessor, Path, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield repository std directories from the active source context.

### Community 61 - "benchmark_main.py"
Cohesion: 0.10
Nodes (20): main(), measure(), Path, Benchmark the generated C program for examples/main.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, device (+12 more)

### Community 62 - "TypeSpec"
Cohesion: 0.17
Nodes (4): Structured representation of a Phils type. The parser still emits the canonical…, TypeSpec, Typed view of one parser scope., TypedScope

### Community 63 - "class_model.py"
Cohesion: 0.29
Nodes (9): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+1 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (5): ClassRegistry, FieldModel, Canonical class metadata and lookup service for the C backend., A field declared directly by one class., Reset all per-compilation mutable state.

### Community 65 - "modules/imports.py"
Cohesion: 0.20
Nodes (4): CImportProcessor, Просто регистрирует C импорт без парсинга, Reset all per-compilation parser state. A Parser instance can safely be reused…, Parse one Phils compilation unit into the legacy graph + typed metadata. The…

### Community 66 - "ClassModel"
Cohesion: 0.28
Nodes (4): ClassModel, Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "CCodeGenerator"
Cohesion: 0.31
Nodes (10): CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, compile_and_run(), test_oop_constructor_method_mutation_and_composition(), test_oop_default_constructor_without_init(), test_oop_inherited_field_access_uses_embedded_base_layout(), test_oop_metadata_has_one_canonical_class_model(), test_oop_rejects_inheritance_cycles() (+2 more)

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 70 - "ExpressionsMixin"
Cohesion: 0.33
Nodes (4): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 71 - "TypedNode"
Cohesion: 0.40
Nodes (3): Semantic metadata for one legacy graph node., Structured OpenMP metadata attached to a loop, if present., TypedNode

### Community 72 - ".resolved_methods"
Cohesion: 0.50
Nodes (3): Resolve methods without rebuilding parser-shaped dictionaries., A method together with the class that provides its implementation., ResolvedMethod

### Community 74 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из JSON

### Community 75 - "DictCodegenMixin"
Cohesion: 0.50
Nodes (3): DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря

## Knowledge Gaps
- **95 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+90 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `RuntimeError`, `TypesMixin`, `OwnershipMixin`, `generator.py`, `main.py`, `ArrayCodegenMixin`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `test_openmp.py`, `test_array_tensor.py`, `OrchestratorMixin`, `TensorCodegenMixin`, `NamingMixin`, `test_memory_safety.py`, `benchmark_main.py`, `ExpressionsMixin`, `ImportsMixin`, `DictCodegenMixin`?**
  _High betweenness centrality (0.282) - this node is a cross-community bridge._
- **Why does `JSONValidator` connect `JSONValidator` to `.validate_scope`, `.add_error`, `.check_undefined_methods`, `test_memory_safety.py`, `.get_type_from_ast`, `main.py`, `benchmark_main.py`, `.validate_openmp_loop`, `.validate`, `.validate_static_method_call`, `test_openmp.py`, `.get_symbol_info`, `test_array_tensor.py`?**
  _High betweenness centrality (0.268) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `modules/imports.py`, `CCodeGenerator`, `SymbolTable`, `.parse_function_arguments_to_ast`, `.parse_expression_to_ast`, `test_memory_safety.py`, `.calculate_indent_level`, `main.py`, `parser.py`, `._parse_line_impl`, `test_openmp.py`, `.parse_function_call`, `.parse_type_annotation`, `benchmark_main.py`, `test_array_tensor.py`?**
  _High betweenness centrality (0.242) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `RuntimeError` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`RuntimeError` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `CCodeGenerator` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`CCodeGenerator` has 34 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _95 weakly-connected nodes found - possible documentation gaps or missing edges._