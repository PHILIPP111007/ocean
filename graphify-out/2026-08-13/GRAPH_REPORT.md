# Graph Report - phils_language  (2026-08-13)

## Corpus Check
- 79 files · ~97,842 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1324 nodes · 2739 edges · 77 communities (66 shown, 11 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 98 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `71b20d6b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- RuntimeError
- .validate_scope
- TypesMixin
- .validate_graph
- OwnershipMixin
- SymbolTable
- Parser
- .parse_expression_to_ast
- .check_undefined_methods
- generator.py
- .add_error
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
- build_typed_ir
- .get_symbol_info
- .validate_static_method_call
- .generate_builtin_function_call
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- NamingMixin
- tensor_runtime.c
- ._parse_with_priorities
- What changed
- TypeSpec
- TypedIRBuilder
- test_memory_safety.py
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- DictCodegenMixin
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- .validate_openmp_loop
- TypedModule
- ._validate_scopes
- ocean-lang
- test_array_tensor.py
- TransformerLanguageModel
- parser.py
- class_model.py
- ClassRegistry
- modules/imports.py
- ClassModel
- CCodeGenerator
- .generate_all_methods
- IRType
- ExpressionsMixin
- benchmark_main.py
- OpenCL tensor backend
- ImportsMixin
- test_debug_validator.py
- .validate_function_return_type

## God Nodes (most connected - your core abstractions)
1. `JSONValidator` - 144 edges
2. `Parser` - 132 edges
3. `run()` - 60 edges
4. `ocean_tensor_fail()` - 42 edges
5. `CCodeGenerator` - 40 edges
6. `TensorCodegenMixin` - 30 edges
7. `OwnershipMixin` - 29 edges
8. `ArrayCodegenMixin` - 25 edges
9. `build_typed_ir()` - 25 edges
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
- `test_tensor_views_keep_owner_and_broadcast_shapes()` --calls--> `CCodeGenerator`  [INFERRED]
  tests/test_array_tensor.py → src/codegen/generator.py

## Import Cycles
- None detected.

## Communities (77 total, 11 thin omitted)

### Community 0 - "run"
Cohesion: 0.05
Nodes (48): ImportProcessor, Path, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield repository std directories from the active source context., run(), test_c_code_math(), test_c_code_pthread() (+40 more)

### Community 1 - "OopMixin"
Cohesion: 0.16
Nodes (8): OopMixin, Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models., Generate an ARC-compatible class layout with safe single inheritance.

### Community 2 - "RuntimeError"
Cohesion: 0.10
Nodes (12): RuntimeError, CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции (+4 more)

### Community 3 - ".validate_scope"
Cohesion: 0.06
Nodes (21): Проверяет дублирование переменных в local_variables, Валидирует таблицу символов scope'а, Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль (+13 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Определяет, является ли тип классом, Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".validate_graph"
Cohesion: 0.07
Nodes (16): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов, Валидирует один аргумент (может быть строкой или AST) (+8 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters., Register the common ``ocean_`` ARC runtime in generated helpers. (+1 more)

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.07
Nodes (13): Parser, Remove standalone triple-quoted blocks while preserving line count., Парсит оператор break, Парсит оператор continue, Парсит оператор del (полное удаление), Возвращает область видимости для заданного уровня отступа, Определяет текущий scope на основе отступа, Parse the supported OpenMP loop directive into structured metadata. (+5 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.10
Nodes (12): Парсит литерал кортежа, Парсит оператор return, Parse an expression into the transitional Phils AST., Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке), Парсит значение опции и определяет его тип, Парсит присваивание результата вызова функции: var x: type = func(args), Парсит условие для циклов и if (+4 more)

### Community 10 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 12 - ".add_error"
Cohesion: 0.08
Nodes (22): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует объявление переменной (+14 more)

### Community 13 - ".calculate_indent_level"
Cohesion: 0.11
Nodes (13): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Parse one line and attach its source location to emitted nodes., Парсит вложенные if внутри других блоков (while, for, других if), Parse a value-semantic struct. Structs intentionally contain fields only in… (+5 more)

### Community 14 - "main.py"
Cohesion: 0.07
Nodes (65): ArgumentParser, build_argument_parser(), cli(), _command(), compile_c(), compile_pipeline(), _compiler_settings(), default_output_paths() (+57 more)

### Community 16 - "._parse_line_impl"
Cohesion: 0.08
Nodes (18): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value (+10 more)

### Community 17 - "ListCodegenMixin"
Cohesion: 0.16
Nodes (9): ListCodegenMixin, Генерирует структуру C для списка любой вложенности, Генерирует все функции для всех зарегистрированных структур списков, Рекурсивно генерирует элементы вложенного списка, Корректно генерирует элементы вложенного списка, Генерирует имя структуры для списка любой вложенности, Генерирует код для повторного объявления списка, Генерирует функции для работы со списком (без дублирования) (+1 more)

### Community 18 - "IndexingMixin"
Cohesion: 0.14
Nodes (8): IndexingMixin, Генерирует присваивание по индексу: list[index] = value или dict[key] = value, Генерирует код для многомерного индексного присваивания: A_data[0][0] = 10, Генерирует присваивание для вложенной индексации любой глубины, Генерирует присваивание среза: list[start:stop] = values, Генерирует составное присваивание по индексу: list[index] += value, Генерирует код для доступа по индексу, Генерирует выражение для вложенной индексации (для использования в выражениях)

### Community 19 - "IoMixin"
Cohesion: 0.16
Nodes (6): IoMixin, Генерирует выражение с input() и возвращает имя переменной с результатом, Генерирует код для чтения ввода с клавиатуры прямо в целевую переменную, Генерирует правильную конкатенацию строк, Генерирует вызов input() как отдельный statement (без присваивания), Генерирует код для чтения ввода с клавиатуры

### Community 20 - ".parse_function_call"
Cohesion: 0.20
Nodes (5): Парсит вызов встроенной функции, Разбирает аргументы функции с учетом строк и вложенных вызовов, Универсальный парсер любого вызова функции с поддержкой опций, Парсит прямой вызов C-функции, Определяет тип возвращаемого значения для встроенной функции

### Community 21 - "StatementsMixin"
Cohesion: 0.09
Nodes (14): Генерирует if statement, Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Release loop-local owners before transferring control., Render validated structured OpenMP metadata as one C pragma., Release current iteration owners before continuing., Return the validated collapse count for an OpenMP directive., Ownership-safe class field assignment. (+6 more)

### Community 22 - "ScopeMixin"
Cohesion: 0.23
Nodes (4): Enter a lexical ownership scope., Leave a lexical scope and deterministically release owned values., Generate a function with borrowed parameters and automatic cleanup., ScopeMixin

### Community 23 - "JSONValidator"
Cohesion: 0.21
Nodes (4): JSONValidator, Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Collect variable references from expression ASTs only.

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует вспомогательные функции для работы со строками, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - "TupleCodegenMixin"
Cohesion: 0.36
Nodes (5): Генерирует имя структуры для tuple, Генерирует код для повторного объявления кортежа, Create a homogeneous immutable tuple with owned element references., Generate an ARC-owned homogeneous tuple[T]., TupleCodegenMixin

### Community 26 - ".parse_type_annotation"
Cohesion: 0.12
Nodes (11): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Parse ``var self.attr: Type [= value]`` with nested types., Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Очищает значение от лишних пробелов, но для сложных выражений возвращает AST, Parse ``name: Type = default`` with nested generic/borrow types., Выводит тип из AST выражения (+3 more)

### Community 27 - "CoreMixin"
Cohesion: 0.22
Nodes (5): CoreMixin, Reset all per-compilation mutable state., Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку

### Community 28 - "Handoff.md"
Cohesion: 0.08
Nodes (24): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+16 more)

### Community 29 - "build_typed_ir"
Cohesion: 0.25
Nodes (16): build_typed_ir(), Convenience entry point used by the compiler pipeline and tests., _parse(), test_openmp_collapse_allows_sequential_loop_after_collapsed_nest(), test_openmp_collapse_requires_enough_nested_loops(), test_openmp_collapse_requires_perfect_nesting(), test_openmp_collapse_two_nested_loops_is_emitted_and_accepted(), test_openmp_pragma_is_attached_to_for_and_emitted_as_omp() (+8 more)

### Community 30 - ".get_symbol_info"
Cohesion: 0.12
Nodes (7): Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет выход за границы массивов/списков, Проверяет операции со строками, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST, Находит родительский узел (если есть)

### Community 33 - "OrchestratorMixin"
Cohesion: 0.17
Nodes (8): OrchestratorMixin, Генерирует имя временной переменной, Generate C from the canonical semantic IR., Генерирует код для узла графа, Compatibility entry point for callers with parser JSON., Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной, Lower the compatibility scope view after typed IR is established.

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 35 - "TensorCodegenMixin"
Cohesion: 0.11
Nodes (8): Generate a device Tensor expression with declaration type context.…, Lower tensor arithmetic to shape-checked broadcasting helpers., Emit view-aware reductions, matrix operations, and broadcasting., Resolve dtype from an explicit receiver or the expected target., Lower Tensor.zeros/from_list without runtime dtype erasure., Prepare checked-once direct access for provably bounded 2D loops., Lower dense row-major ``tensor[T]`` values with owned storage., TensorCodegenMixin

### Community 37 - "tensor_runtime.c"
Cohesion: 0.15
Nodes (58): cl_int, cl_kernel, ocean_tensor_dtype, ocean_tensor_handle_t, ocean_tensor_alloc(), ocean_tensor_alloc_zeros(), ocean_tensor_apply_binary(), ocean_tensor_binary() (+50 more)

### Community 39 - "._parse_with_priorities"
Cohesion: 0.10
Nodes (11): Парсит выражение с учетом приоритетов операторов Python, Парсит выражение на текущем уровне приоритета операторов, Парсит унарные операторы, Проверяет, что оператор в данной позиции является валидным оператором, Парсит цепочки индексации типа a[0][1][2], Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок (+3 more)

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 41 - "TypeSpec"
Cohesion: 0.14
Nodes (5): Structured representation of a Phils type. The parser still emits the canonical…, TypeSpec, Find a typed scope by its parser level., Typed view of one parser scope., TypedScope

### Community 42 - "TypedIRBuilder"
Cohesion: 0.26
Nodes (6): Any, Lower parser dictionaries into typed scopes and effect-annotated nodes., Semantic metadata for one legacy graph node., Structured OpenMP metadata attached to a loop, if present., TypedIRBuilder, TypedNode

### Community 43 - "test_memory_safety.py"
Cohesion: 0.27
Nodes (12): compile_ocean(), test_borrow_cannot_escape_through_return(), test_borrow_cannot_escape_to_non_borrowing_parameter(), test_borrow_is_released_at_block_exit(), test_direct_c_call_requires_unsafe_block(), test_immutable_borrow_cannot_be_passed_to_mutable_parameter(), test_mutable_and_immutable_borrows_are_exclusive(), test_owned_array_is_moved_into_by_value_parameter() (+4 more)

### Community 44 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Safety & Configuration Notes, Testing Guidelines

### Community 45 - "Ocean 🌊"
Cohesion: 0.05
Nodes (37): Borrow, Categories, Containers, FFI, Legacy native tensor views, Move, Ocean automatic ownership model v1, Reference alias (+29 more)

### Community 46 - "4. Memory model"
Cohesion: 0.29
Nodes (7): 4. Memory model, BORROWED, Immutable borrow, Mutable borrow, OWNED, SHARED, VALUE

### Community 47 - "20. Тесты"
Cohesion: 0.33
Nodes (6): 20. Тесты, Level 1 — AST / parser, Level 2 — C generation, Level 3 — compile, Level 4 — memory safety, Обязательные memory tests

### Community 48 - "DictCodegenMixin"
Cohesion: 0.50
Nodes (3): DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря

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
Cohesion: 0.13
Nodes (7): Валидирует узел цикла, Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation., Collect variable references from any expression AST variant.

### Community 56 - "TypedModule"
Cohesion: 0.14
Nodes (7): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, Typed intermediate representation for the Ocean compiler. The parser's…, Iterate semantic nodes in source/scope order., Return the compatibility graph used by legacy callers., Typed compilation unit with an explicit compatibility projection. Compiler…, Return immutable-pass input for legacy lowering code. The returned dictionaries…, TypedModule

### Community 57 - "._validate_scopes"
Cohesion: 0.18
Nodes (6): Собирает информацию о всех символах в системе, Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Validate a typed module or legacy parser graph. ``TypedModule`` is the…, Validate the canonical semantic module before C lowering., Run the existing validation passes over a typed lowering view.

### Community 60 - "test_array_tensor.py"
Cohesion: 0.33
Nodes (8): generate(), test_array_lowering_and_index_mutation(), test_tensor_lowering_shape_and_index_mutation(), test_tensor_numeric_methods_use_shape_and_stride_runtime(), test_tensor_rank_specialization_is_unbounded(), test_tensor_the_big_code(), test_tensor_views_keep_owner_and_broadcast_shapes(), test_tensor_zeros_dynamic_shape()

### Community 61 - "TransformerLanguageModel"
Cohesion: 0.15
Nodes (13): device, main(), A small decoder-only Transformer language model implemented with PyTorch. This…, GPT-style causal language model built from PyTorch modules., Return an additive upper-triangular causal attention mask., Return next-token logits with shape ``(batch, sequence, vocab)``., Compute teacher-forced autoregressive cross-entropy., Greedily append tokens selected from the final position's logits. (+5 more)

### Community 62 - "parser.py"
Cohesion: 0.25
Nodes (6): infer_literal_shape(), Split text only when not nested in (), [], {}, <> or strings., Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, split_top_level(), TypeParser

### Community 63 - "class_model.py"
Cohesion: 0.23
Nodes (11): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+3 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (4): ClassRegistry, Canonical class metadata and lookup service for the C backend., Resolve methods without rebuilding parser-shaped dictionaries., Build the canonical OOP metadata directly from parser output.

### Community 65 - "modules/imports.py"
Cohesion: 0.20
Nodes (4): CImportProcessor, Просто регистрирует C импорт без парсинга, Reset all per-compilation parser state. A Parser instance can safely be reused…, Parse one Phils compilation unit into the legacy graph + typed metadata. The…

### Community 66 - "ClassModel"
Cohesion: 0.24
Nodes (6): ClassModel, FieldModel, A field declared directly by one class., Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "CCodeGenerator"
Cohesion: 0.31
Nodes (10): CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, compile_and_run(), test_oop_constructor_method_mutation_and_composition(), test_oop_default_constructor_without_init(), test_oop_inherited_field_access_uses_embedded_base_layout(), test_oop_metadata_has_one_canonical_class_model(), test_oop_rejects_inheritance_cycles() (+2 more)

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 70 - "ExpressionsMixin"
Cohesion: 0.33
Nodes (4): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 71 - "benchmark_main.py"
Cohesion: 0.36
Nodes (7): main(), measure(), Path, Benchmark the generated C program for examples/main.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess

### Community 72 - "OpenCL tensor backend"
Cohesion: 0.33
Nodes (5): Backend selection, Kernel contract, OpenCL tensor backend, Runtime objects, Safety boundary

### Community 74 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из JSON

### Community 75 - "test_debug_validator.py"
Cohesion: 0.48
Nodes (6): test_validator_accepts_typed_borrow_and_generic_symbols(), test_validator_checks_container_and_index_types_from_ast(), test_validator_does_not_leak_symbols_between_functions(), test_validator_reports_real_source_line_for_type_error(), test_validator_returns_report_for_malformed_input(), validate()

## Knowledge Gaps
- **95 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+90 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `RuntimeError`, `TypesMixin`, `OwnershipMixin`, `generator.py`, `main.py`, `ArrayCodegenMixin`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `build_typed_ir`, `OrchestratorMixin`, `TensorCodegenMixin`, `NamingMixin`, `test_memory_safety.py`, `DictCodegenMixin`, `TypedModule`, `test_array_tensor.py`, `ExpressionsMixin`, `benchmark_main.py`, `ImportsMixin`?**
  _High betweenness centrality (0.270) - this node is a cross-community bridge._
- **Why does `JSONValidator` connect `JSONValidator` to `.validate_scope`, `.validate_graph`, `benchmark_main.py`, `.check_undefined_methods`, `test_debug_validator.py`, `.add_error`, `.validate_function_return_type`, `main.py`, `test_memory_safety.py`, `.validate_openmp_loop`, `TypedModule`, `._validate_scopes`, `test_array_tensor.py`, `build_typed_ir`, `.get_symbol_info`, `.validate_static_method_call`?**
  _High betweenness centrality (0.253) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `modules/imports.py`, `CCodeGenerator`, `SymbolTable`, `benchmark_main.py`, `._parse_with_priorities`, `.parse_expression_to_ast`, `test_debug_validator.py`, `test_memory_safety.py`, `.calculate_indent_level`, `main.py`, `._parse_line_impl`, `.parse_function_call`, `.parse_type_annotation`, `test_array_tensor.py`, `build_typed_ir`, `parser.py`?**
  _High betweenness centrality (0.227) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `RuntimeError` (e.g. with `run_benchmark()` and `compile_pipeline()`) actually correct?**
  _`RuntimeError` has 43 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _95 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run` be split into smaller, more focused modules?**
  _Cohesion score 0.0517503805175038 - nodes in this community are weakly interconnected._