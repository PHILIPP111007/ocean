# Graph Report - phils_language  (2026-08-21)

## Corpus Check
- 135 files · ~139,662 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2068 nodes · 4697 edges · 133 communities (114 shown, 19 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 298 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3b787a61`
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
- AGENTS.md — Ocean repository instructions
- Parser
- .parse_expression_to_ast
- .add_warning
- generator.py
- Validator
- .calculate_indent_level
- parse_cli_paths
- RuntimeError
- test_memory_safety.py
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
- .validate_openmp_loop
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- ImportsMixin
- tensor_runtime.c
- CCodeGenerator
- What changed
- SymbolTable
- Any
- web_runtime.c
- Package
- README.md
- autograd_runtime.c
- ocean_tensor_fail
- debug.py
- ocean_tensor_release
- TypeSpec
- ocean_tensor_load_npy
- .parse_complex_expression
- ocean_autograd_mse_loss
- 53. Ближайший roadmap
- file_runtime.c
- TypedExpression
- net_runtime.c
- ocean-lang
- 46. Frontend/compiler quirks, которые уже встречались
- 6. Memory model
- TypeParser
- class_model.py
- ClassRegistry
- .resolved_methods
- ClassModel
- base.py
- .generate_all_methods
- IRType
- json_runtime.c
- ._parse_line_impl
- os_runtime.c
- compile_c
- OpenCL tensor backend
- TypedNode
- Ocean automatic ownership model v1
- `Tensor[T]`
- parser.py
- TypedModule
- _compile
- 29. Known frontend/compiler quirks
- NamingMixin
- time_runtime.c
- main.py
- test_main_cli.py
- logging_runtime.c
- thread_backend.c
- compiler.py
- ExpressionsMixin
- Benchmarks
- net/README.md
- `Socket` API
- MLP
- test_while_loop.py
- 🌐 Backend development in Ocean
- json/README.md
- 2. HTTP client
- DictCodegenMixin
- 8. Request
- 14. Проверка REST API через `curl`
- 29. GPT primitives
- 10. Response
- 5. Routing
- Импорт
- 32. Optimizers
- 🗺️ Roadmap
- 11. JSON response
- 17. Настройка App
- 🧠 ML in Ocean
- test_tensor_npy.py
- test_dict.py
- test_tiny_gpt_v01_ocean.py
- 62. Backend/server development — `std/net`
- ml/README.md
- test_gpu_training_v01_ocean.py
- test_print.py
- OpenMP.md
- ._generate_complex_attribute_access
- .generate_augmented_assignment
- .generate_delete

## God Nodes (most connected - your core abstractions)
1. `Validator` - 147 edges
2. `Parser` - 136 edges
3. `ocean_tensor_fail()` - 118 edges
4. `compile_c()` - 66 edges
5. `compile_pipeline()` - 66 edges
6. `run()` - 60 edges
7. `ocean_tensor_release()` - 57 edges
8. `ocean_tensor_to()` - 47 edges
9. `AGENTS.md — Ocean repository instructions` - 46 edges
10. `CCodeGenerator` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_parser_does_not_assign_source_or_output_paths_by_default()` --calls--> `build_argument_parser()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `test_compile_c_adds_libm_for_generated_math_import()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `test_compile_c_places_explicit_libraries_after_sources()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `test_compile_c_preserves_explicit_optimization_with_openmp()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `compile_pipeline()` --calls--> `CCodeGenerator`  [INFERRED]
  main.py → src/codegen/generator.py

## Import Cycles
- None detected.

## Communities (133 total, 19 thin omitted)

### Community 0 - "run"
Cohesion: 0.12
Nodes (24): run(), test_c_code_math(), test_c_code_pthread(), test_for_loop_1(), test_for_loop_2(), test_for_loop_3(), test_for_loop_4(), test_functions_1() (+16 more)

### Community 1 - "OopMixin"
Cohesion: 0.15
Nodes (8): OopMixin, Build the canonical OOP metadata directly from parser output., Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Generate constructors from the canonical class models., Generate an ARC-compatible class layout with safe single inheritance.

### Community 2 - "CallsMixin"
Cohesion: 0.10
Nodes (12): CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Return printf format plus an ABI-safe expression for an Ocean type., Dispatch method lowering by semantic type instead of one giant branch., Resolve user-class method result type through ClassRegistry., Resolve static/class method result type through ClassRegistry. (+4 more)

### Community 3 - ".add_error"
Cohesion: 0.06
Nodes (26): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует объявление переменной, Валидирует выражение (правая часть присваивания или инициализации), Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции (+18 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".get_type_from_ast"
Cohesion: 0.09
Nodes (16): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+8 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Raised when Phils ownership/borrow rules are violated during lowering., Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters. (+1 more)

### Community 7 - "AGENTS.md — Ocean repository instructions"
Cohesion: 0.04
Nodes (45): 10. Inheritance, 11. C imports and external symbols, 12. Tensor public API, 13. Tensor identity invariant, 14. ND Tensor, 15. Autograd lifetime invariant, 16. Do not delete autograd metadata on Tensor release, 17. ML stack currently working (+37 more)

### Community 8 - "Parser"
Cohesion: 0.08
Nodes (20): Parser, Remove standalone triple-quoted blocks while preserving line count., Определяет текущий scope на основе отступа, Reset all per-compilation parser state. A Parser instance can safely be reused…, Build the parser's private graph before typed lowering., Parse source into the canonical ``TypedModule`` API., Проверяет, является ли имя именем класса, Извлекает содержимое внутри скобок, учитывая вложенность (+12 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.08
Nodes (17): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит литерал кортежа, Return True only when the entire expression is one quoted literal. Expressions…, Parse an expression into the transitional Phils AST., Парсит выражение с учетом приоритетов операторов Python, Парсит унарные операторы, Парсит цепочки индексации типа a[0][1][2], Универсальный парсер аргументов функции. Возвращает (positional_args,… (+9 more)

### Community 10 - ".add_warning"
Cohesion: 0.06
Nodes (22): Проверяет дублирование переменных в local_variables, Валидирует таблицу символов scope'а, Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль (+14 more)

### Community 12 - "Validator"
Cohesion: 0.15
Nodes (13): Collect variable references from expression ASTs only., Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Validator, test_openmp_collapse_allows_sequential_loop_after_collapsed_nest(), test_openmp_collapse_requires_enough_nested_loops(), test_openmp_collapse_requires_perfect_nesting(), test_openmp_collapse_two_nested_loops_is_emitted_and_accepted() (+5 more)

### Community 13 - ".calculate_indent_level"
Cohesion: 0.11
Nodes (12): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in…, Парсит объявление класса (+4 more)

### Community 14 - "parse_cli_paths"
Cohesion: 0.15
Nodes (27): ArgumentParser, build_argument_parser(), parse_cli_paths(), Resolve paths for both package commands and the old single-file CLI., create_package(), find_manifest(), _flags(), load_package() (+19 more)

### Community 15 - "RuntimeError"
Cohesion: 0.26
Nodes (3): RuntimeError, ArrayCodegenMixin, Lower uniquely-owned one-dimensional ``array[T]`` values to C.

### Community 16 - "test_memory_safety.py"
Cohesion: 0.27
Nodes (12): compile_ocean(), test_borrow_cannot_escape_through_return(), test_borrow_cannot_escape_to_non_borrowing_parameter(), test_borrow_is_released_at_block_exit(), test_direct_c_call_requires_unsafe_block(), test_immutable_borrow_cannot_be_passed_to_mutable_parameter(), test_mutable_and_immutable_borrows_are_exclusive(), test_owned_array_is_moved_into_by_value_parameter() (+4 more)

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
Cohesion: 0.25
Nodes (4): Парсит вызов встроенной функции, Разбирает аргументы функции с учетом строк и вложенных вызовов, Универсальный парсер любого вызова функции с поддержкой опций, Определяет тип возвращаемого значения для встроенной функции

### Community 21 - "StatementsMixin"
Cohesion: 0.09
Nodes (14): Генерирует if statement, Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Release loop-local owners before transferring control., Render validated structured OpenMP metadata as one C pragma., Release current iteration owners before continuing., Return the validated collapse count for an OpenMP directive., Ownership-safe class field assignment. (+6 more)

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
Cohesion: 0.12
Nodes (11): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Parse ``var self.attr: Type [= value]`` with nested types., Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Очищает значение от лишних пробелов, но для сложных выражений возвращает AST, Parse ``name: Type = default`` with nested generic/borrow types., Выводит тип из AST выражения (+3 more)

### Community 27 - "CoreMixin"
Cohesion: 0.33
Nodes (4): CoreMixin, Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку

### Community 28 - "Handoff.md"
Cohesion: 0.03
Nodes (72): 10. Tensor runtime, 11. Tensor storage, 12. Tensor API, 13. ND Tensor milestone, 14. Известный bug: higher-rank gradient reduction, 15. Autograd design, 16. Критический lifetime invariant autograd, 17. Не удалять autograd metadata на Tensor release (+64 more)

### Community 29 - "TypedScope"
Cohesion: 0.20
Nodes (4): Typed view of one parser scope., Return the typed, read-only lowering view consumed by the C backend., Find a typed scope by its parser level., TypedScope

### Community 30 - "._validate_scopes"
Cohesion: 0.17
Nodes (6): Возвращает отчет о проверке, Return the canonical typed report without compatibility projections., Validate the canonical semantic module before C lowering., Строит карту соответствия узлов исходным строкам, Run the existing validation passes over a typed lowering view., Собирает информацию о всех символах в системе

### Community 31 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 32 - ".validate_openmp_loop"
Cohesion: 0.13
Nodes (7): Валидирует узел цикла, Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation., Collect variable references from any expression AST variant.

### Community 33 - "OrchestratorMixin"
Cohesion: 0.19
Nodes (7): OrchestratorMixin, Generate C from the canonical semantic IR., Lower typed scope views after semantic IR construction., Генерирует имя временной переменной, Генерирует код для узла графа, Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 36 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Collect C imports and all forward declarations from semantic metadata.

### Community 37 - "tensor_runtime.c"
Cohesion: 0.09
Nodes (66): cl_event, cl_int, cl_kernel, ocean_tensor_opencl_kernel_key, ocean_tensor_handle_t, ocean_tensor_alloc_uninitialized(), ocean_tensor_apply_binary(), ocean_tensor_binary_cpu() (+58 more)

### Community 39 - "CCodeGenerator"
Cohesion: 0.33
Nodes (9): CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, compile_and_run(), test_oop_constructor_method_mutation_and_composition(), test_oop_default_constructor_without_init(), test_oop_inherited_field_access_uses_embedded_base_layout(), test_oop_metadata_has_one_canonical_class_model(), test_oop_rejects_inheritance_cycles() (+1 more)

### Community 40 - "What changed"
Cohesion: 0.15
Nodes (12): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Structured diagnostics, 5. Deterministic scope cleanup, 6. Ownership-aware containers, 7. Safer class lowering, Important safety boundary (+4 more)

### Community 41 - "SymbolTable"
Cohesion: 0.14
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 42 - "Any"
Cohesion: 0.37
Nodes (4): Any, Lower the parser's private graph into typed scopes and nodes., Wrap expression payloads recursively while preserving AST keys., TypedIRBuilder

### Community 43 - "web_runtime.c"
Cohesion: 0.06
Nodes (90): buffer_t, connection_queue_t, connection_t, ocean_Request, ocean_Response, ocean_web_app_t, ocean_web_handler_t, ocean_web_middleware_t (+82 more)

### Community 44 - "Package"
Cohesion: 0.31
Nodes (4): Package, Path, Resolved package manifest and paths. Paths in the manifest are always…, Return the complete compiler flags for a named profile.

### Community 45 - "README.md"
Cohesion: 0.12
Nodes (14): 🔌 C / POSIX interop, 🧪 Current status, 📚 Documentation, ⚡ GPU, License, 📦 NumPy weights, Ocean 🌊, 🧬 One language, two worlds (+6 more)

### Community 46 - "autograd_runtime.c"
Cohesion: 0.15
Nodes (50): ocean_adamw_parameter_state, ocean_autograd_meta, ocean_autograd_node, ocean_autograd_topology, ocean_tensor_handle_t, ocean_adamw_free_parameter_state(), ocean_adamw_get_parameter_state(), ocean_adamw_shutdown() (+42 more)

### Community 47 - "ocean_tensor_fail"
Cohesion: 0.19
Nodes (42): ocean_tensor_backend_kind, ocean_tensor_backend_ops, ocean_autograd_accumulate(), ocean_autograd_adamw_step(), ocean_autograd_backward_node(), ocean_autograd_contiguous_strides_v04(), ocean_autograd_cross_entropy_forward_v04(), ocean_autograd_embedding_forward_v04() (+34 more)

### Community 48 - "debug.py"
Cohesion: 0.08
Nodes (23): Enum, Diagnostic, DiagnosticReport, DiagnosticSeverity, Any, Structured compiler diagnostics. The compiler keeps diagnostics as typed values…, Diagnostic importance understood by compiler frontends., Location of a diagnostic in an Ocean source file. (+15 more)

### Community 49 - "ocean_tensor_release"
Cohesion: 0.24
Nodes (17): FILE, ocean_tensor_alloc_zeros(), ocean_tensor_column(), ocean_tensor_contiguous(), ocean_tensor_is_contiguous(), ocean_tensor_normalize_dim_v02(), ocean_tensor_npy_write(), ocean_tensor_npy_write_u16() (+9 more)

### Community 51 - "ocean_tensor_load_npy"
Cohesion: 0.16
Nodes (16): ocean_tensor_dtype, ocean_tensor_alloc(), ocean_tensor_dtype_size(), ocean_tensor_elements_from_shape(), ocean_tensor_host_is_little_endian(), ocean_tensor_load_npy(), ocean_tensor_load_npy_typed(), ocean_tensor_npy_descr() (+8 more)

### Community 52 - ".parse_complex_expression"
Cohesion: 0.13
Nodes (8): Парсит выражение на текущем уровне приоритета операторов, Проверяет, что оператор в данной позиции является валидным оператором, Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок, Проверяет, является ли символ частью идентификатора, Находит позицию оператора вне скобок, строк и комментариев, Проверяет, содержит ли выражение какой-либо оператор

### Community 53 - "ocean_autograd_mse_loss"
Cohesion: 0.22
Nodes (12): ocean_adamw_optimizer_state, ocean_adamw_find_state(), ocean_autograd_adamw_begin_step(), ocean_autograd_adamw_create(), ocean_autograd_mse_loss(), ocean_autograd_parameter_uniform(), ocean_tensor_set_2d(), ocean_tensor_zeros() (+4 more)

### Community 54 - "53. Ближайший roadmap"
Cohesion: 0.22
Nodes (9): 53. Ближайший roadmap, P0 — подтвердить GPU training, P1 — GPU-native Transformer path, P2 — autoregressive inference, P3 — KV cache, P4 — positional encoding, P5 — CUDA backend, P6 — optimizer performance (+1 more)

### Community 55 - "file_runtime.c"
Cohesion: 0.44
Nodes (13): ocean_file_handle_t, ocean_file_close(), ocean_file_eof(), ocean_file_fail(), ocean_file_flush(), ocean_file_open(), ocean_file_read(), ocean_file_read_buffer() (+5 more)

### Community 57 - "net_runtime.c"
Cohesion: 0.13
Nodes (44): ocean_buffer, ocean_http_response_t, ocean_socket_handle_t, parsed_url, buf_append(), buf_cstr(), buf_init(), connect_fd() (+36 more)

### Community 60 - "46. Frontend/compiler quirks, которые уже встречались"
Cohesion: 0.29
Nodes (7): 46.1 Multiline class method signatures, 46.2 Method calls in constructors, 46.3 `len(self.parameters)`, 46.4 `Tensor.item()` + reassignment, 46.5 Chained attribute methods в `print`, 46.6 1D Tensor assignment, 46. Frontend/compiler quirks, которые уже встречались

### Community 61 - "6. Memory model"
Cohesion: 0.50
Nodes (4): 6.1 ARC, 6.2 Container ownership, 6.3 `del`, 6. Memory model

### Community 62 - "TypeParser"
Cohesion: 0.21
Nodes (7): infer_literal_shape(), Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, TypeParser, OwnershipEffect, Typed intermediate representation for the Ocean compiler. The parser's…, Explicit ownership transition attached to a typed graph node.

### Community 63 - "class_model.py"
Cohesion: 0.29
Nodes (9): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+1 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (5): ClassRegistry, FieldModel, Canonical class metadata and lookup service for the C backend., A field declared directly by one class., Reset all per-compilation mutable state.

### Community 65 - ".resolved_methods"
Cohesion: 0.50
Nodes (3): Resolve methods without rebuilding parser-shaped dictionaries., A method together with the class that provides its implementation., ResolvedMethod

### Community 66 - "ClassModel"
Cohesion: 0.28
Nodes (4): ClassModel, Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "base.py"
Cohesion: 0.15
Nodes (6): test_del(), test_global_var_declaration(), test_if_else_1(), test_dict(), test_methods(), test_variables()

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 70 - "json_runtime.c"
Cohesion: 0.11
Nodes (64): ocean_json_handle_t, ocean_json_kind_t, ocean_json_alloc(), ocean_json_append_utf8(), ocean_json_array_append(), ocean_json_array_get(), ocean_json_array_reserve(), ocean_json_array_set() (+56 more)

### Community 71 - "._parse_line_impl"
Cohesion: 0.05
Nodes (26): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит присваивание значения указателя переменной: x = *p (+18 more)

### Community 72 - "os_runtime.c"
Cohesion: 0.13
Nodes (27): ocean_os_dir_list_t, ocean_os_chdir(), ocean_os_dir_list_append(), ocean_os_dir_list_get_copy(), ocean_os_dir_list_release(), ocean_os_dir_list_size(), ocean_os_exists(), ocean_os_fail() (+19 more)

### Community 74 - "compile_c"
Cohesion: 0.09
Nodes (36): compile_c(), compile_pipeline(), Compile generated C and return the exact command that was executed., Parse, validate, and generate C without an intermediate serialization artifact., test_adamw_v01_ocean(), test_attention_v01_ocean_frontend(), test_causal_attention_v01_ocean(), test_standard_file_and_binary_file_io() (+28 more)

### Community 75 - "OpenCL tensor backend"
Cohesion: 0.15
Nodes (9): `File` and `BinaryFile`, Current priorities, Imports, Ocean standard library, Backend selection, Kernel contract, OpenCL tensor backend, Runtime objects (+1 more)

### Community 76 - "TypedNode"
Cohesion: 0.20
Nodes (3): Structured OpenMP metadata attached to a loop, if present., Semantic metadata and read-only mapping view for one graph node., TypedNode

### Community 77 - "Ocean automatic ownership model v1"
Cohesion: 0.22
Nodes (8): Borrow, Categories, Containers, FFI, Move, Ocean automatic ownership model v1, Reference alias, Return ABI

### Community 78 - "`Tensor[T]`"
Cohesion: 0.25
Nodes (8): Elementwise operations and layout transforms, Internal representation, `matmul`, NumPy `.npy` files, Public API, Purpose, Semantics of `.to(device)`, `Tensor[T]`

### Community 79 - "parser.py"
Cohesion: 0.13
Nodes (10): CImportProcessor, ImportProcessor, Path, Просто регистрирует C импорт без парсинга, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield real project std directories before incidental nested ``std`` dirs.…, test_import() (+2 more)

### Community 80 - "TypedModule"
Cohesion: 0.25
Nodes (5): _build_typed_module(), Typed compilation unit exchanged by compiler passes., Iterate semantic nodes in source/scope order., Build the typed module from the parser's internal graph., TypedModule

### Community 81 - "_compile"
Cohesion: 0.83
Nodes (3): _compile(), Path, test_v03_math_forward_and_gradients()

### Community 82 - "29. Known frontend/compiler quirks"
Cohesion: 0.29
Nodes (7): 1D Tensor assignment, 29. Known frontend/compiler quirks, bool vs int, Chained attribute calls in `print`, Constructor lowering, Multiline class method signatures, `Tensor.item()` reassignment

### Community 85 - "time_runtime.c"
Cohesion: 0.19
Nodes (20): clockid_t, ocean_time_empty_string(), ocean_time_fail(), ocean_time_format(), ocean_time_format_local(), ocean_time_format_utc(), ocean_time_get_clock(), ocean_time_monotonic() (+12 more)

### Community 86 - "main.py"
Cohesion: 0.24
Nodes (16): _command(), _compiler_settings(), default_output_paths(), _diagnostic_location(), _ensure_parent(), _explicit_source(), _load_package_for_args(), main() (+8 more)

### Community 87 - "test_main_cli.py"
Cohesion: 0.22
Nodes (10): cli(), Discover standard-library C runtimes from generated std/ headers., Entry point used by the installed ``ocean`` console script., _standard_runtime_dependencies(), test_compile_c_adds_libm_for_generated_math_import(), test_compile_c_places_explicit_libraries_after_sources(), test_compile_c_preserves_explicit_optimization_with_openmp(), test_empty_cli_prints_help_without_running_pipeline() (+2 more)

### Community 88 - "logging_runtime.c"
Cohesion: 0.18
Nodes (19): FILE, ocean_logging_close_owned_stream_locked(), ocean_logging_color_reset(), ocean_logging_current_stream(), ocean_logging_enabled(), ocean_logging_flush(), ocean_logging_get_colors(), ocean_logging_get_level() (+11 more)

### Community 89 - "thread_backend.c"
Cohesion: 0.36
Nodes (8): ocean_thread_fn_t, ocean_thread_handle_t, ocean_thread_create(), ocean_thread_detach(), ocean_thread_fail(), ocean_thread_is_joinable(), ocean_thread_join(), ocean_thread_release()

### Community 90 - "compiler.py"
Cohesion: 0.20
Nodes (6): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, generate(), test_array_lowering_and_index_mutation(), test_matmul_1(), test_matmul_2(), test_matmul_3()

### Community 91 - "ExpressionsMixin"
Cohesion: 0.25
Nodes (6): ExpressionsMixin, Encode one decoded Ocean string value as a valid C literal., Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST для конструктора с подстановкой параметров., Генерирует выражение из AST с подстановкой параметров конструктора

### Community 92 - "Benchmarks"
Cohesion: 0.50
Nodes (3): Backend, Benchmarks, Matmul

### Community 93 - "net/README.md"
Cohesion: 0.12
Nodes (15): 12. Response headers, 13. Полный пример REST API, 15. Автоматическое HTTP framing, 16. Автоматические ошибки router, 18. Архитектура, 19. Ownership, 20. `Json` ownership, 21. Текущие ограничения (+7 more)

### Community 94 - "`Socket` API"
Cohesion: 0.12
Nodes (16): 1. TCP sockets, Accept, Bind, Listen, Receive, Send, `Socket` API, TCP client (+8 more)

### Community 96 - "test_while_loop.py"
Cohesion: 0.50
Nodes (3): test_while_loop_1(), test_while_loop_2(), test_while_loop_3()

### Community 97 - "🌐 Backend development in Ocean"
Cohesion: 0.40
Nodes (5): A small API server, 🌐 Backend development in Ocean, Middleware, Native worker pool, Routers

### Community 100 - "2. HTTP client"
Cohesion: 0.15
Nodes (13): 2. HTTP client, Body, DELETE, GET, GET с headers, Headers, HTTP status, `HttpResponse` (+5 more)

### Community 101 - "DictCodegenMixin"
Cohesion: 0.50
Nodes (3): DictCodegenMixin, Generate an ARC-owned chained hash table., Генерирует объявление словаря

### Community 102 - "8. Request"
Cohesion: 0.22
Nodes (9): 8. Request, Body, Client address, Header, Method, Path, Path parameter, Query parameter (+1 more)

### Community 103 - "14. Проверка REST API через `curl`"
Cohesion: 0.25
Nodes (8): 14. Проверка REST API через `curl`, DELETE, GET, PATCH, Path parameter, POST JSON, PUT, Query parameter

### Community 104 - "29. GPT primitives"
Cohesion: 0.67
Nodes (3): 29. GPT primitives, CrossEntropyLoss, Embedding

### Community 105 - "10. Response"
Cohesion: 0.29
Nodes (7): 10. Response, Empty response, HTML response, HTML с status, Redirect, Text response, Text response с status

### Community 106 - "5. Routing"
Cohesion: 0.33
Nodes (6): 5. Routing, DELETE, GET, PATCH, POST, PUT

### Community 107 - "Импорт"
Cohesion: 0.40
Nodes (5): HTTP client, HTTP/Web server, `std/net` — networking и HTTP/Web для Ocean, TCP sockets, Импорт

### Community 108 - "32. Optimizers"
Cohesion: 0.67
Nodes (3): 32. Optimizers, AdamW v0.1, SGD

### Community 109 - "🗺️ Roadmap"
Cohesion: 0.50
Nodes (4): Backend, Language / compiler, ML / GPU, 🗺️ Roadmap

### Community 110 - "11. JSON response"
Cohesion: 0.50
Nodes (4): 11. JSON response, JSON response с status, `Json` из `std/json`, Готовая JSON-строка

### Community 111 - "17. Настройка App"
Cohesion: 0.50
Nodes (4): 17. Настройка App, Maximum request body, Serve, Server header

### Community 112 - "🧠 ML in Ocean"
Cohesion: 0.50
Nodes (4): Build a Transformer block, 🧠 ML in Ocean, TinyGPT, Train a neural network

### Community 113 - "test_tensor_npy.py"
Cohesion: 0.83
Nodes (3): Path, test_tensor_npy_reads_external_file_and_writes_compatible_file(), write_npy()

## Knowledge Gaps
- **277 isolated node(s):** `ocean-lang`, `1. Project identity`, `2. Compiler pipeline`, `3. Repository layout`, `4. Coding style` (+272 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `CallsMixin`, `TypesMixin`, `OwnershipMixin`, `Parser`, `generator.py`, `Validator`, `RuntimeError`, `test_memory_safety.py`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `OrchestratorMixin`, `TensorCodegenMixin`, `ImportsMixin`, `compile_c`, `NamingMixin`, `compiler.py`, `ExpressionsMixin`, `DictCodegenMixin`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `base.py`, `compiler.py`, `._parse_line_impl`, `CCodeGenerator`, `SymbolTable`, `compile_c`, `.parse_expression_to_ast`, `Validator`, `.calculate_indent_level`, `parser.py`, `debug.py`, `test_memory_safety.py`, `.parse_complex_expression`, `.parse_function_call`, `main.py`, `.parse_type_annotation`, `TypeParser`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Why does `Validator` connect `Validator` to `.validate_openmp_loop`, `.add_error`, `.get_type_from_ast`, `Parser`, `compile_c`, `.add_warning`, `debug.py`, `TypedModule`, `test_memory_safety.py`, `main.py`, `.get_symbol_info`, `._validate_scopes`, `.check_undefined_methods`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Validator` (e.g. with `Diagnostic` and `DiagnosticReport`) actually correct?**
  _`Validator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `ocean_tensor_fail()` (e.g. with `ocean_adamw_find_state()` and `ocean_adamw_get_parameter_state()`) actually correct?**
  _`ocean_tensor_fail()` has 34 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `1. Project identity`, `2. Compiler pipeline` to the rest of the system?**
  _277 weakly-connected nodes found - possible documentation gaps or missing edges._