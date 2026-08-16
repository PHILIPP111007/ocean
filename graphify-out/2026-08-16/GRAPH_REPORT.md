# Graph Report - phils_language  (2026-08-16)

## Corpus Check
- 110 files · ~114,920 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1799 nodes · 3856 edges · 115 communities (99 shown, 16 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 101 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `13c7b669`
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
- .validate_static_method_call
- Parser
- .parse_expression_to_ast
- .add_warning
- generator.py
- Validator
- ._parse_line_impl
- Package
- RuntimeError
- .find_symbol_recursive
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
- TypedModule
- ._validate_scopes
- .check_undefined_methods
- .validate_openmp_loop
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- CCodeGenerator
- tensor_runtime.c
- .parse_complex_expression
- What changed
- SymbolTable
- Any
- web_runtime.c
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- debug.py
- 21. Array — принятое устройство
- 28. Исторический статус array/tensor до завершения backend-перехода
- 30. Устаревший план до завершения Tensor backend
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- file_runtime.c
- TypedExpression
- net_runtime.c
- ocean-lang
- ._generate_complex_attribute_access
- benchmark_main.py
- parser.py
- class_model.py
- ClassRegistry
- TypeSpec
- ClassModel
- base.py
- .generate_all_methods
- IRType
- json_runtime.c
- .extract_dependencies_from_ast
- os_runtime.c
- compile_c
- README.md
- TypedNode
- Ocean automatic ownership model v1
- `Tensor[T]`
- ImportProcessor
- OpenCL tensor backend
- .resolved_methods
- Arrays and public tensors
- Ocean standard library
- time_runtime.c
- main.py
- test_main_cli.py
- logging_runtime.c
- thread_backend.c
- compiler.py
- ExpressionsMixin
- ImportsMixin
- net/README.md
- `Socket` API
- MLP
- test_while_loop.py
- test_input.py
- json/README.md
- 2. HTTP client
- .generate_builtin_function_call
- 8. Request
- 14. Проверка REST API через `curl`
- CImportProcessor
- 10. Response
- 5. Routing
- Импорт
- test_oop.py
- ._parse_graph
- 11. JSON response
- 17. Настройка App
- test_matmul.py
- test_tensor_npy.py
- test_dict.py

## God Nodes (most connected - your core abstractions)
1. `Validator` - 149 edges
2. `Parser` - 138 edges
3. `ocean_tensor_fail()` - 72 edges
4. `run()` - 60 edges
5. `compile_c()` - 44 edges
6. `compile_pipeline()` - 42 edges
7. `CCodeGenerator` - 38 edges
8. `OwnershipMixin` - 29 edges
9. `ocean_tensor_to()` - 28 edges
10. `ocean_json_fail()` - 27 edges

## Surprising Connections (you probably didn't know these)
- `run_benchmark()` --calls--> `CCodeGenerator`  [INFERRED]
  benchmarks/benchmark_main.py → src/codegen/generator.py
- `test_compile_c_adds_libm_for_generated_math_import()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `test_compile_c_places_explicit_libraries_after_sources()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `test_compile_c_preserves_explicit_optimization_with_openmp()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_main_cli.py → main.py
- `test_compile_c_adds_openmp_flag_for_generated_pragma()` --calls--> `compile_c()`  [EXTRACTED]
  tests/test_openmp.py → main.py

## Import Cycles
- None detected.

## Communities (115 total, 16 thin omitted)

### Community 0 - "run"
Cohesion: 0.14
Nodes (20): run(), test_c_code_math(), test_c_code_pthread(), test_for_loop_1(), test_for_loop_2(), test_for_loop_3(), test_for_loop_4(), test_functions_1() (+12 more)

### Community 1 - "OopMixin"
Cohesion: 0.14
Nodes (9): OopMixin, Build the canonical OOP metadata directly from parser output., Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models. (+1 more)

### Community 2 - "CallsMixin"
Cohesion: 0.16
Nodes (7): CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует прямой вызов C-функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции

### Community 3 - ".add_error"
Cohesion: 0.07
Nodes (22): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует объявление переменной, Валидирует выражение (правая часть присваивания или инициализации), Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции (+14 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".get_type_from_ast"
Cohesion: 0.08
Nodes (18): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+10 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Raised when Phils ownership/borrow rules are violated during lowering., Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters. (+1 more)

### Community 8 - "Parser"
Cohesion: 0.06
Nodes (27): Parser, Remove standalone triple-quoted blocks while preserving line count., Парсит присваивание значения указателя переменной: x = *p, Парсит оператор break, Парсит оператор continue, Определяет текущий scope на основе отступа, Проверяет, является ли имя именем класса, Извлекает содержимое внутри скобок, учитывая вложенность (+19 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.07
Nodes (18): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит литерал кортежа, Парсит оператор return, Return True only when the entire expression is one quoted literal. Expressions…, Parse an expression into the transitional Phils AST., Парсит выражение с учетом приоритетов операторов Python, Парсит унарные операторы, Парсит цепочки индексации типа a[0][1][2] (+10 more)

### Community 10 - ".add_warning"
Cohesion: 0.06
Nodes (22): Проверяет дублирование переменных в local_variables, Валидирует таблицу символов scope'а, Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль (+14 more)

### Community 12 - "Validator"
Cohesion: 0.15
Nodes (14): Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Collect variable references from expression ASTs only., Validator, test_compile_c_adds_openmp_flag_for_generated_pragma(), test_openmp_collapse_allows_sequential_loop_after_collapsed_nest(), test_openmp_collapse_requires_enough_nested_loops(), test_openmp_collapse_requires_perfect_nesting() (+6 more)

### Community 13 - "._parse_line_impl"
Cohesion: 0.10
Nodes (16): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Парсит оператор del (полное удаление), Возвращает область видимости для заданного уровня отступа, Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Parse the supported OpenMP loop directive into structured metadata., Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла (+8 more)

### Community 14 - "Package"
Cohesion: 0.15
Nodes (22): create_package(), find_manifest(), _flags(), load_package(), Package, PackageError, profile_flags(), Path (+14 more)

### Community 15 - "RuntimeError"
Cohesion: 0.26
Nodes (3): RuntimeError, ArrayCodegenMixin, Lower uniquely-owned one-dimensional ``array[T]`` values to C.

### Community 16 - ".find_symbol_recursive"
Cohesion: 0.29
Nodes (3): Парсит составные операции присваивания, Строит операции из AST выражения, Рекурсивно ищет символ в текущем и родительских scope'ах

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
Cohesion: 0.11
Nodes (8): Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет корректное использование указателей, Проверяет выход за границы массивов/списков, Проверяет операции со строками, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST, Находит родительский узел (если есть)

### Community 24 - "HelpersMixin"
Cohesion: 0.24
Nodes (6): HelpersMixin, Генерирует вспомогательные функции для сортировки, Collect standard-runtime features before any C is emitted. The scan is…, Генерирует вспомогательные функции для работы со строками, Генерирует секцию с вспомогательными функциями и структурами в правильном…, Генерирует вспомогательные функции для конвертации в int

### Community 25 - "TupleCodegenMixin"
Cohesion: 0.36
Nodes (5): Генерирует имя структуры для tuple, Генерирует код для повторного объявления кортежа, Create a homogeneous immutable tuple with owned element references., Generate an ARC-owned homogeneous tuple[T]., TupleCodegenMixin

### Community 26 - ".parse_type_annotation"
Cohesion: 0.12
Nodes (11): Parse ``name: Type`` or ``name: Type = default``., Parse a typed variable declaration. Supported memory-oriented forms: *…, Parse ``var self.attr: Type [= value]`` with nested types., Parse ``self.attr [: Type] = value`` in a constructor., Извлекает информацию о контейнере из AST, Parse ``name: Type = default`` with nested generic/borrow types., Выводит тип из AST выражения, Parse a comma-separated parameter list at top level. (+3 more)

### Community 27 - "CoreMixin"
Cohesion: 0.33
Nodes (4): CoreMixin, Возвращает отступ для текущего уровня, Добавляет строку с правильным отступом, Добавляет пустую строку

### Community 28 - "Handoff.md"
Cohesion: 0.06
Nodes (32): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+24 more)

### Community 29 - "TypedModule"
Cohesion: 0.12
Nodes (9): _build_typed_module(), Typed view of one parser scope., Typed compilation unit exchanged by compiler passes., Return the typed, read-only lowering view consumed by the C backend., Find a typed scope by its parser level., Iterate semantic nodes in source/scope order., Build the typed module from the parser's internal graph., TypedModule (+1 more)

### Community 30 - "._validate_scopes"
Cohesion: 0.17
Nodes (6): Возвращает отчет о проверке, Return the canonical typed report without compatibility projections., Строит карту соответствия узлов исходным строкам, Validate the canonical semantic module before C lowering., Run the existing validation passes over a typed lowering view., Собирает информацию о всех символах в системе

### Community 31 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 32 - ".validate_openmp_loop"
Cohesion: 0.15
Nodes (6): Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation., Collect variable references from any expression AST variant.

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
Cohesion: 0.07
Nodes (110): cl_event, cl_int, cl_kernel, ocean_tensor_backend_kind, ocean_tensor_backend_ops, ocean_tensor_dtype, ocean_tensor_handle_t, ocean_tensor_opencl_kernel_key (+102 more)

### Community 39 - ".parse_complex_expression"
Cohesion: 0.11
Nodes (9): Парсит выражение на текущем уровне приоритета операторов, Проверяет, что оператор в данной позиции является валидным оператором, Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок, Проверяет, является ли символ частью идентификатора, Находит позицию оператора вне скобок, строк и комментариев, Проверяет, содержит ли выражение какой-либо оператор (+1 more)

### Community 40 - "What changed"
Cohesion: 0.15
Nodes (12): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Structured diagnostics, 5. Deterministic scope cleanup, 6. Ownership-aware containers, 7. Safer class lowering, Important safety boundary (+4 more)

### Community 41 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 42 - "Any"
Cohesion: 0.37
Nodes (4): Any, Lower the parser's private graph into typed scopes and nodes., Wrap expression payloads recursively while preserving AST keys., TypedIRBuilder

### Community 43 - "web_runtime.c"
Cohesion: 0.09
Nodes (59): buf_t, ocean_web_app_t, ocean_web_handler_t, ocean_web_request_t, ocean_web_response_t, route_t, socklen_t, ba() (+51 more)

### Community 44 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Safety & Configuration Notes, Testing Guidelines

### Community 45 - "Ocean 🌊"
Cohesion: 0.14
Nodes (14): A small example, Compiler pipeline, Contributing, Honest project status, Imports and the standard library, Install as a Python package, Memory model, Object-oriented ML example (+6 more)

### Community 46 - "4. Memory model"
Cohesion: 0.29
Nodes (7): 4. Memory model, BORROWED, Immutable borrow, Mutable borrow, OWNED, SHARED, VALUE

### Community 47 - "20. Тесты"
Cohesion: 0.33
Nodes (6): 20. Тесты, Level 1 — AST / parser, Level 2 — C generation, Level 3 — compile, Level 4 — memory safety, Обязательные memory tests

### Community 48 - "debug.py"
Cohesion: 0.06
Nodes (35): Enum, Diagnostic, DiagnosticReport, DiagnosticSeverity, Any, Structured compiler diagnostics. The compiler keeps diagnostics as typed values…, Diagnostic importance understood by compiler frontends., Location of a diagnostic in an Ocean source file. (+27 more)

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Исторический статус array/tensor до завершения backend-перехода"
Cohesion: 0.67
Nodes (3): 28. Исторический статус array/tensor до завершения backend-перехода, Backend, Parser

### Community 51 - "30. Устаревший план до завершения Tensor backend"
Cohesion: 0.67
Nodes (3): 30. Устаревший план до завершения Tensor backend, Array, Tensor

### Community 55 - "file_runtime.c"
Cohesion: 0.44
Nodes (13): ocean_file_handle_t, ocean_file_close(), ocean_file_eof(), ocean_file_fail(), ocean_file_flush(), ocean_file_open(), ocean_file_read(), ocean_file_read_buffer() (+5 more)

### Community 57 - "net_runtime.c"
Cohesion: 0.13
Nodes (44): ocean_buffer, ocean_http_response_t, ocean_socket_handle_t, parsed_url, buf_append(), buf_cstr(), buf_init(), connect_fd() (+36 more)

### Community 60 - "._generate_complex_attribute_access"
Cohesion: 0.20
Nodes (3): Генерирует доступ к элементу сложного атрибута (self.data[index]), Generate a scalar compound assignment, including OpenMP reductions., Early deterministic release. Raw pointers are never implicitly freed.

### Community 61 - "benchmark_main.py"
Cohesion: 0.31
Nodes (8): main(), measure(), Path, Benchmark the generated C program for examples/matmul.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, ValueError

### Community 62 - "parser.py"
Cohesion: 0.21
Nodes (7): infer_literal_shape(), Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, TypeParser, OwnershipEffect, Typed intermediate representation for the Ocean compiler. The parser's…, Explicit ownership transition attached to a typed graph node.

### Community 63 - "class_model.py"
Cohesion: 0.29
Nodes (9): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+1 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (5): ClassRegistry, FieldModel, Canonical class metadata and lookup service for the C backend., A field declared directly by one class., Reset all per-compilation mutable state.

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

### Community 71 - ".extract_dependencies_from_ast"
Cohesion: 0.08
Nodes (14): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит присваивание результата вызова функции: var x: type = func(args) (+6 more)

### Community 72 - "os_runtime.c"
Cohesion: 0.13
Nodes (27): ocean_os_dir_list_t, ocean_os_chdir(), ocean_os_dir_list_append(), ocean_os_dir_list_get_copy(), ocean_os_dir_list_release(), ocean_os_dir_list_size(), ocean_os_exists(), ocean_os_fail() (+19 more)

### Community 74 - "compile_c"
Cohesion: 0.15
Nodes (24): compile_c(), compile_pipeline(), Compile generated C and return the exact command that was executed., Parse, validate, and generate C without an intermediate serialization artifact., test_standard_file_and_binary_file_io(), test_std_logging(), test_std_math_compiles(), _serve_once() (+16 more)

### Community 76 - "TypedNode"
Cohesion: 0.20
Nodes (3): Structured OpenMP metadata attached to a loop, if present., Semantic metadata and read-only mapping view for one graph node., TypedNode

### Community 77 - "Ocean automatic ownership model v1"
Cohesion: 0.25
Nodes (8): Borrow, Categories, Containers, FFI, Move, Ocean automatic ownership model v1, Reference alias, Return ABI

### Community 78 - "`Tensor[T]`"
Cohesion: 0.25
Nodes (8): Elementwise operations and layout transforms, Internal representation, `matmul`, NumPy `.npy` files, Public API, Purpose, Semantics of `.to(device)`, `Tensor[T]`

### Community 79 - "ImportProcessor"
Cohesion: 0.18
Nodes (8): ImportProcessor, Path, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield real project std directories before incidental nested ``std`` dirs.…, test_import(), test_relative_and_standard_imports(), test_standard_import_does_not_resolve_to_example_shadow()

### Community 80 - "OpenCL tensor backend"
Cohesion: 0.40
Nodes (5): Backend selection, Kernel contract, OpenCL tensor backend, Runtime objects, Safety boundary

### Community 81 - ".resolved_methods"
Cohesion: 0.50
Nodes (3): Resolve methods without rebuilding parser-shaped dictionaries., A method together with the class that provides its implementation., ResolvedMethod

### Community 82 - "Arrays and public tensors"
Cohesion: 0.67
Nodes (3): Arrays and public tensors, Device-aware `Tensor[T]`, File IO and NumPy weights

### Community 83 - "Ocean standard library"
Cohesion: 0.67
Nodes (3): Current priorities, Imports, Ocean standard library

### Community 85 - "time_runtime.c"
Cohesion: 0.19
Nodes (20): clockid_t, ocean_time_empty_string(), ocean_time_fail(), ocean_time_format(), ocean_time_format_local(), ocean_time_format_utc(), ocean_time_get_clock(), ocean_time_monotonic() (+12 more)

### Community 86 - "main.py"
Cohesion: 0.23
Nodes (18): _command(), _compiler_settings(), default_output_paths(), _diagnostic_location(), _ensure_parent(), _explicit_source(), _load_package_for_args(), main() (+10 more)

### Community 87 - "test_main_cli.py"
Cohesion: 0.17
Nodes (16): ArgumentParser, build_argument_parser(), cli(), Discover standard-library C runtimes from generated std/ headers., Entry point used by the installed ``ocean`` console script., _standard_runtime_dependencies(), test_cli_accepts_custom_paths_flags_and_run_arguments(), test_cli_uses_package_default_paths() (+8 more)

### Community 88 - "logging_runtime.c"
Cohesion: 0.18
Nodes (19): FILE, ocean_logging_close_owned_stream_locked(), ocean_logging_color_reset(), ocean_logging_current_stream(), ocean_logging_enabled(), ocean_logging_flush(), ocean_logging_get_colors(), ocean_logging_get_level() (+11 more)

### Community 89 - "thread_backend.c"
Cohesion: 0.36
Nodes (8): ocean_thread_fn_t, ocean_thread_handle_t, ocean_thread_create(), ocean_thread_detach(), ocean_thread_fail(), ocean_thread_is_joinable(), ocean_thread_join(), ocean_thread_release()

### Community 90 - "compiler.py"
Cohesion: 0.33
Nodes (3): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, generate(), test_array_lowering_and_index_mutation()

### Community 91 - "ExpressionsMixin"
Cohesion: 0.25
Nodes (6): ExpressionsMixin, Encode one decoded Ocean string value as a valid C literal., Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST для конструктора с подстановкой параметров, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 92 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Collect C imports and all forward declarations from semantic metadata.

### Community 93 - "net/README.md"
Cohesion: 0.12
Nodes (15): 12. Response headers, 13. Полный пример REST API, 15. Автоматическое HTTP framing, 16. Автоматические ошибки router, 18. Архитектура, 19. Ownership, 20. `Json` ownership, 21. Текущие ограничения (+7 more)

### Community 94 - "`Socket` API"
Cohesion: 0.12
Nodes (16): 1. TCP sockets, Accept, Bind, Listen, Receive, Send, `Socket` API, TCP client (+8 more)

### Community 96 - "test_while_loop.py"
Cohesion: 0.50
Nodes (3): test_while_loop_1(), test_while_loop_2(), test_while_loop_3()

### Community 100 - "2. HTTP client"
Cohesion: 0.15
Nodes (13): 2. HTTP client, Body, DELETE, GET, GET с headers, Headers, HTTP status, `HttpResponse` (+5 more)

### Community 101 - ".generate_builtin_function_call"
Cohesion: 0.20
Nodes (5): Return printf format plus an ABI-safe expression for an Ocean type., Resolve user-class method result type through ClassRegistry., Resolve static/class method result type through ClassRegistry., Генерирует вызов встроенной функции, Генерирует присваивание результата встроенной функции

### Community 102 - "8. Request"
Cohesion: 0.22
Nodes (9): 8. Request, Body, Client address, Header, Method, Path, Path parameter, Query parameter (+1 more)

### Community 103 - "14. Проверка REST API через `curl`"
Cohesion: 0.25
Nodes (8): 14. Проверка REST API через `curl`, DELETE, GET, PATCH, Path parameter, POST JSON, PUT, Query parameter

### Community 104 - "CImportProcessor"
Cohesion: 0.29
Nodes (3): CImportProcessor, Просто регистрирует C импорт без парсинга, Reset all per-compilation parser state. A Parser instance can safely be reused…

### Community 105 - "10. Response"
Cohesion: 0.29
Nodes (7): 10. Response, Empty response, HTML response, HTML с status, Redirect, Text response, Text response с status

### Community 106 - "5. Routing"
Cohesion: 0.33
Nodes (6): 5. Routing, DELETE, GET, PATCH, POST, PUT

### Community 107 - "Импорт"
Cohesion: 0.40
Nodes (5): HTTP client, HTTP/Web server, `std/net` — networking и HTTP/Web для Ocean, TCP sockets, Импорт

### Community 108 - "test_oop.py"
Cohesion: 0.40
Nodes (4): disabled_test_oop_2(), test_oop_1(), test_oop_3(), test_oop_4()

### Community 110 - "11. JSON response"
Cohesion: 0.50
Nodes (4): 11. JSON response, JSON response с status, `Json` из `std/json`, Готовая JSON-строка

### Community 111 - "17. Настройка App"
Cohesion: 0.50
Nodes (4): 17. Настройка App, Maximum request body, Serve, Server header

### Community 112 - "test_matmul.py"
Cohesion: 0.50
Nodes (3): test_matmul_1(), test_matmul_2(), test_matmul_3()

### Community 113 - "test_tensor_npy.py"
Cohesion: 0.83
Nodes (3): Path, test_tensor_npy_reads_external_file_and_writes_compatible_file(), write_npy()

## Knowledge Gaps
- **186 isolated node(s):** `ocean-lang`, `ocean_web_app`, `ocean_web_request`, `ocean_web_response`, `ocean_Request` (+181 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `CallsMixin`, `TypesMixin`, `OwnershipMixin`, `Parser`, `generator.py`, `Validator`, `RuntimeError`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `OrchestratorMixin`, `TensorCodegenMixin`, `debug.py`, `benchmark_main.py`, `compile_c`, `compiler.py`, `ExpressionsMixin`, `ImportsMixin`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Why does `Validator` connect `Validator` to `.validate_openmp_loop`, `.add_error`, `.get_type_from_ast`, `.validate_static_method_call`, `Parser`, `compile_c`, `.add_warning`, `debug.py`, `TypedModule`, `main.py`, `.get_symbol_info`, `benchmark_main.py`, `._validate_scopes`, `.check_undefined_methods`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `.parse_expression_to_ast`, `Validator`, `._parse_line_impl`, `.find_symbol_recursive`, `.parse_function_call`, `.parse_type_annotation`, `.parse_complex_expression`, `SymbolTable`, `debug.py`, `benchmark_main.py`, `parser.py`, `base.py`, `.extract_dependencies_from_ast`, `compile_c`, `main.py`, `compiler.py`, `CImportProcessor`, `._parse_graph`, `test_matmul.py`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Validator` (e.g. with `Diagnostic` and `DiagnosticReport`) actually correct?**
  _`Validator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `ocean_web_app`, `ocean_web_request` to the rest of the system?**
  _186 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run` be split into smaller, more focused modules?**
  _Cohesion score 0.14333333333333334 - nodes in this community are weakly interconnected._