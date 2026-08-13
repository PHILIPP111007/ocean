# Graph Report - phils_language  (2026-08-13)

## Corpus Check
- 87 files · ~101,642 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1468 nodes · 3159 edges · 89 communities (78 shown, 11 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 141 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9a1f1e26`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- CallsMixin
- .validate_graph
- TypesMixin
- .add_error
- OwnershipMixin
- .validate_static_method_call
- Parser
- .parse_expression_to_ast
- .add_warning
- generator.py
- Validator
- .calculate_indent_level
- Package
- ArrayCodegenMixin
- ._parse_line_impl
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
- test_memory_safety.py
- Repository Guidelines
- Ocean 🌊
- 4. Memory model
- 20. Тесты
- debug.py
- 21. Array — принятое устройство
- 28. Статус array/tensor на момент handoff
- 30. Следующий рекомендуемый этап
- 12. Новый parser v0.2
- 15. Classes
- Handoff — Phils Language / Ocean backend
- file_runtime.c
- TypedExpression
- load_npy.generated.c
- ocean-lang
- RuntimeError
- TransformerLanguageModel
- TypeParser
- class_model.py
- ClassRegistry
- compile_c
- ClassModel
- compiler.py
- .generate_all_methods
- IRType
- ExpressionsMixin
- split_top_level
- parse_cli_paths
- ImportsMixin
- README.md
- TypedNode
- Ocean automatic ownership model v1
- `Tensor[T]`
- parser.py
- OpenCL tensor backend
- .resolved_methods
- Arrays and public tensors
- Ocean standard library
- main.py
- test_openmp.py
- .validate_return
- benchmark_main.py

## God Nodes (most connected - your core abstractions)
1. `Validator` - 149 edges
2. `Parser` - 137 edges
3. `ocean_tensor_fail()` - 71 edges
4. `run()` - 60 edges
5. `CCodeGenerator` - 38 edges
6. `compile_c()` - 30 edges
7. `OwnershipMixin` - 29 edges
8. `compile_pipeline()` - 28 edges
9. `ocean_tensor_to()` - 28 edges
10. `ocean_tensor_release()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `ocean_destroy_Tensor()` --calls--> `ocean_tensor_release()`  [INFERRED]
  examples/load_npy.generated.c → std/tensor/tensor_runtime.c
- `run_benchmark()` --calls--> `CCodeGenerator`  [INFERRED]
  benchmarks/benchmark_main.py → src/codegen/generator.py
- `ocean_Tensor_save_npy()` --calls--> `ocean_tensor_save_npy()`  [INFERRED]
  examples/load_npy.generated.c → std/tensor/tensor_runtime.c
- `ocean_Tensor_to()` --calls--> `ocean_tensor_to()`  [INFERRED]
  examples/load_npy.generated.c → std/tensor/tensor_runtime.c
- `ocean_Tensor_copy()` --calls--> `ocean_tensor_copy()`  [INFERRED]
  examples/load_npy.generated.c → std/tensor/tensor_runtime.c

## Import Cycles
- None detected.

## Communities (89 total, 11 thin omitted)

### Community 0 - "run"
Cohesion: 0.05
Nodes (47): ImportProcessor, Path, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield repository std directories from the active source context., run(), test_c_code_math(), test_c_code_pthread() (+39 more)

### Community 1 - "OopMixin"
Cohesion: 0.14
Nodes (9): OopMixin, Build the canonical OOP metadata directly from parser output., Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models. (+1 more)

### Community 2 - "CallsMixin"
Cohesion: 0.14
Nodes (8): CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует вызов встроенной функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует присваивание результата встроенной функции, Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции

### Community 3 - ".validate_graph"
Cohesion: 0.07
Nodes (15): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов, Валидирует один аргумент (может быть строкой или AST), Извлекает зависимости (имена переменных) из AST (+7 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".add_error"
Cohesion: 0.09
Nodes (19): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует запись через указатель (*p = значение) (+11 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Raised when Phils ownership/borrow rules are violated during lowering., Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters. (+1 more)

### Community 8 - "Parser"
Cohesion: 0.08
Nodes (24): Parser, Remove standalone triple-quoted blocks while preserving line count., Определяет текущий scope на основе отступа, Проверяет, является ли имя именем класса, Извлекает содержимое внутри скобок, учитывая вложенность, Удаляет дублирующиеся методы из классов, Собирает унаследованные методы для всех классов, Remove ``#`` comments without corrupting string literals. (+16 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.11
Nodes (11): Парсит присваивание через разыменование указателя: *p = value, Парсит литерал кортежа, Parse an expression into the transitional Phils AST., Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке), Парсит значение опции и определяет его тип, Парсит условие для циклов и if, Парсит литерал списка: [1, 2, 3] или [[1, 2], [3, 4]] (+3 more)

### Community 10 - ".add_warning"
Cohesion: 0.06
Nodes (22): Проверяет дублирование переменных в local_variables, Валидирует таблицу символов scope'а, Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение, Проверяет деление на ноль (+14 more)

### Community 12 - "Validator"
Cohesion: 0.22
Nodes (4): Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Collect variable references from expression ASTs only., Validator

### Community 13 - ".calculate_indent_level"
Cohesion: 0.11
Nodes (12): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит итерируемое выражение для for цикла, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in…, Парсит объявление класса (+4 more)

### Community 14 - "Package"
Cohesion: 0.15
Nodes (22): create_package(), find_manifest(), _flags(), load_package(), Package, PackageError, profile_flags(), Path (+14 more)

### Community 16 - "._parse_line_impl"
Cohesion: 0.05
Nodes (24): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание значения указателя переменной: x = *p, Парсит оператор break (+16 more)

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
Cohesion: 0.11
Nodes (8): Валидирует удаление переменной, Валидирует объявление указателя, Получает информацию о символе из текущего или родительских scope'ов, Проверяет выход за границы массивов/списков, Проверяет операции со строками, Проверяет вызовы C-функций (начинающиеся с @), Пытается получить статическое значение из AST, Находит родительский узел (если есть)

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

### Community 29 - "TypedModule"
Cohesion: 0.13
Nodes (7): Typed view of one parser scope., Typed compilation unit exchanged by compiler passes., Return the typed, read-only lowering view consumed by the C backend., Find a typed scope by its parser level., Iterate semantic nodes in source/scope order., TypedModule, TypedScope

### Community 30 - "._validate_scopes"
Cohesion: 0.17
Nodes (6): Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Return the canonical typed report without compatibility projections., Validate the canonical semantic module before C lowering., Run the existing validation passes over a typed lowering view., Собирает информацию о всех символах в системе

### Community 31 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 32 - ".validate_openmp_loop"
Cohesion: 0.13
Nodes (7): Валидирует узел цикла, Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation., Collect variable references from any expression AST variant.

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
Nodes (106): cl_event, cl_int, cl_kernel, FILE, ocean_tensor_backend_kind, ocean_tensor_backend_ops, ocean_tensor_dtype, ocean_tensor_opencl_kernel_key (+98 more)

### Community 39 - ".parse_complex_expression"
Cohesion: 0.14
Nodes (7): Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок, Проверяет, является ли символ частью идентификатора, Находит позицию оператора вне скобок, строк и комментариев, Проверяет, содержит ли выражение какой-либо оператор, Очищает значение от лишних пробелов, но для сложных выражений возвращает AST

### Community 40 - "What changed"
Cohesion: 0.15
Nodes (12): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Structured diagnostics, 5. Deterministic scope cleanup, 6. Ownership-aware containers, 7. Safer class lowering, Important safety boundary (+4 more)

### Community 41 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 42 - "Any"
Cohesion: 0.37
Nodes (4): Any, Lower the parser's private graph into typed scopes and nodes., Wrap expression payloads recursively while preserving AST keys., TypedIRBuilder

### Community 43 - "test_memory_safety.py"
Cohesion: 0.27
Nodes (12): compile_ocean(), test_borrow_cannot_escape_through_return(), test_borrow_cannot_escape_to_non_borrowing_parameter(), test_borrow_is_released_at_block_exit(), test_direct_c_call_requires_unsafe_block(), test_immutable_borrow_cannot_be_passed_to_mutable_parameter(), test_mutable_and_immutable_borrows_are_exclusive(), test_owned_array_is_moved_into_by_value_parameter() (+4 more)

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
Cohesion: 0.08
Nodes (23): Enum, Diagnostic, DiagnosticReport, DiagnosticSeverity, Any, Structured compiler diagnostics. The compiler keeps diagnostics as typed values…, Diagnostic importance understood by compiler frontends., Location of a diagnostic in an Ocean source file. (+15 more)

### Community 49 - "21. Array — принятое устройство"
Cohesion: 0.67
Nodes (3): 21. Array — принятое устройство, array, list

### Community 50 - "28. Статус array/tensor на момент handoff"
Cohesion: 0.67
Nodes (3): 28. Статус array/tensor на момент handoff, Backend, Parser

### Community 51 - "30. Следующий рекомендуемый этап"
Cohesion: 0.67
Nodes (3): 30. Следующий рекомендуемый этап, Array, Tensor

### Community 55 - "file_runtime.c"
Cohesion: 0.44
Nodes (13): ocean_file_handle_t, ocean_file_close(), ocean_file_eof(), ocean_file_fail(), ocean_file_flush(), ocean_file_open(), ocean_file_read(), ocean_file_read_buffer() (+5 more)

### Community 57 - "load_npy.generated.c"
Cohesion: 0.12
Nodes (44): ocean_tensor_handle_t, main(), ocean_create_Tensor(), ocean_destroy_Tensor(), ocean_release(), ocean_Tensor_add(), ocean_Tensor_add_scalar(), ocean_Tensor_column() (+36 more)

### Community 60 - "RuntimeError"
Cohesion: 0.10
Nodes (10): RuntimeError, Генерирует прямой вызов C-функции, Генерирует выражение из AST для конструктора с подстановкой параметров, Генерирует доступ к элементу сложного атрибута (self.data[index]), Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Render validated structured OpenMP metadata as one C pragma., Return the validated collapse count for an OpenMP directive. (+2 more)

### Community 61 - "TransformerLanguageModel"
Cohesion: 0.16
Nodes (12): device, main(), A small decoder-only Transformer language model implemented with PyTorch. This…, GPT-style causal language model built from PyTorch modules., Return an additive upper-triangular causal attention mask., Return next-token logits with shape ``(batch, sequence, vocab)``., Compute teacher-forced autoregressive cross-entropy., Greedily append tokens selected from the final position's logits. (+4 more)

### Community 62 - "TypeParser"
Cohesion: 0.12
Nodes (9): infer_literal_shape(), Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, Structured representation of a Phils type. The parser still emits the canonical…, TypeParser, TypeSpec, OwnershipEffect, Typed intermediate representation for the Ocean compiler. The parser's… (+1 more)

### Community 63 - "class_model.py"
Cohesion: 0.29
Nodes (9): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+1 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (5): ClassRegistry, FieldModel, Canonical class metadata and lookup service for the C backend., A field declared directly by one class., Reset all per-compilation mutable state.

### Community 65 - "compile_c"
Cohesion: 0.21
Nodes (19): compile_c(), compile_pipeline(), Compile generated C and return the exact command that was executed., Parse, validate, and generate C without an intermediate serialization artifact., test_standard_file_and_binary_file_io(), test_openmp_collapse_source_compiles_with_gcc(), test_openmp_source_compiles_with_gcc(), Path (+11 more)

### Community 66 - "ClassModel"
Cohesion: 0.28
Nodes (4): ClassModel, Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "compiler.py"
Cohesion: 0.33
Nodes (3): Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may…, generate(), test_array_lowering_and_index_mutation()

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 70 - "ExpressionsMixin"
Cohesion: 0.33
Nodes (4): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 71 - "split_top_level"
Cohesion: 0.15
Nodes (8): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит выражение с учетом приоритетов операторов Python, Парсит выражение на текущем уровне приоритета операторов, Парсит унарные операторы, Проверяет, что оператор в данной позиции является валидным оператором, Парсит цепочки индексации типа a[0][1][2], Split text only when not nested in (), [], {}, <> or strings., split_top_level()

### Community 72 - "parse_cli_paths"
Cohesion: 0.17
Nodes (18): ArgumentParser, build_argument_parser(), cli(), parse_cli_paths(), Discover standard-library C runtimes from generated std/ headers., Resolve paths for both package commands and the old single-file CLI., Entry point used by the installed ``ocean`` console script., _standard_runtime_dependencies() (+10 more)

### Community 74 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из typed AST.

### Community 76 - "TypedNode"
Cohesion: 0.20
Nodes (3): Structured OpenMP metadata attached to a loop, if present., Semantic metadata and read-only mapping view for one graph node., TypedNode

### Community 77 - "Ocean automatic ownership model v1"
Cohesion: 0.25
Nodes (8): Borrow, Categories, Containers, FFI, Move, Ocean automatic ownership model v1, Reference alias, Return ABI

### Community 78 - "`Tensor[T]`"
Cohesion: 0.25
Nodes (8): Elementwise operations and layout transforms, Internal representation, `matmul`, NumPy `.npy` files, Public API, Purpose, Semantics of `.to(device)`, `Tensor[T]`

### Community 79 - "parser.py"
Cohesion: 0.15
Nodes (7): CImportProcessor, Просто регистрирует C импорт без парсинга, Reset all per-compilation parser state. A Parser instance can safely be reused…, Build the parser's private graph before typed lowering., Parse source into the canonical ``TypedModule`` API., _build_typed_module(), Build the typed module from the parser's internal graph.

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

### Community 85 - "main.py"
Cohesion: 0.24
Nodes (16): _command(), _compiler_settings(), default_output_paths(), _diagnostic_location(), _ensure_parent(), _explicit_source(), _load_package_for_args(), main() (+8 more)

### Community 86 - "test_openmp.py"
Cohesion: 0.33
Nodes (10): test_compile_c_adds_openmp_flag_for_generated_pragma(), test_openmp_collapse_allows_sequential_loop_after_collapsed_nest(), test_openmp_collapse_requires_enough_nested_loops(), test_openmp_collapse_requires_perfect_nesting(), test_openmp_collapse_two_nested_loops_is_emitted_and_accepted(), test_openmp_pragma_is_attached_to_for_and_emitted_as_omp(), test_openmp_reduction_is_accepted(), test_openmp_rejects_shared_scalar_write_without_reduction() (+2 more)

### Community 87 - ".validate_return"
Cohesion: 0.22
Nodes (5): Валидирует объявление переменной, Валидирует выражение (правая часть присваивания или инициализации), Валидирует оператор return, Проверяет совместимость типов при присваивании, Пытается определить тип по значению

### Community 88 - "benchmark_main.py"
Cohesion: 0.31
Nodes (8): main(), measure(), Path, Benchmark the generated C program for examples/matmul.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, ValueError

## Knowledge Gaps
- **97 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `CallsMixin`, `TypesMixin`, `OwnershipMixin`, `Parser`, `generator.py`, `ArrayCodegenMixin`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `OrchestratorMixin`, `TensorCodegenMixin`, `test_memory_safety.py`, `compile_c`, `compiler.py`, `ExpressionsMixin`, `ImportsMixin`, `test_openmp.py`, `benchmark_main.py`?**
  _High betweenness centrality (0.232) - this node is a cross-community bridge._
- **Why does `Validator` connect `Validator` to `.validate_openmp_loop`, `compile_c`, `.validate_graph`, `.add_error`, `.validate_static_method_call`, `Parser`, `.add_warning`, `test_memory_safety.py`, `debug.py`, `main.py`, `.validate_return`, `.get_symbol_info`, `benchmark_main.py`, `test_openmp.py`, `TypedModule`, `._validate_scopes`, `.check_undefined_methods`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `compile_c`, `compiler.py`, `.parse_complex_expression`, `split_top_level`, `SymbolTable`, `.parse_expression_to_ast`, `test_memory_safety.py`, `.calculate_indent_level`, `parser.py`, `._parse_line_impl`, `debug.py`, `.parse_function_call`, `main.py`, `test_openmp.py`, `benchmark_main.py`, `.parse_type_annotation`, `TypeParser`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Validator` (e.g. with `Diagnostic` and `DiagnosticReport`) actually correct?**
  _`Validator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run` be split into smaller, more focused modules?**
  _Cohesion score 0.0528169014084507 - nodes in this community are weakly interconnected._