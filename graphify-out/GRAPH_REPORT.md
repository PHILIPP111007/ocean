# Graph Report - phils_language  (2026-08-13)

## Corpus Check
- 79 files · ~95,223 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1317 nodes · 2745 edges · 85 communities (69 shown, 16 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 93 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `17d9bde9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run
- OopMixin
- CallsMixin
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
- ._lifetime_analyze_node
- HelpersMixin
- TupleCodegenMixin
- .parse_type_annotation
- CoreMixin
- Handoff.md
- JSONValidator
- .get_symbol_info
- CCodeGenerator
- parser.py
- OrchestratorMixin
- ColoredFormatter
- TensorCodegenMixin
- NamingMixin
- tensor_runtime.c
- .parse_complex_expression
- What changed
- TypeSpec
- Any
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
- ImportProcessor
- TypedModule
- ocean-lang
- RuntimeError
- TransformerLanguageModel
- TypeParser
- class_model.py
- ClassRegistry
- base.py
- ClassModel
- split_top_level
- .generate_all_methods
- IRType
- ExpressionsMixin
- .find_symbol_recursive
- .__init__
- ImportsMixin
- typed_ir.py
- TypedNode
- .resolved_methods
- TypedScope
- test_for_loop.py
- .validate_static_method_call
- test_while_loop.py
- test_functions.py
- test_print.py
- .to_legacy_json

## God Nodes (most connected - your core abstractions)
1. `JSONValidator` - 143 edges
2. `Parser` - 132 edges
3. `run()` - 60 edges
4. `ocean_tensor_fail()` - 50 edges
5. `CCodeGenerator` - 38 edges
6. `OwnershipMixin` - 29 edges
7. `build_typed_ir()` - 28 edges
8. `ocean_tensor_to()` - 27 edges
9. `compile_c()` - 26 edges
10. `compile_pipeline()` - 25 edges

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

## Communities (85 total, 16 thin omitted)

### Community 0 - "run"
Cohesion: 0.14
Nodes (20): run(), test_c_code_math(), test_c_code_pthread(), test_dict(), test_dict_get(), test_input_1(), test_input_2(), test_list_add_number_to_list_item() (+12 more)

### Community 1 - "OopMixin"
Cohesion: 0.14
Nodes (9): OopMixin, Build the canonical OOP metadata directly from parser output., Return expression addressing the root base subobject at offset zero., Yield (origin_class, field_name, field_type) from root to leaf., Generate an ARC-owned zero-initialized class instance., Initialize a zeroed field, retaining only borrowed incoming references., Initialize fields. Object memory is already zeroed by calloc., Generate constructors from the canonical class models. (+1 more)

### Community 2 - "CallsMixin"
Cohesion: 0.14
Nodes (8): CallsMixin, Генерирует вызов функции, Compatibility lowering for legacy/static_method_call parser nodes. Older parser…, Генерирует вызов встроенной функции, Dispatch method lowering by semantic type instead of one giant branch., Генерирует присваивание результата встроенной функции, Генерирует вызов конструктора, Генерирует объявление с вызовом builtin функции

### Community 3 - ".validate_scope"
Cohesion: 0.06
Nodes (25): Проверяет дублирование переменных в local_variables, Валидирует таблицу символов scope'а, Валидирует отдельный символ, Проверяет, что функция имеет return если нужно, Проверяет циклы на корректность, Проверяет соответствие типа возвращаемого значения, Warn about locals unused across the complete nested graph., Проверяет, что все пути выполнения функции возвращают значение (+17 more)

### Community 4 - "TypesMixin"
Cohesion: 0.07
Nodes (22): Resolve an object receiver and its C expression. Besides local variables and…, Resolve a class field, including fields inherited through ``base``. Derived…, Определяет, является ли тип классом, Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``., Получает имя текущего класса из контекста, Определяет, является ли выражение строкой, Проверяет, является ли выражение None, Извлекает типы ключа и значения из dict[K, V] (+14 more)

### Community 5 - ".validate_graph"
Cohesion: 0.07
Nodes (16): Валидирует граф операций, Находит родительский scope для заданного уровня, Валидирует удаление переменной, Валидирует унарную операцию, Валидирует составное присваивание, Валидирует объявление функции, Валидирует вызов функции с поддержкой AST аргументов, Валидирует один аргумент (может быть строкой или AST) (+8 more)

### Community 6 - "OwnershipMixin"
Cohesion: 0.10
Nodes (9): OwnershipError, OwnershipMixin, Raised when Phils ownership/borrow rules are violated during lowering., Hybrid automatic ownership management for the C backend. Memory model…, Return ``borrowed``, ``owned`` or ``value`` for an expression. Index/attribute…, Transfer a compiler-created temporary owner into its destination., Reject direct owner access while an exclusive borrow is active., Transfer unique buffers passed to by-value function parameters. (+1 more)

### Community 7 - "SymbolTable"
Cohesion: 0.13
Nodes (6): Добавляет атрибут в класс, Получает метод класса (ищет в родительских классах), Проверяет, является ли subclass наследником superclass, Добавляет класс в таблицу символов, Добавляет метод в класс, SymbolTable

### Community 8 - "Parser"
Cohesion: 0.06
Nodes (17): Parser, Remove standalone triple-quoted blocks while preserving line count., Парсит присваивание значения указателя переменной: x = *p, Парсит оператор break, Парсит оператор continue, Парсит оператор del (полное удаление), Возвращает область видимости для заданного уровня отступа, Определяет текущий scope на основе отступа (+9 more)

### Community 9 - ".parse_expression_to_ast"
Cohesion: 0.11
Nodes (11): Парсит литерал кортежа, Парсит оператор return, Parse an expression into the transitional Phils AST., Универсальный парсер аргументов функции. Возвращает (positional_args,…, Проверяет, находится ли "=" внутри скобок (например, в словаре или списке), Парсит значение опции и определяет его тип, Парсит условие для циклов и if, Парсит литерал списка: [1, 2, 3] или [[1, 2], [3, 4]] (+3 more)

### Community 10 - ".check_undefined_methods"
Cohesion: 0.22
Nodes (5): Проверяет, что все используемые методы определены в классе или его родителях, Проверяет, существует ли метод в классе или его иерархии наследования, Извлекает вызовы методов из AST, Проверяет, является ли метод встроенным для данного типа, Добавляет класс в реестр классов

### Community 12 - ".add_error"
Cohesion: 0.08
Nodes (22): Валидирует типы в узле, Валидирует типы в присваивании, Проверяет тип объявления по типизированному AST., Валидирует типы возвращаемых значений, Валидирует типы в условии while, Валидирует типы в условии if/elif, Валидирует типы в операциях, Валидирует объявление переменной (+14 more)

### Community 13 - "._parse_line_impl"
Cohesion: 0.14
Nodes (11): Parse an explicit unsafe region without changing runtime semantics. ``unsafe:``…, Parse a free function with fully nested type annotations., Находит конец блока с отступом, Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла, Парсит вложенные if внутри других блоков (while, for, других if), Parse one line and attach its source location to emitted nodes., Parse a value-semantic struct. Structs intentionally contain fields only in…, Парсит объявление класса (+3 more)

### Community 14 - "main.py"
Cohesion: 0.06
Nodes (72): ArgumentParser, build_argument_parser(), cli(), _command(), compile_c(), compile_pipeline(), _compiler_settings(), default_output_paths() (+64 more)

### Community 16 - ".extract_dependencies_from_ast"
Cohesion: 0.09
Nodes (13): Парсит присваивание срезу: my_list[1:3] = [20, 30], Парсит создание объекта с присваиванием: var x: Class = Class(args), Парсит вызов конструктора без присваивания: Class(args), Парсит присваивание по индексу с поддержкой многомерных массивов, Парсит составную операцию с индексом: my_list[0] += 5, Парсит присваивание атрибуту: obj.attr = value, Парсит присваивание через разыменование указателя: *p = value, Парсит присваивание результата вызова функции: var x: type = func(args) (+5 more)

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
Cohesion: 0.08
Nodes (24): 10. C interop, 11. Parser, 13. `&x` vs borrow, 14. Struct, 16. Strings, 17. Bounds safety, 18. SIMD, 19. Demand-driven runtime (+16 more)

### Community 29 - "JSONValidator"
Cohesion: 0.17
Nodes (21): JSONValidator, Validate the parser's typed graph before C code generation. The validator is…, Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК, Reject raw pointers and direct C FFI unless explicitly marked unsafe., build_typed_ir(), Convenience entry point used by the compiler pipeline and tests., _parse(), test_openmp_collapse_allows_sequential_loop_after_collapsed_nest() (+13 more)

### Community 30 - ".get_symbol_info"
Cohesion: 0.15
Nodes (5): Валидирует объявление указателя, Определяет тип возвращаемого значения, Получает информацию о символе из текущего или родительских scope'ов, Проверяет выход за границы массивов/списков, Пытается получить статическое значение из AST

### Community 31 - "CCodeGenerator"
Cohesion: 0.33
Nodes (9): CCodeGenerator, Public C backend façade. The implementation is split by responsibility into…, compile_and_run(), test_oop_constructor_method_mutation_and_composition(), test_oop_default_constructor_without_init(), test_oop_inherited_field_access_uses_embedded_base_layout(), test_oop_metadata_has_one_canonical_class_model(), test_oop_rejects_inheritance_cycles() (+1 more)

### Community 32 - "parser.py"
Cohesion: 0.16
Nodes (7): CImportProcessor, Просто регистрирует C импорт без парсинга, generate(), test_array_lowering_and_index_mutation(), test_matmul_1(), test_matmul_2(), test_matmul_3()

### Community 33 - "OrchestratorMixin"
Cohesion: 0.17
Nodes (8): OrchestratorMixin, Генерирует имя временной переменной, Генерирует код для узла графа, Generate C from the canonical semantic IR., Compatibility entry point for callers with parser JSON., Return whether an AST value contains a direct ``@c_function(...)``., Генерирует объявление глобальной переменной, Lower typed scope views after semantic IR construction.

### Community 34 - "ColoredFormatter"
Cohesion: 0.33
Nodes (4): LogRecord, ColoredFormatter, Set up a custom logger with optional configuration parameters. :param name:…, setup_logger()

### Community 37 - "tensor_runtime.c"
Cohesion: 0.13
Nodes (66): cl_int, cl_kernel, ocean_tensor_dtype, ocean_tensor_handle_t, ocean_tensor_alloc(), ocean_tensor_alloc_zeros(), ocean_tensor_apply_binary(), ocean_tensor_binary() (+58 more)

### Community 39 - ".parse_complex_expression"
Cohesion: 0.13
Nodes (8): Парсит выражение на текущем уровне приоритета операторов, Проверяет, что оператор в данной позиции является валидным оператором, Разбирает сложные выражения с несколькими операторами и скобками, Проверяет, полностью ли выражение заключено в скобки, Находит оператор с наименьшим приоритетом вне скобок, Проверяет, является ли символ частью идентификатора, Находит позицию оператора вне скобок, строк и комментариев, Проверяет, содержит ли выражение какой-либо оператор

### Community 40 - "What changed"
Cohesion: 0.17
Nodes (11): 1. `ocean_` C namespace, 2. Automatic ownership management, 3. Hybrid borrow checker v1, 4. Deterministic scope cleanup, 5. Ownership-aware containers, 6. Safer class lowering, Important safety boundary, Module layout (+3 more)

### Community 42 - "Any"
Cohesion: 0.35
Nodes (4): Any, Return a lossless compatibility projection of parser scopes. The returned…, Lower parser dictionaries into typed scopes and effect-annotated nodes., TypedIRBuilder

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
Cohesion: 0.15
Nodes (6): Return OpenMP clauses grouped by name, preserving duplicates., Whether a type is safe to create/use as a private scalar., Validate the deliberately conservative, race-aware OpenMP subset., Validate and return the perfectly nested loop chain., Validate structured OpenMP metadata before C code generation., Collect variable references from any expression AST variant.

### Community 56 - "ImportProcessor"
Cohesion: 0.20
Nodes (7): ImportProcessor, Path, Обрабатывает импорт и возвращает содержимое импортируемого файла, Обрабатывает все импорты в коде и вставляет содержимое файлов, Yield repository std directories from the active source context., test_import(), test_relative_and_standard_imports()

### Community 57 - "TypedModule"
Cohesion: 0.12
Nodes (9): Собирает информацию о всех символах в системе, Строит карту соответствия узлов исходным строкам, Возвращает отчет о проверке, Validate a typed module or legacy parser graph. ``TypedModule`` is the…, Validate the canonical semantic module before C lowering., Run the existing validation passes over a typed lowering view., Typed compilation unit with an explicit compatibility projection. Compiler…, Iterate semantic nodes in source/scope order. (+1 more)

### Community 60 - "RuntimeError"
Cohesion: 0.10
Nodes (10): RuntimeError, Генерирует прямой вызов C-функции, Генерирует выражение из AST для конструктора с подстановкой параметров, Генерирует доступ к элементу сложного атрибута (self.data[index]), Lower attribute references in range bounds to their C form., Generate Python-compatible range direction and a per-iteration scope., Render validated structured OpenMP metadata as one C pragma., Return the validated collapse count for an OpenMP directive. (+2 more)

### Community 61 - "TransformerLanguageModel"
Cohesion: 0.15
Nodes (13): device, main(), A small decoder-only Transformer language model implemented with PyTorch. This…, GPT-style causal language model built from PyTorch modules., Return an additive upper-triangular causal attention mask., Return next-token logits with shape ``(batch, sequence, vocab)``., Compute teacher-forced autoregressive cross-entropy., Greedily append tokens selected from the final position's logits. (+5 more)

### Community 62 - "TypeParser"
Cohesion: 0.27
Nodes (4): infer_literal_shape(), Recursive parser for Phils type expressions., Infer a rectangular shape from nested list literals. Returns ``None`` for…, TypeParser

### Community 63 - "class_model.py"
Cohesion: 0.29
Nodes (9): build_class_registry(), _infer_field_type(), MethodModel, Any, Semantic class metadata shared by the OOP lowering passes. The parser graph is…, Infer only the structural type information needed for class layout., Build all class metadata directly from the parser graph and scopes., A class method declaration and its corresponding body scope. (+1 more)

### Community 64 - "ClassRegistry"
Cohesion: 0.21
Nodes (5): ClassRegistry, FieldModel, Canonical class metadata and lookup service for the C backend., A field declared directly by one class., Reset all per-compilation mutable state.

### Community 65 - "base.py"
Cohesion: 0.15
Nodes (6): test_del(), test_global_var_declaration(), test_if_else_1(), test_dict(), test_methods(), test_variables()

### Community 66 - "ClassModel"
Cohesion: 0.28
Nodes (4): ClassModel, Complete semantic metadata for one Ocean class., Yield direct parents first while detecting inheritance cycles., Resolve a field through the single-inheritance chain.

### Community 67 - "split_top_level"
Cohesion: 0.22
Nodes (6): Парсит многомерное присваивание по индексу: A_data[0][0] = 10, Парсит выражение с учетом приоритетов операторов Python, Парсит унарные операторы, Парсит цепочки индексации типа a[0][1][2], Split text only when not nested in (), [], {}, <> or strings., split_top_level()

### Community 68 - ".generate_all_methods"
Cohesion: 0.25
Nodes (4): Генерирует все методы всех классов, включая унаследованные, Генерирует заглушку для унаследованного метода, Build method resolution metadata from canonical class models., Generate a method with borrowed parameters and automatic owner cleanup.

### Community 70 - "ExpressionsMixin"
Cohesion: 0.33
Nodes (4): ExpressionsMixin, Генерирует C выражение из AST с поддержкой tuple и list, Генерирует доступ к атрибуту объекта, Генерирует выражение из AST с подстановкой параметров конструктора

### Community 71 - ".find_symbol_recursive"
Cohesion: 0.29
Nodes (3): Парсит составные операции присваивания, Строит операции из AST выражения, Рекурсивно ищет символ в текущем и родительских scope'ах

### Community 74 - "ImportsMixin"
Cohesion: 0.29
Nodes (4): ImportsMixin, Генерирует #include директивы, Генерирует forward declarations функций, Собирает импорты и объявления функций из JSON

### Community 75 - "typed_ir.py"
Cohesion: 0.13
Nodes (16): main(), measure(), Path, Benchmark the generated C program for examples/matmul.oc. The benchmark…, run_benchmark(), runtime_summary(), CompletedProcess, Compatibility façade for the Phils Ocean C backend v0.2. Existing callers may… (+8 more)

### Community 76 - "TypedNode"
Cohesion: 0.20
Nodes (3): Semantic metadata and read-only mapping view for one graph node., Structured OpenMP metadata attached to a loop, if present., TypedNode

### Community 77 - ".resolved_methods"
Cohesion: 0.50
Nodes (3): Resolve methods without rebuilding parser-shaped dictionaries., A method together with the class that provides its implementation., ResolvedMethod

### Community 78 - "TypedScope"
Cohesion: 0.20
Nodes (4): Return the typed, read-only lowering view consumed by the C backend., Find a typed scope by its parser level., Typed view of one parser scope., TypedScope

### Community 79 - "test_for_loop.py"
Cohesion: 0.40
Nodes (4): test_for_loop_1(), test_for_loop_2(), test_for_loop_3(), test_for_loop_4()

### Community 81 - "test_while_loop.py"
Cohesion: 0.50
Nodes (3): test_while_loop_1(), test_while_loop_2(), test_while_loop_3()

## Knowledge Gaps
- **93 isolated node(s):** `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands`, `Coding Style & Naming Conventions`, `Testing Guidelines` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JSONValidator` connect `JSONValidator` to `.validate_scope`, `.validate_graph`, `.check_undefined_methods`, `typed_ir.py`, `.add_error`, `test_memory_safety.py`, `main.py`, `.validate_static_method_call`, `._lifetime_analyze_node`, `.validate_openmp_loop`, `TypedModule`, `.get_symbol_info`?**
  _High betweenness centrality (0.278) - this node is a cross-community bridge._
- **Why does `CCodeGenerator` connect `CCodeGenerator` to `run`, `OopMixin`, `CallsMixin`, `TypesMixin`, `OwnershipMixin`, `generator.py`, `main.py`, `ArrayCodegenMixin`, `ListCodegenMixin`, `IndexingMixin`, `IoMixin`, `StatementsMixin`, `ScopeMixin`, `HelpersMixin`, `TupleCodegenMixin`, `CoreMixin`, `JSONValidator`, `parser.py`, `OrchestratorMixin`, `TensorCodegenMixin`, `NamingMixin`, `test_memory_safety.py`, `DictCodegenMixin`, `ExpressionsMixin`, `ImportsMixin`, `typed_ir.py`?**
  _High betweenness centrality (0.268) - this node is a cross-community bridge._
- **Why does `Parser` connect `Parser` to `run`, `SymbolTable`, `.parse_expression_to_ast`, `._parse_line_impl`, `main.py`, `.extract_dependencies_from_ast`, `.parse_function_call`, `.parse_type_annotation`, `JSONValidator`, `CCodeGenerator`, `parser.py`, `.parse_complex_expression`, `test_memory_safety.py`, `TypeParser`, `base.py`, `split_top_level`, `.find_symbol_recursive`, `.__init__`, `typed_ir.py`?**
  _High betweenness centrality (0.240) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `SymbolTable` and `TypeParser`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ocean-lang`, `Project Structure & Module Organization`, `Build, Test, and Development Commands` to the rest of the system?**
  _93 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run` be split into smaller, more focused modules?**
  _Cohesion score 0.14333333333333334 - nodes in this community are weakly interconnected._
- **Should `OopMixin` be split into smaller, more focused modules?**
  _Cohesion score 0.1368421052631579 - nodes in this community are weakly interconnected._