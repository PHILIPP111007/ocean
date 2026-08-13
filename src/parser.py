import re
import os

from src.modules.imports import CImportProcessor, ImportProcessor
from src.modules.constants import KEYS, DATA_TYPES, METHOD_DECORATORS
from src.modules.symbol_table import SymbolTable
from src.modules.logger import logger
from src.parsing.type_system import (
    TypeParser,
    find_top_level,
    infer_literal_shape,
    split_top_level,
)


class Parser:
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.type_parser = TypeParser()
        self.builtin_functions = {
            "print",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "range",
            "input",
        }
        self.import_processor = ImportProcessor(base_path=base_path)
        self.c_import_processor = CImportProcessor(base_path=base_path)
        self.reset_state()
    def reset_state(self) -> None:
        """Reset all per-compilation parser state.

        A Parser instance can safely be reused for more than one source file.
        Import processors are deliberately preserved because their base path is
        configuration, not compilation state.
        """
        self.scopes = []
        self.scope_stack = []
        self.symbol_counter = 0
        self.current_indent = 0
        self.indent_size = None
        self.indent_char = None
        self.unsafe_depth = 0
        self.current_source_line = None
        self.current_source_file = ""
        self._pending_openmp_pragma = None

    def _mark_unsafe_nodes(self, value) -> None:
        """Mark graph nodes parsed inside an explicit ``unsafe:`` block."""
        if isinstance(value, dict):
            if "node" in value:
                value["unsafe"] = True
            for child in value.values():
                self._mark_unsafe_nodes(child)
        elif isinstance(value, list):
            for child in value:
                self._mark_unsafe_nodes(child)


    def parse_type_annotation(self, type_text: str) -> tuple[str, dict]:
        """Return canonical type text and structured metadata."""
        spec = self.type_parser.parse(type_text)
        return spec.canonical, spec.to_dict()


    def _strip_comments_preserving_strings(self, code: str) -> str:
        """Remove ``#`` comments without corrupting string literals."""
        result = []
        for line in code.splitlines():
            # OpenMP pragmas are language syntax, not comments.  Preserve
            # both the canonical spelling (``omp``) and the historical typo
            # accepted by the frontend (``opm``); code generation always emits
            # the standard ``omp`` spelling.
            if re.match(r"^\s*#pragma\s+(?:omp|opm)\b", line):
                result.append(line)
                continue
            out = []
            in_string = False
            quote = ""
            escaped = False
            i = 0
            while i < len(line):
                char = line[i]
                if escaped:
                    out.append(char)
                    escaped = False
                    i += 1
                    continue
                if in_string:
                    out.append(char)
                    if char == "\\":
                        escaped = True
                    elif char == quote:
                        in_string = False
                        quote = ""
                    i += 1
                    continue
                if char in {'"', "'"}:
                    in_string = True
                    quote = char
                    out.append(char)
                    i += 1
                    continue
                if char == "#":
                    break
                out.append(char)
                i += 1
            result.append("".join(out))
        return "\n".join(result)


    def _strip_triple_quoted_blocks(self, code: str) -> str:
        """Remove standalone triple-quoted blocks while preserving line count."""
        pattern = re.compile(r"((?:\'{3})|(?:\"{3})).*?\1", flags=re.DOTALL)

        def repl(match):
            return "\n" * match.group(0).count("\n")

        return pattern.sub(repl, code)


    def _type_info_for_symbol(self, type_text: str) -> dict:
        _, info = self.parse_type_annotation(type_text)
        return info


    def _is_owned_container_type(self, type_text: str) -> bool:
        info = self._type_info_for_symbol(type_text)
        return info.get("memory_kind") == "owned"


    def _is_borrow_type(self, type_text: str) -> bool:
        info = self._type_info_for_symbol(type_text)
        return info.get("kind") in {"borrow", "mut_borrow"}


    def _borrow_source_from_ast(self, ast_node: dict) -> str | None:
        if not isinstance(ast_node, dict):
            return None
        if ast_node.get("type") == "variable":
            return ast_node.get("name") or ast_node.get("value")
        if ast_node.get("type") == "borrow":
            source = ast_node.get("source", {})
            if isinstance(source, dict) and source.get("type") == "variable":
                return source.get("name") or source.get("value")
        return None


    def _parse_struct_field(self, line: str) -> dict | None:
        """Parse ``name: Type`` or ``name: Type = default``."""
        colon = find_top_level(line, ":")
        if colon <= 0:
            return None
        name = line[:colon].strip()
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return None
        rest = line[colon + 1 :].strip()
        equals = find_top_level(rest, "=")
        if equals >= 0:
            type_text = rest[:equals].strip()
            default_text = rest[equals + 1 :].strip()
            default_ast = self.parse_expression_to_ast(default_text)
        else:
            type_text = rest
            default_text = None
            default_ast = None
        canonical, type_info = self.parse_type_annotation(type_text)
        return {
            "name": name,
            "type": canonical,
            "type_info": type_info,
            "default": default_ast,
            "default_source": default_text,
        }

    ###############################################################################################
    # INDENT
    ###############################################################################################

    def detect_indent_type(self, line: str):
        if not line.startswith((" ", "\t")):
            return None

        first_char = line[0]
        if first_char == "\t":
            return ("tab", 1)
        elif first_char == " ":
            space_count = 0
            for char in line:
                if char == " ":
                    space_count += 1
                else:
                    break

            common_indents = [2, 4, 8]
            for indent in common_indents:
                if space_count % indent == 0:
                    return ("space", indent)

            return ("space", space_count)

        return None

    def analyze_indent_pattern(self, lines: list) -> tuple:
        tab_lines = 0
        space_lines = 0
        space_counts = {}

        for line in lines:
            if line.startswith("\t"):
                tab_lines += 1
            elif line.startswith(" "):
                space_lines += 1
                space_count = 0
                for char in line:
                    if char == " ":
                        space_count += 1
                    else:
                        break

                if space_count > 0:
                    # Фиксируем размер отступа на 4 пробела
                    # (или определите по первому ненулевому отступу)
                    if space_count >= 4:
                        space_count = 4  # предполагаем, что отступ 4 пробела
                    space_counts[space_count] = space_counts.get(space_count, 0) + 1

        # Всегда используем 4 пробела, если нет табов
        if space_lines > 0:
            return ("space", 4)

        return ("tab", 1)

    def get_current_scope_for_indent(self, indent: int):
        """Возвращает область видимости для заданного уровня отступа"""
        # Если отступ 0 - возвращаем глобальную область
        if indent == 0:
            # Находим глобальную область в стеке
            for scope in self.scope_stack:
                if scope["level"] == 0:
                    return scope
            return self.scope_stack[0] if self.scope_stack else None

        # Находим область с нужным уровнем
        for scope in reversed(self.scope_stack):
            if scope["level"] <= indent:
                return scope

        # Если не нашли, возвращаем последнюю область
        return self.scope_stack[-1] if self.scope_stack else None

    def handle_indent_change(self, indent: int):
        if indent > self.current_indent:
            self.current_indent = indent
        elif indent < self.current_indent:
            while self.current_indent > indent and len(self.scope_stack) > 1:
                self.scope_stack.pop()
                self.current_indent -= 1

    def calculate_indent_level(self, line: str) -> int:
        if not line.startswith((" ", "\t")):
            return 0

        if self.indent_size is None:
            indent_info = self.detect_indent_type(line)
            if indent_info:
                self.indent_char, self.indent_size = indent_info

        if self.indent_char == "tab":
            tab_count = 0
            for char in line:
                if char == "\t":
                    tab_count += 1
                else:
                    break
            return tab_count
        elif self.indent_char == "space":
            space_count = 0
            for char in line:
                if char == " ":
                    space_count += 1
                else:
                    break

            # ВАЖНО: Используем целочисленное деление
            if self.indent_size > 0:
                level = space_count // self.indent_size
                return level
            else:
                return 0

        return 0

    def find_indented_block_end(
        self, lines: list, start_index: int, base_indent: int
    ) -> int:
        """Находит конец блока с отступом"""
        if start_index >= len(lines):
            return start_index

        i = start_index
        while i < len(lines):
            line = lines[i]

            # Пропускаем пустые строки
            if not line.strip():
                i += 1
                continue

            current_indent = self.calculate_indent_level(line)

            # Если отступ стал меньше или равен базовому - конец блока
            if current_indent <= base_indent:
                return i

            i += 1

        return len(lines)  # Дошли до конца файла

    ###############################################################################################
    # SCOPE
    ###############################################################################################

    def get_current_scope(self, indent):
        """Определяет текущий scope на основе отступа"""
        if indent == 0:
            return self.scopes[0]  # Глобальная область

        # Ищем самую глубокую функцию
        for scope in reversed(self.scopes):
            if scope["type"] == "function":
                return scope

        return self.scopes[0]

    ###############################################################################################
    # IMPORTS
    ###############################################################################################

    def parse_cimport(
        self, line: str, scope: dict, all_lines: list, current_index: int
    ):
        """Парсит C импорт"""
        import_info = self.c_import_processor.resolve_cimport(line)

        if import_info:
            # Добавляем узел C импорта
            scope["graph"].append(
                {
                    "node": "c_import",
                    "content": line,
                    "header": import_info["header"],
                    "is_system": import_info["is_system"],
                    "operations": [
                        {
                            "type": "C_IMPORT",
                            "header": import_info["header"],
                            "is_system": import_info["is_system"],
                        }
                    ],
                }
            )
            logger.debug(
                f"Добавлен C импорт: {import_info['header']} (системный: {import_info['is_system']})"
            )

        return current_index + 1

    ###############################################################################################
    # PARSE
    ###############################################################################################

    def _parse_graph(self, code: str, file_path: str = "") -> list[dict]:
        """Build the parser's private graph before typed lowering."""

        def _parse_processed_code(processed: str) -> list[dict]:
            processed = self._strip_triple_quoted_blocks(processed)
            processed = self._strip_comments_preserving_strings(processed)
            lines = processed.split("\n")

            if any(line.startswith((" ", "\t")) for line in lines if line.strip()):
                self.indent_char, self.indent_size = self.analyze_indent_pattern(lines)

            global_scope = {
                "level": 0,
                "type": "module",
                "parent_scope": None,
                "local_variables": [],
                "graph": [],
                "symbol_table": SymbolTable(),
            }
            self.scopes.append(global_scope)
            self.scope_stack = [global_scope]
            self.current_indent = 0
            self._pending_openmp_pragma = None

            i = 0
            while i < len(lines):
                raw_line = lines[i]
                if not raw_line.strip():
                    i += 1
                    continue

                indent_level = self.calculate_indent_level(raw_line)
                line_content = raw_line.strip()

                if indent_level < self.current_indent:
                    while len(self.scope_stack) > 1 and self.current_indent > indent_level:
                        self.scope_stack.pop()
                        self.current_indent -= 1

                self.current_indent = indent_level
                current_scope = self.scope_stack[-1] if self.scope_stack else global_scope
                i = self.parse_line(
                    line_content,
                    current_scope,
                    lines,
                    i,
                    indent_level,
                    source_column=len(raw_line) - len(raw_line.lstrip()) + 1,
                )

            # A directive that reaches EOF without a following ``for`` must
            # remain visible to the validator instead of being silently lost.
            if self._pending_openmp_pragma is not None:
                global_scope["graph"].append(
                    {
                        "node": "openmp_pragma",
                        "content": self._pending_openmp_pragma.get("content", ""),
                        "openmp": self._pending_openmp_pragma,
                        "error": "OpenMP pragma must be immediately followed by a for loop",
                        "source_line": self._pending_openmp_pragma.get("source_line"),
                    }
                )
                self._pending_openmp_pragma = None

            self.remove_duplicate_methods()
            self.collect_inherited_methods_for_all_classes()

            for current_scope in self.scopes:
                if hasattr(current_scope["symbol_table"], "symbols"):
                    current_scope["symbol_table"] = current_scope["symbol_table"].symbols

            return self.scopes

        self.reset_state()
        self.current_source_file = file_path

        if file_path:
            base_dir = os.path.dirname(file_path)
            self.import_processor.base_path = base_dir
            self.c_import_processor.base_path = base_dir

        processed_code = self.import_processor.process_imports(code, file_path)
        return _parse_processed_code(processed_code)

    def parse_typed(self, code: str, file_path: str = ""):
        """Parse source into the canonical ``TypedModule`` API."""
        from src.typed_ir import _build_typed_module

        return _build_typed_module(self._parse_graph(code, file_path=file_path))

    def parse_line(
        self,
        line: str,
        scope: dict,
        all_lines: list,
        current_index: int,
        indent: int,
        source_column: int = 1,
    ):
        """Parse one line and attach its source location to emitted nodes."""
        if source_column == 1 and 0 <= current_index < len(all_lines):
            raw_source_line = all_lines[current_index]
            source_column = len(raw_source_line) - len(raw_source_line.lstrip()) + 1
        previous_source_line = self.current_source_line
        self.current_source_line = current_index + 1
        graph_start = len(scope.get("graph", []))
        try:
            return self._parse_line_impl(line, scope, all_lines, current_index, indent)
        finally:
            for node in scope.get("graph", [])[graph_start:]:
                if isinstance(node, dict):
                    node.setdefault("source_line", current_index + 1)
                    node.setdefault("source_file", self.current_source_file or None)
                    node.setdefault("source_column", source_column)
            self.current_source_line = previous_source_line

    def _parse_line_impl(
        self, line: str, scope: dict, all_lines: list, current_index: int, indent: int
    ):
        """Основной метод парсинга строки с поддержкой всех конструкций"""
        line = line.strip()
        line_content = line.strip()

        if not line:
            return current_index + 1

        # ``#pragma omp parallel for`` belongs to the immediately following
        # Ocean ``for`` node.  Keep the parser state outside the graph until
        # that node is seen so normal code generation never has to deal with
        # a standalone preprocessor directive.
        if line.startswith("#pragma"):
            pragma = self.parse_openmp_pragma(line, current_index + 1)
            if self._pending_openmp_pragma is not None:
                scope["graph"].append(
                    {
                        "node": "openmp_pragma",
                        "content": self._pending_openmp_pragma.get("content", ""),
                        "openmp": self._pending_openmp_pragma,
                        "error": "multiple OpenMP pragmas before one for loop",
                        "source_line": self._pending_openmp_pragma.get("source_line"),
                    }
                )
            self._pending_openmp_pragma = pragma or {
                "backend": "openmp",
                "directive": "invalid",
                "content": line,
                "source_line": current_index + 1,
                "error": "expected '#pragma omp parallel for'",
            }
            return current_index + 1

        if self._pending_openmp_pragma is not None:
            # Blank lines are skipped by _parse_graph and loop-body parsers.  Any
            # actual statement other than ``for`` makes the pragma invalid.
            if not re.match(r"^for\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s+.+:\s*$", line):
                pending = self._pending_openmp_pragma
                scope["graph"].append(
                    {
                        "node": "openmp_pragma",
                        "content": pending.get("content", ""),
                        "openmp": pending,
                        "error": "OpenMP pragma must be immediately followed by a for loop",
                        "source_line": pending.get("source_line"),
                    }
                )
                self._pending_openmp_pragma = None

        # Value-only structures used by OS/HPC code.  This is handled before
        # KEYS so it works even while older constants.py files do not yet list
        # ``struct`` as a keyword.
        if line.startswith("struct "):
            return self.parse_struct_declaration(
                line, scope, all_lines, current_index
            )

        # ========== ПЕРВОЕ: проверяем многомерное индексное присваивание ==========
        # Паттерн: var_name[0][0] = value (любое количество индексов)
        nested_index_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\[[^\]]*\])+)\s*=\s*(.+)$"
        nested_index_match = re.match(nested_index_pattern, line_content)

        if nested_index_match:
            var_name, indices_str, value = nested_index_match.groups()
            parsed = self.parse_nested_index_assignment(
                line_content, scope, var_name, indices_str, value
            )
            return current_index + 1

        # ========== ДОБАВЛЯЕМ ОБРАБОТКУ self.data[index] = value ==========
        # Паттерн: self.attr[index] = value
        self_index_pattern = r"^self\.([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]\s*=\s*(.+)$"
        self_index_match = re.match(self_index_pattern, line_content)

        if self_index_match:
            attr_name, index_expr, value = self_index_match.groups()
            logger.debug(
                f"Найдено присваивание self.{attr_name}[{index_expr}] = {value}"
            )

            # A class tensor uses comma-separated coordinates (self.w[i, j]).
            # Preserve every coordinate instead of turning the whole text into
            # one opaque expression.
            index_parts = split_top_level(index_expr) or [index_expr]
            index_asts = [self.parse_expression_to_ast(part) for part in index_parts]
            value_ast = self.parse_expression_to_ast(value)

            if len(index_asts) > 1:
                scope["graph"].append(
                    {
                        "node": "nested_index_assignment",
                        "content": line,
                        "variable": f"self.{attr_name}",
                        "indices": index_asts,
                        "value": value_ast,
                        "operations": [
                            {
                                "type": "NESTED_INDEX_ASSIGN",
                                "variable": f"self.{attr_name}",
                                "indices": index_asts,
                                "value": value_ast,
                                "depth": len(index_asts),
                            }
                        ],
                        "dependencies": list(
                            dict.fromkeys(
                                [
                                    dependency
                                    for index_ast in index_asts
                                    for dependency in self.extract_dependencies_from_ast(index_ast)
                                ]
                                + self.extract_dependencies_from_ast(value_ast)
                            )
                        ),
                    }
                )
                return current_index + 1

            index_ast = index_asts[0]

            # Создаем узел для присваивания
            operations = [
                {
                    "type": "INDEX_ASSIGN",
                    "variable": f"self.{attr_name}",
                    "index": index_ast,
                    "value": value_ast,
                }
            ]

            dependencies = []
            deps = self.extract_dependencies_from_ast(index_ast)
            dependencies.extend(deps)
            deps = self.extract_dependencies_from_ast(value_ast)
            dependencies.extend(deps)

            scope["graph"].append(
                {
                    "node": "index_assignment",
                    "content": line,
                    "variable": f"self.{attr_name}",
                    "index": index_ast,
                    "value": value_ast,
                    "operations": operations,
                    "dependencies": list(set(dependencies)),
                }
            )

            return current_index + 1

        # ========== ВТОРОЕ: обычное индексное присваивание ==========
        # Паттерн: var_name[0] = value (один индекс)
        simple_index_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]\s*=\s*(.+)$"
        simple_index_match = re.match(simple_index_pattern, line_content)

        if simple_index_match:
            var_name, index_expr, value = simple_index_match.groups()
            parsed = self.parse_index_assignment(
                line_content, scope, var_name, index_expr, value
            )
            return current_index + 1

        # Определяем реальный отступ текущей строки
        actual_indent = (
            self.calculate_indent_level(all_lines[current_index])
            if current_index < len(all_lines)
            else 0
        )

        # Обрабатываем изменение отступа
        self.handle_indent_change(actual_indent)

        # Получаем текущую область видимости для данного отступа
        current_scope = self.get_current_scope_for_indent(actual_indent)
        if not current_scope:
            current_scope = scope

        # ========== СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ КОНСТРУКТОРА ==========
        # Если мы в конструкторе класса и строка начинается с "var self."
        if current_scope.get("type") == "constructor" and line.startswith("self."):
            # Парсим инициализацию атрибута в конструкторе
            pattern = r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)"
            match = re.match(pattern, line)

            if match:
                # Это инициализация атрибута
                result = self.parse_class_attribute_initialization(line, current_scope)
                return current_index + 1 if result else current_index + 1

        # ========== ОБРАБОТКА ПРИСВАИВАНИЯ СРЕЗАМ ==========

        # Паттерн: my_list[1:3] = [20, 30]
        slice_assign_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+):(\d+)\]\s*=\s*(.+)$"
        slice_assign_match = re.match(slice_assign_pattern, line_content)

        if slice_assign_match:
            var_name, start_str, stop_str, value = slice_assign_match.groups()
            parsed = self.parse_slice_assignment(
                line_content, scope, var_name, int(start_str), int(stop_str), value
            )
            return current_index + 1

        # Паттерн: my_list[:3] = [20, 30]
        slice_assign_pattern2 = r"^([a-zA-Z_][a-zA-Z0-9_]*)\[:(\d+)\]\s*=\s*(.+)$"
        slice_assign_match2 = re.match(slice_assign_pattern2, line_content)

        if slice_assign_match2:
            var_name, stop_str, value = slice_assign_match2.groups()
            parsed = self.parse_slice_assignment(
                line_content, scope, var_name, None, int(stop_str), value
            )
            return current_index + 1

        # Паттерн: my_list[1:] = [20, 30]
        slice_assign_pattern3 = r"^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+):\]\s*=\s*(.+)$"
        slice_assign_match3 = re.match(slice_assign_pattern3, line_content)

        if slice_assign_match3:
            var_name, start_str, value = slice_assign_match3.groups()
            parsed = self.parse_slice_assignment(
                line_content, scope, var_name, int(start_str), None, value
            )
            return current_index + 1

        # ========== ОБРАБОТКА СОСТАВНЫХ ОПЕРАЦИЙ С ИНДЕКСАМИ ==========

        # Паттерн: my_list[0] += 5
        augmented_index_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]\s*(\+=|-=|\*=|/=|//=|%=)\s*(.+)$"
        )
        augmented_index_match = re.match(augmented_index_pattern, line_content)

        if augmented_index_match:
            var_name, index_str, operator, value = augmented_index_match.groups()
            parsed = self.parse_augmented_index_assignment(
                line_content, scope, var_name, index_str, operator, value
            )
            return current_index + 1

        # ========== ОБРАБОТКА ИНДЕКСАЦИИ ==========

        # Проверяем присваивание по индексу: my_list[0] = value
        index_assignment_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\[[^\]]*\])+)\s*=\s*(.+)$"
        )
        index_assignment_match = re.match(index_assignment_pattern, line)

        if index_assignment_match:
            var_name, index_expr, value = index_assignment_match.groups()
            # Убираем возможные пробелы вокруг индексов
            index_expr = index_expr.strip()
            parsed = self.parse_index_assignment(
                line, current_scope, var_name, index_expr, value
            )
            return current_index + 1

        # Проверяем составные операции с индексацией: my_list[0] += 1

        # Составные операции с индексами: my_list[0] += 5
        augmented_index_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]\s*(\+=|-=|\*=|/=|//=|%=)\s*(.+)$"
        )
        augmented_index_match = re.match(augmented_index_pattern, line)

        if augmented_index_match:
            var_name, index, operator, value = augmented_index_match.groups()
            parsed = self.parse_augmented_index_assignment(
                line, scope, var_name, index, operator, value
            )
            return current_index + 1

        # ========== ОБРАБОТКА ДЕКОРАТОРОВ И СПЕЦИАЛЬНЫХ СИМВОЛОВ ==========

        # Декораторы методов
        if line in METHOD_DECORATORS:
            # Декораторы обрабатываются в parse_class_method_declaration
            return current_index + 1

        # C-вызовы (@func())
        if line.startswith("@"):
            parsed = self.parse_c_call(line, current_scope)
            return current_index + 1 if parsed else current_index + 1

        # ========== ОБРАБОТКА КЛЮЧЕВЫХ СЛОВ ==========

        for key in KEYS:
            if (
                line.startswith(key + " ")
                or line.startswith(key + ":")
                or line == key
            ):
                if key == "const":
                    parsed = self.parse_const(line, current_scope)
                    return current_index + 1
                elif key == "var":
                    parsed = self.parse_var(line, current_scope)
                    return current_index + 1
                elif key == "def":
                    # Проверяем, не является ли это методом класса
                    if current_scope.get("type") in [
                        "class_body",
                        "class_method",
                        "static_method",
                        "classmethod",
                    ]:
                        # Метод класса уже обрабатывается в parse_class_declaration
                        return current_index + 1
                    else:
                        return self.parse_function_declaration(
                            line, current_scope, all_lines, current_index
                        )
                elif key == "class":
                    return self.parse_class_declaration(
                        line, current_scope, all_lines, current_index
                    )
                elif key == "del":
                    parsed = self.parse_delete(line, current_scope)
                    return current_index + 1
                elif key == "return":
                    parsed = self.parse_return(line, current_scope)
                    return current_index + 1
                elif key == "while":
                    return self.parse_while_loop(
                        line, current_scope, all_lines, current_index, actual_indent
                    )
                elif key == "for":
                    return self.parse_for_loop(
                        line, current_scope, all_lines, current_index, actual_indent
                    )
                elif key == "if":
                    # Проверяем, не вложен ли if в другой блок
                    if current_scope.get("type") in [
                        "while_loop_body",
                        "for_loop_body",
                        "if_body",
                        "elif_body",
                        "else_body",
                    ]:
                        return self.parse_nested_if(
                            line, current_scope, all_lines, current_index, actual_indent
                        )
                    else:
                        return self.parse_if_statement(
                            line, current_scope, all_lines, current_index, actual_indent
                        )
                elif key == "break":
                    parsed = self.parse_break(line, current_scope)
                    return current_index + 1
                elif key == "continue":
                    parsed = self.parse_continue(line, current_scope)
                    return current_index + 1
                elif key == "unsafe":
                    return self.parse_unsafe_block(
                        line, current_scope, all_lines, current_index, actual_indent
                    )

        # ========== ОБРАБОТКА C ИМПОРТОВ ==========

        if line.startswith("cimport "):
            return self.parse_cimport(line, current_scope, all_lines, current_index)

        # ========== ОБРАБОТКА PASS ==========

        if line == "pass":
            current_scope["graph"].append(
                {"node": "pass", "content": "pass", "operations": [{"type": "PASS"}]}
            )
            return current_index + 1

        # ========== ОБРАБОТКА ВЫЗОВОВ МЕТОДОВ И ФУНКЦИЙ ==========

        # 1. Вызов метода объекта: obj.method(args)
        object_method_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\."
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        )
        object_method_match = re.match(object_method_pattern, line)

        # 2. Статический вызов метода: Class.method(args)
        static_method_pattern = (
            r"^([A-Z][a-zA-Z0-9_]*(?:\[[^\]]+\])?)\."
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        )
        static_method_match = re.match(static_method_pattern, line)

        # 3. Обычный вызов функции: func(args)
        function_call_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        function_call_match = re.match(function_call_pattern, line)

        # 4. Присваивание результата вызова: var x: type = func(args)
        func_assignment_pattern = r"^var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        func_assignment_match = re.match(func_assignment_pattern, line)

        # 5. Присваивание с созданием объекта: var x: Class = Class(args)
        obj_creation_pattern = r"^var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([A-Z][a-zA-Z0-9_]*)\s*=\s*([A-Z][a-zA-Z0-9_]*)\s*\((.*)\)$"
        obj_creation_match = re.match(obj_creation_pattern, line)

        # Приоритет проверок:
        # 1. Создание объекта с присваиванием
        if obj_creation_match:
            var_name, var_type, class_name, args_str = obj_creation_match.groups()
            if var_type == class_name:  # Проверяем соответствие типов
                parsed = self.parse_object_creation_assignment(
                    line, current_scope, var_name, class_name, args_str
                )
                return current_index + 1

        # 2. Присваивание результата вызова функции
        if func_assignment_match:
            var_name, var_type, func_name, args_str = func_assignment_match.groups()
            # Создаем выражение для вызова функции
            func_call_expr = f"{func_name}({args_str})"
            # Парсим как обычное присваивание с выражением
            modified_line = f"var {var_name}: {var_type} = {func_call_expr}"
            parsed = self.parse_var(modified_line, current_scope)
            return current_index + 1

        # 3. Вызов статического метода
        if static_method_match:
            class_type, method_name, args_str = static_method_match.groups()
            class_name = class_type.split("[", 1)[0]
            # Упрощенная проверка - начинается с заглавной буквы
            # A variable may legally start with an uppercase letter (for
            # example matrix variables A/B).  Prefer the object call when
            # the receiver is already present in the current symbol scope.
            receiver_symbol, _ = self.find_symbol_recursive(current_scope, class_name)
            receiver_is_variable = bool(
                receiver_symbol
                and receiver_symbol.get("key") not in {"class", "function"}
            )
            if class_name and class_name[0].isupper() and not receiver_is_variable:
                parsed = self.parse_static_method_call_node(
                    line,
                    current_scope,
                    class_name,
                    method_name,
                    args_str,
                    class_type=class_type,
                )
                return current_index + 1

        # 4. Вызов метода объекта
        if object_method_match:
            obj_name, method_name, args_str = object_method_match.groups()
            parsed = self.parse_object_method_call_node(
                line, current_scope, obj_name, method_name, args_str
            )
            return current_index + 1

        # 5. Обычный вызов функции
        if function_call_match:
            func_name, args_str = function_call_match.groups()
            # Проверяем, не является ли это вызовом конструктора без присваивания
            if func_name and func_name[0].isupper():
                # Это возможный вызов конструктора
                parsed = self.parse_constructor_call(
                    line, current_scope, func_name, args_str
                )
                return current_index + 1
            else:
                parsed = self.parse_function_call(line, current_scope)
                return current_index + 1

        # ========== ОБРАБОТКА ВСТРОЕННЫХ ФУНКЦИЙ ==========

        # Проверяем встроенные функции (print, len, str, int, bool, range)
        for func_name in self.builtin_functions:
            if line.startswith(f"{func_name}("):
                parsed = self.parse_builtin_function_call(
                    line, current_scope, func_name
                )
                return current_index + 1

        # ========== ОБРАБОТКА ПРИСВАИВАНИЙ ==========

        # Проверяем различные виды присваиваний

        # 1. Доступ к атрибуту с присваиванием: obj.attr = value
        attr_assignment_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$"
        )
        attr_assignment_match = re.match(attr_assignment_pattern, line)

        if attr_assignment_match:
            obj_name, attr_name, value = attr_assignment_match.groups()
            parsed = self.parse_attribute_assignment(
                line, current_scope, obj_name, attr_name, value
            )
            return current_index + 1

        # 2. Обычное присваивание: var = value
        simple_assignment_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$"
        simple_assignment_match = re.match(simple_assignment_pattern, line)

        if simple_assignment_match:
            var_name, value = simple_assignment_match.groups()

            # Проверяем, не является ли это разыменованием указателя (*p = value)
            if var_name.startswith("*"):
                parsed = self.parse_pointer_dereference_assignment(
                    line, current_scope, var_name, value
                )
                return current_index + 1

            # Проверяем, не является ли значение разыменованием (*p)
            if value.strip().startswith("*"):
                parsed = self.parse_pointer_to_variable_assignment(
                    line, current_scope, var_name, value
                )
                return current_index + 1

            # Обычное присваивание
            parsed = self.parse_assignment(line, current_scope)
            return current_index + 1

        # 3. Составные операции присваивания: var += value
        augmented_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(\+=|-=|\*=|/=|//=|\%=|\*\*=|>>=|<<=|&=|\|=|\^=)\s*(.+)$"
        augmented_match = re.match(augmented_pattern, line)

        if augmented_match:
            var_name, operator, value = augmented_match.groups()
            parsed = self.parse_augmented_assignment(line, current_scope)
            return current_index + 1

        # ========== ОБРАБОТКА ДОСТУПА К АТРИБУТАМ ==========

        # Доступ к атрибуту без вызова: obj.attr
        attr_access_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)$"
        attr_access_match = re.match(attr_access_pattern, line)

        if attr_access_match:
            # В изолированном виде это выражение не имеет смысла,
            # но может быть частью более сложного выражения
            # Парсим как выражение
            expression_ast = self.parse_expression_to_ast(line)
            current_scope["graph"].append(
                {
                    "node": "expression",
                    "content": line,
                    "expression_ast": expression_ast,
                    "operations": [
                        {"type": "EXPRESSION_EVAL", "expression": expression_ast}
                    ],
                    "dependencies": self.extract_dependencies_from_ast(expression_ast),
                }
            )
            return current_index + 1

        # ========== ОБРАБОТКА ВЫРАЖЕНИЙ ==========

        # Если ничего не распознано, пробуем парсить как выражение
        expression_ast = self.parse_expression_to_ast(line)
        if expression_ast["type"] not in ["unknown", "empty"]:
            current_scope["graph"].append(
                {
                    "node": "expression",
                    "content": line,
                    "expression_ast": expression_ast,
                    "operations": [
                        {"type": "EXPRESSION_EVAL", "expression": expression_ast}
                    ],
                    "dependencies": self.extract_dependencies_from_ast(expression_ast),
                }
            )
            return current_index + 1

        # ========== НЕРАСПОЗНАННАЯ СТРОКА ==========

        # Если строка не распознана, создаем узел с ошибкой
        logger.debug(f"Warning: Не удалось распарсить строку: {line}")
        current_scope["graph"].append(
            {
                "node": "unparsed",
                "content": line,
                "operations": [{"type": "UNPARSED", "content": line}],
                "dependencies": [],
            }
        )

        return current_index + 1

    def parse_unsafe_block(
        self, line: str, scope: dict, all_lines: list, current_index: int, indent: int
    ):
        """Parse an explicit unsafe region without changing runtime semantics.

        ``unsafe:`` is a lexical marker.  Its body remains in the surrounding
        function scope, while every emitted graph node is tagged so the
        validator and backend can enforce the same boundary independently.
        """
        if not re.match(r"unsafe\s*:\s*$", line):
            return current_index + 1

        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, indent)
        graph_start = len(scope.get("graph", []))
        self.unsafe_depth += 1
        try:
            i = body_start
            while i < body_end:
                if not all_lines[i].strip():
                    i += 1
                    continue
                body_indent = self.calculate_indent_level(all_lines[i])
                i = self.parse_line(
                    all_lines[i].strip(), scope, all_lines, i, body_indent
                )
        finally:
            self.unsafe_depth -= 1
            # Keep the surrounding function scope active for the next sibling
            # statement (for example ``return`` after ``unsafe:``).
            self.current_indent = indent

        new_nodes = scope.get("graph", [])[graph_start:]
        self._mark_unsafe_nodes(new_nodes)
        return body_end

    def parse_nested_index_assignment(
        self, line: str, scope: dict, var_name: str, indices_str: str, value: str
    ) -> bool:
        """Парсит многомерное присваивание по индексу: A_data[0][0] = 10"""
        # Извлекаем индексы из строки типа "[0][0]"
        indices = []
        current_index = ""
        depth = 0

        for char in indices_str:
            if char == "[":
                if depth == 0:
                    current_index = ""
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    indices.append(current_index)
            else:
                if depth > 0:
                    current_index += char

        if not indices:
            logger.debug(f"Error: Некорректное индексное выражение: {indices_str}")
            return False

        expanded_indices = []
        for index in indices:
            expanded_indices.extend(split_top_level(index) or [index])
        indices = expanded_indices

        # Парсим значение
        value_ast = self.parse_expression_to_ast(value)

        # Парсим каждый индекс
        index_asts = []
        for idx in indices:
            idx_ast = self.parse_expression_to_ast(idx)
            index_asts.append(idx_ast)

        # Проверяем существование переменной
        symbol = scope["symbol_table"].get_symbol(var_name)
        if not symbol:
            logger.debug(f"Error: Переменная '{var_name}' не определена")
            return False

        # Определяем тип переменной
        var_type = symbol.get("type", "")

        # Важное исправление: правильно определяем вложенность списка
        is_nested_list = "list[list" in var_type

        # Определяем глубину вложенности
        depth_level = len(indices)

        # Для одномерных списков используем другой тип операции
        if depth_level == 1:
            # Простое индексное присваивание: list[0] = value
            operations = [
                {
                    "type": "SIMPLE_INDEX_ASSIGN",  # Новый тип операции
                    "variable": var_name,
                    "index": index_asts[0],
                    "value": value_ast,
                    "var_type": var_type,
                }
            ]

            # Создаем узел для простого присваивания
            scope["graph"].append(
                {
                    "node": "index_assignment",
                    "content": line,
                    "variable": var_name,
                    "index": index_asts[0],
                    "value": value_ast,
                    "operations": operations,
                    "dependencies": [var_name]
                    + self.extract_dependencies_from_ast(index_asts[0])
                    + self.extract_dependencies_from_ast(value_ast),
                    "var_type": var_type,
                }
            )
        else:
            # Многомерное присваивание
            operations = [
                {
                    "type": "NESTED_INDEX_ASSIGN",
                    "variable": var_name,
                    "indices": index_asts,
                    "value": value_ast,
                    "depth": depth_level,
                    "is_nested_list": is_nested_list,
                }
            ]

            # Собираем зависимости
            dependencies = [var_name]
            for idx_ast in index_asts:
                deps = self.extract_dependencies_from_ast(idx_ast)
                dependencies.extend(deps)
            deps = self.extract_dependencies_from_ast(value_ast)
            dependencies.extend(deps)

            # Создаем узел
            scope["graph"].append(
                {
                    "node": "nested_index_assignment",
                    "content": line,
                    "variable": var_name,
                    "indices": index_asts,
                    "value": value_ast,
                    "operations": operations,
                    "dependencies": list(set(dependencies)),
                    "var_type": var_type,
                }
            )

        logger.debug(
            f"DEBUG: Добавлено индексное присваивание: {line} (глубина: {len(indices)})"
        )
        return True

    def parse_slice_assignment(
        self, line: str, scope: dict, var_name: str, start: int, stop: int, value: str
    ) -> bool:
        """Парсит присваивание срезу: my_list[1:3] = [20, 30]"""
        # Парсим значение
        value_ast = self.parse_expression_to_ast(value)

        # Проверяем существование переменной
        symbol = scope["symbol_table"].get_symbol(var_name)
        if not symbol:
            logger.debug(f"Error: Переменная '{var_name}' не определена")
            return False

        operations = [
            {
                "type": "SLICE_ASSIGN",
                "variable": var_name,
                "slice_start": start,
                "slice_stop": stop,
                "value": value_ast,
            }
        ]

        # Собираем зависимости
        dependencies = [var_name]
        deps = self.extract_dependencies_from_ast(value_ast)
        dependencies.extend(deps)

        scope["graph"].append(
            {
                "node": "slice_assignment",
                "content": line,
                "variable": var_name,
                "start": {"type": "literal", "value": start, "data_type": "int"}
                if start is not None
                else None,
                "stop": {"type": "literal", "value": stop, "data_type": "int"}
                if stop is not None
                else None,
                "value": value_ast,
                "operations": operations,
                "dependencies": list(set(dependencies)),
            }
        )

        return True

    def parse_object_creation_assignment(
        self, line: str, scope: dict, var_name: str, class_name: str, args_str: str
    ) -> bool:
        """Парсит создание объекта с присваиванием: var x: Class = Class(args)"""
        # Парсим аргументы
        args = []
        if args_str.strip():
            args = self.parse_function_arguments_to_ast(args_str)

        # Проверяем существование класса (упрощенная проверка)
        class_symbol = scope["symbol_table"].get_symbol(class_name)
        is_class = class_symbol and class_symbol.get("key") == "class"

        # Добавляем переменную в таблицу символов
        scope["symbol_table"].add_symbol(name=var_name, key="var", var_type=class_name)

        if var_name not in scope["local_variables"]:
            scope["local_variables"].append(var_name)

        # Создаем AST для вызова конструктора
        constructor_ast = {
            "type": "constructor_call",
            "class_name": class_name,
            "arguments": args,
        }

        # Создаем операции
        operations = [
            {"type": "NEW_VAR", "target": var_name, "var_type": class_name},
            {
                "type": "CONSTRUCTOR_CALL",
                "class_name": class_name,
                "target": var_name,
                "arguments": args,
            },
        ]

        # Собираем зависимости
        dependencies = []
        for arg in args:
            deps = self.extract_dependencies_from_ast(arg)
            dependencies.extend(deps)

        # Создаем узел
        scope["graph"].append(
            {
                "node": "object_creation",
                "content": line,
                "symbols": [var_name],
                "var_name": var_name,
                "var_type": class_name,
                "class_name": class_name,
                "arguments": args,
                "operations": operations,
                "dependencies": dependencies,
                "expression_ast": constructor_ast,
            }
        )

        return True

    def parse_constructor_call(
        self, line: str, scope: dict, class_name: str, args_str: str
    ) -> bool:
        """Парсит вызов конструктора без присваивания: Class(args)"""
        # Парсим аргументы
        args = []
        if args_str.strip():
            args = self.parse_function_arguments_to_ast(args_str)

        operations = [
            {"type": "CONSTRUCTOR_CALL", "class_name": class_name, "arguments": args}
        ]

        # Собираем зависимости
        dependencies = []
        for arg in args:
            deps = self.extract_dependencies_from_ast(arg)
            dependencies.extend(deps)

        scope["graph"].append(
            {
                "node": "constructor_call",
                "content": line,
                "class_name": class_name,
                "arguments": args,
                "operations": operations,
                "dependencies": dependencies,
            }
        )

        return True

    def parse_index_assignment(
        self, line: str, scope: dict, var_name: str, index_expr: str, value: str
    ) -> bool:
        """Парсит присваивание по индексу с поддержкой многомерных массивов"""
        # index_expr может быть "[0]" или "[0][0]" или "[0][1][2]" и т.д.
        # Убираем внешние скобки, если есть
        index_expr = index_expr.strip()

        # Извлекаем все индексы из выражения типа "[0][0]"
        indices = []
        current_index = ""
        depth = 0

        for char in index_expr:
            if char == "[":
                if depth == 0:
                    current_index = ""
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    indices.append(current_index)
            else:
                if depth > 0:
                    current_index += char

        # Парсим значение
        value_ast = self.parse_expression_to_ast(value)

        # Проверяем существование переменной
        symbol = scope["symbol_table"].get_symbol(var_name)
        if not symbol:
            logger.debug(f"Error: Переменная '{var_name}' не определена")
            return False

        # Определяем тип операции в зависимости от количества индексов
        operations = []

        if len(indices) == 1:
            # Одиночная индексация: a[0] = value
            index_ast = self.parse_expression_to_ast(indices[0])
            operations.append(
                {
                    "type": "INDEX_ASSIGN",
                    "variable": var_name,
                    "index": index_ast,
                    "value": value_ast,
                }
            )

            # Создаем узел
            scope["graph"].append(
                {
                    "node": "index_assignment",
                    "content": line,
                    "variable": var_name,
                    "index": index_ast,
                    "value": value_ast,
                    "operations": operations,
                    "dependencies": [var_name]
                    + self.extract_dependencies_from_ast(index_ast)
                    + self.extract_dependencies_from_ast(value_ast),
                }
            )

        elif len(indices) > 1:
            # Многомерная индексация: a[0][0] = value
            index_asts = []
            for idx in indices:
                idx_ast = self.parse_expression_to_ast(idx)
                index_asts.append(idx_ast)

            operations.append(
                {
                    "type": "NESTED_INDEX_ASSIGN",
                    "variable": var_name,
                    "indices": index_asts,
                    "value": value_ast,
                    "depth": len(indices),
                }
            )

            # Создаем узел для многомерного присваивания
            scope["graph"].append(
                {
                    "node": "nested_index_assignment",
                    "content": line,
                    "variable": var_name,
                    "indices": index_asts,
                    "value": value_ast,
                    "operations": operations,
                    "dependencies": [var_name]
                    + self.extract_dependencies_from_ast(value_ast),
                }
            )
        else:
            # Нет индексов - ошибка
            logger.debug(f"Error: Некорректное индексное выражение: {index_expr}")
            return False

        return True

    def parse_augmented_index_assignment(
        self,
        line: str,
        scope: dict,
        var_name: str,
        index: str,
        operator: str,
        value: str,
    ) -> bool:
        """Парсит составную операцию с индексом: my_list[0] += 5"""
        # Parse one or more indices. ``a[0, 1]`` is represented explicitly so
        # the backend can perform a single read-modify-write operation.
        index_asts = self.parse_function_arguments_to_ast(index)
        if not index_asts:
            logger.debug(f"Error: Некорректный индекс: '{index}'")
            return False

        # Парсим значение
        value_ast = self.parse_expression_to_ast(value)

        # Проверяем существование переменной
        symbol = scope["symbol_table"].get_symbol(var_name)
        if not symbol:
            logger.debug(f"Error: Переменная '{var_name}' не определена")
            return False

        # Определяем тип операции
        operator_map = {
            "+=": "ADD",
            "-=": "SUBTRACT",
            "*=": "MULTIPLY",
            "/=": "DIVIDE",
            "//=": "INTEGER_DIVIDE",
            "%=": "MODULO",
        }

        op_type = operator_map.get(operator, "UNKNOWN_AUGMENTED")

        operations = [
            {
                "type": "AUGMENTED_INDEX_ASSIGN",
                "variable": var_name,
                "index": index_asts[0],
                "indices": index_asts,
                "operator": op_type,
                "operator_symbol": operator,
                "value": value_ast,
            }
        ]

        # Собираем зависимости
        dependencies = [var_name]
        deps = self.extract_dependencies_from_ast(value_ast)
        dependencies.extend(deps)

        scope["graph"].append(
            {
                "node": "augmented_index_assignment",
                "content": line,
                "variable": var_name,
                "index": index_asts[0],
                "indices": index_asts,
                "operator": operator,
                "value": value_ast,
                "operations": operations,
                "dependencies": list(set(dependencies)),
            }
        )

        return True

    def parse_attribute_assignment(
        self, line: str, scope: dict, obj_name: str, attr_name: str, value: str
    ) -> bool:
        """Парсит присваивание атрибуту: obj.attr = value"""
        # Парсим значение
        value_ast = self.parse_expression_to_ast(value)

        # Проверяем существование объекта
        obj_symbol = scope["symbol_table"].get_symbol(obj_name)
        if not obj_symbol:
            logger.debug(f"Error: Объект '{obj_name}' не определен")
            return False

        operations = [
            {
                "type": "ATTRIBUTE_ASSIGN",
                "object": obj_name,
                "attribute": attr_name,
                "value": value_ast,
            }
        ]

        # Собираем зависимости
        dependencies = [obj_name]
        deps = self.extract_dependencies_from_ast(value_ast)
        dependencies.extend(deps)

        scope["graph"].append(
            {
                "node": "attribute_assignment",
                "content": line,
                "object": obj_name,
                "attribute": attr_name,
                "value": value_ast,
                "operations": operations,
                "dependencies": dependencies,
            }
        )

        return True

    def parse_pointer_dereference_assignment(
        self, line: str, scope: dict, pointer_expr: str, value: str
    ) -> bool:
        """Парсит присваивание через разыменование указателя: *p = value"""
        # Извлекаем имя указателя
        pointer_name = pointer_expr[1:].strip()

        # Парсим значение
        value_ast = self.parse_expression_to_ast(value)

        # Проверяем существование указателя
        pointer_symbol = scope["symbol_table"].get_symbol(pointer_name)
        if not pointer_symbol:
            logger.debug(f"Error: Указатель '{pointer_name}' не определен")
            return False

        if not pointer_symbol["type"].startswith("*"):
            logger.debug(f"Error: '{pointer_name}' не является указателем")
            return False

        operations = [
            {
                "type": "WRITE_POINTER",
                "pointer": pointer_name,
                "value": value_ast,
                "operation": "*=",
            }
        ]

        # Собираем зависимости
        dependencies = [pointer_name]
        deps = self.extract_dependencies_from_ast(value_ast)
        dependencies.extend(deps)

        scope["graph"].append(
            {
                "node": "dereference_write",
                "content": line,
                "pointer": pointer_name,
                "value": value_ast,
                "operations": operations,
                "dependencies": dependencies,
            }
        )

        return True

    def parse_pointer_to_variable_assignment(
        self, line: str, scope: dict, var_name: str, pointer_expr: str
    ) -> bool:
        """Парсит присваивание значения указателя переменной: x = *p"""
        # Извлекаем имя указателя
        pointer_name = pointer_expr[1:].strip()

        # Проверяем существование указателя
        pointer_symbol = scope["symbol_table"].get_symbol(pointer_name)
        if not pointer_symbol:
            logger.debug(f"Error: Указатель '{pointer_name}' не определен")
            return False

        if not pointer_symbol["type"].startswith("*"):
            logger.debug(f"Error: '{pointer_name}' не является указателем")
            return False

        # Проверяем существование переменной
        var_symbol = scope["symbol_table"].get_symbol(var_name)
        if not var_symbol:
            logger.debug(f"Error: Переменная '{var_name}' не определена")
            return False

        # Создаем AST для разыменования
        deref_ast = {"type": "dereference", "pointer": pointer_name}

        operations = [
            {
                "type": "READ_POINTER",
                "target": var_name,
                "from": pointer_name,
                "operation": "*",
                "value": deref_ast,
                "pointed_type": pointer_symbol["type"][1:],  # Убираем звездочку
            }
        ]

        # Обновляем значение переменной
        scope["symbol_table"].update_symbol(var_name, {"value": deref_ast})

        dependencies = [pointer_name]

        scope["graph"].append(
            {
                "node": "dereference_read",
                "content": line,
                "target": var_name,
                "pointer": pointer_name,
                "operations": operations,
                "dependencies": dependencies,
            }
        )

        return True

    def parse_builtin_function_call(self, line: str, scope: dict, func_name: str):
        """Парсит вызов встроенной функции"""
        pattern = rf"{func_name}\s*\((.*?)\)"
        match = re.match(pattern, line)

        if not match:
            return False

        args_str = match.group(1)
        args = self.parse_function_arguments(args_str)

        # Определяем тип возвращаемого значения
        return_type = self.get_builtin_return_type(func_name, args)

        # Создаем узел для встроенной функции
        operations = [
            {
                "type": "BUILTIN_FUNCTION_CALL",
                "function": func_name,
                "arguments": args,
                "return_type": return_type,
            }
        ]

        # Собираем зависимости
        dependencies = []
        for arg in args:
            if (
                arg
                and not arg.startswith('"')
                and not arg.endswith('"')
                and not arg.startswith("'")
                and not arg.endswith("'")
                and not arg.isdigit()
                and arg not in ["True", "False", "None"]
            ):
                # Извлекаем переменные из аргументов
                var_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)"
                vars_in_arg = re.findall(var_pattern, arg)
                for var in vars_in_arg:
                    if (
                        var not in KEYS
                        and var not in DATA_TYPES
                        and var not in dependencies
                    ):
                        dependencies.append(var)

        scope["graph"].append(
            {
                "node": "builtin_function_call",
                "content": line,
                "function": func_name,
                "arguments": args,
                "return_type": return_type,
                "operations": operations,
                "dependencies": dependencies,
            }
        )

        return True

    def parse_function_arguments(self, args_str: str) -> list:
        """Разбирает аргументы функции с учетом строк и вложенных вызовов"""
        if not args_str.strip():
            return []

        args = []
        current_arg = ""
        in_string = False
        string_char = None
        paren_depth = 0
        bracket_depth = 0

        for char in args_str:
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
                current_arg += char
            elif in_string and char == string_char and current_arg[-1] != "\\":
                in_string = False
                current_arg += char
            elif not in_string and char == "(":
                paren_depth += 1
                current_arg += char
            elif not in_string and char == ")":
                paren_depth -= 1
                current_arg += char
            elif not in_string and char == "[":
                bracket_depth += 1
                current_arg += char
            elif not in_string and char == "]":
                bracket_depth -= 1
                current_arg += char
            elif (
                not in_string
                and paren_depth == 0
                and bracket_depth == 0
                and char == ","
            ):
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char

        if current_arg.strip():
            args.append(current_arg.strip())

        return [arg.strip() for arg in args]

    def parse_tuple_literal(self, value: str) -> dict:
        """Парсит литерал кортежа"""
        value = value.strip()

        # Проверяем, что это действительно кортеж
        if not (value.startswith("(") and value.endswith(")")):
            return {"type": "unknown", "value": value}

        # Проверяем, что это не выражение в скобках
        inner = value[1:-1].strip()
        if "," not in inner:
            # Это выражение в скобках, а не кортеж
            inner_ast = self.parse_expression_to_ast(inner)
            return {
                "type": "tuple_literal",
                "items": [inner_ast],
                "length": 1,
                "is_immutable": True,
            }

        # Парсим элементы кортежа
        items = []
        current_item = ""
        depth = 0
        in_string = False
        string_char = None

        i = 0
        while i < len(inner):
            char = inner[i]

            # Обработка строк
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
                current_item += char
            elif in_string and char == string_char:
                # Проверяем экранирование
                if i > 0 and inner[i - 1] == "\\":
                    current_item += char
                else:
                    in_string = False
                    current_item += char
            # Обработка скобок
            elif not in_string and char == "(":
                depth += 1
                current_item += char
            elif not in_string and char == ")":
                depth -= 1
                current_item += char
            elif not in_string and char == "[":
                depth += 1
                current_item += char
            elif not in_string and char == "]":
                depth -= 1
                current_item += char
            elif not in_string and char == "{":
                depth += 1
                current_item += char
            elif not in_string and char == "}":
                depth -= 1
                current_item += char
            # Разделитель элементов
            elif not in_string and depth == 0 and char == ",":
                if current_item.strip():
                    item_ast = self.parse_expression_to_ast(current_item.strip())
                    items.append(item_ast)
                current_item = ""
            else:
                current_item += char

            i += 1

        # Последний элемент
        if current_item.strip():
            item_ast = self.parse_expression_to_ast(current_item.strip())
            items.append(item_ast)

        # Особый случай: кортеж из одного элемента должен иметь запятую
        if len(items) == 1 and not inner.endswith(","):
            logger.debug(
                f"Warning: кортеж из одного элемента должен иметь запятую: {value}"
            )

        return {
            "type": "tuple_literal",
            "items": items,
            "length": len(items),
            "is_immutable": True,
        }

    def parse_break(self, line: str, scope: dict):
        """Парсит оператор break"""
        scope["graph"].append(
            {
                "node": "break",
                "content": line,
                "operations": [{"type": "BREAK"}],
            }
        )
        return True

    def parse_continue(self, line: str, scope: dict):
        """Парсит оператор continue"""
        scope["graph"].append(
            {
                "node": "continue",
                "content": line,
                "operations": [{"type": "CONTINUE"}],
            }
        )
        return True

    def parse_function_declaration(
        self, line: str, parent_scope: dict, all_lines: list, current_index: int
    ):
        """Parse a free function with fully nested type annotations."""
        pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*:"
        match = re.match(pattern, line)
        if not match:
            return current_index + 1

        func_name, params_str, return_type_text = match.groups()
        return_type, return_type_info = self.parse_type_annotation(
            return_type_text or "None"
        )
        parameters = self.parse_parameters(params_str)

        logger.debug(f"DEBUG: Найдена функция {func_name}() -> {return_type}")

        parent_level = parent_scope["level"]
        func_level = parent_level + 1
        indent_level = (
            self.calculate_indent_level(all_lines[current_index])
            if current_index < len(all_lines)
            else 0
        )

        symbol_id = parent_scope["symbol_table"].add_symbol(
            name=func_name,
            key="function",
            var_type="function",
            value=None,
            parameters=parameters,
            return_type=return_type,
            return_type_info=return_type_info,
        )

        func_decl_node = {
            "node": "function_declaration",
            "content": line,
            "function_name": func_name,
            "symbol_id": symbol_id,
            "parameters": parameters,
            "return_type": return_type,
            "return_type_info": return_type_info,
            "body_level": func_level,
            "is_stub": False,
        }
        parent_scope["graph"].append(func_decl_node)

        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, indent_level)
        is_stub = (
            body_start < len(all_lines)
            and all_lines[body_start].strip() == "pass"
            and self.calculate_indent_level(all_lines[body_start]) == indent_level + 1
        )

        func_scope = {
            "level": func_level,
            "type": "function",
            "parent_scope": parent_scope["level"],
            "function_name": func_name,
            "parameters": parameters,
            "return_type": return_type,
            "return_type_info": return_type_info,
            "local_variables": [],
            "graph": [],
            "symbol_table": SymbolTable(),
            "return_info": {
                "has_return": False,
                "return_value": None,
                "return_type": return_type,
                "return_type_info": return_type_info,
            },
            "is_stub": is_stub,
        }

        for param in parameters:
            func_scope["symbol_table"].add_symbol(
                name=param["name"],
                key="parameter",
                var_type=param["type"],
                type_info=param.get("type_info"),
                memory_kind=param.get("memory_kind"),
            )
            func_scope["local_variables"].append(param["name"])

        if is_stub:
            pass_node = {
                "node": "pass",
                "content": "pass",
                "operations": [{"type": "PASS"}],
            }
            func_scope["graph"].append(pass_node)
            func_decl_node["is_stub"] = True
            func_decl_node["body"] = []
            self.scopes.append(func_scope)
            return body_start + 1

        self.scopes.append(func_scope)
        self.scope_stack.append(func_scope)
        saved_indent = self.current_indent
        self.current_indent = indent_level + 1

        i = body_start
        while i < body_end:
            body_line = all_lines[i]
            if not body_line.strip():
                i += 1
                continue
            body_indent = self.calculate_indent_level(body_line)
            i = self.parse_line(body_line.strip(), func_scope, all_lines, i, body_indent)

        self.current_indent = saved_indent
        if self.scope_stack and self.scope_stack[-1] is func_scope:
            self.scope_stack.pop()
        elif func_scope in self.scope_stack:
            self.scope_stack.remove(func_scope)

        return body_end

    def parse_const(self, line: str, scope: dict):
        match = re.match(r"const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", line)
        if not match:
            return False
        name, rest = match.groups()
        equals = find_top_level(rest, "=")
        if equals < 0:
            return False
        type_text = rest[:equals].strip()
        value_text = rest[equals + 1 :].strip()
        const_type, type_info = self.parse_type_annotation(type_text)
        value = self.clean_value(value_text)

        symbol_id = scope["symbol_table"].add_symbol(
            name=name,
            key="const",
            var_type=const_type,
            value=value,
            is_constant=True,
            type_info=type_info,
            memory_kind=type_info.get("memory_kind"),
        )
        if name not in scope["local_variables"]:
            scope["local_variables"].append(name)
        scope["graph"].append(
            {
                "node": "declaration",
                "content": line,
                "symbols": [name],
                "var_name": name,
                "var_type": const_type,
                "type_info": type_info,
                "memory_kind": type_info.get("memory_kind"),
                "operations": [
                    {"type": "NEW_CONST", "target": name, "const_type": const_type},
                    {"type": "ASSIGN", "target": name, "value": value},
                ],
                "expression_ast": value,
            }
        )
        return True

    def parse_var(self, line: str, scope: dict):
        """Parse a typed variable declaration.

        Supported memory-oriented forms:

        * ``list[T]`` / ``dict[K,V]`` / ``tuple[T]`` -- shared ARC containers;
        * ``&T`` -- immutable lexical borrow;
        * ``&mut T`` -- exclusive mutable lexical borrow;
        * ``*T`` -- raw pointer (unsafe boundary);
        * ``array[T]`` -- uniquely owned contiguous buffer;
        * ``shared[T]`` -- explicit thread-shareable/shared wrapper;
        * ``T?`` -- nullable/optional type.
        """
        match = re.match(r"var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", line)
        if not match:
            return False

        name, type_and_value = match.groups()
        equals_pos = find_top_level(type_and_value, "=")
        if equals_pos < 0:
            return False

        raw_type = type_and_value[:equals_pos].strip()
        value_str = type_and_value[equals_pos + 1 :].strip()
        var_type, type_info = self.parse_type_annotation(raw_type)

        is_pointer = type_info.get("kind") == "raw_pointer"
        is_borrow = type_info.get("kind") in {"borrow", "mut_borrow"}
        is_mut_borrow = type_info.get("kind") == "mut_borrow"
        memory_kind = type_info.get("memory_kind", "value")

        # In raw-pointer declarations ``&x`` retains its historical C address-of
        # meaning.  Everywhere else, borrowing is expressed by the declared &T
        # / &mut T type and the initializer remains a normal source expression.
        if is_pointer and value_str.startswith("&") and not value_str.startswith("&mut "):
            target_text = value_str[1:].strip()
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", target_text):
                value_ast = {"type": "address_of", "variable": target_text}
            else:
                value_ast = {
                    "type": "address_of",
                    "expression": self.parse_expression_to_ast(target_text),
                }
        else:
            value_ast = self.parse_expression_to_ast(value_str)

        existing_symbol = scope["symbol_table"].get_symbol_for_validation(name)
        was_deleted = False
        if existing_symbol:
            was_deleted = existing_symbol.get("is_deleted", False)
            if not was_deleted:
                logger.debug(f"Error: переменная '{name}' уже объявлена")
                return False
            scope["symbol_table"].update_symbol(
                name,
                {
                    "type": var_type,
                    "type_info": type_info,
                    "memory_kind": memory_kind,
                    "value": value_ast,
                    "is_deleted": False,
                },
            )
            if hasattr(scope["symbol_table"], "deleted_symbols"):
                scope["symbol_table"].deleted_symbols.discard(name)
        else:
            scope["symbol_table"].add_symbol(
                name=name,
                key="var",
                var_type=var_type,
                value=value_ast,
                type_info=type_info,
                memory_kind=memory_kind,
                is_pointer=is_pointer,
                is_borrow=is_borrow,
                is_mut_borrow=is_mut_borrow,
            )

        if name not in scope["local_variables"]:
            scope["local_variables"].append(name)

        creation_op_type = "RESTORE_VAR" if was_deleted else "NEW_VAR"
        node_type = "redeclaration" if was_deleted else "declaration"
        operations = [
            {
                "type": creation_op_type,
                "target": name,
                "var_type": var_type,
                "type_info": type_info,
                "memory_kind": memory_kind,
                "was_deleted": was_deleted,
            }
        ]

        base_name = type_info.get("base_name", "")
        generic_name = type_info.get("name") if type_info.get("kind") == "generic" else ""

        if is_borrow:
            source = self._borrow_source_from_ast(value_ast)
            if source is None:
                logger.debug(
                    f"Error: borrow '{name}' должен ссылаться на именованную переменную"
                )
                return False
            operations.append(
                {
                    "type": "BORROW_MUT" if is_mut_borrow else "BORROW_IMMUT",
                    "target": name,
                    "source": source,
                    "value": value_ast,
                    "borrow_type": var_type,
                }
            )

        elif is_pointer:
            if value_ast.get("type") == "address_of":
                operations.append(
                    {
                        "type": "GET_ADDRESS",
                        "target": name,
                        "of": value_ast.get("variable"),
                        "operation": "&",
                    }
                )
            elif value_ast.get("type") == "literal" and value_ast.get("value") is None:
                operations.append({"type": "ASSIGN_NULL", "target": name, "is_null": True})
            else:
                operations.append({"type": "ASSIGN_POINTER", "target": name, "value": value_ast})

        elif generic_name == "array":
            if value_ast.get("type") == "list_literal":
                element_info = type_info.get("arguments", [{}])[0]
                operations.append(
                    {
                        "type": "CREATE_ARRAY",
                        "target": name,
                        "items": value_ast.get("items", []),
                        "size": len(value_ast.get("items", [])),
                        "element_type": element_info.get("canonical", "any"),
                        "element_type_info": element_info,
                        "contiguous": True,
                        "ownership": "unique",
                    }
                )
            else:
                operations.append({"type": "INITIALIZE", "target": name, "value": value_ast})

        elif generic_name == "tensor":
            if value_ast.get("type") == "list_literal":
                shape = infer_literal_shape(value_ast)
                element_info = type_info.get("arguments", [{}])[0]
                operations.append(
                    {
                        "type": "CREATE_TENSOR",
                        "target": name,
                        "items": value_ast.get("items", []),
                        "shape": shape,
                        "rank": len(shape) if shape is not None else None,
                        "is_rectangular": shape is not None,
                        "element_type": element_info.get("canonical", "any"),
                        "element_type_info": element_info,
                        "ownership": "unique",
                    }
                )
            else:
                operations.append({"type": "INITIALIZE", "target": name, "value": value_ast})

        elif generic_name == "shared":
            operations.append(
                {
                    "type": "SHARE_REFERENCE",
                    "target": name,
                    "value": value_ast,
                    "shared_type": var_type,
                }
            )

        elif var_type.startswith("tuple["):
            inner_args = type_info.get("arguments", [])
            if value_ast.get("type") == "tuple_literal":
                items = value_ast.get("items", [])
                operations.append(
                    {
                        "type": "CREATE_TUPLE_UNIFORM" if len(inner_args) == 1 else "CREATE_TUPLE_FIXED",
                        "target": name,
                        "items": items,
                        "size": len(items),
                        "element_type": inner_args[0].get("canonical") if len(inner_args) == 1 else None,
                        "element_types": [arg.get("canonical") for arg in inner_args] if len(inner_args) > 1 else None,
                        "is_immutable": True,
                        "is_uniform": len(inner_args) == 1,
                    }
                )
            else:
                operations.append({"type": "INITIALIZE", "target": name, "value": value_ast})

        elif var_type.startswith("list["):
            element_info = type_info.get("arguments", [{}])[0]
            if value_ast.get("type") == "list_literal":
                items = value_ast.get("items", [])
                operations.append(
                    {
                        "type": "CREATE_LIST",
                        "target": name,
                        "items": items,
                        "size": len(items),
                        "element_type": element_info.get("canonical", "any"),
                        "element_type_info": element_info,
                        "is_pointer_array": True,
                        "is_nested": any(item.get("type") == "list_literal" for item in items),
                    }
                )
            else:
                operations.append({"type": "INITIALIZE", "target": name, "value": value_ast})

        elif var_type.startswith("dict["):
            if value_ast.get("type") == "dict_literal":
                operations.append(
                    {
                        "type": "CREATE_DICT",
                        "target": name,
                        "pairs": value_ast.get("pairs", {}),
                        "size": len(value_ast.get("pairs", {})),
                    }
                )
            else:
                operations.append({"type": "INITIALIZE", "target": name, "value": value_ast})

        elif var_type in {"list", "dict", "set"}:
            if value_ast.get("type") == "list_literal":
                operations.append(
                    {
                        "type": "CREATE_LIST",
                        "target": name,
                        "items": value_ast.get("items", []),
                        "size": len(value_ast.get("items", [])),
                        "element_type": "any",
                    }
                )
            elif value_ast.get("type") == "dict_literal":
                operations.append(
                    {
                        "type": "CREATE_DICT",
                        "target": name,
                        "pairs": value_ast.get("pairs", {}),
                        "size": len(value_ast.get("pairs", {})),
                    }
                )
            elif value_ast.get("type") == "set_literal":
                operations.append(
                    {
                        "type": "CREATE_SET",
                        "target": name,
                        "items": value_ast.get("items", []),
                        "size": len(value_ast.get("items", [])),
                    }
                )
            else:
                operations.append({"type": "INITIALIZE", "target": name, "value": value_ast})
        else:
            operations.append({"type": "ASSIGN", "target": name, "value": value_ast})

        dependencies = self.extract_dependencies_from_ast(value_ast) if value_ast else []
        if is_borrow:
            source = self._borrow_source_from_ast(value_ast)
            if source and source not in dependencies:
                dependencies.append(source)

        data_structure = None
        if generic_name in {"list", "dict", "tuple", "array", "tensor", "shared"}:
            data_structure = generic_name
        elif is_pointer:
            data_structure = "pointer"
        elif is_borrow:
            data_structure = "borrow"

        scope["graph"].append(
            {
                "node": node_type,
                "content": line,
                "symbols": [name],
                "var_name": name,
                "var_type": var_type,
                "type_info": type_info,
                "memory_kind": memory_kind,
                "is_pointer": is_pointer,
                "is_borrow": is_borrow,
                "is_mut_borrow": is_mut_borrow,
                "operations": operations,
                "dependencies": list(dict.fromkeys(dependencies)),
                "expression_ast": value_ast,
                "data_structure": data_structure,
            }
        )
        return True

    def parse_delete(self, line: str, scope: dict):
        """Парсит оператор del (полное удаление)"""
        pattern = r"del\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        match = re.match(pattern, line)

        if not match:
            return False

        name = match.group(1)

        symbol = scope["symbol_table"].get_symbol(name)
        if not symbol:
            return False  # Переменная не существует или уже удалена

        deleted = scope["symbol_table"].delete_symbol(name)

        if deleted:
            # Добавляем флаг, что это полное удаление
            scope["graph"].append(
                {
                    "node": "delete",
                    "content": line,
                    "symbols": [name],
                    "operations": [
                        {
                            "type": "DELETE_FULL",
                            "target": name,
                        }  # Изменено с DELETE на DELETE_FULL
                    ],
                    "is_full_delete": True,  # Добавляем флаг
                }
            )

        return deleted

    def parse_return(self, line: str, scope: dict):
        """Парсит оператор return"""
        pattern = r"return\s+(.+)"
        match = re.match(pattern, line)

        if not match:
            return False

        expression = match.group(1).strip()

        dependencies = []
        var_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)"
        vars_in_value = re.findall(var_pattern, expression)
        for var in vars_in_value:
            if var not in KEYS and var not in DATA_TYPES:
                dependencies.append(var)

        # Парсим выражение
        expression_ast = self.parse_expression_to_ast(expression)

        scope["graph"].append(
            {
                "node": "return",
                "content": line,
                "symbols": [expression] if expression.isalpha() else [],
                "operations": [
                    {
                        "type": "RETURN",
                        "value": expression_ast,  # Используем AST вместо строки
                        "expression": expression,
                    }
                ],
                "dependencies": dependencies,
            }
        )

        if "return_info" in scope:
            scope["return_info"]["has_return"] = True
            scope["return_info"]["return_value"] = expression_ast

        return True

    def parse_expression_to_ast(self, expression: str) -> dict:
        """Parse an expression into the transitional Phils AST."""
        expression = expression.strip()
        if not expression:
            return {"type": "empty", "value": ""}

        # Explicit borrow expression.  Most declarations can simply use
        # ``var view: &T = owner``; this form is also accepted in call arguments.
        if expression.startswith("&mut "):
            inner = expression[5:].strip()
            return {
                "type": "borrow",
                "mutable": True,
                "source": self.parse_expression_to_ast(inner),
            }

        # ``&identifier`` outside raw-pointer declaration context is an immutable
        # borrow expression. Raw pointers are disambiguated in parse_var().
        if expression.startswith("&"):
            inner = expression[1:].strip()
            if inner and not inner.startswith("&"):
                return {
                    "type": "borrow",
                    "mutable": False,
                    "source": self.parse_expression_to_ast(inner),
                }

        if (expression.startswith('"') and expression.endswith('"')) or (
            expression.startswith("'") and expression.endswith("'")
        ):
            content = expression[1:-1]
            content = content.replace('\\"', '"').replace("\\'", "'")
            return {"type": "literal", "value": content, "data_type": "str"}

        if re.match(r"^-?\d+$", expression):
            return {"type": "literal", "value": int(expression), "data_type": "int"}

        if (
            re.match(r"^-?\d+\.\d+$", expression)
            or re.match(r"^-?\d+\.\d+[eE][+-]?\d+$", expression)
            or re.match(r"^-?\d+[eE][+-]?\d+$", expression)
        ):
            try:
                return {"type": "literal", "value": float(expression), "data_type": "float"}
            except ValueError:
                pass

        if expression == "True":
            return {"type": "literal", "value": True, "data_type": "bool"}
        if expression == "False":
            return {"type": "literal", "value": False, "data_type": "bool"}
        if expression == "None":
            return {"type": "literal", "value": None, "data_type": "None"}
        if expression == "null":
            return {"type": "literal", "value": "null", "data_type": "null"}

        if expression.startswith("[") and expression.endswith("]"):
            return self.parse_list_literal(expression)

        if expression.startswith("(") and expression.endswith(")"):
            inner = expression[1:-1].strip()
            if "," in inner or (inner and inner.endswith(",")):
                return self.parse_tuple_literal(expression)
            return self.parse_expression_to_ast(inner)

        if expression.startswith("{") and expression.endswith("}"):
            content = expression[1:-1].strip()
            if self.is_dict_literal(content):
                return self.parse_dict_literal(expression)
            return self.parse_set_literal(expression)

        return self._parse_with_priorities(expression)

    def _parse_with_priorities(self, expression: str) -> dict:
        """Парсит выражение с учетом приоритетов операторов Python"""
        expression = expression.strip()

        # Уровни приоритетов (от низшего к высшему)
        # Каждый уровень проверяется отдельно

        # Уровень 1: Логическое OR (самый низкий приоритет)
        result = self._parse_operator_level(expression, ["or"], "LOGICAL_OR")
        if result:
            return result

        # Уровень 2: Логическое AND
        result = self._parse_operator_level(expression, ["and"], "LOGICAL_AND")
        if result:
            return result

        # Уровень 3: Сравнения (is, is not, in, not in)
        result = self._parse_operator_level(
            expression,
            ["is not", "is", "not in", "in"],
            {"is not": "IS_NOT", "is": "IS", "not in": "NOT_IN", "in": "IN"},
        )
        if result:
            return result

        # Уровень 4: Сравнения (==, !=, >, <, >=, <=)
        result = self._parse_operator_level(
            expression,
            ["==", "!=", ">=", "<=", ">", "<"],
            {
                "==": "EQUAL",
                "!=": "NOT_EQUAL",
                ">=": "GREATER_EQUAL",
                "<=": "LESS_EQUAL",
                ">": "GREATER_THAN",
                "<": "LESS_THAN",
            },
        )
        if result:
            return result

        # Уровень 5: Битовая OR
        result = self._parse_operator_level(expression, ["|"], "BITWISE_OR")
        if result:
            return result

        # Уровень 6: Битовая XOR
        result = self._parse_operator_level(expression, ["^"], "BITWISE_XOR")
        if result:
            return result

        # Уровень 7: Битовая AND
        result = self._parse_operator_level(expression, ["&"], "BITWISE_AND")
        if result:
            return result

        # Уровень 8: Сдвиги
        result = self._parse_operator_level(
            expression, ["<<", ">>"], {"<<": "LEFT_SHIFT", ">>": "RIGHT_SHIFT"}
        )
        if result:
            return result

        # Уровень 9: Сложение/вычитание
        result = self._parse_operator_level(
            expression, ["+", "-"], {"+": "ADD", "-": "SUBTRACT"}
        )
        if result:
            return result

        # Уровень 10: Умножение/деление/остаток
        result = self._parse_operator_level(
            expression,
            ["*", "/", "//", "%"],
            {"*": "MULTIPLY", "/": "DIVIDE", "//": "INTEGER_DIVIDE", "%": "MODULO"},
        )
        if result:
            return result

        # Уровень 11: Возведение в степень (самый высокий приоритет)
        result = self._parse_operator_level(expression, ["**"], "POWER")
        if result:
            return result

        # Уровень 12: Унарные операторы
        # Унарные операторы обрабатываем справа налево
        result = self._parse_unary_operators(expression)
        if result:
            return result

        # ========== 4. ВЫЗОВЫ ФУНКЦИЙ И МЕТОДОВ ==========

        # 4.1 Выражения типа self.data[index]
        complex_attr_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\[(.+)\]$"
        )
        complex_attr_match = re.match(complex_attr_pattern, expression)
        if complex_attr_match:
            obj_name, attr_name, index_expr = complex_attr_match.groups()
            index_parts = split_top_level(index_expr) or [index_expr]
            index_asts = [self.parse_expression_to_ast(part) for part in index_parts]
            return {
                "type": "complex_attribute_access",
                "object": obj_name,
                "attribute": attr_name,
                "index": index_asts[0],
                "indices": index_asts,
            }

        # 4.2 Проверяем индексацию
        if "[" in expression and expression.endswith("]"):
            return self._parse_chained_index_access(expression)

        # 4.3 Вызов метода объекта: obj.method(args)
        static_method_pattern = (
            r"^([A-Z][a-zA-Z0-9_]*(?:\[[^\]]+\])?)\."
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        )
        static_method_match = re.match(static_method_pattern, expression)
        if static_method_match:
            class_type, method_name, args_str = static_method_match.groups()
            class_name = class_type.split("[", 1)[0]
            known_class = class_name == "Tensor"
            if not known_class:
                for parser_scope in self.scopes:
                    table = parser_scope.get("symbol_table")
                    symbol = table.get_symbol(class_name) if table else None
                    if symbol and symbol.get("key") == "class":
                        known_class = True
                        break
            # Uppercase variable names (A, B, C are common matrix names) are
            # object receivers, not static classes. Only registered classes
            # and the imported Tensor facade use this AST shape.
            if not known_class:
                static_method_match = None
            else:
                args = (
                    self.parse_function_arguments_to_ast(args_str)
                    if args_str.strip()
                    else []
                )
                return {
                    "type": "static_method_call",
                    "class_name": class_name,
                    "class_type": class_type,
                    "method": method_name,
                    "arguments": args,
                }

        obj_method_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\."
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        )
        obj_method_match = re.match(obj_method_pattern, expression)
        if obj_method_match:
            obj_name, method_name, args_str = obj_method_match.groups()
            args = (
                self.parse_function_arguments_to_ast(args_str)
                if args_str.strip()
                else []
            )
            return {
                "type": "method_call",
                "object": obj_name,
                "method": method_name,
                "arguments": args,
                "is_standalone": False,
            }

        # 4.4 Вызов конструктора: ClassName(args)
        constructor_pattern = r"^([A-Z][a-zA-Z0-9_]*)\s*\((.*)\)$"
        constructor_match = re.match(constructor_pattern, expression)
        if constructor_match:
            class_name, args_str = constructor_match.groups()
            args = (
                self.parse_function_arguments_to_ast(args_str)
                if args_str.strip()
                else []
            )
            return {
                "type": "constructor_call",
                "class_name": class_name,
                "arguments": args,
            }

        # 4.5 Вызов обычной функции: func(args)
        func_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        func_match = re.match(func_pattern, expression)
        if func_match:
            func_name, args_str = func_match.groups()
            args = (
                self.parse_function_arguments_to_ast(args_str)
                if args_str.strip()
                else []
            )
            return {"type": "function_call", "function": func_name, "arguments": args}

        # 4.6 Доступ к атрибуту: obj.attr
        simple_attr_pattern = (
            r"^([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\."
            r"([a-zA-Z_][a-zA-Z0-9_]*)$"
        )
        simple_attr_match = re.match(simple_attr_pattern, expression)
        if simple_attr_match:
            obj_name, attr_name = simple_attr_match.groups()
            return {
                "type": "attribute_access",
                "object": obj_name,
                "attribute": attr_name,
            }

        # ========== 5. ОПЕРАЦИИ С УКАЗАТЕЛЯМИ ==========

        # 5.1 Адрес переменной: &var
        if expression.startswith("&"):
            rest = expression[1:].strip()
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", rest):
                return {"type": "address_of", "variable": rest}
            else:
                inner_ast = self.parse_expression_to_ast(rest)
                return {"type": "address_of", "expression": inner_ast}

        # 5.2 Разыменование указателя: *ptr
        if expression.startswith("*"):
            rest = expression[1:].strip()
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", rest):
                return {"type": "dereference", "pointer": rest}
            else:
                inner_ast = self.parse_expression_to_ast(rest)
                return {"type": "dereference", "expression": inner_ast}

        # ========== 6. ВЫРАЖЕНИЯ В СКОБКАХ ==========

        # Если выражение полностью в скобках, убираем их и парсим заново
        if self.is_fully_parenthesized(expression):
            inner = expression[1:-1].strip()
            return self.parse_expression_to_ast(inner)

        # ========== 7. ПЕРЕМЕННЫЕ ==========

        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", expression):
            return {"type": "variable", "name": expression, "value": expression}

        # ========== 8. НЕРАСПОЗНАННОЕ ВЫРАЖЕНИЕ ==========

        return {"type": "unknown", "value": expression, "original": expression}

    def _parse_operator_level(self, expression: str, operators, operator_types) -> dict:
        """Парсит выражение на текущем уровне приоритета операторов"""
        if isinstance(operators, str):
            operators = [operators]

        if isinstance(operator_types, str):
            # Все операторы на этом уровне имеют один тип
            op_type = operator_types
            operator_types = {op: op_type for op in operators}

        # Ищем операторы справа налево (левая ассоциативность)
        for i in range(len(expression) - 1, -1, -1):
            # Проверяем все операторы на этой позиции
            for op in operators:
                op_len = len(op)
                if i >= op_len - 1 and expression[i - op_len + 1 : i + 1] == op:
                    # Проверяем, что это действительно оператор, а не часть чего-то
                    if self._is_valid_operator_at(expression, op, i - op_len + 1):
                        left = expression[: i - op_len + 1].strip()
                        right = expression[i + 1 :].strip()

                        if left and right:
                            # Для операторов сравнения с = проверяем правую часть
                            if op in [">=", "<=", "==", "!="] and right.startswith("="):
                                right = right[1:].strip()

                            return {
                                "type": "binary_operation",
                                "operator": operator_types[op],
                                "operator_symbol": op,
                                "left": self.parse_expression_to_ast(left),
                                "right": self.parse_expression_to_ast(right),
                            }

        return None

    def _parse_unary_operators(self, expression: str) -> dict:
        """Парсит унарные операторы"""
        expression = expression.strip()

        if not expression:
            return None

        # Унарное not
        if expression.startswith("not "):
            operand = expression[4:].strip()
            return {
                "type": "unary_operation",
                "operator": "NOT",
                "operator_symbol": "not",
                "operand": self.parse_expression_to_ast(operand),
            }

        # Унарное ~
        if expression.startswith("~"):
            operand = expression[1:].strip()
            return {
                "type": "unary_operation",
                "operator": "BITWISE_NOT",
                "operator_symbol": "~",
                "operand": self.parse_expression_to_ast(operand),
            }

        # Унарные + и - (только если они в начале выражения)
        if expression.startswith("+"):
            # Проверяем, что это действительно унарный плюс
            # (не часть ++ или что-то подобное)
            if len(expression) > 1 and not expression[1].isdigit():
                operand = expression[1:].strip()
                return {
                    "type": "unary_operation",
                    "operator": "POSITIVE",
                    "operator_symbol": "+",
                    "operand": self.parse_expression_to_ast(operand),
                }

        if expression.startswith("-"):
            # Проверяем, что это действительно унарный минус
            if len(expression) > 1 and not expression[1].isdigit():
                operand = expression[1:].strip()
                return {
                    "type": "unary_operation",
                    "operator": "NEGATIVE",
                    "operator_symbol": "-",
                    "operand": self.parse_expression_to_ast(operand),
                }

        return None

    def _is_valid_operator_at(self, expression: str, operator: str, pos: int) -> bool:
        """Проверяет, что оператор в данной позиции является валидным оператором"""
        # Проверяем границы
        if pos < 0 or pos + len(operator) > len(expression):
            return False

        # Проверяем совпадение
        if expression[pos : pos + len(operator)] != operator:
            return False

        # Проверяем, что оператор не часть другого оператора или идентификатора

        # Проверяем символ перед оператором
        if pos > 0:
            before = expression[pos - 1]
            # Оператор не должен следовать за буквой/цифрой/_ (часть идентификатора)
            if before.isalnum() or before == "_":
                return False

            # Специальные проверки для операторов с =
            if operator in [">", "<", "!", "="]:
                # Проверяем, не является ли это частью составного оператора
                if before in [">", "<", "!", "="] and operator == "=":
                    return False

        # Проверяем символ после оператора
        if pos + len(operator) < len(expression):
            after = expression[pos + len(operator)]
            # Оператор не должен предшествовать букве/цифре/_ (часть идентификатора)
            if after.isalnum() or after == "_":
                return False

            # Специальные проверки для операторов
            if operator in [">", "<"] and after == "=":
                # Это часть оператора >= или <=
                return False
            if operator == "!" and after == "=":
                # Это часть оператора !=
                return False
            if operator == "=" and after == "=":
                # Это часть оператора ==
                return False

        # Проверяем, что оператор не внутри скобок, строк и т.д.
        # Используем упрощенную проверку
        return self.find_operator_outside_parentheses(expression, operator) == pos

    def _parse_chained_index_access(self, expression: str) -> dict:
        """Парсит цепочки индексации типа a[0][1][2]"""
        # Основной паттерн для захвата всей индексации
        pattern = r"^([^\[\]]+)((?:\[[^\]]*\])+)$"
        match = re.match(pattern, expression)

        if not match:
            # Не соответствует паттерну, возвращаем как неизвестное
            return {"type": "unknown", "value": expression, "original": expression}

        base_name, indices_str = match.groups()

        # Извлекаем все индексы
        indices = []
        current_index = ""
        depth = 0

        for char in indices_str:
            if char == "[":
                if depth == 0:
                    current_index = ""
                depth += 1
                if depth > 1:
                    current_index += char
            elif char == "]":
                depth -= 1
                if depth == 0:
                    indices.append(current_index)
                elif depth > 0:
                    current_index += char
            else:
                current_index += char

        if not indices:
            return {"type": "unknown", "value": expression, "original": expression}

        # Если только один индекс, парсим как обычную индексацию
        if len(indices) == 1:
            index_expr = indices[0]

            # Проверяем, является ли это срезом
            if ":" in index_expr:
                # Это срез
                slice_parts = index_expr.split(":")
                if len(slice_parts) == 2:
                    start, stop = slice_parts
                    step = None
                elif len(slice_parts) == 3:
                    start, stop, step = slice_parts
                else:
                    # Некорректный срез
                    index_ast = self.parse_expression_to_ast(index_expr)
                    return {
                        "type": "index_access",
                        "variable": base_name,
                        "index": index_ast,
                    }

                # Парсим части среза
                start_ast = (
                    self.parse_expression_to_ast(start.strip())
                    if start.strip()
                    else None
                )
                stop_ast = (
                    self.parse_expression_to_ast(stop.strip()) if stop.strip() else None
                )
                step_ast = (
                    self.parse_expression_to_ast(step.strip())
                    if step and step.strip()
                    else None
                )

                return {
                    "type": "slice_access",
                    "variable": base_name,
                    "start": start_ast,
                    "stop": stop_ast,
                    "step": step_ast,
                }
            else:
                # Обычная индексация
                tensor_indices = split_top_level(index_expr)
                if len(tensor_indices) > 1:
                    return {
                        "type": "tensor_index_access",
                        "variable": base_name,
                        "indices": [self.parse_expression_to_ast(item) for item in tensor_indices],
                    }
                index_ast = self.parse_expression_to_ast(index_expr)
                return {
                    "type": "index_access",
                    "variable": base_name,
                    "index": index_ast,
                }

        # Вложенная индексация - строим цепочку
        # Сначала парсим базовое выражение
        base_ast = {"type": "variable", "name": base_name, "value": base_name}

        # Строим цепочку индексаций
        current_ast = base_ast

        for i, index_expr in enumerate(indices):
            # Парсим индекс
            index_ast = self.parse_expression_to_ast(index_expr)

            if i == len(indices) - 1:
                # Последний индекс в цепочке
                return {
                    "type": "nested_index_access",
                    "base": current_ast,
                    "index": index_ast,
                    "depth": len(indices),
                    "full_expression": expression,
                }
            else:
                # Промежуточный индекс
                current_ast = {
                    "type": "index_access",
                    "variable": current_ast,
                    "index": index_ast,
                }

        # На всякий случай, если что-то пошло не так
        return {"type": "unknown", "value": expression, "original": expression}

    def parse_assignment(self, line: str, scope: dict):
        logger.debug(
            f"      parse_assignment: парсим '{line}' в scope {scope.get('type', 'unknown')}"
        )

        # Проверяем, является ли это разыменованием указателя (*p = значение)
        deref_pattern = r"\*\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)"
        deref_match = re.match(deref_pattern, line)

        if deref_match:
            # Это запись через указатель: *p = значение
            pointer_name, value = deref_match.groups()
            logger.debug(
                f"      parse_assignment: запись через указатель '{pointer_name}' = '{value}'"
            )

            # Ищем указатель в scope'ах
            result = self.find_symbol_recursive(scope, pointer_name)
            if not result:
                logger.debug(
                    f"      parse_assignment: указатель '{pointer_name}' не найден"
                )
                return False

            pointer_symbol, found_scope = result

            # Проверяем, что это действительно указатель
            if not pointer_symbol["type"].startswith("*"):
                logger.debug(
                    f"      parse_assignment: '{pointer_name}' не является указателем"
                )
                return False

            # Парсим значение в AST
            value_ast = self.parse_expression_to_ast(value)

            operations = [
                {
                    "type": "WRITE_POINTER",
                    "pointer": pointer_name,
                    "value": value_ast,
                    "operation": "*=",
                }
            ]

            dependencies = self.extract_dependencies_from_ast(value_ast)

            scope["graph"].append(
                {
                    "node": "dereference_write",
                    "content": line,
                    "symbols": [pointer_name],
                    "operations": operations,
                    "dependencies": dependencies,
                    "is_dereference_write": True,
                }
            )

            return True

        # Обычное присваивание: переменная = выражение
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)"
        match = re.match(pattern, line)

        if not match:
            logger.debug(f"      parse_assignment: не удалось распарсить")
            return False

        name, expression = match.groups()

        logger.debug(
            f"      parse_assignment: name='{name}', expression='{expression}'"
        )

        # Проверяем, является ли выражение разыменованием указателя (*p)
        if expression.strip().startswith("*"):
            # Это чтение через указатель: x = *p
            pointer_name = expression.strip()[1:].strip()
            logger.debug(
                f"      parse_assignment: чтение через указатель '{pointer_name}'"
            )

            # Ищем указатель в scope'ах
            result = self.find_symbol_recursive(scope, pointer_name)
            if not result:
                logger.debug(
                    f"      parse_assignment: указатель '{pointer_name}' не найден"
                )
                return False

            pointer_symbol, found_scope = result

            # Проверяем, что это действительно указатель
            if not pointer_symbol["type"].startswith("*"):
                logger.debug(
                    f"      parse_assignment: '{pointer_name}' не является указателем"
                )
                return False

            # Ищем целевую переменную
            target_result = self.find_symbol_recursive(scope, name)
            if not target_result:
                logger.debug(
                    f"      parse_assignment: целевая переменная '{name}' не найдена"
                )
                return False

            target_symbol, target_scope = target_result

            # Создаем AST для разыменования
            deref_ast = {"type": "dereference", "pointer": pointer_name}

            operations = [
                {
                    "type": "READ_POINTER",
                    "target": name,
                    "from": pointer_name,
                    "operation": "*",
                    "value": deref_ast,
                    "pointed_type": pointer_symbol["type"][1:],  # Убираем звездочку
                }
            ]

            dependencies = [pointer_name]

            # Обновляем значение в symbol table
            scope["symbol_table"].add_symbol(
                name=name,
                key=target_symbol["key"],
                var_type=target_symbol["type"],
                value=deref_ast,
            )

            scope["graph"].append(
                {
                    "node": "dereference_read",
                    "content": line,
                    "symbols": [name],
                    "operations": operations,
                    "dependencies": dependencies,
                    "is_dereference_read": True,
                }
            )

            return True

        # Обычное присваивание с выражением
        # Ищем символ в текущем scope или в родительских scopes
        symbol = None
        current_scope = scope

        def find_symbol_recursive(current_scope, target_name, visited=None):
            if visited is None:
                visited = set()

            scope_id = id(current_scope)
            if scope_id in visited:
                return None
            visited.add(scope_id)

            # Ищем символ в текущем scope
            symbol = current_scope["symbol_table"].get_symbol(target_name)
            if symbol:
                return symbol, current_scope

            # Если не нашли и есть родительский scope, ищем там
            if "parent_scope" in current_scope:
                parent_level = current_scope["parent_scope"]
                # Ищем scope с нужным уровнем
                for parent in self.scopes:
                    if parent["level"] == parent_level:
                        result = find_symbol_recursive(parent, target_name, visited)
                        if result:
                            return result

            return None

        # Ищем символ рекурсивно
        result = find_symbol_recursive(scope, name)
        if result:
            symbol, found_scope = result
            logger.debug(
                f"      parse_assignment: нашли символ '{name}' типа {symbol['type']} в scope {found_scope.get('type', 'unknown')}"
            )
        else:
            logger.debug(
                f"      parse_assignment: символ '{name}' не найден ни в одном scope"
            )
            return False

        # Парсим выражение в AST
        expression_ast = self.parse_expression_to_ast(expression)

        # Обновляем значение в symbol table
        scope["symbol_table"].add_symbol(
            name=name, key=symbol["key"], var_type=symbol["type"], value=expression_ast
        )

        # Создаем операции
        operations = []
        dependencies = self.extract_dependencies_from_ast(expression_ast)

        # Для простых присваиваний создаем ASSIGN операцию
        if expression_ast["type"] in ["variable", "literal", "function_call"]:
            operations.append(
                {"type": "ASSIGN", "target": name, "value": expression_ast}
            )
        else:
            # Для сложных выражений используем build_operations_from_ast
            self.build_operations_from_ast(
                expression_ast, name, operations, dependencies, scope
            )

        # Добавляем узел в граф
        scope["graph"].append(
            {
                "node": "assignment",
                "content": line,
                "symbols": [name],
                "operations": operations,
                "dependencies": dependencies,
                "expression_ast": expression_ast,
            }
        )

        logger.debug(
            f"      parse_assignment: добавлен узел в граф scope {scope.get('type', 'unknown')}"
        )

        return True

    def parse_augmented_assignment(self, line: str, scope: dict):
        """Парсит составные операции присваивания"""
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(\+=|-=|\*=|/=|//=|\%=|\*\*=|>>=|<<=|&=|\|=|\^=)\s*(.+)"
        match = re.match(pattern, line)

        if not match:
            return False

        name, operator, value = match.groups()

        # Используем тот же поиск, что и в parse_assignment
        def find_symbol_recursive(current_scope, target_name, visited=None):
            if visited is None:
                visited = set()

            scope_id = id(current_scope)
            if scope_id in visited:
                return None
            visited.add(scope_id)

            symbol = current_scope["symbol_table"].get_symbol(target_name)
            if symbol:
                return symbol, current_scope

            if "parent_scope" in current_scope:
                parent_level = current_scope["parent_scope"]
                for parent in self.scopes:
                    if parent["level"] == parent_level:
                        result = find_symbol_recursive(parent, target_name, visited)
                        if result:
                            return result

            return None

        result = find_symbol_recursive(scope, name)
        if not result:
            return False

        symbol, found_scope = result

        # Определяем тип операции
        operator_map = {
            "+=": "ADD",
            "-=": "SUBTRACT",
            "*=": "MULTIPLY",
            "/=": "DIVIDE",
            "//=": "INTEGER_DIVIDE",
            "%=": "MODULO",
            "**=": "POWER",
            ">>=": "RIGHT_SHIFT",
            "<<=": "LEFT_SHIFT",
            "&=": "BITWISE_AND",
            "|=": "BITWISE_OR",
            "^=": "BITWISE_XOR",
        }

        op_type = operator_map.get(operator, "UNKNOWN_AUGMENTED")

        operations = [
            {
                "type": "AUGMENTED_ASSIGN",
                "target": name,
                "operator": op_type,
                "operator_symbol": operator,
                "value": value,
            }
        ]

        value_ast = self.parse_expression_to_ast(value)
        dependencies = self.extract_dependencies_from_ast(value_ast)

        # Обновляем значение переменной
        scope["symbol_table"].add_symbol(
            name=name,
            key="var",
            var_type=symbol["type"],
            value=f"{name} {operator} {value}",
        )

        scope["graph"].append(
            {
                "node": "augmented_assignment",
                "content": line,
                "symbols": [name],
                "operations": operations,
                "dependencies": dependencies,
                "value_ast": value_ast,
            }
        )

        return True

    def parse_complex_expression(
        self,
        target: str,
        expression: str,
        operations: list,
        dependencies: list,
        scope: dict,
    ):
        """Разбирает сложные выражения с несколькими операторами и скобками"""
        expression = expression.strip()

        # Убираем внешние скобки, если выражение полностью в них
        while self.is_fully_parenthesized(expression):
            expression = expression[1:-1].strip()

        # Проверяем, содержит ли выражение операторы
        if not self.contains_operator(expression):
            # Нет операторов - это простое значение или переменная
            clean_expr = expression.strip("() ")
            if (
                clean_expr
                and clean_expr.isalpha()
                and clean_expr not in KEYS
                and clean_expr not in DATA_TYPES
            ):
                dependencies.append(clean_expr)

            operations.append(
                {
                    "type": "ASSIGN",
                    "target": target,
                    "value": self.clean_value(expression),
                }
            )
            return

        # Находим оператор с наименьшим приоритетом
        operator_info = self.find_lowest_priority_operator(expression)

        if not operator_info:
            # Если не нашли оператор, возможно выражение в скобках содержит операторы
            # Попробуем разобрать как есть
            clean_expr = expression.strip("() ")
            if clean_expr:
                temp_var = f"{target}_inner"
                self.parse_complex_expression(
                    temp_var, clean_expr, operations, dependencies, scope
                )
                operations.append(
                    {"type": "ASSIGN", "target": target, "value": temp_var}
                )
            return

        op_symbol, op_type, op_index = operator_info
        left = expression[:op_index].strip()
        right = expression[op_index + len(op_symbol) :].strip()

        # Добавляем основную операцию
        operations.append(
            {
                "type": "BINARY_OPERATION",
                "target": target,
                "operator": op_type,
                "operator_symbol": op_symbol,
                "left": left,
                "right": right,
            }
        )

        # Вспомогательная функция для разбора части выражения
        def _parse_subexpression(subexpr: str, side: str):
            subexpr = subexpr.strip()
            if not subexpr:
                return

            # Убираем внешние скобки
            while self.is_fully_parenthesized(subexpr):
                subexpr = subexpr[1:-1].strip()

            if self.contains_operator(subexpr):
                # Создаем временную переменную для подвыражения
                temp_var = f"{target}_{side}_{len(operations)}"
                self.parse_complex_expression(
                    temp_var, subexpr, operations, dependencies, scope
                )
                # Обновляем ссылку в основной операции
                for op in operations:
                    if (
                        op.get("target") == target
                        and op.get("type") == "BINARY_OPERATION"
                    ):
                        if side == "left":
                            op["left"] = temp_var
                        else:
                            op["right"] = temp_var
            else:
                # Проверяем зависимости
                clean_subexpr = subexpr.strip("() ")
                if (
                    clean_subexpr
                    and clean_subexpr.isalpha()
                    and clean_subexpr not in KEYS
                    and clean_subexpr not in DATA_TYPES
                ):
                    dependencies.append(clean_subexpr)

        # Рекурсивно разбираем левую и правую части
        _parse_subexpression(left, "left")
        _parse_subexpression(right, "right")

    def parse_function_call(self, line: str, scope: dict) -> bool:
        """Универсальный парсер любого вызова функции с поддержкой опций"""
        # Паттерн: func_name(arg1, arg2, key=value)
        pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$"
        match = re.match(pattern, line)

        if not match:
            return False

        func_name, args_str = match.groups()

        # Универсальный парсинг аргументов
        positional_args, keyword_options, all_args_ast = (
            self.parse_arguments_with_options(args_str)
        )

        # Определяем тип функции
        if func_name in self.builtin_functions:
            node_type = "builtin_function_call"
            return_type = self.get_builtin_return_type(func_name, positional_args)
        elif func_name[0].isupper():  # Конструктор класса
            node_type = "constructor_call"
            return_type = func_name
        else:
            node_type = "function_call"
            return_type = "any"

        # Создаем операции
        operations = [
            {
                "type": node_type.upper().replace("_CALL", "_CALL"),
                "function": func_name,
                "arguments": positional_args,
                "kwargs": keyword_options,  # Именованные аргументы
                "return_type": return_type,
            }
        ]

        # Собираем зависимости
        dependencies = []
        for arg in all_args_ast:
            if isinstance(arg, dict):
                if arg.get("type") == "keyword_argument":
                    deps = self.extract_dependencies_from_ast(arg.get("value", {}))
                else:
                    deps = self.extract_dependencies_from_ast(arg)
                dependencies.extend(deps)

        # Создаем узел
        scope["graph"].append(
            {
                "node": node_type,
                "content": line,
                "function": func_name,
                "arguments": positional_args,  # Позиционные аргументы
                "kwargs": keyword_options,  # Именованные аргументы/опции
                "all_arguments": all_args_ast,  # Все аргументы вместе
                "return_type": return_type,
                "operations": operations,
                "dependencies": list(set(dependencies)),
            }
        )

        return True

    def parse_arguments_with_options(self, args_str: str) -> tuple:
        """
        Универсальный парсер аргументов функции.
        Возвращает (positional_args, keyword_options, all_args_ast)
        """
        positional_args = []
        keyword_options = {}
        all_args_ast = []  # Все аргументы в виде AST для обратной совместимости

        if not args_str.strip():
            return positional_args, keyword_options, all_args_ast

        # Разделяем по запятым с учетом вложенности
        parts = []
        current = ""
        depth = 0
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        string_char = None

        for char in (
            args_str + ","
        ):  # Добавляем запятую в конце для обработки последнего элемента
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
                current += char
            elif in_string and char == string_char:
                if current and current[-1] != "\\":
                    in_string = False
                current += char
            elif not in_string:
                # Отслеживаем все виды скобок
                if char == "(":
                    depth += 1
                    current += char
                elif char == ")":
                    depth -= 1
                    current += char
                elif char == "[":
                    bracket_depth += 1
                    current += char
                elif char == "]":
                    bracket_depth -= 1
                    current += char
                elif char == "{":
                    brace_depth += 1
                    current += char
                elif char == "}":
                    brace_depth -= 1
                    current += char
                elif (
                    char == ","
                    and depth == 0
                    and bracket_depth == 0
                    and brace_depth == 0
                ):
                    if current.strip():
                        parts.append(current.strip())
                    current = ""
                else:
                    current += char
            else:
                current += char

        # Парсим каждую часть
        for part in parts:
            # Проверяем, является ли часть именованной опцией. Равенство
            # внутри строкового литерала, например в ``print("loss =", x)``,
            # не является синтаксисом ``key=value``.
            keyword_match = re.match(
                r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=(?!=)(.*)$", part
            )
            if keyword_match and not self._is_inside_brackets(part):
                # Это опция
                key = keyword_match.group(1)
                value = keyword_match.group(2).strip()

                # Парсим значение опции
                value_info = self._parse_option_value(value)
                keyword_options[key] = value_info

                # Добавляем в all_args_ast для обратной совместимости
                all_args_ast.append(
                    {"type": "keyword_argument", "key": key, "value": value_info}
                )
            else:
                # Это позиционный аргумент
                arg_ast = self.parse_expression_to_ast(part)
                positional_args.append(arg_ast)
                all_args_ast.append(arg_ast)

        return positional_args, keyword_options, all_args_ast

    def _is_inside_brackets(self, text: str) -> bool:
        """Проверяет, находится ли "=" внутри скобок (например, в словаре или списке)"""
        depth = 0
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        string_char = None

        for i, char in enumerate(text):
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
            elif in_string and char == string_char:
                if i > 0 and text[i - 1] != "\\":
                    in_string = False
            elif not in_string:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                elif char == "=" and (
                    depth > 0 or bracket_depth > 0 or brace_depth > 0
                ):
                    return True

        return False

    def _parse_option_value(self, value: str) -> dict:
        """Парсит значение опции и определяет его тип"""
        value = value.strip()

        # Проверяем булевы значения
        if value.lower() == "true":
            return {"type": "bool", "value": True}
        elif value.lower() == "false":
            return {"type": "bool", "value": False}

        # Проверяем None
        if value.lower() == "none" or value == "null":
            return {"type": "null", "value": None}

        # Проверяем числа
        if value.isdigit():
            return {"type": "int", "value": int(value)}

        if value.replace(".", "").isdigit() and value.count(".") == 1:
            return {"type": "float", "value": float(value)}

        # Проверяем строки
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            str_value = value[1:-1].replace('\\"', '"').replace("\\'", "'")
            return {"type": "str", "value": str_value}

        # Проверяем списки
        if value.startswith("[") and value.endswith("]"):
            return {"type": "list", "value": self.parse_list_literal(value)}

        # Проверяем словари
        if value.startswith("{") and value.endswith("}"):
            if ":" in value:
                return {"type": "dict", "value": self.parse_dict_literal(value)}
            else:
                return {"type": "set", "value": self.parse_set_literal(value)}

        # Проверяем кортежи
        if value.startswith("(") and value.endswith(")"):
            return {"type": "tuple", "value": self.parse_tuple_literal(value)}

        # Если это переменная или выражение
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
            return {"type": "variable", "value": value}

        # Иначе парсим как выражение
        return {"type": "expression", "value": self.parse_expression_to_ast(value)}

    def parse_function_call_assignment(self, line: str, scope: dict) -> bool:
        """Парсит присваивание результата вызова функции: var x: type = func(args)"""
        # Используем более простой паттерн, так как сложный не всегда работает
        pattern = (
            r"var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)"
        )
        match = re.match(pattern, line)

        if not match:
            return False

        var_name, var_type, value_expr = match.groups()

        # Парсим значение как выражение
        value_ast = self.parse_expression_to_ast(value_expr)

        # Проверяем, является ли значение вызовом функции
        if value_ast.get("type") in [
            "function_call",
            "method_call",
            "static_method_call",
            "constructor_call",
        ]:
            # Добавляем переменную
            scope["symbol_table"].add_symbol(
                name=var_name, key="var", var_type=var_type
            )

            if var_name not in scope["local_variables"]:
                scope["local_variables"].append(var_name)

            operations = [
                {"type": "NEW_VAR", "target": var_name, "var_type": var_type},
                {"type": "ASSIGN", "target": var_name, "value": value_ast},
            ]

            # Собираем зависимости из выражения
            dependencies = self.extract_dependencies_from_ast(value_ast)

            # Определяем тип узла
            node_type = "function_call_assignment"
            if value_ast.get("type") == "constructor_call":
                node_type = "object_creation"
            elif value_ast.get("type") == "static_method_call":
                node_type = "static_method_assignment"
            elif value_ast.get("type") == "method_call":
                node_type = "method_call_assignment"

            scope["graph"].append(
                {
                    "node": node_type,
                    "content": line,
                    "symbols": [var_name],
                    "var_name": var_name,
                    "var_type": var_type,
                    "value_ast": value_ast,
                    "operations": operations,
                    "dependencies": dependencies,
                }
            )

            return True

        return False

    def parse_condition(self, condition: str) -> dict:
        """Парсит условие для циклов и if"""
        # Используем AST парсер для сложных условий
        return self.parse_expression_to_ast(condition)

    def parse_iterable(self, iterable_expr: str) -> dict:
        """Парсит итерируемое выражение для for цикла"""
        # Проверяем range вызов с 1, 2 или 3 аргументами
        range_pattern = r"range\s*\(\s*(.+?)\s*\)"
        range_match = re.match(range_pattern, iterable_expr)

        if range_match:
            args_str = range_match.group(1)
            # Разделяем аргументы по запятым, но учитываем возможные вложенные вызовы
            args = []
            current_arg = ""
            depth = 0  # Для отслеживания вложенных скобок

            for char in args_str:
                if char == "(":
                    depth += 1
                    current_arg += char
                elif char == ")":
                    depth -= 1
                    current_arg += char
                elif char == "," and depth == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char

            if current_arg:
                args.append(current_arg.strip())

            # Очищаем аргументы от лишних пробелов
            args = [arg.strip() for arg in args]

            # Определяем количество аргументов
            if len(args) == 1:
                # range(stop)
                return {
                    "type": "RANGE_CALL",
                    "function": "range",
                    "arguments": {"start": "0", "stop": args[0], "step": "1"},
                }
            elif len(args) == 2:
                # range(start, stop)
                return {
                    "type": "RANGE_CALL",
                    "function": "range",
                    "arguments": {"start": args[0], "stop": args[1], "step": "1"},
                }
            elif len(args) == 3:
                # range(start, stop, step)
                return {
                    "type": "RANGE_CALL",
                    "function": "range",
                    "arguments": {"start": args[0], "stop": args[1], "step": args[2]},
                }
            else:
                # Некорректное количество аргументов
                return {"type": "RANGE_CALL", "function": "range", "arguments": args}

        # Другие итерируемые объекты
        return {"type": "ITERABLE", "expression": iterable_expr}

    def parse_openmp_pragma(self, line: str, source_line: int) -> dict | None:
        """Parse the supported OpenMP loop directive into structured metadata."""
        match = re.match(
            r"^#pragma\s+(omp|opm)\s+parallel\s+for(?:\s+(.*?))?\s*$",
            line,
        )
        if not match:
            return None

        clauses_text = (match.group(2) or "").strip()
        clauses = []
        position = 0
        clause_pattern = re.compile(
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\((?P<args>[^()]*)\))?"
        )
        while position < len(clauses_text):
            while position < len(clauses_text) and clauses_text[position].isspace():
                position += 1
            clause_match = clause_pattern.match(clauses_text, position)
            if not clause_match:
                return {
                    "backend": "openmp",
                    "directive": "parallel for",
                    "clauses": [],
                    "content": line,
                    "source_line": source_line,
                    "error": f"invalid OpenMP clause syntax: {clauses_text!r}",
                }
            clauses.append(
                {
                    "name": clause_match.group("name"),
                    "arguments": (clause_match.group("args") or "").strip(),
                }
            )
            position = clause_match.end()

        return {
            "backend": "openmp",
            "directive": "parallel for",
            "clauses": clauses,
            "content": line,
            "source_line": source_line,
        }

    def parse_while_loop(
        self, line: str, scope: dict, all_lines: list, current_index: int, indent: int
    ):
        """Парсит while цикл"""
        pattern = r"while\s+(.+?)\s*:"
        match = re.match(pattern, line)

        if not match:
            return current_index + 1

        condition = match.group(1).strip()

        # Парсим условие
        condition_ast = self.parse_condition(condition)

        # Находим тело цикла
        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, indent)

        # Создаем узел цикла с ПУСТЫМ телом
        loop_node = {
            "node": "while_loop",
            "content": line,
            "condition": condition_ast,
            "body_level": scope["level"] + 1,
            "body": [],  # Пока пустое
        }

        scope["graph"].append(loop_node)

        # НЕ создаем отдельный scope для тела цикла
        # Вместо этого парсим тело прямо в текущем scope
        # но сохраняем его отдельно для узла цикла

        # Сохраняем текущие значения
        saved_indent = self.current_indent
        self.current_indent = indent + 1

        # Создаем временный список для хранения тела цикла
        body_graph = []

        # Парсим тело цикла
        i = body_start
        while i < body_end:
            body_line = all_lines[i]
            if not body_line.strip():
                i += 1
                continue

            body_indent = self.calculate_indent_level(body_line)
            body_content = body_line.strip()

            # Парсим строку в текущем scope, но сохраняем результат отдельно
            current_graph_len = len(scope["graph"])
            i = self.parse_line(body_content, scope, all_lines, i, body_indent)

            # Извлекаем только что добавленные узлы в тело цикла
            if len(scope["graph"]) > current_graph_len:
                # Берем последние добавленные узлы
                new_nodes = scope["graph"][current_graph_len:]
                body_graph.extend(new_nodes)
                # Удаляем их из основного графа scope
                scope["graph"] = scope["graph"][:current_graph_len]

        # Добавляем собранное тело в узел цикла
        loop_node["body"] = body_graph

        # Восстанавливаем отступ
        self.current_indent = saved_indent

        return body_end

    def parse_for_loop(
        self, line: str, scope: dict, all_lines: list, current_index: int, indent: int
    ):
        """Парсит for цикл"""
        pattern = r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.+?)\s*:"
        match = re.match(pattern, line)

        if not match:
            return current_index + 1

        loop_var, iterable_expr = match.groups()
        loop_var = loop_var.strip()
        iterable_expr = iterable_expr.strip()

        # Парсим итерируемое выражение
        iterable_ast = self.parse_iterable(iterable_expr)

        # Находим тело цикла
        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, indent)

        # Создаем узел цикла с ПУСТЫМ телом
        loop_node = {
            "node": "for_loop",
            "content": line,
            "loop_variable": loop_var,
            "iterable": iterable_ast,
            "body_level": scope["level"] + 1,
            "body": [],  # Пока пустое
        }

        if self._pending_openmp_pragma is not None:
            loop_node["openmp"] = self._pending_openmp_pragma
            self._pending_openmp_pragma = None

        # Добавляем узел цикла в граф текущего scope
        scope["graph"].append(loop_node)

        # Добавляем переменную цикла в таблицу символов текущего scope
        scope["symbol_table"].add_symbol(name=loop_var, key="var", var_type="int")
        if loop_var not in scope["local_variables"]:
            scope["local_variables"].append(loop_var)

        # Сохраняем текущие значения
        saved_indent = self.current_indent
        self.current_indent = indent + 1

        # Создаем временный список для хранения тела цикла
        body_graph = []

        # Парсим тело цикла
        i = body_start
        while i < body_end:
            body_line = all_lines[i]
            if not body_line.strip():
                i += 1
                continue

            body_indent = self.calculate_indent_level(body_line)
            body_content = body_line.strip()

            # Парсим строку в текущем scope, но сохраняем результат отдельно
            current_graph_len = len(scope["graph"])
            i = self.parse_line(body_content, scope, all_lines, i, body_indent)

            # Извлекаем только что добавленные узлы в тело цикла
            if len(scope["graph"]) > current_graph_len:
                # Берем последние добавленные узлы (после узла for_loop)
                new_nodes = scope["graph"][current_graph_len:]
                body_graph.extend(new_nodes)
                # Удаляем их из основного графа scope
                scope["graph"] = scope["graph"][:current_graph_len]

        # Добавляем собранное тело в узел цикла
        loop_node["body"] = body_graph

        # Восстанавливаем отступ
        self.current_indent = saved_indent

        return body_end

    def parse_function_arguments_to_ast(self, args_str: str) -> list:
        """Парсит аргументы функции в список AST"""
        if not args_str.strip():
            return []

        args = []
        current_arg = ""
        depth = 0
        in_string = False
        string_char = None
        escaped = False

        i = 0
        while i < len(args_str):
            char = args_str[i]

            # Обработка экранирования
            if escaped:
                current_arg += char
                escaped = False
                i += 1
                continue

            if char == "\\":
                escaped = True
                current_arg += char
                i += 1
                continue

            # Обработка строк
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
                current_arg += char
            elif in_string and char == string_char:
                in_string = False
                string_char = None
                current_arg += char
            elif in_string:
                current_arg += char

            # Обработка скобок (только вне строк)
            elif not in_string:
                if char == "(":
                    depth += 1
                    current_arg += char
                elif char == ")":
                    depth -= 1
                    current_arg += char
                elif char == "[":
                    depth += 1
                    current_arg += char
                elif char == "]":
                    depth -= 1
                    current_arg += char
                elif char == "{":
                    depth += 1
                    current_arg += char
                elif char == "}":
                    depth -= 1
                    current_arg += char
                elif char == "," and depth == 0:
                    # Нашли разделитель аргументов на верхнем уровне
                    if current_arg.strip():
                        args.append(self.parse_expression_to_ast(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char

            i += 1

        # Обрабатываем последний аргумент
        if current_arg.strip():
            args.append(self.parse_expression_to_ast(current_arg.strip()))

        return args

    def parse_list_literal(self, value: str) -> dict:
        """Парсит литерал списка: [1, 2, 3] или [[1, 2], [3, 4]]"""
        if not (value.startswith("[") and value.endswith("]")):
            return {"type": "unknown", "value": value}

        items_str = value[1:-1].strip()
        items = []

        if items_str:
            current_item = ""
            depth = 0
            in_string = False
            string_char = None

            i = 0
            while i < len(items_str):
                char = items_str[i]

                if not in_string and char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_item += char
                elif in_string and char == string_char:
                    if i > 0 and items_str[i - 1] == "\\":
                        current_item += char
                    else:
                        in_string = False
                        current_item += char
                elif not in_string:
                    if char == "[":
                        depth += 1
                        current_item += char
                    elif char == "]":
                        depth -= 1
                        current_item += char
                    elif char == "(":
                        depth += 1
                        current_item += char
                    elif char == ")":
                        depth -= 1
                        current_item += char
                    elif char == "{":
                        depth += 1
                        current_item += char
                    elif char == "}":
                        depth -= 1
                        current_item += char
                    elif depth == 0 and char == ",":
                        if current_item.strip():
                            item_ast = self.parse_expression_to_ast(
                                current_item.strip()
                            )
                            items.append(item_ast)
                        current_item = ""
                    else:
                        current_item += char
                else:
                    current_item += char

                i += 1

            if current_item.strip():
                item_ast = self.parse_expression_to_ast(current_item.strip())
                items.append(item_ast)

        return {
            "type": "list_literal",
            "items": items,
            "length": len(items),
            "is_nested": any(item.get("type") == "list_literal" for item in items),
        }

    def parse_dict_literal(self, value: str) -> dict:
        """Парсит литерал словаря: {"key": "value", "num": 42}"""
        if not (value.startswith("{") and value.endswith("}")):
            return {"type": "unknown", "value": value}

        content = value[1:-1].strip()
        pairs = {}

        if content:
            current_key = ""
            current_value = ""
            parsing_key = True
            depth = 0
            in_string = False
            string_char = None

            i = 0
            while i < len(content):
                char = content[i]

                if not in_string and char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    if parsing_key:
                        current_key += char
                    else:
                        current_value += char
                elif in_string and char == string_char:
                    # Проверяем экранирование
                    if i > 0 and content[i - 1] == "\\":
                        if parsing_key:
                            current_key += char
                        else:
                            current_value += char
                    else:
                        in_string = False
                        if parsing_key:
                            current_key += char
                        else:
                            current_value += char
                elif not in_string and char == ":":
                    if parsing_key and depth == 0:
                        parsing_key = False
                        # Парсим ключ как AST
                        key_ast = self.parse_expression_to_ast(current_key.strip())
                        current_key = key_ast  # Сохраняем AST ключа
                        current_value = ""  # Начинаем собирать значение
                    elif not parsing_key:
                        current_value += char
                elif not in_string and char == "," and depth == 0:
                    if not parsing_key:
                        # Парсим значение как AST
                        value_ast = self.parse_expression_to_ast(current_value.strip())

                        # Ключ должен быть хешируемым (строковый литерал)
                        if (
                            isinstance(current_key, dict)
                            and current_key.get("type") == "literal"
                        ):
                            key_value = current_key.get("value", "")
                            pairs[key_value] = value_ast
                        elif isinstance(current_key, str):
                            # Если ключ еще строка, парсим его
                            key_ast = self.parse_expression_to_ast(current_key.strip())
                            if key_ast.get("type") == "literal":
                                pairs[key_ast.get("value", "")] = value_ast
                            else:
                                # Не литерал - используем строковое представление
                                pairs[str(current_key)] = value_ast

                        # Сбрасываем для следующей пары
                        current_key = ""
                        current_value = ""
                        parsing_key = True
                elif not in_string and char in ["[", "{", "("]:
                    depth += 1
                    if parsing_key:
                        current_key += char
                    else:
                        current_value += char
                elif not in_string and char in ["]", "}", ")"]:
                    depth -= 1
                    if parsing_key:
                        current_key += char
                    else:
                        current_value += char
                else:
                    if parsing_key:
                        current_key += char
                    else:
                        current_value += char

                i += 1

            # Последняя пара
            if current_key and not parsing_key:
                value_ast = self.parse_expression_to_ast(current_value.strip())

                # Обработка ключа
                if (
                    isinstance(current_key, dict)
                    and current_key.get("type") == "literal"
                ):
                    key_value = current_key.get("value", "")
                    pairs[key_value] = value_ast
                elif isinstance(current_key, str):
                    key_ast = self.parse_expression_to_ast(current_key.strip())
                    if key_ast.get("type") == "literal":
                        pairs[key_ast.get("value", "")] = value_ast
                    else:
                        pairs[str(current_key)] = value_ast

        return {"type": "dict_literal", "pairs": pairs, "size": len(pairs)}

    def parse_set_literal(self, value: str) -> dict:
        """Парсит литерал множества: {1, 2, 3}"""
        if not (value.startswith("{") and value.endswith("}") and ":" not in value):
            return {"type": "unknown", "value": value}

        items_str = value[1:-1].strip()
        items = []
        seen_values = set()  # Для уникальности

        if items_str:
            current_item = ""
            depth = 0
            in_string = False
            string_char = None

            i = 0
            while i < len(items_str):
                char = items_str[i]

                if not in_string and char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_item += char
                elif in_string and char == string_char:
                    if i > 0 and items_str[i - 1] == "\\":
                        current_item += char
                    else:
                        in_string = False
                        current_item += char
                elif not in_string and char in ["[", "{", "("]:
                    depth += 1
                    current_item += char
                elif not in_string and char in ["]", "}", ")"]:
                    depth -= 1
                    current_item += char
                elif not in_string and depth == 0 and char == ",":
                    if current_item.strip():
                        item_ast = self.parse_expression_to_ast(current_item.strip())
                        # Проверяем уникальность по строковому представлению
                        item_str = repr(item_ast)
                        if item_str not in seen_values:
                            items.append(item_ast)
                            seen_values.add(item_str)
                    current_item = ""
                else:
                    current_item += char

                i += 1

            if current_item.strip():
                item_ast = self.parse_expression_to_ast(current_item.strip())
                item_str = repr(item_ast)
                if item_str not in seen_values:
                    items.append(item_ast)

        return {"type": "set_literal", "items": items, "size": len(items)}

    def parse_if_statement(
        self, line: str, scope: dict, all_lines: list, current_index: int, indent: int
    ):
        """Парсит if-elif-else конструкцию - РАБОЧАЯ ВЕРСИЯ без бесконечного цикла"""
        pattern = r"if\s+(.+?)\s*:"
        match = re.match(pattern, line)

        if not match:
            return current_index + 1

        condition = match.group(1).strip()
        condition_ast = self.parse_expression_to_ast(condition)

        # Создаем узел if (пока НЕ добавляем в граф)
        if_node = {
            "node": "if_statement",
            "content": line,
            "condition": condition_ast,
            "condition_ast": condition_ast,
            "body_level": scope["level"] + 1,
            "body": [],
            "elif_blocks": [],
            "else_block": None,
        }

        # Начинаем парсинг с текущей строки
        i = current_index

        # Парсим тело if
        body_start = i + 1
        body_end = self.find_indented_block_end(all_lines, body_start, indent)

        # Сохраняем оригинальный граф
        original_graph_len = len(scope["graph"])

        # Парсим тело if
        saved_indent = self.current_indent
        self.current_indent = indent + 1

        body_i = body_start
        while body_i < body_end:
            body_line = all_lines[body_i]
            if not body_line.strip():
                body_i += 1
                continue

            body_indent = self.calculate_indent_level(body_line)
            body_content = body_line.strip()

            # Парсим строку тела if
            body_i = self.parse_line(
                body_content, scope, all_lines, body_i, body_indent
            )

        # Извлекаем узлы тела if
        if len(scope["graph"]) > original_graph_len:
            if_node["body"] = scope["graph"][original_graph_len:]
            # Удаляем эти узлы из основного графа
            scope["graph"] = scope["graph"][:original_graph_len]

        # Теперь i указывает на строку ПОСЛЕ тела if
        i = body_end

        # Парсим elif блоки (если есть)
        while i < len(all_lines):
            current_line = all_lines[i]

            if not current_line.strip():
                i += 1
                continue

            current_line_indent = self.calculate_indent_level(current_line)
            current_line_content = current_line.strip()

            # Проверяем, что это на том же уровне, что и if
            if current_line_indent != indent:
                # Не тот же уровень - выходим
                break

            # Проверяем elif
            if current_line_content.startswith("elif"):
                # Парсим условие elif
                elif_pattern = r"elif\s+(.+?)\s*:"
                elif_match = re.match(elif_pattern, current_line_content)

                if not elif_match:
                    i += 1
                    continue

                elif_condition = elif_match.group(1).strip()
                elif_condition_ast = self.parse_expression_to_ast(elif_condition)

                # Создаем блок elif
                elif_block = {
                    "node": "elif_statement",
                    "content": current_line_content,
                    "condition": elif_condition_ast,
                    "condition_ast": elif_condition_ast,
                    "body_level": scope["level"] + 1,
                    "body": [],
                }

                if_node["elif_blocks"].append(elif_block)

                # Парсим тело elif
                elif_body_start = i + 1
                elif_body_end = self.find_indented_block_end(
                    all_lines, elif_body_start, indent
                )

                # Сохраняем текущую длину графа
                current_graph_len = len(scope["graph"])

                # Парсим тело elif
                self.current_indent = indent + 1

                elif_body_i = elif_body_start
                while elif_body_i < elif_body_end:
                    elif_body_line = all_lines[elif_body_i]
                    if not elif_body_line.strip():
                        elif_body_i += 1
                        continue

                    elif_body_indent = self.calculate_indent_level(elif_body_line)
                    elif_body_content = elif_body_line.strip()

                    # Парсим строку тела elif
                    elif_body_i = self.parse_line(
                        elif_body_content,
                        scope,
                        all_lines,
                        elif_body_i,
                        elif_body_indent,
                    )

                # Извлекаем узлы тела elif
                if len(scope["graph"]) > current_graph_len:
                    elif_block["body"] = scope["graph"][current_graph_len:]
                    # Удаляем эти узлы из основного графа
                    scope["graph"] = scope["graph"][:current_graph_len]

                # Переходим к строке после тела elif
                i = elif_body_end

            # Проверяем else
            elif current_line_content == "else:":
                # Создаем блок else
                else_block = {
                    "node": "else_statement",
                    "content": current_line_content,
                    "body_level": scope["level"] + 1,
                    "body": [],
                }

                if_node["else_block"] = else_block

                # Парсим тело else
                else_body_start = i + 1
                else_body_end = self.find_indented_block_end(
                    all_lines, else_body_start, indent
                )

                # Сохраняем текущую длину графа
                current_graph_len = len(scope["graph"])

                # Парсим тело else
                self.current_indent = indent + 1

                else_body_i = else_body_start
                while else_body_i < else_body_end:
                    else_body_line = all_lines[else_body_i]
                    if not else_body_line.strip():
                        else_body_i += 1
                        continue

                    else_body_indent = self.calculate_indent_level(else_body_line)
                    else_body_content = else_body_line.strip()

                    # Парсим строку тела else
                    else_body_i = self.parse_line(
                        else_body_content,
                        scope,
                        all_lines,
                        else_body_i,
                        else_body_indent,
                    )

                # Извлекаем узлы тела else
                if len(scope["graph"]) > current_graph_len:
                    else_block["body"] = scope["graph"][current_graph_len:]
                    # Удаляем эти узлы из основного графа
                    scope["graph"] = scope["graph"][:current_graph_len]

                # Переходим к строке после тела else
                i = else_body_end
                break  # После else заканчиваем

            else:
                # Не elif и не else - выходим
                break

        # Восстанавливаем отступ
        self.current_indent = saved_indent

        # Теперь добавляем узел if в граф scope
        scope["graph"].append(if_node)

        return i

    def parse_elif_block(
        self,
        line: str,
        scope: dict,
        all_lines: list,
        current_index: int,
        base_indent: int,
        if_node: dict,
    ):
        """Парсит блок elif"""
        pattern = r"elif\s+(.+?)\s*:"
        match = re.match(pattern, line)

        if not match:
            return current_index + 1

        condition = match.group(1).strip()

        # Парсим условие в AST
        condition_ast = self.parse_expression_to_ast(condition)

        # Находим тело elif
        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, base_indent)

        # Создаем блок elif
        elif_block = {
            "node": "elif_statement",
            "content": line,
            "condition": condition_ast,  # AST вместо простого условия
            "condition_ast": condition_ast,  # Дублируем для совместимости
            "body_level": scope["level"] + 1,
            "body": [],  # Пока пустое
        }

        if_node["elif_blocks"].append(elif_block)

        # Сохраняем текущие значения
        saved_indent = self.current_indent
        self.current_indent = base_indent + 1

        # Создаем временный список для хранения тела elif
        body_graph = []

        # Парсим тело elif
        i = body_start
        while i < body_end:
            body_line = all_lines[i]
            if not body_line.strip():
                i += 1
                continue

            body_indent = self.calculate_indent_level(body_line)
            body_content = body_line.strip()

            # Проверяем, является ли строка elif или else
            if body_indent == base_indent:  # Та же глубина отступа, что и if
                if body_content.startswith("elif"):
                    # Сохраняем текущее тело elif
                    elif_block["body"] = body_graph
                    body_graph = []

                    # Рекурсивно парсим следующий elif
                    i = self.parse_elif_block(
                        body_content, scope, all_lines, i, base_indent, if_node
                    )
                    continue
                elif body_content.startswith("else"):
                    # Сохраняем текущее тело elif
                    elif_block["body"] = body_graph
                    body_graph = []

                    # Парсим else блок
                    i = self.parse_else_block(
                        body_content, scope, all_lines, i, base_indent, if_node
                    )
                    break

            # Парсим строку в текущем scope, но сохраняем результат отдельно
            current_graph_len = len(scope["graph"])
            i = self.parse_line(body_content, scope, all_lines, i, body_indent)

            # Извлекаем только что добавленные узлы в тело elif
            if len(scope["graph"]) > current_graph_len:
                new_nodes = scope["graph"][current_graph_len:]
                body_graph.extend(new_nodes)
                scope["graph"] = scope["graph"][:current_graph_len]

        # Сохраняем тело elif
        elif_block["body"] = body_graph

        # Восстанавливаем отступ
        self.current_indent = saved_indent

        return i

    def parse_else_block(
        self,
        line: str,
        scope: dict,
        all_lines: list,
        current_index: int,
        base_indent: int,
        if_node: dict,
    ):
        """Парсит блок else"""
        pattern = r"else\s*:"
        match = re.match(pattern, line)

        if not match:
            return current_index + 1

        # Находим тело else
        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, base_indent)

        # Создаем блок else
        else_block = {
            "node": "else_statement",
            "content": line,
            "body_level": scope["level"] + 1,
            "body": [],  # Пока пустое
        }

        if_node["else_block"] = else_block

        # Сохраняем текущие значения
        saved_indent = self.current_indent
        self.current_indent = base_indent + 1

        # Создаем временный список для хранения тела else
        body_graph = []

        # Парсим тело else
        i = body_start
        while i < body_end:
            body_line = all_lines[i]
            if not body_line.strip():
                i += 1
                continue

            body_indent = self.calculate_indent_level(body_line)
            body_content = body_line.strip()

            # Парсим строку в текущем scope, но сохраняем результат отдельно
            current_graph_len = len(scope["graph"])
            i = self.parse_line(body_content, scope, all_lines, i, body_indent)

            # Извлекаем только что добавленные узлы в тело else
            if len(scope["graph"]) > current_graph_len:
                new_nodes = scope["graph"][current_graph_len:]
                body_graph.extend(new_nodes)
                scope["graph"] = scope["graph"][:current_graph_len]

        # Сохраняем тело else
        else_block["body"] = body_graph

        # Восстанавливаем отступ
        self.current_indent = saved_indent

        return i

    def parse_nested_if(
        self, line: str, scope: dict, all_lines: list, current_index: int, indent: int
    ):
        """Парсит вложенные if внутри других блоков (while, for, других if)"""
        # Используем ту же логику, что и для обычного if
        return self.parse_if_statement(line, scope, all_lines, current_index, indent)

    def parse_c_call(self, line: str, scope: dict):
        """Парсит прямой вызов C-функции"""
        # Убираем @ в начале
        c_call = line[1:].strip()

        # Паттерн для вызова C-функции: func_name(arg1, arg2, ...)
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)"
        match = re.match(pattern, c_call)

        if not match:
            return False

        func_name, args_str = match.groups()

        # Разбираем аргументы
        args = []
        if args_str.strip():
            args = self.parse_function_arguments(args_str)

        operations = [{"type": "C_CALL", "function": func_name, "arguments": args}]

        # Собираем зависимости из аргументов
        dependencies = []
        for arg in args:
            if isinstance(arg, dict):
                # Если аргумент - AST, извлекаем зависимости
                deps = self.extract_dependencies_from_ast(arg)
                dependencies.extend(deps)
            elif arg.isalpha() and arg not in KEYS and arg not in DATA_TYPES:
                dependencies.append(arg)

        # Создаем узел для C-вызова
        scope["graph"].append(
            {
                "node": "c_call",
                "content": line,
                "function": func_name,
                "arguments": args,
                "operations": operations,
                "dependencies": dependencies,
                "unsafe": self.unsafe_depth > 0,
            }
        )

        return True

    ###############################################################################################
    # CLASSES
    ###############################################################################################

    def parse_struct_declaration(
        self, line: str, scope: dict, all_lines: list, current_index: int
    ):
        """Parse a value-semantic struct.

        Structs intentionally contain fields only in v0.2. Methods can be added
        later without changing their value/inline memory semantics.
        """
        match = re.match(r"struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:", line)
        if not match:
            return current_index + 1

        struct_name = match.group(1)
        base_indent = self.calculate_indent_level(all_lines[current_index])
        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, base_indent)

        fields = []
        i = body_start
        while i < body_end:
            raw = all_lines[i]
            if not raw.strip():
                i += 1
                continue
            field = self._parse_struct_field(raw.strip())
            if field is None:
                logger.debug(
                    f"Error: struct '{struct_name}' содержит неподдерживаемую строку: {raw.strip()}"
                )
                i += 1
                continue
            fields.append(field)
            i += 1

        symbol_id = scope["symbol_table"].add_symbol(
            name=struct_name,
            key="struct",
            var_type="struct",
            value=None,
            fields=fields,
            memory_kind="value",
        )

        scope["graph"].append(
            {
                "node": "struct_declaration",
                "content": line,
                "struct_name": struct_name,
                "symbol_id": symbol_id,
                "fields": fields,
                "memory_kind": "value",
                "operations": [
                    {
                        "type": "DECLARE_STRUCT",
                        "struct_name": struct_name,
                        "fields": fields,
                    }
                ],
            }
        )
        return body_end

    def parse_class_declaration(
        self, line: str, scope: dict, all_lines: list, current_index: int
    ):
        """Парсит объявление класса"""
        pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s*:"
        match = re.match(pattern, line)

        if not match:
            return current_index + 1

        class_name = match.group(1)
        base_classes_str = match.group(2)

        # Парсим родительские классы
        base_classes = []
        if base_classes_str:
            base_classes = [bc.strip() for bc in base_classes_str.split(",")]

        logger.debug(f"DEBUG: Класс '{class_name}' наследует от: {base_classes}")

        # Добавляем класс в таблицу символов с информацией о наследовании
        symbol_id = scope["symbol_table"].add_class(
            name=class_name,
            base_classes=base_classes,  # ← Сохраняем родительские классы
        )

        # Находим тело класса
        body_start = current_index + 1
        base_indent = self.calculate_indent_level(all_lines[current_index])
        body_end = self.find_indented_block_end(all_lines, body_start, base_indent)

        class_node = {
            "node": "class_declaration",
            "content": line,
            "class_name": class_name,
            "symbol_id": symbol_id,
            "base_classes": base_classes,  # ← Важно сохранить
            "body_level": scope["level"] + 1,
            "methods": [],
            "attributes": [],
            "static_methods": [],
            "class_methods": [],
            "inherited_methods": {},  # ← Новое поле для унаследованных методов
        }

        scope["graph"].append(class_node)

        # Парсим тело класса
        saved_indent = self.current_indent
        self.current_indent = base_indent + 1

        # Создаем временную область видимости для тела класса
        class_body_scope = {
            "level": scope["level"] + 1,
            "type": "class_body",
            "parent_scope": scope["level"],
            "class_name": class_name,
            "graph": [],
            "symbol_table": SymbolTable(),
            "methods": [],
            "attributes": [],
        }

        i = body_start
        current_decorator = None

        while i < body_end:
            body_line = all_lines[i]
            if not body_line.strip():
                i += 1
                continue

            body_indent = self.calculate_indent_level(body_line)
            body_content = body_line.strip()

            # Проверяем декораторы методов
            if body_content.startswith("@"):
                current_decorator = body_content
                i += 1
                continue

            # Парсим методы класса
            if body_content.startswith("def "):
                # Определяем тип метода
                is_static = current_decorator == "@staticmethod"
                is_classmethod = current_decorator == "@classmethod"

                # Парсим объявление метода
                method_index = self.parse_class_method_declaration(
                    body_content,
                    scope,
                    all_lines,
                    i,
                    body_indent,
                    class_name,
                    is_static,
                    is_classmethod,
                    class_node,
                )

                if method_index > i:
                    i = method_index
                    current_decorator = None
                    continue

            # Парсим атрибуты класса (var self.attr: type = value)
            elif body_content.startswith("var ") and "self." in body_content:
                self.parse_class_attribute(
                    body_content, class_body_scope, class_name, class_node
                )
                i += 1
                continue

            i += 1

        # Восстанавливаем отступ
        self.current_indent = saved_indent

        return body_end

    def parse_class_method_declaration(
        self,
        line: str,
        parent_scope: dict,
        all_lines: list,
        current_index: int,
        indent: int,
        class_name: str,
        is_static: bool,
        is_classmethod: bool,
        class_node: dict,
    ):
        """Parse a class method using the same typed parameter parser as functions."""
        pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*:"
        match = re.match(pattern, line)
        if not match:
            return current_index + 1

        method_name, params_str, return_type_text = match.groups()
        return_type, return_type_info = self.parse_type_annotation(return_type_text or "None")
        parameters = self.parse_parameters(params_str)

        if not is_static and not is_classmethod:
            self_param = next((p for p in parameters if p["name"] == "self"), None)
            if self_param is None:
                self_type, self_type_info = self.parse_type_annotation(f"&mut {class_name}")
                parameters.insert(
                    0,
                    {
                        "name": "self",
                        "type": self_type,
                        "type_info": self_type_info,
                        "memory_kind": "mut_borrow",
                        "is_borrow": True,
                        "is_mut_borrow": True,
                        "is_pointer": False,
                        "default_value": None,
                        "implicit_self": True,
                    },
                )
            else:
                # Preserve the legacy ClassName type string expected by the current
                # OOP backend while tagging the receiver as a mutable borrow in metadata.
                self_type, self_type_info = self.parse_type_annotation(class_name)
                self_param["type"] = self_type
                self_param["type_info"] = self_type_info
                self_param["memory_kind"] = "shared"
                self_param["object_receiver"] = True
                self_param["receiver_borrow"] = "mut"

        method_info = {
            "name": method_name,
            "parameters": parameters,
            "return_type": return_type,
            "return_type_info": return_type_info,
            "is_static": is_static,
            "is_classmethod": is_classmethod,
        }

        for existing in class_node.get("methods", []) + class_node.get("static_methods", []) + class_node.get("class_methods", []):
            if self.methods_equal(existing, method_info):
                body_start = current_index + 1
                return self.find_indented_block_end(all_lines, body_start, indent)

        if is_static:
            class_node.setdefault("static_methods", []).append(method_info)
        elif is_classmethod:
            class_node.setdefault("class_methods", []).append(method_info)
        else:
            class_node.setdefault("methods", []).append(method_info)

        parent_scope["symbol_table"].add_class_method(
            class_name=class_name,
            method_name=method_name,
            is_static=is_static,
            is_classmethod=is_classmethod,
            parameters=parameters,
            return_type=return_type,
        )

        body_start = current_index + 1
        body_end = self.find_indented_block_end(all_lines, body_start, indent)

        if is_static:
            scope_type = "static_method"
        elif is_classmethod:
            scope_type = "classmethod"
        elif method_name == "__init__":
            scope_type = "constructor"
        else:
            scope_type = "class_method"

        method_scope = {
            "level": parent_scope["level"] + 2,
            "type": scope_type,
            "parent_scope": parent_scope["level"],
            "class_name": class_name,
            "method_name": method_name,
            "parameters": parameters,
            "return_type": return_type,
            "return_type_info": return_type_info,
            "local_variables": [],
            "graph": [],
            "symbol_table": SymbolTable(),
            "return_info": {
                "has_return": False,
                "return_value": None,
                "return_type": return_type,
                "return_type_info": return_type_info,
            },
        }

        for param in parameters:
            method_scope["symbol_table"].add_symbol(
                name=param["name"],
                key="parameter",
                var_type=param["type"],
                type_info=param.get("type_info"),
                memory_kind=param.get("memory_kind"),
            )
            method_scope["local_variables"].append(param["name"])

        self.scopes.append(method_scope)
        self.scope_stack.append(method_scope)
        saved_indent = self.current_indent
        self.current_indent = indent + 1

        i = body_start
        while i < body_end:
            method_line = all_lines[i]
            if not method_line.strip():
                i += 1
                continue
            method_indent = self.calculate_indent_level(method_line)
            method_content = method_line.strip()

            # Constructor fields accept generic, borrow, owned and optional types:
            # self.data: array[float32] = [...]
            if method_name == "__init__" and method_content.startswith("self.") and "=" in method_content:
                if self.parse_class_attribute_initialization(method_content, method_scope):
                    i += 1
                    continue

            i = self.parse_line(method_content, method_scope, all_lines, i, method_indent)

        self.current_indent = saved_indent
        if method_scope in self.scope_stack:
            self.scope_stack.remove(method_scope)
        return body_end

    def parse_class_attribute(
        self, line: str, scope: dict, class_name: str, class_node: dict
    ):
        """Parse ``var self.attr: Type [= value]`` with nested types."""
        match = re.match(r"var\s+self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", line)
        if not match:
            return False
        attr_name, rest = match.groups()
        equals = find_top_level(rest, "=")
        if equals >= 0:
            type_text = rest[:equals].strip()
            value_text = rest[equals + 1 :].strip()
            value_ast = self.parse_expression_to_ast(value_text)
        else:
            type_text = rest.strip()
            value_ast = None

        attr_type, type_info = self.parse_type_annotation(type_text)
        attribute_data = {
            "name": attr_name,
            "type": attr_type,
            "type_info": type_info,
            "memory_kind": type_info.get("memory_kind"),
            "access": "public",
        }

        try:
            scope["symbol_table"].add_class_attribute(
                class_name=class_name,
                attribute_name=attr_name,
                attribute_type=attr_type,
            )
        except Exception:
            pass

        if not any(a.get("name") == attr_name for a in class_node.get("attributes", [])):
            class_node.setdefault("attributes", []).append(attribute_data)

        scope["graph"].append(
            {
                "node": "class_attribute_init",
                "content": line,
                "class_name": class_name,
                "attribute_name": attr_name,
                "attribute_type": attr_type,
                "attribute_type_info": type_info,
                "value": value_ast,
                "operations": [
                    {
                        "type": "CLASS_ATTRIBUTE_INIT",
                        "class_name": class_name,
                        "attribute": attr_name,
                        "value": value_ast,
                        "attribute_type": attr_type,
                    }
                ],
            }
        )
        return True

    def parse_object_method_call_node(
        self, line: str, scope: dict, obj_name: str, method_name: str, args_str: str
    ) -> bool:
        """Парсит вызов метода объекта с учетом наследования"""
        # Парсим аргументы
        args = []
        if args_str.strip():
            args = self.parse_function_arguments_to_ast(args_str)

        # Находим объект и его тип
        # A receiver may be a field path such as ``self.hidden``.  The
        # complete type is resolved by the backend; parsing only needs to
        # preserve the path so the call is not mistaken for a free function.
        obj_symbol, found_scope = (
            self.find_symbol_recursive(scope, obj_name)
            if "." not in obj_name
            else (None, None)
        )
        if not obj_symbol and "." not in obj_name:
            logger.debug(f"Error: Object '{obj_name}' not found")
            return False

        # ИНИЦИАЛИЗИРУЕМ ПЕРЕМЕННЫЕ ПО УМОЛЧАНИЮ
        is_inherited = False
        inherited_from = None
        obj_type = obj_symbol.get("type", "") if obj_symbol else ""
        method_found = False

        # Проверяем, есть ли такой метод в классе или его родителях
        if obj_type:
            for global_scope in self.scopes:
                if global_scope.get("level") == 0:
                    class_symbol = global_scope["symbol_table"].get_symbol(obj_type)
                    if class_symbol:
                        # Проверяем собственные методы
                        for method in class_symbol.get("methods", []):
                            if method["name"] == method_name:
                                method_found = True
                                is_inherited = False
                                inherited_from = None
                                break

                        # Если не нашли в классе, проверяем унаследованные
                        if not method_found and "inherited_methods" in class_symbol:
                            if method_name in class_symbol["inherited_methods"]:
                                method_found = True
                                is_inherited = True
                                inherited_from = class_symbol["inherited_methods"][
                                    method_name
                                ].get("inherited_from")

                        if method_found:
                            break

        if obj_type == "Tensor" or obj_type.startswith("Tensor["):
            if method_name in {
                "add", "sub", "mul", "div", "add_scalar", "sub_scalar",
                "mul_scalar", "div_scalar", "fill", "sum", "copy",
                "transpose", "row", "column", "slice", "reshape", "matmul", "to",
                "shape", "ndim", "size", "device", "get", "set", "release",
                "mean", "max", "min", "dtype", "is_contiguous", "contiguous", "item",
            }:
                method_found = True

        if not method_found:
            logger.debug(
                f"Warning: Method '{method_name}' not found in class '{obj_type}' or its parents"
            )
            # Можно продолжить выполнение, но пометить как неизвестный метод

        # Определяем информацию о методе (если нужно)
        method_info = {}
        if obj_type:
            # Определяем информацию о методе
            method_info = (
                self.resolve_method_info(obj_type, method_name) if obj_type else {}
            )

        # A standalone call must be emitted for void-like object methods. A
        # dotted receiver is resolved later by the backend, so preserve it as
        # standalone there as well (for example ``self.hidden.initialize()``).
        builtin_receiver = (
            obj_type == "str"
            or obj_type.startswith(("list[", "dict[", "tuple[", "Tensor["))
            or obj_type == "Tensor"
        )
        is_standalone = "." in obj_name or (
            method_info.get("return_type") in {None, "None", "void"}
            and (not builtin_receiver or obj_type == "Tensor" or obj_type.startswith("Tensor["))
        )

        operations = [
            {
                "type": "METHOD_CALL",
                "object": obj_name,
                "method": method_name,
                "arguments": args,
                "is_inherited": is_inherited,
                "inherited_from": inherited_from,
                "is_standalone": False,
            }
        ]

        # Собираем зависимости
        dependencies = [obj_name]
        for arg in args:
            deps = self.extract_dependencies_from_ast(arg)
            dependencies.extend(deps)

        # Создаем AST для вызова метода
        method_ast = {
            "type": "method_call",
            "object": obj_name,
            "method": method_name,
            "arguments": args,
            "is_standalone": False,
            "is_inherited": is_inherited,
            "inherited_from": inherited_from,
        }

        scope["graph"].append(
            {
                "node": "method_call",
                "content": line,
                "object": obj_name,
                "method": method_name,
                "method_info": method_info,  # Сохраняем полную информацию
                "arguments": args,
                "is_standalone": is_standalone,
                "operations": operations,
                "dependencies": list(set(dependencies)),
                "expression_ast": method_ast,
                "is_inherited": is_inherited,  # <-- Теперь переменная всегда определена
                "inherited_from": inherited_from,  # <-- И эта тоже
            }
        )

        return True

    def parse_static_method_call_node(
        self,
        line: str,
        scope: dict,
        class_name: str,
        method_name: str,
        args_str: str,
        class_type: str | None = None,
    ) -> bool:
        """Парсит вызов статического метода: Class.method(args)"""
        # Парсим аргументы из строки в список AST
        args = []
        if args_str.strip():
            args = self.parse_function_arguments_to_ast(args_str)

        operations = [
            {
                "type": "STATIC_METHOD_CALL",
                "class_name": class_name,
                "class_type": class_type or class_name,
                "method": method_name,
                "arguments": args,
            }
        ]

        # Собираем зависимости
        dependencies = []
        for arg in args:
            deps = self.extract_dependencies_from_ast(arg)
            dependencies.extend(deps)

        scope["graph"].append(
            {
                "node": "static_method_call",
                "content": line,
                "class_name": class_name,
                "method": method_name,
                "arguments": args,
                "operations": operations,
                "dependencies": dependencies,
            }
        )

        return True

    def is_class_name(self, name: str, scope: dict) -> bool:
        """Проверяет, является ли имя именем класса"""
        # Простая проверка - начинается с заглавной буквы
        if name and name[0].isupper():
            # Дополнительно проверяем в таблице символов
            symbol = scope["symbol_table"].get_symbol(name)
            if symbol and symbol.get("key") == "class":
                return True
            # Если символ не найден, все равно считаем это именем класса
            # (может быть определен в другом модуле)
            return True

        # Проверяем в родительских scope'ах
        if "parent_scope" in scope:
            parent_level = scope["parent_scope"]
            for parent in self.scopes:
                if parent["level"] == parent_level:
                    if self.is_class_name(name, parent):
                        return True

        return False

    def parse_class_attribute_initialization(self, line: str, scope: dict) -> bool:
        """Parse ``self.attr [: Type] = value`` in a constructor."""
        match = re.match(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*(.*)$", line)
        if not match:
            return False

        attr_name, rest = match.groups()
        rest = rest.strip()
        declared_type = None

        if rest.startswith(":"):
            rest = rest[1:].strip()
            equals = find_top_level(rest, "=")
            if equals < 0:
                return False
            declared_type = rest[:equals].strip()
            value_text = rest[equals + 1 :].strip()
        elif rest.startswith("="):
            value_text = rest[1:].strip()
        else:
            return False

        value_ast = self.parse_expression_to_ast(value_text)
        if declared_type:
            attr_type, type_info = self.parse_type_annotation(declared_type)
        else:
            attr_type = self._infer_type_from_ast(value_ast)
            attr_type, type_info = self.parse_type_annotation(attr_type)

        container_info = self._extract_container_info(value_ast)
        class_name = scope.get("class_name", "")

        for global_scope in self.scopes:
            if global_scope.get("level") != 0:
                continue
            class_symbol = global_scope["symbol_table"].get_symbol(class_name)
            if not class_symbol:
                break
            attributes = class_symbol.setdefault("attributes", [])
            existing = next((a for a in attributes if a.get("name") == attr_name), None)
            attribute_data = {
                "name": attr_name,
                "type": attr_type,
                "type_info": type_info,
                "memory_kind": type_info.get("memory_kind"),
                "access": "public",
            }
            if container_info:
                attribute_data["container_info"] = container_info
            if existing is None:
                attributes.append(attribute_data)
            else:
                existing.update(attribute_data)

            for node in global_scope["graph"]:
                if node.get("node") == "class_declaration" and node.get("class_name") == class_name:
                    node_attributes = node.setdefault("attributes", [])
                    node_existing = next((a for a in node_attributes if a.get("name") == attr_name), None)
                    if node_existing is None:
                        node_attributes.append(dict(attribute_data))
                    else:
                        node_existing.update(attribute_data)
                    break
            break

        scope["graph"].append(
            {
                "node": "attribute_assignment",
                "content": line,
                "object": "self",
                "attribute": attr_name,
                "value": value_ast,
                "attribute_type": attr_type,
                "attribute_type_info": type_info,
                "container_info": container_info,
                "operations": [
                    {
                        "type": "ATTRIBUTE_ASSIGN",
                        "object": "self",
                        "attribute": attr_name,
                        "value": value_ast,
                        "attribute_type": attr_type,
                        "attribute_type_info": type_info,
                    }
                ],
                "dependencies": self.extract_dependencies_from_ast(value_ast),
            }
        )
        return True

    ###############################################################################################
    # EXTRACT
    ###############################################################################################

    def extract_content_inside_brackets(
        self, s: str, prefix: str, closing_bracket: str
    ) -> str:
        """Извлекает содержимое внутри скобок, учитывая вложенность"""
        if not s.startswith(prefix):
            return ""

        content = s[len(prefix) :]
        depth = 0
        result = []

        for i, char in enumerate(content):
            if char == "[":
                depth += 1
                result.append(char)
            elif char == "]":
                if depth == 0:
                    # Нашли закрывающую скобку
                    return "".join(result)
                depth -= 1
                result.append(char)
            else:
                result.append(char)

        return "".join(result)

    def extract_dependencies_from_ast(self, ast: dict) -> list:
        """Извлекает зависимости (используемые переменные) из AST"""
        dependencies = []

        def traverse(node):
            if not isinstance(node, dict):
                return

            node_type = node.get("type")

            if node_type == "chained_index_access":
                # Для цепочек индексации: a[0][1]
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)

                # Обходим индексы
                for idx in node.get("indices", []):
                    traverse(idx)

            elif node_type == "index_access":
                var_or_expr = node.get("variable") or node.get("expression")
                if isinstance(var_or_expr, dict):
                    traverse(var_or_expr)
                elif var_or_expr and var_or_expr not in dependencies:
                    dependencies.append(var_or_expr)
                traverse(node.get("index"))

            elif node_type == "nested_index_access":
                # Для вложенных индексаций
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)

                # Обходим индексы
                for idx in node.get("indices", []):
                    traverse(idx)

            elif node_type == "slice_access":
                var_or_expr = node.get("variable") or node.get("expression")
                if isinstance(var_or_expr, dict):
                    traverse(var_or_expr)
                elif var_or_expr and var_or_expr not in dependencies:
                    dependencies.append(var_or_expr)
                traverse(node.get("start"))
                traverse(node.get("stop"))
                traverse(node.get("step"))

            elif node_type == "variable":
                var_name = node.get("name") or node.get("value")
                if var_name and var_name not in dependencies:
                    # Проверяем, что это не ключевое слово или тип данных
                    if (
                        var_name not in KEYS
                        and var_name not in DATA_TYPES
                        and var_name not in self.builtin_functions
                        and not var_name.startswith('"')
                        and not var_name.endswith('"')
                    ):
                        dependencies.append(var_name)

            elif node_type == "attribute_access":
                obj_name = node.get("object")
                if obj_name and obj_name not in dependencies:
                    # Проверяем, что это не ключевое слово
                    if (
                        obj_name not in KEYS
                        and obj_name not in DATA_TYPES
                        and obj_name not in self.builtin_functions
                    ):
                        dependencies.append(obj_name)

            elif node_type == "method_call":
                obj_name = node.get("object")
                if obj_name and obj_name not in dependencies:
                    dependencies.append(obj_name)

                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "static_method_call":
                # Статические методы не требуют зависимостей от объектов
                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "constructor_call":
                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "function_call":
                func_name = node.get("function")
                # Только пользовательские функции добавляем как зависимости
                if func_name and func_name not in self.builtin_functions:
                    if func_name not in dependencies:
                        dependencies.append(func_name)

                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "binary_operation":
                traverse(node.get("left"))
                traverse(node.get("right"))

            elif node_type == "unary_operation":
                traverse(node.get("operand"))

            elif node_type == "ternary_operator":
                traverse(node.get("condition"))
                traverse(node.get("true_expr"))
                traverse(node.get("false_expr"))

            elif node_type == "borrow":
                traverse(node.get("source"))

            elif node_type == "address_of":
                expr = node.get("expression") or node.get("variable")
                if isinstance(expr, dict):
                    traverse(expr)
                elif expr and expr not in dependencies:
                    dependencies.append(expr)

            elif node_type == "dereference":
                expr = node.get("expression") or node.get("pointer")
                if isinstance(expr, dict):
                    traverse(expr)
                elif expr and expr not in dependencies:
                    dependencies.append(expr)

            elif node_type == "index_access":
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)
                traverse(node.get("index"))

            elif node_type == "slice_access":
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)
                traverse(node.get("start"))
                traverse(node.get("stop"))
                traverse(node.get("step"))

            elif node_type == "list_literal":
                for item in node.get("items", []):
                    traverse(item)

            elif node_type == "tuple_literal":
                for item in node.get("items", []):
                    traverse(item)

            elif node_type == "dict_literal":
                for key, value in node.get("pairs", {}).items():
                    traverse(value)

            elif node_type == "set_literal":
                for item in node.get("items", []):
                    traverse(item)

            elif node_type == "fstring":
                for part in node.get("parts", []):
                    if part.get("type") == "fstring_expr":
                        traverse(part.get("expression"))

        traverse(ast)
        return list(set(dependencies))  # Убираем дубликаты

    def _extract_container_info(self, ast: dict) -> dict:
        """Извлекает информацию о контейнере из AST"""
        if not ast:
            return None

        node_type = ast.get("type", "")

        if node_type == "list_literal":
            items = ast.get("items", [])
            element_type = "any"

            if items:
                # Определяем тип первого элемента
                first_item_type = self._infer_type_from_ast(items[0])
                element_type = first_item_type

            return {
                "type": f"list[{element_type}]",
                "container_type": "list",
                "element_type": element_type,
                "size": len(items),
                "is_dynamic": True,
            }

        elif node_type == "borrow":
            source_ast = ast.get("source", {})
            source_type = self._infer_type_from_ast(source_ast)
            if ast.get("mutable"):
                return f"&mut {source_type}"
            return f"&{source_type}"

        elif node_type == "variable":
            # Пытаемся определить тип переменной
            var_name = ast.get("name", ast.get("value", ""))

            # Ищем переменную в таблицах символов
            for scope in reversed(self.scope_stack):
                symbol = scope["symbol_table"].get_symbol(var_name)
                if symbol:
                    var_type = symbol.get("type", "")
                    if var_type.startswith("list["):
                        return {
                            "type": var_type,
                            "container_type": "list",
                            "element_type": var_type[
                                5:-1
                            ],  # Извлекаем list[float] -> float
                            "is_dynamic": True,
                        }

        return None

    ###############################################################################################
    # OTHER
    ###############################################################################################

    def is_fully_parenthesized(self, expression: str) -> bool:
        """Проверяет, полностью ли выражение заключено в скобки"""
        if not (expression.startswith("(") and expression.endswith(")")):
            return False

        # Проверяем баланс скобок
        balance = 0
        for i, char in enumerate(expression):
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
                # Если баланс стал 0 до конца строки, это не полное обрамление
                if balance == 0 and i < len(expression) - 1:
                    return False

        return balance == 0

    def find_lowest_priority_operator(self, expression: str):
        """Находит оператор с наименьшим приоритетом вне скобок"""
        # Приоритет операций (от низшего к высшему)
        operator_levels = [
            # Уровень 1 (наименьший приоритет)
            [("|", "BITWISE_OR")],
            # Уровень 2
            [("^", "BITWISE_XOR")],
            # Уровень 3
            [("&", "BITWISE_AND")],
            # Уровень 4
            [("<<", "LEFT_SHIFT"), (">>", "RIGHT_SHIFT")],
            # Уровень 5
            [("+", "ADD"), ("-", "SUBTRACT")],
            # Уровень 6
            [
                ("*", "MULTIPLY"),
                ("/", "DIVIDE"),
                ("//", "INTEGER_DIVIDE"),
                ("%", "MODULO"),
            ],
            # Уровень 7 (наивысший приоритет)
            [("**", "POWER")],
        ]

        # Ищем операторы от низшего приоритета к высшему
        for level in operator_levels:
            for op_symbol, op_type in level:
                # Ищем оператор вне скобок
                index = self.find_operator_outside_parentheses(expression, op_symbol)
                if index != -1:
                    return (op_symbol, op_type, index)

        return None

    def is_identifier_char(self, char: str) -> bool:
        """Проверяет, является ли символ частью идентификатора"""
        return char.isalnum() or char == "_"

    def find_operator_outside_parentheses(self, expression: str, operator: str) -> int:
        """Находит позицию оператора вне скобок, строк и комментариев"""
        balance = 0  # Баланс круглых скобок
        brace_balance = 0  # Баланс фигурных скобок
        bracket_balance = 0  # Баланс квадратных скобок
        in_string = False  # Находимся ли внутри строки
        string_char = None  # Символ, открывший строку
        escaped = False  # Экранирован ли текущий символ

        i = 0
        while i < len(expression):
            char = expression[i]

            # Обработка экранирования
            if escaped:
                escaped = False
                i += 1
                continue

            if char == "\\":
                escaped = True
                i += 1
                continue

            # Обработка строк
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
            elif in_string and char == string_char:
                in_string = False
                string_char = None

            # Обработка скобок (только вне строк)
            if not in_string:
                if char == "(":
                    balance += 1
                elif char == ")":
                    balance -= 1
                elif char == "{":
                    brace_balance += 1
                elif char == "}":
                    brace_balance -= 1
                elif char == "[":
                    bracket_balance += 1
                elif char == "]":
                    bracket_balance -= 1

                # Проверяем оператор, если мы на верхнем уровне всех скобок
                if balance == 0 and brace_balance == 0 and bracket_balance == 0:
                    # Проверяем совпадение оператора
                    if expression[i : i + len(operator)] == operator:
                        # Проверяем контекст, чтобы не перепутать с частью другого оператора или идентификатора
                        before_ok = i == 0 or not self.is_identifier_char(
                            expression[i - 1]
                        )
                        after_ok = i + len(operator) >= len(
                            expression
                        ) or not self.is_identifier_char(expression[i + len(operator)])

                        if before_ok and after_ok:
                            return i

            i += 1

        return -1

    def contains_operator(self, expression: str) -> bool:
        """Проверяет, содержит ли выражение какой-либо оператор"""
        expression = expression.strip()

        # Сначала убираем внешние скобки
        while self.is_fully_parenthesized(expression):
            expression = expression[1:-1].strip()

        operators = ["+", "-", "*", "/", "//", "%", "**", ">>", "<<", "&", "|", "^"]

        balance = 0
        for i, char in enumerate(expression):
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
            elif balance == 0:  # Мы вне скобок
                for op in operators:
                    if expression[i : i + len(op)] == op:
                        # Проверяем контекст
                        before_ok = i == 0 or not expression[i - 1].isalnum()
                        after_ok = (
                            i + len(op) >= len(expression)
                            or not expression[i + len(op)].isalnum()
                        )

                        if before_ok and after_ok:
                            return True

        return False

    def get_builtin_return_type(self, func_name: str, args: list) -> str:
        """Определяет тип возвращаемого значения для встроенной функции"""
        if func_name == "len":
            return "int"
        elif func_name == "str":
            return "str"
        elif func_name == "int":
            return "int"
        elif func_name == "bool":
            return "bool"
        elif func_name == "print":
            return "None"
        elif func_name == "range":
            return "range"
        elif func_name == "input":  # ДОБАВЛЕНО
            return "str"  # input всегда возвращает строку
        return "unknown"

    def build_operations_from_ast(
        self, ast: dict, target: str, operations: list, dependencies: list, scope: dict
    ):
        """Строит операции из AST выражения"""

        if ast["type"] == "variable":
            operations.append({"type": "ASSIGN", "target": target, "value": ast})
            if ast["value"] not in dependencies:
                dependencies.append(ast["value"])

        elif ast["type"] == "literal":
            operations.append({"type": "ASSIGN", "target": target, "value": ast})

        elif ast["type"] == "binary_operation":
            # Создаем временные переменные для левой и правой частей
            left_temp = f"{target}_left"
            right_temp = f"{target}_right"

            # Рекурсивно обрабатываем левую часть
            self.build_operations_from_ast(
                ast["left"], left_temp, operations, dependencies, scope
            )

            # Рекурсивно обрабатываем правую часть
            self.build_operations_from_ast(
                ast["right"], right_temp, operations, dependencies, scope
            )

            # Добавляем бинарную операцию
            operations.append(
                {
                    "type": "BINARY_OPERATION",
                    "target": target,
                    "operator": ast["operator"],
                    "operator_symbol": ast["operator_symbol"],
                    "left": {"type": "variable", "value": left_temp},
                    "right": {"type": "variable", "value": right_temp},
                }
            )

        elif ast["type"] == "function_call":
            # Обрабатываем аргументы
            arg_operations = []
            for i, arg_ast in enumerate(ast["arguments"]):
                arg_temp = f"{target}_arg_{i}"
                self.build_operations_from_ast(
                    arg_ast, arg_temp, arg_operations, dependencies, scope
                )

            # Добавляем операции аргументов
            operations.extend(arg_operations)

            # Добавляем вызов функции
            arg_values = [
                {"type": "variable", "value": f"{target}_arg_{i}"}
                for i in range(len(ast["arguments"]))
            ]

            operations.append(
                {
                    "type": "FUNCTION_CALL_ASSIGN",
                    "target": target,
                    "function": ast["function"],
                    "arguments": arg_values,
                }
            )

        elif ast["type"] == "dereference":
            operations.append(
                {
                    "type": "READ_POINTER",
                    "target": target,
                    "from": ast["pointer"],
                    "operation": "*",
                    "value": ast,
                }
            )
            if ast["pointer"] not in dependencies:
                dependencies.append(ast["pointer"])

    def find_symbol_recursive(self, current_scope, target_name, visited=None):
        """Рекурсивно ищет символ в текущем и родительских scope'ах"""
        if visited is None:
            visited = set()

        # Проверяем, не посещали ли мы уже этот scope
        scope_id = id(current_scope)
        if scope_id in visited:
            return None
        visited.add(scope_id)

        # Ищем символ в текущем scope
        symbol = current_scope["symbol_table"].get_symbol(target_name)
        if symbol:
            return symbol, current_scope

        # Если не нашли и есть родительский scope, ищем там
        if "parent_scope" in current_scope:
            parent_level = current_scope["parent_scope"]
            # Ищем scope с нужным уровнем
            for parent in self.scopes:
                if parent["level"] == parent_level:
                    result = self.find_symbol_recursive(parent, target_name, visited)
                    if result:
                        return result

        return None

    def clean_value(self, value: str):
        """Очищает значение от лишних пробелов, но для сложных выражений возвращает AST"""
        value = value.strip()

        if not value:
            return {"type": "empty", "value": ""}

        # Сначала проверяем простые литералы
        if value.isdigit():
            return {"type": "literal", "value": int(value), "data_type": "int"}
        elif value == "True":
            return {"type": "literal", "value": True, "data_type": "bool"}
        elif value == "False":
            return {"type": "literal", "value": False, "data_type": "bool"}
        elif value == "None":
            return {"type": "literal", "value": None, "data_type": "None"}
        elif value == "null":
            return {"type": "literal", "value": "null", "data_type": "null"}
        elif (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            content = value[1:-1]
            content = content.replace('\\"', '"').replace("\\'", "'")
            return {"type": "literal", "value": content, "data_type": "str"}

        # Если это не простой литерал, парсим как выражение
        ast = self.parse_expression_to_ast(value)

        # Если парсинг не удался, возвращаем как неизвестное
        if ast["type"] == "unknown":
            return {"type": "literal", "value": value, "data_type": "any"}

        return ast

    def parse_single_parameter(self, param_str: str) -> dict:
        """Parse ``name: Type = default`` with nested generic/borrow types."""
        param_str = param_str.strip()
        if not param_str:
            return None

        colon = find_top_level(param_str, ":")
        equals = find_top_level(param_str, "=")

        if colon >= 0:
            name = param_str[:colon].strip()
            typed_part = param_str[colon + 1 :].strip()
            typed_equals = find_top_level(typed_part, "=")
            if typed_equals >= 0:
                type_text = typed_part[:typed_equals].strip()
                default_value = typed_part[typed_equals + 1 :].strip()
            else:
                type_text = typed_part
                default_value = None
        elif equals >= 0:
            name = param_str[:equals].strip()
            type_text = "any"
            default_value = param_str[equals + 1 :].strip()
        else:
            name = param_str
            type_text = "any"
            default_value = None

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return None

        canonical, type_info = self.parse_type_annotation(type_text)
        result = {
            "name": name,
            "type": canonical,
            "type_info": type_info,
            "memory_kind": type_info.get("memory_kind", "value"),
            "is_borrow": type_info.get("kind") in {"borrow", "mut_borrow"},
            "is_mut_borrow": type_info.get("kind") == "mut_borrow",
            "is_pointer": type_info.get("kind") == "raw_pointer",
        }
        if default_value is not None:
            result["default_value"] = default_value
            result["default_ast"] = self.parse_expression_to_ast(default_value)
        else:
            result["default_value"] = None
        return result

    def _infer_type_from_ast(self, ast: dict) -> str:
        """Выводит тип из AST выражения"""
        if not ast:
            return "any"

        node_type = ast.get("type", "")

        if node_type == "literal":
            data_type = ast.get("data_type", "")
            if data_type == "int":
                return "int"
            elif data_type == "float":
                return "float"
            elif data_type == "str":
                return "str"
            elif data_type == "bool":
                return "bool"
            elif data_type == "None":
                return "None"
            else:
                return "any"

        elif node_type == "variable":
            # Пытаемся определить тип переменной
            var_name = ast.get("name", ast.get("value", ""))

            # Ищем тип переменной в таблицах символов
            # Проверяем сначала в текущем scope, затем в родительских
            for scope in reversed(self.scope_stack):
                symbol = scope["symbol_table"].get_symbol(var_name)
                if symbol:
                    return symbol.get("type", "any")

            # Если не нашли в текущих scopes, проверяем во всех scopes
            for scope in self.scopes:
                symbol = scope["symbol_table"].get_symbol(var_name)
                if symbol:
                    return symbol.get("type", "any")

            # Если переменная - self
            if var_name == "self":
                # Определяем класс из текущего scope
                current_scope = self.scope_stack[-1] if self.scope_stack else None
                if current_scope and "class_name" in current_scope:
                    return current_scope["class_name"]

            # Предполагаем int по умолчанию для переменных
            return "int"

        elif node_type == "list_literal":
            # Определяем тип элементов списка
            items = ast.get("items", [])
            if items:
                # Определяем тип первого элемента
                first_item_type = self._infer_type_from_ast(items[0])
                return f"list[{first_item_type}]"
            return "list[any]"

        elif node_type == "attribute_access":
            # Доступ к атрибуту объекта: self.data
            obj_name = ast.get("object")
            attr_name = ast.get("attribute")

            # Если это self.something
            if obj_name == "self":
                # Находим класс
                current_scope = self.scope_stack[-1] if self.scope_stack else None
                if current_scope and "class_name" in current_scope:
                    class_name = current_scope["class_name"]

                    # Ищем атрибут в классе
                    for scope in self.scopes:
                        if scope.get("level") == 0:  # Глобальная область
                            class_symbol = scope["symbol_table"].get_symbol(class_name)
                            if class_symbol:
                                for attr in class_symbol.get("attributes", []):
                                    if attr["name"] == attr_name:
                                        return attr.get("type", "any")

            return "any"

        elif node_type in ["binary_operation", "unary_operation"]:
            # Для операций предполагаем числовой тип
            return "int"

        elif node_type == "constructor_call":
            # Создание объекта: ClassName(args)
            return ast.get("class_name", "any")

        else:
            return "any"

    def methods_equal(self, method1: dict, method2: dict) -> bool:
        """Сравнивает два метода"""
        if method1["name"] != method2["name"]:
            return False

        if method1.get("is_static", False) != method2.get("is_static", False):
            return False

        if method1.get("is_classmethod", False) != method2.get("is_classmethod", False):
            return False

        # Сравниваем параметры
        params1 = method1.get("parameters", [])
        params2 = method2.get("parameters", [])

        if len(params1) != len(params2):
            return False

        for p1, p2 in zip(params1, params2):
            if p1["name"] != p2["name"] or p1["type"] != p2["type"]:
                return False

        return True

    def parse_parameters(self, params_str: str) -> list:
        """Parse a comma-separated parameter list at top level."""
        if not params_str.strip():
            return []
        parameters = []
        for part in split_top_level(params_str):
            parsed = self.parse_single_parameter(part)
            if parsed:
                parameters.append(parsed)
        return parameters

    def remove_duplicate_methods(self):
        """Удаляет дублирующиеся методы из классов"""
        for scope in self.scopes:
            if scope["level"] == 0:  # Глобальная область
                # Удаляем дубликаты в symbol table
                for class_name, class_symbol in scope["symbol_table"].symbols.items():
                    if class_symbol.get("key") == "class":
                        methods = class_symbol.get("methods", [])
                        unique_methods = []
                        seen = set()

                        for method in methods:
                            # Создаем ключ для сравнения
                            method_key = (
                                method["name"],
                                tuple(
                                    (p["name"], p["type"])
                                    for p in method.get("parameters", [])
                                ),
                                method.get("return_type", "None"),
                                method.get("is_static", False),
                                method.get("is_classmethod", False),
                            )

                            if method_key not in seen:
                                seen.add(method_key)
                                unique_methods.append(method)

                        class_symbol["methods"] = unique_methods

                # Также удаляем дубликаты в узлах графа
                for node in scope["graph"]:
                    if node.get("node") == "class_declaration":
                        methods = node.get("methods", [])
                        unique_methods = []
                        seen = set()

                        for method in methods:
                            method_key = (
                                method["name"],
                                tuple(
                                    (p["name"], p["type"])
                                    for p in method.get("parameters", [])
                                ),
                                method.get("return_type", "None"),
                            )

                            if method_key not in seen:
                                seen.add(method_key)
                                unique_methods.append(method)

                        node["methods"] = unique_methods

    def collect_inherited_methods_for_all_classes(self):
        """Собирает унаследованные методы для всех классов"""
        for scope in self.scopes:
            if scope["level"] == 0:  # Глобальная область
                # Проходим по всем классам
                for class_name, class_symbol in scope["symbol_table"].symbols.items():
                    if class_symbol.get("key") == "class":
                        base_classes = class_symbol.get("base_classes", [])
                        if base_classes:
                            inherited_methods = {}

                            # Собираем методы из всех родительских классов
                            for base_class in base_classes:
                                parent_symbol = scope["symbol_table"].get_symbol(
                                    base_class
                                )
                                if parent_symbol:
                                    # Методы родителя
                                    for method in parent_symbol.get("methods", []):
                                        method_name = method["name"]
                                        if method_name not in inherited_methods:
                                            inherited_methods[method_name] = {
                                                "name": method["name"],
                                                "parameters": method.get(
                                                    "parameters", []
                                                ),
                                                "return_type": method.get(
                                                    "return_type", "None"
                                                ),
                                                "is_static": method.get(
                                                    "is_static", False
                                                ),
                                                "is_classmethod": method.get(
                                                    "is_classmethod", False
                                                ),
                                                "inherited_from": base_class,
                                            }

                                    # Также собираем унаследованные методы родителя
                                    if "inherited_methods" in parent_symbol:
                                        for method_name, method_info in parent_symbol[
                                            "inherited_methods"
                                        ].items():
                                            if method_name not in inherited_methods:
                                                inherited_methods[method_name] = (
                                                    method_info
                                                )

                            # Сохраняем в символ класса
                            class_symbol["inherited_methods"] = inherited_methods

                            # Также обновляем узел в графе
                            for node in scope["graph"]:
                                if (
                                    node.get("node") == "class_declaration"
                                    and node.get("class_name") == class_name
                                ):
                                    node["inherited_methods"] = inherited_methods
                                    break

    def resolve_method_info(self, class_name: str, method_name: str) -> dict:
        """Разрешает информацию о методе с учетом наследования"""
        result = {"found": False, "is_inherited": False, "inherited_from": None}

        # ПРОВЕРКА ВСТРОЕННЫХ ТИПОВ
        builtin_methods = {
            "str": [
                "upper",
                "lower",
                "strip",
                "split",
                "join",
                "replace",
                "find",
                "startswith",
                "endswith",
            ],
            "int": ["to_string", "to_float", "abs", "max", "min"],
            "float": ["to_string", "to_int", "abs", "round", "ceil", "floor"],
            "list": [
                "append",
                "extend",
                "insert",
                "remove",
                "pop",
                "clear",
                "index",
                "count",
                "sort",
                "reverse",
            ],
            "dict": [
                "keys",
                "values",
                "items",
                "get",
                "pop",
                "clear",
                "update",
                "copy",
            ],
            "bool": ["__bool__", "__not__"],
        }

        if class_name in builtin_methods:
            if method_name in builtin_methods[class_name]:
                result["found"] = True
                result["is_inherited"] = False
                result["inherited_from"] = None
                result["is_builtin"] = True  # Новое поле
                return result

        # Ищем класс в глобальной области видимости
        for global_scope in self.scopes:
            if global_scope.get("level") == 0:  # Глобальная область
                class_symbol = global_scope["symbol_table"].get_symbol(class_name)

                if not class_symbol or class_symbol.get("key") != "class":
                    return result

                # Проверяем собственные методы класса
                for method in class_symbol.get("methods", []):
                    if method["name"] == method_name:
                        result["found"] = True
                        result["is_inherited"] = False
                        result["inherited_from"] = None
                        # Добавляем полную информацию о методе
                        result.update(
                            {
                                "name": method["name"],
                                "parameters": method.get("parameters", []),
                                "return_type": method.get("return_type", "None"),
                                "is_static": method.get("is_static", False),
                                "is_classmethod": method.get("is_classmethod", False),
                            }
                        )
                        return result

                # Проверяем статические методы
                for method in class_symbol.get("static_methods", []):
                    if method["name"] == method_name:
                        result["found"] = True
                        result["is_inherited"] = False
                        result["inherited_from"] = None
                        result.update(
                            {
                                "name": method["name"],
                                "parameters": method.get("parameters", []),
                                "return_type": method.get("return_type", "None"),
                                "is_static": True,
                                "is_classmethod": False,
                            }
                        )
                        return result

                # Проверяем унаследованные методы
                if "inherited_methods" in class_symbol:
                    if method_name in class_symbol["inherited_methods"]:
                        method_info = class_symbol["inherited_methods"][method_name]
                        result["found"] = True
                        result["is_inherited"] = True
                        result["inherited_from"] = method_info.get("inherited_from")
                        result.update(
                            {
                                "name": method_info["name"],
                                "parameters": method_info.get("parameters", []),
                                "return_type": method_info.get("return_type", "None"),
                                "is_static": method_info.get("is_static", False),
                                "is_classmethod": method_info.get(
                                    "is_classmethod", False
                                ),
                            }
                        )
                        return result

                # Проверяем родительские классы рекурсивно
                base_classes = class_symbol.get("base_classes", [])
                for base_class in base_classes:
                    parent_info = self.resolve_method_info(base_class, method_name)
                    if parent_info["found"]:
                        parent_info["inherited_from"] = parent_info.get(
                            "inherited_from", base_class
                        )
                        return parent_info

                break  # Выходим после проверки глобальной области

        return result

    def is_dict_literal(self, content: str) -> bool:
        """Определяет, является ли содержимое фигурных скобок словарем (а не множеством)"""
        if not content.strip():
            return False  # Пустые {} - это словарь, но для нашего случая вернем False

        # Проходим по содержимому, отслеживая контекст
        in_string = False
        string_char = None
        depth = 0  # глубина вложенных скобок
        has_colon = False
        has_comma = False

        i = 0
        while i < len(content):
            char = content[i]

            # Обработка строк
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
            elif in_string and char == string_char:
                # Проверяем, не экранирован ли символ
                if i > 0 and content[i - 1] == "\\":
                    pass  # экранированная кавычка, продолжаем
                else:
                    in_string = False
                    string_char = None

            # Обработка скобок (только вне строк)
            elif not in_string:
                if char in ["[", "{", "("]:
                    depth += 1
                elif char in ["]", "}", ")"]:
                    depth -= 1
                # Ищем двоеточие на верхнем уровне
                elif char == ":" and depth == 0:
                    has_colon = True
                elif char == "," and depth == 0:
                    has_comma = True

            i += 1

        # Если есть двоеточие на верхнем уровне - это словарь
        # Если нет двоеточия, но есть запятые и все элементы - не пары ключ:значение - это множество
        return has_colon
