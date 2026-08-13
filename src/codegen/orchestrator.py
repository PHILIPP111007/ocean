from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger
from src.typed_ir import TypedModule, build_typed_ir

class OrchestratorMixin:
    def generate_from_typed_ir(self, typed_ir) -> str:
        """Generate C from the canonical semantic IR."""
        if not isinstance(typed_ir, TypedModule):
            raise TypeError("generate_from_typed_ir expects a TypedModule")
        return self._generate_from_scopes(typed_ir.backend_scopes())

    def generate_from_json(self, json_data: List[Dict]) -> str:
        """Compatibility entry point for callers with parser JSON."""
        return self.generate_from_typed_ir(build_typed_ir(json_data))

    def _generate_from_scopes(self, json_data: List[Dict]) -> str:
        """Lower the compatibility scope view after typed IR is established."""
        self.reset()
        logger.debug(f"Starting Ocean C generation with {len(json_data)} scopes")
        self.scan_runtime_requirements(json_data)
        self.phils_function_names = {
            scope.get("function_name")
            for scope in json_data
            if scope.get("type") == "function" and scope.get("function_name")
        }
        self.function_parameters = {
            scope.get("function_name"): scope.get("parameters", [])
            for scope in json_data
            if scope.get("type") == "function" and scope.get("function_name")
        }

        # Register class names before mapping fields/generic containers that may
        # contain class references.
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "class_declaration":
                        name = node.get("class_name", "")
                        if name:
                            self.class_types.add(name)

        # Semantic class prepasses are non-emitting and must happen before the
        # helper section so every field type is known in time.
        self.build_class_registry(json_data)

        all_types = self.extract_all_types_from_ast(json_data)
        for model in self.class_registry.models.values():
            for field in model.fields.values():
                py_type = field.py_type
                all_types.add(py_type)
                self._add_nested_types(py_type, all_types)
        sorted_types = sorted(all_types, key=lambda x: (x.count("["), x))

        # The memory runtime is emitted only when the compilation unit actually
        # contains managed/reference data.  Pure numeric/C-FFI programs should
        # lower to plain C without dragging the Ocean heap runtime into the TU.
        managed_type_present = any(
            self.memory_kind_for_type(py_type) in {
                self.MEMORY_ARC,
                self.MEMORY_STRING,
            }
            for py_type in all_types
        ) or bool(self.class_types)
        if managed_type_present or self.runtime_needs_memory:
            self.generate_ownership_runtime()

        for py_type in sorted_types:
            if py_type.startswith("list["):
                self.generate_list_struct(py_type)
            elif self.is_array_type(py_type):
                self.generate_array_struct(py_type)
        for py_type in sorted_types:
            if py_type.startswith("tuple["):
                self.generate_tuple_struct(py_type)
        for py_type in sorted_types:
            if py_type.startswith("dict["):
                key_type, value_type = self._extract_dict_types(py_type)
                self.generate_dict_struct(key_type, value_type)

        # Declaration collection may instantiate return/parameter container
        # types, so it also precedes helper emission.
        self.collect_imports_and_declarations(json_data)
        self.generate_c_imports()
        self.generate_helpers_section()

        # Class layouts depend on the generated generic type declarations.
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "class_declaration":
                        self.generate_class_declaration_with_fields(node)

        self.generate_forward_declarations()
        self.generate_class_constructors(json_data)
        self.generate_all_methods(json_data)

        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "declaration":
                        self.generate_global_declaration(node)

        for scope in json_data:
            if scope.get("type") == "function" and not scope.get("is_stub", False):
                self.generate_function_scope(scope)

        c_code = "\n".join(self.output)
        return self.apply_ocean_namespace(c_code, json_data)

    def generate_temporary_var(self, var_type: str = "int") -> str:
        """Генерирует имя временной переменной"""
        temp_name = f"temp_{self.temp_var_counter}"
        self.temp_var_counter += 1
        self.declare_variable(temp_name, var_type)
        return temp_name

    def generate_graph_node(self, node: Dict):
        """Генерирует код для узла графа"""
        node_type = node.get("node")

        if self._contains_unsafe_ffi(node) and not node.get("unsafe", False):
            raise RuntimeError(
                "C FFI expressions require an explicit unsafe: block"
            )
        if (
            node_type in {"declaration", "redeclaration"}
            and self.is_raw_pointer_type(node.get("var_type", ""))
            and not node.get("unsafe", False)
        ):
            raise RuntimeError(
                "raw pointer declarations require an explicit unsafe: block"
            )

        if node_type == "declaration":
            self.generate_declaration(node)
        elif node_type == "redeclaration":  # ДОБАВЬТЕ ЭТО!
            self.generate_redeclaration(node)
        elif node_type == "delete":
            self.generate_delete(node)
        elif node_type == "assignment":
            self.generate_assignment(node)
        elif node_type == "augmented_assignment":
            self.generate_augmented_assignment(node)
        elif node_type == "function_call":
            self.generate_function_call(node)
        elif node_type == "builtin_function_call":  # НОВОЕ!
            self.generate_builtin_function_call(node)
        elif node_type == "builtin_function_call_assignment":  # НОВОЕ!
            self.generate_builtin_function_call_assignment(node)
        elif node_type == "return":
            self.generate_return(node)
        elif node_type == "while_loop":
            self.generate_while_loop(node)
        elif node_type == "if_statement":
            self.generate_if_statement(node)
        elif node_type == "for_loop":
            self.generate_for_loop(node)
        elif node_type == "break":  # НОВОЕ: обработка break
            self.generate_break(node)
        elif node_type == "continue":  # НОВОЕ: обработка continue
            self.generate_continue(node)
        elif node_type == "c_call":  # ДОБАВЬТЕ ЭТО
            self.generate_c_call(node)
        elif node_type == "class_declaration":
            # Class layouts are emitted by the semantic OOP prepass above.
            pass
        elif node_type == "attribute_assignment":
            self.generate_attribute_assignment(node)
        elif node_type == "method_call":
            self.generate_method_call(node)
        elif node_type == "static_method_call":  # ДОБАВЬТЕ ЭТО!
            device_tensor_expr = self._device_tensor_static_call(node)
            if device_tensor_expr is not None:
                self.add_line(f"(void){device_tensor_expr};")
                return
            class_name = node.get("class_name", "")
            method_name = node.get("method", "")
            args = node.get("arguments", [])
            arg_strings = [self.generate_expression(arg) for arg in args]
            self.add_line(f"{class_name}_{method_name}({', '.join(arg_strings)});")
        elif node_type in [
            "index_assignment",
            "nested_index_assignment",
        ]:  # НОВОЕ: присваивание по индексу
            self.generate_index_assignment(node)
        elif node_type == "slice_assignment":  # НОВОЕ: присваивание среза
            self.generate_slice_assignment(node)
        elif (
            node_type == "augmented_index_assignment"
        ):  # НОВОЕ: составное присваивание по индексу
            self.generate_augmented_index_assignment(node)
        elif node_type == "c_import":
            # Импорты уже обработаны на уровне модуля
            pass
        elif node_type == "function_declaration":
            # Объявления функций уже обработаны
            pass
        else:
            raise RuntimeError(f"unsupported AST node in safe backend: {node_type!r}")

    def _contains_unsafe_ffi(self, value) -> bool:
        """Return whether an AST value contains a direct ``@c_function(...)``."""
        if isinstance(value, str):
            return bool(re.search(r"@[A-Za-z_][A-Za-z0-9_]*\s*\(", value))
        if isinstance(value, dict):
            return any(self._contains_unsafe_ffi(child) for child in value.values())
        if isinstance(value, list):
            return any(self._contains_unsafe_ffi(child) for child in value)
        return False

    def generate_global_declaration(self, node: Dict):
        """Генерирует объявление глобальной переменной"""
        var_name = node.get("var_name", "")
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast")

        logger.debug(f"Генерация глобального объявления для {var_name}: {var_type}")

        # Объявляем переменную в глобальном scope
        self.declare_variable(var_name, var_type)
        var_info = self.get_variable_info(var_name)

        if not var_info:
            return

        c_type = var_info["c_type"]
        if var_info.get("memory_kind") in {self.MEMORY_ARC, self.MEMORY_STRING, self.MEMORY_OWNED}:
            raise RuntimeError(
                f"managed global '{var_name}' is not enabled in Ocean ownership v1; "
                "use a local owner or add an explicit module initializer"
            )

        # Если есть выражение инициализации
        if expression_ast:
            # Для глобальных переменных нужна константная инициализация
            if expression_ast.get("type") == "literal":
                expr = self.generate_expression(expression_ast)
                self.add_line(f"{c_type} {var_name} = {expr};")
            else:
                # Для не-литералов оставляем только объявление
                self.add_line(f"{c_type} {var_name};")
                # Инициализация будет в main
                self.global_init_nodes.append(node)
        else:
            # Объявление без инициализации
            if c_type.endswith("*"):
                self.add_line(f"{c_type} {var_name} = NULL;")
            else:
                self.add_line(f"{c_type} {var_name};")
