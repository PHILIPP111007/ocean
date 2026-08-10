from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class OrchestratorMixin:
    def generate_from_json(self, json_data: List[Dict]) -> str:
        """Генерирует C код из JSON AST"""
        self.reset()
        logger.debug(f"Starting code generation with {len(json_data)} scopes")

        # 1. Извлекаем ВСЕ типы из AST
        all_types = self.extract_all_types_from_ast(json_data)
        logger.debug(f"Extracted types: {all_types}")

        # 2. Генерируем структуры для всех типов
        # Сортируем по глубине вложенности (сначала простые, потом сложные)
        sorted_types = sorted(all_types, key=lambda x: (x.count("["), x))

        # Сначала генерируем все list[...] структуры
        for py_type in sorted_types:
            if py_type.startswith("list["):
                logger.debug(f"Generating list structure for {py_type}")
                self.generate_list_struct(py_type)

        # Потом генерируем все dict[...] структуры (они могут зависеть от list)
        for py_type in sorted_types:
            if py_type.startswith("dict["):
                key_type, value_type = self._extract_dict_types(py_type)
                logger.debug(f"Generating dict structure for {py_type}")
                self.generate_dict_struct(key_type, value_type)

        # 3. Собираем импорты и объявления
        self.collect_imports_and_declarations(json_data)

        # 4. Генерируем заголовок с импортами
        self.generate_c_imports()

        # 5. Генерируем вспомогательные структуры и функции
        self.generate_helpers_section()

        # 6. Остальная генерация
        self.collect_class_fields_from_init_parameters(json_data)
        self.analyze_class_inheritance(json_data)
        self.analyze_classes(json_data)

        # 7. Генерируем структуры классов
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "class_declaration":
                        self.generate_class_declaration_with_fields(node)

        # 8. Генерируем forward declarations
        self.generate_forward_declarations()

        # 9. Генерируем конструкторы классов
        self.generate_class_constructors(json_data)

        # 10. Генерируем все методы классов
        self.generate_all_methods(json_data)

        # 11. Генерируем глобальные переменные
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "declaration":
                        self.generate_global_declaration(node)

        # 12. Генерируем код для каждой функции
        for scope in json_data:
            if scope.get("type") == "function" and not scope.get("is_stub", False):
                self.generate_function_scope(scope)

        return "\n".join(self.output)

    def generate_temporary_var(self, var_type: str = "int") -> str:
        """Генерирует имя временной переменной"""
        temp_name = f"temp_{self.temp_var_counter}"
        self.temp_var_counter += 1
        self.declare_variable(temp_name, var_type)
        return temp_name

    def generate_graph_node(self, node: Dict):
        """Генерирует код для узла графа"""
        node_type = node.get("node")

        if node_type == "declaration":
            self.generate_declaration(node)
        elif node_type == "redeclaration":  # ДОБАВЬТЕ ЭТО!
            self.generate_redeclaration(node)
        elif node_type == "delete":
            self.generate_delete(node)
        elif node_type == "assignment":
            self.generate_assignment(node)
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
            self.generate_class_declaration(node)
        elif node_type == "attribute_assignment":
            self.generate_attribute_assignment(node)
        elif node_type == "method_call":
            self.generate_method_call(node)
        elif node_type == "static_method_call":  # ДОБАВЬТЕ ЭТО!
            self.generate_object_method_call(node)
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
            logger.warning(f"Неизвестный тип узла: {node_type}")
            self.add_line(f"// Неизвестный узел: {node_type}")

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
