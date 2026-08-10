from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class ScopeMixin:
    def enter_scope(self):
        """Вход в новый scope (увеличение вложенности)"""
        self.current_scope_level += 1
        if len(self.variable_scopes) <= self.current_scope_level:
            self.variable_scopes.append({})

    def exit_scope(self):
        """Выход из текущего scope"""
        if self.current_scope_level > 0:
            if len(self.variable_scopes) > self.current_scope_level:
                self.variable_scopes.pop()
            self.current_scope_level -= 1

    def get_current_scope(self) -> Dict:
        """Получает текущий scope переменных"""
        if self.current_scope_level < len(self.variable_scopes):
            return self.variable_scopes[self.current_scope_level]
        return {}

    def generate_function_scope(self, scope: Dict):
        """Генерирует код для функции"""
        func_name = scope.get("function_name", "")
        return_type = scope.get("return_type", "int")
        parameters = scope.get("parameters", [])

        logger.debug(f"generate_function_scope: {func_name}() -> {return_type}")

        # Входим в новый scope
        self.enter_scope()

        # Объявляем параметры
        param_decls = []
        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "int")
            c_param_type = self.map_type_to_c(param_type)
            param_decls.append(f"{c_param_type} {param_name}")
            self.declare_variable(param_name, param_type)

        # Сигнатура функции
        c_return_type = self.map_type_to_c(return_type)
        params_str = ", ".join(param_decls) if param_decls else "void"

        logger.debug(f"C return type for {return_type} is {c_return_type}")

        self.add_line(f"{c_return_type} {func_name}({params_str}) {{")
        self.indent_level += 1

        # Обрабатываем узлы графа
        processed_declarations = set()

        for node in scope.get("graph", []):
            node_type = node.get("node")

            if node_type == "declaration":
                var_name = node.get("var_name", "")
                if var_name not in processed_declarations:
                    self.generate_graph_node(node)
                    processed_declarations.add(var_name)
            else:
                self.generate_graph_node(node)

        self.indent_level -= 1
        self.add_line("}")
        self.add_empty_line()

        # Выходим из scope
        self.exit_scope()

    def declare_variable(self, name: str, var_type: str, is_pointer: bool = False):
        """Объявляет или обновляет переменную в текущем scope"""
        scope = self.get_current_scope()

        c_type = self.map_type_to_c(var_type, is_pointer)

        # Обновляем или создаем переменную
        scope[name] = {
            "c_type": c_type,
            "py_type": var_type,
            "is_pointer": is_pointer,
            "is_deleted": False,
            "delete_type": None,
        }

        logger.debug(f"Обновлена переменная '{name}': {var_type} -> {c_type}")

    def mark_variable_deleted(self, name: str, delete_type: str = "full") -> bool:
        """Помечает переменную как удаленную"""
        # Ищем переменную в текущем и родительских scope'ах
        for level in range(self.current_scope_level, -1, -1):
            if level < len(self.variable_scopes):
                scope = self.variable_scopes[level]
                if name in scope:
                    scope[name]["is_deleted"] = True
                    scope[name]["delete_type"] = delete_type
                    logger.debug(
                        f"DEBUG: Переменная '{name}' помечена как удаленная ({delete_type})"
                    )
                    return True
        logger.warning(f"Переменная '{name}' не найдена для удаления")
        return False

    def is_variable_declared(self, name: str) -> bool:
        """Проверяет, объявлена ли переменная (и не удалена ли она)"""
        for level in range(self.current_scope_level, -1, -1):
            if level < len(self.variable_scopes):
                if name in self.variable_scopes[level]:
                    var_info = self.variable_scopes[level][name]
                    # Проверяем, не удалена ли переменная
                    if not var_info.get("is_deleted", False):
                        return True
        return False

    def get_variable_info(self, name: str) -> Optional[Dict]:
        """Получает информацию о переменной (даже если она удалена)"""
        for level in range(self.current_scope_level, -1, -1):
            if level < len(self.variable_scopes):
                if name in self.variable_scopes[level]:
                    return self.variable_scopes[level][name]
        return None
