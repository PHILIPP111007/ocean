from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class ImportsMixin:
    def generate_c_imports(self):
        """Генерирует #include директивы"""
        for lib in DEFAULT_C_IMPORTS:
            self.add_line(lib)

        seen = set()
        for c_import in self.c_imports:
            header = c_import.get("header", "")
            is_system = c_import.get("is_system", True)

            if header and header not in seen:
                seen.add(header)
                if is_system:
                    self.add_line(f"#include <{header}>")
                else:
                    self.add_line(f'#include "{header}"')

        if seen:
            self.add_empty_line()

    def generate_forward_declarations(self):
        """Генерирует forward declarations функций"""
        if hasattr(self, "function_declarations") and self.function_declarations:
            # Удаляем дубликаты
            unique_declarations = []
            seen = set()

            for decl in self.function_declarations:
                # Нормализуем декларацию
                decl = decl.strip()
                if decl and decl not in seen:
                    seen.add(decl)
                    unique_declarations.append(decl)

            for decl in unique_declarations:
                self.add_line(decl)

            self.add_empty_line()

    def collect_imports_and_declarations(self, json_data: List[Dict]):
        """Собирает импорты и объявления функций из JSON"""
        self.c_imports = []
        self.function_declarations = []

        # Собираем импорты из module scope
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "c_import":
                        self.c_imports.append(node)

        # Собираем информацию о классах и их методах
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "class_declaration":
                        class_name = node.get("class_name", "")
                        methods = node.get("methods", [])

                        # Генерируем объявления методов
                        for method in methods:
                            if method.get("name") != "__init__":
                                method_name = method.get("name", "")
                                return_type = method.get("return_type", "void")

                                # Определяем C тип возвращаемого значения
                                if return_type.startswith("list["):
                                    self.generate_list_struct(return_type)
                                    struct_name = self.generate_list_struct_name(
                                        return_type
                                    )
                                    c_return_type = f"{struct_name}*"
                                elif return_type.startswith("tuple["):
                                    self.generate_tuple_struct(return_type)
                                    struct_name = self.generate_tuple_struct_name(
                                        return_type
                                    )
                                    c_return_type = f"{struct_name}*"
                                else:
                                    c_return_type = self.map_type_to_c(return_type)

                                params = method.get("parameters", [])

                                # Формируем параметры метода
                                param_decls = []
                                for i, param in enumerate(params):
                                    param_name = param.get("name", "")
                                    param_type = param.get("type", "int")

                                    if i == 0 and param_name == "self":
                                        param_decls.append(f"{class_name}* self")
                                    else:
                                        c_param_type = self.map_type_to_c(param_type)
                                        param_decls.append(
                                            f"{c_param_type} {param_name}"
                                        )

                                params_str = (
                                    ", ".join(param_decls) if param_decls else "void"
                                )
                                declaration = f"{c_return_type} {class_name}_{method_name}({params_str});"
                                self.function_declarations.append(declaration)

        # Добавляем объявление main
        self.function_declarations.append("int main(void);")
