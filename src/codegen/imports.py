from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class ImportsMixin:
    def generate_c_imports(self):
        """Генерирует #include директивы"""
        imports = list(DEFAULT_C_IMPORTS)
        if any(name.startswith(("ocean_array_", "ocean_tensor_")) for name in self.generated_structures):
            imports.extend(["#include <stdint.h>", "#include <stddef.h>"])
        for lib in imports:
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

    def collect_imports_and_declarations(self, scopes: List[Dict]):
        """Collect C imports and all forward declarations from semantic metadata."""
        self.c_imports = []
        self.function_declarations = []

        for scope in scopes:
            if scope.get("type") != "module":
                continue
            for node in scope.get("graph", []):
                if node.get("node") == "c_import":
                    self.c_imports.append(node)

        # ClassRegistry contains ordinary, static and class methods.  Using it
        # fixes missing prototypes for @staticmethod functions such as Json.parse.
        for class_name, model in self.class_registry.models.items():
            for method_name, method in model.methods.items():
                if method_name == "__init__":
                    continue

                c_return_type = self.map_type_to_c(method.return_type)
                param_decls = []
                for param in method.parameters:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "int")
                    if param_name == "self":
                        c_param_type = f"{class_name}*"
                    else:
                        c_param_type = self.map_type_to_c(param_type)
                    param_decls.append(f"{c_param_type} {param_name}")

                params_str = ", ".join(param_decls) if param_decls else "void"
                self.function_declarations.append(
                    f"{c_return_type} {class_name}_{method_name}({params_str});"
                )

        # Imported Ocean module functions can be emitted after their callers
        # (for example std/io.open), so they also require forward declarations.
        for scope in scopes:
            if scope.get("type") != "function" or scope.get("is_stub", False):
                continue

            function_name = scope.get("function_name", "")
            if not function_name:
                continue

            c_return_type = self.map_type_to_c(scope.get("return_type", "None"))
            param_decls = []
            for param in scope.get("parameters", []):
                param_name = param.get("name", "")
                param_type = param.get("type", "int")
                param_decls.append(
                    f"{self.map_type_to_c(param_type)} {param_name}"
                )

            params_str = ", ".join(param_decls) if param_decls else "void"
            self.function_declarations.append(
                f"{c_return_type} {function_name}({params_str});"
            )

        if not any(
            declaration.strip().startswith("int main(")
            for declaration in self.function_declarations
        ):
            self.function_declarations.append("int main(void);")
