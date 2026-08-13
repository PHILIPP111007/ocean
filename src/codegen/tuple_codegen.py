from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class TupleCodegenMixin:
    def generate_tuple_struct_name(self, py_type: str) -> str:
        """Генерирует имя структуры для tuple"""
        # Извлекаем содержимое скобок
        match = re.match(r"tuple\[([^\]]+)\]", py_type)
        if not match:
            # Если не можем распарсить, возвращаем очищенное имя
            return f"tuple_{self.clean_type_name_for_c(py_type)}"

        inner = match.group(1)

        # Если это tuple[T] (один тип)
        if "," not in inner:
            return f"tuple_{self.clean_type_name_for_c(inner)}"

        # Если это tuple[T1, T2, ...]
        # Заменяем запятые на подчеркивания и убираем пробелы
        clean_inner = self.clean_type_name_for_c(
            inner.replace(",", "_").replace(" ", "")
        )
        return f"tuple_{clean_inner}"

    def generate_tuple_creation(self, tuple_ast: Dict, tuple_type: str = None) -> str:
        """Create a homogeneous immutable tuple with owned element references."""
        items = tuple_ast.get("items", [])
        if not tuple_type:
            element_type = "int"
            if items and isinstance(items[0], Mapping):
                element_type = items[0].get("data_type", "int")
            tuple_type = f"tuple[{element_type}]"
        if tuple_type.startswith("tuple_"):
            struct_name = tuple_type
            # This legacy path does not carry a Phils element type; int is the
            # only representation we can prove from the old AST contract.
            element_py_type = "int"
        else:
            inner = re.match(r"tuple\[([^\]]+)\]", tuple_type)
            if not inner or "," in inner.group(1):
                raise RuntimeError("heterogeneous tuples are not enabled in Ocean ownership v1")
            element_py_type = inner.group(1).strip()
            struct_name = self.generate_tuple_struct_name(tuple_type)
            self.generate_tuple_struct(tuple_type)

        if not items:
            return f"create_{struct_name}(NULL, 0)"

        c_element_type = self.map_type_to_c(element_py_type)
        temp_var = f"ocean_tuple_items_{self.temp_var_counter}"
        self.temp_var_counter += 1
        item_exprs = [self.generate_expression(item) for item in items]
        self.add_line(f"{c_element_type} {temp_var}[{len(items)}] = {{")
        self.indent_level += 1
        for i, item_expr in enumerate(item_exprs):
            self.add_line(f"{item_expr}{',' if i < len(items) - 1 else ''}")
        self.indent_level -= 1
        self.add_line("};")
        return f"create_{struct_name}({temp_var}, {len(items)})"

    def _generate_tuple_creation_direct(self, tuple_ast: Dict, base_name: str) -> str:
        items = tuple_ast.get("items", [])
        if not items:
            return "NULL"
        first = items[0]
        element_type = first.get("data_type", "int") if isinstance(first, Mapping) else "int"
        tuple_type = f"tuple[{element_type}]"
        struct_name = self.generate_tuple_struct_name(tuple_type)
        self.generate_tuple_struct(tuple_type)
        c_type = self.map_type_to_c(element_type)
        temp_array_name = f"{base_name}_arr"
        item_exprs = [self.generate_expression(item) for item in items]
        self.add_line(f"{c_type} {temp_array_name}[{len(items)}] = {{")
        self.indent_level += 1
        for i, item_expr in enumerate(item_exprs):
            self.add_line(f"{item_expr}{',' if i < len(items) - 1 else ''}")
        self.indent_level -= 1
        self.add_line("};")
        tuple_var_name = f"{base_name}_val"
        self.add_line(
            f"{struct_name}* {tuple_var_name} = create_{struct_name}({temp_array_name}, {len(items)});"
        )
        return tuple_var_name

    def generate_tuple_struct(self, py_type: str):
        """Generate an ARC-owned homogeneous tuple[T]."""
        struct_name = self.generate_tuple_struct_name(py_type)
        if struct_name in self.generated_structures:
            return
        match = re.match(r"tuple\[([^\]]+)\]", py_type)
        if not match:
            return
        inner = match.group(1).strip()
        if "," in inner:
            raise RuntimeError("heterogeneous tuples are not enabled in Ocean ownership v1")

        self.generated_structures.add(struct_name)
        element_type = self.map_type_to_c(inner)
        is_arc = self.is_arc_type(inner)
        is_string = inner == "str" or element_type == "char*"
        destroy = f"ocean_destroy_{struct_name}"

        self.generated_helpers.append(f"""
typedef struct {struct_name} {{
    ocean_object_header header;
    {element_type}* data;
    int size;
}} {struct_name};
""")
        self.generated_helpers.append(f"static void {destroy}(void* ptr);\n")

        if is_arc:
            create_assign = "        ocean_retain(arr[i]);\n        t->data[i] = arr[i];"
            cleanup = "    for (int i = 0; i < t->size; ++i) ocean_release(t->data[i]);\n"
        elif is_string:
            create_assign = "        t->data[i] = ocean_strdup(arr[i]);"
            cleanup = "    for (int i = 0; i < t->size; ++i) free(t->data[i]);\n"
        else:
            create_assign = "        t->data[i] = arr[i];"
            cleanup = ""

        self.generated_helpers.append(f"""
{struct_name}* create_{struct_name}(const {element_type} arr[], int size) {{
    if (size < 0) {{ fprintf(stderr, "Invalid tuple size\\n"); exit(1); }}
    {struct_name}* t = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!t) {{ fprintf(stderr, "Memory allocation failed for tuple\\n"); exit(1); }}
    t->header.refcount = 1;
    t->header.destroy = {destroy};
    t->size = size;
    if (size > 0) {{
        t->data = ({element_type}*)calloc((size_t)size, sizeof({element_type}));
        if (!t->data) {{ free(t); fprintf(stderr, "Memory allocation failed for tuple data\\n"); exit(1); }}
    }}
    for (int i = 0; i < size; ++i) {{
{create_assign}
    }}
    return t;
}}

static inline {element_type} get_{struct_name}(const {struct_name}* t, int index) {{
    if (!t || index < 0 || index >= t->size) {{
        fprintf(stderr, "Index out of bounds in tuple\\n"); exit(1);
    }}
    return t->data[index];
}}

static inline int builtin_len_{struct_name}(const {struct_name}* t) {{
    return t ? t->size : 0;
}}

{struct_name}* copy_{struct_name}(const {struct_name}* t) {{
    return t ? create_{struct_name}(t->data, t->size) : NULL;
}}

{struct_name}* slice_{struct_name}(const {struct_name}* t, int start, int stop, int step) {{
    if (!t) return NULL;
    if (step == 0) {{ fprintf(stderr, "ValueError: slice step cannot be zero\\n"); exit(1); }}
    int n = t->size;
    if (start < 0) start += n;
    if (stop < 0) stop += n;
    if (step > 0) {{ if (start < 0) start = 0; if (stop > n) stop = n; }}
    else {{ if (start >= n) start = n - 1; if (stop < -1) stop = -1; }}
    int count = 0;
    if (step > 0) for (int i = start; i < stop; i += step) ++count;
    else for (int i = start; i > stop; i += step) ++count;
    {element_type}* tmp = count ? ({element_type}*)malloc((size_t)count * sizeof({element_type})) : NULL;
    if (count && !tmp) {{ fprintf(stderr, "Memory allocation failed for tuple slice\\n"); exit(1); }}
    int pos = 0;
    if (step > 0) for (int i = start; i < stop; i += step) tmp[pos++] = t->data[i];
    else for (int i = start; i > stop; i += step) tmp[pos++] = t->data[i];
    {struct_name}* out = create_{struct_name}(tmp, count);
    free(tmp);
    return out;
}}

static void {destroy}(void* ptr) {{
    {struct_name}* t = ({struct_name}*)ptr;
    if (!t) return;
{cleanup}    free(t->data);
    free(t);
}}

static inline void free_{struct_name}({struct_name}* t) {{
    ocean_release(t);
}}
""")
        for name in (
            f"create_{struct_name}", f"get_{struct_name}", f"builtin_len_{struct_name}",
            f"copy_{struct_name}", f"slice_{struct_name}", f"free_{struct_name}",
        ):
            self.generated_functions.add(name)

    def _generate_tuple_redeclaration(
        self, var_name: str, var_type: str, tuple_ast: Dict
    ):
        """Генерирует код для повторного объявления кортежа"""
        items = tuple_ast.get("items", [])

        # Генерируем структуру для кортежа если нужно
        self.generate_tuple_struct(var_type)
        struct_name = self.generate_tuple_struct_name(var_type)

        if items:
            # Создаем временный массив
            temp_array = f"temp_{var_name}"
            element_type = self.map_type_to_c("int")  # по умолчанию int

            self.add_line(f"{element_type} {temp_array}[{len(items)}] = {{")
            self.indent_level += 1
            for i, item_ast in enumerate(items):
                item_expr = self.generate_expression(item_ast)
                self.add_line(f"{item_expr}{',' if i < len(items) - 1 else ''}")
            self.indent_level -= 1
            self.add_line("};")

            # Создаем новый кортеж
            self.add_line(
                f"{var_name} = create_{struct_name}({temp_array}, {len(items)});"
            )
        else:
            # Пустой кортеж
            self.add_line(f"{var_name} = create_{struct_name}(NULL, 0);")
