from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
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
        """Генерирует выражение для создания кортежа"""
        items = tuple_ast.get("items", [])

        # Если tuple_type уже задан и является именем структуры (начинается с tuple_), не анализируем
        if tuple_type and tuple_type.startswith("tuple_"):
            # Это уже имя структуры, а не тип
            struct_name = tuple_type
            logger.debug(
                f"DEBUG generate_tuple_creation: struct_name={struct_name} (уже задано)"
            )
        else:
            if not tuple_type:
                # Определяем тип кортежа на основе элементов
                if items:
                    # Проверяем, все ли элементы одного типа
                    element_types = set()
                    for item in items:
                        if isinstance(item, dict):
                            if item.get("type") == "literal":
                                data_type = item.get("data_type", "int")
                                element_types.add(data_type)

                    if len(element_types) == 1:
                        element_type = next(iter(element_types))
                        tuple_type = f"tuple[{element_type}]"
                    else:
                        # Разные типы - используем фиксированный кортеж
                        element_types_list = []
                        for item in items:
                            if isinstance(item, dict) and item.get("type") == "literal":
                                data_type = item.get("data_type", "int")
                                element_types_list.append(data_type)

                        if element_types_list:
                            tuple_type = f"tuple[{', '.join(element_types_list)}]"
                        else:
                            tuple_type = "tuple[int]"
                else:
                    tuple_type = "tuple[int]"

            struct_name = self.generate_tuple_struct_name(tuple_type)
            logger.debug(
                f"DEBUG generate_tuple_creation: tuple_type={tuple_type}, struct_name={struct_name}"
            )

        if items:
            # Для универсального кортежа tuple[T]
            if "," not in tuple_type:  # tuple[int] (нет запятых)
                # Создаем временный массив
                temp_var = self.generate_temporary_var("array")

                # Генерируем элементы массива
                item_exprs = [self.generate_expression(item) for item in items]

                # Создаем массив
                self.add_line(f"int {temp_var}[{len(items)}] = {{")
                self.indent_level += 1
                for i, item_expr in enumerate(item_exprs):
                    self.add_line(f"{item_expr}{',' if i < len(items) - 1 else ''}")
                self.indent_level -= 1
                self.add_line("};")

                # Возвращаем вызов create_tuple_int
                return f"create_{struct_name}({temp_var}, {len(items)})"

            else:
                # Для фиксированного кортежа tuple[T1, T2, ...]
                item_exprs = [self.generate_expression(item) for item in items]
                return f"create_{struct_name}({', '.join(item_exprs)})"

        # Пустой кортеж
        return f"({struct_name}){{NULL, 0}}"

    def _generate_tuple_creation_direct(self, tuple_ast: Dict, base_name: str) -> str:
        """Генерирует создание кортежа напрямую, возвращая имя переменной с кортежем"""
        items = tuple_ast.get("items", [])

        if not items:
            return "NULL"

        # Определяем тип кортежа
        element_types = set()
        for item in items:
            if isinstance(item, dict):
                if item.get("type") == "literal":
                    data_type = item.get("data_type", "int")
                    element_types.add(data_type)

        if len(element_types) == 1:
            element_type = next(iter(element_types))
            tuple_type = f"tuple[{element_type}]"
        else:
            # По умолчанию int
            tuple_type = "tuple[int]"

        struct_name = self.generate_tuple_struct_name(tuple_type)

        # Создаем временный массив
        temp_array_name = f"{base_name}_arr"

        # Генерируем элементы массива
        item_exprs = [self.generate_expression(item) for item in items]

        # Создаем массив
        self.add_line(f"int {temp_array_name}[{len(items)}] = {{")
        self.indent_level += 1
        for i, item_expr in enumerate(item_exprs):
            self.add_line(f"{item_expr}{',' if i < len(items) - 1 else ''}")
        self.indent_level -= 1
        self.add_line("};")

        # Создаем кортеж и возвращаем его
        tuple_var_name = f"{base_name}_val"
        self.add_line(
            f"tuple_int {tuple_var_name} = create_{struct_name}({temp_array_name}, {len(items)});"
        )

        return tuple_var_name

    def generate_tuple_struct(self, py_type: str):
        """Генерирует структуру C для tuple типа"""
        if py_type in self.generated_structures:
            return

        self.generated_structures.add(py_type)

        match = re.match(r"tuple\[([^\]]+)\]", py_type)
        if not match:
            return

        inner = match.group(1)
        struct_name = self.generate_tuple_struct_name(py_type)
        element_type = self.map_type_to_c(inner)

        # Структура tuple
        struct_code = f"typedef struct {{\n"
        struct_code += f"    {element_type}* data;\n"
        struct_code += f"    int size;\n"
        struct_code += f"}} {struct_name};\n\n"

        self.generated_helpers.append(struct_code)

        # Функции для tuple
        functions = []

        # Создание
        functions.append(f"""
    {struct_name}* create_{struct_name}(const {element_type} arr[], int size) {{
        {struct_name}* t = malloc(sizeof({struct_name}));
        if (!t) {{
            fprintf(stderr, "Memory allocation failed for tuple\\n");
            exit(1);
        }}
        
        t->size = size;
        t->data = malloc(size * sizeof({element_type}));
        if (!t->data) {{
            fprintf(stderr, "Memory allocation failed for tuple data\\n");
            free(t);
            exit(1);
        }}
        
        for (int i = 0; i < size; i++) {{
            t->data[i] = arr[i];
        }}
        
        return t;
    }}
    """)

        # Получение элемента (read-only)
        functions.append(f"""
    {element_type} get_{struct_name}(const {struct_name}* t, int index) {{
        if (!t || index < 0 || index >= t->size) {{
            fprintf(stderr, "Index out of bounds in tuple\\n");
            exit(1);
        }}
        return t->data[index];
    }}
    """)

        # Длина
        functions.append(f"""
    int builtin_len_{struct_name}(const {struct_name}* t) {{
        if (!t) return 0;
        return t->size;
    }}
    """)

        # Срез (возвращает новый tuple)
        functions.append(f"""
    {struct_name}* slice_{struct_name}(const {struct_name}* t, int start, int stop, int step) {{
        if (!t) return NULL;
        
        // Нормализация индексов
        if (start < 0) start = t->size + start;
        if (stop < 0) stop = t->size + stop;
        if (start < 0) start = 0;
        if (stop > t->size) stop = t->size;
        
        // Вычисляем размер результата
        int new_size;
        if (step > 0) {{
            if (start >= stop) new_size = 0;
            else new_size = (stop - start + step - 1) / step;
        }} else if (step < 0) {{
            if (start <= stop) new_size = 0;
            else new_size = (start - stop - step - 1) / (-step);
        }} else {{
            fprintf(stderr, "ValueError: slice step cannot be zero\\n");
            exit(1);
        }}
        
        // Создаем новый tuple
        {struct_name}* result = malloc(sizeof({struct_name}));
        if (!result) {{
            fprintf(stderr, "Memory allocation failed for tuple slice\\n");
            exit(1);
        }}
        
        result->size = new_size;
        result->data = malloc(new_size * sizeof({element_type}));
        if (!result->data) {{
            fprintf(stderr, "Memory allocation failed for tuple slice data\\n");
            free(result);
            exit(1);
        }}
        
        // Копируем элементы
        int pos = 0;
        if (step > 0) {{
            for (int i = start; i < stop && pos < new_size; i += step) {{
                result->data[pos++] = t->data[i];
            }}
        }} else {{
            for (int i = start; i > stop && pos < new_size; i += step) {{
                result->data[pos++] = t->data[i];
            }}
        }}
        
        return result;
    }}
    """)

        # Освобождение памяти
        functions.append(f"""
    void free_{struct_name}({struct_name}* t) {{
        if (t) {{
            if (t->data) {{
                free(t->data);
            }}
            free(t);
        }}
    }}
    """)

        # Копирование (глубокая копия)
        functions.append(f"""
    {struct_name}* copy_{struct_name}(const {struct_name}* t) {{
        if (!t) return NULL;
        
        {struct_name}* copy = malloc(sizeof({struct_name}));
        if (!copy) {{
            fprintf(stderr, "Memory allocation failed for tuple copy\\n");
            exit(1);
        }}
        
        copy->size = t->size;
        copy->data = malloc(t->size * sizeof({element_type}));
        if (!copy->data) {{
            fprintf(stderr, "Memory allocation failed for tuple copy data\\n");
            free(copy);
            exit(1);
        }}
        
        for (int i = 0; i < t->size; i++) {{
            copy->data[i] = t->data[i];
        }}
        
        return copy;
    }}
    """)

        self.generated_helpers.extend(functions)

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
