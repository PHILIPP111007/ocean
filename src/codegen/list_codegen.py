from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class ListCodegenMixin:
    def generate_list_struct(self, py_type: str):
        """Генерирует структуру C для списка любой вложенности"""
        # Получаем полную информацию о типе
        type_info = self.extract_nested_type_info(py_type)

        if not type_info:
            logger.error(f"Не удалось получить информацию о типе {py_type}")
            return

        struct_name = type_info.get("struct_name")
        if not struct_name:
            logger.error(f"Нет struct_name для типа {py_type}")
            return

        logger.debug(f"generate_list_struct: {py_type}")
        logger.debug(f"  struct_name: {struct_name}")

        # Генерируем структуру только если еще не генерировали
        if struct_name not in self.generated_structures:
            self.generated_structures.add(struct_name)

            # Определяем element_type
            element_type = type_info.get("element_type", "void*")
            element_py_type = type_info.get("element_py_type")
            is_c_type = type_info.get("is_c_type", False)

            logger.debug(f"  element_type={element_type}, is_c_type={is_c_type}")

            # Создаем правильную структуру
            struct_code = f"typedef struct {struct_name} {{\n"
            struct_code += "    ocean_object_header header;\n"
            struct_code += f"    {element_type}* data;  // Указатель на массив элементов типа {element_type}\n"
            struct_code += f"    int size;\n"
            struct_code += f"    int capacity;\n"
            struct_code += f"}} {struct_name};\n\n"

            self.generated_helpers.append(struct_code)

            # ВСЕГДА вызываем _generate_list_functions, она сама решит как генерировать
            self._generate_list_functions(
                struct_name, element_type, element_py_type, is_c_type
            )
        else:
            logger.debug(f"Структура {struct_name} уже сгенерирована")

    def generate_list_struct_name(self, py_type: str) -> str:
        """Генерирует имя структуры для списка любой вложенности"""
        if not py_type.startswith("list["):
            # Если это уже базовый тип (например, pthread_t)
            clean_name = self.clean_type_name_for_c(py_type)
            # pthread_t -> pthread_t, Object* -> ObjectPtr
            if clean_name.endswith("*"):
                clean_name = clean_name[:-1] + "Ptr"
            return f"list_{clean_name}"

        # Используем уже существующий метод _generate_struct_name_recursive
        return self._generate_struct_name_recursive(py_type)

    def _generate_list_functions(
        self,
        struct_name: str,
        element_type: str,
        element_py_type: str = None,
        is_c_type: bool = False,
    ):
        """Генерирует функции для работы со списком (без дублирования)"""

        logger.debug(f"_generate_list_functions: struct_name={struct_name}")

        # Проверяем, не генерировали ли мы уже функции для этой структуры
        if struct_name in self.generated_functions:
            logger.debug(f"Функции для {struct_name} уже сгенерированы, пропускаем")
            return

        # Помечаем как сгенерированные
        self.generated_functions.add(struct_name)

        # Генерируем ВСЕ функции независимо от типа
        self._generate_standard_list_functions(
            struct_name, element_type, element_py_type
        )

        logger.debug(
            f"DEBUG _generate_list_functions: struct_name={struct_name}, element_type={element_type}, is_c_type={is_c_type}"
        )

    def _generate_standard_list_functions(
        self, struct_name: str, element_type: str, element_py_type: str = None
    ):
        """Generate a memory-safe ARC list implementation.

        Manual type-agnostic SIMD from the previous version is intentionally
        removed: copying 8 elements with AVX2 was only correct for 4-byte T.
        Clang/GCC can vectorize the value-type memcpy path safely.
        """
        is_arc_element = bool(element_py_type and self.is_arc_type(element_py_type))
        is_string_element = element_py_type == "str" or element_type == "char*"
        min_capacity = 4

        destroy_name = f"ocean_destroy_{struct_name}"
        self.generated_helpers.append(
            f"static void {destroy_name}(void* ptr);\n"
        )

        create_name = f"create_{struct_name}"
        if create_name not in self.generated_functions:
            self.generated_helpers.append(f"""
{struct_name}* {create_name}(int initial_capacity) {{
    if (initial_capacity < {min_capacity}) initial_capacity = {min_capacity};
    {struct_name}* list = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!list) {{
        fprintf(stderr, "Memory allocation failed for {struct_name}\\n");
        exit(1);
    }}
    list->data = ({element_type}*)calloc((size_t)initial_capacity, sizeof({element_type}));
    if (!list->data) {{
        free(list);
        fprintf(stderr, "Memory allocation failed for {struct_name} data\\n");
        exit(1);
    }}
    list->header.refcount = 1;
    list->header.destroy = {destroy_name};
    list->size = 0;
    list->capacity = initial_capacity;
    return list;
}}
""")
            self.generated_functions.add(create_name)

        append_name = f"append_{struct_name}"
        if append_name not in self.generated_functions:
            pre_store = ""
            value_expr = "value"
            if is_arc_element:
                pre_store = "    ocean_retain(value);\n"
            elif is_string_element:
                pre_store = "    char* owned_value = ocean_strdup(value);\n"
                value_expr = "owned_value"
            self.generated_helpers.append(f"""
void {append_name}({struct_name}* list, {element_type} value) {{
    if (!list) {{
        fprintf(stderr, "Null list in append\\n");
        exit(1);
    }}
    if (list->size >= list->capacity) {{
        int new_capacity = list->capacity > 0 ? list->capacity * 2 : {min_capacity};
        {element_type}* new_data = ({element_type}*)realloc(
            list->data, (size_t)new_capacity * sizeof({element_type})
        );
        if (!new_data) {{
            fprintf(stderr, "Memory reallocation failed for {struct_name}\\n");
            exit(1);
        }}
        list->data = new_data;
        list->capacity = new_capacity;
    }}
{pre_store}    list->data[list->size++] = {value_expr};
}}
""")
            self.generated_functions.add(append_name)

        extend_name = f"extend_{struct_name}"
        if extend_name not in self.generated_functions:
            self.generated_helpers.append(f"""
void {extend_name}({struct_name}* list, {element_type} const* values, int count) {{
    if (!list || !values || count <= 0) return;
    // Snapshot the source array before appending. This keeps self.extend(self)
    // safe even if append() reallocates list->data during the operation.
    {element_type}* snapshot = ({element_type}*)malloc((size_t)count * sizeof({element_type}));
    if (!snapshot) {{ fprintf(stderr, "Memory allocation failed in list.extend\\n"); exit(1); }}
    memcpy(snapshot, values, (size_t)count * sizeof({element_type}));
    for (int i = 0; i < count; ++i) {{
        {append_name}(list, snapshot[i]);
    }}
    free(snapshot);
}}
""")
            self.generated_functions.add(extend_name)

        len_name = f"builtin_len_{struct_name}"
        if len_name not in self.generated_functions:
            self.generated_helpers.append(f"""
static inline int {len_name}(const {struct_name}* list) {{
    return list ? list->size : 0;
}}
""")
            self.generated_functions.add(len_name)

        get_name = f"get_{struct_name}"
        if get_name not in self.generated_functions:
            self.generated_helpers.append(f"""
static inline {element_type} {get_name}(const {struct_name}* list, int index) {{
    if (!list || index < 0 || index >= list->size) {{
        fprintf(stderr, "Index out of bounds in {struct_name}\\n");
        exit(1);
    }}
    return list->data[index];
}}
""")
            self.generated_functions.add(get_name)

        set_name = f"set_{struct_name}"
        if set_name not in self.generated_functions:
            if is_arc_element:
                replace = """    ocean_retain(value);\n    ocean_release(list->data[index]);\n    list->data[index] = value;"""
            elif is_string_element:
                replace = """    char* copy = ocean_strdup(value);\n    free(list->data[index]);\n    list->data[index] = copy;"""
            else:
                replace = "    list->data[index] = value;"
            self.generated_helpers.append(f"""
static inline void {set_name}({struct_name}* list, int index, {element_type} value) {{
    if (!list || index < 0 || index >= list->size) {{
        fprintf(stderr, "Index out of bounds in {struct_name}\\n");
        exit(1);
    }}
{replace}
}}
""")
            self.generated_functions.add(set_name)

        insert_name = f"insert_{struct_name}"
        if insert_name not in self.generated_functions:
            if is_arc_element:
                prep = "    ocean_retain(value);\n"
                stored = "value"
            elif is_string_element:
                prep = "    char* owned_value = ocean_strdup(value);\n"
                stored = "owned_value"
            else:
                prep = ""
                stored = "value"
            self.generated_helpers.append(f"""
void {insert_name}({struct_name}* list, int index, {element_type} value) {{
    if (!list || index < 0 || index > list->size) {{
        fprintf(stderr, "Index out of bounds in list insert\\n"); exit(1);
    }}
    if (list->size >= list->capacity) {{
        int new_capacity = list->capacity > 0 ? list->capacity * 2 : {min_capacity};
        {element_type}* new_data = ({element_type}*)realloc(
            list->data, (size_t)new_capacity * sizeof({element_type})
        );
        if (!new_data) {{ fprintf(stderr, "Memory reallocation failed in insert\\n"); exit(1); }}
        list->data = new_data;
        list->capacity = new_capacity;
    }}
{prep}    for (int i = list->size; i > index; --i) list->data[i] = list->data[i - 1];
    list->data[index] = {stored};
    list->size += 1;
}}
""")
            self.generated_functions.add(insert_name)

        pop_name = f"pop_{struct_name}"
        if pop_name not in self.generated_functions:
            self.generated_helpers.append(f"""
{element_type} {pop_name}({struct_name}* list, int index) {{
    if (!list || list->size <= 0) {{ fprintf(stderr, "IndexError: pop from empty list\\n"); exit(1); }}
    if (index < 0) index += list->size;
    if (index < 0 || index >= list->size) {{ fprintf(stderr, "IndexError: pop index out of range\\n"); exit(1); }}
    {element_type} value = list->data[index];
    for (int i = index; i < list->size - 1; ++i) list->data[i] = list->data[i + 1];
    list->size -= 1;
    memset(&list->data[list->size], 0, sizeof({element_type}));
    // Ownership of reference/string elements is transferred to the caller.
    return value;
}}
""")
            self.generated_functions.add(pop_name)

        remove_name = f"remove_{struct_name}"
        if remove_name not in self.generated_functions:
            if is_string_element:
                equals = "strcmp(list->data[i], value) == 0"
                removed_cleanup = "free(removed);"
            elif is_arc_element:
                equals = "list->data[i] == value"
                removed_cleanup = "ocean_release(removed);"
            else:
                equals = "list->data[i] == value"
                removed_cleanup = "(void)removed;"
            self.generated_helpers.append(f"""
void {remove_name}({struct_name}* list, {element_type} value) {{
    if (!list) {{ fprintf(stderr, "Null list in remove\\n"); exit(1); }}
    for (int i = 0; i < list->size; ++i) {{
        if ({equals}) {{
            {element_type} removed = list->data[i];
            for (int j = i; j < list->size - 1; ++j) list->data[j] = list->data[j + 1];
            list->size -= 1;
            memset(&list->data[list->size], 0, sizeof({element_type}));
            {removed_cleanup}
            return;
        }}
    }}
    fprintf(stderr, "ValueError: list.remove(x): x not in list\\n");
    exit(1);
}}
""")
            self.generated_functions.add(remove_name)

        clear_name = f"clear_{struct_name}"
        if clear_name not in self.generated_functions:
            if is_arc_element:
                clear_cleanup = "    for (int i = 0; i < list->size; ++i) ocean_release(list->data[i]);\n"
            elif is_string_element:
                clear_cleanup = "    for (int i = 0; i < list->size; ++i) free(list->data[i]);\n"
            else:
                clear_cleanup = ""
            self.generated_helpers.append(f"""
void {clear_name}({struct_name}* list) {{
    if (!list) return;
{clear_cleanup}    if (list->data && list->size > 0) memset(list->data, 0, (size_t)list->size * sizeof({element_type}));
    list->size = 0;
}}
""")
            self.generated_functions.add(clear_name)

        reverse_name = f"reverse_{struct_name}"
        if reverse_name not in self.generated_functions:
            self.generated_helpers.append(f"""
void {reverse_name}({struct_name}* list) {{
    if (!list) return;
    for (int i = 0, j = list->size - 1; i < j; ++i, --j) {{
        {element_type} tmp = list->data[i];
        list->data[i] = list->data[j];
        list->data[j] = tmp;
    }}
}}
""")
            self.generated_functions.add(reverse_name)

        count_name = f"count_{struct_name}"
        index_name = f"index_{struct_name}"
        if is_string_element:
            eq_count = "strcmp(list->data[i], value) == 0"
        else:
            eq_count = "list->data[i] == value"
        if count_name not in self.generated_functions:
            self.generated_helpers.append(f"""
int {count_name}(const {struct_name}* list, {element_type} value) {{
    if (!list) return 0;
    int count = 0;
    for (int i = 0; i < list->size; ++i) if ({eq_count}) ++count;
    return count;
}}
""")
            self.generated_functions.add(count_name)
        if index_name not in self.generated_functions:
            self.generated_helpers.append(f"""
int {index_name}(const {struct_name}* list, {element_type} value) {{
    if (!list) {{ fprintf(stderr, "ValueError: value not in list\\n"); exit(1); }}
    for (int i = 0; i < list->size; ++i) if ({eq_count}) return i;
    fprintf(stderr, "ValueError: value not in list\\n"); exit(1);
}}
""")
            self.generated_functions.add(index_name)

        # Destruction owns one reference to every stored reference element.
        if is_arc_element:
            element_cleanup = """
    for (int i = 0; i < list->size; ++i) {
        ocean_release(list->data[i]);
    }
"""
        elif is_string_element:
            element_cleanup = """
    for (int i = 0; i < list->size; ++i) {
        free(list->data[i]);
    }
"""
        else:
            element_cleanup = ""
        self.generated_helpers.append(f"""
static void {destroy_name}(void* ptr) {{
    {struct_name}* list = ({struct_name}*)ptr;
    if (!list) return;
{element_cleanup}    free(list->data);
    free(list);
}}
""")

        free_name = f"free_{struct_name}"
        if free_name not in self.generated_functions:
            self.generated_helpers.append(f"""
static inline void {free_name}({struct_name}* list) {{
    ocean_release(list);
}}
""")
            self.generated_functions.add(free_name)

        slice_name = f"slice_{struct_name}"
        if slice_name not in self.generated_functions:
            self.generated_helpers.append(f"""
{struct_name}* {slice_name}(const {struct_name}* list, int start, int stop, int step) {{
    if (!list) return NULL;
    if (step == 0) {{
        fprintf(stderr, "ValueError: slice step cannot be zero\\n");
        exit(1);
    }}
    int n = list->size;
    if (start < 0) start += n;
    if (stop < 0) stop += n;
    if (step > 0) {{
        if (start < 0) start = 0;
        if (stop > n) stop = n;
    }} else {{
        if (start >= n) start = n - 1;
        if (stop < -1) stop = -1;
    }}
    {struct_name}* out = {create_name}(4);
    if (step > 0) {{
        for (int i = start; i < stop; i += step) {append_name}(out, list->data[i]);
    }} else {{
        for (int i = start; i > stop; i += step) {append_name}(out, list->data[i]);
    }}
    return out;
}}
""")
            self.generated_functions.add(slice_name)

        copy_name = f"copy_{struct_name}"
        if copy_name not in self.generated_functions:
            self.generated_helpers.append(f"""
{struct_name}* {copy_name}(const {struct_name}* src) {{
    if (!src) return NULL;
    {struct_name}* dst = {create_name}(src->size);
    for (int i = 0; i < src->size; ++i) {{
        {append_name}(dst, src->data[i]);
    }}
    return dst;
}}
""")
            self.generated_functions.add(copy_name)

    def _generate_all_list_functions(self):
        """Генерирует все функции для всех зарегистрированных структур списков"""
        # Собираем все структуры списков
        list_structures = []

        for helper in self.generated_helpers:
            if "typedef struct" in helper:
                # Извлекаем имя структуры
                lines = helper.split("\n")
                for line in lines:
                    if "} " in line and ";" in line and "list_" in line:
                        # Находим имя структуры
                        parts = line.strip().split()
                        for part in parts:
                            if part.endswith(";"):
                                struct_name = part[:-1]
                                if struct_name.startswith("list_"):
                                    list_structures.append(struct_name)
                                break

        # Удаляем дубликаты
        list_structures = list(set(list_structures))

        logger.debug(
            f"_generate_all_list_functions: Найдено структур: {list_structures}"
        )

        # Генерируем функции для каждой структуры
        for struct_name in list_structures:
            # Находим соответствующий helper чтобы определить element_type
            element_type = None
            for helper in self.generated_helpers:
                if f"}} {struct_name};" in helper:
                    # Парсим element_type из структуры
                    lines = helper.split("\n")
                    for line in lines:
                        if "* data;" in line and "//" in line:
                            # Пример: "    int* data;  // Указатель на массив элементов типа int"
                            parts = line.split("*")[0].strip()
                            element_type = parts
                            break
                    if element_type:
                        break

            if element_type:
                logger.debug(
                    f"Генерация функций для {struct_name} с element_type={element_type}"
                )
                # Определяем, является ли это C-типом
                is_c_type = self._is_c_type(element_type)

                # Генерируем ВСЕ функции
                self._generate_list_functions(
                    struct_name,
                    element_type,
                    element_type,  # element_py_type
                    is_c_type,
                )

    def _generate_nested_list_elements(
        self, parent_var: str, items: List, type_info: Dict, level: int
    ):
        """Рекурсивно генерирует элементы вложенного списка"""
        indent = "    " * (level + 1)  # Уровень вложенности для отступа

        if type_info["is_leaf"]:
            # Дошли до листовых элементов (int, float и т.д.)
            for i, item_ast in enumerate(items):
                item_expr = self.generate_expression(item_ast)
                self.add_line(
                    f"append_{type_info['struct_name']}({parent_var}, {item_expr});"
                )
            return

        # Еще есть вложенность
        for i, item_ast in enumerate(items):
            if item_ast.get("type") == "list_literal":
                # Создаем внутренний список
                inner_items = item_ast.get("items", [])
                inner_info = type_info["inner_info"]

                if not inner_info or not inner_info["struct_name"]:
                    logger.error(f"Нет информации о внутреннем типе на уровне {level}")
                    continue

                # Генерируем структуру для внутреннего типа
                self.generate_list_struct(inner_info["py_type"])

                # Создаем внутренний список
                temp_name = f"{parent_var}_l{level}_{i}"
                inner_struct_name = inner_info["struct_name"]
                inner_c_type = f"{inner_struct_name}*"

                self.add_line(
                    f"{inner_c_type} {temp_name} = create_{inner_struct_name}({max(len(inner_items), INITIAL_LIST_CAPACITY)});"
                )

                # Рекурсивно обрабатываем элементы внутреннего списка
                self._generate_nested_list_elements(
                    temp_name, inner_items, inner_info, level + 1
                )

                # Добавляем внутренний список в родительский
                self.add_line(
                    f"append_{type_info['struct_name']}({parent_var}, {temp_name});"
                )
            else:
                # Листовой элемент в промежуточном списке (должен быть list_literal)
                logger.error(
                    f"ERROR: Ожидался list_literal на уровне {level}, получено {item_ast.get('type')}"
                )

    def _generate_nested_list_elements_correctly(
        self, parent_var: str, items: List, type_info: Dict, level: int
    ):
        """Корректно генерирует элементы вложенного списка"""
        if not items:
            return

        struct_name = type_info.get("struct_name", "")
        if not struct_name:
            logger.error(f"Нет struct_name на уровне {level}")
            return

        logger.debug(f"generate_elements уровень {level}:")
        logger.debug(f"  parent_var: {parent_var}")
        logger.debug(f"  struct_name: {struct_name}")
        logger.debug(f"  is_leaf: {type_info.get('is_leaf')}")
        logger.debug(f"  element_type: {type_info.get('element_type')}")
        logger.debug(f"  items count: {len(items)}")

        # Проверяем, является ли текущий уровень листовым
        # is_leaf=True означает list[int] (элементы int)
        # is_leaf=False означает list[list[...]] (элементы указатели на списки)
        if type_info.get("is_leaf", True):
            logger.debug("  ЛИСТОВОЙ УРОВЕНЬ - добавляем простые элементы")
            for i, item_ast in enumerate(items):
                logger.debug(f"    элемент {i}: {item_ast.get('type')}")

                # Для кортежей используем специальную обработку
                if item_ast.get("type") == "tuple_literal":
                    # Создаем кортеж напрямую, без вызова generate_expression
                    tuple_expr = self._generate_tuple_creation_direct(
                        item_ast, f"{parent_var}_tuple_{i}"
                    )
                    self.add_line(f"append_{struct_name}({parent_var}, {tuple_expr});")
                else:
                    # Для других типов используем обычный generate_expression
                    item_expr = self.generate_expression(item_ast)
                    self.add_line(f"append_{struct_name}({parent_var}, {item_expr});")
            return

        # Есть вложенность - элементы это указатели на списки
        inner_info = type_info.get("inner_info")
        if not inner_info:
            logger.error(f"Нет информации о внутреннем типе на уровне {level}")
            return

        inner_struct_name = inner_info.get("struct_name", "")
        if not inner_struct_name:
            logger.error(f"Нет имени структуры для внутреннего типа на уровне {level}")
            return

        logger.debug(f"  ВЛОЖЕННЫЙ УРОВЕНЬ - создаем внутренние списки")
        logger.debug(f"  inner_struct_name: {inner_struct_name}")
        logger.debug(f"  inner_is_leaf: {inner_info.get('is_leaf')}")

        # Обрабатываем каждый элемент
        for i, item_ast in enumerate(items):
            logger.debug(f"  обработка элемента {i}: {item_ast.get('type')}")

            if item_ast.get("type") == "list_literal":
                # Создаем внутренний список
                inner_items = item_ast.get("items", [])
                temp_name = f"{parent_var}_l{level}_{i}"

                logger.debug(
                    f"    создаем {inner_struct_name}* {temp_name} с {len(inner_items)} элементами"
                )

                # Создаем внутренний список
                self.add_line(
                    f"{inner_struct_name}* {temp_name} = create_{inner_struct_name}({max(len(inner_items), INITIAL_LIST_CAPACITY)});"
                )

                # Рекурсивно обрабатываем элементы внутреннего списка
                logger.debug(f"    рекурсивный вызов для {temp_name}")
                self._generate_nested_list_elements_correctly(
                    temp_name, inner_items, inner_info, level + 1
                )

                # Добавляем внутренний список в родительский
                self.add_line(f"append_{struct_name}({parent_var}, {temp_name});")
            else:
                logger.warning(f"Не list_literal: {item_ast.get('type')}")
                # Если это уже созданная переменная, просто добавляем ее
                item_expr = self.generate_expression(item_ast)
                self.add_line(f"append_{struct_name}({parent_var}, {item_expr});")

    def _generate_list_redeclaration(
        self, var_name: str, var_type: str, list_ast: Dict
    ):
        """Генерирует код для повторного объявления списка"""
        items = list_ast.get("items", [])

        # Генерируем структуру для списка если нужно
        self.generate_list_struct(var_type)
        struct_name = self.generate_list_struct_name(var_type)

        # Создаем новый список
        self.add_line(
            f"{struct_name}* {var_name} = create_{struct_name}({max(len(items), INITIAL_LIST_CAPACITY)});"
        )

        # Добавляем элементы
        for item_ast in items:
            item_expr = self.generate_expression(item_ast)
            self.add_line(f"append_{struct_name}({var_name}, {item_expr});")
