from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class DictCodegenMixin:
    def generate_dict_struct(self, key_type: str, value_type: str) -> str:
        """Generate an ARC-owned chained hash table."""
        key_c_type = self.map_type_to_c(key_type)
        value_c_type = self.map_type_to_c(value_type)
        key_name = self.clean_type_name_for_c(key_type)
        value_name = self.clean_type_name_for_c(value_type)
        struct_name = f"dict_{key_name}_{value_name}"

        if struct_name in self.generated_structures:
            return struct_name
        self.generated_structures.add(struct_name)

        # keys()/values() return regular Ocean lists whose element ownership
        # policy is generated from the same type information.
        self.generate_list_struct(f"list[{key_type}]")
        self.generate_list_struct(f"list[{value_type}]")

        is_key_string = key_type == "str" or key_c_type == "char*"
        is_value_arc = self.is_arc_type(value_type)
        is_value_string = value_type == "str" or value_c_type == "char*"
        bucket_count = 256
        destroy_name = f"ocean_destroy_{struct_name}"

        self.generated_helpers.append(f"""
typedef struct {struct_name}_node {{
    {key_c_type} key;
    {value_c_type} value;
    struct {struct_name}_node* next;
}} {struct_name}_node;

typedef struct {struct_name} {{
    ocean_object_header header;
    {struct_name}_node** buckets;
    int size;
    int bucket_count;
}} {struct_name};
""")
        self.generated_helpers.append(f"static void {destroy_name}(void* ptr);\n")

        if is_key_string:
            hash_body = f"""
static unsigned int hash_{struct_name}({key_c_type} key) {{
    if (!key) return 0;
    unsigned long hash = 5381;
    unsigned char c;
    while ((c = (unsigned char)*key++)) hash = ((hash << 5) + hash) + c;
    return (unsigned int)(hash % {bucket_count});
}}
"""
            key_equal = "strcmp(current->key, key) == 0"
            key_store = "ocean_strdup(key)"
            key_cleanup = "free(current->key);"
        else:
            hash_body = f"""
static unsigned int hash_{struct_name}({key_c_type} key) {{
    unsigned long long x = (unsigned long long)key;
    return (unsigned int)(x % {bucket_count});
}}
"""
            key_equal = "current->key == key"
            key_store = "key"
            key_cleanup = ""
        self.generated_helpers.append(hash_body)

        self.generated_helpers.append(f"""
{struct_name}* create_{struct_name}(int initial_capacity) {{
    (void)initial_capacity;
    {struct_name}* dict = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!dict) {{ fprintf(stderr, "Memory allocation failed for {struct_name}\\n"); exit(1); }}
    dict->bucket_count = {bucket_count};
    dict->buckets = ({struct_name}_node**)calloc((size_t)dict->bucket_count, sizeof({struct_name}_node*));
    if (!dict->buckets) {{ free(dict); fprintf(stderr, "Memory allocation failed for dict buckets\\n"); exit(1); }}
    dict->header.refcount = 1;
    dict->header.destroy = {destroy_name};
    return dict;
}}
""")

        if is_value_arc:
            value_prepare = "    ocean_retain(value);\n"
            replace_cleanup = "ocean_release(current->value);"
            node_cleanup = "ocean_release(current->value);"
        elif is_value_string:
            value_prepare = "    char* owned_value = ocean_strdup(value);\n"
            replace_cleanup = "free(current->value);"
            node_cleanup = "free(current->value);"
        else:
            value_prepare = ""
            replace_cleanup = ""
            node_cleanup = ""
        stored_value = "owned_value" if is_value_string else "value"

        self.generated_helpers.append(f"""
void set_{struct_name}({struct_name}* dict, {key_c_type} key, {value_c_type} value) {{
    if (!dict) {{ fprintf(stderr, "Null dict in set\\n"); exit(1); }}
    unsigned int index = hash_{struct_name}(key);
    {struct_name}_node* current = dict->buckets[index];
    while (current) {{
        if ({key_equal}) {{
{value_prepare}            {replace_cleanup}
            current->value = {stored_value};
            return;
        }}
        current = current->next;
    }}
{value_prepare}    {struct_name}_node* node = ({struct_name}_node*)calloc(1, sizeof({struct_name}_node));
    if (!node) {{
        {'ocean_release(value);' if is_value_arc else 'free(owned_value);' if is_value_string else ''}
        fprintf(stderr, "Memory allocation failed for dict node\\n"); exit(1);
    }}
    node->key = {key_store};
    if ({'node->key == NULL' if is_key_string else '0'}) {{
        {'ocean_release(value);' if is_value_arc else 'free(owned_value);' if is_value_string else ''}
        free(node); fprintf(stderr, "Memory allocation failed for dict key\\n"); exit(1);
    }}
    node->value = {stored_value};
    node->next = dict->buckets[index];
    dict->buckets[index] = node;
    dict->size += 1;
}}
""")

        self.generated_helpers.append(f"""
{value_c_type} get_{struct_name}(const {struct_name}* dict, {key_c_type} key) {{
    if (!dict) {{ fprintf(stderr, "KeyError: dict is NULL\\n"); exit(1); }}
    unsigned int index = hash_{struct_name}(key);
    {struct_name}_node* current = dict->buckets[index];
    while (current) {{
        if ({key_equal}) return current->value;
        current = current->next;
    }}
    fprintf(stderr, "KeyError: key not found in dict\\n");
    exit(1);
}}

{value_c_type} get_default_{struct_name}(const {struct_name}* dict, {key_c_type} key, {value_c_type} default_value) {{
    if (!dict) return default_value;
    unsigned int index = hash_{struct_name}(key);
    {struct_name}_node* current = dict->buckets[index];
    while (current) {{
        if ({key_equal}) return current->value;
        current = current->next;
    }}
    return default_value;
}}

int contains_{struct_name}(const {struct_name}* dict, {key_c_type} key) {{
    if (!dict) return 0;
    unsigned int index = hash_{struct_name}(key);
    {struct_name}_node* current = dict->buckets[index];
    while (current) {{
        if ({key_equal}) return 1;
        current = current->next;
    }}
    return 0;
}}
""")

        self.generated_helpers.append(f"""
void delete_{struct_name}({struct_name}* dict, {key_c_type} key) {{
    if (!dict) return;
    unsigned int index = hash_{struct_name}(key);
    {struct_name}_node* current = dict->buckets[index];
    {struct_name}_node* prev = NULL;
    while (current) {{
        if ({key_equal}) {{
            if (prev) prev->next = current->next;
            else dict->buckets[index] = current->next;
            {key_cleanup}
            {node_cleanup}
            free(current);
            dict->size -= 1;
            return;
        }}
        prev = current;
        current = current->next;
    }}
}}

static inline int len_{struct_name}(const {struct_name}* dict) {{
    return dict ? dict->size : 0;
}}
""")

        list_key = self.generate_list_struct_name(f"list[{key_type}]")
        list_value = self.generate_list_struct_name(f"list[{value_type}]")
        self.generated_helpers.append(f"""
{list_key}* keys_{struct_name}(const {struct_name}* dict) {{
    if (!dict) return NULL;
    {list_key}* out = create_{list_key}(dict->size);
    for (int i = 0; i < dict->bucket_count; ++i) {{
        for ({struct_name}_node* current = dict->buckets[i]; current; current = current->next) {{
            append_{list_key}(out, current->key);
        }}
    }}
    return out;
}}

{list_value}* values_{struct_name}(const {struct_name}* dict) {{
    if (!dict) return NULL;
    {list_value}* out = create_{list_value}(dict->size);
    for (int i = 0; i < dict->bucket_count; ++i) {{
        for ({struct_name}_node* current = dict->buckets[i]; current; current = current->next) {{
            append_{list_value}(out, current->value);
        }}
    }}
    return out;
}}
""")

        self.generated_helpers.append(f"""
static void {destroy_name}(void* ptr) {{
    {struct_name}* dict = ({struct_name}*)ptr;
    if (!dict) return;
    for (int i = 0; i < dict->bucket_count; ++i) {{
        {struct_name}_node* current = dict->buckets[i];
        while (current) {{
            {struct_name}_node* next = current->next;
            {key_cleanup}
            {node_cleanup}
            free(current);
            current = next;
        }}
    }}
    free(dict->buckets);
    free(dict);
}}

static inline void free_{struct_name}({struct_name}* dict) {{
    ocean_release(dict);
}}
""")

        for name in (
            f"create_{struct_name}", f"set_{struct_name}", f"get_{struct_name}",
            f"get_default_{struct_name}", f"contains_{struct_name}",
            f"delete_{struct_name}", f"len_{struct_name}", f"free_{struct_name}",
            f"keys_{struct_name}", f"values_{struct_name}",
        ):
            self.generated_functions.add(name)
        return struct_name

    def _generate_dict_declaration(
        self, var_name: str, var_type: str, value_ast: Dict, node: Dict
    ):
        """Генерирует объявление словаря"""
        # Извлекаем типы ключа и значения
        key_type, value_type = self._extract_dict_types(var_type)

        # Генерируем структуру для словаря (ВАЖНО: вызываем ДО объявления переменной)
        struct_name = self.generate_dict_struct(key_type, value_type)

        # Объявляем переменную
        self.declare_variable(var_name, var_type)

        # Получаем C тип для переменной
        c_type = f"{struct_name}*"

        if value_ast and value_ast.get("type") == "dict_literal":
            pairs = value_ast.get("pairs", {})

            # Создаем словарь
            self.add_line(
                f"{c_type} {var_name} = create_{struct_name}({max(len(pairs), 16)});"
            )

            # Добавляем элементы
            for key_str, value_node in pairs.items():
                # Обработка ключа
                if key_type == "str":
                    key_expr = f'"{key_str}"'
                elif key_type == "int":
                    key_expr = key_str
                else:
                    key_expr = key_str

                # Генерация значения
                temp_owned_value = None
                if value_node.get("type") == "list_literal" and value_type.startswith("list["):
                    items = value_node.get("items", [])
                    list_struct = self.generate_list_struct_name(value_type)
                    temp_var = f"ocean_dict_value_{self.temp_var_counter}"
                    self.temp_var_counter += 1
                    self.declare_variable(temp_var, value_type)
                    self.add_line(
                        f"{list_struct}* {temp_var} = create_{list_struct}({max(len(items), 4)});"
                    )
                    for item in items:
                        item_expr = self.generate_expression(item)
                        self.add_line(f"append_{list_struct}({temp_var}, {item_expr});")
                    value_expr = temp_var
                    temp_owned_value = temp_var
                else:
                    value_expr = self.generate_expression(value_node)

                self.add_line(
                    f"set_{struct_name}({var_name}, {key_expr}, {value_expr});"
                )
                if temp_owned_value:
                    # set_dict retained the reference; drop the temporary owner.
                    self.add_line(f"ocean_release({temp_owned_value});")
                    info = self.get_variable_info(temp_owned_value)
                    if info:
                        info["is_deleted"] = True
                        info["owns_reference"] = False
        else:
            # Пустой словарь
            self.add_line(f"{c_type} {var_name} = create_{struct_name}(16);")
