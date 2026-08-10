from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class DictCodegenMixin:
    def generate_dict_struct(self, key_type: str, value_type: str) -> str:
        """Генерирует структуру C для словаря с хеш-таблицей (O(1) доступ)"""
        key_c_type = self.map_type_to_c(key_type)
        value_c_type = self.map_type_to_c(value_type)

        # Очищаем имена для использования в идентификаторах
        key_name = self.clean_type_name_for_c(key_type)
        value_name = self.clean_type_name_for_c(value_type)
        struct_name = f"dict_{key_name}_{value_name}"

        logger.debug(
            f"generate_dict_struct: {struct_name} for {key_type} -> {value_type}"
        )

        # Проверяем, не генерировали ли уже
        if struct_name in self.generated_structures:
            logger.debug(f"  already generated")
            return struct_name

        self.generated_structures.add(struct_name)

        # Для ключей-строк нужно особое сравнение и хеш-функция
        is_key_string = key_type == "str" or key_c_type == "char*"

        # Размер таблицы по умолчанию
        DEFAULT_HASH_SIZE = 256

        # Определяем хеш-функцию в зависимости от типа ключа
        if is_key_string:
            hash_func_name = f"hash_{struct_name}"
            hash_func = f"""
        static unsigned int {hash_func_name}({key_c_type} key) {{
            if (!key) return 0;
            unsigned int hash = 5381;
            int c;
            while ((c = *key++)) {{
                hash = ((hash << 5) + hash) + c;
            }}
            return hash % {DEFAULT_HASH_SIZE};
        }}
        """
            self.generated_helpers.append(hash_func)
            logger.debug(f"  added hash function for strings")
        else:
            # Для целочисленных ключей - простой модуль с обработкой отрицательных
            hash_func_name = f"hash_{struct_name}"
            hash_func = f"""
        static unsigned int {hash_func_name}({key_c_type} key) {{
            // Обрабатываем отрицательные ключи
            unsigned int ukey = (key < 0) ? -key : key;
            return ukey % {DEFAULT_HASH_SIZE};
        }}
        """
            self.generated_helpers.append(hash_func)
            logger.debug(f"  added hash function for integers")

        # Узел для цепочек коллизий
        node_struct = f"""typedef struct {struct_name}_node {{
            {key_c_type} key;
            {value_c_type} value;
            struct {struct_name}_node* next;
        }} {struct_name}_node;
        """
        self.generated_helpers.append(node_struct)
        logger.debug(f"  added node struct")

        # Структура для словаря (хеш-таблица)
        dict_struct = f"""typedef struct {{
            {struct_name}_node** buckets;
            int size;
            int capacity;
            int bucket_count;
        }} {struct_name};
        """
        self.generated_helpers.append(dict_struct)
        logger.debug(f"  added dict struct")

        # Функция создания словаря
        create_func = f"""
        {struct_name}* create_{struct_name}(int initial_capacity) {{
            (void)initial_capacity;  // Параметр не используется, оставлен для совместимости
            {struct_name}* dict = malloc(sizeof({struct_name}));
            if (!dict) {{
                fprintf(stderr, "Memory allocation failed for dict\\n");
                exit(1);
            }}
            
            dict->bucket_count = {DEFAULT_HASH_SIZE};
            dict->buckets = ({struct_name}_node**)calloc(dict->bucket_count, sizeof({struct_name}_node*));
            if (!dict->buckets) {{
                fprintf(stderr, "Memory allocation failed for dict buckets\\n");
                free(dict);
                exit(1);
            }}
            
            dict->size = 0;
            dict->capacity = {DEFAULT_HASH_SIZE};
            return dict;
        }}
        """
        self.generated_helpers.append(create_func)
        logger.debug(f"  added create function")

        # Функция установки значения (с хешированием)
        if is_key_string:
            set_func = f"""
        void set_{struct_name}({struct_name}* dict, {key_c_type} key, {value_c_type} value) {{
            if (!dict) return;
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            
            // Ищем существующий ключ
            while (current) {{
                if (strcmp(current->key, key) == 0) {{
                    current->value = value;
                    return;
                }}
                current = current->next;
            }}
            
            // Если ключ не найден, добавляем новый узел
            {struct_name}_node* new_node = malloc(sizeof({struct_name}_node));
            if (!new_node) {{
                fprintf(stderr, "Memory allocation failed for dict node\\n");
                exit(1);
            }}
            
            new_node->key = malloc(strlen(key) + 1);
            if (!new_node->key) {{
                fprintf(stderr, "Memory allocation failed for dict key\\n");
                free(new_node);
                exit(1);
            }}
            strcpy(new_node->key, key);
            new_node->value = value;
            new_node->next = dict->buckets[index];
            dict->buckets[index] = new_node;
            dict->size++;
        }}
        """
        else:
            set_func = f"""
        void set_{struct_name}({struct_name}* dict, {key_c_type} key, {value_c_type} value) {{
            if (!dict) return;
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            
            // Ищем существующий ключ
            while (current) {{
                if (current->key == key) {{
                    current->value = value;
                    return;
                }}
                current = current->next;
            }}
            
            // Если ключ не найден, добавляем новый узел
            {struct_name}_node* new_node = malloc(sizeof({struct_name}_node));
            if (!new_node) {{
                fprintf(stderr, "Memory allocation failed for dict node\\n");
                exit(1);
            }}
            
            new_node->key = key;
            new_node->value = value;
            new_node->next = dict->buckets[index];
            dict->buckets[index] = new_node;
            dict->size++;
        }}
        """
        self.generated_helpers.append(set_func)
        logger.debug(f"  added set function")

        # Функция получения значения (O(1) в среднем)
        if is_key_string:
            get_func = f"""
        {value_c_type} get_{struct_name}({struct_name}* dict, {key_c_type} key) {{
            if (!dict) {{
                fprintf(stderr, "KeyError: dict is NULL\\n");
                exit(1);
            }}
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            
            while (current) {{
                if (strcmp(current->key, key) == 0) {{
                    return current->value;
                }}
                current = current->next;
            }}
            
            fprintf(stderr, "KeyError: key not found in dict\\n");
            exit(1);
        }}
        """
        else:
            get_func = f"""
        {value_c_type} get_{struct_name}({struct_name}* dict, {key_c_type} key) {{
            if (!dict) {{
                fprintf(stderr, "KeyError: dict is NULL\\n");
                exit(1);
            }}
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            
            while (current) {{
                if (current->key == key) {{
                    return current->value;
                }}
                current = current->next;
            }}
            
            fprintf(stderr, "KeyError: key not found in dict\\n");
            exit(1);
        }}
        """
        self.generated_helpers.append(get_func)
        logger.debug(f"  added get function")

        # Функция проверки наличия ключа (O(1) в среднем)
        if is_key_string:
            contains_func = f"""
        int contains_{struct_name}({struct_name}* dict, {key_c_type} key) {{
            if (!dict) return 0;
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            
            while (current) {{
                if (strcmp(current->key, key) == 0) {{
                    return 1;
                }}
                current = current->next;
            }}
            return 0;
        }}
        """
        else:
            contains_func = f"""
        int contains_{struct_name}({struct_name}* dict, {key_c_type} key) {{
            if (!dict) return 0;
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            
            while (current) {{
                if (current->key == key) {{
                    return 1;
                }}
                current = current->next;
            }}
            return 0;
        }}
        """
        self.generated_helpers.append(contains_func)
        logger.debug(f"  added contains function")

        # Функция get с значением по умолчанию (аналог Python dict.get)
        if is_key_string:
            get_default_func = f"""
            {value_c_type} get_default_{struct_name}({struct_name}* dict, {key_c_type} key, {value_c_type} default_value) {{
                if (!dict) return default_value;
                unsigned int index = hash_{struct_name}(key);
                {struct_name}_node* current = dict->buckets[index];
                
                while (current) {{
                    if (strcmp(current->key, key) == 0) {{
                        return current->value;
                    }}
                    current = current->next;
                }}
                return default_value;
            }}
            """
        else:
            get_default_func = f"""
            {value_c_type} get_default_{struct_name}({struct_name}* dict, {key_c_type} key, {value_c_type} default_value) {{
                if (!dict) return default_value;
                unsigned int index = hash_{struct_name}(key);
                {struct_name}_node* current = dict->buckets[index];
                
                while (current) {{
                    if (current->key == key) {{
                        return current->value;
                    }}
                    current = current->next;
                }}
                return default_value;
            }}
            """
        self.generated_helpers.append(get_default_func)
        logger.debug(f"  added get_default function")
        self.generated_functions.add(f"get_default_{struct_name}")

        # Функция удаления ключа
        if is_key_string:
            delete_func = f"""
        void delete_{struct_name}({struct_name}* dict, {key_c_type} key) {{
            if (!dict) return;
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            {struct_name}_node* prev = NULL;
            
            while (current) {{
                if (strcmp(current->key, key) == 0) {{
                    if (prev) {{
                        prev->next = current->next;
                    }} else {{
                        dict->buckets[index] = current->next;
                    }}
                    free(current->key);
                    free(current);
                    dict->size--;
                    return;
                }}
                prev = current;
                current = current->next;
            }}
        }}
        """
        else:
            delete_func = f"""
        void delete_{struct_name}({struct_name}* dict, {key_c_type} key) {{
            if (!dict) return;
            unsigned int index = hash_{struct_name}(key);
            {struct_name}_node* current = dict->buckets[index];
            {struct_name}_node* prev = NULL;
            
            while (current) {{
                if (current->key == key) {{
                    if (prev) {{
                        prev->next = current->next;
                    }} else {{
                        dict->buckets[index] = current->next;
                    }}
                    free(current);
                    dict->size--;
                    return;
                }}
                prev = current;
                current = current->next;
            }}
        }}
        """
        self.generated_helpers.append(delete_func)
        logger.debug(f"  added delete function")

        # Функция размера
        len_func = f"""
        int len_{struct_name}({struct_name}* dict) {{
            return dict ? dict->size : 0;
        }}
        """
        self.generated_helpers.append(len_func)
        logger.debug(f"  added len function")

        # Функция освобождения памяти
        if is_key_string:
            free_func = f"""
        void free_{struct_name}({struct_name}* dict) {{
            if (dict) {{
                for (int i = 0; i < dict->bucket_count; i++) {{
                    {struct_name}_node* current = dict->buckets[i];
                    while (current) {{
                        {struct_name}_node* next = current->next;
                        free(current->key);
                        free(current);
                        current = next;
                    }}
                }}
                free(dict->buckets);
                free(dict);
            }}
        }}
        """
        else:
            free_func = f"""
        void free_{struct_name}({struct_name}* dict) {{
            if (dict) {{
                for (int i = 0; i < dict->bucket_count; i++) {{
                    {struct_name}_node* current = dict->buckets[i];
                    while (current) {{
                        {struct_name}_node* next = current->next;
                        free(current);
                        current = next;
                    }}
                }}
                free(dict->buckets);
                free(dict);
            }}
        }}
        """
        self.generated_helpers.append(free_func)
        logger.debug(f"  added free function")

        # Функция получения всех ключей (для совместимости)
        list_struct_name = f"list_{key_name}"
        keys_func = f"""
        {list_struct_name}* keys_{struct_name}({struct_name}* dict) {{
            if (!dict) return NULL;
            {list_struct_name}* result = create_{list_struct_name}(dict->size);
            for (int i = 0; i < dict->bucket_count; i++) {{
                {struct_name}_node* current = dict->buckets[i];
                while (current) {{
                    append_{list_struct_name}(result, current->key);
                    current = current->next;
                }}
            }}
            return result;
        }}
        """
        self.generated_helpers.append(keys_func)
        logger.debug(f"  added keys function")

        # Функция получения всех значений
        list_value_struct = f"list_{value_name}"
        values_func = f"""
        {list_value_struct}* values_{struct_name}({struct_name}* dict) {{
            if (!dict) return NULL;
            {list_value_struct}* result = create_{list_value_struct}(dict->size);
            for (int i = 0; i < dict->bucket_count; i++) {{
                {struct_name}_node* current = dict->buckets[i];
                while (current) {{
                    append_{list_value_struct}(result, current->value);
                    current = current->next;
                }}
            }}
            return result;
        }}
        """
        self.generated_helpers.append(values_func)
        logger.debug(f"  added values function")

        logger.debug(f"  total helpers now: {len(self.generated_helpers)}")
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
                if value_node.get("type") == "list_literal":
                    # Для списков нужно создать временную переменную
                    items = value_node.get("items", [])
                    list_struct = self.generate_list_struct_name(f"list[{value_type}]")
                    temp_var = self.generate_temporary_var("list")
                    self.add_line(
                        f"{list_struct}* {temp_var} = create_{list_struct}({len(items)});"
                    )

                    for item in items:
                        item_expr = self.generate_expression(item)
                        self.add_line(f"append_{list_struct}({temp_var}, {item_expr});")

                    value_expr = temp_var
                else:
                    value_expr = self.generate_expression(value_node)

                self.add_line(
                    f"set_{struct_name}({var_name}, {key_expr}, {value_expr});"
                )
        else:
            # Пустой словарь
            self.add_line(f"{c_type} {var_name} = create_{struct_name}(16);")
