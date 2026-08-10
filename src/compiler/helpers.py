from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class HelpersMixin:
    def generate_helpers_section(self):
        """Генерирует секцию с вспомогательными функциями и структурами в правильном порядке"""

        # Сначала генерируем вспомогательные функции (они добавляются в self.generated_helpers)
        self.generate_sort_helpers()
        self.generate_string_helpers()
        self.generate_builtin_int_helpers()

        # Проверяем, есть ли что генерировать
        if not self.generated_helpers:
            logger.debug("No helpers to generate")
            return

        logger.debug(
            f"Generating helpers section with {len(self.generated_helpers)} helpers"
        )

        # Разделяем структуры и функции
        structures = []
        functions = []

        for helper in self.generated_helpers:
            if "typedef struct" in helper:
                structures.append(helper)
                logger.debug(f"Found structure: {helper[:50]}...")
            else:
                functions.append(helper)
                logger.debug(f"Found function: {helper[:50]}...")

        logger.debug(f"Structures: {len(structures)}, Functions: {len(functions)}")

        # Сортируем структуры по глубине вложенности
        def get_structure_depth(struct_code):
            lines = struct_code.split("\n")
            for line in lines:
                if "} " in line and ";" in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith(";"):
                            name = part[:-1]
                            # Для словарей тоже учитываем
                            if name.startswith("dict_"):
                                return 1
                            return name.count("list_")
            return 0

        structures.sort(key=get_structure_depth)

        # Добавляем заголовок
        self.add_line("// =========================================")
        self.add_line("// Вспомогательные структуры и функции")
        self.add_line("// =========================================")
        self.add_empty_line()

        # Добавляем все структуры
        for struct in structures:
            logger.debug(f"Adding structure to output")
            for line in struct.split("\n"):
                if line.strip():
                    self.add_line(line)
            self.add_empty_line()

        # Добавляем все функции
        for func in functions:
            logger.debug(f"Adding function to output")
            for line in func.split("\n"):
                if line.strip():
                    self.add_line(line)
            self.add_empty_line()

        logger.debug(f"Total helpers generated: {len(self.generated_helpers)}")

    def generate_sort_helpers(self):
        """Генерирует вспомогательные функции для сортировки"""
        helpers = []

        # Для целых чисел
        helpers.append("""
    int compare_int(const void* a, const void* b) {
        return (*(int*)a - *(int*)b);
    }
    """)

        # Для чисел с плавающей точкой
        helpers.append("""
    int compare_float(const void* a, const void* b) {
        float float_a = *(float*)a;
        float float_b = *(float*)b;
        if (float_a < float_b) return -1;
        if (float_a > float_b) return 1;
        return 0;
    }
    """)

        # Для double
        helpers.append("""
    int compare_double(const void* a, const void* b) {
        double double_a = *(double*)a;
        double double_b = *(double*)b;
        if (double_a < double_b) return -1;
        if (double_a > double_b) return 1;
        return 0;
    }
    """)

        # Для str
        helpers.append("""
    int compare_string(const void* a, const void* b) {
        return strcmp(*(const char**)a, *(const char**)b);
    }
    """)

        self.generated_helpers.extend(helpers)

    def generate_string_helpers(self):
        """Генерирует вспомогательные функции для работы со строками"""
        helpers = []

        if "list_str" not in self.generated_structures:
            # 1. Сначала определим структуру list_str
            helpers.append("""
            // Структура для списка строк (list[str])
            typedef struct {
                char** data;     // Массив указателей на строки
                int size;        // Текущее количество элементов
                int capacity;    // Вместимость массива
            } list_str;
            """)

        # 2. Функции для работы с list_str
        if "create_list_str" not in self.generated_functions:
            helpers.append("""
            // Создание списка строк
            list_str* create_list_str(int initial_capacity) {
                list_str* list = (list_str*)malloc(sizeof(list_str));
                if (!list) {
                    fprintf(stderr, "Memory allocation failed for list_str\\n");
                    exit(1);
                }
                list->data = (char**)malloc(initial_capacity * sizeof(char*));
                if (!list->data) {
                    fprintf(stderr, "Memory allocation failed for list_str data\\n");
                    free(list);
                    exit(1);
                }
                list->size = 0;
                list->capacity = initial_capacity;
                return list;
            }
            """)

        if "append_list_str" not in self.generated_functions:
            helpers.append("""
            // Добавление строки в список (создается копия строки)
            void append_list_str(list_str* list, const char* value) {
                if (!list || !value) return;
                
                if (list->size >= list->capacity) {
                    list->capacity = list->capacity == 0 ? 4 : list->capacity * 2;
                    list->data = (char**)realloc(list->data, list->capacity * sizeof(char*));
                    if (!list->data) {
                        fprintf(stderr, "Memory reallocation failed for list_str\\n");
                        exit(1);
                    }
                }
                
                // Создаем копию строки
                char* copy = (char*)malloc(strlen(value) + 1);
                if (!copy) {
                    fprintf(stderr, "Memory allocation failed for string copy\\n");
                    exit(1);
                }
                strcpy(copy, value);
                list->data[list->size] = copy;
                list->size++;
            }
            """)

        if "free_list_str" not in self.generated_functions:
            helpers.append("""
            // Освобождение памяти списка строк
            void free_list_str(list_str* list) {
                if (list) {
                    // Освобождаем все строки
                    for (int i = 0; i < list->size; i++) {
                        if (list->data[i]) {
                            free(list->data[i]);
                        }
                    }
                    // Освобождаем массив указателей
                    free(list->data);
                    // Освобождаем саму структуру
                    free(list);
                }
            }
            """)

        if "get_list_str" not in self.generated_functions:
            helpers.append("""
            // Получение элемента списка строк по индексу
            char* get_list_str(list_str* list, int index) {
                if (!list || index < 0 || index >= list->size) {
                    fprintf(stderr, "Index out of bounds in list_str\\n");
                    exit(1);
                }
                return list->data[index];
            }
            """)

        helpers.append("""
        // Функция преобразования int в строку
        char* int_to_string(int value) {
            // Определяем максимальную длину int (включая знак)
            char buffer[12]; // достаточно для 32-битного int
            sprintf(buffer, "%d", value);
            
            char* result = (char*)malloc(strlen(buffer) + 1);
            if (result == NULL) return NULL;
            
            strcpy(result, buffer);
            return result;
        }

        // Универсальная функция преобразования в строку
        char* builtin_str(int value) {
            return int_to_string(value);
        }
        """)

        if "set_list_str" not in self.generated_functions:
            helpers.append("""
            // Установка элемента списка строк по индексу
            void set_list_str(list_str* list, int index, const char* value) {
                if (!list || !value || index < 0 || index >= list->size) {
                    fprintf(stderr, "Index out of bounds in list_str\\n");
                    exit(1);
                }
                // Освобождаем старую строку
                if (list->data[index]) {
                    free(list->data[index]);
                }
                // Создаем копию новой строки
                char* copy = (char*)malloc(strlen(value) + 1);
                if (!copy) {
                    fprintf(stderr, "Memory allocation failed for string copy\\n");
                    exit(1);
                }
                strcpy(copy, value);
                list->data[index] = copy;
            }
            """)

        # 1. Функция upper
        helpers.append("""
    char* string_upper(const char* str) {
        if (!str) return NULL;
        int len = strlen(str);
        char* result = malloc(len + 1);
        if (!result) return NULL;
        for (int i = 0; i < len; i++) {
            if (str[i] >= 'a' && str[i] <= 'z') {
                result[i] = str[i] - 32;
            } else {
                result[i] = str[i];
            }
        }
        result[len] = '\\0';
        return result;
    }
    """)

        # 2. Функция lower
        helpers.append("""
    char* string_lower(const char* str) {
        if (!str) return NULL;
        int len = strlen(str);
        char* result = malloc(len + 1);
        if (!result) return NULL;
        for (int i = 0; i < len; i++) {
            if (str[i] >= 'A' && str[i] <= 'Z') {
                result[i] = str[i] + 32;
            } else {
                result[i] = str[i];
            }
        }
        result[len] = '\\0';
        return result;
    }
    """)

        # 3. Функция capitalize
        helpers.append("""
    char* string_capitalize(const char* str) {
        if (!str || strlen(str) == 0) return NULL;
        int len = strlen(str);
        char* result = malloc(len + 1);
        if (!result) return NULL;
        
        // Первый символ в верхний регистр
        if (str[0] >= 'a' && str[0] <= 'z') {
            result[0] = str[0] - 32;
        } else {
            result[0] = str[0];
        }
        
        // Остальные в нижний регистр
        for (int i = 1; i < len; i++) {
            if (str[i] >= 'A' && str[i] <= 'Z') {
                result[i] = str[i] + 32;
            } else {
                result[i] = str[i];
            }
        }
        result[len] = '\\0';
        return result;
    }
    """)

        # 4. Функция title
        helpers.append("""
    char* string_title(const char* str) {
        if (!str) return NULL;
        int len = strlen(str);
        char* result = malloc(len + 1);
        if (!result) return NULL;
        
        int new_word = 1;
        for (int i = 0; i < len; i++) {
            if (new_word && str[i] >= 'a' && str[i] <= 'z') {
                result[i] = str[i] - 32;
                new_word = 0;
            } else if (!new_word && str[i] >= 'A' && str[i] <= 'Z') {
                result[i] = str[i] + 32;
            } else {
                result[i] = str[i];
            }
            
            // Проверяем, начинается ли новое слово
            if (str[i] == ' ' || str[i] == '\\t' || str[i] == '\\n') {
                new_word = 1;
            }
        }
        result[len] = '\\0';
        return result;
    }
    """)

        # 5. Функция strip
        helpers.append("""
    char* string_strip(const char* str) {
        if (!str) return NULL;
        
        int start = 0;
        int end = strlen(str) - 1;
        
        // Находим начало без пробельных символов
        while (start <= end && (str[start] == ' ' || str[start] == '\\t' || str[start] == '\\n')) {
            start++;
        }
        
        // Находим конец без пробельных символов
        while (end >= start && (str[end] == ' ' || str[end] == '\\t' || str[end] == '\\n')) {
            end--;
        }
        
        int len = end - start + 1;
        char* result = malloc(len + 1);
        if (!result) return NULL;
        
        strncpy(result, str + start, len);
        result[len] = '\\0';
        return result;
    }
    """)

        # 6. Функция lstrip
        helpers.append("""
    char* string_lstrip(const char* str) {
        if (!str) return NULL;
        
        int start = 0;
        int len = strlen(str);
        
        // Находим начало без пробельных символов
        while (start < len && (str[start] == ' ' || str[start] == '\\t' || str[start] == '\\n')) {
            start++;
        }
        
        int result_len = len - start;
        char* result = malloc(result_len + 1);
        if (!result) return NULL;
        
        strcpy(result, str + start);
        return result;
    }
    """)

        # 7. Функция rstrip
        helpers.append("""
    char* string_rstrip(const char* str) {
        if (!str) return NULL;
        
        int end = strlen(str) - 1;
        
        // Находим конец без пробельных символов
        while (end >= 0 && (str[end] == ' ' || str[end] == '\\t' || str[end] == '\\n')) {
            end--;
        }
        
        int result_len = end + 1;
        char* result = malloc(result_len + 1);
        if (!result) return NULL;
        
        strncpy(result, str, result_len);
        result[result_len] = '\\0';
        return result;
    }
    """)

        # 8. Функция split - возвращает list_str
        helpers.append("""
        list_str* string_split(const char* str, const char* delimiter) {
            if (!str) return NULL;
            
            // Создаем список строк
            list_str* result = create_list_str(10);
            
            if (!delimiter || delimiter[0] == '\\0') {
                // Разделение по пробелам (по умолчанию)
                const char* start = str;
                const char* end = str;
                
                while (*end) {
                    if (*end == ' ' || *end == '\\t' || *end == '\\n') {
                        if (start != end) {
                            // Добавляем токен в список
                            int token_len = end - start;
                            char* token = malloc(token_len + 1);
                            if (!token) {
                                free_list_str(result);
                                return NULL;
                            }
                            strncpy(token, start, token_len);
                            token[token_len] = '\\0';
                            
                            append_list_str(result, token);
                        }
                        start = end + 1;
                    }
                    end++;
                }
                
                // Последний токен
                if (start != end) {
                    int token_len = end - start;
                    char* token = malloc(token_len + 1);
                    if (!token) {
                        free_list_str(result);
                        return NULL;
                    }
                    strncpy(token, start, token_len);
                    token[token_len] = '\\0';
                    
                    append_list_str(result, token);
                }
            } else {
                // Разделение по указанному разделителю
                int delim_len = strlen(delimiter);
                const char* start = str;
                const char* pos = strstr(start, delimiter);
                
                while (pos) {
                    int token_len = pos - start;
                    char* token = malloc(token_len + 1);
                    if (!token) {
                        free_list_str(result);
                        return NULL;
                    }
                    strncpy(token, start, token_len);
                    token[token_len] = '\\0';
                    
                    append_list_str(result, token);
                    
                    start = pos + delim_len;
                    pos = strstr(start, delimiter);
                }
                
                // Последний токен
                int token_len = strlen(start);
                if (token_len > 0) {
                    char* token = malloc(token_len + 1);
                    if (!token) {
                        free_list_str(result);
                        return NULL;
                    }
                    strcpy(token, start);
                    
                    append_list_str(result, token);
                }
            }
            
            return result;
        }
        """)

        # Альтернативная версия с оптимизацией для избежания множественных strcat
        helpers.append("""
        char* string_join(const char* separator, list_str* list) {
            if (!list || list->size == 0) {
                char* empty = malloc(1);
                if (!empty) return NULL;
                empty[0] = '\\0';
                return empty;
            }
            
            // Быстрая проверка для одного элемента
            if (list->size == 1) {
                char* result = malloc(strlen(list->data[0]) + 1);
                if (!result) return NULL;
                strcpy(result, list->data[0]);
                return result;
            }
            
            // Вычисляем длины всех элементов
            int* lengths = malloc(list->size * sizeof(int));
            if (!lengths) return NULL;
            
            int total_len = 0;
            int sep_len = strlen(separator);
            
            for (int i = 0; i < list->size; i++) {
                lengths[i] = strlen(list->data[i]);
                total_len += lengths[i];
            }
            total_len += (list->size - 1) * sep_len;
            
            // Выделяем память
            char* result = malloc(total_len + 1);
            if (!result) {
                free(lengths);
                return NULL;
            }
            
            // Собираем строку
            char* current = result;
            for (int i = 0; i < list->size; i++) {
                // Копируем элемент
                const char* item = list->data[i];
                int item_len = lengths[i];
                memcpy(current, item, item_len);
                current += item_len;
                
                // Добавляем разделитель
                if (i < list->size - 1) {
                    memcpy(current, separator, sep_len);
                    current += sep_len;
                }
            }
            
            result[total_len] = '\\0';
            free(lengths);
            return result;
        }
        """)

        # 10. Функция replace
        helpers.append("""
    char* string_replace(const char* str, const char* old, const char* new) {
        if (!str || !old || !new) return NULL;
        
        int str_len = strlen(str);
        int old_len = strlen(old);
        int new_len = strlen(new);
        
        // Считаем количество вхождений
        int count = 0;
        const char* pos = str;
        while ((pos = strstr(pos, old)) != NULL) {
            count++;
            pos += old_len;
        }
        
        // Вычисляем длину результата
        int result_len = str_len + count * (new_len - old_len);
        char* result = malloc(result_len + 1);
        if (!result) return NULL;
        
        // Заменяем
        const char* src = str;
        char* dest = result;
        while ((pos = strstr(src, old)) != NULL) {
            // Копируем часть до старой подстроки
            int copy_len = pos - src;
            strncpy(dest, src, copy_len);
            dest += copy_len;
            
            // Копируем новую подстроку
            strcpy(dest, new);
            dest += new_len;
            
            // Пропускаем старую подстроку
            src = pos + old_len;
        }
        
        // Копируем остаток
        strcpy(dest, src);
        
        return result;
    }
    """)

        # 11. Функция find
        helpers.append("""
    int string_find(const char* str, const char* sub) {
        if (!str || !sub) return -1;
        char* pos = strstr(str, sub);
        if (pos) {
            return pos - str;
        }
        return -1;
    }
    """)

        # 12. Функция index
        helpers.append("""
    int string_index(const char* str, const char* sub) {
        if (!str || !sub) return -1;
        char* pos = strstr(str, sub);
        if (!pos) {
            fprintf(stderr, "ValueError: substring not found\\n");
            exit(1);
        }
        return pos - str;
    }
    """)

        # 13. Функция count
        helpers.append("""
    int string_count(const char* str, const char* sub) {
        if (!str || !sub || sub[0] == '\\0') return 0;
        
        int count = 0;
        int sub_len = strlen(sub);
        const char* pos = str;
        
        while ((pos = strstr(pos, sub)) != NULL) {
            count++;
            pos += sub_len;
        }
        
        return count;
    }
    """)

        # 14. Функция startswith
        helpers.append("""
    bool string_startswith(const char* str, const char* prefix) {
        if (!str || !prefix) return false;
        return strncmp(str, prefix, strlen(prefix)) == 0;
    }
    """)

        # 15. Функция endswith
        helpers.append("""
    bool string_endswith(const char* str, const char* suffix) {
        if (!str || !suffix) return false;
        int str_len = strlen(str);
        int suffix_len = strlen(suffix);
        
        if (suffix_len > str_len) return false;
        return strcmp(str + str_len - suffix_len, suffix) == 0;
    }
    """)

        # 16. Функция isdigit
        helpers.append("""
    bool string_isdigit(const char* str) {
        if (!str || str[0] == '\\0') return false;
        
        for (int i = 0; str[i]; i++) {
            if (!(str[i] >= '0' && str[i] <= '9')) {
                return false;
            }
        }
        return true;
    }
    """)

        # 17. Функция isalpha
        helpers.append("""
    bool string_isalpha(const char* str) {
        if (!str || str[0] == '\\0') return false;
        
        for (int i = 0; str[i]; i++) {
            if (!((str[i] >= 'a' && str[i] <= 'z') || (str[i] >= 'A' && str[i] <= 'Z'))) {
                return false;
            }
        }
        return true;
    }
    """)

        # 18. Функция isalnum
        helpers.append("""
    bool string_isalnum(const char* str) {
        if (!str || str[0] == '\\0') return false;
        
        for (int i = 0; str[i]; i++) {
            char c = str[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
                return false;
            }
        }
        return true;
    }
    """)

        # 19. Функция islower
        helpers.append("""
    bool string_islower(const char* str) {
        if (!str || str[0] == '\\0') return false;
        
        int has_letter = 0;
        for (int i = 0; str[i]; i++) {
            if (str[i] >= 'A' && str[i] <= 'Z') {
                return false;
            }
            if (str[i] >= 'a' && str[i] <= 'z') {
                has_letter = 1;
            }
        }
        return has_letter;
    }
    """)

        # 20. Функция isupper
        helpers.append("""
    bool string_isupper(const char* str) {
        if (!str || str[0] == '\\0') return false;
        
        int has_letter = 0;
        for (int i = 0; str[i]; i++) {
            if (str[i] >= 'a' && str[i] <= 'z') {
                return false;
            }
            if (str[i] >= 'A' && str[i] <= 'Z') {
                has_letter = 1;
            }
        }
        return has_letter;
    }
    """)

        # 21. Функция zfill
        helpers.append("""
    char* string_zfill(const char* str, int width) {
        if (!str) return NULL;
        
        int str_len = strlen(str);
        int total_len = (width > str_len) ? width : str_len;
        
        char* result = malloc(total_len + 1);
        if (!result) return NULL;
        
        if (str_len >= width) {
            strcpy(result, str);
        } else {
            int zeros = width - str_len;
            for (int i = 0; i < zeros; i++) {
                result[i] = '0';
            }
            strcpy(result + zeros, str);
        }
        
        result[total_len] = '\\0';
        return result;
    }
    """)

        # 22. Функция center
        helpers.append("""
    char* string_center(const char* str, int width, char fillchar) {
        if (!str) return NULL;
        
        int str_len = strlen(str);
        if (str_len >= width) {
            char* result = malloc(str_len + 1);
            strcpy(result, str);
            return result;
        }
        
        char* result = malloc(width + 1);
        if (!result) return NULL;
        
        int left = (width - str_len) / 2;
        int right = width - str_len - left;
        
        for (int i = 0; i < left; i++) {
            result[i] = fillchar;
        }
        strcpy(result + left, str);
        for (int i = left + str_len; i < width; i++) {
            result[i] = fillchar;
        }
        
        result[width] = '\\0';
        return result;
    }
    """)

        # 23. Функция ljust
        helpers.append("""
    char* string_ljust(const char* str, int width, char fillchar) {
        if (!str) return NULL;
        
        int str_len = strlen(str);
        if (str_len >= width) {
            char* result = malloc(str_len + 1);
            strcpy(result, str);
            return result;
        }
        
        char* result = malloc(width + 1);
        if (!result) return NULL;
        
        strcpy(result, str);
        for (int i = str_len; i < width; i++) {
            result[i] = fillchar;
        }
        
        result[width] = '\\0';
        return result;
    }
    """)

        # 24. Функция rjust
        helpers.append("""
    char* string_rjust(const char* str, int width, char fillchar) {
        if (!str) return NULL;
        
        int str_len = strlen(str);
        if (str_len >= width) {
            char* result = malloc(str_len + 1);
            strcpy(result, str);
            return result;
        }
        
        char* result = malloc(width + 1);
        if (!result) return NULL;
        
        int padding = width - str_len;
        for (int i = 0; i < padding; i++) {
            result[i] = fillchar;
        }
        strcpy(result + padding, str);
        
        result[width] = '\\0';
        return result;
    }
    """)

        # 25. Функция format (простая версия для одного аргумента)
        helpers.append("""
    char* string_format(const char* format_str, const char* arg) {
        if (!format_str) return NULL;
        
        // Ищем {} в строке
        char* pos = strstr(format_str, "{}");
        if (!pos) {
            // Если нет {}, просто копируем строку
            char* result = malloc(strlen(format_str) + 1);
            strcpy(result, format_str);
            return result;
        }
        
        // Вычисляем длину результата
        int format_len = strlen(format_str);
        int arg_len = arg ? strlen(arg) : 0;
        int result_len = format_len - 2 + arg_len; // -2 для удаления {}
        
        char* result = malloc(result_len + 1);
        if (!result) return NULL;
        
        // Копируем часть до {}
        int before_len = pos - format_str;
        strncpy(result, format_str, before_len);
        
        // Копируем аргумент
        if (arg) {
            strcpy(result + before_len, arg);
        }
        
        // Копируем часть после {}
        strcpy(result + before_len + arg_len, pos + 2);
        
        return result;
    }
    """)

        # Функции для конвертации в строку
        if "builtin_str_int" not in self.generated_functions:
            helpers.append("""
        char* builtin_str_int(int value) {
            char buffer[12];
            sprintf(buffer, "%d", value);
            char* result = malloc(strlen(buffer) + 1);
            if (!result) return NULL;
            strcpy(result, buffer);
            return result;
        }
        """)
            self.generated_functions.add("builtin_str_int")

        if "builtin_str_float" not in self.generated_functions:
            helpers.append("""
        char* builtin_str_float(float value) {
            char buffer[32];
            sprintf(buffer, "%f", value);
            // Убираем лишние нули
            char* dot = strchr(buffer, '.');
            if (dot) {
                char* end = buffer + strlen(buffer) - 1;
                while (end > dot && *end == '0') *end-- = '\\0';
            }
            char* result = malloc(strlen(buffer) + 1);
            if (!result) return NULL;
            strcpy(result, buffer);
            return result;
        }
        """)
            self.generated_functions.add("builtin_str_float")

        self.generated_helpers.extend(helpers)

    def generate_builtin_int_helpers(self):
        """Генерирует вспомогательные функции для конвертации в int"""
        helpers = []

        # Функция builtin_int для конвертации строки в int
        helpers.append("""
        int builtin_int(const char* str) {
            if (!str || strlen(str) == 0) {
                fprintf(stderr, "ValueError: empty string cannot be converted to int\\n");
                exit(1);
            }
            
            // Проверяем, является ли строка допустимым целым числом
            int i = 0;
            int sign = 1;
            
            // Обрабатываем знак
            if (str[0] == '-') {
                sign = -1;
                i = 1;
            } else if (str[0] == '+') {
                i = 1;
            }
            
            // Проверяем все символы
            for (; str[i]; i++) {
                if (str[i] < '0' || str[i] > '9') {
                    fprintf(stderr, "ValueError: invalid literal for int(): '%s'\\n", str);
                    exit(1);
                }
            }
            
            // Используем стандартную функцию atoi
            return atoi(str);
        }
        """)

        # Также добавим функцию для конвертации float в int
        helpers.append("""
        int builtin_int_from_float(double value) {
            // Простое приведение float/double к int
            return (int)value;
        }
        """)

        # Функция для конвертации bool в int
        helpers.append("""
        int builtin_int_from_bool(bool value) {
            return value ? 1 : 0;
        }
        """)

        # Универсальная функция builtin_int с поддержкой разных типов
        helpers.append("""
        int builtin_int_universal(void* value, const char* type_hint) {
            if (!value) {
                return 0;
            }
            
            if (type_hint) {
                if (strcmp(type_hint, "str") == 0) {
                    return builtin_int((char*)value);
                } else if (strcmp(type_hint, "float") == 0 || strcmp(type_hint, "double") == 0) {
                    return builtin_int_from_float(*(double*)value);
                } else if (strcmp(type_hint, "bool") == 0) {
                    return builtin_int_from_bool(*(bool*)value);
                }
            }
            
            // По умолчанию пытаемся конвертировать строку
            return builtin_int((char*)value);
        }
        """)

        self.generated_helpers.extend(helpers)
