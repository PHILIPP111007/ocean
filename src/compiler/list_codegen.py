from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
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
            struct_code = f"typedef struct {{\n"
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
        """Генерирует стандартные функции для списка (всегда включает set)"""

        # Константы для оптимизации
        LIST_MIN_CAPACITY = 16
        LIST_GROWTH_FACTOR = 1.5

        # 1. Функция создания (оптимизированная с выравниванием)
        create_func_name = f"create_{struct_name}"
        if create_func_name not in self.generated_functions:
            create_func = f"""{struct_name}* {create_func_name}(int initial_capacity) {{
            // Минимальный размер для предотвращения частых реаллокаций
            if (initial_capacity < {LIST_MIN_CAPACITY}) initial_capacity = {LIST_MIN_CAPACITY};
            
            // Динамическое выравнивание в зависимости от архитектуры
            #if defined(__AVX512F__)  // Intel/AMD AVX-512
                initial_capacity = (initial_capacity + 31) & ~31;
            #elif defined(__AVX2__)    // Intel/AMD AVX2
                initial_capacity = (initial_capacity + 7) & ~7;
            #elif defined(__ARM_NEON) || defined(__ARM_NEON__)  // Apple M1/M2/M3 (ARM NEON)
                // NEON работает лучше с выравниванием по 16 байт
                initial_capacity = (initial_capacity + 15) & ~15;
            #else  // Базовое выравнивание
                initial_capacity = (initial_capacity + 3) & ~3;
            #endif

            // Используем posix_memalign для лучшей совместимости
            {struct_name}* list = NULL;
            
            #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
                // posix_memalign доступен на большинстве Unix систем (включая macOS)
                if (posix_memalign((void**)&list, 64, sizeof({struct_name})) != 0) {{
                    list = NULL;
                }}
            #else
                // Fallback на aligned_alloc (C11)
                list = ({struct_name}*)aligned_alloc(64, sizeof({struct_name}));
            #endif
            
            // Если выровненное выделение не сработало, используем malloc
            if (!list) {{
                list = ({struct_name}*)malloc(sizeof({struct_name}));
                if (!list) {{
                    fprintf(stderr, "Memory allocation failed for list\\n");
                    exit(1);
                }}
            }}

            // Выровненное выделение для данных
            {element_type}* data = NULL;
            size_t data_size = initial_capacity * sizeof({element_type});
            
            #if defined(__AVX512F__)
                // AVX-512 требует выравнивания по 64 байта
                if (posix_memalign((void**)&data, 64, data_size) != 0) {{
                    data = NULL;
                }}
            #elif defined(__AVX2__)
                // AVX2 требует выравнивания по 32 байта
                if (posix_memalign((void**)&data, 32, data_size) != 0) {{
                    data = NULL;
                }}
            #elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                // ARM NEON (Apple M1/M2/M3) работает лучше с выравниванием по 16 байт
                if (posix_memalign((void**)&data, 16, data_size) != 0) {{
                    data = NULL;
                }}
            #else
                // Базовое выравнивание
                if (posix_memalign((void**)&data, 16, data_size) != 0) {{
                    data = NULL;
                }}
            #endif
            
            // Если выровненное выделение не сработало, используем calloc
            if (!data) {{
                data = ({element_type}*)calloc(initial_capacity, sizeof({element_type}));
                if (!data) {{
                    fprintf(stderr, "Memory allocation failed for list data\\n");
                    free(list);
                    exit(1);
                }}
            }} else {{
                // Обнуляем память (для aligned_alloc/posix_memalign)
                memset(data, 0, data_size);
            }}
            
            list->data = data;
            list->size = 0;
            list->capacity = initial_capacity;
            
            return list;
        }}

        """
            self.generated_helpers.append(create_func)
            self.generated_functions.add(create_func_name)

        # 2. Функция добавления (оптимизированная с предсказанием ветвлений и поддержкой архитектур)
        append_func_name = f"append_{struct_name}"
        if append_func_name not in self.generated_functions:
            append_func = f"""void {append_func_name}({struct_name}* list, {element_type} value) {{
            // Быстрая проверка с предсказанием ветвлений
            if (__builtin_expect(list->size >= list->capacity, 0)) {{
                // Медленный путь - реаллокация
                int new_capacity = (int)(list->capacity * {LIST_GROWTH_FACTOR});
                if (new_capacity < 64) new_capacity = 64;

                // Динамическое выравнивание в зависимости от архитектуры
                #if defined(__AVX512F__)  // Intel/AMD AVX-512
                    new_capacity = (new_capacity + 31) & ~31;
                #elif defined(__AVX2__)    // Intel/AMD AVX2
                    new_capacity = (new_capacity + 7) & ~7;
                #elif defined(__ARM_NEON) || defined(__ARM_NEON__)  // Apple M1/M2/M3 (ARM NEON)
                    // NEON работает лучше с выравниванием по 16 байт
                    new_capacity = (new_capacity + 15) & ~15;
                #else  // Базовое выравнивание
                    new_capacity = (new_capacity + 3) & ~3;
                #endif

                // Сохраняем старый указатель для возможного восстановления
                {element_type}* old_data = list->data;
                size_t old_size = list->size;
                
                // Используем выделение с учетом архитектуры
                {element_type}* new_data = NULL;
                size_t new_size_bytes = new_capacity * sizeof({element_type});
                
                #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
                    // Определяем выравнивание для нового блока
                    size_t alignment = 16;  // По умолчанию
                    
                    #if defined(__AVX512F__)
                        alignment = 64;
                    #elif defined(__AVX2__)
                        alignment = 32;
                    #elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                        alignment = 16;
                    #endif
                    
                    // Пробуем выделить выровненную память
                    if (posix_memalign((void**)&new_data, alignment, new_size_bytes) == 0) {{
                        // Копируем старые данные
                        memcpy(new_data, old_data, old_size * sizeof({element_type}));
                        // Освобождаем старую память
                        free(old_data);
                        list->data = new_data;
                        list->capacity = new_capacity;
                    }} else {{
                        // Если не получилось, используем realloc как fallback
                        new_data = ({element_type}*)realloc(old_data, new_size_bytes);
                        if (!new_data) {{
                            fprintf(stderr, "Memory reallocation failed for list\\n");
                            exit(1);
                        }}
                        list->data = new_data;
                        list->capacity = new_capacity;
                    }}
                #else
                    // Если posix_memalign недоступен, используем обычный realloc
                    new_data = ({element_type}*)realloc(old_data, new_size_bytes);
                    if (!new_data) {{
                        fprintf(stderr, "Memory reallocation failed for list\\n");
                        exit(1);
                    }}
                    list->data = new_data;
                    list->capacity = new_capacity;
                #endif
            }}

            list->data[list->size++] = value;
        }}

        """
            self.generated_helpers.append(append_func)
            self.generated_functions.add(append_func_name)

        # 3. Функция расширения (пакетное добавление) - НОВАЯ!
        extend_func_name = f"extend_{struct_name}"
        if extend_func_name not in self.generated_functions:
            extend_func = f"""void {extend_func_name}({struct_name}* list, const {element_type}* values, int count) {{
    if (count <= 0) return;

    int new_size = list->size + count;

    // Предварительное расширение если нужно
    if (new_size > list->capacity) {{
        int new_capacity = list->capacity;
        while (new_capacity < new_size) {{
            new_capacity = (int)(new_capacity * {LIST_GROWTH_FACTOR});
        }}

        // Выравнивание
        #ifdef __AVX512F__
            new_capacity = (new_capacity + 31) & ~31;
        #elif defined(__AVX2__)
            new_capacity = (new_capacity + 7) & ~7;
        #else
            new_capacity = (new_capacity + 3) & ~3;
        #endif

        {element_type}* new_data = ({element_type}*)realloc(list->data, new_capacity * sizeof({element_type}));
        if (!new_data) {{
            fprintf(stderr, "Memory reallocation failed for list\\n");
            exit(1);
        }}

        list->data = new_data;
        list->capacity = new_capacity;
    }}

    // SIMD-оптимизированное копирование
    #ifdef __AVX512F__
        int i = 0;
        for (; i + 15 < count; i += 16) {{
            __m512i vec = _mm512_loadu_si512((__m512i*)&values[i]);
            _mm512_storeu_si512((__m512i*)&list->data[list->size + i], vec);
        }}
        for (; i < count; i++) {{
            list->data[list->size + i] = values[i];
        }}
    #elif defined(__AVX2__)
        int i = 0;
        for (; i + 7 < count; i += 8) {{
            __m256i vec = _mm256_loadu_si256((__m256i*)&values[i]);
            _mm256_storeu_si256((__m256i*)&list->data[list->size + i], vec);
        }}
        for (; i < count; i++) {{
            list->data[list->size + i] = values[i];
        }}
    #elif defined(__SSE2__)
        int i = 0;
        for (; i + 3 < count; i += 4) {{
            __m128i vec = _mm_loadu_si128((__m128i*)&values[i]);
            _mm_storeu_si128((__m128i*)&list->data[list->size + i], vec);
        }}
        for (; i < count; i++) {{
            list->data[list->size + i] = values[i];
        }}
    #else
        memcpy(&list->data[list->size], values, count * sizeof({element_type}));
    #endif

    list->size = new_size;
}}

"""
            self.generated_helpers.append(extend_func)
            self.generated_functions.add(extend_func_name)

        # 4. Функция len() (инлайн рекомендация)
        len_func_name = f"builtin_len_{struct_name}"
        if len_func_name not in self.generated_functions:
            len_func = f"""static inline int {len_func_name}(const {struct_name}* list) {{
    return list ? list->size : 0;
}}

"""
            self.generated_helpers.append(len_func)
            self.generated_functions.add(len_func_name)

        # 5. Функция очистки (оптимизированная)
        free_func_name = f"free_{struct_name}"
        if free_func_name not in self.generated_functions:
            free_func = f"""void {free_func_name}({struct_name}* list) {{
            if (list) {{
        """

            # Если элементы - указатели на другие структуры, освобождаем их
            if element_py_type and (
                element_py_type.startswith("dict[")
                or element_py_type.startswith("list[")
                or element_py_type.startswith("tuple[")
            ):
                # Определяем функцию для освобождения элементов
                if element_py_type.startswith("dict["):
                    key_type, value_type = self._extract_dict_types(element_py_type)
                    key_name = self.clean_type_name_for_c(key_type)
                    value_name = self.clean_type_name_for_c(value_type)
                    inner_free_func = f"free_dict_{key_name}_{value_name}"
                elif element_py_type.startswith("list["):
                    inner_struct = self.generate_list_struct_name(element_py_type)
                    inner_free_func = f"free_{inner_struct}"
                else:
                    inner_free_func = (
                        f"free_{self.clean_type_name_for_c(element_py_type)}"
                    )

                free_func += f"""        // Освобождаем каждый элемент списка
                for (int i = 0; i < list->size; i++) {{
                    if (list->data[i]) {{
                        {inner_free_func}(list->data[i]);
                    }}
                }}
        """
            # Если элементы - простые указатели (char*), освобождаем их
            elif element_py_type == "str" or element_type == "char*":
                free_func += f"""        // Освобождаем каждую строку
                for (int i = 0; i < list->size; i++) {{
                    if (list->data[i]) {{
                        free(list->data[i]);
                    }}
                }}
        """

            free_func += f"""        free(list->data);
                free(list);
            }}
        }}

        """
            self.generated_helpers.append(free_func)
            self.generated_functions.add(free_func_name)

        # 7. Функция установки элемента (set) - инлайн для производительности
        set_func_name = f"set_{struct_name}"
        if set_func_name not in self.generated_functions:
            set_func = f"""static inline void {set_func_name}({struct_name}* list, int index, {element_type} value) {{
    #ifndef NDEBUG
    if (!list || index < 0 || index >= list->size) {{
        fprintf(stderr, "Index out of bounds in list\\n");
        exit(1);
    }}
    #endif
    list->data[index] = value;
}}

"""
            self.generated_helpers.append(set_func)
            self.generated_functions.add(set_func_name)

        # 9. Функция slice для списка (оптимизированная)
        slice_func_name = f"slice_{struct_name}"
        if slice_func_name not in self.generated_functions:
            slice_func = f"""{struct_name}* slice_{struct_name}(const {struct_name}* list, int start, int stop, int step) {{
    if (!list) return NULL;

    // Нормализация индексов
    if (start < 0) start = list->size + start;
    if (stop < 0) stop = list->size + stop;
    if (start < 0) start = 0;
    if (stop > list->size) stop = list->size;

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

    // Создаем новый список
    {struct_name}* result = create_{struct_name}(new_size);

    // Оптимизированное копирование с учетом шага
    if (step > 0) {{
        for (int i = start; i < stop && result->size < new_size; i += step) {{
            result->data[result->size++] = list->data[i];
        }}
    }} else {{
        for (int i = start; i > stop && result->size < new_size; i += step) {{
            result->data[result->size++] = list->data[i];
        }}
    }}

    return result;
}}

"""
            self.generated_helpers.append(slice_func)
            self.generated_functions.add(slice_func_name)

        # 11. Функция копирования (НОВАЯ!)
        copy_func_name = f"copy_{struct_name}"
        if copy_func_name not in self.generated_functions:
            copy_func = f"""{struct_name}* copy_{struct_name}(const {struct_name}* src) {{
    if (!src) return NULL;
    
    {struct_name}* dst = create_{struct_name}(src->capacity);
    dst->size = src->size;
    
    #ifdef __AVX2__
        int i = 0;
        for (; i + 7 < src->size; i += 8) {{
            __m256i vec = _mm256_loadu_si256((__m256i*)&src->data[i]);
            _mm256_storeu_si256((__m256i*)&dst->data[i], vec);
        }}
        for (; i < src->size; i++) {{
            dst->data[i] = src->data[i];
        }}
    #else
        memcpy(dst->data, src->data, src->size * sizeof({element_type}));
    #endif
    
    return dst;
}}

"""
            self.generated_helpers.append(copy_func)
            self.generated_functions.add(copy_func_name)

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
