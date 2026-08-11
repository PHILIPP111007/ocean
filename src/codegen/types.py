from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class TypesMixin:
    def map_type_to_c(self, py_type: str, is_pointer: bool = False) -> str:
        """Map a Phils type to C, including zero-cost borrows ``&T``/``&mut T``."""

        py_type = (py_type or "").strip()
        if py_type.startswith("&mut "):
            return self.map_type_to_c(py_type[5:].strip(), is_pointer=False)
        if py_type.startswith("&"):
            return self.map_type_to_c(py_type[1:].strip(), is_pointer=False)

        # Проверяем, не является ли это уже C типом
        if self._is_c_type(py_type):
            # Если это известный C тип, возвращаем как есть
            return py_type

        if self._is_class_type(py_type):
            # Классы в C - это указатели на структуры
            if is_pointer:
                return f"{py_type}**"  # Указатель на указатель
            return f"{py_type}*"  # Обычный указатель на структуру

        if py_type == "None":
            return "void*"  # None -> void*
        elif py_type.startswith("*"):
            base_type = py_type[1:]
            c_base_type = self.map_type_to_c(base_type)
            return f"{c_base_type}*"
        elif py_type == "pointer":
            return "void*"
        elif py_type.startswith("tuple["):
            # Генерируем структуру для кортежа
            self.generate_tuple_struct(py_type)
            struct_name = self.generate_tuple_struct_name(py_type)
            return f"{struct_name}*"
        elif py_type.startswith("array["):
            self.generate_array_struct(py_type)
            return self.array_c_type(py_type)
        elif py_type.startswith("tensor["):
            self.generate_tensor_struct(py_type)
            return self.tensor_c_type(py_type)
        elif py_type.startswith("list["):
            # Генерируем структуру для list
            self.generate_list_struct(py_type)
            struct_name = self.generate_list_struct_name(py_type)

            # list всегда указатель на структуру
            c_type = f"{struct_name}*"

            if is_pointer:
                return f"{c_type}*"
            return c_type
        elif py_type.startswith("dict["):
            # Генерируем структуру для словаря
            key_type, value_type = self._extract_dict_types(py_type)
            struct_name = self.generate_dict_struct(key_type, value_type)
            logger.debug(f"map_type_to_c: dict {py_type} -> {struct_name}*")

            if is_pointer:
                return f"{struct_name}**"
            return f"{struct_name}*"
        else:
            c_type = self.type_map.get(py_type, "int")
            if is_pointer:
                return f"{c_type}*"
            return c_type

    def _is_c_type(self, type_name: str) -> bool:
        """Определяет, является ли тип известным C типом"""
        if not isinstance(type_name, str):
            return False

        # Проверяем, является ли это известным C типом
        if type_name in self.known_c_types:
            return True

        # Проверяем по шаблонам C типов
        c_type_patterns = [
            r"^[a-zA-Z_][a-zA-Z0-9_]*_t$",  # _t типы (pthread_t, size_t и т.д.)
            r"^FILE$",
            r"^clock_t$",
            r"^time_t$",
        ]

        for pattern in c_type_patterns:
            if re.match(pattern, type_name):
                # Добавляем в известные типы
                self.known_c_types.add(type_name)
                return True

        # Проверяем, содержит ли тип указатель
        if "*" in type_name:
            # Разделяем на базовый тип и указатели
            base = type_name.replace("*", "").strip()
            if self._is_c_type(base):
                self.known_c_types.add(type_name)
                return True

        return False

    def _is_class_type(self, type_name: str) -> bool:
        """Определяет, является ли тип классом"""
        if not isinstance(type_name, str):
            return False

        # Проверяем по зарегистрированным классам
        if hasattr(self, "class_types") and type_name in self.class_types:
            return True

        # Проверяем по типу (классы обычно с большой буквы)
        if type_name and len(type_name) > 0 and type_name[0].isupper():
            # Проверяем, не является ли это базовым типом или встроенным типом
            base_types = {"int", "float", "double", "char", "bool", "void", "None"}
            if type_name not in base_types:
                return True

        return False

    def _get_current_class(self) -> Optional[str]:
        """Получает имя текущего класса из контекста"""
        # Ищем в текущем и родительских scope'ах
        for i in range(len(self.variable_scopes) - 1, -1, -1):
            if "class_name" in self.variable_scopes[i]:
                return self.variable_scopes[i]["class_name"]
        return None

    def _is_string_expression(self, ast: Dict) -> bool:
        """Определяет, является ли выражение строкой"""
        if not ast:
            return False

        node_type = ast.get("type", "")

        if node_type == "literal":
            return ast.get("data_type", "") == "str"

        elif node_type == "variable":
            var_name = ast.get("value", "")
            var_info = self.get_variable_info(var_name)
            if var_info:
                return var_info.get("py_type", "") == "str"

        elif node_type == "binary_operation":
            left_ast = ast.get("left", {})
            right_ast = ast.get("right", {})
            operator = ast.get("operator_symbol", "")

            if operator == "+":
                return self._is_string_expression(
                    left_ast
                ) or self._is_string_expression(right_ast)

        return False

    def _is_none_expression(self, ast: Dict) -> bool:
        """Проверяет, является ли выражение None"""
        if not ast:
            return False

        if ast.get("type") == "literal":
            return ast.get("data_type") == "None" or ast.get("value") == "None"

        if ast.get("type") == "variable":
            var_name = ast.get("value", "")
            var_info = self.get_variable_info(var_name)
            if var_info:
                return var_info.get("py_type") == "None"

        return False

    def _extract_dict_types(self, dict_type: str) -> tuple:
        """Извлекает типы ключа и значения из dict[K, V]"""
        if not dict_type.startswith("dict[") or not dict_type.endswith("]"):
            return "int", "int"  # По умолчанию

        # Извлекаем содержимое между скобками
        inner = dict_type[5:-1].strip()

        # Ищем запятую, которая не находится внутри вложенных скобок
        depth = 0
        comma_pos = -1

        for i, char in enumerate(inner):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            elif char == "," and depth == 0:
                comma_pos = i
                break

        if comma_pos == -1:
            return "int", "int"

        key_type = inner[:comma_pos].strip()
        value_type = inner[comma_pos + 1 :].strip()

        return key_type, value_type

    def clean_type_name_for_c(self, type_name: str) -> str:
        """Очищает имя типа для использования в C идентификаторах"""
        if not isinstance(type_name, str):
            return "unknown"

        # Удаляем все небуквенно-цифровые символы и заменяем на _
        cleaned = re.sub(r"[^a-zA-Z0-9]", "_", type_name)
        # Убираем множественные подчеркивания
        cleaned = re.sub(r"_+", "_", cleaned)
        # Убираем подчеркивания в начале и конце
        cleaned = cleaned.strip("_")

        # Если после очистки строка пустая, используем дефолтное имя
        if not cleaned:
            return "unknown"

        # Делаем первую букву строчной для согласованности
        if cleaned[0].isupper():
            cleaned = cleaned[0].lower() + cleaned[1:]

        return cleaned

    def extract_nested_type_info(self, py_type: str) -> Dict:
        """Извлекает информацию о вложенном типе списка с рекурсивным анализом"""
        if not py_type or not isinstance(py_type, str):
            return self._create_default_type_info()

        logger.debug(f"extract_nested_type_info: {py_type}")

        # Базовый случай: не список
        if not py_type.startswith("list["):
            # Для простых типов
            is_c_type = self._is_c_type(py_type)
            c_type = py_type if is_c_type else self.map_type_to_c(py_type)
            struct_name = f"list_{self.clean_type_name_for_c(py_type)}"

            logger.debug(
                f"DEBUG: Базовый тип - is_c_type={is_c_type}, c_type={c_type}, struct_name={struct_name}"
            )

            # НЕ ГЕНЕРИРУЕМ здесь - структура будет сгенерирована позже
            return {
                "py_type": py_type,
                "c_type": f"{struct_name}*",
                "struct_name": struct_name,
                "element_type": c_type,
                "element_py_type": py_type,
                "is_leaf": True,
                "is_c_type": is_c_type,
                "inner_info": None,
            }

        try:
            # Извлекаем внутренний тип
            inner_type = self._parse_list_type(py_type)
            if not inner_type:
                logger.debug(f"Не удалось извлечь внутренний тип из {py_type}")
                return self._create_default_type_info()

            logger.debug(f"Внутренний тип: {inner_type}")

            # Генерируем имя структуры
            struct_name = self._generate_struct_name_recursive(py_type)
            logger.debug(f"Сгенерированное имя структуры: {struct_name}")

            # Рекурсивно анализируем внутренний тип
            inner_info = self.extract_nested_type_info(inner_type)

            # Определяем информацию о текущем уровне
            is_leaf = not inner_type.startswith("list[")

            # Определяем element_type
            if is_leaf:
                # Если это list[T], то element_type = T
                if self._is_c_type(inner_type):
                    element_type = inner_type
                else:
                    element_type = self.map_type_to_c(inner_type)
            else:
                # Если это list[list[...]], то element_type = inner_struct*
                if inner_info.get("struct_name"):
                    element_type = f"{inner_info['struct_name']}*"
                else:
                    element_type = "void*"

            result = {
                "py_type": py_type,
                "c_type": f"{struct_name}*",
                "struct_name": struct_name,
                "element_type": element_type,
                "element_py_type": inner_type,
                "is_leaf": is_leaf,
                "is_c_type": inner_info.get("is_c_type", False) if is_leaf else False,
                "inner_info": inner_info,
            }

            logger.debug(
                f"DEBUG результат: struct_name={struct_name}, element_type={element_type}"
            )

            return result

        except Exception as e:
            logger.debug(f"ERROR в extract_nested_type_info для {py_type}: {e}")
            return self._create_default_type_info()

    def _parse_list_type(self, list_type: str) -> Optional[str]:
        """Парсит тип списка и извлекает внутренний тип"""
        if not list_type.startswith("list["):
            return None

        # Счетчик скобок для правильного парсинга вложенных типов
        bracket_count = 0
        start_idx = 4  # индекс после "list"

        # Находим начало внутреннего типа
        for i in range(start_idx, len(list_type)):
            if list_type[i] == "[":
                bracket_count += 1
            elif list_type[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    # Нашли закрывающую скобку
                    inner_type = list_type[start_idx + 1 : i]  # +1 чтобы пропустить '['
                    return inner_type.strip()

        return None

    def _create_default_type_info(self) -> Dict:
        """Создает информацию о типе по умолчанию"""
        return {
            "py_type": "unknown",
            "c_type": "void*",
            "struct_name": None,
            "element_type": None,
            "is_leaf": True,
            "inner_info": None,
        }

    def _generate_struct_name_recursive(self, py_type: str) -> str:
        """Рекурсивно генерирует имя структуры для вложенного списка"""
        if not py_type.startswith("list["):
            # Если это не список, проверяем, является ли это C типом
            if self._is_c_type(py_type):
                # Для C типов возвращаем list_имя_типа
                clean_name = self.clean_type_name_for_c(py_type)
                return f"list_{clean_name}"
            else:
                # Для других типов (int, float и т.д.)
                clean_name = self.clean_type_name_for_c(py_type)
                return f"list_{clean_name}"

        # Извлекаем внутренний тип
        inner_type = self._parse_list_type(py_type)
        if not inner_type:
            return "list_unknown"

        # Если внутренний тип тоже список, рекурсивно генерируем имя
        if inner_type.startswith("list["):
            inner_struct_name = self._generate_struct_name_recursive(inner_type)
            # Для list[list[int]] -> list_list_int
            return f"list_{inner_struct_name}"
        else:
            # list[int] -> list_int
            clean_inner = self.clean_type_name_for_c(inner_type)
            return f"list_{clean_inner}"

    def _collect_all_nested_list_types(self, list_type: str, type_set: set):
        """Рекурсивно собирает все вложенные типы списков"""
        if not list_type.startswith("list["):
            return

        type_set.add(list_type)

        # Извлекаем внутренний тип
        inner_type = self._parse_list_type(list_type)
        if inner_type:
            if inner_type.startswith("list["):
                # Если внутренний тип тоже список, рекурсивно обрабатываем
                self._collect_all_nested_list_types(inner_type, type_set)
            else:
                # Листовой тип - создаем базовую структуру list_тип
                leaf_struct = f"list[{inner_type}]"
                type_set.add(leaf_struct)

    def extract_all_types_from_ast(self, json_data: List[Dict]) -> set:
        """Извлекает все типы из AST для генерации структур"""
        all_types = set()

        def process_value(value):
            """Рекурсивно обрабатывает значение для извлечения типов"""
            if isinstance(value, dict):
                # Обрабатываем литералы списков
                if value.get("type") == "list_literal":
                    items = value.get("items", [])
                    if items:
                        # Определяем тип элементов
                        first_item = items[0]
                        if first_item.get("type") == "list_literal":
                            element_type = "list"
                        elif first_item.get("type") == "literal":
                            element_type = first_item.get("data_type", "int")
                        else:
                            element_type = "int"

                        list_type = f"list[{element_type}]"
                        all_types.add(list_type)

                        # Рекурсивно обрабатываем вложенные списки
                        for item in items:
                            process_value(item)

                # Обрабатываем литералы словарей
                elif value.get("type") == "dict_literal":
                    pairs = value.get("pairs", {})
                    if pairs:
                        # Определяем типы ключа и значения
                        first_key = next(iter(pairs))
                        first_val = pairs[first_key]

                        key_type = "str" if isinstance(first_key, str) else "int"
                        val_type = self._infer_type_from_value(first_val)

                        dict_type = f"dict[{key_type}, {val_type}]"
                        all_types.add(dict_type)

                        # Рекурсивно обрабатываем значения
                        for val in pairs.values():
                            process_value(val)

        def process_node(node):
            """Рекурсивно обрабатывает узел AST"""
            if not isinstance(node, dict):
                return

            node_type = node.get("node", "")

            # Обрабатываем объявления переменных
            if node_type == "declaration":
                var_type = node.get("var_type", "")
                if var_type:
                    all_types.add(var_type)
                    # Если это сложный тип, добавляем все вложенные типы
                    self._add_nested_types(var_type, all_types)

                # Array/tensor literals are consumed by their dedicated
                # runtime generators, not by the generic list runtime.
                expr_ast = node.get("expression_ast", {})
                if not (self.is_array_type(var_type) or self.is_tensor_type(var_type)):
                    process_value(expr_ast)

            # Обрабатываем присваивания
            elif node_type == "assignment":
                expr_ast = node.get("expression_ast", {})
                process_value(expr_ast)

            # Обрабатываем вызовы функций
            elif node_type == "function_call":
                for arg in node.get("arguments", []):
                    if isinstance(arg, dict):
                        process_value(arg)

            # Обрабатываем вызовы методов
            elif node_type == "method_call":
                for arg in node.get("arguments", []):
                    if isinstance(arg, dict):
                        process_value(arg)

            # Обрабатываем циклы
            elif node_type == "for_loop":
                for body_node in node.get("body", []):
                    process_node(body_node)

            # Обрабатываем условия
            elif node_type == "if_statement":
                for body_node in node.get("body", []):
                    process_node(body_node)
                for elif_block in node.get("elif_blocks", []):
                    for body_node in elif_block.get("body", []):
                        process_node(body_node)
                if node.get("else_block"):
                    for body_node in node.get("else_block").get("body", []):
                        process_node(body_node)

            # Обрабатываем все поля узла на предмет типов
            for key, value in node.items():
                if key.endswith("_type") and isinstance(value, str):
                    if value and value not in ["int", "float", "str", "bool", "None"]:
                        all_types.add(value)
                        self._add_nested_types(value, all_types)

        # Проходим по всем scope
        for scope in json_data:
            if scope.get("type") in [
                "module",
                "function",
                "class_method",
                "constructor",
            ]:
                for node in scope.get("graph", []):
                    process_node(node)

                # Также проверяем локальные переменные
                for var_name in scope.get("local_variables", []):
                    var_info = scope.get("symbol_table", {}).get(var_name, {})
                    var_type = var_info.get("type", "")
                    if var_type:
                        all_types.add(var_type)
                        self._add_nested_types(var_type, all_types)

        return all_types

    def collect_types_from_ast(self, json_data: List[Dict]):
        """Собирает все типы из AST для генерации структур"""
        all_types = set()

        def process_node(node):
            if not isinstance(node, dict):
                return

            # Обрабатываем declaration узлы
            if node.get("node") == "declaration":
                var_type = node.get("var_type", "")
                if var_type:
                    if var_type.startswith("list["):
                        all_types.add(var_type)
                        # Также добавляем ВСЕ ВЛОЖЕННЫЕ ТИПЫ
                        self._collect_all_nested_list_types(var_type, all_types)
                    elif var_type.startswith("tuple["):
                        all_types.add(var_type)
                    elif var_type.startswith("dict["):
                        all_types.add(var_type)  # Добавляем словари
                        # Также добавляем типы ключа и значения
                        key_type, value_type = self._extract_dict_types(var_type)
                        if key_type.startswith("list["):
                            all_types.add(key_type)
                        if value_type.startswith("list["):
                            all_types.add(value_type)

            # Обрабатываем временные переменные (temp_0, temp_1 и т.д.)
            if node.get("node") == "declaration" and node.get(
                "var_name", ""
            ).startswith("temp_"):
                var_type = node.get("var_type", "")
                if var_type and var_type.startswith("list["):
                    all_types.add(var_type)
                    self._collect_all_nested_list_types(var_type, all_types)

        # Проходим по всем scope и узлам
        for scope in json_data:
            if scope.get("type") in ["module", "function"]:
                # Обрабатываем graph узлы
                for node in scope.get("graph", []):
                    process_node(node)

        # Генерируем структуры для всех найденных типов
        # Сортируем по глубине вложенности (от простых к сложным)
        sorted_types = sorted(all_types, key=lambda x: (x.count("["), x))

        # ВАЖНО: Сначала генерируем ВСЕ структуры
        for py_type in sorted_types:
            if py_type.startswith("list["):
                logger.debug(
                    f"collect_types_from_ast: Генерация структуры для {py_type}"
                )
                self.generate_list_struct(py_type)
            elif py_type.startswith("tuple["):
                self.generate_tuple_struct(py_type)
            elif py_type.startswith("dict["):
                # Для словарей нужно вызвать generate_dict_struct
                key_type, value_type = self._extract_dict_types(py_type)
                logger.debug(
                    f"collect_types_from_ast: Генерация структуры для {py_type}"
                )
                self.generate_dict_struct(key_type, value_type)

        # Затем генерируем ВСЕ функции для ВСЕХ структур
        self._generate_all_list_functions()

    def _add_nested_types(self, type_str: str, types_set: set):
        """Рекурсивно добавляет все вложенные типы"""
        if not isinstance(type_str, str):
            return

        # Обработка list[T]
        if type_str.startswith("list["):
            inner = type_str[5:-1].strip()
            types_set.add(type_str)
            self._add_nested_types(inner, types_set)

        # Обработка dict[K, V]
        elif type_str.startswith("dict["):
            inner = type_str[5:-1].strip()
            # Ищем запятую вне скобок
            depth = 0
            comma_pos = -1
            for i, char in enumerate(inner):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                elif char == "," and depth == 0:
                    comma_pos = i
                    break

            if comma_pos != -1:
                key_type = inner[:comma_pos].strip()
                val_type = inner[comma_pos + 1 :].strip()
                types_set.add(type_str)
                self._add_nested_types(key_type, types_set)
                self._add_nested_types(val_type, types_set)

        # Обработка tuple[T1, T2, ...]
        elif type_str.startswith("tuple["):
            inner = type_str[6:-1].strip()
            if "," in inner:
                # Разделяем по запятым вне скобок
                elements = []
                current = ""
                depth = 0
                for char in inner:
                    if char == "[":
                        depth += 1
                        current += char
                    elif char == "]":
                        depth -= 1
                        current += char
                    elif char == "," and depth == 0:
                        elements.append(current.strip())
                        current = ""
                    else:
                        current += char
                if current:
                    elements.append(current.strip())

                types_set.add(type_str)
                for elem_type in elements:
                    self._add_nested_types(elem_type, types_set)
            else:
                types_set.add(type_str)
                self._add_nested_types(inner, types_set)

    def _infer_type_from_value(self, value) -> str:
        """Определяет тип Python из значения (AST узла или примитива)"""

        # Если значение - словарь (AST узел)
        if isinstance(value, dict):
            node_type = value.get("type", "")

            # Литералы
            if node_type == "literal":
                return value.get("data_type", "int")

            # Литералы списков
            elif node_type == "list_literal":
                items = value.get("items", [])
                if items:
                    # Определяем тип первого элемента
                    first_item_type = self._infer_type_from_value(items[0])
                    return f"list[{first_item_type}]"
                return "list[int]"  # Пустой список по умолчанию

            # Литералы словарей
            elif node_type == "dict_literal":
                pairs = value.get("pairs", {})
                if pairs:
                    # Берем первую пару для определения типов
                    first_key = next(iter(pairs))
                    first_val = pairs[first_key]

                    # Определяем тип ключа
                    if isinstance(first_key, str):
                        key_type = "str"
                    elif isinstance(first_key, int):
                        key_type = "int"
                    else:
                        key_type = self._infer_type_from_value(first_key)

                    # Определяем тип значения
                    val_type = self._infer_type_from_value(first_val)

                    return f"dict[{key_type}, {val_type}]"
                return "dict[str, int]"  # Пустой словарь по умолчанию

            # Литералы кортежей
            elif node_type == "tuple_literal":
                items = value.get("items", [])
                if items:
                    # Определяем типы всех элементов
                    item_types = []
                    for item in items:
                        item_types.append(self._infer_type_from_value(item))

                    # Если все элементы одного типа
                    if len(set(item_types)) == 1:
                        return f"tuple[{item_types[0]}]"
                    else:
                        return f"tuple[{', '.join(item_types)}]"
                return "tuple[int]"  # Пустой кортеж по умолчанию

            # Переменные - пытаемся найти их тип
            elif node_type == "variable":
                var_name = value.get("value", "") or value.get("name", "")
                var_info = self.get_variable_info(var_name)
                if var_info:
                    return var_info.get("py_type", "int")
                return "int"

            # Вызовы функций - определяем по имени функции
            elif node_type == "function_call":
                func_name = value.get("function", "")
                builtin_returns = {
                    "len": "int",
                    "str": "str",
                    "int": "int",
                    "float": "float",
                    "bool": "bool",
                    "range": "range",
                    "input": "str",
                    "print": "None",
                }
                return builtin_returns.get(func_name, "int")

            # Вызовы методов
            elif node_type == "method_call":
                obj_name = value.get("object", "")
                method_name = value.get("method", "")

                # Пытаемся определить тип объекта
                obj_info = self.get_variable_info(obj_name)
                if obj_info:
                    obj_type = obj_info.get("py_type", "")

                    # Маппинг методов к возвращаемым типам
                    if obj_type.startswith("list["):
                        if method_name == "pop":
                            # Извлекаем тип элемента из list[T]
                            match = re.match(r"list\[([^\]]+)\]", obj_type)
                            if match:
                                return match.group(1)
                        elif method_name == "copy":
                            return obj_type
                    elif obj_type == "str":
                        if method_name in ["upper", "lower", "strip", "replace"]:
                            return "str"
                        elif method_name == "split":
                            return "list[str]"

                return "int"

            # Бинарные операции
            elif node_type == "binary_operation":
                left = value.get("left", {})
                right = value.get("right", {})
                operator = value.get("operator_symbol", "")

                left_type = self._infer_type_from_value(left)
                right_type = self._infer_type_from_value(right)

                # Для арифметических операций
                if operator in ["+", "-", "*", "/", "//", "%", "**"]:
                    if "float" in left_type or "float" in right_type:
                        return "float"
                    return "int"

                # Для сравнений
                elif operator in ["<", ">", "<=", ">=", "==", "!="]:
                    return "bool"

                # Для логических операций
                elif operator in ["and", "or"]:
                    return "bool"

            # Унарные операции
            elif node_type == "unary_operation":
                operator = value.get("operator_symbol", "")
                if operator == "not":
                    return "bool"
                elif operator in ["+", "-"]:
                    operand = value.get("operand", {})
                    return self._infer_type_from_value(operand)

            # Доступ по индексу
            elif node_type == "index_access":
                var_name = value.get("variable", "")
                var_info = self.get_variable_info(var_name)
                if var_info:
                    var_type = var_info.get("py_type", "")
                    if var_type.startswith("list["):
                        # Извлекаем тип элемента из list[T]
                        match = re.match(r"list\[([^\]]+)\]", var_type)
                        if match:
                            return match.group(1)
                    elif var_type.startswith("dict["):
                        # Извлекаем тип значения из dict[K, V]
                        match = re.match(r"dict\[[^,]+,\s*([^\]]+)\]", var_type)
                        if match:
                            return match.group(1)
                return "int"

            # Доступ к атрибуту
            elif node_type == "attribute_access":
                obj_name = value.get("object", "")
                attr_name = value.get("attribute", "")

                obj_info = self.get_variable_info(obj_name)
                if obj_info:
                    obj_type = obj_info.get("py_type", "")

                    # Если это класс, ищем тип атрибута
                    if self._is_class_type(obj_type):
                        if obj_type in self.class_fields:
                            attr_type = self.class_fields[obj_type].get(attr_name)
                            if attr_type:
                                return attr_type

                return "int"

        # Если значение - примитив
        elif isinstance(value, str):
            if value.startswith('"') or value.startswith("'"):
                return "str"
            elif value.isdigit():
                return "int"
            elif value.replace(".", "").isdigit() and "." in value:
                return "float"
            elif value in ["True", "False"]:
                return "bool"
            elif value in ["None", "null"]:
                return "None"
            else:
                # Возможно это переменная
                var_info = self.get_variable_info(value)
                if var_info:
                    return var_info.get("py_type", "int")
                return "int"

        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, bool):
            return "bool"
        elif value is None:
            return "None"

        return "int"

    def _infer_field_type_from_ast(self, ast: Dict, context_vars: Dict = None) -> str:
        """Определяет тип поля по AST выражению (без хардкода)"""
        if not ast:
            return "int"  # По умолчанию

        node_type = ast.get("type", "")

        # Литералы
        if node_type == "literal":
            data_type = ast.get("data_type", "int")
            return data_type

        # Переменные - проверяем в контексте
        elif node_type == "variable":
            var_name = ast.get("value", "")

            # Сначала проверяем переданный контекст (параметры конструктора)
            if context_vars and var_name in context_vars:
                return context_vars[var_name]

            # Затем проверяем объявленные переменные
            var_info = self.get_variable_info(var_name)
            if var_info:
                return var_info.get("py_type", "int")

            # Если переменная не найдена, возвращаем "int" по умолчанию
            return "int"

        # Атрибуты объектов
        elif node_type == "attribute_access":
            obj_name = ast.get("object", "")
            attr_name = ast.get("attribute", "")

            # Если обращаемся к self.атрибут, нужно анализировать граф конструктора
            # чтобы найти тип этого атрибута
            if obj_name == "self":
                # Пока возвращаем "int" по умолчанию
                # TODO: рекурсивный анализ для определения типа атрибута
                return "int"

            return "int"

        # Бинарные операции
        elif node_type == "binary_operation":
            left = ast.get("left", {})
            right = ast.get("right", {})
            operator = ast.get("operator_symbol", "")

            left_type = self._infer_field_type_from_ast(left, context_vars)
            right_type = self._infer_field_type_from_ast(right, context_vars)

            # Для арифметических операций определяем результирующий тип
            if operator in ["+", "-", "*", "/", "//", "%", "**"]:
                # Если один из операндов float/double, результат float
                if "float" in left_type or "double" in left_type:
                    return "float"
                if "float" in right_type or "double" in right_type:
                    return "float"
                # Если оба int, результат int
                if left_type == "int" and right_type == "int":
                    return "int"

            # Для сравнений и логических операций результат bool
            elif operator in ["<", ">", "<=", ">=", "==", "!=", "and", "or", "not"]:
                return "bool"

            return "int"

        # Вызовы функций/методов
        elif node_type == "function_call":
            # Пока возвращаем "int" по умолчанию
            # TODO: анализировать возвращаемый тип функции
            return "int"

        # Списки и кортежи
        elif node_type == "list_literal":
            items = ast.get("items", [])
            if items:
                # Определяем тип первого элемента
                first_item_type = self._infer_field_type_from_ast(
                    items[0], context_vars
                )
                return f"list[{first_item_type}]"
            return "list[int]"

        elif node_type == "tuple_literal":
            items = ast.get("items", [])
            if items:
                # Определяем типы всех элементов
                element_types = []
                for item in items:
                    element_types.append(
                        self._infer_field_type_from_ast(item, context_vars)
                    )

                # Если все элементы одного типа
                if len(set(element_types)) == 1:
                    return f"tuple[{element_types[0]}]"
                else:
                    return f"tuple[{', '.join(element_types)}]"
            return "tuple[int]"

        # Доступ по индексу
        elif node_type == "index_access":
            variable = ast.get("variable", "")
            var_info = self.get_variable_info(variable)
            if var_info:
                py_type = var_info.get("py_type", "")
                if py_type.startswith("list["):
                    # Извлекаем тип элемента списка
                    match = re.match(r"list\[([^\]]+)\]", py_type)
                    if match:
                        return match.group(1)
                elif py_type.startswith("tuple["):
                    # Для кортежей возвращаем тип элемента
                    match = re.match(r"tuple\[([^\]]+)\]", py_type)
                    if match:
                        inner = match.group(1)
                        # Если это один тип (tuple[int])
                        if "," not in inner:
                            return inner
                        # Если это несколько типов (tuple[int, float]), возвращаем первый
                        return inner.split(",")[0].strip()
            return "int"

        # По умолчанию
        return "int"

    def _get_attribute_type(self, obj_name: str, attr_name: str) -> Optional[str]:
        """Получает тип атрибута объекта"""
        # Если это self, ищем в текущем классе
        if obj_name == "self":
            # Ищем текущий класс в scope
            for scope in reversed(self.variable_scopes):
                if "class_name" in scope:
                    class_name = scope.get("class_name")
                    if class_name in self.class_fields:
                        return self.class_fields[class_name].get(attr_name)
            return None

        # Получаем информацию об объекте
        var_info = self.get_variable_info(obj_name)
        if not var_info:
            return None

        obj_type = var_info.get("py_type", "")

        # Если это класс, ищем в его полях
        if self._is_class_type(obj_type):
            if obj_type in self.class_fields:
                return self.class_fields[obj_type].get(attr_name)

        return None

    def resolve_object_path(self, object_path: str):
        """Resolve an object receiver and its C expression.

        Besides local variables and ``self``, methods may be invoked through
        object-valued fields, for example ``self.hidden.forward(x)``.  Keep
        this resolution in the type layer so calls, attributes, and future
        dispatch mechanisms share one implementation.
        """
        parts = [part for part in str(object_path).split(".") if part]
        if not parts:
            return "", object_path

        first = parts[0]
        if first == "self":
            current_type = self._get_current_class() or ""
            c_expression = "self"
        else:
            info = self.get_variable_info(first)
            if not info:
                return "", object_path
            current_type = self.strip_borrow_type(info.get("py_type", ""))
            c_expression = first

        for attribute in parts[1:]:
            current_type = self.strip_borrow_type(current_type)
            if not self._is_class_type(current_type):
                return "", c_expression
            field_type, field_expression = self.resolve_class_field(
                current_type, c_expression, attribute
            )
            if not field_type:
                return "", c_expression
            c_expression = field_expression
            current_type = field_type

        return current_type, c_expression

    def resolve_class_field(self, class_name: str, object_expression: str, attribute: str):
        """Resolve a class field, including fields inherited through ``base``.

        Derived objects embed their parent at offset zero.  A field declared by
        the current class is therefore addressed with ``->field`` while a
        field declared by an ancestor is addressed through one or more
        embedded ``base`` members (for example ``self->base.value``).
        """
        current = self.strip_borrow_type(class_name)
        path = object_expression
        visited = set()

        while current:
            if current in visited:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            visited.add(current)

            fields = self.class_fields.get(current, {})
            if attribute in fields:
                if current == class_name:
                    expression = f"{path}->{attribute}"
                else:
                    expression = f"{path}.{attribute}"
                return fields[attribute], expression

            parents = self.class_hierarchy.get(current, [])
            if not parents:
                break
            if len(parents) > 1:
                raise RuntimeError(
                    f"multiple inheritance for class '{class_name}' is not supported"
                )
            path = f"{path}->base"
            current = self.strip_borrow_type(parents[0])

        return None, None
