from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class CallsMixin:
    def generate_function_call(self, node: Dict):
        """Генерирует вызов функции"""
        func_name = node.get("function", "")
        args = node.get("arguments", [])

        # Удаляем @ из имени функции для C кода
        if func_name.startswith("@"):
            func_name = func_name[1:]

        # Генерируем аргументы
        arg_strings = []
        for arg in args:
            if isinstance(arg, dict):
                arg_strings.append(self.generate_expression(arg))
            else:
                if str(arg) == "None":
                    arg_strings.append("NULL")
                else:
                    arg_strings.append(str(arg))

        args_str = ", ".join(arg_strings)
        self.add_line(f"{func_name}({args_str});")

    def generate_method_call(self, node: Dict):
        """Генерирует вызов метода объекта"""
        object_name = node.get("object", "")
        method_name = node.get("method", "")
        args = node.get("arguments", [])
        is_standalone = node.get("is_standalone", False)  # Новое поле из парсера
        # Добавляем получение целевой переменной для присваивания
        target_var = node.get("target", "")  # Добавьте эту строку

        # Проверяем тип объекта
        var_info = self.get_variable_info(object_name)
        if not var_info:
            self.add_line(f"// ERROR: Объект '{object_name}' не найден")
            return

        obj_type = var_info.get("py_type", "")

        # Генерируем аргументы
        arg_strings = []
        for arg in args:
            if isinstance(arg, dict):
                arg_strings.append(self.generate_expression(arg))
            else:
                arg_strings.append(str(arg))

        args_str = ", ".join(arg_strings) if arg_strings else ""

        if self._is_class_type(obj_type):
            # Это класс - используем формат ClassName_methodName
            # Первым аргументом идет указатель на объект (self)
            full_args = f"{object_name}"
            if args_str:
                full_args = f"{object_name}, {args_str}"

            # Если это standalone вызов (statement)
            if is_standalone:
                self.add_line(f"{obj_type}_{method_name}({full_args});")
            else:
                # Если это выражение (возвращаем результат)
                return f"{obj_type}_{method_name}({full_args})"

        # Для атрибутов объектов (self.attribute)
        elif object_name == "self":
            # self.get(i, j) должно стать Matrix_get(self, i, j)
            # Получаем класс из текущего scope
            current_scope = None
            for scope in reversed(self.variable_scopes):
                if "class_name" in scope:
                    current_scope = scope
                    break

            if current_scope:
                class_name = current_scope.get("class_name", "")
                full_args = f"self"
                if args_str:
                    full_args = f"self, {args_str}"

                if is_standalone:
                    self.add_line(f"{class_name}_{method_name}({full_args});")
                else:
                    return f"{class_name}_{method_name}({full_args})"
        # Обработка методов для списков
        # Обработка методов для строк
        elif obj_type == "str":
            if method_name == "upper":
                if is_standalone:
                    # Для a.upper() как standalone - результат должен быть присвоен обратно в a
                    self.add_line("// upper")
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(f"char* {temp_var} = string_upper({object_name});")
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для upper() внутри выражения
                    return f"string_upper({object_name})"

            elif method_name == "lower":
                if is_standalone:
                    self.add_line("// lower")
                    # Для a.lower() как standalone
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(f"char* {temp_var} = string_lower({object_name});")
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для lower() внутри выражения
                    return f"string_lower({object_name})"

            elif method_name == "capitalize":
                if is_standalone:
                    self.add_line("// capitalize")

                    # Для a.capitalize() как standalone
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(
                        f"char* {temp_var} = string_capitalize({object_name});"
                    )
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для capitalize() внутри выражения
                    return f"string_capitalize({object_name})"

            elif method_name == "title":
                if is_standalone:
                    self.add_line("// title")

                    # Для a.title() как standalone
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(f"char* {temp_var} = string_title({object_name});")
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для title() внутри выражения
                    return f"string_title({object_name})"

            elif method_name == "strip":
                if is_standalone:
                    self.add_line("// strip")

                    # Для a.strip() как standalone
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(f"char* {temp_var} = string_strip({object_name});")
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для strip() внутри выражения
                    return f"string_strip({object_name})"

            elif method_name == "lstrip":
                if is_standalone:
                    self.add_line("// lstrip")

                    # Для a.lstrip() как standalone
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(f"char* {temp_var} = string_lstrip({object_name});")
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для lstrip() внутри выражения
                    return f"string_lstrip({object_name})"

            elif method_name == "rstrip":
                if is_standalone:
                    self.add_line("// rstrip")

                    # Для a.rstrip() как standalone
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(f"char* {temp_var} = string_rstrip({object_name});")
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для rstrip() внутри выражения
                    return f"string_rstrip({object_name})"

            elif method_name == "format":
                if is_standalone:
                    self.add_line("// format")

                    # Для a.format("world") как standalone - результат должен быть присвоен обратно в a
                    temp_var = self.generate_temporary_var("str")
                    self.add_line(
                        f"char* {temp_var} = string_format({object_name}, {args_str});"
                    )
                    # Освобождаем старую строку
                    self.add_line(f"if ({object_name}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({object_name});")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    # Присваиваем новую строку
                    self.add_line(f"{object_name} = {temp_var};")
                else:
                    # Для format() внутри выражения
                    return f"string_format({object_name}, {args_str})"

            elif method_name == "split":
                # Определяем разделитель
                if len(arg_strings) > 0:
                    delimiter = arg_strings[0]
                else:
                    delimiter = '" "'  # По умолчанию пробел

                if is_standalone:
                    # Для a.split() как standalone - результат игнорируется
                    temp_var = self.generate_temporary_var("str_list")
                    self.add_line(
                        f"string_list* {temp_var} = string_split({object_name}, {delimiter});"
                    )
                    self.add_line(f"// Результат split() игнорируется")
                else:
                    # Для split() внутри выражения - возвращаем результат
                    return f"string_split({object_name}, {delimiter})"

            elif method_name == "replace":
                # replace(old, new) - заменяет все вхождения подстроки
                if len(arg_strings) >= 2:
                    old = arg_strings[0]
                    new = arg_strings[1]

                    if is_standalone:
                        self.add_line("// replace")
                        temp_var = self.generate_temporary_var("str")
                        self.add_line(
                            f"char* {temp_var} = string_replace({object_name}, {old}, {new});"
                        )
                        # Освобождаем старую строку
                        self.add_line(f"if ({object_name}) {{")
                        self.indent_level += 1
                        self.add_line(f"free({object_name});")
                        self.indent_level -= 1
                        self.add_line(f"}}")
                        # Присваиваем новую строку
                        self.add_line(f"{object_name} = {temp_var};")
                    else:
                        if target_var:
                            self.add_line(
                                f"{target_var} = string_replace({object_name}, {old}, {new});"
                            )
                        else:
                            return f"string_replace({object_name}, {old}, {new})"
                else:
                    self.add_line(f"// replace() requires 2 arguments")

        elif obj_type.startswith("list["):
            if method_name == "append":
                if args_str:
                    struct_name = self.generate_list_struct_name(obj_type)
                    self.add_line(f"append_{struct_name}({object_name}, {args_str});")

            elif method_name == "extend":
                if args_str:
                    # args[0] должен быть другим списком
                    struct_name = self.generate_list_struct_name(obj_type)
                    other_list = arg_strings[0]
                    self.add_line(f"// extend: добавление элементов из другого списка")
                    self.add_line(f"for (int i = 0; i < {other_list}->size; i++) {{")
                    self.indent_level += 1
                    self.add_line(
                        f"append_{struct_name}({object_name}, {other_list}->data[i]);"
                    )
                    self.indent_level -= 1
                    self.add_line("}")

            elif method_name == "insert":
                if len(arg_strings) >= 2:
                    index_var = arg_strings[0]
                    value_var = arg_strings[1]
                    struct_name = self.generate_list_struct_name(obj_type)
                    self.add_line(
                        f"if ({index_var} >= 0 && {index_var} <= {object_name}->size) {{"
                    )
                    self.indent_level += 1
                    self.add_line(
                        f"if ({object_name}->size >= {object_name}->capacity) {{"
                    )
                    self.indent_level += 1
                    self.add_line(
                        f"{object_name}->capacity = {object_name}->capacity == 0 ? {INITIAL_LIST_CAPACITY} : {object_name}->capacity * 2;"
                    )
                    self.add_line(
                        f"{object_name}->data = realloc({object_name}->data, {object_name}->capacity * sizeof(int));"
                    )
                    self.indent_level -= 1
                    self.add_line("}")
                    self.add_line(
                        f"for (int i = {object_name}->size; i > {index_var}; i--) {{"
                    )
                    self.indent_level += 1
                    self.add_line(
                        f"{object_name}->data[i] = {object_name}->data[i - 1];"
                    )
                    self.indent_level -= 1
                    self.add_line("}")
                    self.add_line(f"{object_name}->data[{index_var}] = {value_var};")
                    self.add_line(f"{object_name}->size++;")
                    self.indent_level -= 1
                    self.add_line("}")

            elif method_name == "remove":
                if args_str:
                    value_var = arg_strings[0]
                    self.add_line(f"// remove первый элемент со значением {value_var}")
                    self.add_line(f"int found_index = -1;")
                    self.add_line(f"for (int i = 0; i < {object_name}->size; i++) {{")
                    self.indent_level += 1
                    self.add_line(f"if ({object_name}->data[i] == {value_var}) {{")
                    self.indent_level += 1
                    self.add_line(f"found_index = i;")
                    self.add_line(f"break;")
                    self.indent_level -= 1
                    self.add_line("}")
                    self.indent_level -= 1
                    self.add_line("}")
                    self.add_line(f"if (found_index != -1) {{")
                    self.indent_level += 1
                    self.add_line(
                        f"for (int i = found_index; i < {object_name}->size - 1; i++) {{"
                    )
                    self.indent_level += 1
                    self.add_line(
                        f"{object_name}->data[i] = {object_name}->data[i + 1];"
                    )
                    self.indent_level -= 1
                    self.add_line("}")
                    self.add_line(f"{object_name}->size--;")
                    self.indent_level -= 1
                    self.add_line("}")

            elif method_name == "pop":
                # Проверяем, есть ли присваивание результата
                # Если is_standalone == False, значит результат используется (в выражении или присваивании)
                if not args_str:
                    # pop() без аргументов - удалить последний
                    self.add_line(f"if ({object_name} && {object_name}->size > 0) {{")
                    self.indent_level += 1

                    # Получаем удаляемое значение
                    temp_var = self.generate_temporary_var("int")
                    self.add_line(
                        f"int {temp_var} = {object_name}->data[{object_name}->size - 1];"
                    )

                    # Уменьшаем размер
                    self.add_line(f"{object_name}->size--;")

                    # Если результат используется (не standalone), присваиваем его
                    if not is_standalone:
                        if target_var:
                            self.add_line(f"{target_var} = {temp_var};")
                        else:
                            self.add_line(
                                f"// Результат pop() используется, но не присвоен"
                            )

                    self.indent_level -= 1
                    self.add_line("} else {")
                    self.indent_level += 1
                    self.add_line(
                        f'fprintf(stderr, "IndexError: pop from empty list\\n");'
                    )
                    self.add_line(f"exit(1);")
                    self.indent_level -= 1
                    self.add_line("}")
                else:
                    # pop(index) - удалить по индексу
                    index_var = arg_strings[0]
                    self.add_line(
                        f"if ({object_name} && {index_var} >= 0 && {index_var} < {object_name}->size) {{"
                    )
                    self.indent_level += 1

                    # Получаем удаляемое значение
                    temp_var = self.generate_temporary_var("int")
                    self.add_line(f"int {temp_var} = {object_name}->data[{index_var}];")

                    # Сдвигаем элементы
                    self.add_line(
                        f"for (int i = {index_var}; i < {object_name}->size - 1; i++) {{"
                    )
                    self.indent_level += 1
                    self.add_line(
                        f"{object_name}->data[i] = {object_name}->data[i + 1];"
                    )
                    self.indent_level -= 1
                    self.add_line("}")

                    # Уменьшаем размер
                    self.add_line(f"{object_name}->size--;")

                    # Если результат используется (не standalone), присваиваем его
                    if not is_standalone:
                        if target_var:
                            self.add_line(f"{target_var} = {temp_var};")
                        else:
                            self.add_line(
                                f"// Результат pop() используется, но не присвоен"
                            )

                    self.indent_level -= 1
                    self.add_line("} else {")
                    self.indent_level += 1
                    self.add_line(
                        f'fprintf(stderr, "IndexError: pop index out of range\\n");'
                    )
                    self.add_line(f"exit(1);")
                    self.indent_level -= 1
                    self.add_line("}")

            elif method_name == "clear":
                self.add_line(f"{object_name}->size = 0;")

            elif method_name == "index":
                if args_str:
                    value_var = arg_strings[0]
                    temp_var = self.generate_temporary_var("int")
                    self.add_line(f"int {temp_var} = -1;")
                    self.add_line(f"for (int i = 0; i < {object_name}->size; i++) {{")
                    self.indent_level += 1
                    self.add_line(f"if ({object_name}->data[i] == {value_var}) {{")
                    self.indent_level += 1
                    self.add_line(f"{temp_var} = i;")
                    self.add_line(f"break;")
                    self.indent_level -= 1
                    self.add_line("}")
                    self.indent_level -= 1
                    self.add_line("}")
                    # TODO: Проверить на -1 и выдать ошибку как в Python

            elif method_name == "count":
                if args_str:
                    value_var = arg_strings[0]
                    temp_var = self.generate_temporary_var("int")
                    self.add_line(f"int {temp_var} = 0;")
                    self.add_line(f"for (int i = 0; i < {object_name}->size; i++) {{")
                    self.indent_level += 1
                    self.add_line(f"if ({object_name}->data[i] == {value_var}) {{")
                    self.indent_level += 1
                    self.add_line(f"{temp_var}++;")
                    self.indent_level -= 1
                    self.add_line("}")
                    self.indent_level -= 1
                    self.add_line("}")

            elif method_name == "sort":
                # Используем qsort для эффективности
                # Определяем тип элементов списка
                match = re.match(r"list\[([^\]]+)\]", obj_type)
                element_type = match.group(1) if match else "int"
                c_element_type = self.map_type_to_c(element_type)

                # Выбираем соответствующую функцию сравнения
                if element_type == "int":
                    compare_func = "compare_int"
                elif element_type == "float":
                    compare_func = "compare_float"
                elif element_type == "double":
                    compare_func = "compare_double"
                elif element_type == "str":
                    compare_func = "compare_string"
                else:
                    # По умолчанию для неизвестных типов
                    compare_func = "compare_int"

                self.add_line(
                    f"qsort({object_name}->data, {object_name}->size, sizeof({c_element_type}), {compare_func});"
                )

            elif method_name == "reverse":
                self.add_line(f"for (int i = 0; i < {object_name}->size / 2; i++) {{")
                self.indent_level += 1
                self.add_line(f"int temp = {object_name}->data[i];")
                self.add_line(
                    f"{object_name}->data[i] = {object_name}->data[{object_name}->size - i - 1];"
                )
                self.add_line(
                    f"{object_name}->data[{object_name}->size - i - 1] = temp;"
                )
                self.indent_level -= 1
                self.add_line("}")

            else:
                self.add_line(f"// Метод списка '{method_name}' не реализован")

        # Обработка методов для кортежей
        elif obj_type.startswith("tuple["):
            if method_name == "count":
                if args_str:
                    struct_name = self.generate_tuple_struct_name(obj_type)
                    self.add_line(f"// count в кортеже")
                    temp_var = self.generate_temporary_var("int")
                    self.add_line(f"int {temp_var} = 0;")
                    self.add_line(f"for (int i = 0; i < {object_name}.size; i++) {{")
                    self.indent_level += 1
                    self.add_line(f"if ({object_name}.data[i] == {args_str}) {{")
                    self.indent_level += 1
                    self.add_line(f"{temp_var}++;")
                    self.indent_level -= 1
                    self.add_line("}")
                    self.indent_level -= 1
                    self.add_line("}")
                    # Возвращаем значение
                    # Но в вашем коде нет присваивания результата, так что просто вычисляем

            elif method_name == "index":
                if args_str:
                    struct_name = self.generate_tuple_struct_name(obj_type)
                    self.add_line(f"// index в кортеже")
                    temp_var = self.generate_temporary_var("int")
                    self.add_line(f"int {temp_var} = -1;")
                    self.add_line(f"for (int i = 0; i < {object_name}.size; i++) {{")
                    self.indent_level += 1
                    self.add_line(
                        f"if ({object_name}.data[i] == {args_str} && {temp_var} == -1) {{"
                    )
                    self.indent_level += 1
                    self.add_line(f"{temp_var} = i;")
                    self.indent_level -= 1
                    self.add_line("}")
                    self.indent_level -= 1
                    self.add_line("}")
                    # Возвращаем значение
                    # Но в вашем коде нет присваивания результата

            else:
                self.add_line(f"// Метод '{method_name}' для кортежа не реализован")

        # Обработка методов для словарей
        elif obj_type.startswith("dict["):
            # Извлекаем типы ключа и значения
            key_type, value_type = self._extract_dict_types(obj_type)
            key_name = self.clean_type_name_for_c(key_type)
            value_name = self.clean_type_name_for_c(value_type)
            struct_name = f"dict_{key_name}_{value_name}"

            if method_name == "keys":
                # keys() - возвращает список ключей
                list_struct = f"list_{key_name}"

                if target_var:
                    self.add_line(
                        f"{list_struct}* {target_var} = keys_{struct_name}({object_name});"
                    )
                elif is_standalone:
                    # Если вызов без присваивания, создаем временную переменную и освобождаем
                    temp_var = self.generate_temporary_var(list_struct)
                    self.add_line(
                        f"{list_struct}* {temp_var} = keys_{struct_name}({object_name});"
                    )
                    self.add_line(f"free_{list_struct}({temp_var});")
                else:
                    # Если используется в выражении, возвращаем вызов функции
                    return f"keys_{struct_name}({object_name})"

            elif method_name == "values":
                # values() - возвращает список значений
                list_struct = f"list_{value_name}"

                if target_var:
                    self.add_line(
                        f"{list_struct}* {target_var} = values_{struct_name}({object_name});"
                    )
                elif is_standalone:
                    temp_var = self.generate_temporary_var(list_struct)
                    self.add_line(
                        f"{list_struct}* {temp_var} = values_{struct_name}({object_name});"
                    )
                    self.add_line(f"free_{list_struct}({temp_var});")
                else:
                    return f"values_{struct_name}({object_name})"

            elif method_name == "items":
                # items() - возвращает список пар (ключ, значение)
                # Для этого нужно создать структуру для пары
                pair_struct = f"{struct_name}_pair"
                list_pair_struct = f"list_{key_name}_{value_name}_pair"

                # Создаем структуру для списка пар, если еще не создана
                if list_pair_struct not in self.generated_structures:
                    # Здесь нужно сгенерировать структуру для списка пар
                    pass

                # TODO: реализовать items()
                self.add_line(f"// items() method not fully implemented yet")

            elif method_name == "get":
                # get(key, default_value)
                if len(arg_strings) >= 1:
                    key_arg = arg_strings[0]

                    # Определяем значение по умолчанию
                    if len(arg_strings) >= 2:
                        default_arg = arg_strings[1]
                    else:
                        # Если default не указан, используем 0, NULL или false в зависимости от типа
                        if value_type == "int":
                            default_arg = "0"
                        elif value_type == "float" or value_type == "double":
                            default_arg = "0.0"
                        elif value_type == "bool":
                            default_arg = "false"
                        elif value_type == "str" or value_type == "char*":
                            default_arg = "NULL"
                        else:
                            default_arg = "0"

                    if target_var:
                        # Присваивание результата переменной
                        self.add_line(
                            f"{target_var} = get_default_{struct_name}({object_name}, {key_arg}, {default_arg});"
                        )
                    elif is_standalone:
                        # Вызов без присваивания (обычно так не делают, но поддержим)
                        temp_var = self.generate_temporary_var(value_type)
                        self.add_line(
                            f"{self.map_type_to_c(value_type)} {temp_var} = get_default_{struct_name}({object_name}, {key_arg}, {default_arg});"
                        )
                    else:
                        # Используется в выражении (например, в print)
                        return f"get_default_{struct_name}({object_name}, {key_arg}, {default_arg})"
                else:
                    self.add_line("// get() requires at least a key argument")
                return

            else:
                self.add_line(f"// Метод словаря '{method_name}' не реализован")

            return

    def generate_object_method_call(self, node: Dict):
        """Генерирует вызов метода объекта (obj.method())"""
        object_name = node.get(
            "class_name", ""
        )  # В JSON это поле называется class_name
        method_name = node.get("method", "")
        args = node.get("arguments", [])

        logger.debug(
            f"DEBUG generate_object_method_call: {object_name}.{method_name}()"
        )

        # Универсальная реализация
        var_info = self.get_variable_info(object_name)
        if not var_info:
            self.add_line(f"// ERROR: Object '{object_name}' not found")
            return

        object_type = var_info.get("py_type", "")
        if not object_type:
            self.add_line(f"// ERROR: No type for '{object_name}'")
            return

        # ПРОВЕРЯЕМ, ЯВЛЯЕТСЯ ЛИ ОБЪЕКТ СПИСКОМ
        if object_type.startswith("list["):
            # Для списков используем специальные функции
            if method_name == "append" and args:
                # Получаем правильное имя структуры
                struct_name = self.generate_list_struct_name(object_type)
                logger.debug(
                    f"DEBUG: append для {object_name} типа {object_type}, struct_name={struct_name}"
                )

                # Генерируем аргумент
                arg = args[0]
                if isinstance(arg, dict):
                    arg_expr = self.generate_expression(arg)
                else:
                    arg_expr = str(arg)

                # Генерируем правильный вызов
                self.add_line(f"append_{struct_name}({object_name}, {arg_expr});")
                return

            # Добавляем обработку других методов списков если нужно
            else:
                self.add_line(f"// Метод списка '{method_name}' для типа {object_type}")
                return

        # ДЛЯ КЛАССОВ (не списков) - оригинальная логика
        # Генерируем аргументы
        arg_exprs = []
        for arg in args:
            if isinstance(arg, dict):
                expr = self.generate_expression(arg)
                if expr is None:
                    expr = "0"
                arg_exprs.append(expr)
            else:
                arg_exprs.append(str(arg))

        # Собираем все аргументы: self + остальные
        all_args = [object_name] + arg_exprs
        args_str = ", ".join(all_args)

        # Формируем имя функции: TypeName_methodName
        # Но для списков это не должно использоваться!
        self.add_line(f"{object_type}_{method_name}({args_str});")

    def generate_c_call(self, node: Dict):
        """Генерирует прямой вызов C-функции"""
        func_name = node.get("function", "")
        args = node.get("arguments", [])

        # Генерируем аргументы
        arg_strings = []
        for arg in args:
            if isinstance(arg, dict):
                # Если аргумент - AST, генерируем выражение
                arg_strings.append(self.generate_expression(arg))
            else:
                # Если это простая строка
                arg_strings.append(str(arg))

        args_str = ", ".join(arg_strings)

        # Просто генерируем вызов C-функции
        self.add_line(f"{func_name}({args_str});")

    def generate_builtin_function_call(self, node: Dict):
        """Генерирует вызов встроенной функции"""
        func_name = node.get("function", "")
        args = node.get("arguments", [])
        kwargs: dict = node.get("kwargs", {})  # Получаем опции

        if func_name == "print":
            # Генерируем printf для print
            if not args:
                self.add_line('printf("\\n");')
                return

            # Создаем форматную строку
            format_parts = []
            value_parts = []

            for arg in args:
                if isinstance(arg, dict):
                    if arg.get("type") == "attribute_access":
                        expr = self.generate_attribute_access(arg)
                        format_parts.append("%d")
                        value_parts.append(expr)
                    elif arg.get("type") == "variable":
                        var_name = arg.get("value", "")
                        var_info = self.get_variable_info(var_name)
                        if var_info:
                            var_type = var_info.get("py_type", "")
                            if var_type == "int":
                                format_parts.append("%d")
                                value_parts.append(var_name)
                            elif var_type in ["float", "double"]:
                                format_parts.append("%f")
                                value_parts.append(var_name)
                            elif var_type == "str":
                                format_parts.append("%s")
                                value_parts.append(var_name)
                            else:
                                format_parts.append("%d")
                                value_parts.append(var_name)
                        else:
                            format_parts.append("%d")
                            value_parts.append(var_name)
                    elif arg.get("type") == "literal":
                        value = arg.get("value", "")
                        data_type = arg.get("data_type", "")
                        if data_type == "str":
                            format_parts.append("%s")
                            value_parts.append(f'"{value}"')
                        else:
                            format_parts.append("%d")
                            value_parts.append(str(value))
                    else:
                        expr = self.generate_expression(arg)
                        format_parts.append("%d")
                        value_parts.append(expr)
                else:
                    format_parts.append("%d")
                    value_parts.append(str(arg))

            sep_node = kwargs.get("sep", {})
            end_node = kwargs.get("end", {})

            sep = sep_node.get("value", " ")
            end = end_node.get("value", "\\n")

            # Собираем форматную строку
            format_str = '"' + sep.join(format_parts) + f'{end}"'
            args_str = ", ".join(value_parts)

            self.add_line(f"printf({format_str}, {args_str});")
        elif func_name == "input":
            # Для input() без присваивания
            self.generate_input_statement(node)
            return
        else:
            # Обработка других встроенных функций
            # Генерируем аргументы
            arg_strings = []
            for arg in args:
                if isinstance(arg, dict):
                    arg_strings.append(self.generate_expression(arg))
                else:
                    arg_strings.append(str(arg))

            args_str = ", ".join(arg_strings)

            # Маппинг других встроенных функций
            builtin_map = {
                "len": "builtin_len",
                "str": "builtin_str",
                "int": "builtin_int",
                "bool": "builtin_bool",
                "range": "builtin_range",
            }

            c_func_name = builtin_map.get(func_name, func_name)
            self.add_line(f"{c_func_name}({args_str});")

    def generate_builtin_function_call_assignment(self, node: Dict):
        """Генерирует присваивание результата встроенной функции"""
        target = node.get("symbols", [])[0] if node.get("symbols") else ""
        func_name = node.get("function", "")
        args = node.get("arguments", [])
        return_type = node.get("return_type", "")

        if not target:
            # Просто вызов функции без присваивания
            if func_name == "input":
                self.generate_input_statement(node)
            else:
                self.generate_builtin_function_call(node)
            return

        var_info = self.get_variable_info(target)
        if not var_info:
            node_type = node.get("var_type", "int")
            # Для input() по умолчанию возвращается строка
            if func_name == "input" and not node_type:
                node_type = "str"
            self.declare_variable(target, node_type)
            var_info = self.get_variable_info(target)

        # Специальная обработка для input()
        if func_name == "input":
            c_type = var_info["c_type"] if var_info else "char*"

            # Генерируем prompt если есть аргументы
            if args:
                self._generate_input_prompt(args)

            # Для разных типов переменных разная обработка
            if c_type == "char*":
                # Для строковых переменных - прямой ввод в целевую переменную
                self._generate_input_read_code_direct(target)
            else:
                # Для других типов (int, float и т.д.)
                buffer_var = f"{target}_input_buffer"
                self.add_line(f"char {buffer_var}[256];")
                self.add_line(f"fgets({buffer_var}, sizeof({buffer_var}), stdin);")
                self.add_line(f'{buffer_var}[strcspn({buffer_var}, "\\n")] = 0;')

                if c_type == "int":
                    self.add_line(f"{target} = atoi({buffer_var});")
                elif c_type == "float" or c_type == "double":
                    self.add_line(f"{target} = atof({buffer_var});")
                elif c_type == "bool":
                    self.add_line(
                        f'{target} = (strcmp({buffer_var}, "true") == 0 || strcmp({buffer_var}, "1") == 0);'
                    )
                else:
                    self.add_line(f"// Неподдерживаемый тип для input: {c_type}")
                    self.add_line(f"{target} = 0;")

            return

        elif func_name == "float":
            c_type = var_info["c_type"] if var_info else "float"

            if args:
                arg_expr = self.generate_expression(args[0])
                var_info = self.get_variable_info(target)

                if var_info:
                    py_type = var_info.get("py_type", "")

                    # Конвертация из строки в float
                    self.add_line(f"{c_type} {target} = atof({arg_expr});")
                else:
                    self.add_line(f"{c_type} {target} = atof({arg_expr});")
            return

        # Специальная обработка для str()
        elif func_name == "str":
            c_type = var_info["c_type"] if var_info else "char*"

            if args:
                arg_expr = self.generate_expression(args[0])

                # Определяем тип аргумента для выбора правильной функции
                if isinstance(args[0], dict):
                    arg_type = args[0].get("type", "")
                    if arg_type == "variable":
                        var_name = args[0].get("value", "")
                        arg_var_info = self.get_variable_info(var_name)
                        if arg_var_info:
                            py_type = arg_var_info.get("py_type", "")

                            if py_type == "int":
                                self.add_line(
                                    f"{c_type} {target} = builtin_str_int({arg_expr});"
                                )
                            elif py_type == "float" or py_type == "double":
                                self.add_line(
                                    f"{c_type} {target} = builtin_str_float({arg_expr});"
                                )
                            elif py_type == "bool":
                                self.add_line(
                                    f"{c_type} {target} = builtin_str_bool({arg_expr});"
                                )
                            elif py_type == "str":
                                self.add_line(
                                    f"{c_type} {target} = malloc(strlen({arg_expr}) + 1);"
                                )
                                self.add_line(f"strcpy({target}, {arg_expr});")
                            else:
                                self.add_line(
                                    f'{c_type} {target} = builtin_str({arg_expr}, "{py_type}");'
                                )
                        else:
                            self.add_line(
                                f"{c_type} {target} = builtin_str_int({arg_expr});"
                            )
                    elif arg_type == "literal":
                        data_type = args[0].get("data_type", "")
                        if data_type == "int":
                            self.add_line(
                                f"{c_type} {target} = builtin_str_int({arg_expr});"
                            )
                        elif data_type == "float":
                            self.add_line(
                                f"{c_type} {target} = builtin_str_float({arg_expr});"
                            )
                        elif data_type == "bool":
                            self.add_line(
                                f"{c_type} {target} = builtin_str_bool({arg_expr});"
                            )
                        else:
                            self.add_line(
                                f"{c_type} {target} = builtin_str_int({arg_expr});"
                            )
                    else:
                        self.add_line(
                            f"{c_type} {target} = builtin_str_int({arg_expr});"
                        )
                else:
                    self.add_line(f"{c_type} {target} = builtin_str_int({arg_expr});")
            return

        # Маппинг встроенных функций Python -> C
        builtin_map = {
            "len": "builtin_len",
            "str": "builtin_str",
            "int": "builtin_int",
            "bool": "builtin_bool",
            "range": "builtin_range",
        }

        c_func_name = builtin_map.get(func_name, func_name)

        # Генерируем аргументы
        arg_strings = []
        for arg in args:
            if isinstance(arg, dict):
                arg_strings.append(self.generate_expression(arg))
            else:
                arg_strings.append(str(arg))

        args_str = ", ".join(arg_strings)

        # Специальная обработка для len()
        if func_name == "len" and args:
            # Определяем тип аргумента
            arg_expr = args[0]
            if isinstance(arg_expr, dict):
                # Если это переменная, получаем ее тип
                if arg_expr.get("type") == "variable":
                    var_name = arg_expr.get("value", "")
                    var_info = self.get_variable_info(var_name)
                    if var_info:
                        py_type = var_info.get("py_type", "")
                        if py_type.startswith("tuple["):
                            struct_name = self.generate_tuple_struct_name(py_type)
                            c_func_name = f"builtin_len_{struct_name}"
                        elif py_type.startswith("list["):
                            struct_name = self.generate_list_struct_name(py_type)
                            c_func_name = f"builtin_len_{struct_name}"

        c_type = var_info["c_type"] if var_info else self.map_type_to_c(return_type)
        self.add_line(f"{c_type} {target} = {c_func_name}({args_str});")

    def generate_constructor_call(self, ast: Dict) -> str:
        """Генерирует вызов конструктора"""
        class_name = ast.get("class_name", "")
        args = ast.get("arguments", [])

        # Генерируем аргументы
        arg_strings = []
        for arg in args:
            if isinstance(arg, dict):
                arg_strings.append(self.generate_expression(arg))
            else:
                arg_strings.append(str(arg))

        args_str = ", ".join(arg_strings)
        return f"create_{class_name}({args_str})"

    def _generate_builtin_declaration(
        self, var_name: str, c_type: str, call_ast: Dict, is_redeclaration: bool
    ):
        """Генерирует объявление с вызовом builtin функции"""
        func_name = call_ast.get("function", "")
        args = call_ast.get("arguments", [])

        if not args:
            if is_redeclaration:
                self.add_line(f"{var_name} = 0;")
            else:
                self.add_line(f"{c_type} {var_name} = 0;")
            return

        arg_expr = self.generate_expression(args[0])

        # Определяем тип аргумента для выбора правильной функции
        arg_type = "unknown"
        if args and isinstance(args[0], dict):
            if args[0].get("type") == "variable":
                var_name_arg = args[0].get("value", "")
                arg_var_info = self.get_variable_info(var_name_arg)
                if arg_var_info:
                    arg_type = arg_var_info.get("py_type", "unknown")
            elif args[0].get("type") == "literal":
                arg_type = args[0].get("data_type", "unknown")

        # Выбираем правильную реализацию для str()
        if func_name == "str":
            if arg_type == "int":
                func_call = f"builtin_str_int({arg_expr})"
            elif arg_type == "float":
                func_call = f"builtin_str_float({arg_expr})"
            elif arg_type == "double":
                func_call = f"builtin_str_double({arg_expr})"
            elif arg_type == "bool":
                func_call = f"builtin_str_bool({arg_expr})"
            elif arg_type == "str":
                # Для строки - просто копируем
                if is_redeclaration:
                    self.add_line(f"if ({var_name}) free({var_name});")
                    self.add_line(f"{var_name} = malloc(strlen({arg_expr}) + 1);")
                    self.add_line(f"if ({var_name}) strcpy({var_name}, {arg_expr});")
                else:
                    self.add_line(
                        f"{c_type} {var_name} = malloc(strlen({arg_expr}) + 1);"
                    )
                    self.add_line(f"if ({var_name}) strcpy({var_name}, {arg_expr});")
                return
            else:
                # По умолчанию для неизвестного типа используем float
                func_call = f"builtin_str_float({arg_expr})"

            if is_redeclaration:
                self.add_line(f"{var_name} = {func_call};")
            else:
                self.add_line(f"{c_type} {var_name} = {func_call};")
            return

        # Для int()
        elif func_name == "int":
            if arg_type == "str":
                func_call = f"atoi({arg_expr})"
            elif arg_type == "float" or arg_type == "double":
                func_call = f"(int)({arg_expr})"
            elif arg_type == "bool":
                func_call = f"({arg_expr} ? 1 : 0)"
            else:
                func_call = f"({int})({arg_expr})"

            if is_redeclaration:
                self.add_line(f"{var_name} = {func_call};")
            else:
                self.add_line(f"{c_type} {var_name} = {func_call};")
            return

        # Для float()
        elif func_name == "float":
            if arg_type == "str":
                func_call = f"atof({arg_expr})"
            elif arg_type == "int":
                func_call = f"(float)({arg_expr})"
            elif arg_type == "bool":
                func_call = f"({arg_expr} ? 1.0 : 0.0)"
            else:
                func_call = f"(float)({arg_expr})"

            if is_redeclaration:
                self.add_line(f"{var_name} = {func_call};")
            else:
                self.add_line(f"{c_type} {var_name} = {func_call};")
            return

        # Для bool()
        elif func_name == "bool":
            if arg_type == "int":
                func_call = f"({arg_expr} != 0)"
            elif arg_type == "float" or arg_type == "double":
                func_call = f"({arg_expr} != 0.0)"
            elif arg_type == "str":
                func_call = f"({arg_expr} && strlen({arg_expr}) > 0)"
            else:
                func_call = f"({arg_expr} != 0)"

            if is_redeclaration:
                self.add_line(f"{var_name} = {func_call};")
            else:
                self.add_line(f"{c_type} {var_name} = {func_call};")
            return

        # Для len()
        elif func_name == "len":
            # Определяем функцию len в зависимости от типа аргумента
            if arg_type.startswith("list["):
                struct_name = self.generate_list_struct_name(arg_type)
                func_call = f"builtin_len_{struct_name}({arg_expr})"
            elif arg_type.startswith("dict["):
                key_type, value_type = self._extract_dict_types(arg_type)
                key_name = self.clean_type_name_for_c(key_type)
                value_name = self.clean_type_name_for_c(value_type)
                struct_name = f"dict_{key_name}_{value_name}"
                func_call = f"len_{struct_name}({arg_expr})"
            elif arg_type == "str":
                func_call = f"strlen({arg_expr})"
            else:
                func_call = f"builtin_len({arg_expr})"

            if is_redeclaration:
                self.add_line(f"{var_name} = {func_call};")
            else:
                self.add_line(f"{c_type} {var_name} = {func_call};")
            return
