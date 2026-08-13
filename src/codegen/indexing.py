from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class IndexingMixin:
    def generate_index_assignment(self, node: Dict):
        """Генерирует присваивание по индексу: list[index] = value или dict[key] = value"""
        variable = node.get("variable", "")
        if isinstance(variable, str) and not variable.startswith("self."):
            self.assert_can_mutate(variable)
        index_ast = node.get("index", {})
        value_ast = node.get("value", {})
        node_type = node.get("node")

        # Проверяем по node_type
        if node_type == "nested_index_assignment":
            logger.debug(
                "  -> это nested_index_assignment, вызываем _generate_nested_index_assignment"
            )
            self._generate_nested_index_assignment(node)
            return

        # Проверяем, не является ли переменная словарем с индексами
        if isinstance(variable, dict):
            var_type = variable.get("type", "")
            logger.debug(f"variable is dict with type: {var_type}")
            if var_type in [
                "nested_index_access",
                "nested_index_assignment",
                "index_access",
            ]:
                logger.debug(
                    "  -> это вложенная индексация, вызываем _generate_nested_index_assignment"
                )
                self._generate_nested_index_assignment(node)
                return

        index_expr = self.generate_expression(index_ast)
        value_expr = self.generate_expression(value_ast)

        # Обработка для self.attr
        if isinstance(variable, str) and variable.startswith("self."):
            attr_name = variable[5:]
            logger.debug(f"  -> обработка self.{attr_name}")
            current_class = self._get_current_class()
            field = self.class_registry.field(current_class, attr_name) if current_class else None
            attr_type = field.py_type if field else ""
            if attr_type and self.is_device_tensor_type(attr_type):
                self.generate_tensor_index_assignment(
                    f"self->{attr_name}", attr_type, [index_ast], value_expr
                )
                return

            # Для атрибутов, которые могут быть списками
            if attr_name == "data":
                self.add_line(f"set_list_int(self->data, {index_expr}, {value_expr});")
            else:
                # Пробуем определить тип атрибута
                current_class = self._get_current_class()
                if current_class:
                    field = self.class_registry.field(current_class, attr_name)
                    attr_type = field.py_type if field else None
                    if attr_type and attr_type.startswith("list["):
                        struct_name = self.generate_list_struct_name(attr_type)
                        self.add_line(
                            f"set_{struct_name}(self->{attr_name}, {index_expr}, {value_expr});"
                        )
                    elif attr_type and attr_type.startswith("dict["):
                        # Обработка словаря как атрибута
                        key_type, value_type = self._extract_dict_types(attr_type)
                        dict_struct = f"dict_{self.clean_type_name_for_c(key_type)}_{self.clean_type_name_for_c(value_type)}"
                        self.add_line(
                            f"set_{dict_struct}(self->{attr_name}, {index_expr}, {value_expr});"
                        )
                    else:
                        self.add_line(
                            f"self->{attr_name}[{index_expr}] = {value_expr};"
                        )
                else:
                    self.add_line(f"self->{attr_name}[{index_expr}] = {value_expr};")
            return

        # Получаем информацию о переменной
        var_info = self.get_variable_info(variable)

        if var_info:
            py_type = var_info.get("py_type", "")
            logger.debug(f"  -> тип переменной: {py_type}")

            # Обработка для словарей
            if py_type.startswith("dict["):
                # Извлекаем типы ключа и значения
                key_type, value_type = self._extract_dict_types(py_type)

                # Генерируем имя структуры для словаря
                key_name = self.clean_type_name_for_c(key_type)
                value_name = self.clean_type_name_for_c(value_type)
                dict_struct = f"dict_{key_name}_{value_name}"

                # Добавляем присваивание для словаря
                self.add_line(
                    f"set_{dict_struct}({variable}, {index_expr}, {value_expr});"
                )

            # Обработка для списков
            elif py_type.startswith("list["):
                struct_name = self.generate_list_struct_name(py_type)
                self.add_line(
                    f"set_{struct_name}({variable}, {index_expr}, {value_expr});"
                )
            elif self.is_array_type(py_type):
                self.generate_array_struct(py_type)
                self.add_line(
                    f"{self.array_struct_name(py_type)}_set({variable}, (size_t)({index_expr}), {value_expr});"
                )
            # Обработка для кортежей (неизменяемые)
            elif py_type.startswith("tuple["):
                struct_name = self.generate_tuple_struct_name(py_type)
                self.add_line(
                    f"{variable}.data[{index_expr}] = {value_expr}; // Note: tuples are immutable in Python"
                )

            # Обычный массив или другой тип
            else:
                self.add_line(f"{variable}[{index_expr}] = {value_expr};")

        else:
            # Если информация о переменной не найдена, пробуем прямую генерацию
            logger.debug("  -> переменная не найдена, прямая генерация")
            self.add_line(f"{variable}[{index_expr}] = {value_expr};")

    def generate_nested_index_assignment(self, node: Dict):
        """Генерирует код для многомерного индексного присваивания: A_data[0][0] = 10"""
        variable = node.get("variable", "")
        if isinstance(variable, str) and not variable.startswith("self."):
            self.assert_can_mutate(variable)
        indices = node.get("indices", [])
        value_ast = node.get("value", {})
        var_type = node.get("var_type", "")

        # Генерируем выражение для значения
        value_expr = self.generate_expression(value_ast)

        # Получаем информацию о переменной
        var_info = self.get_variable_info(variable)
        if not var_info:
            self.add_line(f"// ERROR: Variable '{variable}' not found")
            return

        # Определяем тип переменной
        py_type = var_info.get("py_type", "")

        # Проверяем, является ли это вложенным списком
        if py_type.startswith("list[list["):
            # Это многомерный массив
            match = re.match(r"list\[list\[([^\]]+)\]\]", py_type)
            if not match:
                self.add_line(f"// ERROR: Invalid nested list type: {py_type}")
                return

            inner_type = match.group(1)  # "int"

            # Генерируем структуры для обоих уровней
            outer_struct_name = self.generate_list_struct_name(
                f"list[list[{inner_type}]]"
            )
            inner_struct_name = self.generate_list_struct_name(f"list[{inner_type}]")

            # Убедимся, что структуры сгенерированы
            self.generate_list_struct(f"list[{inner_type}]")
            self.generate_list_struct(f"list[list[{inner_type}]]")

            # Генерируем индексы
            if len(indices) == 2:
                index1_expr = self.generate_expression(indices[0])
                index2_expr = self.generate_expression(indices[1])

                # Получаем внутренний список
                temp_var = f"{variable}_inner_{self.temp_var_counter}"
                self.temp_var_counter += 1

                self.add_line(
                    f"// Доступ к элементу {variable}[{index1_expr}][{index2_expr}]"
                )
                self.add_line(
                    f"{inner_struct_name}* {temp_var} = get_{outer_struct_name}({variable}, {index1_expr});"
                )

                # Устанавливаем значение во внутреннем списке
                self.add_line(
                    f"set_{inner_struct_name}({temp_var}, {index2_expr}, {value_expr});"
                )
            else:
                self.add_line(f"// ERROR: Unsupported nesting depth {len(indices)}")
        elif py_type.startswith("list["):
            # Это одномерный список (но с вложенной индексацией - ошибка)
            if len(indices) == 1:
                # Это фактически обычное индексное присваивание
                index_expr = self.generate_expression(indices[0])
                struct_name = self.generate_list_struct_name(py_type)
                self.add_line(
                    f"set_{struct_name}({variable}, {index_expr}, {value_expr});"
                )
            else:
                self.add_line(f"// ERROR: Too many indices for type {py_type}")
        else:
            # Неизвестный тип
            self.add_line(f"// ERROR: Cannot assign to nested index of type {py_type}")

    def _generate_nested_index_assignment(self, node: Dict):
        """Генерирует присваивание для вложенной индексации любой глубины"""
        logger.debug(f"_generate_nested_index_assignment: {node}")

        # Получаем данные из узла
        var_name = node.get("variable", "")
        if isinstance(var_name, str):
            self.assert_can_mutate(var_name)
        indices_ast = node.get("indices", [])
        value_ast = node.get("value", {})

        logger.debug(
            f"nested assignment variable={var_name} indices={indices_ast} value={value_ast}"
        )

        if not var_name or not indices_ast:
            logger.error("nested assignment has no variable or indices")
            return

        # Генерируем выражения для всех индексов
        indices = []
        for idx_ast in indices_ast:
            indices.append(self.generate_expression(idx_ast))

        logger.debug(f"nested assignment indices: {indices}")
        value_expr = self.generate_expression(value_ast)

        if isinstance(var_name, str) and var_name.startswith("self."):
            attr_name = var_name[5:]
            current_class = self._get_current_class()
            field = self.class_registry.field(current_class, attr_name) if current_class else None
            attr_type = field.py_type if field else ""
            if attr_type and self.is_device_tensor_type(attr_type):
                self.generate_tensor_index_assignment(
                    f"self->{attr_name}", attr_type, indices_ast, value_expr
                )
                return

        var_info = self.get_variable_info(var_name)
        if not var_info:
            logger.error(f"nested assignment variable not found: {var_name}")
            return

        py_type = var_info.get("py_type", "")

        if self.is_device_tensor_type(py_type):
            self.generate_tensor_index_assignment(var_name, py_type, indices_ast, value_expr)
            return

        logger.debug(f"nested assignment type={py_type} value={value_expr}")

        # Если глубина = 1, используем простую set_функцию
        if len(indices) == 1:
            if py_type.startswith("list["):
                struct_name = self.generate_list_struct_name(py_type)
                self.add_line(
                    f"set_{struct_name}({var_name}, {indices[0]}, {value_expr});"
                )
            else:
                self.add_line(f"{var_name}[{indices[0]}] = {value_expr};")
            return

        # Для глубины > 1 нужно дойти до последнего уровня и там установить значение
        current_var = var_name
        current_type = py_type

        # Проходим все уровни КРОМЕ ПОСЛЕДНЕГО, чтобы получить список последнего уровня
        for i, idx in enumerate(indices[:-1]):  # Все кроме последнего индекса
            if current_type.startswith("list["):
                struct_name = self.generate_list_struct_name(current_type)

                # Определяем тип следующего уровня
                inner_type = current_type[5:-1]  # list[X] -> X

                # Создаем временную переменную для следующего уровня
                temp_var = self.generate_temporary_var(f"level_{i}")

                # Определяем C тип для временной переменной
                inner_struct_name = self.generate_list_struct_name(inner_type)
                c_type = f"{inner_struct_name}*"

                # Получаем вложенный список
                self.add_line(
                    f"{c_type} {temp_var} = get_{struct_name}({current_var}, {idx});"
                )

                # Обновляем текущие переменные для следующей итерации
                current_var = temp_var
                current_type = inner_type
            else:
                self.add_line(f"// ERROR: Too many indices for type {py_type}")
                return

        # Теперь current_var указывает на список последнего уровня
        # Устанавливаем значение на последнем уровне
        last_idx = indices[-1]

        if current_type.startswith("list["):
            struct_name = self.generate_list_struct_name(current_type)
            self.add_line(
                f"set_{struct_name}({current_var}, {last_idx}, {value_expr});"
            )
        else:
            self.add_line(f"{current_var}[{last_idx}] = {value_expr};")

    def generate_slice_assignment(self, node: Dict):
        """Генерирует присваивание среза: list[start:stop] = values"""
        variable = node.get("variable", "")
        if isinstance(variable, str) and not variable.startswith("self."):
            self.assert_can_mutate(variable)
        start_ast = node.get("start", {})
        stop_ast = node.get("stop", {})
        step_ast = node.get("step", {})
        value_ast = node.get("value", {})

        start_expr = self.generate_expression(start_ast) if start_ast else "0"
        stop_expr = (
            self.generate_expression(stop_ast) if stop_ast else f"{variable}->size"
        )

        var_info = self.get_variable_info(variable)

        if var_info and var_info.get("py_type", "").startswith("list["):
            if value_ast.get("type") == "list_literal":
                items = value_ast.get("items", [])
                if items:
                    # Присваивание списка значений срезу
                    for i, item in enumerate(items):
                        item_expr = self.generate_expression(item)
                        idx = f"{start_expr} + {i}"
                        self.add_line(
                            f"if ({idx} < {stop_expr} && {idx} < {variable}->size) {{"
                        )
                        self.indent_level += 1
                        self.add_line(
                            f"set_{self.generate_list_struct_name(var_info['py_type'])}({variable}, {idx}, {item_expr});"
                        )
                        self.indent_level -= 1
                        self.add_line("}")
            else:
                # Присваивание одного значения всем элементам среза
                value_expr = self.generate_expression(value_ast)
                temp_var = self.generate_temporary_var("int")
                self.add_line(
                    f"for (int {temp_var} = {start_expr}; {temp_var} < {stop_expr}; {temp_var}++) {{"
                )
                self.indent_level += 1
                self.add_line(f"if ({temp_var} < {variable}->size) {{")
                self.indent_level += 1
                self.add_line(
                    f"set_{self.generate_list_struct_name(var_info['py_type'])}({variable}, {temp_var}, {value_expr});"
                )
                self.indent_level -= 1
                self.add_line("}")
                self.indent_level -= 1
                self.add_line("}")

    def generate_augmented_index_assignment(self, node: Dict):
        """Генерирует составное присваивание по индексу: list[index] += value"""
        variable = node.get("variable", "")
        if isinstance(variable, str) and not variable.startswith("self."):
            self.assert_can_mutate(variable)
        index_ast = node.get("index", {})
        indices_ast = node.get("indices") or [index_ast]
        operator = node.get("operator", "")
        value_ast = node.get("value", {})

        index_exprs = [self.generate_expression(index) for index in indices_ast]
        index_expr = index_exprs[0]
        value_expr = self.generate_expression(value_ast)

        var_info = self.get_variable_info(variable)

        target_type = var_info.get("py_type", "") if var_info else ""
        if not target_type and isinstance(variable, str) and variable.startswith("self."):
            current_class = self._get_current_class()
            field = self.class_registry.field(current_class, variable[5:]) if current_class else None
            target_type = field.py_type if field else ""

        if self.is_device_tensor_type(target_type):
            target_expr = variable
            if variable.startswith("self."):
                target_expr = f"self->{variable[5:]}"
            literal = ", ".join(f"(size_t)({index})" for index in index_exprs)
            current_expr = (
                f"ocean_tensor_get_nd({target_expr}->handle, "
                f"(const size_t[]){{{literal}}}, {len(index_exprs)})"
            )
            op_symbol = operator.replace("=", "")
            updated_expr = f"({current_expr} {op_symbol} {value_expr})"
            self.generate_tensor_index_assignment(
                target_expr, target_type, indices_ast, updated_expr
            )
            return

        if var_info and var_info.get("py_type", "").startswith("list["):
            struct_name = self.generate_list_struct_name(var_info["py_type"])
            # Получаем текущее значение
            temp_var = self.generate_temporary_var("int")
            self.add_line(
                f"int {temp_var} = get_{struct_name}({variable}, {index_expr});"
            )
            # Применяем оператор
            op_symbol = operator.replace("=", "")
            c_op = self.operator_map.get(op_symbol, op_symbol)
            if c_op == "pow":
                self.add_line(f"{temp_var} = pow({temp_var}, {value_expr});")
            else:
                self.add_line(f"{temp_var} {operator} {value_expr};")
            # Устанавливаем новое значение
            self.add_line(f"set_{struct_name}({variable}, {index_expr}, {temp_var});")

    def _generate_index_access(self, ast: Dict) -> str:
        """Генерирует код для доступа по индексу"""
        variable = ast.get("variable", "")
        index_ast = ast.get("index", {})
        index_expr = self.generate_expression(index_ast)

        # Если variable - это attribute_access (например, self.a)
        if isinstance(variable, dict) and variable.get("type") == "attribute_access":
            # Генерируем выражение для доступа к атрибуту
            obj_name = variable.get("object", "")
            attr_name = variable.get("attribute", "")

            # Получаем тип атрибута
            attr_type = self._get_attribute_type(obj_name, attr_name)

            # Генерируем выражение для доступа к атрибуту
            attr_expr = self.generate_attribute_access(variable)

            if attr_type and attr_type.startswith("list["):
                # Для списков используем get_функцию
                struct_name = self.generate_list_struct_name(attr_type)
                logger.debug(
                    f"DEBUG: Использую get_{struct_name} для {attr_expr}[{index_expr}]"
                )
                return f"get_{struct_name}({attr_expr}, {index_expr})"
            else:
                # Для других типов - обычная индексация
                logger.debug(f"DEBUG: Обычная индексация для {attr_expr}[{index_expr}]")
                return f"{attr_expr}[{index_expr}]"

        # Если variable - обычная строка
        elif isinstance(variable, str):
            self.assert_can_read(variable)
            var_info = self.get_variable_info(variable)

            if var_info:
                py_type = var_info.get("py_type", "")

                if py_type.startswith("list["):
                    struct_name = self.generate_list_struct_name(py_type)
                    return f"get_{struct_name}({variable}, {index_expr})"
                elif self.is_array_type(py_type):
                    self.generate_array_struct(py_type)
                    return f"{self.array_struct_name(py_type)}_get({variable}, (size_t)({index_expr}))"
                elif py_type.startswith("tuple["):
                    struct_name = self.generate_tuple_struct_name(py_type)
                    return f"get_{struct_name}({variable}, {index_expr})"
                elif py_type.startswith("dict["):
                    # Это словарь
                    key_type, value_type = self._extract_dict_types(py_type)
                    key_name = self.clean_type_name_for_c(key_type)
                    value_name = self.clean_type_name_for_c(value_type)
                    struct_name = f"dict_{key_name}_{value_name}"

                    # Убеждаемся, что структура сгенерирована
                    self.generate_dict_struct(key_type, value_type)

                    return f"get_{struct_name}({variable}, {index_expr})"

            return f"{variable}[{index_expr}]"
        else:
            # Сложный случай
            var_expr = self.generate_expression(variable)
            return f"{var_expr}[{index_expr}]"

    def _generate_nested_index_access(self, ast: Dict) -> str:
        """Генерирует выражение для вложенной индексации (для использования в выражениях)"""
        logger.debug(f"_generate_nested_index_access_expr: {ast}")

        # Собираем все индексы
        indices = []
        current = ast
        var_name = None

        while True:
            if current.get("type") == "nested_index_access":
                # Добавляем текущий индекс
                indices.append(self.generate_expression(current.get("index", {})))
                # Переходим к base
                current = current.get("base", {})
            elif current.get("type") == "index_access":
                # Добавляем индекс
                indices.append(self.generate_expression(current.get("index", {})))
                # Получаем переменную
                var_node = current.get("variable", {})

                # Извлекаем имя переменной из вложенной структуры
                if isinstance(var_node, dict):
                    if var_node.get("type") == "index_access":
                        # Это вложенный index_access, продолжаем
                        current = var_node
                        continue
                    elif var_node.get("type") == "variable":
                        # Это конечная переменная
                        var_name = var_node.get("value", "")
                        if not var_name:
                            var_name = var_node.get("name", "")
                        break
                    else:
                        # Пробуем другие поля
                        var_name = (
                            var_node.get("value", "")
                            or var_node.get("name", "")
                            or str(var_node)
                        )
                        break
                else:
                    var_name = str(var_node)
                    break
            elif current.get("type") == "variable":
                # Прямой узел переменной
                var_name = (
                    current.get("value", "") or current.get("name", "") or str(current)
                )
                break
            else:
                # Если ничего не подошло
                var_name = str(current)
                break

        # Индексы собраны в обратном порядке
        indices.reverse()
        logger.debug(f"var_name={var_name}, indices={indices}")

        if not var_name or not isinstance(var_name, str):
            logger.error(f"Invalid var_name: {var_name}")
            return "0"

        self.assert_can_read(var_name)
        var_info = self.get_variable_info(var_name)
        if not var_info:
            # Обычный массив
            result = var_name
            for idx in indices:
                result += f"[{idx}]"
            return result

        # Для списков строим цепочку get_функций
        py_type = var_info.get("py_type", "")
        result = var_name
        current_type = py_type

        for idx in indices:
            if current_type.startswith("list["):
                struct_name = self.generate_list_struct_name(current_type)
                result = f"get_{struct_name}({result}, {idx})"

                # Обновляем текущий тип для следующего уровня
                current_type = current_type[5:-1]  # list[X] -> X
            else:
                result += f"[{idx}]"

        return result

    def _generate_complex_attribute_access(self, ast: Dict) -> str:
        """Генерирует доступ к элементу сложного атрибута (self.data[index])"""
        obj_name = ast.get("object", "")
        attr_name = ast.get("attribute", "")
        index_asts = ast.get("indices") or [ast.get("index", {})]
        index_exprs = [self.generate_expression(index_ast) for index_ast in index_asts]
        index_expr = index_exprs[0] if len(index_exprs) == 1 else ", ".join(index_exprs)

        # Получаем информацию об объекте
        var_info = self.get_variable_info(obj_name)

        if not var_info:
            # Если переменная не найдена, возможно это self внутри метода класса
            if obj_name == "self":
                # Ищем текущий класс
                current_class = self._get_current_class()
                if current_class:
                    # Проверяем тип атрибута
                    field = self.class_registry.field(current_class, attr_name)
                    attr_type = field.py_type if field else None
                    if attr_type and self.is_device_tensor_type(attr_type):
                        literal = ", ".join(f"(size_t)({index})" for index in index_exprs)
                        return (
                            f"ocean_tensor_get_nd(self->{attr_name}->handle, "
                            f"(const size_t[]){{{literal}}}, {len(index_exprs)})"
                        )
                    if attr_type and attr_type.startswith("list["):
                        # Генерируем вызов специализированной функции
                        struct_name = self.generate_list_struct_name(attr_type)
                        return f"get_{struct_name}(self->{attr_name}, {index_expr})"

        # Если это переменная (не self)
        if var_info:
            obj_py_type = var_info.get("py_type", "")

            if self.is_device_tensor_type(obj_py_type) and attr_name == "shape":
                if len(index_exprs) != 1:
                    raise RuntimeError("Tensor.shape expects one axis")
                return f"Tensor_shape({obj_name}, {index_exprs[0]})"
            # Если это класс или указатель на класс
            if self._is_class_type(obj_py_type) or var_info.get("is_pointer", False):
                # Получаем информацию о классе
                class_name = obj_py_type.replace("*", "").strip()
                field = self.class_registry.field(class_name, attr_name)
                attr_type = field.py_type if field else None
                if attr_type and self.is_device_tensor_type(attr_type):
                    literal = ", ".join(f"(size_t)({index})" for index in index_exprs)
                    return (
                        f"ocean_tensor_get_nd({obj_name}->{attr_name}->handle, "
                        f"(const size_t[]){{{literal}}}, {len(index_exprs)})"
                    )
                if attr_type and attr_type.startswith("list["):
                    struct_name = self.generate_list_struct_name(attr_type)
                    return (
                        f"get_{struct_name}({obj_name}->{attr_name}, {index_expr})"
                    )

        # По умолчанию - прямой доступ к массиву
        return f"{obj_name}->{attr_name}[{index_expr}]"
