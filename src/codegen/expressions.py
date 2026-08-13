from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class ExpressionsMixin:
    def generate_expression(self, ast: Dict) -> str:
        """Генерирует C выражение из AST с поддержкой tuple и list"""
        if not ast:
            return "0"

        node_type = ast.get("type", "")

        if node_type == "nested_index_access":
            return self._generate_nested_index_access(ast)

        if node_type == "tensor_index_access":
            return self.generate_tensor_index_access(ast)

        if node_type == "index_access":
            return self._generate_index_access(ast)

        elif node_type == "complex_attribute_access":
            return self._generate_complex_attribute_access(ast)

        elif node_type == "slice_access":
            variable = ast.get("variable", "")
            start_ast = ast.get("start", {})
            stop_ast = ast.get("stop", {})
            step_ast = ast.get("step", {})

            # Генерируем выражения для границ
            start_expr = self.generate_expression(start_ast) if start_ast else "0"
            stop_expr = self.generate_expression(stop_ast) if stop_ast else ""
            step_expr = self.generate_expression(step_ast) if step_ast else "1"

            var_info = self.get_variable_info(variable)
            if var_info:
                py_type = self.strip_borrow_type(var_info.get("py_type", ""))

                if py_type.startswith("list["):
                    # Для списка stop по умолчанию: list->size
                    if not stop_ast:
                        stop_expr = f"{variable}->size"

                    struct_name = self.generate_list_struct_name(py_type)
                    # Создаем срез списка напрямую без временной переменной
                    return f"slice_{struct_name}({variable}, {start_expr}, {stop_expr}, {step_expr})"

                elif py_type.startswith("tuple["):
                    # Для кортежа stop по умолчанию: tuple.size
                    if not stop_ast:
                        stop_expr = f"{variable}->size"

                    struct_name = self.generate_tuple_struct_name(py_type)
                    return f"slice_{struct_name}({variable}, {start_expr}, {stop_expr}, {step_expr})"

            # Если не list и не tuple, генерируем обычный slice
            return f"/* slice of {variable}[{start_expr}:{stop_expr}] */"

        if node_type == "tuple_literal":
            # Для tuple литералов используем метод generate_tuple_creation
            return self.generate_tuple_creation(ast)
        elif node_type == "literal":
            value = ast.get("value")
            data_type = ast.get("data_type", "")

            if data_type == "str":
                return f'"{value}"'
            elif data_type == "bool":
                return "true" if value else "false"
            elif data_type == "None":
                return "NULL"
            else:
                return str(value)
        elif node_type == "variable":
            var_name = ast.get("value", "")
            self.assert_can_read(var_name)
            return var_name

        # Добавляем обработку новых типов узлов
        elif node_type == "attribute_access":
            return self.generate_attribute_access(ast)
        elif node_type == "constructor_call":
            return self.generate_constructor_call(ast)
        elif node_type == "method_call":
            return self.generate_method_call(ast)

        if node_type == "literal":
            value = ast.get("value")
            data_type = ast.get("data_type", "")

            if data_type == "str":
                return f'"{value}"'
            elif data_type == "bool":
                return "true" if value else "false"
            elif data_type == "None":
                return "NULL"
            else:
                return str(value)

        elif node_type == "variable":
            var_name = ast.get("value", "")
            self.assert_can_read(var_name)
            return var_name

        elif node_type == "binary_operation":
            left_ast = ast.get("left", {})
            right_ast = ast.get("right", {})
            operator = ast.get("operator_symbol", "")

            left = self.generate_expression(left_ast)
            right = self.generate_expression(right_ast)

            # Проверяем, являются ли операнды строками
            left_is_string = self._is_string_expression(left_ast)
            right_is_string = self._is_string_expression(right_ast)

            right_is_none = self._is_none_expression(right_ast)
            left_is_none = self._is_none_expression(left_ast)

            if operator == "==" and (right_is_none or left_is_none):
                # Сравнение с NULL
                non_none_expr = left if right_is_none else right
                return f"({non_none_expr} == NULL)"
            elif operator == "!=" and (right_is_none or left_is_none):
                non_none_expr = left if right_is_none else right
                return f"({non_none_expr} != NULL)"

            if operator == "==" and (left_is_string or right_is_string):
                return f"(strcmp({left}, {right}) == 0)"
            elif operator == "!=" and (left_is_string or right_is_string):
                return f"(strcmp({left}, {right}) != 0)"
            # Для сложения строк используем временную переменную
            elif operator == "+" and (left_is_string or right_is_string):
                # Создаем временную переменную для результата
                temp_var = self.generate_temporary_var("str")

                # Генерируем код для конкатенации строк
                self._generate_string_concatenation(temp_var, left, right, "char*")

                return temp_var

            if operator == "**":
                return f"pow({left}, {right})"

            c_operator = self.operator_map.get(operator, operator)
            return f"({left} {c_operator} {right})"

        elif node_type == "unary_operation":
            operand_ast = ast.get("operand", {})
            operator = ast.get("operator_symbol", "")

            operand = self.generate_expression(operand_ast)
            c_operator = self.operator_map.get(operator, operator)

            return f"{c_operator}({operand})"

        elif node_type == "method_call":
            # Это вызов метода внутри выражения
            object_name = ast.get("object", "")
            method_name = ast.get("method", "")
            args = ast.get("arguments", [])

            # Проверяем тип объекта
            var_info = self.get_variable_info(object_name)

            # Parameters such as ``A: &Tensor[T]`` may be resolved through a
            # class/borrow path even when they are not present as a direct
            # local binding. Preserve the public Tensor method ABI there.
            resolved_type, resolved_object = self.resolve_object_path(object_name)
            if not var_info and self.is_device_tensor_type(resolved_type or ""):
                arg_strings = [self.generate_expression(arg) for arg in args]
                full_args = resolved_object
                if arg_strings:
                    full_args += ", " + ", ".join(arg_strings)
                return f"Tensor_{method_name}({full_args})"

            if var_info:
                obj_type = var_info.get("py_type", "")

                if self.is_device_tensor_type(obj_type):
                    arg_strings = [self.generate_expression(arg) for arg in args]
                    full_args = object_name
                    if arg_strings:
                        full_args += ", " + ", ".join(arg_strings)
                    return f"Tensor_{method_name}({full_args})"

                if self._is_class_type(obj_type):
                    # Это вызов метода класса: obj.method(args)
                    arg_strings = [self.generate_expression(arg) for arg in args]
                    args_str = ", ".join(arg_strings) if arg_strings else ""
                    full_args = f"{object_name}"
                    if args_str:
                        full_args = f"{object_name}, {args_str}"
                    class_name = obj_type.split("[", 1)[0]
                    return f"{class_name}_{method_name}({full_args})"

                elif object_name == "self":
                    # self.method(args) внутри метода класса
                    # Находим текущий класс
                    current_class = None
                    for scope in reversed(self.variable_scopes):
                        if "class_name" in scope:
                            current_class = scope.get("class_name")
                            break

                    if current_class:
                        arg_strings = [self.generate_expression(arg) for arg in args]
                        args_str = ", ".join(arg_strings) if arg_strings else ""
                        full_args = f"self"
                        if args_str:
                            full_args = f"self, {args_str}"
                        return f"{current_class}_{method_name}({full_args})"

            # Если не смогли определить, генерируем ошибку
            return (
                f"/* ERROR: Неизвестный вызов метода {object_name}.{method_name}() */"
            )

        elif node_type == "static_method_call":
            device_tensor_expr = self._device_tensor_static_call(ast)
            if device_tensor_expr is not None:
                return device_tensor_expr
            class_name = ast.get("class_name", "")
            method_name = ast.get("method", "")
            args = ast.get("arguments", [])
            arg_strings = [self.generate_expression(arg) for arg in args]
            args_str = ", ".join(arg_strings)
            return f"{class_name}_{method_name}({args_str})"

        elif node_type == "function_call":
            func_name = ast.get("function", "")

            if func_name.startswith("@"):
                func_name = func_name[1:]

            builtin_funcs = ["len", "str", "int", "bool", "range", "input"]
            if func_name in builtin_funcs:
                args = ast.get("arguments", [])

                # Для len() определяем тип аргумента
                if func_name == "len" and args:
                    arg_ast = args[0]
                    if arg_ast.get("type") == "variable":
                        var_name = arg_ast.get("value", "")
                        var_info = self.get_variable_info(var_name)

                        if var_info:
                            py_type = var_info.get("py_type", "")

                            if py_type.startswith("tuple["):
                                struct_name = self.generate_tuple_struct_name(py_type)
                                # Используем специализированную функцию
                                c_func_name = f"builtin_len_{struct_name}"
                            elif py_type.startswith("list["):
                                struct_name = self.generate_list_struct_name(py_type)
                                # Используем специализированную функцию для списков
                                c_func_name = f"builtin_len_{struct_name}"
                            elif self.is_array_type(py_type):
                                self.generate_array_struct(py_type)
                                c_func_name = f"{self.array_struct_name(py_type)}_len"
                            else:
                                c_func_name = "builtin_len"
                        else:
                            c_func_name = "builtin_len"
                    else:
                        c_func_name = "builtin_len"
                elif func_name == "input":
                    # Для input в выражениях генерируем код и возвращаем переменную
                    return self.generate_input_expression(ast)
                else:
                    c_func_name = f"builtin_{func_name}"
            else:
                c_func_name = func_name

            args = ast.get("arguments", [])
            arg_strings = [self.generate_expression(arg_ast) for arg_ast in args]
            self.consume_owned_call_arguments(func_name, args)
            args_str = ", ".join(arg_strings)
            return f"{c_func_name}({args_str})"

        elif node_type == "tuple_literal":
            # Для tuple литералов генерируем временную структуру
            items = ast.get("items", [])
            if items:
                item_strs = [self.generate_expression(item) for item in items]

                # Создаем временный tuple
                temp_name = self.generate_temporary_var("tuple")
                struct_name = f"tuple_{len(items)}_{'_'.join(['item' for _ in items])}"

                # Регистрируем тип
                elements_type = ", ".join(["int" for _ in items])  # Упрощенно
                py_type = f"tuple[{elements_type}]"
                self.generate_tuple_struct(py_type)

                return f"create_{self.generate_tuple_struct_name(py_type)}({', '.join(item_strs)})"
            return "{}"

        elif node_type == "list_literal":
            # Для list литералов генерируем создание списка
            items = ast.get("items", [])
            if items:
                # Определяем тип элементов
                if items:
                    first_item = items[0]
                    if isinstance(first_item, Mapping):
                        if first_item.get("type") == "tuple_literal":
                            element_type = "tuple"
                        elif first_item.get("type") == "list_literal":
                            element_type = "list"
                        else:
                            element_type = "int"  # По умолчанию
                    else:
                        element_type = "int"
                else:
                    element_type = "int"

                py_type = f"list[{element_type}]"
                struct_name = self.generate_list_struct_name(py_type)

                # Генерируем код для создания списка
                temp_name = self.generate_temporary_var("list")
                self.generate_list_struct(py_type)

                # Создаем список
                code_parts = []
                code_parts.append(f"create_{struct_name}({len(items)})")

                # Добавляем элементы
                for item_ast in items:
                    item_expr = self.generate_expression(item_ast)
                    code_parts.append(f"append_{struct_name}({temp_name}, {item_expr})")

                return temp_name
            return "NULL"

        elif node_type == "address_of":
            variable = ast.get("variable", "")
            return f"&{variable}"

        elif node_type == "dereference":
            pointer = ast.get("pointer", "")
            return f"*{pointer}"

        # Для неизвестных типов пытаемся извлечь значение
        ast_value = str(ast.get("value", "0"))
        if ast_value.startswith("@"):  # C - code
            ast_value = ast_value[1:]

        return ast_value

    def generate_attribute_access(self, ast: Dict) -> str:
        """Генерирует доступ к атрибуту объекта"""
        obj_name = ast.get("object", "")
        attr_name = ast.get("attribute", "")

        if obj_name != "self" and "." not in obj_name:
            self.assert_can_read(obj_name)

        object_type, object_expression = self.resolve_object_path(obj_name)
        _, field_expression = self.resolve_class_field(
            object_type, object_expression, attr_name
        ) if object_type else (None, None)
        if field_expression:
            return field_expression
        if object_expression != obj_name or obj_name == "self":
            return f"{object_expression}->{attr_name}"

        var_info = self.get_variable_info(obj_name)
        if var_info and (
            var_info.get("is_pointer", False)
            or self._is_class_type(var_info.get("py_type", ""))
            or self.is_array_type(var_info.get("py_type", ""))
        ):
            if self.is_device_tensor_type(var_info.get("py_type", "")):
                if attr_name in {"ndim", "size", "device"}:
                    return f"Tensor_{attr_name}({obj_name})"
            return f"{obj_name}->{attr_name}"
        return f"{obj_name}.{attr_name}"

    def _generate_expression_from_ast_for_init(
        self,
        ast: Dict,
        param_names: List[str],
        target_type: str = "",
        target_name: str = "field",
    ) -> str:
        """Генерирует выражение из AST для конструктора с подстановкой параметров"""
        if not ast:
            return ""

        node_type = ast.get("type", "")
        logger.debug(
            f"DEBUG _generate_expression_from_ast_for_init: type={node_type}, ast={ast}"
        )

        if node_type == "literal":
            value = ast.get("value", "")
            data_type = ast.get("data_type", "")
            logger.debug(f"Found literal: {value} (type: {data_type})")
            if data_type == "str":
                return f'"{value}"'
            else:
                return str(value)

        elif node_type == "variable":
            # Поддерживаем оба формата: 'value' и 'name'
            var_name = ast.get("value") or ast.get("name", "")
            logger.debug(f"Found variable: {var_name}")
            # Если это параметр конструктора, используем как есть
            if var_name in param_names:
                logger.debug(f"Is a constructor parameter")
                return var_name
            # Если это не параметр, возможно это атрибут self
            logger.debug(f"Not a constructor parameter")
            return var_name

        elif node_type == "constructor_call":
            class_name = ast.get("class_name", "")
            arguments = [
                self._generate_expression_from_ast_for_init(argument, param_names)
                for argument in ast.get("arguments", [])
            ]
            return f"create_{class_name}({', '.join(arguments)})"

        elif node_type == "method_call":
            raise RuntimeError(
                f"method call '{ast.get('object', '')}.{ast.get('method', '')}' "
                "is not supported in a constructor initializer"
            )

        elif node_type == "static_method_call":
            if ast.get("class_name") == "Tensor" and self.is_device_tensor_type(target_type):
                previous = getattr(self, "device_tensor_argument_generator", None)
                self.device_tensor_argument_generator = (
                    lambda argument: self._generate_expression_from_ast_for_init(
                        argument, param_names
                    )
                )
                try:
                    return self._generate_device_tensor_expression(ast, target_type)
                finally:
                    self.device_tensor_argument_generator = previous
            raise RuntimeError(
                f"static call '{ast.get('class_name', '')}.{ast.get('method', '')}' "
                "is not supported in a constructor initializer"
            )

        elif node_type == "list_literal":
            # Keep the established constructor behavior for an empty list:
            # calloc leaves the field NULL and later method initialization can
            # replace it with a concrete list. Non-empty list fields need a
            # dedicated owned-list initializer rather than a raw expression.
            if target_type.startswith("list[") and not ast.get("items"):
                return ""
            raise RuntimeError(
                f"list literal cannot initialize field '{target_name}' of type '{target_type}'"
            )

        elif node_type == "binary_operation":
            left_ast = ast.get("left", {})
            right_ast = ast.get("right", {})
            operator = ast.get("operator_symbol") or ast.get("operator", "")

            logger.debug(f"Binary operation: {operator}")

            left = self._generate_expression_from_ast_for_init(left_ast, param_names)
            right = self._generate_expression_from_ast_for_init(right_ast, param_names)

            if operator in ["**", "POW"]:
                return f"pow({left}, {right})"

            c_operator = self.operator_map.get(operator, operator)

            # Правильно расставляем скобки для сохранения приоритета операций
            if operator in ["+", "-", "ADD", "SUBTRACT"]:
                # Для сложения/вычитания в сложных выражениях нужны скобки
                if left_ast.get("type") == "binary_operation":
                    left_operator = left_ast.get("operator_symbol") or left_ast.get(
                        "operator", ""
                    )
                    if left_operator in ["*", "/", "%", "MULTIPLY", "DIVIDE", "MODULO"]:
                        left = f"({left})"
                if right_ast.get("type") == "binary_operation":
                    right_operator = right_ast.get("operator_symbol") or right_ast.get(
                        "operator", ""
                    )
                    if right_operator in [
                        "*",
                        "/",
                        "%",
                        "MULTIPLY",
                        "DIVIDE",
                        "MODULO",
                    ]:
                        right = f"({right})"

            result = f"{left} {c_operator} {right}"
            logger.debug(f"Generated binary expression: {result}")
            return result

        logger.debug(
            f"DEBUG _generate_expression_from_ast_for_init: Unknown AST type: {node_type}"
        )
        return ""

    def _generate_expression_from_ast(self, ast: Dict, param_names: List[str]) -> str:
        """Генерирует выражение из AST с подстановкой параметров конструктора"""
        if not ast:
            return ""

        node_type = ast.get("type", "")
        logger.debug(f"_generate_expression_from_ast: type={node_type}, ast={ast}")

        if node_type == "variable":
            # Поддерживаем оба формата: 'value' и 'name'
            var_name = ast.get("value") or ast.get("name", "")
            # Если это параметр конструктора, используем как есть
            if var_name in param_names:
                logger.debug(f"Found parameter: {var_name}")
                return var_name
            logger.debug(f"Variable not a parameter: {var_name}")
            return var_name

        elif node_type == "literal":
            value = ast.get("value", "")
            data_type = ast.get("data_type", "")
            logger.debug(f"Found literal: {value} (type: {data_type})")
            if data_type == "str":
                return f'"{value}"'
            else:
                return str(value)

        elif node_type == "constructor_call":
            class_name = ast.get("class_name", "")
            arguments = [
                self._generate_expression_from_ast(argument, param_names)
                for argument in ast.get("arguments", [])
            ]
            return f"create_{class_name}({', '.join(arguments)})"

        elif node_type == "binary_operation":
            left_ast = ast.get("left", {})
            right_ast = ast.get("right", {})
            operator = ast.get("operator_symbol") or ast.get("operator", "")

            logger.debug(f"Binary operation: {operator}")

            left = self._generate_expression_from_ast(left_ast, param_names)
            right = self._generate_expression_from_ast(right_ast, param_names)

            if operator == "**" or operator == "POW":
                return f"pow({left}, {right})"

            c_operator = self.operator_map.get(operator, operator)

            # Правильно расставляем скобки для сохранения приоритета операций
            if operator in ["+", "-", "ADD", "SUBTRACT"]:
                # Для сложения/вычитания в сложных выражениях нужны скобки
                if left_ast.get("type") == "binary_operation":
                    left_operator = left_ast.get("operator_symbol") or left_ast.get(
                        "operator", ""
                    )
                    if left_operator in ["*", "/", "%", "MULTIPLY", "DIVIDE", "MODULO"]:
                        left = f"({left})"
                if right_ast.get("type") == "binary_operation":
                    right_operator = right_ast.get("operator_symbol") or right_ast.get(
                        "operator", ""
                    )
                    if right_operator in [
                        "*",
                        "/",
                        "%",
                        "MULTIPLY",
                        "DIVIDE",
                        "MODULO",
                    ]:
                        right = f"({right})"

            result = f"{left} {c_operator} {right}"
            logger.debug(f"Generated binary expression: {result}")
            return result

        elif node_type == "attribute_access":
            obj_name = ast.get("object", "")
            attr_name = ast.get("attribute", "")

            logger.debug(f"Attribute access: {obj_name}.{attr_name}")

            # В конструкторе атрибуты объекта еще не инициализированы
            # Это не должно случиться при правильном анализе
            self.add_line(
                f"// WARNING: Accessing attribute {attr_name} of {obj_name} in constructor"
            )
            return f"obj->{attr_name}"

        logger.debug(f"Unknown AST type: {node_type}")
        return ""
