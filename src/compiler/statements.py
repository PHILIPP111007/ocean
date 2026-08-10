from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class StatementsMixin:
    def generate_break(self, node: Dict):
        """Генерирует оператор break"""
        self.add_line("break;")
        self.add_line("// break statement")

    def generate_continue(self, node: Dict):
        """Генерирует оператор continue"""
        self.add_line("continue;")
        self.add_line("// continue statement")

    def generate_return(self, node: Dict):
        """Генерирует return с поддержкой кастомных типов"""
        operations = node.get("operations", [])

        for op in operations:
            if op.get("type") == "RETURN":
                value_ast = op.get("value", {})

                if not value_ast:
                    self.add_line("return;")
                    return

                # Определяем тип возвращаемого значения
                if value_ast.get("type") == "variable":
                    var_name = value_ast.get("value", "")
                    var_info = self.get_variable_info(var_name)

                    if var_info:
                        py_type = var_info.get("py_type", "")

                        # Если возвращаем список или кортеж
                        if py_type.startswith("list[") or py_type.startswith("tuple["):
                            # Просто возвращаем указатель
                            self.add_line(f"return {var_name};")
                            return

                # Для других случаев генерируем выражение
                expr = self.generate_expression(value_ast)
                self.add_line(f"return {expr};")
                return

        # Если ничего не нашли
        self.add_line("return;")

    def generate_while_loop(self, node: Dict):
        """Генерирует while loop с правильной обработкой структуры JSON"""
        # В вашем JSON ключ "condition", а не "condition_ast"
        condition_ast = node.get("condition")
        if not condition_ast:
            return

        condition = self.generate_expression(condition_ast)

        self.add_line(f"while ({condition}) {{")
        self.indent_level += 1

        # Входим в scope цикла
        self.enter_scope()

        # Генерируем тело цикла из списка body
        body_nodes = node.get("body", [])
        for body_node in body_nodes:
            self.generate_graph_node(body_node)

        # Выходим из scope цикла
        self.exit_scope()

        self.indent_level -= 1
        self.add_line("}")

    def generate_if_statement(self, node: Dict):
        """Генерирует if statement"""
        condition_ast = node.get("condition_ast")
        if not condition_ast:
            return

        condition = self.generate_expression(condition_ast)

        self.add_line(f"if ({condition}) {{")
        self.indent_level += 1

        # Входим в scope if
        self.enter_scope()

        # Генерируем тело if
        for body_node in node.get("body", []):
            self.generate_graph_node(body_node)

        # Выходим из scope if
        self.exit_scope()

        self.indent_level -= 1
        self.add_line("}")

        # elif блоки
        for elif_block in node.get("elif_blocks", []):
            elif_condition = self.generate_expression(
                elif_block.get("condition_ast", {})
            )
            self.add_line(f"else if ({elif_condition}) {{")
            self.indent_level += 1

            # Входим в scope elif
            self.enter_scope()

            # Генерируем тело elif
            for body_node in elif_block.get("body", []):
                self.generate_graph_node(body_node)

            # Выходим из scope elif
            self.exit_scope()

            self.indent_level -= 1
            self.add_line("}")

        # else блок
        else_block = node.get("else_block")
        if else_block:
            self.add_line("else {")
            self.indent_level += 1

            # Входим в scope else
            self.enter_scope()

            # Генерируем тело else
            for body_node in else_block.get("body", []):
                self.generate_graph_node(body_node)

            # Выходим из scope else
            self.exit_scope()

            self.indent_level -= 1
            self.add_line("}")

    def generate_for_loop(self, node: Dict):
        """Генерирует for loop"""
        loop_var = node.get("loop_variable", "i")
        iterable = node.get("iterable", {})

        if iterable.get("type") == "RANGE_CALL":
            args = iterable.get("arguments", {})
            start = args.get("start", "0")
            stop = args.get("stop", "10")
            step = args.get("step", "1")

            # Объявляем переменную цикла
            self.declare_variable(loop_var, "int")

            self.add_line(
                f"for (int {loop_var} = {start}; {loop_var} < {stop}; {loop_var} += {step}) {{"
            )
            self.indent_level += 1

            # Входим в scope цикла
            self.enter_scope()

            # Генерируем тело цикла
            for body_node in node.get("body", []):
                self.generate_graph_node(body_node)

            # Выходим из scope цикла
            self.exit_scope()

            self.indent_level -= 1
            self.add_line("}")

    def generate_attribute_assignment(self, node: Dict):
        """Генерирует присваивание атрибуту объекта (self.attr = value)"""
        object_name = node.get("object", "")
        attribute = node.get("attribute", "")
        value_ast = node.get("value", {})

        logger.debug(
            f"generate_attribute_assignment: {object_name}.{attribute} = {value_ast}"
        )

        # Если это self внутри метода класса
        if object_name == "self":
            # Находим текущий класс
            current_class = self._get_current_class()

            if current_class:
                # Генерируем выражение для значения
                value_expr = self.generate_expression(value_ast)

                # Добавляем присваивание
                self.add_line(f"self->{attribute} = {value_expr};")
                return

        # Если это другой объект
        var_info = self.get_variable_info(object_name)
        if var_info:
            obj_type = var_info.get("py_type", "")
            if self._is_class_type(obj_type):
                # Это объект класса
                value_expr = self.generate_expression(value_ast)
                self.add_line(f"{object_name}->{attribute} = {value_expr};")
                return

        # Fallback
        value_expr = self.generate_expression(value_ast)
        self.add_line(f"{object_name}.{attribute} = {value_expr};")

    def generate_assignment(self, node: Dict):
        """Генерирует присваивание с поддержкой строковых операций"""
        symbols = node.get("symbols", [])
        if not symbols:
            return

        target = symbols[0]
        expression_ast = node.get("expression_ast")

        if expression_ast:
            expression_ast["target"] = target

            # Проверяем, является ли это строковой операцией
            if expression_ast.get("type") == "binary_operation":
                operator = expression_ast.get("operator_symbol", "")
                left_ast = expression_ast.get("left", {})
                right_ast = expression_ast.get("right", {})

                left_is_string = self._is_string_expression(left_ast)
                right_is_string = self._is_string_expression(right_ast)

                if operator == "+" and (left_is_string or right_is_string):
                    # Генерируем конкатенацию строк
                    left_expr = self.generate_expression(left_ast)
                    right_expr = self.generate_expression(right_ast)

                    # Освобождаем старую память, если переменная уже была инициализирована
                    var_info = self.get_variable_info(target)
                    if var_info and var_info.get("py_type") == "str":
                        self.add_line(f"if ({target}) {{")
                        self.indent_level += 1
                        self.add_line(f"free({target});")
                        self.indent_level -= 1
                        self.add_line(f"}}")

                    self.add_line(
                        f"{target} = malloc(strlen({left_expr}) + strlen({right_expr}) + 1);"
                    )
                    self.add_line(f"if (!{target}) {{")
                    self.indent_level += 1
                    self.add_line(
                        f'fprintf(stderr, "Memory allocation failed for string concatenation\\n");'
                    )
                    self.add_line(f"exit(1);")
                    self.indent_level -= 1
                    self.add_line(f"}}")
                    self.add_line(f"strcpy({target}, {left_expr});")
                    self.add_line(f"strcat({target}, {right_expr});")
                    return

            # Обычное присваивание
            expr = self.generate_expression(expression_ast)

            # Для строковых литералов при присваивании
            if (
                expression_ast.get("type") == "literal"
                and expression_ast.get("data_type") == "str"
            ):
                var_info = self.get_variable_info(target)
                if var_info and var_info.get("py_type") == "str":
                    self.add_line(f"if ({target}) {{")
                    self.indent_level += 1
                    self.add_line(f"free({target});")
                    self.indent_level -= 1
                    self.add_line("}}")
                    self.add_line(f"{target} = malloc(strlen({expr}) + 1);")
                    self.add_line(f"strcpy({target}, {expr});")
                    return

            if expr is not None:
                self.add_line(f"{target} = {expr};")

    def generate_declaration(self, node: Dict):
        """Генерирует объявление переменной с поддержкой повторных объявлений"""
        var_name = node.get("var_name", "")
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast", {})

        logger.debug(f"Генерация объявления для {var_name}: {var_type}")

        if var_type.startswith("dict["):
            self._generate_dict_declaration(var_name, var_type, expression_ast, node)
            return

        # Проверяем, объявлена ли уже переменная
        var_info = self.get_variable_info(var_name)
        is_redeclaration = var_info is not None and not var_info.get(
            "is_deleted", False
        )

        if is_redeclaration:
            logger.debug(f"Переменная '{var_name}' уже объявлена, переобъявляем")

            # Освобождаем старую память
            old_py_type = var_info.get("py_type", "")
            if old_py_type.startswith("list["):
                struct_name = self.generate_list_struct_name(old_py_type)
                self.add_line(f"if ({var_name}) {{")
                self.indent_level += 1
                self.add_line(f"free_{struct_name}({var_name});")
                self.indent_level -= 1
                self.add_line("}")
            elif old_py_type == "str":
                self.add_line(f"if ({var_name}) {{")
                self.indent_level += 1
                self.add_line(f"free({var_name});")
                self.indent_level -= 1
                self.add_line("}")

        # Объявляем/обновляем переменную
        self.declare_variable(var_name, var_type)
        var_info = self.get_variable_info(var_name)

        if not var_info:
            return

        c_type = var_info["c_type"]

        # Проверяем, является ли выражение вызовом builtin функции
        if expression_ast and expression_ast.get("type") == "function_call":
            func_name = expression_ast.get("function", "")

            # Специальная обработка для builtin функций
            if func_name in ["str", "int", "float", "bool", "len"]:
                self._generate_builtin_declaration(
                    var_name, c_type, expression_ast, is_redeclaration
                )
                return

        # Обработка list[int] с литералом
        if expression_ast.get("type") == "list_literal" and var_type.startswith(
            "list["
        ):
            items = expression_ast.get("items", [])

            if is_redeclaration:
                # Для повторного объявления используем присваивание
                struct_name = self.generate_list_struct_name(var_type)
                self.add_line(
                    f"{var_name} = create_{struct_name}({max(len(items), INITIAL_LIST_CAPACITY)});"
                )
            else:
                # Для первого объявления генерируем объявление с инициализацией
                struct_name = self.generate_list_struct_name(var_type)
                self.add_line(
                    f"{c_type} {var_name} = create_{struct_name}({max(len(items), INITIAL_LIST_CAPACITY)});"
                )

            # Добавляем элементы
            for item_ast in items:
                item_expr = self.generate_expression(item_ast)
                self.add_line(f"append_{struct_name}({var_name}, {item_expr});")

            return

        # Обычная инициализация
        if expression_ast:
            expr = self.generate_expression(expression_ast)

            if is_redeclaration:
                # Повторное объявление = присваивание
                self.add_line(f"{var_name} = {expr};")
            else:
                # Первое объявление
                self.add_line(f"{c_type} {var_name} = {expr};")
        else:
            # Объявление без инициализации
            if not is_redeclaration:
                if c_type.endswith("*"):
                    self.add_line(f"{c_type} {var_name} = NULL;")
                else:
                    self.add_line(f"{c_type} {var_name};")

    def generate_redeclaration(self, node: Dict):
        """Генерирует код для повторного объявления переменной"""
        var_name = node.get("var_name", "")
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast", {})

        logger.debug(f"generate_redeclaration: {var_name}: {var_type}")

        # Получаем информацию о старой переменной
        old_var_info = self.get_variable_info(var_name)

        # Освобождаем старую память если нужно
        if old_var_info:
            old_py_type = old_var_info.get("py_type", "")

            if old_py_type.startswith("list["):
                struct_name = self.generate_list_struct_name(old_py_type)
                self.add_line(f"if ({var_name}) {{")
                self.indent_level += 1
                self.add_line(f"free_{struct_name}({var_name});")
                self.indent_level -= 1
                self.add_line("}")
            elif old_py_type.startswith("tuple["):
                struct_name = self.generate_tuple_struct_name(old_py_type)
                self.add_line(f"free_{struct_name}({var_name});")
            elif old_py_type == "str":
                self.add_line(f"if ({var_name}) {{")
                self.indent_level += 1
                self.add_line(f"free({var_name});")
                self.indent_level -= 1
                self.add_line("}")

        # Обновляем переменную в scope
        self.declare_variable(var_name, var_type)

        # Генерируем код для нового значения
        if expression_ast:
            if (
                var_type.startswith("list[")
                and expression_ast.get("type") == "list_literal"
            ):
                # Генерируем код для нового списка
                self._generate_list_redeclaration(var_name, var_type, expression_ast)
            elif (
                var_type.startswith("tuple[")
                and expression_ast.get("type") == "tuple_literal"
            ):
                # Генерируем код для нового кортежа
                self._generate_tuple_redeclaration(var_name, var_type, expression_ast)
            else:
                # Обычное присваивание
                expr = self.generate_expression(expression_ast)
                self.add_line(f"{var_name} = {expr};")

    def generate_delete(self, node: Dict):
        """Генерирует код для del с поддержкой tuple и list"""
        symbols = node.get("symbols", [])

        for target in symbols:
            self.mark_variable_deleted(target, "full")
            var_info = self.get_variable_info(target)

            if not var_info:
                self.add_line(f"// ERROR: Переменная '{target}' не найдена для del")
                continue

            self.add_line(f"// del {target}")

            py_type = var_info.get("py_type", "")
            c_type = var_info.get("c_type", "")

            if py_type.startswith("dict["):
                # Для словаря вызываем функцию очистки
                # Извлекаем типы ключа и значения для имени структуры
                key_type, value_type = self._extract_dict_types(py_type)
                key_name = self.clean_type_name_for_c(key_type)
                value_name = self.clean_type_name_for_c(value_type)
                struct_name = f"dict_{key_name}_{value_name}"

                self.add_line(f"if ({target}) {{")
                self.indent_level += 1
                self.add_line(f"free_{struct_name}({target});")
                self.indent_level -= 1
                self.add_line("}")
                self.add_line(f"{target} = NULL;")
            elif py_type.startswith("list["):
                # Для list вызываем функцию очистки
                struct_name = self.generate_list_struct_name(py_type)
                self.add_line(f"if ({target}) {{")
                self.indent_level += 1
                self.add_line(f"free_{struct_name}({target});")
                self.indent_level -= 1
                self.add_line("}")
                self.add_line(f"{target} = NULL;")

            elif py_type.startswith("tuple["):
                # Для tuple вызываем функцию очистки
                struct_name = self.generate_tuple_struct_name(py_type)
                self.add_line(f"free_{struct_name}(&{target});")

            elif var_info["is_pointer"]:
                self.add_line(f"if ({target} != NULL) {{")
                self.indent_level += 1
                self.add_line(f"free({target});")
                self.indent_level -= 1
                self.add_line("}")
                self.add_line(f"{target} = NULL;")
            else:
                if c_type in ["int", "float", "double", "long"]:
                    self.add_line(f"{target} = 0;")
                elif c_type == "bool":
                    self.add_line(f"{target} = false;")
                elif "char*" in c_type or c_type.endswith("*"):
                    self.add_line(f"{target} = NULL;")
                else:
                    self.add_line(f"// {target} обнулена")
