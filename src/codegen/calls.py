from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
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
            if isinstance(arg, Mapping):
                arg_strings.append(self.generate_expression(arg))
            else:
                if str(arg) == "None":
                    arg_strings.append("NULL")
                else:
                    arg_strings.append(str(arg))

        self.consume_owned_call_arguments(node.get("function", ""), args)

        args_str = ", ".join(arg_strings)
        self.add_line(f"{func_name}({args_str});")

    def generate_method_call(self, node: Dict):
        """Dispatch method lowering by semantic type instead of one giant branch."""
        object_name = node.get("object", "")
        method_name = node.get("method", "")
        args = node.get("arguments", [])
        is_standalone = node.get("is_standalone", False)
        target_var = node.get("target", "")

        obj_type, object_expression = self.resolve_object_path(object_name)
        if not obj_type:
            raise RuntimeError(f"unknown object '{object_name}'")
        if "." not in object_name and object_name != "self":
            info = self.get_variable_info(object_name)
            if info and (info.get("is_deleted") or info.get("is_moved")):
                raise RuntimeError(f"use of dead value '{object_name}'")

        arg_strings = [
            self.generate_expression(arg) if isinstance(arg, Mapping) else str(arg)
            for arg in args
        ]

        if obj_type.startswith("list["):
            return self._generate_list_method_call(
                object_expression, obj_type, method_name, arg_strings, is_standalone, target_var
            )
        if obj_type == "str":
            return self._generate_string_method_call(
                object_expression, method_name, arg_strings, is_standalone, target_var
            )
        if obj_type.startswith("dict["):
            return self._generate_dict_method_call(
                object_expression, obj_type, method_name, arg_strings, is_standalone, target_var
            )
        if obj_type.startswith("tuple["):
            return self._generate_tuple_method_call(
                object_expression, obj_type, method_name, arg_strings, is_standalone, target_var
            )
        if self.is_device_tensor_type(obj_type):
            full_args = object_expression
            if arg_strings:
                full_args += ", " + ", ".join(arg_strings)
            expr = f"Tensor_{method_name}({full_args})"
            if is_standalone:
                self.add_line(f"{expr};")
                return None
            return expr

        if self._is_class_type(obj_type):
            full_args = object_expression
            if arg_strings:
                full_args += ", " + ", ".join(arg_strings)
            expr = f"{obj_type}_{method_name}({full_args})"
            if is_standalone:
                self.add_line(f"{expr};")
                return None
            return expr

        raise RuntimeError(f"method '{method_name}' is not available for type '{obj_type}'")

    def _store_owned_result(self, target_var: str, expr: str, py_type: str) -> None:
        info = self.get_variable_info(target_var)
        if not info:
            self.add_line(f"{target_var} = {expr};")
            return
        kind = self.memory_kind_for_type(py_type or info.get("py_type", ""))
        if kind == self.MEMORY_ARC:
            self.assert_can_move_or_delete(target_var)
            self.add_line(f"ocean_release({target_var});")
            self.add_line(f"{target_var} = {expr};")
            info["owns_reference"] = True
            info["is_deleted"] = False
        elif kind == self.MEMORY_STRING:
            self.assert_can_move_or_delete(target_var)
            self.add_line(f"free({target_var});")
            self.add_line(f"{target_var} = {expr};")
            info["owns_reference"] = True
            info["is_deleted"] = False
        else:
            self.add_line(f"{target_var} = {expr};")

    def _generate_list_method_call(
        self, object_name, obj_type, method_name, arg_strings, is_standalone, target_var
    ):
        struct_name = self.generate_list_struct_name(obj_type)
        element_type = self._parse_list_type(obj_type) or "int"
        c_element_type = self.map_type_to_c(element_type)
        mutating = {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse"}
        if method_name in mutating:
            self.assert_can_mutate(object_name)

        if method_name == "append":
            if len(arg_strings) != 1: raise RuntimeError("list.append() expects one argument")
            self.add_line(f"append_{struct_name}({object_name}, {arg_strings[0]});")
            return None
        if method_name == "extend":
            if len(arg_strings) != 1: raise RuntimeError("list.extend() expects one list")
            other = arg_strings[0]
            self.add_line(f"extend_{struct_name}({object_name}, {other}->data, {other}->size);")
            return None
        if method_name == "insert":
            if len(arg_strings) != 2: raise RuntimeError("list.insert() expects index and value")
            self.add_line(f"insert_{struct_name}({object_name}, {arg_strings[0]}, {arg_strings[1]});")
            return None
        if method_name == "remove":
            if len(arg_strings) != 1: raise RuntimeError("list.remove() expects one value")
            self.add_line(f"remove_{struct_name}({object_name}, {arg_strings[0]});")
            return None
        if method_name == "clear":
            self.add_line(f"clear_{struct_name}({object_name});")
            return None
        if method_name == "reverse":
            self.add_line(f"reverse_{struct_name}({object_name});")
            return None
        if method_name == "sort":
            if element_type == "int": compare = "compare_int"
            elif element_type in {"float", "double"}: compare = "compare_double"
            elif element_type == "str": compare = "compare_string"
            else: raise RuntimeError(f"sort is not defined for list[{element_type}] yet")
            self.add_line(
                f"qsort({object_name}->data, {object_name}->size, sizeof({c_element_type}), {compare});"
            )
            return None
        if method_name == "count":
            if len(arg_strings) != 1: raise RuntimeError("list.count() expects one value")
            expr = f"count_{struct_name}({object_name}, {arg_strings[0]})"
            if target_var: self.add_line(f"{target_var} = {expr};"); return None
            if is_standalone: self.add_line(f"(void){expr};"); return None
            return expr
        if method_name == "index":
            if len(arg_strings) != 1: raise RuntimeError("list.index() expects one value")
            expr = f"index_{struct_name}({object_name}, {arg_strings[0]})"
            if target_var: self.add_line(f"{target_var} = {expr};"); return None
            if is_standalone: self.add_line(f"(void){expr};"); return None
            return expr
        if method_name == "pop":
            index = arg_strings[0] if arg_strings else "-1"
            expr = f"pop_{struct_name}({object_name}, {index})"
            if target_var:
                self._store_owned_result(target_var, expr, element_type)
                return None
            if is_standalone:
                temp = f"ocean_pop_tmp_{self.temp_var_counter}"
                self.temp_var_counter += 1
                self.add_line(f"{c_element_type} {temp} = {expr};")
                kind = self.memory_kind_for_type(element_type)
                if kind == self.MEMORY_ARC: self.add_line(f"ocean_release({temp});")
                elif kind == self.MEMORY_STRING: self.add_line(f"free({temp});")
                return None
            return expr
        raise RuntimeError(f"list method '{method_name}' is not implemented")

    def _generate_string_method_call(
        self, object_name, method_name, arg_strings, is_standalone, target_var
    ):
        unary = {
            "upper": "string_upper", "lower": "string_lower", "capitalize": "string_capitalize",
            "title": "string_title", "strip": "string_strip", "lstrip": "string_lstrip",
            "rstrip": "string_rstrip",
        }
        if method_name in unary:
            expr = f"{unary[method_name]}({object_name})"
            result_type = "str"
        elif method_name == "format":
            if len(arg_strings) != 1: raise RuntimeError("str.format currently supports one argument")
            expr = f"string_format({object_name}, {arg_strings[0]})"; result_type = "str"
        elif method_name == "replace":
            if len(arg_strings) != 2: raise RuntimeError("str.replace expects old,new")
            expr = f"string_replace({object_name}, {arg_strings[0]}, {arg_strings[1]})"; result_type = "str"
        elif method_name == "split":
            delimiter = arg_strings[0] if arg_strings else '" "'
            self.generate_list_struct("list[str]")
            expr = f"string_split({object_name}, {delimiter})"; result_type = "list[str]"
        elif method_name in {"find", "index", "count"}:
            if len(arg_strings) != 1: raise RuntimeError(f"str.{method_name} expects one argument")
            expr = f"string_{method_name}({object_name}, {arg_strings[0]})"; result_type = "int"
        elif method_name in {"startswith", "endswith"}:
            if len(arg_strings) != 1: raise RuntimeError(f"str.{method_name} expects one argument")
            expr = f"string_{method_name}({object_name}, {arg_strings[0]})"; result_type = "bool"
        elif method_name in {"isdigit", "isalpha", "isalnum", "islower", "isupper"}:
            expr = f"string_{method_name}({object_name})"; result_type = "bool"
        else:
            raise RuntimeError(f"str method '{method_name}' is not implemented")

        if target_var:
            self._store_owned_result(target_var, expr, result_type)
            return None
        if is_standalone:
            kind = self.memory_kind_for_type(result_type)
            if kind == self.MEMORY_ARC: self.add_line(f"ocean_release({expr});")
            elif kind == self.MEMORY_STRING: self.add_line(f"free({expr});")
            else: self.add_line(f"(void){expr};")
            return None
        return expr

    def _generate_dict_method_call(
        self, object_name, obj_type, method_name, arg_strings, is_standalone, target_var
    ):
        key_type, value_type = self._extract_dict_types(obj_type)
        struct_name = f"dict_{self.clean_type_name_for_c(key_type)}_{self.clean_type_name_for_c(value_type)}"
        if method_name == "keys": expr, result_type = f"keys_{struct_name}({object_name})", f"list[{key_type}]"
        elif method_name == "values": expr, result_type = f"values_{struct_name}({object_name})", f"list[{value_type}]"
        elif method_name == "get":
            if not arg_strings: raise RuntimeError("dict.get expects key")
            default = arg_strings[1] if len(arg_strings) > 1 else self.default_value_for_type(value_type)
            expr, result_type = f"get_default_{struct_name}({object_name}, {arg_strings[0]}, {default})", value_type
        else:
            raise RuntimeError(f"dict method '{method_name}' is not implemented")
        if target_var:
            # get() returns a borrow; keys()/values() return owned collections.
            if method_name == "get" and self.memory_kind_for_type(result_type) == self.MEMORY_ARC:
                self.add_line(f"ocean_retain({expr});")
            if method_name == "get" and self.memory_kind_for_type(result_type) == self.MEMORY_STRING:
                expr = f"ocean_strdup({expr})"
            self._store_owned_result(target_var, expr, result_type)
            return None
        if is_standalone:
            if method_name in {"keys", "values"}: self.add_line(f"ocean_release({expr});")
            else: self.add_line(f"(void){expr};")
            return None
        return expr

    def _generate_tuple_method_call(
        self, object_name, obj_type, method_name, arg_strings, is_standalone, target_var
    ):
        if method_name not in {"count", "index"} or len(arg_strings) != 1:
            raise RuntimeError(f"tuple method '{method_name}' is not implemented")
        # Immutable tuple operations can be expressed directly and never mutate ownership.
        inner = re.match(r"tuple\[([^\]]+)\]", obj_type)
        element_type = inner.group(1).strip() if inner else "int"
        if element_type == "str": comparison = f"strcmp({object_name}->data[i], {arg_strings[0]}) == 0"
        else: comparison = f"{object_name}->data[i] == {arg_strings[0]}"
        temp = f"ocean_tuple_{method_name}_{self.temp_var_counter}"
        self.temp_var_counter += 1
        if method_name == "count":
            self.add_line(f"int {temp} = 0;")
            self.add_line(f"for (int i = 0; i < {object_name}->size; ++i) if ({comparison}) ++{temp};")
        else:
            self.add_line(f"int {temp} = -1;")
            self.add_line(f"for (int i = 0; i < {object_name}->size; ++i) if ({comparison}) {{ {temp} = i; break; }}")
            self.add_line(f"if ({temp} < 0) {{ fprintf(stderr, \"ValueError: tuple.index(x): x not in tuple\\n\"); exit(1); }}")
        if target_var: self.add_line(f"{target_var} = {temp};"); return None
        if is_standalone: return None
        return temp

    def default_value_for_type(self, py_type: str) -> str:
        kind = self.memory_kind_for_type(py_type)
        if kind in {self.MEMORY_ARC, self.MEMORY_STRING, self.MEMORY_RAW, self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
            return "NULL"
        if py_type == "bool": return "false"
        if py_type in {"float", "double"}: return "0.0"
        return "0"

    def generate_object_method_call(self, node: Dict):
        """Compatibility lowering for legacy/static_method_call parser nodes.

        Older parser versions stored the receiver in ``class_name`` even when
        the node represented an instance method.  Route it through the single
        ownership-aware method dispatcher so list/class mutations cannot bypass
        borrow checks.
        """
        normalized = {
            "object": node.get("object") or node.get("class_name", ""),
            "method": node.get("method", ""),
            "arguments": node.get("arguments", []),
            "is_standalone": node.get("is_standalone", True),
            "target": node.get("target", ""),
        }
        return self.generate_method_call(normalized)

    def generate_c_call(self, node: Dict):
        """Генерирует прямой вызов C-функции"""
        if not node.get("unsafe", False):
            raise RuntimeError(
                "direct C calls require an explicit unsafe: block"
            )
        func_name = node.get("function", "")
        args = node.get("arguments", [])

        # Генерируем аргументы
        arg_strings = []
        for arg in args:
            if isinstance(arg, Mapping):
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
            temporary_cleanup = []

            for arg in args:
                if isinstance(arg, Mapping):
                    if arg.get("type") == "attribute_access":
                        expr = self.generate_attribute_access(arg)
                        object_name = arg.get("object", "")
                        object_info = self.get_variable_info(object_name)
                        object_type = (
                            self.strip_borrow_type(object_info.get("py_type", ""))
                            if object_info
                            else ""
                        )
                        if (
                            (
                                self.is_array_type(object_type)
                                or self.is_device_tensor_type(object_type)
                            )
                            and arg.get("attribute") in {"size", "capacity", "ndim"}
                        ):
                            format_parts.append("%zu")
                        else:
                            format_parts.append("%d")
                        value_parts.append(expr)
                    elif arg.get("type") == "complex_attribute_access":
                        expr = self.generate_expression(arg)
                        object_name = arg.get("object", "")
                        object_info = self.get_variable_info(object_name)
                        object_type = (
                            self.strip_borrow_type(object_info.get("py_type", ""))
                            if object_info
                            else ""
                        )
                        if (
                            self.is_device_tensor_type(object_type)
                            and arg.get("attribute") == "shape"
                        ):
                            format_parts.append("%d")
                        else:
                            format_parts.append("%d")
                        value_parts.append(expr)
                    elif arg.get("type") == "variable":
                        var_name = arg.get("value", "")
                        var_info = self.get_variable_info(var_name)
                        if var_info:
                            var_type = var_info.get("py_type", "")
                            if var_type in {"int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
                                format_parts.append("%d")
                                value_parts.append(var_name)
                            elif var_type in {"float", "float16", "float32", "float64", "double"}:
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
                        elif data_type in {"float", "float16", "float32", "float64", "double"}:
                            format_parts.append("%f")
                            value_parts.append(str(value))
                        else:
                            format_parts.append("%d")
                            value_parts.append(str(value))
                    elif arg.get("type") == "method_call":
                        expr = self.generate_expression(arg)
                        method_name = arg.get("method", "")
                        if method_name in {"device", "dtype"}:
                            temporary = f"ocean_print_tmp_{self.temp_var_counter}"
                            self.temp_var_counter += 1
                            self.add_line(f"char* {temporary} = {expr};")
                            expr = temporary
                            temporary_cleanup.append(temporary)
                            format_parts.append("%s")
                        elif method_name == "size":
                            format_parts.append("%zu")
                        elif method_name in {"sum", "mean", "max", "min", "item"}:
                            format_parts.append("%f")
                        elif method_name == "is_contiguous":
                            format_parts.append("%d")
                        elif method_name == "get":
                            format_parts.append("%f")
                        else:
                            format_parts.append("%d")
                        value_parts.append(expr)
                    elif arg.get("type") in {"index_access", "tensor_index_access"}:
                        expr = self.generate_expression(arg)
                        source = arg.get("variable", "")
                        source_info = self.get_variable_info(source)
                        element_type = "int"
                        if source_info:
                            source_type = self.strip_borrow_type(source_info.get("py_type", ""))
                            if self.is_array_type(source_type):
                                element_type = self.array_element_type(source_type)
                            elif self.is_device_tensor_type(source_type):
                                # Public Tensor.get() exposes numeric values
                                # through the stable float64 scalar ABI.
                                element_type = "float64"
                        format_parts.append("%f" if element_type in {"float", "float16", "float32", "float64", "double"} else "%d")
                        value_parts.append(expr)
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
            for temporary in temporary_cleanup:
                self.add_line(f"free({temporary});")
        elif func_name == "input":
            # Для input() без присваивания
            self.generate_input_statement(node)
            return
        else:
            # Обработка других встроенных функций
            # Генерируем аргументы
            arg_strings = []
            for arg in args:
                if isinstance(arg, Mapping):
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
                if isinstance(args[0], Mapping):
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
            if isinstance(arg, Mapping):
                arg_strings.append(self.generate_expression(arg))
            else:
                arg_strings.append(str(arg))

        args_str = ", ".join(arg_strings)

        # Специальная обработка для len()
        if func_name == "len" and args:
            # Определяем тип аргумента
            arg_expr = args[0]
        if isinstance(arg_expr, Mapping):
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
            if isinstance(arg, Mapping):
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
        if args and isinstance(args[0], Mapping):
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
                    self.add_line(f"char* ocean_builtin_tmp = ocean_strdup({arg_expr});")
                    self.add_line(f"free({var_name});")
                    self.add_line(f"{var_name} = ocean_builtin_tmp;")
                else:
                    self.add_line(f"{c_type} {var_name} = ocean_strdup({arg_expr});")
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
                func_call = f"(int)({arg_expr})"

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
            elif self.is_array_type(arg_type):
                self.generate_array_struct(arg_type)
                func_call = f"{self.array_struct_name(arg_type)}_len({arg_expr})"
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
