from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class IoMixin:
    def generate_input_expression(self, node: Dict) -> str:
        """Генерирует выражение с input() и возвращает имя переменной с результатом"""
        args = node.get("arguments", [])

        # Создаем уникальное имя для временной переменной
        temp_var = self.generate_temporary_var("str")

        # Объявляем переменную
        self.declare_variable(temp_var, "str")

        # Получаем информацию о переменной для генерации правильного типа
        var_info = self.get_variable_info(temp_var)
        c_type = var_info["c_type"] if var_info else "char*"

        # Объявляем переменную
        self.add_line(f"{c_type} {temp_var} = NULL;")

        # Генерируем prompt если есть аргументы
        if args:
            self._generate_input_prompt(args)

        # Генерируем код для чтения ввода
        self._generate_input_read_code_direct(temp_var)

        return temp_var

    def generate_input_statement(self, node: Dict):
        """Генерирует вызов input() как отдельный statement (без присваивания)"""
        args = node.get("arguments", [])

        # Генерируем prompt если есть аргументы
        if args:
            self._generate_input_prompt(args)

        # Читаем ввод, но игнорируем результат
        temp_var = self.generate_temporary_var("str")
        buffer_var = f"{temp_var}_buffer"

        self.add_line(f"char {buffer_var}[256];")
        self.add_line(f"fgets({buffer_var}, sizeof({buffer_var}), stdin);")
        self.add_line(f'{buffer_var}[strcspn({buffer_var}, "\\n")] = 0;')
        self.add_line(f"// Ввод прочитан, результат игнорируется")

    def _generate_input_prompt(self, args: List):
        """Генерирует код для вывода prompt в input()"""
        format_parts = []
        value_parts = []

        for arg in args:
            if isinstance(arg, dict):
                if arg.get("type") == "literal" and arg.get("data_type") == "str":
                    # Строковый литерал
                    value = arg.get("value", "")
                    format_parts.append(f"{value}")
                else:
                    # Другие выражения (переменные, вызовы функций и т.д.)
                    expr = self.generate_expression(arg)
                    format_parts.append("%s")
                    value_parts.append(expr)
            else:
                # Простая строка (не должно быть в нормальном AST)
                format_parts.append(str(arg))

        # Собираем prompt строку
        if format_parts:
            prompt = " ".join(format_parts)

            if value_parts:
                # Если есть динамические части (переменные)
                args_str = ", ".join(value_parts)
                self.add_line(f'printf("{prompt}", {args_str});')
            else:
                # Простой строковый литерал
                self.add_line(f'printf("{prompt}");')

    def _generate_input_read_code(self, target_var: str):
        """Генерирует код для чтения ввода с клавиатуры"""
        # Создаем буфер для ввода
        buffer_var = f"{target_var}_buffer"

        # Выделяем память для буфера
        self.add_line(f"char {buffer_var}[256];")

        # Читаем строку с stdin
        self.add_line(f"fgets({buffer_var}, sizeof({buffer_var}), stdin);")

        # Убираем символ новой строки
        self.add_line(f'{buffer_var}[strcspn({buffer_var}, "\\n")] = 0;')

        # Выделяем память для результата и копируем
        self.add_line(f"{target_var} = malloc(strlen({buffer_var}) + 1);")
        self.add_line(f"if (!{target_var}) {{")
        self.indent_level += 1
        self.add_line(
            f'fprintf(stderr, "Memory allocation failed for input result\\n");'
        )
        self.add_line(f"exit(1);")
        self.indent_level -= 1
        self.add_line(f"}}")
        self.add_line(f"strcpy({target_var}, {buffer_var});")

    def _generate_input_read_code_direct(self, target_var: str):
        """Генерирует код для чтения ввода с клавиатуры прямо в целевую переменную"""
        buffer_var = f"{target_var}_buffer"

        # Выделяем память для буфера
        self.add_line(f"char {buffer_var}[256];")

        # Читаем строку с stdin
        self.add_line(
            f"if (fgets({buffer_var}, sizeof({buffer_var}), stdin) == NULL) {{"
        )
        self.indent_level += 1
        self.add_line(f"// Достигнут конец файла (EOF)")
        self.add_line(f"{target_var} = NULL;")
        self.indent_level -= 1
        self.add_line(f"}} else {{")
        self.indent_level += 1
        self.add_line(f"// Успешно прочитали строку")
        self.add_line(f'{buffer_var}[strcspn({buffer_var}, "\\n")] = 0;')

        # Освобождаем предыдущую память
        self.add_line(f"if ({target_var} != NULL) {{")
        self.indent_level += 1
        self.add_line(f"free({target_var});")
        self.indent_level -= 1
        self.add_line(f"}}")

        # Выделяем память для результата
        self.add_line(f"{target_var} = malloc(strlen({buffer_var}) + 1);")
        self.add_line(f"if (!{target_var}) {{")
        self.indent_level += 1
        self.add_line(
            f'fprintf(stderr, "Memory allocation failed for input result\\n");'
        )
        self.add_line(f"exit(1);")
        self.indent_level -= 1
        self.add_line(f"}}")
        self.add_line(f"strcpy({target_var}, {buffer_var});")
        self.indent_level -= 1
        self.add_line(f"}}")

    def _generate_string_concatenation(
        self, target_var: str, left_expr: str, right_expr: str, c_type: str
    ):
        """Генерирует правильную конкатенацию строк"""
        # Определяем, являются ли выражения строковыми литералами
        left_is_literal = left_expr.startswith('"') and left_expr.endswith('"')
        right_is_literal = right_expr.startswith('"') and right_expr.endswith('"')

        if left_is_literal and right_is_literal:
            # Оба литерала - можно вычислить на этапе компиляции
            left_str = left_expr[1:-1]  # Убираем кавычки
            right_str = right_expr[1:-1]
            result_str = f'"{left_str}{right_str}"'

            self.add_line(f"{c_type} {target_var} = malloc(strlen({result_str}) + 1);")
            self.add_line(f"if (!{target_var}) {{")
            self.indent_level += 1
            self.add_line(f'fprintf(stderr, "Memory allocation failed\\n");')
            self.add_line(f"exit(1);")
            self.indent_level -= 1
            self.add_line(f"}}")
            self.add_line(f"strcpy({target_var}, {result_str});")

        else:
            # Одно или оба выражения - переменные или сложные выражения
            # Создаем временные переменные для длин
            temp_len1 = self.generate_temporary_var("int")
            temp_len2 = self.generate_temporary_var("int")

            self.add_line(f"int {temp_len1} = strlen({left_expr});")
            self.add_line(f"int {temp_len2} = strlen({right_expr});")

            # Выделяем память
            self.add_line(
                f"{c_type} {target_var} = malloc({temp_len1} + {temp_len2} + 1);"
            )
            self.add_line(f"if (!{target_var}) {{")
            self.indent_level += 1
            self.add_line(
                f'fprintf(stderr, "Memory allocation failed for string concatenation\\n");'
            )
            self.add_line(f"exit(1);")
            self.indent_level -= 1
            self.add_line(f"}}")

            # Копируем строки
            self.add_line(f"strcpy({target_var}, {left_expr});")
            self.add_line(f"strcat({target_var}, {right_expr});")

    def generate_input(self, node: Dict):
        """Генерирует код для функции input()"""
        args = node.get("arguments", [])

        # Форматная строка для prompt (если есть)
        format_str = ""
        value_parts = []

        if args:
            # Создаем форматную строку для prompt
            format_parts = []
            for arg in args:
                if isinstance(arg, dict):
                    if arg.get("type") == "literal" and arg.get("data_type") == "str":
                        # Строковый литерал
                        value = arg.get("value", "")
                        format_parts.append(f"{value}")
                    else:
                        # Другие выражения
                        expr = self.generate_expression(arg)
                        format_parts.append("%s")
                        value_parts.append(expr)
                else:
                    # Простая строка
                    format_parts.append(str(arg))

            # Собираем строку
            prompt = " ".join(format_parts)
            format_str = f'printf("{prompt}"); '

        # Добавляем чтение ввода
        # Создаем временную переменную для результата input()
        temp_var = self.generate_temporary_var("str")
        self.add_line(
            f"{format_str}char {temp_var}[256]; fgets({temp_var}, sizeof({temp_var}), stdin);"
        )

        # Убираем символ новой строки в конце
        self.add_line(f'{temp_var}[strcspn({temp_var}, "\\n")] = 0;')

        # Если input() используется в выражении, нужно вернуть значение
        # Для этого создадим узел с результатом
        return temp_var
