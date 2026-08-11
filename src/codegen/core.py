from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger
from src.codegen.class_model import ClassRegistry

class CoreMixin:
    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.temp_var_counter = 0
        self.function_declarations = []
        self.function_parameters = {}
        self.c_imports = []

        # Улучшенная система управления переменными
        self.variable_scopes = []  # Стек scope'ов
        self.current_scope_level = 0

        # Структуры для типов
        self.generated_helpers = []
        self.helper_declarations = []  # Декларации helper-функций

        # Для отслеживания типов классов
        self.class_types = set()  # Имена классов
        self.struct_types = set()  # Имена структур (включая tuple/list)
        # Типы, которые являются указателями (используют ->)
        self.pointer_types = set()

        self.class_registry = ClassRegistry()

        # Расширенный маппинг типов Python -> C
        self.type_map = {
            "int": "int",
            "float": "double",
            "float16": "_Float16",
            "float32": "float",
            "float64": "double",
            "int8": "int8_t",
            "int16": "int16_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "uint8": "uint8_t",
            "uint16": "uint16_t",
            "uint32": "uint32_t",
            "uint64": "uint64_t",
            "str": "char*",
            "bool": "bool",
            "None": "void",
            "null": "void*",
            "list": "void*",
            "dict": "void*",
            "set": "void*",
            "function": "void*",
            "tuple": "void*",
            "bytes": "unsigned char*",
            "bytearray": "unsigned char*",
        }

        self.known_c_types = set(KNOWN_C_TYPES)

        # Поддержка обобщенных типов
        self.generic_type_map = {}  # Кэш для сгенерированных типов

        # Поддерживаемые операции
        self.operator_map = {
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "//": "/",  # Целочисленное деление
            "%": "%",
            "**": "pow",  # Степень
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "==": "==",
            "!=": "!=",
            "and": "&&",
            "or": "||",
            "not": "!",
        }

        # Для отслеживания уже сгенерированных функций
        self.generated_functions = set()  # Имена уже сгенерированных функций
        self.generated_structures = set()  # Имена уже сгенерированных структур
        self.global_init_nodes = []
        self.current_function_return_type = None
        self.current_function_name = None
        self.phils_function_names = set()
        self.runtime_needs_memory = False
        self.runtime_needs_sort_helpers = False
        self.runtime_needs_string_helpers = False
        self.runtime_needs_int_helpers = False
        self.tensor_index_ranks = set()
        self.tensor_fast_access = {}
        self.tensor_fast_loop_bounds = {}
        self.tensor_fast_patterns = set()

    def reset(self):
        """Reset all per-compilation mutable state."""
        self.output = []
        self.indent_level = 0
        self.temp_var_counter = 0
        self.function_declarations = []
        self.function_parameters = {}
        self.c_imports = []
        self.variable_scopes = [{}]
        self.current_scope_level = 0
        self.generated_helpers = []
        self.helper_declarations = []
        self.generic_type_map = {}
        self.class_types = set()
        self.struct_types = set()
        self.pointer_types = set()
        self.class_registry = ClassRegistry()
        self.generated_functions = set()
        self.generated_structures = set()
        self.global_init_nodes = []
        self.current_function_return_type = None
        self.current_function_name = None
        self.phils_function_names = set()
        self.runtime_needs_memory = False
        self.runtime_needs_sort_helpers = False
        self.runtime_needs_string_helpers = False
        self.runtime_needs_int_helpers = False
        self.tensor_index_ranks = set()
        self.tensor_fast_access = {}
        self.tensor_fast_loop_bounds = {}
        self.tensor_fast_patterns = set()
        self.known_c_types = set(KNOWN_C_TYPES)
        logger.debug("CCodeGenerator state reset")

    def indent(self) -> str:
        """Возвращает отступ для текущего уровня"""
        return "    " * self.indent_level

    def add_line(self, line: str):
        """Добавляет строку с правильным отступом"""
        self.output.append(self.indent() + line)

    def add_empty_line(self):
        """Добавляет пустую строку"""
        self.output.append("")
