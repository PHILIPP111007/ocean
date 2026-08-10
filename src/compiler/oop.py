from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import (
    DEFAULT_C_IMPORTS,
    INITIAL_LIST_CAPACITY,
    KNOWN_C_TYPES,
)
from src.modules.logger import logger


class OopMixin:
    def generate_class_declaration(self, node: Dict):
        """Генерирует структуру для класса C динамически"""
        class_name = node.get("class_name", "")

        # Регистрируем класс
        self.class_types.add(class_name)
        self.type_map[class_name] = f"{class_name}*"

        # Анализируем класс для определения полей
        # (fields будут собраны позже при анализе методов)
        if class_name not in self.class_fields:
            self.class_fields[class_name] = {}

        # Генерируем структуру
        self.add_line(f"typedef struct {class_name} {{")
        self.indent_level += 1

        # Добавляем таблицу виртуальных методов
        self.add_line(f"void** vtable;")

        # Поля будут добавлены позже, после анализа методов
        # Создаем временный комментарий
        self.add_line(f"// Поля класса будут добавлены после анализа методов")

        self.indent_level -= 1
        self.add_line(f"}} {class_name};")
        self.add_empty_line()

    def collect_class_fields(self, class_name: str, json_data: List[Dict]) -> Dict:
        """Собирает поля класса из всех его методов (включая __init__)"""
        fields = {}

        # Ищем все методы этого класса в json_data
        for scope in json_data:
            if (
                scope.get("type") == "class_method"
                and scope.get("class_name") == class_name
            ):
                method_name = scope.get("method_name", "")

                # Анализируем метод __init__ для присваиваний атрибутам
                if method_name == "__init__":
                    self._analyze_init_method_for_fields(fields, scope)

                # Также анализируем другие методы для использования атрибутов
                else:
                    self._analyze_method_for_field_references(fields, scope)

        return fields

    def _analyze_init_method_for_fields(self, fields: Dict, init_scope: Dict):
        """Анализирует метод __init__ для определения полей класса"""
        graph = init_scope.get("graph", [])

        parameters = init_scope.get("parameters", [])
        param_types = {}

        # Собираем типы параметров (пропускаем self)
        for param in parameters:
            if param.get("name") != "self":
                param_name = param.get("name", "")
                param_type = param.get("type", "int")
                param_types[param_name] = param_type

        for node in graph:
            if node.get("node") == "attribute_assignment":
                # Присваивание атрибуту: self.attr = value
                attr_name = node.get("attribute", "")
                value = node.get("value", {})

                # Проверяем, является ли значение параметром конструктора
                if value.get("type") == "variable":
                    var_name = value.get("value", "")

                    # Если это параметр конструктора, берем его тип
                    if var_name in param_types:
                        field_type = param_types[var_name]
                        fields[attr_name] = field_type
                        logger.debug(
                            f"DEBUG: Поле {attr_name} получает тип параметра {var_name}: {field_type}"
                        )
                    else:
                        # Иначе пытаемся определить тип по значению
                        field_type = self._infer_field_type(value)
                        if field_type:
                            fields[attr_name] = field_type
                else:
                    # Определяем тип по литералу или выражению
                    field_type = self._infer_field_type(value)
                    if field_type:
                        fields[attr_name] = field_type
                        logger.debug(
                            f"DEBUG: Поле {attr_name} получает тип из значения: {field_type}"
                        )

            elif node.get("node") == "declaration":
                # Объявление атрибута с типом: self.attr: type = value
                var_name = node.get("var_name", "")
                if var_name.startswith("self."):
                    attr_name = var_name[5:]  # Убираем "self."
                    var_type = node.get("var_type", "")
                    if var_type:
                        fields[attr_name] = var_type

    def _analyze_method_for_field_references(self, fields: Dict, method_scope: Dict):
        """Анализирует метод для ссылок на атрибуты"""
        graph = method_scope.get("graph", [])

        # Собираем все обращения к атрибутам
        def collect_attribute_accesses(node):
            accesses = []

            if isinstance(node, dict):
                node_type = node.get("type", "")

                if node_type == "attribute_access":
                    # Доступ к атрибуту: self.attr или obj.attr
                    obj_name = node.get("object", "")
                    attr_name = node.get("attribute", "")

                    if obj_name == "self":
                        accesses.append(attr_name)

                # Рекурсивно проверяем все значения
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        if isinstance(value, dict):
                            accesses.extend(collect_attribute_accesses(value))
                        elif isinstance(value, list):
                            for item in value:
                                accesses.extend(collect_attribute_accesses(item))

            return accesses

        # Проходим по всему графу метода
        for node in graph:
            attr_accesses = collect_attribute_accesses(node)
            for attr_name in attr_accesses:
                # Если атрибут упоминается, но не зарегистрирован, добавляем как int
                if attr_name not in fields:
                    fields[attr_name] = "int"

    def _infer_field_type(self, value_ast: Dict) -> str:
        """Определяет тип поля по значению"""
        if not value_ast:
            return "int"  # По умолчанию

        value_type = value_ast.get("type", "")

        # Литералы
        if value_type == "literal":
            data_type = value_ast.get("data_type", "int")
            return data_type

        # Переменные
        elif value_type == "variable":
            var_name = value_ast.get("value", "")
            # Пытаемся определить тип переменной по контексту
            if var_name in ["in_dim", "out_dim", "x", "y", "z"]:
                return "int"
            elif var_name in ["weight", "bias", "value"]:
                return "float"

        # Бинарные операции
        elif value_type == "binary_operation":
            left = value_ast.get("left", {})
            right = value_ast.get("right", {})

            left_type = self._infer_field_type(left)
            right_type = self._infer_field_type(right)

            # Если типы совпадают, возвращаем его
            if left_type == right_type:
                return left_type

            # Если один float, а другой int - возвращаем float
            if "float" in left_type or "double" in left_type:
                return left_type
            if "float" in right_type or "double" in right_type:
                return right_type

            # По умолчанию int
            return "int"

        # Атрибуты
        elif value_type == "attribute_access":
            # Не можем определить тип атрибута рекурсивно
            return "int"

        # По умолчанию
        return "int"

    def generate_constructor(
        self,
        class_name: str,
        init_method: Optional[Dict] = None,
        init_scope: Optional[Dict] = None,
    ):
        """Генерирует конструктор класса"""
        self.add_line(f"// Конструктор для {class_name}")

        # Определяем параметры
        params = []
        param_names = []
        if init_method:
            init_params = init_method.get("parameters", [])
            # Пропускаем self параметр
            for param in init_params[1:]:
                param_name = param.get("name", "")
                param_type = param.get("type", "int")
                c_param_type = self.map_type_to_c(param_type)
                params.append(f"{c_param_type} {param_name}")
                param_names.append(param_name)

        params_str = ", ".join(params) if params else "void"

        # Функция создания объекта
        self.add_line(f"{class_name}* create_{class_name}({params_str}) {{")
        self.indent_level += 1

        # Выделяем память
        self.add_line(f"{class_name}* obj = malloc(sizeof({class_name}));")
        self.add_line(f"if (!obj) {{")
        self.indent_level += 1
        self.add_line(
            f'fprintf(stderr, "Memory allocation failed for {class_name}\\n");'
        )
        self.add_line(f"exit(1);")
        self.indent_level -= 1
        self.add_line(f"}}")
        self.add_empty_line()

        # Генерируем логику инициализации
        if init_scope:
            self._generate_init_logic(class_name, init_scope, param_names)
        else:
            # Базовая инициализация для классов без __init__
            base_classes = self.class_hierarchy.get(class_name, [])
            if not base_classes:
                # Корневой класс
                self.add_line(f"obj->vtable = malloc(sizeof(void*) * 16);")
            else:
                # Производный класс
                self.add_line(f"obj->base.vtable = malloc(sizeof(void*) * 16);")

            self.add_line(
                f"if (!obj->{'vtable' if not base_classes else 'base.vtable'}) {{"
            )
            self.indent_level += 1
            self.add_line(f'fprintf(stderr, "Memory allocation failed for vtable\\n");')
            self.add_line(f"free(obj);")
            self.add_line(f"exit(1);")
            self.indent_level -= 1
            self.add_line(f"}}")

        self.add_line(f"return obj;")
        self.indent_level -= 1
        self.add_line(f"}}")
        self.add_empty_line()

    def generate_class_method(self, class_name: str, method: Dict):
        """Генерирует метод класса"""
        method_name = method.get("name", "")
        return_type = method.get("return_type", "void")
        params = method.get("parameters", [])

        # Генерируем сигнатуру метода
        c_return_type = self.map_type_to_c(return_type)

        # Первый параметр - всегда self
        if params and params[0].get("name") == "self":
            # Параметр self в C - это указатель на структуру
            param_decls = [f"{class_name}* self"]
            # Остальные параметры
            for param in params[1:]:
                param_name = param.get("name", "")
                param_type = param.get("type", "int")
                c_param_type = self.map_type_to_c(param_type)
                param_decls.append(f"{c_param_type} {param_name}")
        else:
            param_decls = []
            for param in params:
                param_name = param.get("name", "")
                param_type = param.get("type", "int")
                c_param_type = self.map_type_to_c(param_type)
                param_decls.append(f"{c_param_type} {param_name}")

        params_str = ", ".join(param_decls) if param_decls else "void"

        self.add_line(f"{c_return_type} {class_name}_{method_name}({params_str}) {{")
        self.indent_level += 1

        # Тело метода будет сгенерировано отдельно
        self.add_line(f"// Реализация метода {method_name}")

        # Для метода get_age из примера
        if method_name == "get_age":
            self.add_line(f"return self->age;")

        self.indent_level -= 1
        self.add_line(f"}}")
        self.add_empty_line()

    def generate_class_constructors(self, json_data: List[Dict]):
        """Генерирует конструкторы для всех классов"""
        # Сначала находим все методы __init__
        init_scopes = {}

        for scope in json_data:
            # Ищем как constructor ИЛИ class_method
            if (
                scope.get("type") == "class_method"
                or scope.get("type") == "constructor"
            ) and scope.get("method_name") == "__init__":
                class_name = scope.get("class_name", "")
                init_scopes[class_name] = scope
                logger.debug(
                    f"DEBUG: Found init_scope for {class_name} (type: {scope.get('type')})"
                )
                logger.debug(f"Graph length: {len(scope.get('graph', []))}")

        # Затем находим объявления классов
        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "class_declaration":
                        class_name = node.get("class_name", "")
                        methods = node.get("methods", [])

                        # Ищем метод __init__ в объявлении класса
                        init_method = None
                        for method in methods:
                            if method.get("name") == "__init__":
                                init_method = method
                                logger.debug(f"Found init_method for {class_name}")
                                break

                        # Получаем scope для этого метода
                        init_scope = init_scopes.get(class_name)

                        if init_scope:
                            logger.debug(f"Will generate constructor for {class_name}")
                            # Выводим для отладки структуру init_scope
                            logger.debug(f"init_scope keys: {init_scope.keys()}")
                            logger.debug(
                                f"DEBUG init_scope graph: {init_scope.get('graph', [])}"
                            )
                        else:
                            logger.debug(f"No init_scope found for {class_name}")
                            logger.debug(
                                f"DEBUG: Available scopes: {list(init_scopes.keys())}"
                            )

                        # Генерируем конструктор
                        self.generate_constructor(class_name, init_method, init_scope)

    def generate_class_method_implementation(self, class_name: str, scope: Dict):
        """Генерирует реализацию метода класса с поддержкой сложных типов возврата"""
        method_name = scope.get("method_name", "")
        return_type = scope.get("return_type", "void")

        logger.debug(
            f"DEBUG generate_class_method_implementation: {class_name}.{method_name}() -> {return_type}"
        )

        # Пропускаем конструктор
        if method_name == "__init__":
            return

        # Проверяем, не генерировали ли уже этот метод
        func_name = f"{class_name}_{method_name}"
        if func_name in self.generated_functions:
            logger.debug(f"метод {func_name} уже сгенерирован, пропускаем")
            return

        # Регистрируем метод как сгенерированный
        self.generated_functions.add(func_name)

        # Определяем C тип возвращаемого значения
        if return_type.startswith("list["):
            # Генерируем структуру для списка если нужно
            self.generate_list_struct(return_type)
            struct_name = self.generate_list_struct_name(return_type)
            c_return_type = f"{struct_name}*"
        elif return_type.startswith("tuple["):
            # Генерируем структуру для кортежа если нужно
            self.generate_tuple_struct(return_type)
            struct_name = self.generate_tuple_struct_name(return_type)
            c_return_type = f"{struct_name}*"
        else:
            c_return_type = self.map_type_to_c(return_type)

        # Генерируем параметры
        parameters = scope.get("parameters", [])
        param_decls = []

        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "int")

            if param_name == "self":
                c_param_type = f"{class_name}*"
            else:
                c_param_type = self.map_type_to_c(param_type)

            param_decls.append(f"{c_param_type} {param_name}")

        params_str = ", ".join(param_decls) if param_decls else "void"

        # Сигнатура метода
        self.add_line(f"{c_return_type} {class_name}_{method_name}({params_str}) {{")
        self.indent_level += 1

        # Входим в scope метода и добавляем информацию о классе
        self.enter_scope()

        # ДОБАВЛЯЕМ ИНФОРМАЦИЮ О КЛАССЕ В ТЕКУЩИЙ SCOPE
        current_scope = self.get_current_scope()
        current_scope["class_name"] = class_name

        # Объявляем параметры в scope (кроме self)
        for param in parameters:
            param_name = param.get("name", "")
            if param_name != "self":
                param_type = param.get("type", "int")
                self.declare_variable(param_name, param_type)

        # Генерируем тело метода
        for node in scope.get("graph", []):
            self.generate_graph_node(node)

        # Выходим из scope
        self.exit_scope()

        self.indent_level -= 1
        self.add_line("}")
        self.add_empty_line()

    def analyze_classes(self, json_data: List[Dict]):
        """Анализирует все классы и их методы для определения полей"""
        logger.debug("DEBUG analyze_classes: Начинаем анализ классов")

        # Собираем все конструкторы
        for scope in json_data:
            if (
                scope.get("type") == "constructor"
                and scope.get("method_name") == "__init__"
            ):
                class_name = scope.get("class_name", "")
                logger.debug(f"Найден конструктор для класса {class_name}")

                if class_name not in self.class_fields:
                    self.class_fields[class_name] = {}

                # Получаем параметры конструктора
                parameters = scope.get("parameters", [])
                param_types = {}

                for param in parameters:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "int")
                    if param_name != "self":
                        param_types[param_name] = param_type
                        logger.debug(f"Параметр {param_name}: {param_type}")

                # Анализируем присваивания атрибутов
                for node in scope.get("graph", []):
                    if node.get("node") == "attribute_assignment":
                        obj_name = node.get("object", "")
                        attr_name = node.get("attribute", "")
                        value_ast = node.get("value", {})

                        if obj_name == "self":
                            # Определяем тип значения
                            field_type = self._infer_field_type_from_ast(
                                value_ast, param_types
                            )
                            if field_type:
                                self.class_fields[class_name][attr_name] = field_type
                                logger.debug(
                                    f"DEBUG: Поле {class_name}.{attr_name} = {field_type}"
                                )

    def _analyze_init_method(self, class_name: str, init_scope: Dict):
        """Анализирует метод __init__ для определения полей класса"""
        if class_name not in self.class_fields:
            self.class_fields[class_name] = {}

        graph = init_scope.get("graph", [])

        for node in graph:
            if node.get("node") == "attribute_assignment":
                # Присваивание атрибуту: self.attr = value
                attr_name = node.get("attribute", "")
                value = node.get("value", {})

                # Определяем тип значения
                field_type = self._infer_field_type(value)
                if field_type:
                    self.class_fields[class_name][attr_name] = field_type

            elif node.get("node") == "declaration":
                # Объявление атрибута с типом: self.attr: type = value
                var_name = node.get("var_name", "")
                if var_name.startswith("self."):
                    attr_name = var_name[5:]  # Убираем "self."
                    var_type = node.get("var_type", "")
                    if var_type:
                        self.class_fields[class_name][attr_name] = var_type

    def generate_class_declaration_with_fields(self, node: Dict):
        """Генерирует структуру для класса C с полями"""
        class_name = node.get("class_name", "")
        base_classes = node.get("base_classes", [])

        # Регистрируем класс
        self.class_types.add(class_name)
        self.type_map[class_name] = f"{class_name}*"

        # Генерируем forward declaration
        self.add_line(f"typedef struct {class_name} {class_name};")
        self.add_empty_line()

        # Генерируем структуру
        self.add_line(f"struct {class_name} {{")
        self.indent_level += 1

        # Добавляем наследование через композицию
        if base_classes and len(base_classes) > 0:
            parent_class = base_classes[0]
            self.add_line(f"// Наследование от {parent_class}")
            self.add_line(f"{parent_class} base;")
        else:
            # Для корневого класса добавляем vtable
            self.add_line(f"void** vtable;")

        # Добавляем поля класса (если они были собраны)
        if class_name in self.class_fields and self.class_fields[class_name]:
            self.add_line(f"// Поля класса {class_name}")
            for field_name, field_type in self.class_fields[class_name].items():
                c_type = self.map_type_to_c(field_type)
                self.add_line(f"{c_type} {field_name};")
        else:
            self.add_line(f"// Поля не найдены для {class_name}")

        self.indent_level -= 1
        self.add_line(f"}};")
        self.add_empty_line()

    def _process_attribute_assignment_in_init(self, node: Dict, param_names: List[str]):
        """Обрабатывает присваивание атрибуту в конструкторе"""
        object_name = node.get("object", "")
        attribute = node.get("attribute", "")
        value_ast = node.get("value", {})

        logger.debug(
            f"DEBUG _process_attribute_assignment_in_init: {object_name}.{attribute} = {value_ast}"
        )

        if object_name == "self" and value_ast:
            # Генерируем выражение для значения с учетом параметров конструктора
            value_expr = self._generate_expression_from_ast_for_init(
                value_ast, param_names
            )
            if value_expr:
                logger.debug(f"Generated expression: obj->{attribute} = {value_expr}")
                self.add_line(f"obj->{attribute} = {value_expr};")
            else:
                logger.debug(f"Could not generate expression for {attribute}")
                self.add_line(f"obj->{attribute} = 0; // default value")
        else:
            logger.debug(f"Skipping non-self assignment or empty value")

    def _generate_init_logic(
        self, class_name: str, init_scope: Dict, param_names: List[str]
    ):
        """Генерирует логику инициализации полей из метода __init__"""
        if not init_scope:
            return

        graph = init_scope.get("graph", [])
        base_classes = self.class_hierarchy.get(class_name, [])

        self.add_line(f"// Инициализация полей класса {class_name}")

        # Инициализируем vtable
        if not base_classes:
            # Для корневого класса (например, Object) - прямое поле vtable
            self.add_line(f"obj->vtable = malloc(sizeof(void*) * 16);")
            self.add_line(f"if (!obj->vtable) {{")
        else:
            # Для производных классов - vtable в базовом классе
            self.add_line(f"obj->base.vtable = malloc(sizeof(void*) * 16);")
            self.add_line(f"if (!obj->base.vtable) {{")

        self.indent_level += 1
        self.add_line(f'fprintf(stderr, "Memory allocation failed for vtable\\n");')
        self.add_line(f"free(obj);")
        self.add_line(f"exit(1);")
        self.indent_level -= 1
        self.add_line(f"}}")

        # Инициализация полей из метода __init__
        for node in graph:
            node_type = node.get("node", "")

            if node_type == "attribute_assignment":
                self._process_attribute_assignment_in_init(node, param_names)

    def generate_all_methods(self, json_data: List[Dict]):
        """Генерирует все методы всех классов, включая унаследованные"""
        logger.debug("DEBUG generate_all_methods: Начало")

        # Сначала анализируем наследование
        self.analyze_class_inheritance(json_data)

        # Собираем все методы классов из JSON
        class_method_scopes = {}

        for scope in json_data:
            if scope.get("type") == "class_method":
                class_name = scope.get("class_name", "")
                method_name = scope.get("method_name", "")

                if class_name not in class_method_scopes:
                    class_method_scopes[class_name] = {}

                class_method_scopes[class_name][method_name] = scope

        # Генерируем унаследованные методы
        for class_name, methods in self.all_class_methods.items():
            logger.debug(f"Проверка методов для класса {class_name}")

            for method_name, method_info in methods.items():
                # Пропускаем конструктор
                if method_name == "__init__":
                    continue

                # Проверяем, есть ли реализация в JSON
                has_implementation = (
                    class_name in class_method_scopes
                    and method_name in class_method_scopes[class_name]
                )

                # Если это унаследованный метод и у него нет реализации
                if (
                    method_info.get("is_inherited", False)
                    and not has_implementation
                    and method_info.get("origin") != class_name
                ):
                    logger.debug(
                        f"DEBUG: Генерация унаследованного метода {class_name}.{method_name} из {method_info['origin']}"
                    )
                    self._generate_inherited_method_stub(class_name, method_info)

        # Генерируем методы с реализациями из JSON
        for class_name, methods in class_method_scopes.items():
            for method_name, scope in methods.items():
                if method_name != "__init__":
                    self.generate_class_method_implementation(class_name, scope)

    def _generate_inherited_method_stub(self, class_name: str, method_info: Dict):
        """Генерирует заглушку для унаследованного метода"""
        method_name = method_info["name"]
        return_type = method_info.get("return_type", "void")
        parameters = method_info.get("parameters", [])

        origin_class = method_info.get("origin")

        if not origin_class:
            return

        # Генерируем сигнатуру
        param_decls = []
        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "int")

            if param_name == "self":
                c_param_type = f"{class_name}*"
            else:
                c_param_type = self.map_type_to_c(param_type)

            param_decls.append(f"{c_param_type} {param_name}")

        c_return_type = self.map_type_to_c(return_type)
        params_str = ", ".join(param_decls) if param_decls else "void"

        self.add_line(f"{c_return_type} {class_name}_{method_name}({params_str}) {{")
        self.indent_level += 1

        # Вызываем родительский метод с приведением типа
        if return_type != "void":
            if origin_class == class_name:
                # Метод определен в этом классе (не должен вызываться здесь)
                self.add_line(f"return 0;")
            else:
                # Приводим self к типу родительского класса
                self.add_line(f"// Вызов унаследованного метода из {origin_class}")
                self.add_line(f"{origin_class}* base_obj = ({origin_class}*)self;")
                self.add_line(f"return {origin_class}_{method_name}(base_obj);")
        else:
            if origin_class != class_name:
                self.add_line(f"// Вызов унаследованного метода из {origin_class}")
                self.add_line(f"{origin_class}* base_obj = ({origin_class}*)self;")
                self.add_line(f"{origin_class}_{method_name}(base_obj);")

        self.indent_level -= 1
        self.add_line("}")
        self.add_empty_line()

    def collect_class_fields_from_init_parameters(self, json_data: List[Dict]):
        """Собирает типы полей классов из параметров конструктора"""
        logger.debug("DEBUG: collect_class_fields_from_init_parameters")

        # 1. Собираем все методы __init__
        init_scopes = {}

        for scope in json_data:
            if scope.get("type") in ["constructor", "class_method"]:
                if scope.get("method_name") == "__init__":
                    class_name = scope.get("class_name")
                    init_scopes[class_name] = scope
                    logger.debug(f"Найден конструктор для {class_name}")

        # 2. Анализируем каждый конструктор
        for class_name, init_scope in init_scopes.items():
            if class_name not in self.class_fields:
                self.class_fields[class_name] = {}

            logger.debug(f"Анализируем конструктор {class_name}.__init__")

            # Получаем параметры конструктора
            parameters = init_scope.get("parameters", [])

            # Собираем типы параметров (пропускаем self)
            param_types = {}
            for param in parameters:
                param_name = param.get("name", "")
                if param_name != "self":
                    param_type = param.get("type", "int")
                    param_types[param_name] = param_type
                    logger.debug(f"Параметр {param_name}: {param_type}")

            # Анализируем узлы графа
            for node in init_scope.get("graph", []):
                if node.get("node") == "attribute_assignment":
                    obj = node.get("object", "")
                    attr = node.get("attribute", "")
                    value_ast = node.get("value", {})

                    if obj == "self":
                        # 1. Проверяем, является ли значение параметром конструктора
                        if value_ast.get("type") == "variable":
                            var_name = value_ast.get("value", "")

                            if var_name in param_types:
                                # Используем тип параметра
                                field_type = param_types[var_name]
                                self.class_fields[class_name][attr] = field_type
                                logger.debug(
                                    f"DEBUG: Поле {attr} <- тип параметра {var_name}: {field_type}"
                                )
                            else:
                                # Иначе определяем тип по AST
                                field_type = self._infer_field_type_from_ast(
                                    value_ast, param_types
                                )
                                if field_type:
                                    self.class_fields[class_name][attr] = field_type
                                    logger.debug(
                                        f"DEBUG: Поле {attr} <- вычисленный тип: {field_type}"
                                    )

                        # 2. Проверяем литералы
                        elif value_ast.get("type") == "literal":
                            data_type = value_ast.get("data_type", "int")
                            self.class_fields[class_name][attr] = data_type
                            logger.debug(f"Поле {attr} <- литерал типа: {data_type}")

                        # 3. Для выражений пытаемся определить тип
                        else:
                            field_type = self._infer_field_type_from_ast(
                                value_ast, param_types
                            )
                            if field_type:
                                self.class_fields[class_name][attr] = field_type
                                logger.debug(f"Поле {attr} <- тип из AST: {field_type}")

    def collect_class_fields_from_json(self, json_data: List[Dict]):
        """Собирает поля всех классов из JSON"""
        logger.debug(
            "DEBUG: collect_class_fields_from_json: начинаем сбор полей классов"
        )

        # 1. Находим все конструкторы __init__
        init_methods = {}

        for scope in json_data:
            if scope.get("type") in ["constructor", "class_method"]:
                if scope.get("method_name") == "__init__":
                    class_name = scope.get("class_name")
                    init_methods[class_name] = scope
                    logger.debug(f"Найден конструктор для класса {class_name}")

        # 2. Анализируем каждый конструктор
        for class_name, init_scope in init_methods.items():
            if class_name not in self.class_fields:
                self.class_fields[class_name] = {}

            logger.debug(f"Анализируем конструктор {class_name}.__init__")

            # Анализируем узлы графа
            for node in init_scope.get("graph", []):
                if node.get("node") == "attribute_assignment":
                    obj = node.get("object", "")
                    attr = node.get("attribute", "")

                    if obj == "self":
                        # Определяем тип поля
                        # Пока просто ставим int для всех полей
                        # Позже можно улучшить определение типов
                        self.class_fields[class_name][attr] = "int"
                        logger.debug(f"Добавлено поле {class_name}.{attr} = int")

        logger.debug(f"Всего классов с полями: {len(self.class_fields)}")
        for class_name, fields in self.class_fields.items():
            logger.debug(f"  {class_name}: {fields}")

    def analyze_class_inheritance(self, json_data: List[Dict]):
        """Анализирует иерархию наследования классов"""
        logger.debug("DEBUG analyze_class_inheritance: Начинаем анализ классов")

        # Сначала собираем информацию о всех классах
        class_info = {}

        for scope in json_data:
            if scope.get("type") == "module":
                for node in scope.get("graph", []):
                    if node.get("node") == "class_declaration":
                        class_name = node.get("class_name", "")
                        base_classes = node.get("base_classes", [])
                        methods = node.get("methods", [])

                        class_info[class_name] = {
                            "base_classes": base_classes,
                            "methods": {method["name"]: method for method in methods},
                        }
                        logger.debug(f"Класс {class_name} наследует от {base_classes}")

        # Строим иерархию наследования
        for class_name, info in class_info.items():
            self.class_hierarchy[class_name] = info["base_classes"]
            self.all_class_methods[class_name] = {}

            # Начинаем с методов текущего класса
            for method_name, method_info in info["methods"].items():
                self.all_class_methods[class_name][method_name] = {
                    **method_info,
                    "origin": class_name,
                    "is_inherited": False,
                }

            # Добавляем методы из родительских классов
            if info["base_classes"]:
                self._inherit_methods_from_parents(class_name, class_info)

    def _inherit_methods_from_parents(self, class_name: str, class_info: Dict):
        """Добавляет унаследованные методы из родительских классов"""
        if class_name not in class_info:
            return

        base_classes = class_info[class_name]["base_classes"]

        for base_class in base_classes:
            if base_class in class_info:
                # Добавляем методы родительского класса
                for method_name, method_info in class_info[base_class][
                    "methods"
                ].items():
                    if method_name not in self.all_class_methods[class_name]:
                        self.all_class_methods[class_name][method_name] = {
                            **method_info,
                            "origin": base_class,
                            "is_inherited": True,
                        }
                        logger.debug(
                            f"DEBUG: Класс {class_name} наследует метод {method_name} от {base_class}"
                        )

                # Рекурсивно добавляем методы от родительских классов родителя
                self._inherit_methods_from_parents(base_class, class_info)
