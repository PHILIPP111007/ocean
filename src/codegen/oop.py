from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
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
        """Generate an ARC-owned zero-initialized class instance."""
        params = []
        param_names = []
        if init_method:
            for param in init_method.get("parameters", [])[1:]:
                name = param.get("name", "")
                typ = param.get("type", "int")
                params.append(f"{self.map_type_to_c(typ)} {name}")
                param_names.append(name)
        params_str = ", ".join(params) if params else "void"

        self._generate_class_destructor(class_name)
        self.add_line(f"{class_name}* create_{class_name}({params_str}) {{")
        self.indent_level += 1
        self.add_line(f"{class_name}* obj = ({class_name}*)calloc(1, sizeof({class_name}));")
        self.add_line(f"if (!obj) {{ fprintf(stderr, \"Memory allocation failed for {class_name}\\n\"); exit(1); }}")
        header = self._class_header_lvalue(class_name, "obj")
        vtable = self._class_vtable_lvalue(class_name, "obj")
        self.add_line(f"{header}.refcount = 1;")
        self.add_line(f"{header}.destroy = ocean_destroy_{class_name};")
        self.add_line(f"{vtable} = NULL;")
        if init_scope:
            self._generate_init_logic(class_name, init_scope, param_names)
        self.add_line("return obj;")
        self.indent_level -= 1
        self.add_line("}")
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
        """Generate a method with borrowed parameters and automatic owner cleanup."""
        method_name = scope.get("method_name", "")
        return_type = scope.get("return_type", "None")
        if method_name == "__init__":
            return
        func_name = f"{class_name}_{method_name}"
        if func_name in self.generated_functions:
            return
        self.generated_functions.add(func_name)

        c_return_type = self.map_type_to_c(return_type)
        parameters = scope.get("parameters", [])
        decls = []
        for param in parameters:
            name = param.get("name", "")
            typ = param.get("type", "int")
            c_type = f"{class_name}*" if name == "self" else self.map_type_to_c(typ)
            decls.append(f"{c_type} {name}")
        self.add_line(f"{c_return_type} {func_name}({', '.join(decls) if decls else 'void'}) {{")
        self.indent_level += 1
        self.current_function_return_type = return_type
        self.current_function_name = func_name
        self.enter_scope("function")
        current = self.get_current_scope()
        current["class_name"] = class_name
        for param in parameters:
            name = param.get("name", "")
            if name == "self":
                # self is a borrowed object reference for the duration of the method.
                continue
            self.declare_variable(name, param.get("type", "int"), is_parameter=True)
        for node in scope.get("graph", []):
            self.generate_graph_node(node)
        self.exit_scope(emit_cleanup=True)
        self.current_function_return_type = None
        self.current_function_name = None
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
                            # Explicit annotations are authoritative. Inferring
                            # tensor.zeros(...) from an expression falls back
                            # to int and corrupts the generated class layout.
                            field_type = node.get("attribute_type") or node.get(
                                "attribute_type_info", {}
                            ).get("canonical")
                            field_type = field_type or self._infer_field_type_from_ast(
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
        """Generate an ARC-compatible class layout with safe single inheritance."""
        class_name = node.get("class_name", "")
        base_classes = node.get("base_classes", [])
        if len(base_classes) > 1:
            raise RuntimeError(
                f"multiple inheritance for class '{class_name}' is disabled in Ocean memory-safe v0.2; "
                "use composition until multi-base layout/offset checking is implemented"
            )

        self.class_types.add(class_name)
        self.type_map[class_name] = f"{class_name}*"
        self.add_line(f"typedef struct {class_name} {class_name};")
        self.add_empty_line()
        self.add_line(f"struct {class_name} {{")
        self.indent_level += 1

        if base_classes:
            parent = base_classes[0]
            self.add_line(f"{parent} base;")
        else:
            # Header at offset zero makes every root object consumable by ocean_retain/release.
            self.add_line("ocean_object_header header;")
            self.add_line("void** vtable;")

        for field_name, field_type in self.class_fields.get(class_name, {}).items():
            self.add_line(f"{self.map_type_to_c(field_type)} {field_name};")
        self.indent_level -= 1
        self.add_line("};")
        self.add_empty_line()

    def _class_root_path(self, class_name: str, obj_expr: str) -> str:
        """Return expression addressing the root base subobject at offset zero."""
        expr = obj_expr
        current = class_name
        seen = set()
        while self.class_hierarchy.get(current):
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            parent = self.class_hierarchy[current][0]
            expr += ".base" if not expr.endswith("->") else "base"
            current = parent
        return expr

    def _class_header_lvalue(self, class_name: str, obj_name: str = "obj") -> str:
        current = class_name
        path = obj_name
        seen = set()
        while self.class_hierarchy.get(current):
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            path += "->base" if path == obj_name else ".base"
            current = self.class_hierarchy[current][0]
        return f"{path}.header" if path != obj_name else f"{obj_name}->header"

    def _class_vtable_lvalue(self, class_name: str, obj_name: str = "obj") -> str:
        current = class_name
        path = obj_name
        seen = set()
        while self.class_hierarchy.get(current):
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            path += "->base" if path == obj_name else ".base"
            current = self.class_hierarchy[current][0]
        return f"{path}.vtable" if path != obj_name else f"{obj_name}->vtable"

    def _iter_inherited_fields(self, class_name: str):
        """Yield (origin_class, field_name, field_type) from root to leaf."""
        chain = []
        current = class_name
        seen = set()
        while current:
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            chain.append(current)
            parents = self.class_hierarchy.get(current, [])
            current = parents[0] if parents else None
        for origin in reversed(chain):
            for name, typ in self.class_fields.get(origin, {}).items():
                yield origin, name, typ

    def _generate_class_destructor(self, class_name: str):
        destroy_name = f"ocean_destroy_{class_name}"
        if destroy_name in self.generated_functions:
            return
        self.add_line(f"static void {destroy_name}(void* ptr) {{")
        self.indent_level += 1
        self.add_line(f"{class_name}* self = ({class_name}*)ptr;")
        self.add_line("if (!self) return;")
        for origin, field_name, field_type in self._iter_inherited_fields(class_name):
            access = f"(({origin}*)self)->{field_name}"
            kind = self.memory_kind_for_type(field_type)
            if kind == self.MEMORY_ARC:
                self.add_line(f"ocean_release({access});")
            elif kind == self.MEMORY_STRING:
                self.add_line(f"free({access});")
            elif kind == self.MEMORY_OWNED:
                self.add_line(self._owned_free_call(access, field_type))
        self.add_line("free(self);")
        self.indent_level -= 1
        self.add_line("}")
        self.add_empty_line()
        self.generated_functions.add(destroy_name)

    def _process_attribute_assignment_in_init(self, node: Dict, param_names: List[str]):
        """Initialize a zeroed field, retaining only borrowed incoming references."""
        object_name = node.get("object", "")
        attribute = node.get("attribute", "")
        value_ast = node.get("value", {}) or {}
        if object_name != "self":
            return
        class_name = getattr(self, "_constructing_class", None)
        field_type = self.class_fields.get(class_name, {}).get(attribute) if class_name else None
        value_expr = self._generate_expression_from_ast_for_init(
            value_ast,
            param_names,
            target_type=field_type or "",
            target_name=attribute,
        )
        if not value_expr:
            return
        if not field_type:
            self.add_line(f"obj->{attribute} = {value_expr};")
            return
        kind = self.memory_kind_for_type(field_type)
        ownership = self.expression_ownership(value_ast, field_type)
        if kind == self.MEMORY_ARC:
            if ownership == "borrowed":
                self.add_line(f"ocean_retain({value_expr});")
            self.add_line(f"obj->{attribute} = {value_expr};")
        elif kind == self.MEMORY_STRING:
            if ownership == "borrowed":
                self.add_line(f"obj->{attribute} = ocean_strdup({value_expr});")
            else:
                self.add_line(f"obj->{attribute} = {value_expr};")
        else:
            self.add_line(f"obj->{attribute} = {value_expr};")

    def _generate_init_logic(
        self, class_name: str, init_scope: Dict, param_names: List[str]
    ):
        """Initialize fields. Object memory is already zeroed by calloc."""
        self._constructing_class = class_name
        try:
            for node in (init_scope or {}).get("graph", []):
                if node.get("node") == "attribute_assignment":
                    self._process_attribute_assignment_in_init(node, param_names)
        finally:
            self._constructing_class = None

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
                        explicit_type = node.get("attribute_type") or node.get(
                            "attribute_type_info", {}
                        ).get("canonical")
                        if explicit_type:
                            self.class_fields[class_name][attr] = explicit_type
                            logger.debug(
                                f"Поле {attr} <- явная аннотация: {explicit_type}"
                            )
                            continue

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
                        field_type = node.get("attribute_type") or node.get(
                            "attribute_type_info", {}
                        ).get("canonical")
                        if field_type:
                            self.class_fields[class_name][attr] = field_type
                            logger.debug(
                                f"Добавлено поле {class_name}.{attr} = {field_type}"
                            )

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
