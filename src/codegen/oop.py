from __future__ import annotations

from typing import Dict, List, Optional

from src.modules.logger import logger
from src.codegen.class_model import ClassRegistry, MethodModel, build_class_registry

class OopMixin:
    def build_class_registry(self, scopes: List[Dict]) -> ClassRegistry:
        """Build the canonical OOP metadata directly from parser output."""
        self.class_registry = build_class_registry(scopes)
        return self.class_registry

    def generate_constructor(
        self,
        class_name: str,
        init_method: Optional[MethodModel] = None,
        init_scope: Optional[Dict] = None,
    ):
        """Generate an ARC-owned zero-initialized class instance."""
        params = []
        param_names = []
        if init_method:
            for param in init_method.parameters[1:]:
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

    def generate_class_constructors(self, scopes: List[Dict]):
        """Generate constructors from the canonical class models."""
        for class_name, model in self.class_registry.models.items():
            init_model = model.direct_method("__init__")
            init_scope = init_model.scope if init_model else None
            self.generate_constructor(class_name, init_model, init_scope)

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

    def generate_class_declaration_with_fields(self, node: Dict):
        """Generate an ARC-compatible class layout with safe single inheritance."""
        class_name = node.get("class_name", "")
        model = self.class_registry.get(class_name)
        base_classes = model.bases if model else node.get("base_classes", [])
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

        fields = model.fields.values() if model else ()
        for field in fields:
            self.add_line(f"{self.map_type_to_c(field.py_type)} {field.name};")
        self.indent_level -= 1
        self.add_line("};")
        self.add_empty_line()

    def _class_root_path(self, class_name: str, obj_expr: str) -> str:
        """Return expression addressing the root base subobject at offset zero."""
        expr = obj_expr
        current = class_name
        seen = set()
        while self.class_registry.bases_for(current):
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            parent = self.class_registry.bases_for(current)[0]
            expr += ".base" if not expr.endswith("->") else "base"
            current = parent
        return expr

    def _class_header_lvalue(self, class_name: str, obj_name: str = "obj") -> str:
        current = class_name
        path = obj_name
        seen = set()
        while self.class_registry.bases_for(current):
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            path += "->base" if path == obj_name else ".base"
            current = self.class_registry.bases_for(current)[0]
        return f"{path}.header" if path != obj_name else f"{obj_name}->header"

    def _class_vtable_lvalue(self, class_name: str, obj_name: str = "obj") -> str:
        current = class_name
        path = obj_name
        seen = set()
        while self.class_registry.bases_for(current):
            if current in seen:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            seen.add(current)
            path += "->base" if path == obj_name else ".base"
            current = self.class_registry.bases_for(current)[0]
        return f"{path}.vtable" if path != obj_name else f"{obj_name}->vtable"

    def _iter_inherited_fields(self, class_name: str):
        """Yield (origin_class, field_name, field_type) from root to leaf."""
        model = self.class_registry.get(class_name)
        if not model:
            return
        for origin, field in self.class_registry.inherited_fields(class_name):
            yield origin, field.name, field.py_type

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
            elif field_type == "ocean_tensor_handle_t":
                # The standard Tensor facade stores an opaque C handle.  It is
                # not a raw pointer in Ocean source, but its destructor still
                # has an explicit runtime release operation.
                self.add_line(f"ocean_tensor_release({access});")
            elif field_type == "ocean_file_handle_t":
                self.add_line(f"ocean_file_close({access});")
            elif field_type == "ocean_thread_handle_t":
                self.add_line(f"ocean_thread_release({access});")
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
        field_model = self.class_registry.field(class_name, attribute) if class_name else None
        field_type = field_model.py_type if field_model else None
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

    def generate_all_methods(self, scopes: List[Dict]):
        """Генерирует все методы всех классов, включая унаследованные"""
        logger.debug("DEBUG generate_all_methods: Начало")

        # Сначала анализируем наследование
        self.analyze_class_inheritance(scopes)

        # Собираем реализации из canonical MethodModel scopes.
        class_method_scopes = {}
        for class_name, model in self.class_registry.models.items():
            for method_name, method in model.methods.items():
                if method.scope and method_name != "__init__":
                    class_method_scopes.setdefault(class_name, {})[method_name] = method.scope

        # Генерируем унаследованные методы
        for class_name, methods in self.class_registry.resolved_methods().items():
            logger.debug(f"Проверка методов для класса {class_name}")

            for method_name, method_info in methods.items():
                # Пропускаем конструктор
                if method_name == "__init__":
                    continue

                # Проверяем, есть ли реализация в typed AST
                has_implementation = (
                    class_name in class_method_scopes
                    and method_name in class_method_scopes[class_name]
                )

                # Если это унаследованный метод и у него нет реализации
                if method_info.inherited and not has_implementation and method_info.origin != class_name:
                    logger.debug(
                        f"DEBUG: Генерация унаследованного метода {class_name}.{method_name} из {method_info.origin}"
                    )
                    self._generate_inherited_method_stub(class_name, method_info)

        # Генерируем методы с реализациями из typed AST
        for class_name, methods in class_method_scopes.items():
            for method_name, scope in methods.items():
                if method_name != "__init__":
                    self.generate_class_method_implementation(class_name, scope)

    def _generate_inherited_method_stub(self, class_name: str, method_info):
        """Генерирует заглушку для унаследованного метода"""
        method = method_info.method
        method_name = method.name
        return_type = method.return_type
        parameters = method.parameters
        origin_class = method_info.origin

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

    def analyze_class_inheritance(self, scopes: List[Dict]):
        """Build method resolution metadata from canonical class models."""
        logger.debug("Class inheritance is resolved by ClassRegistry")
        self.class_registry.resolved_methods()
        return
