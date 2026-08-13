from __future__ import annotations

from typing import Dict, Optional

from src.parsing.type_system import TENSOR_DTYPES, TypeParser


class ArrayCodegenMixin:
    """Lower uniquely-owned one-dimensional ``array[T]`` values to C."""

    def is_array_type(self, py_type: str) -> bool:
        return self.strip_borrow_type(py_type).startswith("array[")

    def is_device_tensor_type(self, py_type: str) -> bool:
        base = self.strip_borrow_type(py_type)
        return base == "Tensor" or (
            base.startswith("Tensor[") and base.endswith("]")
        )

    def device_tensor_dtype(self, py_type: str) -> str:
        base = self.strip_borrow_type(py_type)
        if base == "Tensor":
            return "float32"
        dtype = base[len("Tensor[") : -1].strip()
        if dtype not in TENSOR_DTYPES:
            raise RuntimeError(
                f"Tensor dtype '{dtype}' is not numeric or is not supported"
            )
        return dtype

    def is_owned_buffer_type(self, py_type: str) -> bool:
        return self.is_array_type(py_type)

    def _generic_element_type(self, py_type: str, generic_name: str) -> str:
        base = self.strip_borrow_type(py_type)
        spec = TypeParser().parse(base)
        if spec.kind != "generic" or spec.name != generic_name or not spec.args:
            raise RuntimeError(f"invalid {generic_name} type: {py_type}")
        return spec.args[0].canonical

    def array_element_type(self, py_type: str) -> str:
        return self._generic_element_type(py_type, "array")

    def array_struct_name(self, py_type: str) -> str:
        return f"ocean_array_{self.clean_type_name_for_c(self.array_element_type(py_type))}"

    def array_c_type(self, py_type: str) -> str:
        return f"{self.array_struct_name(py_type)}*"

    def _owned_free_call(self, name: str, py_type: str) -> str:
        if self.is_array_type(py_type):
            return f"{self.array_struct_name(py_type)}_free({name});"
        raise RuntimeError(f"not an owned buffer type: {py_type}")

    def _mark_owned_move(self, source: str) -> None:
        info = self.get_variable_info(source)
        if info is not None:
            info["is_moved"] = True
            info["owns_reference"] = False

    def _owned_source(self, expression_ast: Optional[Dict], target_type: str) -> Optional[str]:
        if not expression_ast or expression_ast.get("type") != "variable":
            return None
        source = expression_ast.get("value") or expression_ast.get("name")
        info = self.get_variable_info(source)
        if not info:
            return None
        source_type = self.strip_borrow_type(info.get("py_type", ""))
        target = self.strip_borrow_type(target_type)
        if source_type != target:
            return None
        if info.get("memory_kind") in {self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
            raise RuntimeError(
                f"cannot move owned value from borrow '{source}'; use an explicit copy"
            )
        return source

    def _generate_owned_assignment(self, target: str, target_type: str, expression_ast: Dict) -> None:
        info = self.get_variable_info(target)
        if not info:
            raise RuntimeError(f"unknown owned target '{target}'")
        self.assert_can_move_or_delete(target)

        source = self._owned_source(expression_ast, target_type)
        if source:
            if source == target:
                return
            self.assert_can_read(source)
            source_info = self.get_variable_info(source)
            self.assert_can_move_or_delete(source)
            self.add_line(self._owned_free_call(target, target_type))
            self.add_line(f"{target} = {source};")
            self.add_line(f"{source} = NULL;")
            self._mark_owned_move(source)
            info["is_deleted"] = False
            info["is_moved"] = False
            info["owns_reference"] = True
            return

        expr = (
            self._generate_owned_literal_expr(target, target_type, expression_ast)
            if expression_ast.get("type") == "list_literal"
            else self.generate_expression(expression_ast)
        )
        if self._is_none_expression(expression_ast):
            expr = "NULL"
        self.add_line(self._owned_free_call(target, target_type))
        self.add_line(f"{target} = {expr};")
        info["is_deleted"] = False
        info["is_moved"] = False
        info["owns_reference"] = True

    def generate_array_declaration(self, node: Dict) -> None:
        var_name = node.get("var_name", "")
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast", {}) or {}
        self.declare_variable(var_name, var_type)
        info = self.get_variable_info(var_name)
        if not info:
            return

        if expression_ast.get("type") == "list_literal":
            expr = self._generate_array_literal_expr(var_name, var_type, expression_ast)
        elif self._is_none_expression(expression_ast):
            expr = "NULL"
        else:
            source = self._owned_source(expression_ast, var_type)
            if source:
                self.assert_can_read(source)
                self.assert_can_move_or_delete(source)
                expr = source
                self.add_line(f"{self.array_c_type(var_type)} {var_name} = {expr};")
                self.add_line(f"{source} = NULL;")
                self._mark_owned_move(source)
                return
            expr = self.generate_expression(expression_ast)

        self.add_line(f"{self.array_c_type(var_type)} {var_name} = {expr};")

    def generate_array_assignment(self, node: Dict) -> None:
        self._generate_owned_assignment(
            node.get("symbols", [""])[0],
            node.get("var_type") or self.get_variable_info(node.get("symbols", [""])[0]).get("py_type"),
            node.get("expression_ast") or {},
        )

    def _generate_array_literal_expr(self, var_name: str, py_type: str, ast: Dict) -> str:
        element_type = self.array_element_type(py_type)
        if self.is_array_type(element_type):
            raise RuntimeError("array elements must be scalar/value types in backend v1")
        c_element_type = self.map_type_to_c(element_type)
        items = ast.get("items", [])
        if not items:
            self.generate_array_struct(py_type)
            return f"{self.array_struct_name(py_type)}_create(NULL, 0)"
        values_name = f"ocean_array_{var_name}_{self.temp_var_counter}_values"
        self.temp_var_counter += 1
        values = ", ".join(self.generate_expression(item) for item in items)
        self.add_line(f"{c_element_type} {values_name}[{len(items)}] = {{ {values} }};")
        self.generate_array_struct(py_type)
        return f"{self.array_struct_name(py_type)}_create({values_name}, {len(items)})"

    def _generate_owned_literal_expr(self, var_name: str, py_type: str, ast: Dict) -> str:
        if self.is_array_type(py_type):
            return self._generate_array_literal_expr(var_name, py_type, ast)
        raise RuntimeError(f"unsupported owned buffer type: {py_type}")

    def generate_array_struct(self, py_type: str) -> None:
        base = self.strip_borrow_type(py_type)
        if not self.is_array_type(base):
            return
        struct_name = self.array_struct_name(base)
        if struct_name in self.generated_structures:
            return
        element_type = self.array_element_type(base)
        if self.is_array_type(element_type):
            raise RuntimeError("nested array/tensor elements are not supported in backend v1")
        c_element_type = self.map_type_to_c(element_type)
        self.generated_structures.add(struct_name)
        self.generated_helpers.append(f"""
typedef struct {struct_name} {{
    {c_element_type}* data;
    size_t size;
    size_t capacity;
}} {struct_name};
""")
        self.generated_helpers.append(f"""
static {struct_name}* {struct_name}_create(const {c_element_type}* values, size_t size) {{
    {struct_name}* array = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!array) {{ fprintf(stderr, "Ocean allocation error: {struct_name}\\n"); exit(1); }}
    array->size = size;
    array->capacity = size;
    if (size > 0) {{
        array->data = ({c_element_type}*)malloc(size * sizeof({c_element_type}));
        if (!array->data) {{ free(array); fprintf(stderr, "Ocean allocation error: {struct_name} data\\n"); exit(1); }}
        memcpy(array->data, values, size * sizeof({c_element_type}));
    }}
    return array;
}}

static void {struct_name}_free({struct_name}* array) {{
    if (!array) return;
    free(array->data);
    free(array);
}}

static inline size_t {struct_name}_len(const {struct_name}* array) {{
    return array ? array->size : 0;
}}

static inline {c_element_type} {struct_name}_get(const {struct_name}* array, size_t index) {{
    if (!array || index >= array->size) {{
        fprintf(stderr, "Index out of bounds in {struct_name}\\n"); exit(1);
    }}
    return array->data[index];
}}

static inline void {struct_name}_set({struct_name}* array, size_t index, {c_element_type} value) {{
    if (!array || index >= array->size) {{
        fprintf(stderr, "Index out of bounds in {struct_name}\\n"); exit(1);
    }}
    array->data[index] = value;
}}
""")
