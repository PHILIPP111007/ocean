from __future__ import annotations

from typing import Dict, Iterable, List


class TensorCodegenMixin:
    """Lower dense row-major ``tensor[T]`` values with owned storage."""

    def tensor_struct_name(self, py_type: str) -> str:
        return f"ocean_tensor_{self.clean_type_name_for_c(self.tensor_element_type(py_type))}"

    def tensor_c_type(self, py_type: str) -> str:
        return f"{self.tensor_struct_name(py_type)}*"

    def _flatten_tensor_items(self, ast: Dict) -> List[Dict]:
        if ast.get("type") != "list_literal":
            return [ast]
        result: List[Dict] = []
        for item in ast.get("items", []):
            result.extend(self._flatten_tensor_items(item))
        return result

    def _tensor_index_call(self, tensor_expr: str, py_type: str, indices: Iterable[str]) -> str:
        values = list(indices)
        struct_name = self.tensor_struct_name(py_type)
        literal = ", ".join(f"(size_t)({value})" for value in values)
        return f"{struct_name}_get({tensor_expr}, (size_t[]){{{literal}}}, {len(values)})"

    def _tensor_set_call(self, tensor_expr: str, py_type: str, indices: Iterable[str], value: str) -> str:
        values = list(indices)
        struct_name = self.tensor_struct_name(py_type)
        literal = ", ".join(f"(size_t)({index})" for index in values)
        return f"{struct_name}_set({tensor_expr}, (size_t[]){{{literal}}}, {len(values)}, {value});"

    def generate_tensor_index_access(self, ast: Dict) -> str:
        variable = ast.get("variable", "")
        info = self.get_variable_info(variable)
        if not info:
            raise RuntimeError(f"unknown tensor '{variable}'")
        self.assert_can_read(variable)
        py_type = self.strip_borrow_type(info.get("py_type", ""))
        indices = [self.generate_expression(index) for index in ast.get("indices", [])]
        return self._tensor_index_call(variable, py_type, indices)

    def generate_tensor_index_assignment(self, variable: str, py_type: str, indices, value: str) -> None:
        expressions = [self.generate_expression(index) for index in indices]
        self.add_line(self._tensor_set_call(variable, py_type, expressions, value))

    def generate_tensor_declaration(self, node: Dict) -> None:
        var_name = node.get("var_name", "")
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast", {}) or {}
        self.declare_variable(var_name, var_type)
        info = self.get_variable_info(var_name)
        if not info:
            return

        if expression_ast.get("type") == "list_literal":
            expr = self._generate_tensor_literal_expr(var_name, var_type, expression_ast)
        elif self._is_none_expression(expression_ast):
            expr = "NULL"
        else:
            source = self._owned_source(expression_ast, var_type)
            if source:
                self.assert_can_read(source)
                self.assert_can_move_or_delete(source)
                self.add_line(f"{self.tensor_c_type(var_type)} {var_name} = {source};")
                self.add_line(f"{source} = NULL;")
                self._mark_owned_move(source)
                return
            expr = self.generate_expression(expression_ast)

        self.add_line(f"{self.tensor_c_type(var_type)} {var_name} = {expr};")

    def generate_tensor_assignment(self, node: Dict) -> None:
        self._generate_owned_assignment(
            node.get("symbols", [""])[0],
            node.get("var_type") or self.get_variable_info(node.get("symbols", [""])[0]).get("py_type"),
            node.get("expression_ast") or {},
        )

    def _generate_tensor_literal_expr(self, var_name: str, py_type: str, ast: Dict) -> str:
        shape = self._infer_tensor_shape(ast)
        if shape is None:
            raise RuntimeError(f"tensor literal for '{var_name}' must be rectangular")
        element_type = self.tensor_element_type(py_type)
        if self.is_array_type(element_type) or self.is_tensor_type(element_type):
            raise RuntimeError("tensor elements must be scalar/value types in backend v1")
        c_element_type = self.map_type_to_c(element_type)
        struct_name = self.tensor_struct_name(py_type)
        self.generate_tensor_struct(py_type)
        flat_items = self._flatten_tensor_items(ast)
        data_name = f"ocean_tensor_{var_name}_{self.temp_var_counter}_data"
        shape_name = f"ocean_tensor_{var_name}_{self.temp_var_counter}_shape"
        self.temp_var_counter += 1
        if flat_items:
            values = ", ".join(self.generate_expression(item) for item in flat_items)
            self.add_line(f"{c_element_type} {data_name}[{len(flat_items)}] = {{ {values} }};")
        else:
            data_name = "NULL"
        shape_values = ", ".join(str(value) for value in shape)
        self.add_line(f"size_t {shape_name}[{len(shape)}] = {{ {shape_values} }};")
        return f"{struct_name}_create({data_name}, {len(flat_items)}, {shape_name}, {len(shape)})"

    def _infer_tensor_shape(self, ast: Dict):
        if ast.get("type") != "list_literal":
            return None
        items = ast.get("items", [])
        if not items:
            return [0]
        child_shapes = []
        for item in items:
            if item.get("type") == "list_literal":
                shape = self._infer_tensor_shape(item)
                if shape is None:
                    return None
                child_shapes.append(shape)
            else:
                child_shapes.append([])
        first = child_shapes[0]
        if any(shape != first for shape in child_shapes[1:]):
            return None
        return [len(items), *first]

    def generate_tensor_struct(self, py_type: str) -> None:
        base = self.strip_borrow_type(py_type)
        if not self.is_tensor_type(base):
            return
        struct_name = self.tensor_struct_name(base)
        if struct_name in self.generated_structures:
            return
        element_type = self.tensor_element_type(base)
        if self.is_array_type(element_type) or self.is_tensor_type(element_type):
            raise RuntimeError("nested array/tensor elements are not supported in backend v1")
        c_element_type = self.map_type_to_c(element_type)
        self.generated_structures.add(struct_name)
        self.generated_helpers.append(f"""
typedef struct {struct_name} {{
    {c_element_type}* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
}} {struct_name};
""")
        self.generated_helpers.append(f"""
static {struct_name}* {struct_name}_create(const {c_element_type}* values, size_t value_count, const size_t* shape, size_t ndim) {{
    {struct_name}* tensor = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!tensor) {{ fprintf(stderr, "Ocean allocation error: {struct_name}\\n"); exit(1); }}
    tensor->ndim = ndim;
    tensor->size = 1;
    if (ndim == 0) tensor->size = 0;
    tensor->shape = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    tensor->strides = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    if ((ndim && !tensor->shape) || (ndim && !tensor->strides)) {{
        free(tensor->shape); free(tensor->strides); free(tensor);
        fprintf(stderr, "Ocean allocation error: {struct_name} metadata\\n"); exit(1);
    }}
    for (size_t i = 0; i < ndim; ++i) tensor->shape[i] = shape[i];
    size_t stride = 1;
    for (size_t i = ndim; i-- > 0;) {{
        tensor->strides[i] = stride;
        if (tensor->shape[i] != 0 && stride > (size_t)-1 / tensor->shape[i]) {{
            fprintf(stderr, "Tensor size overflow in {struct_name}\\n"); exit(1);
        }}
        stride *= tensor->shape[i];
    }}
    tensor->size = ndim ? stride : 0;
    if (tensor->size != value_count) {{
        fprintf(stderr, "Tensor literal size mismatch in {struct_name}\\n"); exit(1);
    }}
    if (tensor->size) {{
        tensor->data = ({c_element_type}*)malloc(tensor->size * sizeof({c_element_type}));
        if (!tensor->data) {{ fprintf(stderr, "Ocean allocation error: {struct_name} data\\n"); exit(1); }}
        memcpy(tensor->data, values, tensor->size * sizeof({c_element_type}));
    }}
    return tensor;
}}

static void {struct_name}_free({struct_name}* tensor) {{
    if (!tensor) return;
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}}

static inline size_t {struct_name}_len(const {struct_name}* tensor) {{
    return tensor ? tensor->size : 0;
}}

static inline size_t {struct_name}_shape_at(const {struct_name}* tensor, size_t axis) {{
    if (!tensor || axis >= tensor->ndim) {{
        fprintf(stderr, "Tensor shape index out of bounds in {struct_name}\\n"); exit(1);
    }}
    return tensor->shape[axis];
}}

static size_t {struct_name}_offset(const {struct_name}* tensor, const size_t* indices, size_t rank) {{
    if (!tensor || rank != tensor->ndim || (rank && !indices)) {{
        fprintf(stderr, "Tensor rank mismatch in {struct_name}\\n"); exit(1);
    }}
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i) {{
        if (indices[i] >= tensor->shape[i]) {{
            fprintf(stderr, "Tensor index out of bounds in {struct_name}\\n"); exit(1);
        }}
        offset += indices[i] * tensor->strides[i];
    }}
    return offset;
}}

static inline {c_element_type} {struct_name}_get(const {struct_name}* tensor, const size_t* indices, size_t rank) {{
    return tensor->data[{struct_name}_offset(tensor, indices, rank)];
}}

static inline void {struct_name}_set({struct_name}* tensor, const size_t* indices, size_t rank, {c_element_type} value) {{
    tensor->data[{struct_name}_offset(tensor, indices, rank)] = value;
}}
""")
