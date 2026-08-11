from __future__ import annotations

from typing import Dict, Iterable, List


class TensorCodegenMixin:
    """Lower dense row-major ``tensor[T]`` values with owned storage."""

    def _tensor_fast_index_expression(self, tensor_expr: str, indices: Iterable[str]):
        values = list(indices)
        aliases = getattr(self, "tensor_fast_access", {})
        if tensor_expr not in aliases or len(values) != 2:
            return None
        if (tensor_expr, tuple(value.strip() for value in values)) not in self.tensor_fast_patterns:
            return None
        if any(value.strip() not in self.tensor_fast_loop_bounds for value in values):
            return None
        alias = aliases[tensor_expr]
        return (
            f"{alias['data']}[(size_t)({values[0]}) * {alias['stride0']} + "
            f"(size_t)({values[1]}) * {alias['stride1']}]"
        )

    def tensor_struct_name(self, py_type: str) -> str:
        return f"ocean_tensor_{self.clean_type_name_for_c(self.tensor_element_type(py_type))}"

    def tensor_c_type(self, py_type: str) -> str:
        return f"{self.tensor_struct_name(py_type)}*"

    def _tensor_ast_type(self, ast: Dict) -> str | None:
        if not isinstance(ast, dict) or ast.get("type") != "variable":
            return None
        info = self.get_variable_info(ast.get("value") or ast.get("name", ""))
        if not info:
            return None
        py_type = self.strip_borrow_type(info.get("py_type", ""))
        return py_type if self.is_tensor_type(py_type) else None

    def generate_tensor_broadcast_binary(self, left_ast: Dict, right_ast: Dict, operator: str):
        """Lower tensor arithmetic to shape-checked broadcasting helpers."""
        left_type = self._tensor_ast_type(left_ast)
        right_type = self._tensor_ast_type(right_ast)
        if not left_type and not right_type:
            return None
        if operator not in {"+", "-", "*", "/"}:
            raise RuntimeError(f"tensor broadcasting does not support '{operator}'")

        operation = {"+": 0, "-": 1, "*": 2, "/": 3}[operator]
        if left_type and right_type:
            if self.tensor_element_type(left_type) != self.tensor_element_type(right_type):
                raise RuntimeError("tensor broadcasting requires matching element types")
            struct_name = self.tensor_struct_name(left_type)
            left = self.generate_expression(left_ast)
            right = self.generate_expression(right_ast)
            self.generate_tensor_struct(left_type)
            return f"{struct_name}_binary_broadcast({left}, {right}, {operation})"

        tensor_ast = left_ast if left_type else right_ast
        scalar_ast = right_ast if left_type else left_ast
        tensor_type = left_type or right_type
        struct_name = self.tensor_struct_name(tensor_type)
        tensor = self.generate_expression(tensor_ast)
        scalar = self.generate_expression(scalar_ast)
        self.generate_tensor_struct(tensor_type)
        scalar_left = 1 if not left_type else 0
        return f"{struct_name}_scalar_broadcast({tensor}, {scalar}, {operation}, {scalar_left})"

    def _flatten_tensor_items(self, ast: Dict) -> List[Dict]:
        if ast.get("type") != "list_literal":
            return [ast]
        result: List[Dict] = []
        for item in ast.get("items", []):
            result.extend(self._flatten_tensor_items(item))
        return result

    def _tensor_index_call(self, tensor_expr: str, py_type: str, indices: Iterable[str]) -> str:
        values = list(indices)
        fast_expression = self._tensor_fast_index_expression(tensor_expr, values)
        if fast_expression is not None:
            return fast_expression
        struct_name = self.tensor_struct_name(py_type)
        if len(values) in getattr(self, "tensor_index_ranks", set()):
            arguments = ", ".join(values)
            return f"{struct_name}_get{len(values)}({tensor_expr}, {arguments})"
        literal = ", ".join(f"(size_t)({value})" for value in values)
        return f"{struct_name}_get({tensor_expr}, (size_t[]){{{literal}}}, {len(values)})"

    def _tensor_set_call(self, tensor_expr: str, py_type: str, indices: Iterable[str], value: str) -> str:
        values = list(indices)
        fast_expression = self._tensor_fast_index_expression(tensor_expr, values)
        if fast_expression is not None:
            return f"{fast_expression} = {value};"
        struct_name = self.tensor_struct_name(py_type)
        if len(values) in getattr(self, "tensor_index_ranks", set()):
            arguments = ", ".join(values)
            return f"{struct_name}_set{len(values)}({tensor_expr}, {arguments}, {value});"
        literal = ", ".join(f"(size_t)({index})" for index in values)
        return f"{struct_name}_set({tensor_expr}, (size_t[]){{{literal}}}, {len(values)}, {value});"

    def _tensor_rank_helpers(self, struct_name: str, c_element_type: str) -> str:
        helpers = []
        for rank in sorted(getattr(self, "tensor_index_ranks", set())):
            if rank < 1:
                continue
            parameters = ", ".join(f"size_t index_{axis}" for axis in range(rank))
            checks = " || ".join(
                f"index_{axis} >= tensor->shape[{axis}]" for axis in range(rank)
            )
            offset = " + ".join(
                f"index_{axis} * tensor->strides[{axis}]" for axis in range(rank)
            )
            helpers.append(f"""
static inline {c_element_type} {struct_name}_get{rank}(const {struct_name}* tensor, {parameters}) {{
    if (!tensor || {checks}) {{
        fprintf(stderr, "Tensor index out of bounds in {struct_name}\\n"); exit(1);
    }}
    return tensor->data[{offset}];
}}

static inline void {struct_name}_set{rank}({struct_name}* tensor, {parameters}, {c_element_type} value) {{
    if (!tensor || {checks}) {{
        fprintf(stderr, "Tensor index out of bounds in {struct_name}\\n"); exit(1);
    }}
    tensor->data[{offset}] = value;
}}
""")
        return "\n".join(helpers)

    def _tensor_numeric_helpers(self, struct_name: str, c_element_type: str) -> str:
        """Emit view-aware reductions, matrix operations, and broadcasting."""
        return f"""
static inline void {struct_name}_fill({struct_name}* tensor, {c_element_type} value) {{
    if (!tensor) {{ fprintf(stderr, "Tensor fill on NULL in {struct_name}\\n"); exit(1); }}
    for (size_t linear = 0; linear < tensor->size; ++linear) {{
        size_t remaining = linear;
        size_t offset = 0;
        for (size_t axis = tensor->ndim; axis-- > 0;) {{
            size_t coordinate = tensor->shape[axis] ? remaining % tensor->shape[axis] : 0;
            remaining = tensor->shape[axis] ? remaining / tensor->shape[axis] : 0;
            offset += coordinate * tensor->strides[axis];
        }}
        tensor->data[offset] = value;
    }}
}}

static inline {c_element_type} {struct_name}_sum(const {struct_name}* tensor) {{
    if (!tensor) {{ fprintf(stderr, "Tensor sum on NULL in {struct_name}\\n"); exit(1); }}
    {c_element_type} result = ({c_element_type})0;
    for (size_t linear = 0; linear < tensor->size; ++linear) {{
        size_t remaining = linear;
        size_t offset = 0;
        for (size_t axis = tensor->ndim; axis-- > 0;) {{
            size_t coordinate = tensor->shape[axis] ? remaining % tensor->shape[axis] : 0;
            remaining = tensor->shape[axis] ? remaining / tensor->shape[axis] : 0;
            offset += coordinate * tensor->strides[axis];
        }}
        result += tensor->data[offset];
    }}
    return result;
}}

static {struct_name}* {struct_name}_copy(const {struct_name}* source) {{
    if (!source) return NULL;
    {struct_name}* result = {struct_name}_zeros(source->shape, source->ndim);
    for (size_t linear = 0; linear < source->size; ++linear) {{
        size_t remaining = linear;
        size_t offset = 0;
        for (size_t axis = source->ndim; axis-- > 0;) {{
            size_t coordinate = source->shape[axis] ? remaining % source->shape[axis] : 0;
            remaining = source->shape[axis] ? remaining / source->shape[axis] : 0;
            offset += coordinate * source->strides[axis];
        }}
        result->data[linear] = source->data[offset];
    }}
    return result;
}}

static {struct_name}* {struct_name}_transpose2(const {struct_name}* source) {{
    if (!source || source->ndim != 2) {{
        fprintf(stderr, "Tensor transpose() expects a 2D tensor in {struct_name}\\n"); exit(1);
    }}
    size_t shape[2] = {{ source->shape[1], source->shape[0] }};
    {struct_name}* result = {struct_name}_zeros(shape, 2);
    for (size_t i = 0; i < source->shape[0]; ++i)
        for (size_t j = 0; j < source->shape[1]; ++j)
            result->data[j * result->strides[0] + i * result->strides[1]] =
                source->data[i * source->strides[0] + j * source->strides[1]];
    return result;
}}

static {struct_name}* {struct_name}_matmul2(
    const {struct_name}* left, const {struct_name}* right
) {{
    if (!left || !right || left->ndim != 2 || right->ndim != 2 ||
        left->shape[1] != right->shape[0]) {{
        fprintf(stderr, "Tensor matmul() expects compatible 2D tensors in {struct_name}\\n"); exit(1);
    }}
    size_t shape[2] = {{ left->shape[0], right->shape[1] }};
    {struct_name}* result = {struct_name}_zeros(shape, 2);
    for (size_t i = 0; i < left->shape[0]; ++i)
        for (size_t j = 0; j < right->shape[1]; ++j) {{
            {c_element_type} value = ({c_element_type})0;
            for (size_t k = 0; k < left->shape[1]; ++k)
                value += left->data[i * left->strides[0] + k * left->strides[1]] *
                         right->data[k * right->strides[0] + j * right->strides[1]];
            result->data[i * result->strides[0] + j * result->strides[1]] = value;
        }}
    return result;
}}

static {struct_name}* {struct_name}_binary_broadcast(
    const {struct_name}* left, const {struct_name}* right, int operation
) {{
    if (!left || !right) {{
        fprintf(stderr, "Tensor broadcast on NULL in {struct_name}\\n"); exit(1);
    }}
    size_t ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    size_t* shape = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    if (ndim && !shape) {{ fprintf(stderr, "Tensor broadcast shape allocation failed\\n"); exit(1); }}
    for (size_t axis = 0; axis < ndim; ++axis) {{
        size_t left_axis = axis + left->ndim >= ndim ? left->shape[axis + left->ndim - ndim] : 1;
        size_t right_axis = axis + right->ndim >= ndim ? right->shape[axis + right->ndim - ndim] : 1;
        if (left_axis != right_axis && left_axis != 1 && right_axis != 1) {{
            free(shape);
            fprintf(stderr, "Incompatible tensor shapes for broadcasting in {struct_name}\\n");
            exit(1);
        }}
        shape[axis] = left_axis > right_axis ? left_axis : right_axis;
    }}
    {struct_name}* result = {struct_name}_zeros(shape, ndim);
    for (size_t linear = 0; linear < result->size; ++linear) {{
        size_t remaining = linear;
        size_t left_offset = 0;
        size_t right_offset = 0;
        for (size_t axis = ndim; axis-- > 0;) {{
            size_t coordinate = shape[axis] ? remaining % shape[axis] : 0;
            remaining = shape[axis] ? remaining / shape[axis] : 0;
            if (axis + left->ndim >= ndim) {{
                size_t source_axis = axis + left->ndim - ndim;
                left_offset += (left->shape[source_axis] == 1 ? 0 : coordinate) * left->strides[source_axis];
            }}
            if (axis + right->ndim >= ndim) {{
                size_t source_axis = axis + right->ndim - ndim;
                right_offset += (right->shape[source_axis] == 1 ? 0 : coordinate) * right->strides[source_axis];
            }}
        }}
        {c_element_type} left_value = left->data[left_offset];
        {c_element_type} right_value = right->data[right_offset];
        if (operation == 0) result->data[linear] = left_value + right_value;
        else if (operation == 1) result->data[linear] = left_value - right_value;
        else if (operation == 2) result->data[linear] = left_value * right_value;
        else result->data[linear] = left_value / right_value;
    }}
    free(shape);
    return result;
}}

static {struct_name}* {struct_name}_scalar_broadcast(
    const {struct_name}* tensor, {c_element_type} scalar, int operation, int scalar_left
) {{
    if (!tensor) {{ fprintf(stderr, "Tensor scalar broadcast on NULL in {struct_name}\\n"); exit(1); }}
    {struct_name}* result = {struct_name}_zeros(tensor->shape, tensor->ndim);
    for (size_t linear = 0; linear < tensor->size; ++linear) {{
        size_t remaining = linear;
        size_t offset = 0;
        for (size_t axis = tensor->ndim; axis-- > 0;) {{
            size_t coordinate = tensor->shape[axis] ? remaining % tensor->shape[axis] : 0;
            remaining = tensor->shape[axis] ? remaining / tensor->shape[axis] : 0;
            offset += coordinate * tensor->strides[axis];
        }}
        {c_element_type} value = tensor->data[offset];
        if (operation == 0) result->data[linear] = scalar_left ? scalar + value : value + scalar;
        else if (operation == 1) result->data[linear] = scalar_left ? scalar - value : value - scalar;
        else if (operation == 2) result->data[linear] = scalar * value;
        else result->data[linear] = scalar_left ? scalar / value : value / scalar;
    }}
    return result;
}}
"""

    def _generate_tensor_method_call(
        self,
        object_name: str,
        object_type: str,
        method_name: str,
        arg_strings: list[str],
        is_standalone: bool,
        target_var: str = "",
    ):
        struct_name = self.tensor_struct_name(object_type)
        if method_name == "fill":
            if len(arg_strings) != 1:
                raise RuntimeError("tensor.fill() expects one value")
            self.assert_can_mutate(object_name)
            self.add_line(f"{struct_name}_fill({object_name}, {arg_strings[0]});")
            return None

        self.assert_can_read(object_name)

        if method_name == "sum":
            if arg_strings:
                raise RuntimeError("tensor.sum() expects no arguments")
            expression = f"{struct_name}_sum({object_name})"
        elif method_name == "copy":
            if arg_strings:
                raise RuntimeError("tensor.copy() expects no arguments")
            expression = f"{struct_name}_copy({object_name})"
        elif method_name == "transpose":
            if arg_strings:
                raise RuntimeError("tensor.transpose() expects no arguments")
            expression = f"{struct_name}_transpose2({object_name})"
        elif method_name == "transpose_view":
            if arg_strings:
                raise RuntimeError("tensor.transpose_view() expects no arguments")
            expression = f"{struct_name}_transpose_view({object_name})"
        elif method_name == "row":
            if len(arg_strings) != 1:
                raise RuntimeError("tensor.row() expects one index")
            expression = f"{struct_name}_row({object_name}, (size_t)({arg_strings[0]}))"
        elif method_name == "column":
            if len(arg_strings) != 1:
                raise RuntimeError("tensor.column() expects one index")
            expression = f"{struct_name}_column({object_name}, (size_t)({arg_strings[0]}))"
        elif method_name == "slice":
            if len(arg_strings) not in {3, 4}:
                raise RuntimeError("tensor.slice() expects axis, start, stop[, step]")
            step = arg_strings[3] if len(arg_strings) == 4 else "1"
            expression = (
                f"{struct_name}_slice({object_name}, (size_t)({arg_strings[0]}), "
                f"(size_t)({arg_strings[1]}), (size_t)({arg_strings[2]}), "
                f"(size_t)({step}))"
            )
        elif method_name == "matmul":
            if len(arg_strings) != 1:
                raise RuntimeError("tensor.matmul() expects one tensor")
            expression = f"{struct_name}_matmul2({object_name}, {arg_strings[0]})"
        else:
            raise RuntimeError(
                f"tensor method '{method_name}' is not implemented for '{object_type}'"
            )

        if target_var:
            self.add_line(f"{target_var} = {expression};")
            return None
        if is_standalone:
            self.add_line(f"(void){expression};")
            return None
        return expression

    def _tensor_fast_shape_expression(self, ast: Dict, declarations: Dict[str, Dict]):
        if not isinstance(ast, dict):
            return None
        if ast.get("type") == "complex_attribute_access" and ast.get("attribute") == "shape":
            index = ast.get("index", {})
            if index.get("type") == "literal":
                return f"{ast.get('object', '')}->shape[{index.get('value')}]"
        if ast.get("type") == "variable":
            name = ast.get("value")
            declaration = declarations.get(name)
            if declaration:
                return self._tensor_fast_shape_expression(
                    declaration.get("expression_ast"), declarations
                )
        return None

    def _collect_tensor_fast_accesses(self, nodes, tensor_types, declarations):
        accesses = []

        def visit(node, loop_bounds):
            if not isinstance(node, dict):
                return
            if node.get("node") == "for_loop":
                iterable = node.get("iterable", {})
                arguments = iterable.get("arguments", {})
                start = str(arguments.get("start", "0")).strip()
                step = str(arguments.get("step", "1")).strip()
                stop = str(arguments.get("stop", "")).strip()
                if iterable.get("type") != "RANGE_CALL" or start != "0" or step != "1":
                    return
                nested_bounds = dict(loop_bounds)
                nested_bounds[node.get("loop_variable", "i")] = stop
                for child in node.get("body", []):
                    visit(child, nested_bounds)
                return

            def visit_ast(value):
                if not isinstance(value, dict):
                    return
                if value.get("type") in {"tensor_index_access", "nested_index_access"}:
                    variable = value.get("variable", "")
                    indices = value.get("indices", [])
                    if variable in tensor_types and len(indices) == 2:
                        index_names = [item.get("value") for item in indices]
                        if all(name in loop_bounds for name in index_names):
                            accesses.append(
                                (variable, tensor_types[variable], index_names, dict(loop_bounds))
                            )
                for child in value.values():
                    if isinstance(child, dict):
                        visit_ast(child)
                    elif isinstance(child, list):
                        for item in child:
                            visit_ast(item)

            if node.get("node") == "nested_index_assignment":
                variable = node.get("variable", "")
                indices = node.get("indices", [])
                if variable in tensor_types and len(indices) == 2:
                    index_names = [item.get("value") for item in indices]
                    if all(name in loop_bounds for name in index_names):
                        accesses.append(
                            (variable, tensor_types[variable], index_names, dict(loop_bounds))
                        )

            for key, value in node.items():
                if key in {"body", "iterable"}:
                    continue
                if isinstance(value, dict):
                    visit_ast(value)
                elif isinstance(value, list):
                    for item in value:
                        visit_ast(item)

            for child in node.get("body", []):
                visit(child, loop_bounds)

        for node in nodes:
            visit(node, {})
        return accesses

    def _prepare_tensor_fast_path(self, scope: Dict) -> None:
        """Prepare checked-once direct access for provably bounded 2D loops."""
        symbol_table = scope.get("symbol_table", {}) or {}
        parameter_names = {
            parameter.get("name")
            for parameter in scope.get("parameters", []) or []
            if parameter.get("name")
        }
        tensor_types = {
            name: self.strip_borrow_type(info.get("type", ""))
            for name, info in symbol_table.items()
            if name in parameter_names
            and isinstance(info, dict)
            and self.is_tensor_type(self.strip_borrow_type(info.get("type", "")))
        }
        if not tensor_types:
            return

        declarations = {
            node.get("var_name"): node
            for node in scope.get("graph", [])
            if isinstance(node, dict) and node.get("node") == "declaration"
        }
        accesses = self._collect_tensor_fast_accesses(
            scope.get("graph", []), tensor_types, declarations
        )
        if not accesses:
            return

        constraints = set()
        fast_variables = set()
        fast_patterns = set()
        for variable, _, index_names, loop_bounds in accesses:
            resolved_bounds = []
            for axis, index_name in enumerate(index_names):
                bound = self._tensor_fast_shape_expression(
                    {"type": "variable", "value": loop_bounds[index_name]},
                    declarations,
                )
                if bound is None:
                    bound = loop_bounds[index_name]
                    if not bound.isdigit():
                        break
                if bound == f"{variable}->shape[{axis}]":
                    continue
                resolved_bounds.append((axis, bound))
            else:
                fast_variables.add(variable)
                fast_patterns.add((variable, tuple(index_names)))
                constraints.update(
                    (variable, axis, bound) for axis, bound in resolved_bounds
                )

        check_parts = []
        for variable in sorted(fast_variables):
            check_parts.extend([f"{variable} == NULL", f"{variable}->ndim != 2"])
        for variable, axis, bound in sorted(constraints):
            check_parts.append(f"{variable}->shape[{axis}] < ({bound})")
        if check_parts:
            self.add_line("if (" + " || ".join(check_parts) + ") {")
            self.indent_level += 1
            self.add_line('fprintf(stderr, "Tensor fast-path shape check failed\\n");')
            self.add_line("exit(1);")
            self.indent_level -= 1
            self.add_line("}")

        for variable in sorted(fast_variables):
            py_type = tensor_types[variable]
            alias_base = f"ocean_fast_{self.clean_type_name_for_c(variable)}"
            original_type = next(
                info.get("type", "")
                for name, info in symbol_table.items()
                if name == variable and isinstance(info, dict)
            )
            qualifier = "" if self.is_mut_borrow_type(original_type) or not self.is_borrow_type(original_type) else "const "
            self.add_line(f"{qualifier}{self.map_type_to_c(self.tensor_element_type(py_type))}* {alias_base}_data = {variable}->data;")
            self.add_line(f"const size_t {alias_base}_stride0 = {variable}->strides[0];")
            self.add_line(f"const size_t {alias_base}_stride1 = {variable}->strides[1];")
            self.tensor_fast_access[variable] = {
                "data": f"{alias_base}_data",
                "stride0": f"{alias_base}_stride0",
                "stride1": f"{alias_base}_stride1",
            }
        self.tensor_fast_patterns = fast_patterns

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

    def _is_tensor_zeros_expression(self, ast: Dict) -> bool:
        return (
            isinstance(ast, dict)
            and ast.get("type") == "method_call"
            and ast.get("object") == "tensor"
            and ast.get("method") == "zeros"
        )

    def _generate_tensor_zeros_expr(
        self,
        var_name: str,
        py_type: str,
        ast: Dict,
        expression_generator=None,
    ) -> str:
        args = ast.get("arguments", []) or []
        if not args:
            raise RuntimeError("tensor.zeros expects at least one dimension")
        if not self.is_tensor_type(py_type):
            raise RuntimeError(f"tensor.zeros target must be tensor[T], got {py_type}")
        self.generate_tensor_struct(py_type)
        shape_name = f"ocean_tensor_{var_name}_{self.temp_var_counter}_shape"
        self.temp_var_counter += 1
        generate_argument = expression_generator or self.generate_expression
        dimensions = ", ".join(
            f"(size_t)({generate_argument(argument)})" for argument in args
        )
        self.add_line(f"size_t {shape_name}[{len(args)}] = {{ {dimensions} }};")
        return f"{self.tensor_struct_name(py_type)}_zeros({shape_name}, {len(args)})"

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
        elif self._is_tensor_zeros_expression(expression_ast):
            expr = self._generate_tensor_zeros_expr(var_name, var_type, expression_ast)
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
        target = node.get("symbols", [""])[0]
        target_type = node.get("var_type") or self.get_variable_info(target).get("py_type")
        expression_ast = node.get("expression_ast") or {}
        if self._is_tensor_zeros_expression(expression_ast):
            self.assert_can_move_or_delete(target)
            self.add_line(self._owned_free_call(target, target_type))
            expr = self._generate_tensor_zeros_expr(target, target_type, expression_ast)
            self.add_line(f"{target} = {expr};")
            info = self.get_variable_info(target)
            info["is_deleted"] = False
            info["is_moved"] = False
            info["owns_reference"] = True
            return
        self._generate_owned_assignment(target, target_type, expression_ast)

    def _generate_tensor_literal_expr(
        self,
        var_name: str,
        py_type: str,
        ast: Dict,
        expression_generator=None,
    ) -> str:
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
        generate_item = expression_generator or self.generate_expression
        if flat_items:
            values = ", ".join(generate_item(item) for item in flat_items)
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
    size_t refcount;
    bool is_view;
    struct {struct_name}* owner;
}} {struct_name};

enum {{ {struct_name}_MAX_RANK = 64 }};
static void {struct_name}_release({struct_name}* tensor);
""")
        self.generated_helpers.append(f"""
static {struct_name}* {struct_name}_create(const {c_element_type}* values, size_t value_count, const size_t* shape, size_t ndim) {{
    {struct_name}* tensor = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!tensor) {{ fprintf(stderr, "Ocean allocation error: {struct_name}\\n"); exit(1); }}
    tensor->refcount = 1;
    tensor->is_view = false;
    tensor->owner = NULL;
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
    if (tensor->size != value_count && !(values == NULL && value_count == 0)) {{
        fprintf(stderr, "Tensor literal size mismatch in {struct_name}\\n"); exit(1);
    }}
    if (tensor->size) {{
        tensor->data = ({c_element_type}*)calloc(tensor->size, sizeof({c_element_type}));
        if (!tensor->data) {{ fprintf(stderr, "Ocean allocation error: {struct_name} data\\n"); exit(1); }}
        if (values) memcpy(tensor->data, values, tensor->size * sizeof({c_element_type}));
    }}
    return tensor;
}}

static {struct_name}* {struct_name}_zeros(const size_t* shape, size_t ndim) {{
    return {struct_name}_create(NULL, 0, shape, ndim);
}}

static void {struct_name}_retain({struct_name}* tensor) {{
    if (!tensor) return;
    if (tensor->refcount == (size_t)-1) {{
        fprintf(stderr, "Tensor owner reference count overflow in {struct_name}\\n"); exit(1);
    }}
    tensor->refcount += 1;
}}

static void {struct_name}_destroy({struct_name}* tensor) {{
    if (!tensor) return;
    if (tensor->is_view) {{
        {struct_name}_release(tensor->owner);
    }} else {{
        free(tensor->data);
    }}
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}}

static void {struct_name}_release({struct_name}* tensor) {{
    if (!tensor) return;
    if (tensor->refcount == 0) {{
        fprintf(stderr, "Tensor owner release of dead object in {struct_name}\\n"); exit(1);
    }}
    tensor->refcount -= 1;
    if (tensor->refcount == 0) {struct_name}_destroy(tensor);
}}

static void {struct_name}_free({struct_name}* tensor) {{
    {struct_name}_release(tensor);
}}

static {struct_name}* {struct_name}_view(
    const {struct_name}* source, size_t offset, const size_t* shape,
    const size_t* strides, size_t ndim
) {{
    if (!source || (offset > source->size && source->size != 0)) {{
        fprintf(stderr, "Tensor view offset out of bounds in {struct_name}\\n"); exit(1);
    }}
    {struct_name}* view = ({struct_name}*)calloc(1, sizeof({struct_name}));
    if (!view) {{ fprintf(stderr, "Ocean allocation error: {struct_name} view\\n"); exit(1); }}
    view->refcount = 1;
    view->is_view = true;
    view->owner = ({struct_name}*)source;
    {struct_name}_retain(view->owner);
    view->ndim = ndim;
    view->shape = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    view->strides = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    if ((ndim && !view->shape) || (ndim && !view->strides)) {{
        free(view->shape); free(view->strides); {struct_name}_release(view->owner); free(view);
        fprintf(stderr, "Ocean allocation error: {struct_name} view metadata\\n"); exit(1);
    }}
    view->size = 1;
    for (size_t axis = 0; axis < ndim; ++axis) {{
        view->shape[axis] = shape[axis];
        view->strides[axis] = strides[axis];
        if (view->shape[axis] != 0 && view->size > (size_t)-1 / view->shape[axis]) {{
            {struct_name}_release(view->owner); free(view->shape); free(view->strides); free(view);
            fprintf(stderr, "Tensor view size overflow in {struct_name}\\n"); exit(1);
        }}
        view->size *= view->shape[axis];
    }}
    if (ndim == 0) view->size = 0;
    view->data = source->data ? source->data + offset : NULL;
    return view;
}}

static {struct_name}* {struct_name}_row(const {struct_name}* source, size_t index) {{
    if (!source || source->ndim != 2 || index >= source->shape[0]) {{
        fprintf(stderr, "Tensor row() expects a valid 2D row in {struct_name}\\n"); exit(1);
    }}
    size_t shape[1] = {{ source->shape[1] }};
    size_t strides[1] = {{ source->strides[1] }};
    return {struct_name}_view(source, index * source->strides[0], shape, strides, 1);
}}

static {struct_name}* {struct_name}_column(const {struct_name}* source, size_t index) {{
    if (!source || source->ndim != 2 || index >= source->shape[1]) {{
        fprintf(stderr, "Tensor column() expects a valid 2D column in {struct_name}\\n"); exit(1);
    }}
    size_t shape[1] = {{ source->shape[0] }};
    size_t strides[1] = {{ source->strides[0] }};
    return {struct_name}_view(source, index * source->strides[1], shape, strides, 1);
}}

static {struct_name}* {struct_name}_transpose_view(const {struct_name}* source) {{
    if (!source || source->ndim != 2) {{
        fprintf(stderr, "Tensor transpose_view() expects a 2D tensor in {struct_name}\\n"); exit(1);
    }}
    size_t shape[2] = {{ source->shape[1], source->shape[0] }};
    size_t strides[2] = {{ source->strides[1], source->strides[0] }};
    return {struct_name}_view(source, 0, shape, strides, 2);
}}

static {struct_name}* {struct_name}_slice(
    const {struct_name}* source, size_t axis, size_t start, size_t stop, size_t step
) {{
    if (!source || axis >= source->ndim || step == 0 || start > stop || stop > source->shape[axis]) {{
        fprintf(stderr, "Invalid tensor slice in {struct_name}\\n"); exit(1);
    }}
    size_t shape[{struct_name}_MAX_RANK];
    size_t strides[{struct_name}_MAX_RANK];
    if (source->ndim > {struct_name}_MAX_RANK) {{
        fprintf(stderr, "Tensor rank exceeds view limit in {struct_name}\\n"); exit(1);
    }}
    for (size_t i = 0; i < source->ndim; ++i) {{ shape[i] = source->shape[i]; strides[i] = source->strides[i]; }}
    shape[axis] = stop <= start ? 0 : 1 + (stop - 1 - start) / step;
    strides[axis] *= step;
    return {struct_name}_view(source, start * source->strides[axis], shape, strides, source->ndim);
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

{self._tensor_rank_helpers(struct_name, c_element_type)}
{self._tensor_numeric_helpers(struct_name, c_element_type)}
""")
