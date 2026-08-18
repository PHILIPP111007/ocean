from __future__ import annotations

from typing import Dict, Iterable, List


class TensorCodegenMixin:
    """Lower the public, opaque ``Tensor[T]`` facade.

    Tensor storage lives exclusively in the standard runtime; this mixin only
    lowers constructors and indexed access for the public device-aware type.
    """

    def _generate_device_tensor_expression(self, ast: Dict, expected_type: str) -> str:
        previous = getattr(self, "device_tensor_expected_type", None)
        self.device_tensor_expected_type = expected_type
        try:
            return self.generate_expression(ast)
        finally:
            self.device_tensor_expected_type = previous

    def _device_tensor_call_dtype(self, class_type: str) -> str:
        if class_type == "Tensor":
            expected_type = getattr(self, "device_tensor_expected_type", None)
            if expected_type and self.is_device_tensor_type(expected_type):
                return self.device_tensor_dtype(expected_type)
        return self.device_tensor_dtype(class_type)

    def _device_tensor_static_call(self, ast: Dict):
        class_type = ast.get("class_type") or ast.get("class_name", "")
        if not class_type.startswith("Tensor"):
            return None

        dtype = self._device_tensor_call_dtype(class_type)
        args = ast.get("arguments", []) or []
        method = ast.get("method", "")
        generate_argument = getattr(
            self, "device_tensor_argument_generator", None
        ) or self.generate_expression

        if method == "zeros":
            if len(args) < 2:
                raise RuntimeError("Tensor[T].zeros expects dimensions and device")
            device_ast = args[-1]
            dimensions = args[:-1]
            shape_name = f"ocean_device_tensor_shape_{self.temp_var_counter}"
            self.temp_var_counter += 1
            values = ", ".join(
                f"(size_t)({generate_argument(argument)})"
                for argument in dimensions
            )
            self.add_line(
                f"size_t {shape_name}[{len(dimensions)}] = {{ {values} }};"
            )
            device = generate_argument(device_ast)
            return (
                f"create_Tensor(ocean_tensor_zeros_nd({shape_name}, "
                f"{len(dimensions)}, \"{dtype}\", {device}))"
            )

        if method == "from_list":
            if len(args) != 2:
                raise RuntimeError(
                    "Tensor[T].from_list expects a list and device"
                )
            source_ast = args[0]
            if source_ast.get("type") == "variable":
                source_name = source_ast.get("value") or source_ast.get("name")
                source_info = self.get_variable_info(source_name)
                source_type = self.strip_borrow_type(
                    source_info.get("py_type", "") if source_info else ""
                )
                if source_type.startswith("list["):
                    return self._generate_tensor_from_list_variable(
                        source_name, source_type, dtype, args[1]
                    )
                raise RuntimeError(
                    "Tensor[T].from_list expects a list or rectangular list literal"
                )
            if source_ast.get("type") != "list_literal":
                raise RuntimeError(
                    "Tensor[T].from_list expects a list or rectangular list literal"
                )
            shape = self._infer_tensor_shape(source_ast)
            if shape is None:
                raise RuntimeError(
                    "Tensor[T].from_list expects a rectangular list literal"
                )
            flat_items = self._flatten_tensor_items(source_ast)
            c_element_type = self.map_type_to_c(dtype)
            data_name = f"ocean_device_tensor_data_{self.temp_var_counter}"
            shape_name = f"ocean_device_tensor_shape_{self.temp_var_counter}"
            strides_name = f"ocean_device_tensor_strides_{self.temp_var_counter}"
            self.temp_var_counter += 1
            if flat_items:
                values = ", ".join(generate_argument(item) for item in flat_items)
                self.add_line(
                    f"{c_element_type} {data_name}[{len(flat_items)}] = {{ {values} }};"
                )
            else:
                data_name = "NULL"

            shape_values = ", ".join(str(value) for value in shape)
            strides: List[int] = []
            stride = 1
            for dimension in reversed(shape):
                strides.insert(0, stride)
                stride *= dimension
            stride_values = ", ".join(str(value) for value in strides)
            self.add_line(f"size_t {shape_name}[{len(shape)}] = {{ {shape_values} }};")
            self.add_line(
                f"size_t {strides_name}[{len(shape)}] = {{ {stride_values} }};"
            )
            device = generate_argument(args[1])
            result_name = f"ocean_device_tensor_{self.temp_var_counter}"
            self.temp_var_counter += 1
            self.add_line(
                f"Tensor* {result_name} = create_Tensor(ocean_tensor_from_cpu_strided("
                f"(const void*){data_name}, {shape_name}, {strides_name}, "
                f"{len(shape)}, \"{dtype}\", {device}));"
            )
            return result_name

        if method == "load_npy":
            if len(args) != 2:
                raise RuntimeError("Tensor.load_npy expects path and device")
            path = generate_argument(args[0])
            device = generate_argument(args[1])
            return (
                f"create_Tensor(ocean_tensor_load_npy_typed({path}, {device}, "
                f"\"{dtype}\"))"
            )

        return None


    def _device_tensor_instance_call(self, ast: Dict, object_name: str, obj_type: str):
        if not self.is_device_tensor_type(obj_type):
            return None

        method = ast.get("method", "")
        args = ast.get("arguments", []) or []
        gen = getattr(self, "device_tensor_argument_generator", None) or self.generate_expression

        if method == "reshape" and len(args) == 1:
            shape_ast = args[0]
            if shape_ast.get("type") != "list_literal":
                raise RuntimeError("Tensor.reshape(shape) expects a list literal")
            dims = shape_ast.get("items", []) or []
            if not dims:
                raise RuntimeError("Tensor.reshape(shape) requires rank >= 1")
            suffix = self.temp_var_counter
            self.temp_var_counter += 1
            name = f"ocean_device_tensor_reshape_shape_{suffix}"
            vals = ", ".join(f"(size_t)({gen(item)})" for item in dims)
            self.add_line(f"size_t {name}[{len(dims)}] = {{ {vals} }};")
            return (
                f"create_Tensor(ocean_autograd_reshape("
                f"{object_name}->handle, {name}, {len(dims)}))"
            )

        if method == "transpose" and len(args) == 2:
            d0 = gen(args[0])
            d1 = gen(args[1])
            return (
                f"create_Tensor(ocean_autograd_transpose_dims("
                f"{object_name}->handle, {d0}, {d1}))"
            )

        return None

    def _tensor_list_types(self, source_type: str) -> List[str]:
        types: List[str] = []
        current = source_type
        while current.startswith("list[") and current.endswith("]"):
            types.append(current)
            current = current[5:-1].strip()
        if not types or not current:
            raise RuntimeError("Tensor[T].from_list expects a typed numeric list")
        return types

    def _generate_tensor_from_list_variable(
        self, source_name: str, source_type: str, dtype: str, device_ast: Dict
    ) -> str:
        list_types = self._tensor_list_types(source_type)
        rank = len(list_types)
        for list_type in list_types:
            self.generate_list_struct(list_type)

        suffix = self.temp_var_counter
        self.temp_var_counter += 1
        shape_name = f"ocean_device_tensor_shape_{suffix}"
        strides_name = f"ocean_device_tensor_strides_{suffix}"
        data_name = f"ocean_device_tensor_data_{suffix}"
        offset_name = f"ocean_device_tensor_offset_{suffix}"
        self.add_line(f"size_t {shape_name}[{rank}] = {{0}};")

        current_expr = source_name
        for level, list_type in enumerate(list_types):
            struct_name = self.generate_list_struct_name(list_type)
            length_expr = f"builtin_len_{struct_name}({current_expr})"
            if level == 0:
                self.add_line(
                    f"{shape_name}[{level}] = (size_t)({length_expr});"
                )
            else:
                self.add_line(
                    f"{shape_name}[{level}] = {shape_name}[{level - 1}] > 0 "
                    f"? (size_t)({length_expr}) : 0;"
                )
            if level < rank - 1:
                child_type = list_types[level + 1]
                child_struct = self.generate_list_struct_name(child_type)
                child_name = f"ocean_device_tensor_shape_source_{suffix}_{level}"
                self.add_line(
                    f"{child_struct}* {child_name} = "
                    f"{shape_name}[{level}] > 0 ? "
                    f"get_{struct_name}({current_expr}, 0) : NULL;"
                )
                current_expr = child_name

        self.add_line(f"size_t {strides_name}[{rank}] = {{0}};")
        self.add_line(f"size_t {offset_name} = 0;")
        self.add_line(f"size_t ocean_device_tensor_total_{suffix} = 1;")
        self.add_line(
            f"for (size_t axis = 0; axis < {rank}; ++axis) "
            f"ocean_device_tensor_total_{suffix} *= {shape_name}[axis];"
        )
        self.add_line(
            f"{self.map_type_to_c(dtype)}* {data_name} = "
            f"ocean_device_tensor_total_{suffix} ? "
            f"({self.map_type_to_c(dtype)}*)malloc("
            f"ocean_device_tensor_total_{suffix} * sizeof({self.map_type_to_c(dtype)})) : NULL;"
        )
        self.add_line(
            f"if (ocean_device_tensor_total_{suffix} && !{data_name}) "
            "ocean_tensor_fail(\"out of memory flattening Tensor list\");"
        )

        self._emit_tensor_list_flatten(
            source_name, list_types, 0, data_name, offset_name, shape_name, suffix
        )
        for level in range(rank - 1, -1, -1):
            if level == rank - 1:
                self.add_line(f"{strides_name}[{level}] = 1;")
            else:
                self.add_line(
                    f"{strides_name}[{level}] = {strides_name}[{level + 1}] * "
                    f"{shape_name}[{level + 1}];"
                )
        device = self.generate_expression(device_ast)
        result_name = f"ocean_device_tensor_{suffix}"
        self.add_line(
            f"Tensor* {result_name} = create_Tensor(ocean_tensor_from_cpu_strided("
            f"(const void*){data_name}, {shape_name}, {strides_name}, {rank}, "
            f"\"{dtype}\", {device}));"
        )
        self.add_line(f"free({data_name});")
        return result_name

    def _emit_tensor_list_flatten(
        self,
        list_expr: str,
        list_types: List[str],
        level: int,
        data_name: str,
        offset_name: str,
        shape_name: str,
        suffix: int,
    ) -> None:
        list_type = list_types[level]
        struct_name = self.generate_list_struct_name(list_type)
        index_name = f"ocean_device_tensor_index_{suffix}_{level}"
        self.add_line(
            f"for (int {index_name} = 0; {index_name} < "
            f"builtin_len_{struct_name}({list_expr}); ++{index_name}) {{"
        )
        self.indent_level += 1
        item_type = list_type[5:-1].strip()
        item_expr = f"get_{struct_name}({list_expr}, {index_name})"
        if level == len(list_types) - 1:
            self.add_line(f"{data_name}[{offset_name}++] = {item_expr};")
        else:
            child_struct = self.generate_list_struct_name(item_type)
            child_name = f"ocean_device_tensor_flatten_source_{suffix}_{level}"
            self.add_line(f"{child_struct}* {child_name} = {item_expr};")
            self.add_line(
                f"ocean_tensor_validate_list_length("
                f"builtin_len_{child_struct}({child_name}), {shape_name}[{level + 1}]);"
            )
            self._emit_tensor_list_flatten(
                child_name, list_types, level + 1, data_name,
                offset_name, shape_name, suffix
            )
        self.indent_level -= 1
        self.add_line("}")

    def _flatten_tensor_items(self, ast: Dict) -> List[Dict]:
        if ast.get("type") != "list_literal":
            return [ast]
        result: List[Dict] = []
        for item in ast.get("items", []):
            result.extend(self._flatten_tensor_items(item))
        return result

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

    def generate_tensor_index_access(self, ast: Dict) -> str:
        variable = ast.get("variable", "")
        info = self.get_variable_info(variable)
        if not info:
            raise RuntimeError(f"unknown Tensor '{variable}'")
        self.assert_can_read(variable)
        py_type = self.strip_borrow_type(info.get("py_type", ""))
        if not self.is_device_tensor_type(py_type):
            raise RuntimeError("indexed access is supported only for Tensor[T]")
        indices = [self.generate_expression(index) for index in ast.get("indices", [])]
        if not indices:
            raise RuntimeError("Tensor indexing expects at least one index")
        literal = ", ".join(f"(size_t)({index})" for index in indices)
        return (
            f"ocean_tensor_get_nd({variable}->handle, "
            f"(const size_t[]){{{literal}}}, {len(indices)})"
        )

    def generate_tensor_index_assignment(
        self, variable: str, py_type: str, indices: Iterable[Dict], value: str
    ) -> None:
        if not self.is_device_tensor_type(py_type):
            raise RuntimeError("indexed assignment is supported only for Tensor[T]")
        expressions = [self.generate_expression(index) for index in indices]
        if not expressions:
            raise RuntimeError("Tensor indexing expects at least one index")
        literal = ", ".join(f"(size_t)({index})" for index in expressions)
        self.add_line(
            f"ocean_tensor_set_nd({variable}->handle, (const size_t[]){{{literal}}}, "
            f"{len(expressions)}, {value});"
        )
