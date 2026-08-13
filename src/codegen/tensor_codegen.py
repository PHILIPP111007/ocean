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
            if len(args) != 2 or args[0].get("type") != "list_literal":
                raise RuntimeError(
                    "Tensor[T].from_list expects a rectangular list literal and device"
                )
            shape = self._infer_tensor_shape(args[0])
            if shape is None:
                raise RuntimeError(
                    "Tensor[T].from_list expects a rectangular list literal"
                )
            flat_items = self._flatten_tensor_items(args[0])
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

        return None

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
