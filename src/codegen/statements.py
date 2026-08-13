from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.modules.constants import DEFAULT_C_IMPORTS, INITIAL_LIST_CAPACITY, KNOWN_C_TYPES
from src.modules.logger import logger

class StatementsMixin:
    @staticmethod
    def _normalize_loop_expression(value) -> str:
        """Lower attribute references in range bounds to their C form."""
        text = str(value)
        return re.sub(
            r"\bself((?:\.[A-Za-z_][A-Za-z0-9_]*)+)",
            lambda match: "self" + match.group(1).replace(".", "->"),
            text,
        )

    def generate_break(self, node: Dict):
        """Release loop-local owners before transferring control."""
        self.emit_cleanup_to_loop()
        self.add_line("break;")

    def generate_continue(self, node: Dict):
        """Release current iteration owners before continuing."""
        self.emit_cleanup_to_loop()
        self.add_line("continue;")

    def generate_return(self, node: Dict):
        """Ownership-aware return: evaluate, establish return ownership, cleanup."""
        operations = node.get("operations", [])
        value_ast = None
        for op in operations:
            if op.get("type") == "RETURN":
                value_ast = op.get("value", {})
                break

        return_type = self.current_function_return_type or "None"
        if not value_ast:
            self.emit_all_scope_cleanup()
            self.add_line("return NULL;" if return_type == "None" else "return;")
            return
        if return_type == "None":
            self.emit_all_scope_cleanup()
            self.add_line("return NULL;")
            return

        if self.is_device_tensor_type(return_type):
            expr = self._generate_device_tensor_expression(value_ast, return_type)
        else:
            expr = self.generate_expression(value_ast)
        c_type = self.map_type_to_c(return_type)
        temp = f"ocean_return_{self.temp_var_counter}"
        self.temp_var_counter += 1
        ownership = self.expression_ownership(value_ast, return_type)
        kind = self.memory_kind_for_type(return_type)
        source = self.source_variable_from_expression(value_ast)
        source_info = self.get_variable_info(source) if source else None
        transfer_local_owner = bool(
            source_info
            and source_info.get("owns_reference")
            and kind in {self.MEMORY_ARC, self.MEMORY_STRING, self.MEMORY_OWNED}
        )

        if kind == self.MEMORY_OWNED:
            self.add_line(f"{c_type} {temp} = {expr};")
            if transfer_local_owner and source:
                self._mark_owned_move(source)
        elif kind == self.MEMORY_STRING and ownership == "borrowed" and not transfer_local_owner:
            self.add_line(f"{c_type} {temp} = ocean_strdup({expr});")
        else:
            self.add_line(f"{c_type} {temp} = {expr};")
            if kind == self.MEMORY_ARC and ownership == "borrowed" and not transfer_local_owner:
                self.add_line(f"ocean_retain({temp});")
            elif ownership == "owned":
                self.consume_owned_expression(expr, ownership)

        exclude = {source} if transfer_local_owner and source else set()
        self.emit_all_scope_cleanup(exclude=exclude)
        self.add_line(f"return {temp};")

    def generate_while_loop(self, node: Dict):
        """Генерирует while loop с правильной обработкой структуры AST."""
        # В AST ключ "condition", а не "condition_ast"
        condition_ast = node.get("condition")
        if not condition_ast:
            return

        condition = self.generate_expression(condition_ast)

        self.add_line(f"while ({condition}) {{")
        self.indent_level += 1

        # Входим в scope цикла
        self.enter_scope("loop")

        # Генерируем тело цикла из списка body
        body_nodes = node.get("body", [])
        for body_node in body_nodes:
            self.generate_graph_node(body_node)

        # Выходим из scope цикла
        self.exit_scope()

        self.indent_level -= 1
        self.add_line("}")

    def generate_if_statement(self, node: Dict):
        """Генерирует if statement"""
        condition_ast = node.get("condition_ast")
        if not condition_ast:
            return

        condition = self.generate_expression(condition_ast)

        self.add_line(f"if ({condition}) {{")
        self.indent_level += 1

        # Входим в scope if
        self.enter_scope("block")

        # Генерируем тело if
        for body_node in node.get("body", []):
            self.generate_graph_node(body_node)

        # Выходим из scope if
        self.exit_scope()

        self.indent_level -= 1
        self.add_line("}")

        # elif блоки
        for elif_block in node.get("elif_blocks", []):
            elif_condition = self.generate_expression(
                elif_block.get("condition_ast", {})
            )
            self.add_line(f"else if ({elif_condition}) {{")
            self.indent_level += 1

            # Входим в scope elif
            self.enter_scope("block")

            # Генерируем тело elif
            for body_node in elif_block.get("body", []):
                self.generate_graph_node(body_node)

            # Выходим из scope elif
            self.exit_scope()

            self.indent_level -= 1
            self.add_line("}")

        # else блок
        else_block = node.get("else_block")
        if else_block:
            self.add_line("else {")
            self.indent_level += 1

            # Входим в scope else
            self.enter_scope("block")

            # Генерируем тело else
            for body_node in else_block.get("body", []):
                self.generate_graph_node(body_node)

            # Выходим из scope else
            self.exit_scope()

            self.indent_level -= 1
            self.add_line("}")

    def generate_for_loop(self, node: Dict):
        """Generate Python-compatible range direction and a per-iteration scope."""
        loop_var = node.get("loop_variable", "i")
        iterable = node.get("iterable", {})
        if iterable.get("type") != "RANGE_CALL":
            return
        args = iterable.get("arguments", {})
        start = self._normalize_loop_expression(args.get("start", "0"))
        stop = self._normalize_loop_expression(args.get("stop", "10"))
        step = self._normalize_loop_expression(args.get("step", "1"))
        if str(step).strip() in {"0", "+0", "-0"}:
            raise RuntimeError("range() step cannot be zero")

        openmp = node.get("openmp")
        if openmp:
            if openmp.get("error"):
                raise RuntimeError(openmp["error"])
            if openmp.get("backend") != "openmp" or openmp.get("directive") != "parallel for":
                raise RuntimeError("unsupported OpenMP loop directive")
            self.add_line(self._format_openmp_pragma(openmp))

        collapse_count = self._openmp_collapse_count(openmp) if openmp else 1
        canonical_openmp_loop = bool(openmp) or self.openmp_collapse_remaining > 0

        # loop variable belongs to the loop's C declaration, not an owning object.
        if canonical_openmp_loop:
            try:
                numeric_step = int(str(step).strip())
            except ValueError as error:
                raise RuntimeError(
                    "OpenMP parallel for requires a constant integer range step"
                ) from error
            comparison = "<" if numeric_step > 0 else ">"
            loop_header = (
                f"for (int {loop_var} = {start}; {loop_var} {comparison} {stop}; "
                f"{loop_var} += {step}) {{"
            )
        else:
            loop_header = (
                f"for (int {loop_var} = {start}; "
                f"(({step}) > 0 ? {loop_var} < {stop} : {loop_var} > {stop}); "
                f"{loop_var} += {step}) {{"
            )
        self.add_line(loop_header)
        self.indent_level += 1
        self.enter_scope("loop")
        self.declare_variable(loop_var, "int")
        previous_collapse_remaining = self.openmp_collapse_remaining
        if openmp:
            self.openmp_collapse_remaining = collapse_count - 1
        elif self.openmp_collapse_remaining > 0:
            self.openmp_collapse_remaining -= 1
        previous_bounds = self.tensor_fast_loop_bounds
        self.tensor_fast_loop_bounds = dict(previous_bounds)
        self.tensor_fast_loop_bounds[loop_var] = {
            "start": str(start).strip(),
            "stop": str(stop).strip(),
            "step": str(step).strip(),
        }
        try:
            for body_node in node.get("body", []):
                self.generate_graph_node(body_node)
        finally:
            self.tensor_fast_loop_bounds = previous_bounds
            self.openmp_collapse_remaining = previous_collapse_remaining
        self.exit_scope()
        self.indent_level -= 1
        self.add_line("}")

    def _format_openmp_pragma(self, metadata: Dict) -> str:
        """Render validated structured OpenMP metadata as one C pragma."""
        allowed = {
            "schedule",
            "collapse",
            "reduction",
            "private",
            "firstprivate",
            "lastprivate",
            "shared",
            "default",
            "nowait",
            "ordered",
        }
        rendered = []
        for clause in metadata.get("clauses", []) or []:
            name = str(clause.get("name", "")).strip()
            arguments = str(clause.get("arguments", "")).strip()
            if name not in allowed:
                raise RuntimeError(f"unsupported OpenMP clause: {name!r}")
            if name in {"nowait", "ordered"}:
                if arguments:
                    raise RuntimeError(f"OpenMP clause '{name}' does not take arguments")
                rendered.append(name)
                continue
            if not arguments or not re.match(r"^[A-Za-z0-9_+%*/:.,\-\s]+$", arguments):
                raise RuntimeError(f"invalid arguments for OpenMP clause '{name}'")
            rendered.append(f"{name}({arguments})")
        suffix = " " + " ".join(rendered) if rendered else ""
        return f"#pragma omp parallel for{suffix}"

    def _openmp_collapse_count(self, metadata: Dict | None) -> int:
        """Return the validated collapse count for an OpenMP directive."""
        if not metadata:
            return 1
        values = [
            str(clause.get("arguments", "")).strip()
            for clause in metadata.get("clauses", []) or []
            if clause.get("name") == "collapse"
        ]
        if not values:
            return 1
        if len(values) != 1 or not re.match(r"^[1-9][0-9]*$", values[0]):
            raise RuntimeError("OpenMP collapse requires one positive integer")
        return int(values[0])

    def generate_attribute_assignment(self, node: Dict):
        """Ownership-safe class field assignment."""
        object_name = node.get("object", "")
        attribute = node.get("attribute", "")
        value_ast = node.get("value", {}) or {}

        if object_name == "self":
            class_name = self._get_current_class()
            object_expr = "self"
        else:
            info = self.get_variable_info(object_name)
            class_name = self.strip_borrow_type(info.get("py_type", "")) if info else None
            _, object_expr = self.resolve_object_path(object_name)
            if info:
                self.assert_can_mutate(object_name)

        field_type, field_lvalue = self.resolve_class_field(
            class_name, object_expr, attribute
        ) if class_name else (None, None)
        field_lvalue = field_lvalue or f"{object_expr}->{attribute}"

        if field_type and self.is_device_tensor_type(field_type):
            value_expr = self._generate_device_tensor_expression(value_ast, field_type)
        else:
            value_expr = self.generate_expression(value_ast)

        if not field_type:
            self.add_line(f"{field_lvalue} = {value_expr};")
            return

        memory_kind = self.memory_kind_for_type(field_type)
        ownership = self.expression_ownership(value_ast, field_type)
        c_type = self.map_type_to_c(field_type)

        if memory_kind == self.MEMORY_ARC:
            temp = f"ocean_field_tmp_{self.temp_var_counter}"
            self.temp_var_counter += 1
            self.add_line(f"{c_type} {temp} = {value_expr};")
            if ownership == "borrowed":
                self.add_line(f"ocean_retain({temp});")
            self.add_line(f"ocean_release({field_lvalue});")
            self.add_line(f"{field_lvalue} = {temp};")
            self.consume_owned_expression(value_expr, ownership)
        elif memory_kind == self.MEMORY_STRING:
            temp = f"ocean_field_string_{self.temp_var_counter}"
            self.temp_var_counter += 1
            if ownership == "borrowed":
                self.add_line(f"char* {temp} = ocean_strdup({value_expr});")
            else:
                self.add_line(f"char* {temp} = {value_expr};")
            self.add_line(f"free({field_lvalue});")
            self.add_line(f"{field_lvalue} = {temp};")
            self.consume_owned_expression(value_expr, ownership)
        else:
            self.add_line(f"{field_lvalue} = {value_expr};")

    def generate_assignment(self, node: Dict):
        """Generate assignment with ARC/string ownership and borrow checks."""
        symbols = node.get("symbols", [])
        if not symbols:
            return
        target = symbols[0]
        expression_ast = node.get("expression_ast") or {}
        if not expression_ast:
            return

        info = self.get_variable_info(target)
        if info is None:
            expr = self.generate_expression(expression_ast)
            self.add_line(f"{target} = {expr};")
            return

        self.assert_can_move_or_delete(target)
        kind = info.get("memory_kind")
        if kind == self.MEMORY_OWNED:
            self.generate_array_assignment(node)
            return
        if kind in {self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
            raise RuntimeError(
                f"borrow variable '{target}' cannot be rebound in borrow-checker v1"
            )

        # String concatenation must evaluate into a fresh buffer before freeing target.
        if (
            info.get("py_type") == "str"
            and expression_ast.get("type") == "binary_operation"
            and expression_ast.get("operator_symbol") == "+"
        ):
            left = self.generate_expression(expression_ast.get("left", {}))
            right = self.generate_expression(expression_ast.get("right", {}))
            temp = f"ocean_string_tmp_{self.temp_var_counter}"
            self.temp_var_counter += 1
            self.add_line(
                f"char* {temp} = malloc(strlen({left}) + strlen({right}) + 1);"
            )
            self.add_line(f"if (!{temp}) {{ fprintf(stderr, \"Ocean allocation error\\n\"); exit(1); }}")
            self.add_line(f"strcpy({temp}, {left});")
            self.add_line(f"strcat({temp}, {right});")
            self.add_line(f"free({target});")
            self.add_line(f"{target} = {temp};")
            return

        if self.is_device_tensor_type(info.get("py_type", "")):
            expr = self._generate_device_tensor_expression(
                expression_ast, info.get("py_type", "")
            )
        else:
            expr = self.generate_expression(expression_ast)
        if self._is_none_expression(expression_ast) and kind == self.MEMORY_VALUE:
            expr = f"({info['c_type']}){{0}}"
        ownership = self.expression_ownership(expression_ast, info.get("py_type", ""))

        if kind == self.MEMORY_ARC:
            temp = f"ocean_ref_tmp_{self.temp_var_counter}"
            self.temp_var_counter += 1
            self.add_line(f"{info['c_type']} {temp} = {expr};")
            if ownership == "borrowed":
                self.add_line(f"ocean_retain({temp});")
            self.add_line(f"ocean_release({target});")
            self.add_line(f"{target} = {temp};")
            self.consume_owned_expression(expr, ownership)
            info["is_deleted"] = False
            info["is_moved"] = False
            info["owns_reference"] = True
            return

        if kind == self.MEMORY_STRING:
            temp = f"ocean_string_tmp_{self.temp_var_counter}"
            self.temp_var_counter += 1
            if ownership == "borrowed":
                self.add_line(f"char* {temp} = ocean_strdup({expr});")
            else:
                self.add_line(f"char* {temp} = {expr};")
            self.add_line(f"free({target});")
            self.add_line(f"{target} = {temp};")
            self.consume_owned_expression(expr, ownership)
            info["is_deleted"] = False
            info["is_moved"] = False
            info["owns_reference"] = True
            return

        self.add_line(f"{target} = {expr};")

    def generate_augmented_assignment(self, node: Dict):
        """Generate a scalar compound assignment, including OpenMP reductions."""
        symbols = node.get("symbols", [])
        operations = node.get("operations", []) or []
        if not symbols or not operations:
            return
        target = symbols[0]
        operator = operations[0].get("operator_symbol", "+=")
        value_ast = node.get("value_ast") or {}
        info = self.get_variable_info(target)
        if info:
            self.assert_can_mutate(target)
            if self.memory_kind_for_type(info.get("py_type", "")) != self.MEMORY_VALUE:
                raise RuntimeError(
                    f"compound assignment is only supported for scalar values, got '{target}'"
                )
        value_expr = self.generate_expression(value_ast)
        self.add_line(f"{target} {operator} {value_expr};")

    def generate_declaration(self, node: Dict):
        """Declare a value and establish its ownership state."""
        var_name = node.get("var_name", "")
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast", {}) or {}
        logger.debug(f"declaration {var_name}: {var_type}")

        existing = self.get_variable_info(var_name)
        if existing is not None and not existing.get("is_deleted", False):
            self.generate_redeclaration(node)
            return

        # Lexical borrows are represented as the same pointer type in C, but
        # they carry no ownership and are checked statically here.
        if self.is_borrow_type(var_type):
            source = self.source_variable_from_expression(expression_ast)
            if not source:
                raise RuntimeError(
                    f"borrow '{var_name}' must currently originate from a named variable"
                )
            source_info = self.get_variable_info(source)
            if not source_info:
                raise RuntimeError(f"cannot borrow unknown variable '{source}'")
            source_type = self.strip_borrow_type(source_info.get("py_type", ""))
            target_type = self.strip_borrow_type(var_type)
            if source_type != target_type:
                raise RuntimeError(
                    f"borrow type mismatch: '{var_name}' expects {target_type}, "
                    f"but '{source}' has type {source_type}"
                )
            if source_info.get("memory_kind") in {self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
                raise RuntimeError(
                    "reborrowing a borrow is not enabled in Ocean borrow-checker v1"
                )
            if source_info.get("memory_kind") == self.MEMORY_VALUE:
                raise RuntimeError(
                    "borrowing inline/value types is not enabled in backend v1; "
                    "managed objects (&list/&dict/&class) are supported first"
                )
            if self.is_mut_borrow_type(var_type) and source_info.get("memory_kind") == self.MEMORY_STRING:
                raise RuntimeError("&mut str is not enabled because Phils strings are immutable values")
            self.declare_variable(var_name, var_type, owns_reference=False)
            info = self.get_variable_info(var_name)
            expr = self.generate_expression(expression_ast)
            self.add_line(f"{info['c_type']} {var_name} = {expr};")
            self.register_borrow(var_name, source, self.is_mut_borrow_type(var_type))
            return

        if self.is_array_type(var_type):
            self.generate_array_declaration(node)
            return
        if var_type.startswith("dict["):
            self._generate_dict_declaration(var_name, var_type, expression_ast, node)
            return

        self.declare_variable(var_name, var_type)
        info = self.get_variable_info(var_name)
        if not info:
            return
        c_type = info["c_type"]
        kind = info["memory_kind"]

        if not expression_ast:
            if c_type.endswith("*"):
                self.add_line(f"{c_type} {var_name} = NULL;")
            elif kind == self.MEMORY_STRING:
                self.add_line(f"{c_type} {var_name} = NULL;")
            else:
                self.add_line(f"{c_type} {var_name};")
            return

        # Builtins have specialized declaration lowering in the existing backend.
        if expression_ast.get("type") == "function_call":
            func_name = expression_ast.get("function", "")
            if func_name in ["str", "int", "float", "bool", "len"]:
                self._generate_builtin_declaration(var_name, c_type, expression_ast, False)
                return

        if expression_ast.get("type") == "list_literal" and var_type.startswith("list["):
            items = expression_ast.get("items", [])
            struct_name = self.generate_list_struct_name(var_type)
            self.add_line(
                f"{c_type} {var_name} = create_{struct_name}({max(len(items), INITIAL_LIST_CAPACITY)});"
            )
            for item_ast in items:
                item_expr = self.generate_expression(item_ast)
                self.add_line(f"append_{struct_name}({var_name}, {item_expr});")
            return

        if self.is_device_tensor_type(var_type):
            expr = self._generate_device_tensor_expression(expression_ast, var_type)
        else:
            expr = self.generate_expression(expression_ast)
        if self._is_none_expression(expression_ast) and kind == self.MEMORY_VALUE:
            expr = f"({c_type}){{0}}"
        ownership = self.expression_ownership(expression_ast, var_type)
        if kind == self.MEMORY_ARC:
            self.add_line(f"{c_type} {var_name} = {expr};")
            if ownership == "borrowed":
                self.add_line(f"ocean_retain({var_name});")
            else:
                self.consume_owned_expression(expr, ownership)
        elif kind == self.MEMORY_STRING:
            if ownership == "borrowed":
                self.add_line(f"{c_type} {var_name} = ocean_strdup({expr});")
            else:
                self.add_line(f"{c_type} {var_name} = {expr};")
                self.consume_owned_expression(expr, ownership)
        else:
            self.add_line(f"{c_type} {var_name} = {expr};")

    def generate_redeclaration(self, node: Dict):
        """Redeclaration is a destruction boundary followed by a new owner."""
        var_name = node.get("var_name", "")
        old = self.get_variable_info(var_name)
        if old:
            self.assert_can_move_or_delete(var_name)
            # Evaluate through normal assignment when the type is unchanged.
            if old.get("py_type") == node.get("var_type", ""):
                assignment = {
                    "symbols": [var_name],
                    "expression_ast": node.get("expression_ast", {}),
                }
                self.generate_assignment(assignment)
                return
            self.emit_variable_cleanup(var_name, old)

        # Remove the old binding in the current lexical scope if present.
        current = self.get_current_scope()
        current.pop(var_name, None)
        fresh = dict(node)
        fresh["node"] = "declaration"
        self.generate_declaration(fresh)

    def generate_delete(self, node: Dict):
        """Early deterministic release. Raw pointers are never implicitly freed."""
        for target in node.get("symbols", []):
            info = self.get_variable_info(target)
            if not info:
                raise RuntimeError(f"unknown variable '{target}' in del")
            self.assert_can_move_or_delete(target)
            self.add_line(f"// del {target}")
            kind = info.get("memory_kind")

            if kind == self.MEMORY_ARC:
                if info.get("owns_reference"):
                    self.add_line(f"ocean_release({target});")
                self.add_line(f"{target} = NULL;")
            elif kind == self.MEMORY_STRING:
                if info.get("owns_reference"):
                    self.add_line(f"free({target});")
                self.add_line(f"{target} = NULL;")
            elif kind == self.MEMORY_OWNED:
                if info.get("owns_reference"):
                    self.add_line(self._owned_free_call(target, info["py_type"]))
                self.add_line(f"{target} = NULL;")
            elif kind in {self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
                self._unregister_borrow_info(info)
                self.add_line(f"{target} = NULL;")
            elif kind == self.MEMORY_RAW:
                # Ownership of C pointers cannot be inferred. Explicit C/unsafe
                # code must free them; del only invalidates the Phils binding.
                self.add_line(f"{target} = NULL;")
            else:
                c_type = info.get("c_type", "")
                if c_type == "bool":
                    self.add_line(f"{target} = false;")
                elif c_type in {"int", "float", "double", "long", "short"}:
                    self.add_line(f"{target} = 0;")

            info["is_deleted"] = True
            info["owns_reference"] = False
