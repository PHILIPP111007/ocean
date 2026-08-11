from __future__ import annotations

from typing import Dict, Optional

from src.modules.logger import logger


class ScopeMixin:
    def enter_scope(self, kind: str = "block"):
        """Enter a lexical ownership scope."""
        self.current_scope_level += 1
        if len(self.variable_scopes) <= self.current_scope_level:
            self.variable_scopes.append({})
        else:
            self.variable_scopes[self.current_scope_level] = {}
        self.variable_scopes[self.current_scope_level]["__scope_kind__"] = kind

    def exit_scope(self, emit_cleanup: bool = True):
        """Leave a lexical scope and deterministically release owned values."""
        if self.current_scope_level <= 0:
            return
        scope = self.variable_scopes[self.current_scope_level]
        if emit_cleanup:
            self.emit_scope_cleanup(scope)
        # Borrow state is compile-time state and ends lexically here.
        for name, info in list(scope.items()):
            if name.startswith("__") or not isinstance(info, dict):
                continue
            if info.get("memory_kind") in {self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
                self._unregister_borrow_info(info)
        self.variable_scopes.pop()
        self.current_scope_level -= 1

    def get_current_scope(self) -> Dict:
        if self.current_scope_level < len(self.variable_scopes):
            return self.variable_scopes[self.current_scope_level]
        return {}

    def generate_function_scope(self, scope: Dict):
        """Generate a function with borrowed parameters and automatic cleanup."""
        func_name = scope.get("function_name", "")
        return_type = scope.get("return_type", "int")
        parameters = scope.get("parameters", [])

        logger.debug(f"generate_function_scope: {func_name}() -> {return_type}")
        self.current_function_return_type = return_type
        self.current_function_name = func_name
        self.enter_scope("function")

        param_decls = []
        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "int")
            c_param_type = self.map_type_to_c(param_type)
            param_decls.append(f"{c_param_type} {param_name}")
            self.declare_variable(param_name, param_type, is_parameter=True)

        c_return_type = self.map_type_to_c(return_type)
        params_str = ", ".join(param_decls) if param_decls else "void"
        self.add_line(f"{c_return_type} {func_name}({params_str}) {{")
        self.indent_level += 1
        self.tensor_fast_access = {}
        self.tensor_fast_loop_bounds = {}
        self.tensor_fast_patterns = set()
        self._prepare_tensor_fast_path(scope)

        processed_declarations = set()
        for node in scope.get("graph", []):
            node_type = node.get("node")
            if node_type == "declaration":
                var_name = node.get("var_name", "")
                if var_name not in processed_declarations:
                    self.generate_graph_node(node)
                    processed_declarations.add(var_name)
            else:
                self.generate_graph_node(node)

        # Fall-through cleanup. Return paths emit their own cleanup before return.
        self.exit_scope(emit_cleanup=True)
        self.indent_level -= 1
        self.add_line("}")
        self.add_empty_line()
        self.tensor_fast_access = {}
        self.tensor_fast_loop_bounds = {}
        self.tensor_fast_patterns = set()
        self.current_function_return_type = None
        self.current_function_name = None

    def declare_variable(
        self,
        name: str,
        var_type: str,
        is_pointer: bool = False,
        *,
        is_parameter: bool = False,
        owns_reference: Optional[bool] = None,
    ):
        scope = self.get_current_scope()
        c_type = self.map_type_to_c(var_type, is_pointer)
        memory_kind = self.memory_kind_for_type(var_type)
        if owns_reference is None:
            owns_reference = (
                not is_parameter
                and memory_kind in {self.MEMORY_ARC, self.MEMORY_STRING, self.MEMORY_OWNED}
            )

        scope[name] = {
            "c_type": c_type,
            "py_type": var_type,
            "is_pointer": is_pointer,
            "is_deleted": False,
            "is_moved": False,
            "delete_type": None,
            "memory_kind": memory_kind,
            "owns_reference": owns_reference,
            "is_parameter": is_parameter,
            "shared_borrows": 0,
            "mutable_borrow": False,
            "borrow_source": None,
            "borrow_mutable": False,
        }
        logger.debug(
            f"variable {name}: {var_type} -> {c_type} "
            f"({memory_kind}, owner={owns_reference})"
        )

    def mark_variable_deleted(self, name: str, delete_type: str = "full") -> bool:
        for level in range(self.current_scope_level, -1, -1):
            if level < len(self.variable_scopes):
                scope = self.variable_scopes[level]
                if name in scope:
                    scope[name]["is_deleted"] = True
                    scope[name]["delete_type"] = delete_type
                    return True
        logger.warning(f"Variable '{name}' not found for deletion")
        return False

    def is_variable_declared(self, name: str) -> bool:
        for level in range(self.current_scope_level, -1, -1):
            if level < len(self.variable_scopes) and name in self.variable_scopes[level]:
                info = self.variable_scopes[level][name]
                if isinstance(info, dict) and not info.get("is_deleted", False) and not info.get("is_moved", False):
                    return True
        return False

    def get_variable_info(self, name: str) -> Optional[Dict]:
        for level in range(self.current_scope_level, -1, -1):
            if level < len(self.variable_scopes) and name in self.variable_scopes[level]:
                info = self.variable_scopes[level][name]
                return info if isinstance(info, dict) else None
        return None
