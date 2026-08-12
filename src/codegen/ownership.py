from __future__ import annotations

from typing import Dict, Iterable, Optional

from src.modules.logger import logger


class OwnershipError(RuntimeError):
    """Raised when Phils ownership/borrow rules are violated during lowering."""


class OwnershipMixin:
    """Hybrid automatic ownership management for the C backend.

    Memory model implemented by this first checker iteration:

    * value types: copied directly, no runtime ownership metadata;
    * ARC types: list/dict/tuple/class instances are reference counted;
    * strings: uniquely-owned C strings (copied on aliasing);
    * ``&T``: immutable lexical borrow, no retain/release;
    * ``&mut T``: exclusive lexical mutable borrow, no retain/release;
    * raw C pointers: outside automatic ownership and therefore unsafe.

    The backend's runtime lowering remains lexical in v1.  The JSON validator
    performs the conservative intra-function data-flow checks before lowering;
    both layers share the same ownership metadata and generated ABI.
    """

    MEMORY_VALUE = "value"
    MEMORY_ARC = "arc"
    MEMORY_STRING = "string"
    MEMORY_BORROW = "borrow"
    MEMORY_MUT_BORROW = "mut_borrow"
    MEMORY_RAW = "raw"
    MEMORY_OWNED = "owned"

    def generate_ownership_runtime(self) -> None:
        """Register the common ``ocean_`` ARC runtime in generated helpers."""
        marker = "ocean_object_header"
        if marker in self.generated_structures:
            return
        self.generated_structures.add(marker)

        self.generated_helpers.append(
            """
typedef struct ocean_object_header {
    size_t refcount;
    void (*destroy)(void*);
} ocean_object_header;
"""
        )
        self.generated_helpers.append(
            """
static inline void ocean_retain(void* ptr) {
    if (!ptr) return;
    ocean_object_header* header = (ocean_object_header*)ptr;
    if (header->refcount == (size_t)-1) {
        fprintf(stderr, "Ocean ownership error: reference count overflow\\n");
        abort();
    }
    header->refcount += 1;
}

static inline void ocean_release(void* ptr) {
    if (!ptr) return;
    ocean_object_header* header = (ocean_object_header*)ptr;
    if (header->refcount == 0) {
        fprintf(stderr, "Ocean ownership error: release of dead object\\n");
        abort();
    }
    header->refcount -= 1;
    if (header->refcount == 0) {
        void (*destroy)(void*) = header->destroy;
        if (destroy) destroy(ptr);
    }
}

static char* ocean_strdup(const char* src) {
    if (!src) return NULL;
    size_t n = strlen(src) + 1;
    char* dst = (char*)malloc(n);
    if (!dst) {
        fprintf(stderr, "Ocean allocation error: string copy\\n");
        exit(1);
    }
    memcpy(dst, src, n);
    return dst;
}
"""
        )

    # ------------------------------------------------------------------
    # Type categories
    # ------------------------------------------------------------------

    def strip_borrow_type(self, py_type: str) -> str:
        value = (py_type or "").strip()
        if value.startswith("&mut "):
            return value[5:].strip()
        if value.startswith("&"):
            return value[1:].strip()
        return value

    def is_borrow_type(self, py_type: str) -> bool:
        return (py_type or "").strip().startswith("&")

    def is_mut_borrow_type(self, py_type: str) -> bool:
        return (py_type or "").strip().startswith("&mut ")

    def is_string_type(self, py_type: str) -> bool:
        return self.strip_borrow_type(py_type) == "str"

    def is_arc_type(self, py_type: str) -> bool:
        base = self.strip_borrow_type(py_type)
        if not base:
            return False
        if self.is_device_tensor_type(base):
            return True
        if base.startswith(("list[", "dict[", "tuple[")):
            return True
        return self._is_class_type(base)

    def is_owned_type(self, py_type: str) -> bool:
        base = self.strip_borrow_type(py_type)
        return self.is_owned_buffer_type(base)

    def is_raw_pointer_type(self, py_type: str) -> bool:
        value = (py_type or "").strip()
        if value in {"pointer", "void*", "null"}:
            return True
        if value.startswith("*"):
            return True
        return "*" in value and self._is_c_type(value)

    def memory_kind_for_type(self, py_type: str) -> str:
        if self.is_mut_borrow_type(py_type):
            return self.MEMORY_MUT_BORROW
        if self.is_borrow_type(py_type):
            return self.MEMORY_BORROW
        if self.is_string_type(py_type):
            return self.MEMORY_STRING
        if self.is_owned_type(py_type):
            return self.MEMORY_OWNED
        if self.is_arc_type(py_type):
            return self.MEMORY_ARC
        if self.is_raw_pointer_type(py_type):
            return self.MEMORY_RAW
        return self.MEMORY_VALUE

    def type_needs_runtime_cleanup(self, py_type: str) -> bool:
        return self.memory_kind_for_type(py_type) in {
            self.MEMORY_ARC,
            self.MEMORY_STRING,
            self.MEMORY_OWNED,
        }

    # ------------------------------------------------------------------
    # Expression ownership
    # ------------------------------------------------------------------

    def expression_ownership(self, ast: Optional[Dict], target_type: str = "") -> str:
        """Return ``borrowed``, ``owned`` or ``value`` for an expression.

        Index/attribute reads and variable aliases are borrowed views.  Calls
        and constructors returning a managed type obey the Phils ABI and return
        one owned reference.
        """
        if not ast:
            return "value"
        if self.memory_kind_for_type(target_type) not in {
            self.MEMORY_ARC,
            self.MEMORY_STRING,
        }:
            return "value"

        kind = ast.get("type", "")
        if kind in {
            "variable",
            "attribute_access",
            "index_access",
            "nested_index_access",
            "complex_attribute_access",
        }:
            return "borrowed"
        if kind == "literal" and ast.get("data_type") == "str":
            return "borrowed"
        if kind == "method_call" and ast.get("method") == "get":
            return "borrowed"
        return "owned"

    def source_variable_from_expression(self, ast: Optional[Dict]) -> Optional[str]:
        if not ast:
            return None
        if ast.get("type") == "variable":
            return ast.get("value") or ast.get("name")
        return None

    def consume_owned_expression(self, expr: str, ownership: str) -> None:
        """Transfer a compiler-created temporary owner into its destination."""
        if ownership != "owned" or not expr or not expr.replace("_", "a").isalnum():
            return
        info = self.get_variable_info(expr)
        if not info or not info.get("owns_reference") or info.get("is_parameter"):
            return
        if info.get("memory_kind") not in {self.MEMORY_ARC, self.MEMORY_STRING}:
            return
        info["owns_reference"] = False
        info["is_moved"] = True

    # ------------------------------------------------------------------
    # Borrow checker
    # ------------------------------------------------------------------

    def register_borrow(self, borrow_name: str, source_name: str, mutable: bool) -> None:
        source = self.get_variable_info(source_name)
        borrow = self.get_variable_info(borrow_name)
        if source is None:
            raise OwnershipError(
                f"cannot borrow unknown variable '{source_name}' for '{borrow_name}'"
            )
        if source.get("is_deleted") or source.get("is_moved"):
            raise OwnershipError(f"cannot borrow dead value '{source_name}'")

        shared = int(source.get("shared_borrows", 0))
        mut_active = bool(source.get("mutable_borrow", False))

        if mutable:
            if shared or mut_active:
                raise OwnershipError(
                    f"cannot mutably borrow '{source_name}': another borrow is active"
                )
            source["mutable_borrow"] = True
        else:
            if mut_active:
                raise OwnershipError(
                    f"cannot immutably borrow '{source_name}': mutable borrow is active"
                )
            source["shared_borrows"] = shared + 1

        if borrow is not None:
            borrow["borrow_source"] = source_name
            borrow["borrow_mutable"] = mutable
        logger.debug(
            f"borrow registered: {borrow_name} -> {source_name} "
            f"(mutable={mutable})"
        )

    def _unregister_borrow_info(self, info: Dict) -> None:
        source_name = info.get("borrow_source")
        if not source_name:
            return
        source = self.get_variable_info(source_name)
        if source is None:
            return
        if info.get("borrow_mutable"):
            source["mutable_borrow"] = False
        else:
            source["shared_borrows"] = max(0, int(source.get("shared_borrows", 0)) - 1)

    def assert_live(self, name: str) -> None:
        info = self.get_variable_info(name)
        if info is None:
            if name in getattr(self, "phils_function_names", set()):
                return
            raise OwnershipError(f"use of undeclared value '{name}'")
        if info.get("is_deleted"):
            raise OwnershipError(f"use of deleted value '{name}'")
        if info.get("is_moved"):
            raise OwnershipError(f"use of moved value '{name}'")

    def assert_can_read(self, name: str) -> None:
        """Reject direct owner access while an exclusive borrow is active."""
        self.assert_live(name)
        info = self.get_variable_info(name)
        if not info:
            return
        kind = info.get("memory_kind")
        if kind in {self.MEMORY_BORROW, self.MEMORY_MUT_BORROW}:
            return
        if info.get("mutable_borrow"):
            raise OwnershipError(
                f"cannot access owner '{name}' while a mutable borrow is active"
            )

    def assert_can_mutate(self, name: str) -> None:
        info = self.get_variable_info(name)
        if not info:
            return
        kind = info.get("memory_kind")
        if kind == self.MEMORY_BORROW:
            raise OwnershipError(f"cannot mutate through immutable borrow '{name}'")
        if kind == self.MEMORY_MUT_BORROW:
            return
        if int(info.get("shared_borrows", 0)) > 0:
            raise OwnershipError(
                f"cannot mutate '{name}' while immutable borrow(s) are active"
            )
        if info.get("mutable_borrow"):
            raise OwnershipError(
                f"cannot mutate owner '{name}' while a mutable borrow is active"
            )

    def assert_can_move_or_delete(self, name: str) -> None:
        info = self.get_variable_info(name)
        if not info:
            return
        if int(info.get("shared_borrows", 0)) > 0 or info.get("mutable_borrow"):
            raise OwnershipError(
                f"cannot move/delete '{name}' while it is borrowed"
            )

    # ------------------------------------------------------------------
    # Cleanup / transfer
    # ------------------------------------------------------------------

    def emit_variable_cleanup(self, name: str, info: Dict, *, mark_state: bool = True) -> None:
        if not isinstance(info, dict) or "py_type" not in info:
            return
        if info.get("is_deleted") or info.get("is_moved"):
            return
        if not info.get("owns_reference", False):
            return

        kind = info.get("memory_kind")
        if kind == self.MEMORY_ARC:
            self.add_line(f"ocean_release({name});")
            self.add_line(f"{name} = NULL;")
            if mark_state:
                info["is_deleted"] = True
        elif kind == self.MEMORY_STRING:
            self.add_line(f"free({name});")
            self.add_line(f"{name} = NULL;")
            if mark_state:
                info["is_deleted"] = True
        elif kind == self.MEMORY_OWNED:
            self.add_line(self._owned_free_call(name, info["py_type"]))
            self.add_line(f"{name} = NULL;")
            if mark_state:
                info["is_deleted"] = True

    def emit_scope_cleanup(
        self, scope: Dict, exclude: Iterable[str] = (), *, mark_state: bool = True
    ) -> None:
        excluded = set(exclude)
        # Reverse declaration order approximates deterministic destruction order.
        for name, info in reversed(list(scope.items())):
            if name.startswith("__") or name in excluded:
                continue
            self.emit_variable_cleanup(name, info, mark_state=mark_state)

    def emit_all_scope_cleanup(self, exclude: Iterable[str] = ()) -> None:
        excluded = set(exclude)
        for level in range(self.current_scope_level, 0, -1):
            if level < len(self.variable_scopes):
                self.emit_scope_cleanup(
                    self.variable_scopes[level], excluded, mark_state=False
                )

    def consume_owned_call_arguments(self, function_name: str, arguments: Iterable[Dict]) -> None:
        """Transfer unique buffers passed to by-value function parameters."""
        parameters = getattr(self, "function_parameters", {}).get(function_name, [])
        for index, argument in enumerate(arguments):
            if index >= len(parameters) or not isinstance(argument, dict):
                continue
            expected = parameters[index].get("type", "")
            if self.is_borrow_type(expected) or not self.is_owned_type(expected):
                continue
            if argument.get("type") != "variable":
                continue
            source = argument.get("value") or argument.get("name")
            info = self.get_variable_info(source)
            if not info or not self.is_owned_type(info.get("py_type", "")):
                continue
            self.assert_can_read(source)
            self.assert_can_move_or_delete(source)
            self._mark_owned_move(source)

    def emit_cleanup_to_loop(self) -> None:
        for level in range(self.current_scope_level, 0, -1):
            scope = self.variable_scopes[level]
            self.emit_scope_cleanup(scope, mark_state=False)
            if scope.get("__scope_kind__") == "loop":
                return
        raise OwnershipError("break/continue used outside a loop")

    def mark_moved(self, name: str) -> None:
        info = self.get_variable_info(name)
        if info:
            self.assert_can_move_or_delete(name)
            info["is_moved"] = True
            info["owns_reference"] = False
