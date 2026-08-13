"""Typed intermediate representation for the Ocean compiler.

The parser's dictionary graph remains the compatibility interchange format.  This
module adds a semantic view of that graph so later passes can consume structured
types and ownership effects without repeatedly guessing them from strings.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Iterable

from src.parsing.type_system import TypeParser, TypeSpec, infer_literal_shape


@dataclass(frozen=True)
class IRType:
    """A compiler-facing type backed by the parser's canonical ``TypeSpec``."""

    spec: TypeSpec

    @classmethod
    def parse(cls, value: str | None) -> "IRType":
        return cls(TypeParser().parse(value or "any"))

    @property
    def canonical(self) -> str:
        return self.spec.canonical

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def memory_kind(self) -> str:
        return self.spec.memory_kind

    @property
    def is_unique(self) -> bool:
        return self.spec.memory_kind == "owned"

    @property
    def is_borrow(self) -> bool:
        return self.spec.is_borrow

    @property
    def is_mut_borrow(self) -> bool:
        return self.spec.is_mut_borrow

    def __str__(self) -> str:
        return self.canonical


@dataclass(frozen=True)
class TypedNode(Mapping[str, Any]):
    """Semantic metadata and read-only mapping view for one graph node."""

    raw: dict[str, Any]
    node_type: str
    result_type: IRType
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    effect: str

    @property
    def source_line(self) -> int | None:
        return self.raw.get("source_line")

    @property
    def source_file(self) -> str | None:
        return self.raw.get("source_file")

    @property
    def source_column(self) -> int | None:
        return self.raw.get("source_column")

    @property
    def openmp(self) -> dict[str, Any] | None:
        """Structured OpenMP metadata attached to a loop, if present."""
        value = self.raw.get("openmp")
        return value if isinstance(value, dict) else None

    # Keep the backend's existing read interface while preventing mutation of
    # the parser graph through the typed pipeline.
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


@dataclass(frozen=True)
class TypedScope(Mapping[str, Any]):
    """Typed view of one parser scope."""

    raw: dict[str, Any]
    symbols: dict[str, IRType]
    nodes: tuple[TypedNode, ...]

    def __getitem__(self, key: str) -> Any:
        if key == "graph":
            return self.nodes
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "graph":
            return self.nodes
        return self.raw.get(key, default)


@dataclass(frozen=True)
class TypedModule:
    """Typed compilation unit exchanged by compiler passes."""

    scopes: tuple[TypedScope, ...]

    def backend_scopes(self) -> list[TypedScope]:
        """Return the typed, read-only lowering view consumed by the C backend."""
        return list(self.scopes)

    def scope(self, level: int) -> TypedScope | None:
        """Find a typed scope by its parser level."""
        return next(
            (scope for scope in self.scopes if scope.raw.get("level") == level),
            None,
        )

    def iter_nodes(self):
        """Iterate semantic nodes in source/scope order."""
        for scope in self.scopes:
            yield from scope.nodes

    def __len__(self) -> int:
        return len(self.scopes)

    def __iter__(self):
        return iter(self.scopes)


class TypedIRBuilder:
    """Lower parser dictionaries into typed scopes and effect-annotated nodes."""

    def __init__(self) -> None:
        self.type_parser = TypeParser()
        self._scope_by_level: dict[int, dict[str, Any]] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def build(self, scopes: list[dict[str, Any]]) -> TypedModule:
        if not isinstance(scopes, list):
            raise TypeError("typed IR expects a list of parser scopes")

        self._scope_by_level = {
            scope.get("level", 0): scope
            for scope in scopes
            if isinstance(scope, dict)
        }
        self._functions = {}
        for scope in scopes:
            if not isinstance(scope, dict):
                continue
            for name, info in (scope.get("symbol_table") or {}).items():
                if isinstance(info, dict) and info.get("key") == "function":
                    self._functions[name] = info

        typed_scopes = []
        for scope in scopes:
            if not isinstance(scope, dict):
                continue
            symbols = {
                name: IRType.parse(info.get("type", "any"))
                for name, info in (scope.get("symbol_table") or {}).items()
                if isinstance(info, dict) and info.get("key") not in {"function", "class"}
            }
            nodes = tuple(
                self._type_node(node, scope)
                for node in scope.get("graph", [])
                if isinstance(node, dict)
            )
            typed_scopes.append(TypedScope(deepcopy(scope), symbols, nodes))
        return TypedModule(tuple(typed_scopes))

    def _type_node(self, node: dict[str, Any], scope: dict[str, Any]) -> TypedNode:
        node_type = node.get("node", "unknown")
        expression = node.get("expression_ast")
        result = self._infer_expression(expression, scope)
        if node_type in {"declaration", "redeclaration"}:
            result = IRType.parse(node.get("var_type", "any"))
        elif node_type in {"assignment", "augmented_assignment"}:
            target = (node.get("symbols") or [""])[0]
            info = self._scope_symbol(target, scope)
            if info:
                result = IRType.parse(info.get("type", "any"))
        reads = tuple(sorted(self._collect_reads(node)))
        writes = tuple(self._node_writes(node))

        effect = "pure"
        if node_type in {"declaration", "redeclaration"}:
            effect = "declare"
        elif node_type in {"assignment", "augmented_assignment", "index_assignment", "nested_index_assignment", "slice_assignment", "attribute_assignment"}:
            effect = "write"
        elif node_type == "delete":
            effect = "drop"
        elif any(str(op.get("type", "")).startswith("BORROW_") for op in node.get("operations", [])):
            effect = "borrow"
        elif node_type in {"function_call", "function_call_assignment", "method_call", "c_call"}:
            effect = "call"
        elif node_type == "return":
            effect = "return"

        if node_type == "for_loop" and node.get("openmp"):
            effect = "parallel_loop"

        return TypedNode(deepcopy(node), node_type, result, reads, writes, effect)

    def _scope_symbol(self, name: str, scope: dict[str, Any]) -> dict[str, Any] | None:
        current = scope
        visited: set[int] = set()
        while isinstance(current, dict) and id(current) not in visited:
            visited.add(id(current))
            symbols = current.get("symbol_table") or {}
            info = symbols.get(name)
            if isinstance(info, dict):
                return info
            parent_level = current.get("parent_scope")
            current = self._scope_by_level.get(parent_level) if parent_level is not None else None
        return None

    def _infer_expression(self, ast: Any, scope: dict[str, Any]) -> IRType:
        if not isinstance(ast, dict):
            return IRType.parse("any")
        ast_type = ast.get("type")

        if ast_type == "literal":
            data_type = ast.get("data_type")
            if data_type:
                return IRType.parse(data_type)
            value = ast.get("value")
            if isinstance(value, bool):
                return IRType.parse("bool")
            if isinstance(value, int):
                return IRType.parse("int")
            if isinstance(value, float):
                return IRType.parse("float")
            if isinstance(value, str):
                return IRType.parse("str")
            if value is None:
                return IRType.parse("None")

        if ast_type == "variable":
            name = ast.get("name") or ast.get("value")
            info = self._scope_symbol(name, scope) if isinstance(name, str) else None
            if info:
                return IRType.parse(info.get("type", "any"))

        if ast_type == "list_literal":
            shape = infer_literal_shape(ast)
            items = ast.get("items", []) or []
            scalar = self._infer_expression(self._first_scalar(items), scope)
            if shape and len(shape) > 1:
                return IRType.parse(f"array[{scalar.canonical}]")
            return IRType.parse(f"array[{scalar.canonical}]")

        if ast_type in {"index_access", "tensor_index_access", "nested_index_access"}:
            name = ast.get("variable")
            info = self._scope_symbol(name, scope) if isinstance(name, str) else None
            if info:
                spec = self.type_parser.parse(info.get("type", "any"))
                if spec.kind == "generic" and spec.args:
                    return IRType(spec.args[0])
                if info.get("type") == "str":
                    return IRType.parse("str")

        if ast_type in {"attribute_access", "complex_attribute_access"}:
            if ast.get("attribute") in {"length", "size", "ndim", "shape"}:
                return IRType.parse("int")

        if ast_type == "method_call":
            info = self._scope_symbol(ast.get("object", ""), scope)
            object_type = info.get("type", "") if info else ""
            if (
                object_type.startswith("Tensor[")
                or object_type == "Tensor"
            ) and ast.get("method") in {
                "row", "column", "slice", "transpose_view", "copy", "transpose", "matmul",
                "add", "sub", "mul", "div", "add_scalar", "sub_scalar",
                "mul_scalar", "div_scalar", "reshape", "sum", "fill", "to",
                "shape", "ndim", "size", "device", "get", "set",
                "mean", "max", "min", "dtype", "is_contiguous", "contiguous", "item",
                }:
                if ast.get("method") == "sum":
                    return IRType.parse("float64")
                if ast.get("method") in {"mean", "max", "min", "item"}:
                    return IRType.parse("float64")
                if ast.get("method") in {"shape", "ndim"}:
                    return IRType.parse("int")
                if ast.get("method") == "size":
                    return IRType.parse("size_t")
                if ast.get("method") == "device":
                    return IRType.parse("str")
                if ast.get("method") == "dtype":
                    return IRType.parse("str")
                if ast.get("method") == "is_contiguous":
                    return IRType.parse("bool")
                if ast.get("method") == "get":
                    return IRType.parse("float64")
                if ast.get("method") == "set":
                    return IRType.parse("None")
                return IRType.parse(object_type or "Tensor[float32]")
            return IRType.parse("any")

        if ast_type == "static_method_call":
            class_name = ast.get("class_name", "")
            class_type = ast.get("class_type") or class_name
            if class_name == "Tensor" and ast.get("method") in {
                "zeros", "from_list"
            }:
                return IRType.parse(
                    class_type if class_type != "Tensor" else "Tensor[float32]"
                )
            return IRType.parse("any")

        if ast_type == "function_call":
            function = ast.get("function")
            info = self._functions.get(function)
            if info:
                return IRType.parse(info.get("return_type", "any"))

        if ast_type == "binary_operation":
            left = self._infer_expression(ast.get("left"), scope)
            right = self._infer_expression(ast.get("right"), scope)
            if ast.get("operator_symbol") in {"<", ">", "<=", ">=", "==", "!=", "and", "or"}:
                return IRType.parse("bool")
            if "float" in {left.canonical, right.canonical}:
                return IRType.parse("float")
            return left if left.canonical != "any" else right

        return IRType.parse("any")

    def _first_scalar(self, items: Iterable[Any]) -> dict[str, Any]:
        for item in items:
            if isinstance(item, dict) and item.get("type") == "list_literal":
                return self._first_scalar(item.get("items", []))
            if isinstance(item, dict):
                return item
        return {"type": "literal", "data_type": "any", "value": None}

    def _collect_reads(self, node: dict[str, Any]) -> set[str]:
        names: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "variable":
                    name = value.get("name") or value.get("value")
                    if isinstance(name, str):
                        names.add(name)
                    return
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for key in ("expression_ast", "condition_ast", "condition", "iterable", "arguments", "value", "index", "indices"):
            visit(node.get(key))
        return names

    def _node_writes(self, node: dict[str, Any]) -> list[str]:
        values = node.get("symbols") or []
        if node.get("node") in {"index_assignment", "nested_index_assignment", "slice_assignment"}:
            values = [node.get("variable", "")]
        elif node.get("node") == "attribute_assignment":
            values = [node.get("object", "")]
        return [value for value in values if isinstance(value, str) and value]


def build_typed_module(scopes: list[dict[str, Any]]) -> TypedModule:
    """Build the typed module from the parser's internal graph."""
    return TypedIRBuilder().build(scopes)


__all__ = ["IRType", "TypedNode", "TypedScope", "TypedModule", "TypedIRBuilder", "build_typed_module"]
