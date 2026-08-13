"""Semantic class metadata shared by the OOP lowering passes.

The parser graph is deliberately kept as the source input, but code generation
should not repeatedly rediscover class fields and inheritance from that graph.
These small models provide one stable view for layout, field access and method
resolution throughout the C backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FieldModel:
    """A field declared directly by one class."""

    name: str
    py_type: str
    owner: str
    declaration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodModel:
    """A class method declaration and its corresponding body scope."""

    name: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "None"
    scope: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, declaration: Dict[str, Any], scope: Optional[Dict[str, Any]] = None
    ) -> "MethodModel":
        return cls(
            name=declaration.get("name", ""),
            parameters=list(declaration.get("parameters", [])),
            return_type=declaration.get("return_type", "None"),
            scope=scope,
            metadata=dict(declaration),
        )

@dataclass(frozen=True)
class ResolvedMethod:
    """A method together with the class that provides its implementation."""

    method: MethodModel
    origin: str
    inherited: bool = False


@dataclass
class ClassModel:
    """Complete semantic metadata for one Ocean class."""

    name: str
    bases: List[str] = field(default_factory=list)
    fields: Dict[str, FieldModel] = field(default_factory=dict)
    methods: Dict[str, MethodModel] = field(default_factory=dict)
    declaration: Dict[str, Any] = field(default_factory=dict)
    init_scope: Optional[Dict[str, Any]] = None

    def direct_field(self, name: str) -> Optional[FieldModel]:
        return self.fields.get(name)

    def direct_method(self, name: str) -> Optional[MethodModel]:
        return self.methods.get(name)

    def iter_ancestor_names(self, models: Dict[str, "ClassModel"]) -> Iterable[str]:
        """Yield direct parents first while detecting inheritance cycles."""
        current = self.name
        visited = {current}
        while True:
            current_model = models.get(current)
            if not current_model or not current_model.bases:
                return
            if len(current_model.bases) > 1:
                raise RuntimeError(
                    f"multiple inheritance for class '{self.name}' is not supported"
                )
            parent = current_model.bases[0]
            if parent in visited:
                raise RuntimeError(f"inheritance cycle involving {self.name}")
            visited.add(parent)
            yield parent
            current = parent

    def field(self, name: str, models: Dict[str, "ClassModel"]) -> Optional[FieldModel]:
        """Resolve a field through the single-inheritance chain."""
        direct = self.direct_field(name)
        if direct:
            return direct
        for ancestor_name in self.iter_ancestor_names(models):
            ancestor = models.get(ancestor_name)
            if ancestor:
                field_model = ancestor.direct_field(name)
                if field_model:
                    return field_model
        return None


class ClassRegistry:
    """Canonical class metadata and lookup service for the C backend."""

    def __init__(self, models: Optional[Dict[str, ClassModel]] = None):
        self.models: Dict[str, ClassModel] = models or {}

    def __bool__(self) -> bool:
        return bool(self.models)

    def __iter__(self):
        return iter(self.models)

    def get(self, class_name: str) -> Optional[ClassModel]:
        return self.models.get(class_name)

    def field(self, class_name: str, field_name: str) -> Optional[FieldModel]:
        model = self.get(class_name)
        return model.field(field_name, self.models) if model else None

    def fields_for(self, class_name: str) -> Dict[str, FieldModel]:
        model = self.get(class_name)
        return model.fields if model else {}

    def methods_for(self, class_name: str) -> Dict[str, MethodModel]:
        model = self.get(class_name)
        return model.methods if model else {}

    def bases_for(self, class_name: str) -> List[str]:
        model = self.get(class_name)
        return list(model.bases) if model else []

    def resolved_methods(self) -> Dict[str, Dict[str, ResolvedMethod]]:
        """Resolve methods without rebuilding parser-shaped dictionaries."""
        resolved: Dict[str, Dict[str, ResolvedMethod]] = {}

        def visit(class_name: str, active: set[str]) -> None:
            if class_name in active:
                raise RuntimeError(f"inheritance cycle involving {class_name}")
            active = set(active)
            active.add(class_name)
            model = self.get(class_name)
            if not model:
                return
            methods = resolved.setdefault(class_name, {})
            for name, method in model.methods.items():
                methods[name] = ResolvedMethod(method, class_name)
            if len(model.bases) > 1:
                raise RuntimeError(
                    f"multiple inheritance for class '{class_name}' is not supported"
                )
            for parent in model.bases:
                visit(parent, active)
                for name, method in resolved.get(parent, {}).items():
                    if name not in methods:
                        methods[name] = ResolvedMethod(
                            method.method, method.origin, inherited=True
                        )

        for class_name in self.models:
            visit(class_name, set())
        return resolved

    def inherited_fields(self, class_name: str) -> Iterable[Tuple[str, FieldModel]]:
        model = self.get(class_name)
        if not model:
            return
        chain = list(model.iter_ancestor_names(self.models))
        chain.reverse()
        chain.append(class_name)
        for owner in chain:
            for field_model in self.fields_for(owner).values():
                yield owner, field_model


def _infer_field_type(ast: Mapping[str, Any], context: Dict[str, str]) -> str:
    """Infer only the structural type information needed for class layout."""
    if not ast:
        return "int"
    node_type = ast.get("type", "")
    if node_type == "literal":
        return ast.get("data_type", "int")
    if node_type == "variable":
        return context.get(ast.get("value", ""), "int")
    if node_type == "constructor_call":
        return ast.get("class_name", "int")
    if node_type == "list_literal":
        items = ast.get("items", [])
        return f"list[{_infer_field_type(items[0], context) if items else 'int'}]"
    if node_type == "tuple_literal":
        items = ast.get("items", [])
        types = [_infer_field_type(item, context) for item in items] or ["int"]
        return f"tuple[{types[0]}]" if len(set(types)) == 1 else f"tuple[{', '.join(types)}]"
    if node_type == "binary_operation":
        left = _infer_field_type(ast.get("left", {}), context)
        right = _infer_field_type(ast.get("right", {}), context)
        if left == right:
            return left
        if "float" in left or "double" in left:
            return left
        if "float" in right or "double" in right:
            return right
        return "int"
    return "int"


def build_class_registry(json_data: Iterable[Mapping[str, Any]]) -> ClassRegistry:
    """Build all class metadata directly from the parser graph and scopes."""
    declarations: Dict[str, Dict[str, Any]] = {}
    method_scopes: Dict[tuple[str, str], Dict[str, Any]] = {}

    for scope in json_data:
        if scope.get("type") in {"constructor", "class_method", "static_method", "classmethod"}:
            class_name = scope.get("class_name", "")
            method_name = scope.get("method_name", "")
            if class_name and method_name:
                method_scopes[(class_name, method_name)] = scope

        if scope.get("type") != "module":
            continue
        for node in scope.get("graph", []):
            if node.get("node") == "class_declaration":
                declarations[node.get("class_name", "")] = node

    inferred_fields: Dict[str, Dict[str, str]] = {}
    declared_fields: Dict[str, set[str]] = {}
    for (class_name, method_name), scope in method_scopes.items():
        fields = inferred_fields.setdefault(class_name, {})
        parameters = {
            parameter.get("name", ""): parameter.get("type", "int")
            for parameter in scope.get("parameters", [])
            if parameter.get("name") != "self"
        }
        for node in scope.get("graph", []):
            if node.get("node") != "attribute_assignment" or node.get("object") != "self":
                continue
            attribute = node.get("attribute", "")
            declared_fields.setdefault(class_name, set()).add(attribute)
            field_type = node.get("attribute_type") or node.get(
                "attribute_type_info", {}
            ).get("canonical")
            if field_type:
                fields[attribute] = field_type
            elif attribute not in fields:
                fields[attribute] = _infer_field_type(
                    node.get("value", {}), parameters
                )
        # A field used by a method but initialized without an annotation is
        # still part of the C layout; int is the conservative legacy default.
        for node in scope.get("graph", []):
            for nested in _walk_dicts(node):
                if nested.get("type") == "attribute_access" and nested.get("object") == "self":
                    fields.setdefault(nested.get("attribute", ""), "int")

    models: Dict[str, ClassModel] = {}
    for class_name, declaration in declarations.items():
        model = ClassModel(
            name=class_name,
            bases=list(declaration.get("base_classes", [])),
            declaration=declaration,
        )
        for field_name, field_type in inferred_fields.get(class_name, {}).items():
            model.fields[field_name] = FieldModel(
                name=field_name,
                py_type=field_type,
                owner=class_name,
            )
        for method_group in (
            "methods",
            "static_methods",
            "class_methods",
        ):
            for declaration_info in declaration.get(method_group, []):
                method_name = declaration_info.get("name", "")
                if not method_name:
                    continue
                scope = method_scopes.get((class_name, method_name))
                model.methods[method_name] = MethodModel.from_dict(
                    declaration_info, scope
                )
        model.init_scope = method_scopes.get((class_name, "__init__"))
        models[class_name] = model

    # An untyped ``self.field`` read in a derived method is a reference to the
    # embedded parent field, not a second field in the derived layout.
    for class_name, model in models.items():
        inherited_names = set()
        for parent in model.iter_ancestor_names(models):
            parent_model = models.get(parent)
            if parent_model:
                inherited_names.update(parent_model.fields)
        for field_name in inherited_names - declared_fields.get(class_name, set()):
            model.fields.pop(field_name, None)

    return ClassRegistry(models)


def _walk_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)
