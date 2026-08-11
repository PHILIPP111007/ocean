"""Semantic class metadata shared by the OOP lowering passes.

The parser graph is deliberately kept as the source input, but code generation
should not repeatedly rediscover class fields and inheritance from that graph.
These small models provide one stable view for layout, field access and method
resolution while retaining the original dictionaries for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


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

    def to_dict(self) -> Dict[str, Any]:
        """Return the legacy method shape expected by existing generators."""
        result = dict(self.metadata)
        result.update(
            {
                "name": self.name,
                "parameters": self.parameters,
                "return_type": self.return_type,
            }
        )
        return result


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


def build_class_models(
    json_data: List[Dict[str, Any]], class_fields: Dict[str, Dict[str, str]]
) -> Dict[str, ClassModel]:
    """Build class metadata once from declarations, scopes and inferred fields."""
    declarations: Dict[str, Dict[str, Any]] = {}
    method_scopes: Dict[tuple[str, str], Dict[str, Any]] = {}

    for scope in json_data:
        if scope.get("type") in {"constructor", "class_method"}:
            class_name = scope.get("class_name", "")
            method_name = scope.get("method_name", "")
            if class_name and method_name:
                method_scopes[(class_name, method_name)] = scope

        if scope.get("type") != "module":
            continue
        for node in scope.get("graph", []):
            if node.get("node") == "class_declaration":
                declarations[node.get("class_name", "")] = node

    models: Dict[str, ClassModel] = {}
    for class_name, declaration in declarations.items():
        model = ClassModel(
            name=class_name,
            bases=list(declaration.get("base_classes", [])),
            declaration=declaration,
        )
        for field_name, field_type in class_fields.get(class_name, {}).items():
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

    return models
