from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


GENERIC_TYPES = {
    "list",
    "dict",
    "tuple",
    "set",
    "array",
    "tensor",
    "shared",
    "optional",
}

VALUE_TYPES = {
    "int",
    "float",
    "double",
    "bool",
    "char",
    "byte",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "size_t",
    "uintptr_t",
    "intptr_t",
}

# Device tensors deliberately accept numeric scalar types only.  Strings and
# managed containers are not valid tensor elements.
TENSOR_DTYPES = {
    "bool",
    "int",
    "float",
    "double",
    "float16",
    "float32",
    "float64",
    "int8",
    "int8_t",
    "int16",
    "int16_t",
    "int32",
    "int32_t",
    "int64",
    "int64_t",
    "uint8",
    "uint8_t",
    "uint16",
    "uint16_t",
    "uint32",
    "uint32_t",
    "uint64",
    "uint64_t",
    "size_t",
    "uintptr_t",
    "intptr_t",
}

SHARED_TYPES = {"str", "list", "dict", "tuple"}
OWNED_TYPES = {"array", "tensor"}


@dataclass(frozen=True)
class TypeSpec:
    """Structured representation of a Phils type.

    The parser still emits the canonical string in ``var_type``/``type`` for
    backwards compatibility with the current code generator.  ``TypeSpec`` is
    additionally serialized into ``type_info`` so semantic/ownership passes do
    not have to re-parse strings later.
    """

    kind: str
    name: str = ""
    args: tuple["TypeSpec", ...] = field(default_factory=tuple)
    inner: "TypeSpec | None" = None

    @property
    def canonical(self) -> str:
        if self.kind == "borrow":
            return f"&{self.inner.canonical}"
        if self.kind == "mut_borrow":
            return f"&mut {self.inner.canonical}"
        if self.kind == "raw_pointer":
            return f"*{self.inner.canonical}"
        if self.kind == "optional":
            return f"{self.inner.canonical}?"
        if self.kind == "generic":
            return f"{self.name}[{', '.join(arg.canonical for arg in self.args)}]"
        return self.name

    @property
    def base_name(self) -> str:
        if self.kind in {"borrow", "mut_borrow", "raw_pointer", "optional"}:
            return self.inner.base_name
        return self.name

    @property
    def memory_kind(self) -> str:
        if self.kind == "borrow":
            return "borrow"
        if self.kind == "mut_borrow":
            return "mut_borrow"
        if self.kind == "raw_pointer":
            return "raw"
        if self.kind == "optional":
            return self.inner.memory_kind
        if self.kind == "generic":
            if self.name == "shared":
                return "shared"
            if self.name == "Tensor":
                # Device-aware Tensor is an ARC-managed public facade.  Its
                # backend storage is opaque and released by the class
                # destructor, unlike the compiler-native tensor[T] buffer.
                return "shared"
            if self.name in OWNED_TYPES:
                return "owned"
            if self.name in SHARED_TYPES:
                return "shared"
            return "value"
        if self.name == "str":
            return "shared"
        if self.name in VALUE_TYPES or self.name in {"None", "void"}:
            return "value"
        # User classes remain shared/ARC in the current Phils memory model.
        # Value-only user structures are tagged as ``struct`` by the symbol
        # table and can be refined by the semantic pass.
        return "shared"

    @property
    def is_borrow(self) -> bool:
        return self.kind in {"borrow", "mut_borrow"}

    @property
    def is_mut_borrow(self) -> bool:
        return self.kind == "mut_borrow"

    @property
    def is_raw_pointer(self) -> bool:
        return self.kind == "raw_pointer"

    @property
    def is_optional(self) -> bool:
        return self.kind == "optional"

    def to_dict(self) -> dict:
        result = {
            "kind": self.kind,
            "canonical": self.canonical,
            "base_name": self.base_name,
            "memory_kind": self.memory_kind,
        }
        if self.name:
            result["name"] = self.name
        if self.inner is not None:
            result["inner"] = self.inner.to_dict()
        if self.args:
            result["arguments"] = [arg.to_dict() for arg in self.args]
        return result


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split text only when not nested in (), [], {}, <> or strings."""

    if not text:
        return []

    parts: list[str] = []
    current: list[str] = []
    round_depth = 0
    square_depth = 0
    curly_depth = 0
    angle_depth = 0
    in_string = False
    quote = ""
    escaped = False
    i = 0

    while i < len(text):
        char = text[i]

        if escaped:
            current.append(char)
            escaped = False
            i += 1
            continue

        if char == "\\" and in_string:
            current.append(char)
            escaped = True
            i += 1
            continue

        if in_string:
            current.append(char)
            if char == quote:
                in_string = False
                quote = ""
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            quote = char
            current.append(char)
            i += 1
            continue

        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth = max(0, angle_depth - 1)

        if (
            char == delimiter
            and round_depth == 0
            and square_depth == 0
            and curly_depth == 0
            and angle_depth == 0
        ):
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)

        i += 1

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def find_top_level(text: str, needle: str) -> int:
    """Return the first top-level occurrence of ``needle`` or -1."""

    if not text or not needle:
        return -1

    round_depth = square_depth = curly_depth = angle_depth = 0
    in_string = False
    quote = ""
    escaped = False
    i = 0

    while i <= len(text) - len(needle):
        char = text[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if in_string:
            if char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            quote = char
            i += 1
            continue

        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth = max(0, angle_depth - 1)

        if (
            round_depth == square_depth == curly_depth == angle_depth == 0
            and text.startswith(needle, i)
        ):
            return i

        i += 1

    return -1


class TypeParser:
    """Recursive parser for Phils type expressions."""

    def parse(self, text: str) -> TypeSpec:
        value = (text or "").strip()
        if not value:
            return TypeSpec(kind="named", name="any")

        if value.startswith("&mut "):
            return TypeSpec(kind="mut_borrow", inner=self.parse(value[5:].strip()))

        if value.startswith("&"):
            return TypeSpec(kind="borrow", inner=self.parse(value[1:].strip()))

        if value.startswith("*"):
            return TypeSpec(kind="raw_pointer", inner=self.parse(value[1:].strip()))

        if value.endswith("?"):
            return TypeSpec(kind="optional", inner=self.parse(value[:-1].strip()))

        generic = self._parse_generic(value)
        if generic is not None:
            return generic

        return TypeSpec(kind="named", name=value)

    def _parse_generic(self, value: str) -> TypeSpec | None:
        open_pos = value.find("[")
        if open_pos <= 0 or not value.endswith("]"):
            return None

        name = value[:open_pos].strip()
        if not name or not self._matching_outer_brackets(value, open_pos):
            return None

        inner = value[open_pos + 1 : -1].strip()
        args = tuple(self.parse(part) for part in split_top_level(inner))
        return TypeSpec(kind="generic", name=name, args=args)

    @staticmethod
    def _matching_outer_brackets(value: str, open_pos: int) -> bool:
        depth = 0
        in_string = False
        quote = ""
        escaped = False
        for index in range(open_pos, len(value)):
            char = value[index]
            if escaped:
                escaped = False
                continue
            if in_string:
                if char == "\\":
                    escaped = True
                elif char == quote:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                quote = char
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return index == len(value) - 1
        return False


def infer_literal_shape(ast: dict) -> list[int] | None:
    """Infer a rectangular shape from nested list literals.

    Returns ``None`` for ragged literals.  This is parser metadata only; the
    semantic pass remains responsible for element-type validation.
    """

    if not isinstance(ast, dict) or ast.get("type") != "list_literal":
        return None

    items = ast.get("items", [])
    if not items:
        return [0]

    child_shapes = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "list_literal":
            shape = infer_literal_shape(item)
            if shape is None:
                return None
            child_shapes.append(shape)
        else:
            child_shapes.append([])

    first = child_shapes[0]
    if any(shape != first for shape in child_shapes[1:]):
        return None
    return [len(items), *first]
