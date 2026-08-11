from __future__ import annotations

import re
from typing import Dict, List, Set


class NamingMixin:
    """Namespace every Phils-generated C type/function with ``ocean_``.

    The pass first discovers declarations/definitions in the generated C and
    then rewrites only those exact identifiers. User variables and external C
    ABI symbols are therefore left untouched.
    """

    def _discover_generated_identifiers(self, code: str) -> Set[str]:
        names: Set[str] = set()

        # typedef struct { ... } name;
        for match in re.finditer(
            r"typedef\s+struct(?:\s+[A-Za-z_]\w*)?\s*\{.*?\}\s*([A-Za-z_]\w*)\s*;",
            code,
            re.S,
        ):
            names.add(match.group(1))

        # typedef struct Tag Alias; and struct Tag { ... }
        for match in re.finditer(
            r"typedef\s+struct\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;", code
        ):
            names.update(match.groups())
        for match in re.finditer(r"\bstruct\s+([A-Za-z_]\w*)\s*\{", code):
            names.add(match.group(1))

        # Function definitions and forward declarations produced by the backend.
        function_pattern = re.compile(
            r"(?m)^\s*(?!(?:return|if|for|while|switch|else)\b)(?:static\s+)?(?:inline\s+)?"
            r"(?:const\s+)?[A-Za-z_]\w*(?:\s*\*+)?"
            r"(?:\s+[A-Za-z_]\w*(?:\s*\*+)?)?\s+"
            r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:\{|;)"
        )
        for match in function_pattern.finditer(code):
            names.add(match.group(1))

        names.discard("main")
        return {name for name in names if name and not name.startswith("ocean_")}

    def _rewrite_identifiers_outside_literals(self, code: str, mapping: Dict[str, str]) -> str:
        out: list[str] = []
        i = 0
        n = len(code)
        state = "normal"
        while i < n:
            ch = code[i]
            nxt = code[i + 1] if i + 1 < n else ""
            if state == "normal":
                if ch == '"':
                    state = "string"; out.append(ch); i += 1; continue
                if ch == "'":
                    state = "char"; out.append(ch); i += 1; continue
                if ch == "/" and nxt == "/":
                    state = "line_comment"; out.extend([ch, nxt]); i += 2; continue
                if ch == "/" and nxt == "*":
                    state = "block_comment"; out.extend([ch, nxt]); i += 2; continue
                if ch.isalpha() or ch == "_":
                    j = i + 1
                    while j < n and (code[j].isalnum() or code[j] == "_"):
                        j += 1
                    token = code[i:j]
                    out.append(mapping.get(token, token))
                    i = j
                    continue
                out.append(ch); i += 1; continue

            out.append(ch)
            if state in {"string", "char"}:
                if ch == "\\" and i + 1 < n:
                    out.append(code[i + 1]); i += 2; continue
                if (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                    state = "normal"
            elif state == "line_comment" and ch == "\n":
                state = "normal"
            elif state == "block_comment" and ch == "*" and nxt == "/":
                out.append(nxt); i += 2; state = "normal"; continue
            i += 1
        return "".join(out)

    def apply_ocean_namespace(self, code: str, json_data: List[Dict]) -> str:
        del json_data  # discovery is based on emitted symbols, not spelling heuristics.
        identifiers = self._discover_generated_identifiers(code)
        mapping = {name: f"ocean_{name}" for name in identifiers}
        return self._rewrite_identifiers_outside_literals(code, mapping)
