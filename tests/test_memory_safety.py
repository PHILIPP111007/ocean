import pytest

from src.compiler import CCodeGenerator
from src.debug import JSONValidator
from src.parser import Parser


def compile_ocean(source: str) -> str:
    data = Parser().parse_code(source)
    return CCodeGenerator().generate_from_json(data)


def test_direct_c_call_requires_unsafe_block():
    source = """
cimport <stdio.h>

def main() -> int:
    @puts(\"hello\")
    return 0
"""
    data = Parser().parse_code(source)
    report = JSONValidator().validate(data)

    assert not report["is_valid"]
    assert any("unsafe" in error["message"] for error in report["errors"])
    with pytest.raises(RuntimeError, match="unsafe"):
        CCodeGenerator().generate_from_json(data)


def test_raw_pointer_requires_unsafe_block():
    source = """
def main() -> int:
    var value: int = 1
    var pointer: *int = &value
    return value
"""
    with pytest.raises(RuntimeError, match="raw pointer"):
        compile_ocean(source)


def test_unsafe_block_allows_explicit_ffi():
    source = """
cimport <stdio.h>

def main() -> int:
    unsafe:
        @puts(\"hello\")
    return 0
"""
    output = compile_ocean(source)
    assert "puts(\"hello\");" in output
