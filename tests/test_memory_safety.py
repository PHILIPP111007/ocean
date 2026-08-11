import pytest

from src.compiler import CCodeGenerator
from src.debug import JSONValidator
from src.parser import Parser


def compile_ocean(source: str) -> str:
    data = Parser().parse_code(source)
    return CCodeGenerator().generate_from_json(data)


def validate_ocean(source: str) -> dict:
    return JSONValidator().validate(Parser().parse_code(source))


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


def test_owner_cannot_be_deleted_while_borrowed():
    report = validate_ocean("""
def main() -> None:
    var values: list[int] = [1]
    var view: &list[int] = values
    del values
    return None
""")

    assert not report["is_valid"]
    assert any("while it is borrowed" in error["message"] for error in report["errors"])


def test_mutable_and_immutable_borrows_are_exclusive():
    report = validate_ocean("""
def main() -> None:
    var values: list[int] = [1]
    var mutable: &mut list[int] = values
    var readonly: &list[int] = values
    return None
""")

    assert not report["is_valid"]
    assert any("mutable borrow is active" in error["message"] for error in report["errors"])


def test_borrow_cannot_escape_through_return():
    report = validate_ocean("""
def get_view() -> &list[int]:
    var values: list[int] = [1]
    var view: &list[int] = values
    return view
""")

    assert not report["is_valid"]
    assert any("escape through a function return" in error["message"] for error in report["errors"])


def test_borrow_is_released_at_block_exit():
    report = validate_ocean("""
def main(flag: bool) -> None:
    var values: list[int] = [1]
    if flag:
        var view: &list[int] = values
        print(view)
    values.append(2)
    return None
""")

    assert report["is_valid"]


def test_borrow_cannot_escape_to_non_borrowing_parameter():
    report = validate_ocean("""
def consume(values: list[int]) -> None:
    print(values)
    return None

def main() -> None:
    var values: list[int] = [1]
    var view: &list[int] = values
    consume(view)
    return None
""")

    assert not report["is_valid"]
    assert any("escapes through call" in error["message"] for error in report["errors"])
