from src.debug import JSONValidator
from src.parser import Parser


def validate(source: str) -> dict:
    return JSONValidator().validate(Parser().parse_code(source))


def test_validator_returns_report_for_malformed_input():
    report = JSONValidator().validate({"not": "a scope list"})

    assert not report["is_valid"]
    assert report["error_count"] == 1
    assert "списком scope" in report["errors"][0]["message"]

    report = JSONValidator().validate([None])
    assert not report["is_valid"]
    assert "Scope 0" in report["errors"][0]["message"]


def test_validator_does_not_leak_symbols_between_functions():
    report = validate("""
def first() -> None:
    var private_value: int = 1
    return None

def second() -> None:
    print(private_value)
    return None
""")

    assert not report["is_valid"]
    assert any("private_value" in error["message"] for error in report["errors"])


def test_validator_accepts_typed_borrow_and_generic_symbols():
    report = validate("""
def show(values: &list[int]) -> None:
    print(values)
    return None

def main() -> None:
    var values: list[int] = [1]
    show(values)
    return None
""")

    assert report["is_valid"]
    assert not any("unknown" in warning["message"] for warning in report["warnings"])


def test_validator_reports_real_source_line_for_type_error():
    report = validate("""
def main() -> int:
    return "not an integer"
""")

    assert not report["is_valid"]
    error = next(error for error in report["errors"] if error["line_number"])
    assert error["line_number"] == 3
    assert "return \"not an integer\"" in error["message"]


def test_validator_reports_file_line_and_column(tmp_path):
    source = tmp_path / "diagnostic.oc"
    source.write_text(
        "def main() -> int:\n    return \"wrong\"\n",
        encoding="utf-8",
    )
    parsed = Parser().parse_code(source.read_text(), file_path=str(source))
    report = JSONValidator().validate(parsed)
    error = next(error for error in report["errors"] if error["line_number"])

    assert error["source_file"] == str(source)
    assert error["line_number"] == 2
    assert error["column_number"] == 5
    assert f"{source}:2:5" in report["formatted_errors"][0]


def test_validator_checks_container_and_index_types_from_ast():
    report = validate("""
def main() -> None:
    var values: list[int] = [1, 2]
    var item: int = values[0]
    var count: int = values.length
    return None
""")

    assert report["is_valid"]
