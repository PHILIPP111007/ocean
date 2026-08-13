from src.parser import Parser
from src.typed_ir import build_typed_ir
from src.codegen import CCodeGenerator
from src.debug import JSONValidator


def test_typed_ir_tracks_types_reads_writes_and_effects():
    source = """
def main() -> int:
    var matrix: tensor[float32] = tensor.zeros(2, 2)
    var view: &tensor[float32] = matrix
    matrix[0, 0] = 1.0
    return 0
"""
    module = build_typed_ir(Parser().parse_code(source))
    function = next(scope for scope in module.scopes if scope.raw.get("type") == "function")

    declaration, borrow, write, _ = function.nodes
    assert declaration.result_type.canonical == "tensor[float32]"
    assert declaration.effect == "declare"
    assert borrow.result_type.canonical == "&tensor[float32]"
    assert borrow.reads == ("matrix",)
    assert write.effect == "write"
    assert write.writes == ("matrix",)


def test_typed_ir_keeps_codegen_compatibility_format():
    source = """
def main() -> int:
    var values: array[float32] = [1.0, 2.0]
    return 0
"""
    parsed = Parser().parse_code(source)
    module = build_typed_ir(parsed)

    assert module.to_legacy_json() == parsed


def test_typed_ir_is_the_canonical_validator_and_codegen_api():
    source = """
def main() -> int:
    var values: tensor[float32] = tensor.zeros(2, 2)
    return 0
"""
    module = build_typed_ir(Parser().parse_code(source))

    report = JSONValidator().validate_typed_ir(module)
    generated = CCodeGenerator().generate_from_typed_ir(module)

    assert report["is_valid"] is True
    assert "ocean_tensor_float32" in generated


def test_typed_ir_models_device_tensor_as_managed_public_type():
    module = build_typed_ir(
        Parser().parse_code(
            """
def main() -> int:
    var value: Tensor[int32] = Tensor[int32].zeros(2, 2, "cpu")
    return 0
"""
        )
    )
    value_scope = next(scope for scope in module.scopes if scope.raw.get("type") == "function")
    value_type = value_scope.symbols["value"]

    assert value_type.canonical == "Tensor[int32]"
    assert value_type.memory_kind == "shared"


def test_legacy_native_tensor_is_reported_for_migration():
    module = build_typed_ir(
        Parser().parse_code(
            """
def main() -> int:
    var value: tensor[float32] = tensor.zeros(1, 1)
    return 0
"""
        )
    )

    report = JSONValidator().validate_typed_ir(module)

    assert report["is_valid"]
    assert any("tensor[T]" in warning["message"] for warning in report["warnings"])
