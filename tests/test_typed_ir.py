from src.parser import Parser
from src.typed_ir import build_typed_ir


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
