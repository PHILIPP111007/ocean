from src.parser import Parser
from src.typed_ir import (
    OwnershipEffect,
    TypedExpression,
    TypedModule,
    TypedNode,
    TypedScope,
)
from src.codegen import CCodeGenerator
from src.debug import Validator


def test_typed_ir_tracks_types_reads_writes_and_effects():
    source = """
def main() -> int:
    var values: array[float32] = [0.0, 0.0]
    var view: &array[float32] = values
    values[0] = 1.0
    return 0
"""
    module = Parser().parse_typed(source)
    function = next(scope for scope in module.scopes if scope.raw.get("type") == "function")

    declaration, borrow, write, _ = function.nodes
    assert declaration.result_type.canonical == "array[float32]"
    assert declaration.effect == "declare"
    assert borrow.result_type.canonical == "&array[float32]"
    assert borrow.reads == ("values",)
    assert write.effect == "write"
    assert write.writes == ("values",)


def test_parser_typed_entrypoint_is_ready_for_compiler_callers():
    module = Parser().parse_typed(
        "def main() -> int:\n    return 0\n"
    )

    assert isinstance(module, TypedModule)
    assert any(scope.raw.get("type") == "function" for scope in module.scopes)


def test_typed_ir_exposes_typed_backend_views():
    source = """
def main() -> int:
    var values: array[float32] = [1.0, 2.0]
    return 0
"""
    module = Parser().parse_typed(source)

    backend_scopes = module.backend_scopes()
    assert all(isinstance(scope, TypedScope) for scope in backend_scopes)
    assert all(
        isinstance(node, TypedNode)
        for scope in backend_scopes
        for node in scope.get("graph", ())
    )


def test_typed_ir_wraps_nested_expression_nodes():
    module = Parser().parse_typed(
        """
def main() -> int:
    var value: int = 1 + 2
    return value
"""
    )
    declaration = next(
        node for node in module.iter_nodes() if node.node_type == "declaration"
    )
    expression = declaration.get("expression_ast")

    assert isinstance(expression, TypedExpression)
    assert expression.result_type.canonical == "int"
    assert isinstance(expression.get("left"), TypedExpression)
    assert isinstance(expression.get("right"), TypedExpression)


def test_typed_ir_exposes_ownership_transitions():
    module = Parser().parse_typed(
        """
def consume(value: array[int]) -> None:
    return None

def main() -> int:
    var values: array[int] = [1]
    var view: &array[int] = values
    consume(values)
    del view
    return 0
"""
    )

    declarations = {
        node.get("var_name"): node
        for node in module.iter_nodes()
        if node.node_type == "declaration"
    }
    consume = next(node for node in module.iter_nodes() if node.node_type == "function_call")
    delete = next(node for node in module.iter_nodes() if node.node_type == "delete")

    borrow = declarations["view"].ownership_effects[0]
    create = declarations["values"].ownership_effects[0]
    move = consume.ownership_effects[0]
    drop = delete.ownership_effects[0]

    assert isinstance(borrow, OwnershipEffect)
    assert (borrow.kind, borrow.source, borrow.target, borrow.mutable) == (
        "borrow",
        "values",
        "view",
        False,
    )
    assert (create.kind, create.target, create.ownership) == ("create", "values", "unique")
    assert (move.kind, move.source, move.target) == ("move", "values", "consume")
    assert (drop.kind, drop.target) == ("drop", "view")


def test_typed_ir_exposes_source_location_metadata(tmp_path):
    source = tmp_path / "location.oc"
    source.write_text(
        "def main() -> int:\n    var value: int = 1\n    return 0\n",
        encoding="utf-8",
    )
    module = Parser().parse_typed(source.read_text(), file_path=str(source))
    node = next(node for node in module.iter_nodes() if node.node_type == "declaration")

    assert node.source_file == str(source)
    assert node.source_line == 2
    assert node.source_column == 5


def test_typed_ir_is_the_canonical_validator_and_codegen_api():
    source = """
def main() -> int:
    var values: array[float32] = [1.0, 2.0]
    return 0
"""
    module = Parser().parse_typed(source)

    report = Validator().validate(module)
    generated = CCodeGenerator().generate_from_typed_ir(module)

    assert report["is_valid"] is True
    assert "ocean_array_float32" in generated


def test_typed_ir_models_device_tensor_as_managed_public_type():
    module = Parser().parse_typed(
        """
def main() -> int:
    var value: Tensor[int32] = Tensor[int32].zeros(2, 2, "cpu")
    return 0
"""
    )
    value_scope = next(scope for scope in module.scopes if scope.raw.get("type") == "function")
    value_type = value_scope.symbols["value"]

    assert value_type.canonical == "Tensor[int32]"
    assert value_type.memory_kind == "shared"


def test_removed_native_tensor_is_rejected():
    module = Parser().parse_typed(
        """
def main() -> int:
    var value: tensor[float32] = tensor.zeros(1, 1)
    return 0
"""
    )

    report = Validator().validate(module)

    assert not report["is_valid"]
    assert any("тип tensor[T] удален" in error["message"] for error in report["errors"])
