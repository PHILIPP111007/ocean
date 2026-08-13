from tests.constants import base_path
from src.parser import Parser
from src.compiler import CCodeGenerator


def generate(source: str) -> str:
    typed_ir = Parser(base_path=base_path).parse_typed(source)
    return CCodeGenerator().generate_from_typed_ir(typed_ir)


def test_array_lowering_and_index_mutation():
    code = generate(
        """
def main() -> int:
    var values: array[float32] = [1.0, 2.0, 3.0]
    values[1] = values[0] + 2.0
    print(values[1])
    return 0
"""
    )

    assert "typedef struct ocean_array_float32" in code
    assert "ocean_array_float32_create" in code
    assert "ocean_array_float32_get" in code
    assert "ocean_array_float32_set" in code
    assert "ocean_array_float32_free(values);" in code
    assert "#include <stddef.h>" in code
    assert "ocean_tensor_float32" not in code
