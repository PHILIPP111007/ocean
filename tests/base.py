from tests.constants import base_path
from src.parser import Parser
from src.compiler import CCodeGenerator


def run(P: str, C: str):
    # Парсим код
    parser = Parser(base_path=base_path)
    typed_ir = parser.parse_typed(P)

    # Генерируем C код
    generator = CCodeGenerator()
    output = generator.generate_from_typed_ir(typed_ir)

    # Проверяем результат
    assert C in output, f"\n--- FAILED ---\nExpected:\n{C}\n\nActual:\n{output}\n"
