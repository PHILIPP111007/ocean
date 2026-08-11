import subprocess

import pytest

from src.compiler import CCodeGenerator
from src.parser import Parser


def compile_and_run(source: str, tmp_path) -> str:
    data = Parser().parse_code(source)
    c_code = CCodeGenerator().generate_from_json(data)
    c_file = tmp_path / "program.c"
    executable = tmp_path / "program"
    c_file.write_text(c_code, encoding="utf-8")
    subprocess.run(
        ["gcc", "-std=c11", "-Wall", str(c_file), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    return result.stdout


def test_oop_constructor_method_mutation_and_composition(tmp_path):
    source = """
class Counter:
    def __init__(self, value: int) -> None:
        self.value: int = value

    def increment(self, amount: int) -> int:
        self.value = self.value + amount
        return self.value

class Box:
    def __init__(self, value: int) -> None:
        self.counter: Counter = Counter(value)

    def bump(self, amount: int) -> int:
        return self.counter.increment(amount)

def main() -> int:
    var box: Box = Box(4)
    print(box.bump(3))
    print(box.bump(2))
    return 0
"""

    assert compile_and_run(source, tmp_path) == "7\n9\n"


def test_oop_default_constructor_without_init(tmp_path):
    source = """
class Marker:
    def value(self) -> int:
        return 42

def main() -> int:
    var marker: Marker = Marker()
    print(marker.value())
    return 0
"""

    assert compile_and_run(source, tmp_path) == "42\n"


def test_oop_tensor_fields_use_generic_constructor_initializers(tmp_path):
    source = """
class TensorBox:
    def __init__(self, rows: int, cols: int) -> None:
        self.storage: tensor[float32] = tensor.zeros(rows, cols)
        self.seed: tensor[float32] = [[1.0, 2.0]]

    def size(self) -> int:
        return self.storage.size

    def first(self) -> float32:
        return self.seed[0, 0]

def main() -> int:
    var box: TensorBox = TensorBox(2, 3)
    var first: float32 = box.first()
    print(box.size())
    print(first)
    return 0
"""

    assert compile_and_run(source, tmp_path) == "6\n1.000000\n"


def test_oop_inherited_field_access_uses_embedded_base_layout(tmp_path):
    source = """
class Base:
    def __init__(self) -> None:
        self.value: int = 7

class Child(Base):
    def __init__(self) -> None:
        pass

    def doubled(self) -> int:
        return self.value * 2

def main() -> int:
    var child: Child = Child()
    print(child.doubled())
    return 0
"""

    data = Parser().parse_code(source)
    c_code = CCodeGenerator().generate_from_json(data)
    assert "self->base.value" in c_code
    c_file = tmp_path / "inherited.c"
    executable = tmp_path / "inherited"
    c_file.write_text(c_code, encoding="utf-8")
    subprocess.run(
        ["gcc", "-std=c11", "-Wall", str(c_file), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_oop_rejects_multiple_inheritance():
    source = """
class Left:
    pass

class Right:
    pass

class Invalid(Left, Right):
    pass
"""

    with pytest.raises(RuntimeError, match="multiple inheritance"):
        CCodeGenerator().generate_from_json(Parser().parse_code(source))
