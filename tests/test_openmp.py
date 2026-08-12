import subprocess

from main import compile_c, compile_pipeline
from src.compiler import CCodeGenerator
from src.debug import JSONValidator
from src.parser import Parser
from src.typed_ir import build_typed_ir


def _parse(source: str):
    return Parser().parse_code(source)


def test_openmp_pragma_is_attached_to_for_and_emitted_as_omp():
    source = r"""
def main() -> int:
    var values: array[float32] = [1.0, 2.0, 3.0]
    #pragma omp parallel for schedule(static)
    for i in range(0, 3):
        values[i] = values[i] * 2.0

    return 0
"""
    data = _parse(source)
    loop = data[1]["graph"][1]

    assert loop["openmp"]["directive"] == "parallel for"
    assert loop["openmp"]["clauses"] == [{"name": "schedule", "arguments": "static"}]

    report = JSONValidator().validate(build_typed_ir(data))
    assert report["errors"] == []

    generated = CCodeGenerator().generate_from_typed_ir(build_typed_ir(data))
    assert "#pragma omp parallel for schedule(static)" in generated



def test_openmp_reduction_is_accepted():
    source = r"""
def main() -> int:
    var total: int = 0
    #pragma omp parallel for reduction(+:total)
    for i in range(0, 10):
        total += i

    return 0
"""
    report = JSONValidator().validate(build_typed_ir(_parse(source)))
    assert report["errors"] == []


def test_openmp_rejects_shared_scalar_write_without_reduction():
    source = r"""
def main() -> int:
    var total: int = 0
    #pragma omp parallel for
    for i in range(0, 10):
        total += i

    return 0
"""
    report = JSONValidator().validate(build_typed_ir(_parse(source)))
    assert report["error_count"] == 1
    assert "reduction/private" in report["errors"][0]["message"]


def test_openmp_requires_a_following_for_loop():
    source = r"""
def main() -> int:
    #pragma omp parallel for
    var value: int = 1
    return value
"""
    report = JSONValidator().validate(build_typed_ir(_parse(source)))
    assert report["error_count"] == 1
    assert "immediately followed" in report["errors"][0]["message"]


def test_compile_c_adds_openmp_flag_for_generated_pragma(tmp_path, monkeypatch):
    c_file = tmp_path / "openmp.c"
    binary = tmp_path / "openmp"
    c_file.write_text(
        "#include <omp.h>\n"
        "int main(void) {\n"
        "#pragma omp parallel for\n"
        "for (int i = 0; i < 1; ++i) {}\n"
        "return 0;\n}\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        assert check is True

    monkeypatch.setattr("main.subprocess.run", fake_run)
    command = compile_c(c_file, binary)

    assert command == captured["command"]
    assert "-fopenmp" in command


def test_openmp_source_compiles_with_gcc(tmp_path):
    source = tmp_path / "openmp.oc"
    json_path = tmp_path / "openmp.json"
    c_path = tmp_path / "openmp.c"
    binary = tmp_path / "openmp"
    source.write_text(
        "def main() -> int:\n"
        "    var total: int = 0\n"
        "    #pragma omp parallel for reduction(+:total)\n"
        "    for i in range(0, 100):\n"
        "        total += i\n"
        "    return 0\n",
        encoding="utf-8",
    )

    compile_pipeline(tmp_path, source, json_path, c_path, quiet=True)
    command = compile_c(c_path, binary)
    subprocess.run([str(binary)], check=True)
    assert "-fopenmp" in command
