from pathlib import Path

from main import compile_c, compile_pipeline


def test_std_math_compiles(tmp_path):
    source = tmp_path / "math_test.oc"
    source.write_text(
        """
import <std/math/math.oc>


def main() -> int:
    var a: float64 = Math.sqrt(9.0)
    var b: float64 = Math.pow(2.0, 3.0)
    var c: float64 = Math.sin(0.0)
    var d: float64 = Math.hypot(3.0, 4.0)
    var e: bool = Math.isfinite(a)

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    return 0
""",
        encoding="utf-8",
    )

    c_path = tmp_path / "math_test.generated.c"
    binary = tmp_path / "math_test"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]),
        source,
        c_path,
        quiet=True,
    )

    command = compile_c(c_path, binary)
    assert "-lm" in command
