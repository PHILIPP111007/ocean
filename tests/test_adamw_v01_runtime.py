
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_adamw_v01_runtime(tmp_path):
    source = ROOT / "tests" / "adamw_v01_runtime.c"
    binary = tmp_path / "adamw_v01_runtime"

    command = [
        "gcc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        f"-I{ROOT}",
        str(source),
        str(ROOT / "std/tensor/autograd_runtime.c"),
        str(ROOT / "std/tensor/tensor_runtime.c"),
        "-lm",
        "-o",
        str(binary),
    ]

    subprocess.run(command, check=True)
    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[ok] Ocean AdamW runtime v0.1" in result.stdout
