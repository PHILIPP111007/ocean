import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_standard_thread_create_join_and_raw_argument(tmp_path):
    source = tmp_path / "thread_test.oc"

    source.write_text(
        """
import <std/multiprocessing/thread.oc>
cimport <stdio.h>

def worker(arg: *void) -> None:
    unsafe:
        var typed_arg: *int = arg
        var value: int = *typed_arg
        @printf("worker=%d\\n", value)
    return None

def main() -> int:
    var value: int = 42
    unsafe:
        var thread: Thread = Thread.create(worker, &value)
        thread.join()
    print("done")
    return 0
""",
        encoding="utf-8",
    )

    c_path = tmp_path / "thread_test.generated.c"
    binary = tmp_path / "thread_test"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]),
        source,
        c_path,
        quiet=True,
    )

    command = compile_c(c_path, binary)

    assert "-pthread" in command

    generated = c_path.read_text(encoding="utf-8")
    assert "void* arg" in generated
    assert "*void arg" not in generated
    assert "self->handle" in generated
    assert "ocean_thread_is_joinable(self.handle)" not in generated
    assert "ocean_thread_is_joinable(handle)" in generated
    assert "&value" in generated
    assert "int* typed_arg = arg;" in generated

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.stdout.splitlines() == [
        "worker=42",
        "done",
    ]
