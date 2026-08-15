import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_std_os(tmp_path):
    source = tmp_path / "os_test.oc"
    source.write_text(
        """
import <std/os/os.oc>
import <std/io/file.oc>

def main() -> int:
    OS.makedirs("a/b")

    print(OS.exists("a"))
    print(OS.is_dir("a/b"))

    var file: File = open("a/b/test.txt", "w")
    file.write("hello")
    file.close()

    print(OS.is_file("a/b/test.txt"))

    OS.rename("a/b/test.txt", "a/b/renamed.txt")
    print(OS.exists("a/b/renamed.txt"))

    OS.setenv("OCEAN_TEST_ENV", "ok", True)
    print(OS.getenv("OCEAN_TEST_ENV", "missing"))
    OS.unsetenv("OCEAN_TEST_ENV")
    print(OS.has_env("OCEAN_TEST_ENV"))

    var entries: list[str] = OS.listdir("a/b")
    print(len(entries))

    OS.remove("a/b/renamed.txt")
    OS.rmdir("a/b")
    OS.rmdir("a")

    return 0
""",
        encoding="utf-8",
    )

    c_path = tmp_path / "os_test.generated.c"
    binary = tmp_path / "os_test"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]),
        source,
        c_path,
        quiet=True,
    )
    compile_c(c_path, binary)

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.stdout.splitlines() == [
        "1",
        "1",
        "1",
        "1",
        "ok",
        "0",
        "1",
    ]
