import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_standard_time_runtime(tmp_path):
    source = tmp_path / "time_test.oc"
    source.write_text(
        """
import <std/time/time.oc>

def main() -> int:
    var before: int64 = Time.monotonic_ns()
    Time.sleep_ms(20)
    var after: int64 = Time.monotonic_ns()

    var utc: DateTime = Time.utc_from_unix(0)

    print(after > before)
    print(utc.date())
    print(utc.clock())
    return 0
""",
        encoding="utf-8",
    )

    c_path = tmp_path / "time_test.generated.c"
    binary = tmp_path / "time_test"

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
        "1970-01-01",
        "00:00:00",
    ]
