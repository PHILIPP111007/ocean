import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_std_logging(tmp_path):
    source = tmp_path / "logging_test.oc"
    source.write_text(
        """
import <std/logging/logging.oc>

def main() -> int:
    Logging.to_stdout()
    Logging.set_timestamps(False)
    Logging.set_level(LogLevel.INFO())

    var logger: Logger = Logging.get_logger("worker")

    logger.debug("hidden")
    logger.info("hello")
    logger.error("failure")

    Logging.to_file("test.log", False)
    logger.critical("file-message")
    Logging.shutdown()

    return 0
""",
        encoding="utf-8",
    )

    c_path = tmp_path / "logging_test.generated.c"
    binary = tmp_path / "logging_test"

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
        "INFO     | worker | hello",
        "ERROR    | worker | failure",
    ]

    log_text = (tmp_path / "test.log").read_text(encoding="utf-8")
    assert log_text == "CRITICAL | worker | file-message\n"
