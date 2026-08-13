import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_standard_file_and_binary_file_io(tmp_path):
    source = tmp_path / "file_io.oc"
    text_path = tmp_path / "values.txt"
    binary_path = tmp_path / "values.bin"
    source.write_text(
        '''
import <std/io/file.oc>

def main() -> int:
    var text: File = open("values.txt", "w")
    var lines_to_write: list[str] = ["one\\n", "two\\n"]
    text.writelines(lines_to_write)
    text.close()

    var reader: File = open("values.txt", "r")
    var lines: list[str] = reader.readlines()
    reader.close()

    var binary: BinaryFile = open_binary("values.bin", "wb")
    var bytes_to_write: list[int] = [0, 1, 255]
    binary.write_bytes(bytes_to_write)
    binary.close()

    var binary_reader: BinaryFile = open_binary("values.bin", "rb")
    var values: list[int] = binary_reader.read_bytes(4)
    binary_reader.close()

    print(len(lines))
    print(values[0])
    print(values[1])
    print(values[2])
    return 0
''',
        encoding="utf-8",
    )
    c_path = tmp_path / "file_io.generated.c"
    binary = tmp_path / "file_io"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]), source, c_path, quiet=True
    )
    compile_c(c_path, binary)
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, cwd=tmp_path
    )

    assert result.stdout.splitlines() == ["2", "0", "1", "255"]
    assert text_path.read_text(encoding="utf-8") == "one\ntwo\n"
    assert binary_path.read_bytes() == bytes([0, 1, 255])
