import struct
import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def write_npy(path: Path, descr: str, shape: tuple[int, ...], payload: bytes, version=(2, 0)):
    shape_text = "(" + ", ".join(str(value) for value in shape)
    if len(shape) == 1:
        shape_text += ","
    shape_text += ")"
    header = (
        "{'descr': '" + descr + "', 'fortran_order': False, 'shape': "
        + shape_text
        + ", }"
    ).encode("ascii")
    prefix_size = 10 if version[0] == 1 else 12
    padding = (64 - (prefix_size + len(header) + 1) % 64) % 64
    header += b" " * padding + b"\n"
    with path.open("wb") as stream:
        stream.write(b"\x93NUMPY")
        stream.write(bytes(version))
        if version[0] == 1:
            stream.write(struct.pack("<H", len(header)))
        else:
            stream.write(struct.pack("<I", len(header)))
        stream.write(header)
        stream.write(payload)


def test_tensor_npy_reads_external_file_and_writes_compatible_file(tmp_path):
    write_npy(
        tmp_path / "external.npy",
        "<f4",
        (2, 2),
        struct.pack("<4f", 1.5, 2.5, 3.5, 4.5),
        version=(2, 0),
    )
    write_npy(
        tmp_path / "external_big_endian.npy",
        ">i2",
        (1, 2),
        struct.pack(">2h", -12, 300),
        version=(3, 0),
    )
    source = tmp_path / "tensor_npy.oc"
    source.write_text(
        """
import <std/tensor/tensor.oc>

def main() -> int:
    var external: Tensor[float32] = Tensor.load_npy("external.npy", "cpu")
    external.save_npy("roundtrip.npy")
    var restored: Tensor[float32] = Tensor.load_npy("roundtrip.npy", "cpu")
    var integers: Tensor[int32] = Tensor.from_list([[1, 2], [3, 4]], "cpu")
    integers.save_npy("integers.npy")
    var restored_integers: Tensor[int32] = Tensor.load_npy("integers.npy", "cpu")
    var big_endian: Tensor[int16] = Tensor.load_npy("external_big_endian.npy", "cpu")
    print(restored.ndim())
    print(restored.shape(0))
    print(restored.shape(1))
    print(restored[1, 0])
    print(restored_integers[1, 1])
    print(big_endian[0, 0])
    print(big_endian[0, 1])
    return 0
""",
        encoding="utf-8",
    )
    c_path = tmp_path / "tensor_npy.generated.c"
    binary_path = tmp_path / "tensor_npy"
    repository_root = Path(__file__).resolve().parents[1]

    compile_pipeline(str(repository_root), source, c_path, quiet=True)
    compile_c(c_path, binary_path)
    result = subprocess.run(
        [str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.stdout.splitlines() == [
        "2", "2", "2", "3.500000", "4.000000", "-12.000000", "300.000000"
    ]
    roundtrip = (tmp_path / "roundtrip.npy").read_bytes()
    assert roundtrip[:8] == b"\x93NUMPY\x01\x00"
    header_size = struct.unpack_from("<H", roundtrip, 8)[0]
    header = roundtrip[10 : 10 + header_size]
    assert b"'descr': '<f4'" in header
    assert b"'fortran_order': False" in header
    assert b"'shape': (2, 2)" in header
    payload_offset = 10 + header_size
    assert struct.unpack_from("<4f", roundtrip, payload_offset) == (1.5, 2.5, 3.5, 4.5)

    integers = (tmp_path / "integers.npy").read_bytes()
    integer_header_size = struct.unpack_from("<H", integers, 8)[0]
    integer_payload_offset = 10 + integer_header_size
    assert struct.unpack_from("<4i", integers, integer_payload_offset) == (1, 2, 3, 4)
