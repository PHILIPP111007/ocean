from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_tensor_nd_v02_ocean_facade(tmp_path):
    source = tmp_path / "tensor_nd_v02.oc"
    source.write_text(
        """
import <std/tensor/tensor_nd.oc>

def main() -> int:
    var base: Tensor[float32] = Tensor.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], "cpu")
    var x: Tensor[float32] = reshape4(base, 1, 2, 2, 3)
    var xt: Tensor[float32] = transpose_dims(x, -2, -1)
    var reduced: Tensor[float32] = sum_dim(x, -1, True)

    print(x.ndim())
    print(x.shape(0))
    print(x.shape(1))
    print(x.shape(2))
    print(x.shape(3))
    print(xt.shape(2))
    print(xt.shape(3))
    print(reduced.shape(2))
    print(reduced.shape(3))
    print(reduced[0, 0, 0, 0])
    return 0
""",
        encoding="utf-8",
    )

    root = Path(__file__).resolve().parents[1]
    c_path = tmp_path / "tensor_nd_v02.generated.c"
    binary_path = tmp_path / "tensor_nd_v02"

    compile_pipeline(str(root), source, c_path, quiet=True)
    compile_c(c_path, binary_path)

    result = subprocess.run(
        [str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "4", "1", "2", "2", "3",
        "3", "2", "2", "1", "6.000000",
    ]
