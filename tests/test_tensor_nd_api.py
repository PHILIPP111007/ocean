from __future__ import annotations
import subprocess
from pathlib import Path
from main import compile_c, compile_pipeline


def test_tensor_nd_normal_api(tmp_path):
    src = tmp_path / "nd_api.oc"
    src.write_text(
        """
import <std/tensor/tensor.oc>

def main() -> int:
    var base: Tensor[float32] = Tensor.from_list([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0],[10.0,11.0,12.0]], "cpu")
    var x: Tensor[float32] = base.reshape([1,2,2,3])
    var xt: Tensor[float32] = x.transpose(-2, -1)
    var reduced: Tensor[float32] = x.sum_dim(-1, True)
    print(x.ndim())
    print(x.shape(0))
    print(x.shape(1))
    print(x.shape(2))
    print(x.shape(3))
    print(xt.shape(2))
    print(xt.shape(3))
    print(reduced[0,0,0,0])
    return 0
""",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    c = tmp_path / "nd_api.c"
    b = tmp_path / "nd_api"
    compile_pipeline(str(root), src, c, quiet=True)
    compile_c(c, b)
    r = subprocess.run([str(b)], check=True, capture_output=True, text=True)
    assert r.stdout.splitlines() == ["4","1","2","2","3","3","2","6.000000"]
