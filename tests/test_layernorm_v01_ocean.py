from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_affine_layernorm_v01_ocean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/layernorm_v01.oc"
    c_path = tmp_path / "layernorm_v01.generated.c"
    binary = tmp_path / "layernorm_v01"

    compile_pipeline(
        source.parent,
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
    )

    stdout = result.stdout.lower()
    assert "shape = 2 3 8" in stdout
    assert "x grad = 1" in stdout
    assert "gamma grad = 1" in stdout
    assert "beta grad = 1" in stdout
    assert "[ok] ocean affine layernorm v0.1" in stdout
