from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_transformer_block_v01_ocean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/transformer_block_v01.oc"
    c_path = tmp_path / "transformer_block_v01.generated.c"
    binary = tmp_path / "transformer_block_v01"

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

    assert "output shape = 2 3 8" in stdout
    assert "x has grad = 1" in stdout

    expected_grad_lines = [
        "norm1 gamma grad = 1",
        "norm1 beta grad = 1",
        "attention q grad = 1",
        "attention k grad = 1",
        "attention v grad = 1",
        "attention out grad = 1",
        "norm2 gamma grad = 1",
        "norm2 beta grad = 1",
        "ff1 weight grad = 1",
        "ff1 bias grad = 1",
        "ff2 weight grad = 1",
        "ff2 bias grad = 1",
    ]

    for line in expected_grad_lines:
        assert line in stdout

    match = re.search(r"loss\s*=\s*([0-9eE+.\-]+)", stdout)
    assert match is not None
    loss = float(match.group(1))
    assert math.isfinite(loss)
    assert loss >= 0.0

    assert "[ok] ocean transformerblock v0.1" in stdout
