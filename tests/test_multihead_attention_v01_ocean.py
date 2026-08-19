from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_multihead_attention_v01_ocean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/multihead_attention_v01.oc"
    c_path = tmp_path / "multihead_attention_v01.generated.c"
    binary = tmp_path / "multihead_attention_v01"

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
    assert "q weight has grad = 1" in stdout
    assert "k weight has grad = 1" in stdout
    assert "v weight has grad = 1" in stdout
    assert "out weight has grad = 1" in stdout
    assert "[ok] ocean multiheadattention v0.1" in stdout
