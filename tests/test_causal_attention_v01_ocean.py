from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_causal_attention_v01_ocean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/causal_attention_v01.oc"
    c_path = tmp_path / "causal_attention_v01.generated.c"
    binary = tmp_path / "causal_attention_v01"

    compile_pipeline(
        source.parent,
        source,
        c_path,
        quiet=True,
    )
    compile_c(
        c_path,
        binary,
    )

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[ok] Ocean causal attention v0.1" in result.stdout
    assert "q has grad = 1" in result.stdout.lower()
    assert "k has grad = 1" in result.stdout.lower()
    assert "v has grad = 1" in result.stdout.lower()
