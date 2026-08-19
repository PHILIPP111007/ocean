from __future__ import annotations

import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_attention_v01_ocean_frontend(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/attention_v01.oc"
    c_path = tmp_path / "attention_v01.generated.c"
    binary = tmp_path / "attention_v01"

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
    assert "[ok] Ocean 4D attention v0.1" in result.stdout
