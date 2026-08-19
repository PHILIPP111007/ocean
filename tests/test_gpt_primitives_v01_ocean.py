from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def test_gpt_primitives_v01_ocean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/gpt_primitives_v01.oc"
    c_path = tmp_path / "gpt_primitives_v01.generated.c"
    binary = tmp_path / "gpt_primitives_v01"

    compile_pipeline(source.parent, source, c_path, quiet=True)
    compile_c(c_path, binary)

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.lower()

    assert "embedding shape = 2 3 6" in stdout
    assert "embedding grad = 1" in stdout
    assert "projection grad = 1" in stdout

    match = re.search(r"loss\s*=\s*([0-9eE+.\-]+)", stdout)
    assert match is not None
    loss = float(match.group(1))
    assert math.isfinite(loss)
    assert loss > 0.0
    assert "[ok] ocean gpt primitives v0.1" in stdout
