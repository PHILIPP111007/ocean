from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


def _value(pattern: str, stdout: str) -> float:
    match = re.search(pattern, stdout)
    assert match is not None, stdout
    return float(match.group(1))


def test_tiny_gpt_adamw_v01_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/ML/tiny_gpt_adamw_v01.oc"
    c_path = tmp_path / "tiny_gpt_adamw_v01.generated.c"
    binary = tmp_path / "tiny_gpt_adamw_v01"

    compile_pipeline(source.parent, source, c_path, quiet=True)
    compile_c(c_path, binary)

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    stdout = result.stdout.lower()

    initial = _value(
        r"initial loss\s*=\s*([0-9eE+.\-]+)",
        stdout,
    )
    final = _value(
        r"final loss\s*=\s*([0-9eE+.\-]+)",
        stdout,
    )

    assert math.isfinite(initial)
    assert math.isfinite(final)
    assert initial > 0.0
    assert final >= 0.0

    # This is an actual training criterion, not merely a gradient smoke test.
    assert final < initial * 0.70, (initial, final, stdout)

    assert "predicted next token = 7" in stdout
    assert "token embedding grad = 1" in stdout
    assert "position embedding grad = 1" in stdout
    assert "lm head grad = 1" in stdout
    assert "[ok] ocean tinygpt adamw v0.1" in stdout
