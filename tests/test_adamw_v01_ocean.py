from pathlib import Path
import math
import re
import subprocess

from main import compile_c, compile_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_adamw_v01_ocean(tmp_path):
    source = ROOT / "examples" / "ML" / "adamw_v01.oc"
    generated_c = tmp_path / "adamw_v01.c"
    binary = tmp_path / "adamw_v01"

    compile_pipeline(
        source.parent,
        source,
        generated_c,
        quiet=True,
    )

    compile_c(
        generated_c,
        binary,
    )

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    initial_match = re.search(
        r"initial loss\s*=\s*([-+0-9.eE]+)",
        result.stdout,
    )
    final_match = re.search(
        r"final loss\s*=\s*([-+0-9.eE]+)",
        result.stdout,
    )

    assert initial_match
    assert final_match

    initial = float(initial_match.group(1))
    final = float(final_match.group(1))

    assert math.isfinite(initial)
    assert math.isfinite(final)
    assert final < initial * 0.05
    assert "[ok] Ocean AdamW v0.1" in result.stdout
