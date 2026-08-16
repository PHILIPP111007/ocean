from pathlib import Path

from main import compile_c, compile_pipeline


def test_std_web_compiles(tmp_path):
    source = Path("examples/std/net/server_app.oc").resolve()
    c_path = tmp_path / "server.generated.c"
    binary = tmp_path / "server"

    compile_pipeline(
        str(Path.cwd()),
        source,
        c_path,
        quiet=True,
    )
    compile_c(c_path, binary)

    assert binary.exists()
