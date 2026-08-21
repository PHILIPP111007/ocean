import socket
import subprocess
import threading
import time
from pathlib import Path

from main import compile_c, compile_pipeline


def _serve_once(port_holder, errors):
    server = None
    conn = None
    try:
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port_holder.append(server.getsockname()[1])

        conn, _ = server.accept()
        request = b""
        while b"\r\n\r\n" not in request:
            request += conn.recv(4096)

        conn.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 5\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b"hello"
        )
    except Exception as exc:
        errors.append(exc)
    finally:
        if conn is not None:
            conn.close()
        if server is not None:
            server.close()


def test_std_net_http(tmp_path):
    ports = []
    errors = []
    thread = threading.Thread(
        target=_serve_once,
        args=(ports, errors),
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 2.0
    while not ports and not errors and time.monotonic() < deadline:
        time.sleep(0.001)

    assert not errors, f"test server failed to start: {errors[0]}"
    assert ports, "test server did not publish a port before the timeout"

    source = tmp_path / "net_test.oc"
    source.write_text(
        f"""
import <std/net/http.oc>

def main() -> int:
    var response: HttpResponse = HTTP.get("http://127.0.0.1:{ports[0]}/", 5000)
    print(response.status())
    print(response.ok())
    print(response.body())
    return 0
""",
        encoding="utf-8",
    )

    c_path = tmp_path / "net_test.generated.c"
    binary = tmp_path / "net_test"

    compile_pipeline(
        str(Path(__file__).resolve().parents[1]),
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

    thread.join(timeout=2)

    assert result.stdout.splitlines() == ["200", "1", "hello"]
