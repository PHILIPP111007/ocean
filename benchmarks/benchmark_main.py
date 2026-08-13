"""Benchmark the generated C program for examples/matmul.oc.

The benchmark intentionally compiles without an optimization flag.  This
gives us a reproducible baseline for improving the generated structure before
comparing compiler optimizations such as -O3.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler import CCodeGenerator
from src.debug import JSONValidator
from src.parser import Parser
from main import compile_c


def measure(
    command: list[str], timeout: float
) -> tuple[float, subprocess.CompletedProcess[str]]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return time.perf_counter() - started, result


def runtime_summary(values: list[float]) -> dict[str, float]:
    return {
        "min_seconds": min(values),
        "median_seconds": statistics.median(values),
        "max_seconds": max(values),
    }


def run_benchmark(source_path: Path, runs: int, timeout: float, keep: bool) -> dict:
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc is required to run the C benchmark")
    if runs < 1:
        raise ValueError("runs must be positive")

    source = source_path.read_text()
    parser = Parser(base_path=str(source_path.parent))

    started = time.perf_counter()
    parsed = parser.parse_code(source)
    parse_seconds = time.perf_counter() - started

    validation_started = time.perf_counter()
    report = JSONValidator().validate(parsed)
    validation_seconds = time.perf_counter() - validation_started
    if not report["is_valid"]:
        messages = "\n".join(report["formatted_errors"])
        raise RuntimeError(f"source validation failed:\n{messages}")

    generation_started = time.perf_counter()
    # A few legacy codegen paths still print debug details directly.  Keep
    # stdout reserved for benchmark output/JSON and route those diagnostics to
    # stderr instead.
    with contextlib.redirect_stdout(sys.stderr):
        generated = CCodeGenerator().generate_from_json(parsed)
    generation_seconds = time.perf_counter() - generation_started

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="ocean-benchmark-", dir=os.environ.get("TMPDIR")
    )
    try:
        artifact_dir = Path(temporary_directory.name)
        c_path = artifact_dir / "generated_code.c"
        binary_path = artifact_dir / "generated_code"
        c_path.write_text(generated)

        # Deliberately no -O3 (or any other optimization flag).  Use the
        # regular compiler facade so paired standard-library runtimes (for
        # example std/tensor/tensor_runtime.c) are linked as well.
        compile_started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            compile_command = compile_c(
                c_path,
                binary_path,
                cflags=["-std=c11", "-Wall", "-Wextra", "-Wpedantic"],
                timeout=timeout,
            )
        compile_seconds = time.perf_counter() - compile_started

        run_times = []
        output = ""
        for _ in range(runs):
            run_seconds, result = measure([str(binary_path)], timeout)
            run_times.append(run_seconds)
            output = result.stdout

        result = {
            "source": str(source_path),
            "compiler": "gcc",
            "compile_flags": ["-std=c11", "-Wall", "-Wextra", "-Wpedantic"],
            "runs": runs,
            "parse_seconds": parse_seconds,
            "validation_seconds": validation_seconds,
            "generation_seconds": generation_seconds,
            "compile_seconds": compile_seconds,
            "runtime": runtime_summary(run_times),
            "stdout": output.strip().splitlines(),
        }

        if keep:
            kept_dir = ROOT / "benchmarks" / "artifacts"
            kept_dir.mkdir(parents=True, exist_ok=True)
            kept_c = kept_dir / "generated_main.c"
            kept_binary = kept_dir / "generated_main"
            kept_c.write_text(generated)
            shutil.copy2(binary_path, kept_binary)
            result["artifacts"] = {
                "c": str(kept_c),
                "binary": str(kept_binary),
            }
        return result
    finally:
        temporary_directory.cleanup()


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "examples" / "matmul.oc",
        help="Ocean source file to benchmark",
    )
    argument_parser.add_argument(
        "--runs", type=int, default=1, help="number of runtime repetitions"
    )
    argument_parser.add_argument(
        "--timeout", type=float, default=120.0, help="timeout per compile/run step"
    )
    argument_parser.add_argument(
        "--keep", action="store_true", help="keep generated C and executable"
    )
    argument_parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    args = argument_parser.parse_args()

    source_path = args.source if args.source.is_absolute() else ROOT / args.source
    result = run_benchmark(source_path.resolve(), args.runs, args.timeout, args.keep)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"source: {result['source']}")
        print("compile flags: " + " ".join(result["compile_flags"]))
        print(f"parse:      {result['parse_seconds']:.4f}s")
        print(f"validation: {result['validation_seconds']:.4f}s")
        print(f"generation: {result['generation_seconds']:.4f}s")
        print(f"compile:    {result['compile_seconds']:.4f}s")
        runtime = result["runtime"]
        print(f"runtime:    {runtime['median_seconds']:.4f}s")
        print(f"stdout:     {' | '.join(result['stdout'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
