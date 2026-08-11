"""Command-line entry point for the Ocean-to-C compiler pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from src.compiler import CCodeGenerator
from src.debug import JSONValidator
from src.modules.logger import logger
from src.parser import Parser


DEFAULT_CFLAGS = ["-std=c11"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse Ocean source, validate it, generate C11, and optionally compile/run it."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="examples/main.oc",
        help="Ocean source file (default: examples/main.oc)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        help="base directory used to resolve imports (default: source directory)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="path for parsed JSON (default: alongside the source)",
    )
    parser.add_argument(
        "--c-output",
        type=Path,
        help="path for generated C (default: alongside the source)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="binary_output",
        type=Path,
        help="path for the compiled executable (default: alongside the source)",
    )
    parser.add_argument(
        "--compiler",
        default="gcc",
        help="C compiler executable (default: gcc)",
    )
    parser.add_argument(
        "--cflag",
        dest="cflag_list",
        action="append",
        default=[],
        metavar="FLAG",
        help="additional compiler flag; repeat for multiple flags (for example --cflag=-O3)",
    )
    parser.add_argument(
        "--cflags",
        default="",
        metavar="FLAGS",
        help="compiler flags as one shell-style string (for example '-Wall -Wextra')",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="stop after generating JSON and C; do not invoke the C compiler",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the compiled executable after a successful build",
    )
    parser.add_argument(
        "--run-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="argument passed to the executable; repeat for multiple arguments",
    )
    return parser


def default_output_paths(source_path: Path) -> tuple[Path, Path, Path]:
    """Return JSON, C, and executable defaults without changing old main.py paths."""
    if source_path.name == "main.oc" and source_path.parent.name == "examples":
        return (
            source_path.parent / "parsed_code.json",
            source_path.parent / "generated_code.c",
            source_path.parent / "generated_code",
        )
    return (
        source_path.with_suffix(".parsed.json"),
        source_path.with_suffix(".generated.c"),
        source_path.with_suffix(""),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compile_c(
    c_path: Path,
    binary_path: Path,
    compiler: str = "gcc",
    cflags: list[str] | None = None,
) -> list[str]:
    """Compile generated C and return the exact command that was executed."""
    command = [compiler, *(cflags or DEFAULT_CFLAGS), str(c_path), "-o", str(binary_path)]
    _ensure_parent(binary_path)
    print("\n=========== C compiler ===========")
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)
    return command


def compile_pipeline(
    base_path: str | Path,
    p_path: str | Path,
    json_path: str | Path,
    c_path: str | Path,
) -> dict:
    """Parse, validate, and generate C while preserving the public old API."""
    source_path = Path(p_path)
    json_output_path = Path(json_path)
    c_output_path = Path(c_path)

    code = source_path.read_text(encoding="utf-8")

    print("\n=========== PARSER ===========")
    parser = Parser(base_path=str(base_path))
    data = parser.parse_code(code)

    _ensure_parent(json_output_path)
    json_output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n=========== DEBUGGER ===========")
    result_validation = JSONValidator().validate(data)

    print("\nРезультат валидации:")
    print(f"Валидный: {result_validation['is_valid']}")
    print(f"Ошибок: {result_validation['error_count']}")
    print(f"Предупреждений: {result_validation['warning_count']}")

    for warning in result_validation["warnings"]:
        logger.warning(f"Строка {warning['line_number']}: {warning['message']}")
    for error in result_validation["errors"]:
        logger.error(f"Строка {error['line_number']}: {error['message']}")

    if result_validation["errors"]:
        raise RuntimeError(
            "Compilation stopped: validation failed; generated C was not emitted"
        )

    print("\n=========== CCodeGenerator ===========")
    c_code = CCodeGenerator().generate_from_json(data)
    _ensure_parent(c_output_path)
    c_output_path.write_text(c_code, encoding="utf-8")
    print(f"Generated C: {c_output_path}")
    return result_validation


def parse_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    source_path = Path(args.source).expanduser().resolve()
    default_json, default_c, default_binary = default_output_paths(source_path)
    base_path = (
        Path(args.base_path).expanduser().resolve()
        if args.base_path
        else source_path.parent
    )
    json_path = (
        Path(args.json_output).expanduser().resolve()
        if args.json_output
        else default_json
    )
    c_path = Path(args.c_output).expanduser().resolve() if args.c_output else default_c
    binary_path = (
        Path(args.binary_output).expanduser().resolve()
        if args.binary_output
        else default_binary
    )
    return base_path, source_path, json_path, c_path, binary_path


def run_cli(args: argparse.Namespace) -> int:
    base_path, source_path, json_path, c_path, binary_path = parse_cli_paths(args)
    compile_pipeline(base_path, source_path, json_path, c_path)

    if args.no_compile and args.run:
        raise ValueError("--run cannot be used with --no-compile")
    if args.run_arg and not args.run:
        raise ValueError("--run-arg requires --run")
    if args.no_compile:
        print("Compilation skipped (--no-compile).")
        return 0

    cflags = [*DEFAULT_CFLAGS, *args.cflag_list, *shlex.split(args.cflags)]
    compile_c(c_path, binary_path, args.compiler, cflags)

    if args.run:
        print("\n=========== Program ===========")
        subprocess.run([str(binary_path), *args.run_arg], check=True)
    return 0


def main(base_path: str, p_path: str, json_path: str, c_path: str):
    """Backward-compatible entry point used by existing callers."""
    return compile_pipeline(base_path, p_path, json_path, c_path)


if __name__ == "__main__":
    cli_parser = build_argument_parser()
    try:
        raise SystemExit(run_cli(cli_parser.parse_args()))
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
