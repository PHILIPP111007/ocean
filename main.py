"""The ``ocean`` command line interface and legacy compiler entry point."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from src.compiler import CCodeGenerator
from src.debug import JSONValidator
from src.modules.logger import logger
from src.package_model import (
    Package,
    PackageError,
    create_package,
    find_manifest,
    load_package,
    profile_flags,
)
from src.parser import Parser
from src.typed_ir import build_typed_ir


DEFAULT_CFLAGS = ["-std=c11"]
COMMANDS = {"init", "check", "build", "run", "test", "clean"}


def _diagnostic_location(diagnostic: dict) -> str:
    """Format a stable source location for compiler diagnostics."""
    source_file = diagnostic.get("source_file")
    line = diagnostic.get("line_number")
    column = diagnostic.get("column_number")
    if source_file and line:
        return f"{source_file}:{line}:{column or 1}"
    if line:
        return f"строка {line}"
    return "неизвестное место"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocean",
        description="Build, check, run, and test Ocean packages or single source files.",
        epilog=(
            "Commands: init, check, build, run, test, clean. "
            "For compatibility, a source path without a command means build."
        ),
    )
    parser.add_argument(
        "command_or_source",
        nargs="?",
        metavar="COMMAND|SOURCE",
        help="command name, or a source file for the legacy single-file workflow",
    )
    parser.add_argument(
        "source_arg",
        nargs="?",
        metavar="SOURCE",
        help="source file for check/build/run, or target directory for init",
    )
    parser.add_argument("--manifest", type=Path, help="path to ocean.toml (default: search parent directories)")
    parser.add_argument("--source", dest="source_override", type=Path, help="entry source file")
    parser.add_argument("--path", dest="init_path", type=Path, help="target directory for init")
    parser.add_argument("--force", action="store_true", help="allow init to create files in an existing package")
    parser.add_argument("--profile", choices=("debug", "release"), default="debug", help="build profile (default: debug)")
    parser.add_argument("--base-path", type=Path, help="base directory used to resolve imports")
    parser.add_argument("--json-output", type=Path, help="path for parsed JSON")
    parser.add_argument("--c-output", type=Path, help="path for generated C")
    parser.add_argument("-o", "--output", dest="binary_output", type=Path, help="path for the compiled executable")
    parser.add_argument("--compiler", help="C compiler executable (overrides ocean.toml)")
    parser.add_argument(
        "--cflag",
        dest="cflag_list",
        action="append",
        default=[],
        metavar="FLAG",
        help="additional compiler flag; repeat for multiple flags",
    )
    parser.add_argument("--cflags", default="", metavar="FLAGS", help="compiler flags as one shell-style string")
    parser.add_argument("--no-compile", action="store_true", help="stop after generating JSON and C")
    parser.add_argument("--run", action="store_true", help="run the executable after building")
    parser.add_argument("--run-arg", action="append", default=[], metavar="ARG", help="argument passed to the executable")
    parser.add_argument("--test-path", type=Path, help="test directory or file for the test command")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress compiler progress output")
    return parser


def _command(args: argparse.Namespace) -> str:
    value = args.command_or_source
    return value if value in COMMANDS else "build"


def _explicit_source(args: argparse.Namespace) -> Path | None:
    if args.source_override:
        return args.source_override
    if _command(args) != "build" or args.command_or_source in COMMANDS:
        return Path(args.source_arg) if args.source_arg else None
    return Path(args.command_or_source) if args.command_or_source else None


def _load_package_for_args(args: argparse.Namespace, source: Path | None = None) -> Package | None:
    if args.manifest:
        return load_package(args.manifest)
    # A positional source path selects the single-file workflow. The package
    # manifest must not silently replace it with its configured entry file.
    # ``--source`` remains an explicit package-level override.
    if source is not None and not args.source_override:
        return None
    if _command(args) == "build" and args.command_or_source not in COMMANDS and source is None:
        return None
    manifest = find_manifest(source.parent if source else Path.cwd())
    return load_package(manifest) if manifest else None


def default_output_paths(source_path: Path) -> tuple[Path, Path, Path]:
    """Return JSON, C, and executable locations for one source file."""
    return source_path.with_suffix(".parsed.json"), source_path.with_suffix(".generated.c"), source_path.with_suffix("")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


_STANDARD_INCLUDE_PATTERN = re.compile(r'#include\s*[<"](?P<header>std/[^>"]+)[>"]')


def _standard_runtime_dependencies(
    c_path: Path, generated_c: str
) -> tuple[list[str], list[str], bool]:
    """Discover standard-library C runtimes from generated std/ headers."""
    headers = sorted(
        {
            match.group("header")
            for match in _STANDARD_INCLUDE_PATTERN.finditer(generated_c)
        }
    )
    if not headers:
        return [], [], False

    candidates = [c_path.parent, *c_path.parents, Path.cwd()]
    runtime_sources: list[str] = []
    include_flags: list[str] = []
    requires_opencl = False
    roots_seen: set[Path] = set()
    sources_seen: set[Path] = set()

    for header in headers:
        header_path = Path(header)
        for candidate in candidates:
            root = candidate.resolve()
            resolved_header = root / header_path
            if not resolved_header.is_file():
                continue

            if root not in roots_seen:
                roots_seen.add(root)
                include_flags.append(f"-I{root}")

            runtime_source = resolved_header.with_suffix(".c")
            if runtime_source.is_file() and runtime_source not in sources_seen:
                sources_seen.add(runtime_source)
                runtime_sources.append(str(runtime_source))
                requires_opencl = requires_opencl or (
                    "#include <CL/cl.h>" in runtime_source.read_text(encoding="utf-8")
                )
            break
        else:
            raise RuntimeError(
                f"standard-library header '{header}' could not be resolved"
            )

    return runtime_sources, include_flags, requires_opencl


def compile_c(
    c_path: Path,
    binary_path: Path,
    compiler: str = "gcc",
    cflags: list[str] | None = None,
    timeout: float | None = None,
) -> list[str]:
    """Compile generated C and return the exact command that was executed."""
    flags = list(cflags or DEFAULT_CFLAGS)
    generated_c = c_path.read_text(encoding="utf-8")
    runtime_sources, include_flags, runtime_requires_opencl = (
        _standard_runtime_dependencies(c_path, generated_c)
    )
    for include_flag in include_flags:
        if include_flag not in flags:
            flags.append(include_flag)

    runtime_link_flags: list[str] = []
    if runtime_requires_opencl and shutil.which("pkg-config"):
        opencl_probe = subprocess.run(
            ["pkg-config", "--exists", "OpenCL"],
            check=False,
        )
        if opencl_probe.returncode == 0:
            opencl_cflags = subprocess.run(
                ["pkg-config", "--cflags", "OpenCL"],
                check=True,
                capture_output=True,
                text=True,
            )
            opencl_libs = subprocess.run(
                ["pkg-config", "--libs", "OpenCL"],
                check=True,
                capture_output=True,
                text=True,
            )
            flags.extend(
                flag for flag in shlex.split(opencl_cflags.stdout)
                if flag not in flags
            )
            flags.extend(
                flag for flag in shlex.split(opencl_libs.stdout)
                if not flag.startswith("-l") and flag not in flags
            )
            runtime_link_flags.extend(
                flag for flag in shlex.split(opencl_libs.stdout)
                if flag.startswith("-l") and flag not in runtime_link_flags
            )
            if "-DOCEAN_TENSOR_ENABLE_OPENCL" not in flags:
                flags.append("-DOCEAN_TENSOR_ENABLE_OPENCL")
    # A bare OpenMP pragma is otherwise accepted by some C compilers as an
    # ignored extension, silently changing a parallel program into a serial
    # one.  Derive the required compiler/linker flag from generated C unless
    # the caller already supplied an OpenMP setting explicitly.
    if "#pragma omp " in generated_c and not any(
        flag in {"-fopenmp", "-fopenmp-simd"} for flag in flags
    ):
        flags.append("-fopenmp")
        
        if not any(flag.startswith("-O") for flag in flags):
            flags.append("-O3")

    # ``math.h`` declares functions supplied by libm on GCC and Clang.  Keep
    # the FFI source syntax simple: a ``cimport <math.h>`` automatically makes
    # the corresponding linker dependency available to the generated program.
    link_flags = [flag for flag in flags if flag.startswith("-l")]
    flags = [flag for flag in flags if not flag.startswith("-l")]
    if "#include <math.h>" in generated_c and not link_flags:
        link_flags.append("-lm")
    command = [
        compiler,
        *flags,
        str(c_path),
        *runtime_sources,
        "-o",
        str(binary_path),
        *link_flags,
        *runtime_link_flags,
    ]
    _ensure_parent(binary_path)
    print("\n=========== C compiler ===========")
    print("$ " + " ".join(command))
    if timeout is None:
        subprocess.run(command, check=True)
    else:
        subprocess.run(command, check=True, timeout=timeout)
    return command


def compile_pipeline(base_path: str | Path, p_path: str | Path, json_path: str | Path, c_path: str | Path, quiet: bool = False) -> dict:
    """Parse, validate, and generate C while preserving the public old API."""
    source_path, json_output_path, c_output_path = Path(p_path), Path(json_path), Path(c_path)
    code = source_path.read_text(encoding="utf-8")

    if not quiet:
        print("\n=========== PARSER ===========")
    data = Parser(base_path=str(base_path)).parse_code(code, file_path=str(source_path))
    _ensure_parent(json_output_path)
    json_output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if not quiet:
        print("\n=========== DEBUGGER ===========")
    typed_ir = build_typed_ir(data)
    result_validation = JSONValidator().validate_typed_ir(typed_ir)
    if not quiet:
        print("\nРезультат валидации:")
        print(f"Валидный: {result_validation['is_valid']}")
        print(f"Ошибок: {result_validation['error_count']}")
        print(f"Предупреждений: {result_validation['warning_count']}")
    for warning in result_validation["warnings"]:
        logger.warning(f"{_diagnostic_location(warning)}: {warning['message']}")
    for error in result_validation["errors"]:
        logger.error(f"{_diagnostic_location(error)}: {error['message']}")
    if result_validation["errors"]:
        details = "\n".join(result_validation["formatted_errors"])
        raise RuntimeError(
            "Compilation stopped: validation failed; generated C was not emitted\n"
            + details
        )

    if not quiet:
        print("\n=========== CCodeGenerator ===========")
    c_code = CCodeGenerator().generate_from_typed_ir(typed_ir)
    _ensure_parent(c_output_path)
    c_output_path.write_text(c_code, encoding="utf-8")
    if not quiet:
        print(f"Generated C: {c_output_path}")
    return result_validation


def parse_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve paths for both package commands and the old single-file CLI."""
    source_hint = _explicit_source(args)
    package = _load_package_for_args(args, source_hint)
    if source_hint:
        source_path = source_hint.expanduser().resolve()
    elif package:
        source_path = package.entry_path
    else:
        source_path = Path("examples/tensor_std.oc").resolve()

    if package:
        default_json, default_c, default_binary = package.artifact_paths(args.profile)
        default_base = source_path.parent
    else:
        default_json, default_c, default_binary = default_output_paths(source_path)
        default_base = source_path.parent
    base_path = args.base_path.expanduser().resolve() if args.base_path else default_base
    json_path = args.json_output.expanduser().resolve() if args.json_output else default_json.resolve()
    c_path = args.c_output.expanduser().resolve() if args.c_output else default_c.resolve()
    binary_path = args.binary_output.expanduser().resolve() if args.binary_output else default_binary.resolve()
    return base_path, source_path, json_path, c_path, binary_path


def _compiler_settings(args: argparse.Namespace, package: Package | None) -> tuple[str, list[str]]:
    compiler = args.compiler or (package.compiler if package else "gcc")
    flags = list(DEFAULT_CFLAGS)
    if package:
        flags.extend(profile_flags(package, args.profile))
    flags.extend(args.cflag_list)
    flags.extend(shlex.split(args.cflags))
    # Keep command lines readable when a manifest repeats the C11 default.
    return compiler, list(dict.fromkeys(flags))


def _run_tests(args: argparse.Namespace, package: Package | None) -> int:
    test_path = args.test_path
    if test_path is None and package:
        test_path = package.root / package.tests_dir
    command = [sys.executable, "-m", "pytest", "-v"]
    if test_path and test_path.exists():
        command.append(str(test_path))
    print("$ " + " ".join(command))
    return subprocess.run(command, cwd=str(package.root if package else Path.cwd())).returncode


def run_cli(args: argparse.Namespace) -> int:
    command = _command(args)
    if args.no_compile and command not in {"build", "check"}:
        raise ValueError("--no-compile is only valid with build or check")
    if args.run_arg and command != "run":
        raise ValueError("--run-arg requires the run command")
    if args.run and command != "run":
        raise ValueError("--run is only valid with the run command")

    if command == "init":
        target = args.init_path or (Path(args.source_arg) if args.source_arg else Path.cwd())
        package = create_package(target, force=args.force)
        print(f"Created package {package.name} at {package.root}")
        return 0

    source_hint = _explicit_source(args)
    package = _load_package_for_args(args, source_hint)
    if command == "clean":
        if not package:
            raise PackageError("clean requires an ocean.toml package")
        if package.build_path.exists():
            shutil.rmtree(package.build_path)
            print(f"Removed {package.build_path}")
        else:
            print(f"Build directory does not exist: {package.build_path}")
        return 0
    if command == "test":
        return _run_tests(args, package)

    base_path, source_path, json_path, c_path, binary_path = parse_cli_paths(args)
    previous_logging_disabled = logger.disabled
    logger.disabled = args.quiet
    try:
        compile_pipeline(base_path, source_path, json_path, c_path, quiet=args.quiet)
    finally:
        logger.disabled = previous_logging_disabled
    if command == "check" or args.no_compile:
        print("Check passed." if command == "check" else "Compilation skipped (--no-compile).")
        return 0

    compiler, cflags = _compiler_settings(args, package)
    compile_c(c_path, binary_path, compiler, cflags)
    if command == "run":
        print("\n=========== Program ===========")
        subprocess.run([str(binary_path), *args.run_arg], check=True)
    return 0


def main(base_path: str, p_path: str, json_path: str, c_path: str):
    """Backward-compatible entry point used by existing callers."""
    return compile_pipeline(base_path, p_path, json_path, c_path)


def cli(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the installed ``ocean`` console script."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        build_argument_parser().print_help()
        return 0
    try:
        return run_cli(build_argument_parser().parse_args(arguments))
    except (OSError, PackageError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
