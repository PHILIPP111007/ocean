from pathlib import Path

from main import (
    _standard_runtime_dependencies,
    build_argument_parser,
    cli,
    compile_c,
    parse_cli_paths,
)


def test_empty_cli_prints_help_without_running_pipeline(capsys):
    assert cli([]) == 0

    output = capsys.readouterr().out
    assert "usage: ocean" in output
    assert "Commands: init, check, build, run, test, clean" in output


def test_parser_does_not_assign_source_or_output_paths_by_default():
    args = build_argument_parser().parse_args([])

    assert args.command_or_source is None
    assert args.source_arg is None
    assert args.source_override is None
    assert args.json_output is None
    assert args.c_output is None
    assert args.binary_output is None


def test_cli_preserves_legacy_default_paths():
    args = build_argument_parser().parse_args([])

    base_path, source_path, json_path, c_path, binary_path = parse_cli_paths(args)

    expected_source = Path("examples/main.oc").resolve()
    assert base_path == expected_source.parent
    assert source_path == expected_source
    assert json_path == expected_source.parent / "parsed_code.json"
    assert c_path == expected_source.parent / "generated_code.c"
    assert binary_path == expected_source.parent / "generated_code"


def test_cli_accepts_custom_paths_flags_and_run_arguments(tmp_path):
    source = tmp_path / "sample.oc"
    json_output = tmp_path / "artifacts" / "sample.json"
    c_output = tmp_path / "artifacts" / "sample.c"
    binary_output = tmp_path / "bin" / "sample"

    args = build_argument_parser().parse_args(
        [
            str(source),
            "--base-path",
            str(tmp_path / "imports"),
            "--json-output",
            str(json_output),
            "--c-output",
            str(c_output),
            "--output",
            str(binary_output),
            "--compiler",
            "clang",
            "--cflag=-O3",
            "--cflag=-pthread",
            "--cflags",
            "-Wall -Wextra",
            "--run",
            "--run-arg",
            "hello",
        ]
    )

    paths = parse_cli_paths(args)

    assert paths == (
        (tmp_path / "imports").resolve(),
        source.resolve(),
        json_output.resolve(),
        c_output.resolve(),
        binary_output.resolve(),
    )
    assert args.compiler == "clang"
    assert args.cflag_list == ["-O3", "-pthread"]
    assert args.cflags == "-Wall -Wextra"
    assert args.run_arg == ["hello"]


def test_positional_source_does_not_use_package_entry():
    args = build_argument_parser().parse_args(["build", "examples/neural_network.oc"])

    _, source_path, json_path, c_path, binary_path = parse_cli_paths(args)

    source = (Path("examples") / "neural_network.oc").resolve()
    assert source_path == source
    assert json_path == source.with_suffix(".parsed.json")
    assert c_path == source.with_suffix(".generated.c")
    assert binary_path == source.with_suffix("")


def test_compile_c_adds_libm_for_generated_math_import(tmp_path, monkeypatch):
    c_file = tmp_path / "math.c"
    binary = tmp_path / "math"
    c_file.write_text("#include <math.h>\nint main(void) { return (int)sqrt(4.0); }\n", encoding="utf-8")
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        assert check is True

    monkeypatch.setattr("main.subprocess.run", fake_run)

    command = compile_c(c_file, binary)

    assert command == captured["command"]
    assert command[-1] == "-lm"


def test_compile_c_places_explicit_libraries_after_sources(tmp_path, monkeypatch):
    c_file = tmp_path / "opencl.c"
    binary = tmp_path / "opencl"
    c_file.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        assert check is True

    monkeypatch.setattr("main.subprocess.run", fake_run)
    command = compile_c(
        c_file,
        binary,
        cflags=["-std=c11", "-L/opt/opencl/lib", "-lOpenCL"],
    )

    assert command == captured["command"]
    assert command.index(str(c_file)) < command.index("-lOpenCL")
    assert "-L/opt/opencl/lib" in command


def test_standard_runtime_is_discovered_from_paired_header(tmp_path):
    header = tmp_path / "std" / "demo" / "demo.h"
    source = header.with_suffix(".c")
    header.parent.mkdir(parents=True)
    header.write_text("int demo(void);\n", encoding="utf-8")
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    generated_c = '#include <std/demo/demo.h>\n'

    runtime_sources, include_flags, requires_opencl = (
        _standard_runtime_dependencies(tmp_path / "generated.c", generated_c)
    )

    assert runtime_sources == [str(source)]
    assert include_flags == [f"-I{tmp_path}"]
    assert requires_opencl is False
