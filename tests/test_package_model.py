from main import build_argument_parser, parse_cli_paths, run_cli
from src.package_model import create_package, find_manifest, load_package, profile_flags


def test_create_and_load_package(tmp_path):
    package = create_package(tmp_path / "demo", name="demo_app")

    assert package.name == "demo_app"
    assert package.entry_path == tmp_path / "demo" / "src" / "main.oc"
    assert find_manifest(package.root / "src") == package.manifest_path
    assert load_package(package.manifest_path) == package


def test_package_paths_and_release_flags(tmp_path):
    package = create_package(tmp_path / "demo")
    package.manifest_path.write_text(
        package.manifest_path.read_text(encoding="utf-8")
        + '\n[build.profiles.release]\n'
        + 'cflags = ["-O3"]\n',
        encoding="utf-8",
    )
    package = load_package(package.manifest_path)

    assert package.artifact_paths("debug")[0] == package.root / "build" / "debug" / "demo.c"
    assert profile_flags(package, "release") == ["-std=c11", "-O3"]


def test_package_cli_uses_manifest_entry_and_build_directory(tmp_path):
    package = create_package(tmp_path / "demo", name="cli_demo")
    args = build_argument_parser().parse_args(["build", "--manifest", str(package.manifest_path)])

    base, source, c_path, binary = parse_cli_paths(args)

    assert base == package.entry_path.parent
    assert source == package.entry_path
    assert c_path == package.root / "build" / "debug" / "cli_demo.c"
    assert binary == package.root / "build" / "debug" / "cli_demo"


def test_package_build_command_generates_isolated_artifacts(tmp_path):
    package = create_package(tmp_path / "demo", name="built_demo")
    args = build_argument_parser().parse_args(
        ["build", "--manifest", str(package.manifest_path), "--quiet"]
    )

    assert run_cli(args) == 0
    _, _, c_path, binary = parse_cli_paths(args)
    assert c_path.is_file()
    assert binary.is_file()
    assert not (package.root / "build" / "debug" / "built_demo.parsed.json").exists()


def test_legacy_source_argument_still_defaults_to_single_file_paths(tmp_path):
    source = tmp_path / "sample.oc"
    args = build_argument_parser().parse_args([str(source)])

    _, resolved_source, c_path, binary = parse_cli_paths(args)

    assert resolved_source == source.resolve()
    assert c_path == source.with_suffix(".generated.c").resolve()
    assert binary == source.with_suffix("").resolve()
