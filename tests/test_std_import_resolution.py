from pathlib import Path

from src.modules.imports import ImportProcessor


def test_standard_import_does_not_resolve_to_example_shadow(tmp_path):
    root = tmp_path / "ocean"
    (root / "src").mkdir(parents=True)
    (root / "std/os").mkdir(parents=True)
    (root / "examples/std/os").mkdir(parents=True)

    real_std = root / "std/os/os.oc"
    example = root / "examples/std/os/os.oc"

    real_std.write_text("class OS:\n    pass\n", encoding="utf-8")
    example.write_text("import <std/os/os.oc>\n", encoding="utf-8")

    processor = ImportProcessor(base_path=str(example.parent))

    resolved = processor._resolve_import_path(
        "import <std/os/os.oc>",
        str(example),
    )

    assert resolved == real_std.resolve()
    assert resolved != example.resolve()
