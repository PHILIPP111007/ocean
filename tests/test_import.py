from tests.base import run
from pathlib import Path

from src.modules.imports import ImportProcessor


def test_import():
    P = r"""
cimport <stdio.h>
cimport <stdlib.h>
cimport <string.h>
cimport <stdbool.h>
cimport "my_header.h"
import "./module.p"
"""

    C = """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "my_header.h"

int main(void);
"""
    run(P, C)


def test_relative_and_standard_imports(tmp_path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "nested.oc").write_text("nested_value = 1\n", encoding="utf-8")
    (package_dir / "main.oc").write_text(
        'import "./nested.oc"\nmain_value = 2\n',
        encoding="utf-8",
    )
    source_path = tmp_path / "source.oc"
    processor = ImportProcessor(base_path=str(tmp_path))

    relative = processor.process_imports(
        'import "./pkg/main.oc"\n',
        str(source_path),
    )
    assert "nested_value = 1" in relative
    assert "main_value = 2" in relative

    repository_root = Path(__file__).resolve().parents[1]
    standard_processor = ImportProcessor(
        base_path=str(repository_root / "examples")
    )
    standard_path = standard_processor._resolve_import_path(
        "import <std/tensor/tensor.oc>",
        str(repository_root / "examples/tensor_std.oc"),
    )
    assert standard_path == repository_root / "std/tensor/tensor.oc"
