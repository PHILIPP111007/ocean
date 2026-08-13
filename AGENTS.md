# Repository Guidelines

## Project Structure & Module Organization

Ocean is a Python-like language that lowers Phils/Ocean source to C. The command-line entry point is
`main.py`; package commands read `ocean.toml` and write profile artifacts under `build/`, while the
no-argument command retains the legacy `examples/` workflow. The compiler implementation is under
`src/`: parsing and validation live in
`src/parser.py`, `src/typed_ir.py`, and `src/debug.py`, while C generation is organized in
`src/codegen/` (ownership, scope, types, statements, expressions, containers, tensors, and OOP).
`src/compiler.py` provides the public generator import used by existing callers. Tests are in
`tests/`, documentation in `docs/`, and illustrations in `images/`.

## Build, Test, and Development Commands

Run the full Python test suite with:

```bash
pytest --verbose
```

Run the sample compiler pipeline with:

```bash
python main.py build
```

This parses the package entry configured in `ocean.toml`, emits package artifacts under
`build/<profile>/`, then compiles the C output with `gcc`. Running `python main.py` with no
arguments prints help and performs no build.
For generated C, use C11 and
strict diagnostics; add `-pthread` for pthread examples. Sanitizers are recommended during runtime
testing (`-fsanitize=address,undefined`).

The CLI accepts a custom source path plus `--base-path`, `--json-output`, `--c-output`, `-o`, and
`--compiler`. Pass repeatable C options with `--cflag=-O2` or a group with `--cflags "-Wall -g"`;
use `--no-compile` for generation-only checks and `--run` to execute the compiled binary.
Package workflows use `python main.py init|check|build|run|test|clean` and configure
`ocean.toml`; package artifacts are written to `build/<profile>/`. A source path without a command
continues to use the legacy single-file layout, while an empty CLI invocation only prints help.

Build the Python distribution with `python -m build` and inspect it with `python -m twine check
dist/*`. The installed console entry point is `ocean`; package metadata and the entry point live in
`pyproject.toml`.

## Coding Style & Naming Conventions

Use four-space indentation, readable snake_case for Python functions and variables, and PascalCase
for classes. Keep compiler passes focused in their existing `src/codegen/` modules. Generated Ocean
symbols use the `ocean_` prefix; preserve external C/POSIX names for FFI compatibility. No formatter
or linter configuration is currently committed, so keep changes consistent with nearby code.

## Testing Guidelines

Tests use pytest-style functions named `test_*`. Most compiler tests call `tests.base.run`, supplying
Ocean input and an expected C fragment. Add a focused test beside the relevant feature, and run the
entire suite before submitting changes. Include typed-IR metadata, ownership/move, borrow, or
generated-C assertions when modifying these passes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase summaries (for example, `updated compiler` and
`added dict.get func`). Follow that concise style while naming the affected area. Pull requests
should explain the language/compiler behavior changed, list test commands run, call out generated
C or memory-model implications, and include documentation or example updates when syntax changes.

## Safety & Configuration Notes

Direct C/POSIX calls use `@` and must be inside an explicit `unsafe:` block; raw pointer declarations
(`*T`) follow the same rule. Safe code cannot cross this boundary implicitly. The ownership model is
non-atomic and intended for thread-confined managed objects; validate generated code with strict
warnings and sanitizers before relying on it.
