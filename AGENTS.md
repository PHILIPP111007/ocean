# Repository Guidelines

## Project Structure & Module Organization

Ocean is a Python-like language that lowers Phils/Ocean source to C. The command-line entry point is
`main.py`; it reads `examples/main.oc` and writes parsed JSON, generated C, and an executable in
`examples/`. The compiler implementation is under `src/`: parsing and validation live in
`src/parser.py` and `src/debug.py`, while C generation is organized in `src/codegen/` (ownership,
scope, types, statements, expressions, containers, and OOP). `src/compiler.py` provides the public
generator import used by existing callers. Tests are in `tests/`, documentation in `docs/`, and
illustrations in `images/`.

## Build, Test, and Development Commands

Run the full Python test suite with:

```bash
pytest --verbose
```

Run the sample compiler pipeline with:

```bash
python main.py
```

This parses `examples/main.oc`, emits `examples/parsed_code.json` and
`examples/generated_code.c`, then compiles the C output with `gcc`. For generated C, use C11 and
strict diagnostics; add `-pthread` for pthread examples. Sanitizers are recommended during runtime
testing (`-fsanitize=address,undefined`).

## Coding Style & Naming Conventions

Use four-space indentation, readable snake_case for Python functions and variables, and PascalCase
for classes. Keep compiler passes focused in their existing `src/codegen/` modules. Generated Ocean
symbols use the `ocean_` prefix; preserve external C/POSIX names for FFI compatibility. No formatter
or linter configuration is currently committed, so keep changes consistent with nearby code.

## Testing Guidelines

Tests use pytest-style functions named `test_*`. Most compiler tests call `tests.base.run`, supplying
Ocean input and an expected C fragment. Add a focused test beside the relevant feature, and run the
entire suite before submitting changes. Include ownership, borrow, or generated-C assertions when
modifying memory-management behavior.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase summaries (for example, `updated compiler` and
`added dict.get func`). Follow that concise style while naming the affected area. Pull requests
should explain the language/compiler behavior changed, list test commands run, call out generated
C or memory-model implications, and include documentation or example updates when syntax changes.

## Safety & Configuration Notes

Direct C/POSIX calls are an unsafe FFI boundary. The ownership model is non-atomic and intended for
thread-confined managed objects; validate generated code with strict warnings and sanitizers before
relying on it.
