"""Project and package configuration for the Ocean command line tool.

The compiler itself remains usable as a library.  This module only describes
how a project on disk is laid out and how its build settings are resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
try:  # Python 3.11+; ``tomli`` is the small backport for Python 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter version
    tomllib = None


PACKAGE_FILE = "ocean.toml"
DEFAULT_ENTRY = "src/main.oc"
DEFAULT_SOURCE_DIR = "src"
DEFAULT_BUILD_DIR = "build"


class PackageError(ValueError):
    """Raised when an Ocean package manifest is missing or invalid."""


def _read_toml(path: Path) -> dict:
    if tomllib is None:
        try:
            import tomli
        except ModuleNotFoundError as error:  # pragma: no cover - Python 3.10 setup path
            raise PackageError("reading ocean.toml on Python < 3.11 requires 'tomli'") from error
        parser = tomli
    else:
        parser = tomllib
    try:
        return parser.loads(path.read_text(encoding="utf-8"))
    except parser.TOMLDecodeError as error:
        raise PackageError(f"invalid {PACKAGE_FILE}: {error}") from error


@dataclass(frozen=True)
class Package:
    """Resolved package manifest and paths.

    Paths in the manifest are always interpreted relative to ``root``.  The
    object is immutable so one compilation cannot accidentally change another
    command's project settings.
    """

    root: Path
    manifest_path: Path
    name: str
    version: str
    entry: str
    source_dir: str
    build_dir: str
    compiler: str
    cflags: tuple[str, ...]
    tests_dir: str

    @property
    def entry_path(self) -> Path:
        return self.root / self.entry

    @property
    def source_path(self) -> Path:
        return self.root / self.source_dir

    @property
    def build_path(self) -> Path:
        return self.root / self.build_dir

    def profile_path(self, profile: str) -> Path:
        return self.build_path / profile

    def artifact_paths(self, profile: str) -> tuple[Path, Path, Path]:
        output_dir = self.profile_path(profile)
        return (
            output_dir / f"{self.name}.parsed.json",
            output_dir / f"{self.name}.c",
            output_dir / self.name,
        )

    def profile_cflags(self, profile: str) -> list[str]:
        """Return the complete compiler flags for a named profile."""
        return profile_flags(self, profile)


def find_manifest(start: str | Path = ".") -> Path | None:
    """Find ``ocean.toml`` in *start* or one of its parents."""
    path = Path(start).expanduser().resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        manifest = candidate / PACKAGE_FILE
        if manifest.is_file():
            return manifest
    return None


def _string(value, field: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise PackageError(f"[package].{field} must be a non-empty string")
    return value.strip()


def _flags(value, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PackageError(f"[build].{field} must be an array of strings")
    return tuple(value)


def load_package(manifest: str | Path) -> Package:
    """Load and validate an ``ocean.toml`` manifest."""
    manifest_path = Path(manifest).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path /= PACKAGE_FILE
    if not manifest_path.is_file():
        raise PackageError(f"package manifest not found: {manifest_path}")

    data = _read_toml(manifest_path)

    package_data = data.get("package", {})
    build_data = data.get("build", {})
    if not isinstance(package_data, dict) or not isinstance(build_data, dict):
        raise PackageError("[package] and [build] must be TOML tables")

    root = manifest_path.parent
    name = _string(package_data.get("name"), "name", root.name)
    if not re.match(r"^[A-Za-z][A-Za-z0-9_-]*$", name):
        raise PackageError("[package].name must contain letters, digits, '-' or '_'")

    entry = _string(package_data.get("entry"), "entry", DEFAULT_ENTRY)
    source_dir = _string(package_data.get("source"), "source", DEFAULT_SOURCE_DIR)
    build_dir = _string(package_data.get("build"), "build", DEFAULT_BUILD_DIR)
    tests_dir = _string(package_data.get("tests"), "tests", "tests")
    compiler = _string(build_data.get("compiler"), "compiler", "gcc")
    cflags = _flags(build_data.get("cflags"), "cflags")

    profile_data = build_data.get("profiles", {})
    if profile_data is not None and not isinstance(profile_data, dict):
        raise PackageError("[build.profiles] must be a TOML table")
    for profile_name, profile in (profile_data or {}).items():
        if not isinstance(profile, dict):
            raise PackageError(f"[build.profiles.{profile_name}] must be a TOML table")
        # Validate profile flags while retaining the compact immutable model.
        _flags(profile.get("cflags"), f"profiles.{profile_name}.cflags")

    package = Package(
        root=root,
        manifest_path=manifest_path,
        name=name,
        version=_string(package_data.get("version"), "version", "0.1.0"),
        entry=entry,
        source_dir=source_dir,
        build_dir=build_dir,
        compiler=compiler,
        cflags=cflags,
        tests_dir=tests_dir,
    )
    if not package.entry_path.is_file():
        raise PackageError(f"package entry file not found: {package.entry_path}")
    return package


def profile_flags(package: Package, profile: str) -> list[str]:
    """Resolve flags including optional profile-specific manifest settings."""
    data = _read_toml(package.manifest_path)
    build_data = data.get("build", {})
    profiles = build_data.get("profiles", {}) if isinstance(build_data, dict) else {}
    profile_data = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
    profile_flags_value = []
    if isinstance(profile_data, dict):
        profile_flags_value = list(_flags(profile_data.get("cflags"), f"profiles.{profile}.cflags"))
    flags = list(package.cflags)
    has_profile_optimization = any(flag.startswith("-O") for flag in profile_flags_value)
    if profile == "release" and not any(flag.startswith("-O") for flag in flags) and not has_profile_optimization:
        flags.append("-O2")
    if isinstance(profile_data, dict):
        flags.extend(profile_flags_value)
    return flags


def create_package(root: str | Path, name: str | None = None, force: bool = False) -> Package:
    """Create a small runnable package skeleton and return its model."""
    package_root = Path(root).expanduser().resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    manifest_path = package_root / PACKAGE_FILE
    entry_path = package_root / DEFAULT_ENTRY
    if not force and (manifest_path.exists() or entry_path.exists()):
        raise PackageError(f"package already exists at {package_root} (use --force to replace files)")

    package_name = name or package_root.name
    manifest_path.write_text(
        "[package]\n"
        f'name = "{package_name}"\n'
        'version = "0.1.0"\n'
        'entry = "src/main.oc"\n'
        'source = "src"\n'
        'build = "build"\n\n'
        "[build]\n"
        'compiler = "gcc"\n'
        'cflags = ["-std=c11"]\n',
        encoding="utf-8",
    )
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(
        "def main() -> int:\n"
        '    print("Hello from Ocean")\n'
        "    return 0\n",
        encoding="utf-8",
    )
    return load_package(manifest_path)


__all__ = ["PACKAGE_FILE", "Package", "PackageError", "create_package", "find_manifest", "load_package", "profile_flags"]
