import os
import re
from pathlib import Path

from src.modules.logger import logger


class CImportProcessor:
    def __init__(self, base_path=""):
        self.base_path = base_path

    def resolve_cimport(
        self, import_statement: str, current_file_path: str = ""
    ) -> dict:
        """Просто регистрирует C импорт без парсинга"""
        patterns = [
            r"cimport\s+<(.+?)>",  # cimport <stdio.h>
            r'cimport\s+"(.+?)"',  # cimport "my_header.h"
        ]

        for pattern in patterns:
            match = re.match(pattern, import_statement.strip())
            if match:
                header_path = match.group(1)

                # Определяем тип импорта
                is_system = import_statement.strip().startswith("cimport <")

                return {
                    "type": "c_import",
                    "header": header_path,
                    "is_system": is_system,
                    "original_statement": import_statement.strip(),
                }

        return {}


class ImportProcessor:
    def __init__(self, base_path=""):
        self.base_path = base_path
        self.processed_files = set()  # Чтобы избежать циклических импортов

    def _candidate_standard_roots(self, current_file_path: str = ""):
        """Yield real project std directories before incidental nested ``std`` dirs.

        Example/source trees may themselves contain paths such as
        ``examples/std/os/os.oc``.  Those directories are not the standard
        library root and must not shadow ``<std/...>`` imports.

        Prefer ancestors that look like the Ocean project root
        (``src/`` + ``std/``), then fall back to legacy candidates.
        """
        locations = []

        # CWD is commonly the repository root for the CLI. Check it first, but
        # also walk the source/base ancestors so compilation remains robust
        # when invoked from another working directory.
        for value in (os.getcwd(), current_file_path, self.base_path):
            if not value:
                continue

            path = Path(value).expanduser().resolve()
            if path.is_file():
                path = path.parent

            if path not in locations:
                locations.append(path)

        seen = set()

        # Pass 1: only roots that look like an Ocean checkout.
        for location in locations:
            for ancestor in (location, *location.parents):
                candidate = (ancestor / "std").resolve()
                if candidate in seen:
                    continue

                if (ancestor / "src").is_dir() and candidate.is_dir():
                    seen.add(candidate)
                    yield candidate

        # Pass 2: backward-compatible fallback for non-repository embeddings.
        for location in locations:
            for ancestor in (location, *location.parents):
                candidate = (ancestor / "std").resolve()
                if candidate in seen:
                    continue

                seen.add(candidate)
                yield candidate

    def _parse_import(self, import_statement: str) -> tuple[str, str] | None:
        match = re.match(
            r'^import\s+(?:"([^"]+)"|\'([^\']+)\'|<([^>]+)>)\s*$',
            import_statement.strip(),
        )
        if not match:
            return None

        quoted_path, single_quoted_path, standard_path = match.groups()
        if standard_path is not None:
            return "standard", standard_path.strip()
        return "relative", (quoted_path or single_quoted_path or "").strip()

    def _resolve_import_path(
        self, import_statement: str, current_file_path: str = ""
    ) -> Path | None:
        parsed = self._parse_import(import_statement)
        if not parsed:
            return None

        import_kind, import_path = parsed
        if import_kind == "standard":
            if not import_path.startswith("std/"):
                logger.error(
                    f"Стандартный импорт должен начинаться с 'std/': {import_path}"
                )
                return None
            relative_path = import_path[len("std/") :]
            current_source = (
                Path(current_file_path).expanduser().resolve()
                if current_file_path
                else None
            )

            for standard_root in self._candidate_standard_roots(current_file_path):
                candidate = (standard_root / relative_path).resolve()

                # A source fixture may live under examples/std/... and have the
                # same relative path as a real stdlib module. Never resolve a
                # standard import back to the importing source itself.
                if current_source is not None and candidate == current_source:
                    continue

                if candidate.is_file():
                    return candidate
            logger.error(f"Файл стандартной библиотеки не найден: {import_path}")
            return None

        # Quoted imports remain compatible with the old base-path form, while
        # an explicit ./ or ../ path is relative to the importing file.
        if import_path.startswith("./") or import_path.startswith("../"):
            base_dir = (
                Path(current_file_path).expanduser().resolve().parent
                if current_file_path
                else Path(self.base_path or os.getcwd()).expanduser().resolve()
            )
            return (base_dir / import_path).resolve()
        return (
            Path(self.base_path or os.getcwd()).expanduser().resolve() / import_path
        ).resolve()

    def _read_import(
        self, import_statement: str, current_file_path: str = ""
    ) -> tuple[str, str]:
        full_path = self._resolve_import_path(import_statement, current_file_path)
        if full_path is None:
            return "", ""

        path_key = str(full_path)
        if path_key in self.processed_files:
            logger.warning(f"Предупреждение: циклический импорт файла {full_path}")
            return "", ""

        self.processed_files.add(path_key)
        try:
            content = full_path.read_text(encoding="utf-8")
            logger.debug(f"Импортирован файл: {full_path}")
            return content, str(full_path)
        except FileNotFoundError:
            logger.error(f"Ошибка: файл не найден {full_path}")
            return "", ""
        except Exception as error:
            logger.error(f"Ошибка при чтении файла {full_path}: {error}")
            return "", ""

    def resolve_import(self, import_statement: str, current_file_path: str = "") -> str:
        """Обрабатывает импорт и возвращает содержимое импортируемого файла"""
        content, _ = self._read_import(import_statement, current_file_path)
        return content

    def process_imports(self, code: str, current_file_path: str = "") -> str:
        """Обрабатывает все импорты в коде и вставляет содержимое файлов"""
        lines = code.split("\n")
        result_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Проверяем, является ли строка импортом
            if line.strip().startswith("import"):
                # Обрабатываем импорт
                imported_content, imported_file_path = self._read_import(
                    line, current_file_path
                )

                if imported_content:
                    # Рекурсивно обрабатываем импорты в импортированном файле
                    processed_import = self.process_imports(
                        imported_content, imported_file_path
                    )
                    result_lines.append(f"# Импорт из {line.strip()}")
                    result_lines.extend(processed_import.split("\n"))
                    result_lines.append("# Конец импорта")
                else:
                    result_lines.append(f"# Ошибка импорта: {line.strip()}")
            else:
                result_lines.append(line)

            i += 1

        return "\n".join(result_lines)
