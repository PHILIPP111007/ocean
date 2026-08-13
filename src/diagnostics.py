"""Structured compiler diagnostics.

The compiler keeps diagnostics as typed values internally.  A small mapping
projection is provided for older callers that still consume the validator's
report dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiagnosticSeverity(str, Enum):
    """Diagnostic importance understood by compiler frontends."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class SourceLocation:
    """Location of a diagnostic in an Ocean source file."""

    source_file: str | None = None
    line: int | None = None
    column: int | None = None
    source_content: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    """One compiler diagnostic with a stable machine-readable code."""

    severity: DiagnosticSeverity
    message: str
    code: str
    location: SourceLocation = field(default_factory=SourceLocation)
    scope_idx: int | None = None
    node_idx: int | None = None

    @property
    def source_file(self) -> str | None:
        return self.location.source_file

    @property
    def line_number(self) -> int | None:
        return self.location.line

    @property
    def column_number(self) -> int | None:
        return self.location.column

    def format(self) -> str:
        """Return the human-readable form used by CLI errors."""
        if self.source_file and self.line_number:
            return f"{self.message} ({self.source_file}:{self.line_number}:{self.column_number or 1})"
        if self.line_number:
            return f"{self.message} (строка {self.line_number})"
        return self.message

    def as_dict(self) -> dict[str, Any]:
        """Return the compatibility representation used by old callers."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "scope_idx": self.scope_idx,
            "node_idx": self.node_idx,
            "line_number": self.line_number,
            "source_file": self.source_file,
            "column_number": self.column_number,
            "source_content": self.location.source_content,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Immutable typed report produced by validation."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is DiagnosticSeverity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is DiagnosticSeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy report shape plus the typed diagnostics."""
        errors = [item.as_dict() for item in self.errors]
        warnings = [item.as_dict() for item in self.warnings]
        return {
            "is_valid": self.is_valid,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "diagnostic_report": self,
            "diagnostics": self.diagnostics,
            "errors": errors,
            "warnings": warnings,
            "formatted_errors": [item.format() for item in self.errors],
            "formatted_warnings": [item.format() for item in self.warnings],
        }
