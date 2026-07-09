"""Shared typed exceptions for Phase 2 (Patch Generation).

Mirrors the Phase 1 ``dataset_foundation.errors`` discipline: a common base
exception plus specific subtypes for configuration, input loading, layout,
write, and visualization failures. Every exception stores the offending path
or identifier and renders it in ``str()`` so failures always name what went
wrong. The pipeline decides whether a raised exception is fatal (halt the run)
or per-fragment (flag and continue).

This module is self-contained (no intra-package imports).

Requirements: 1.3, 7.5, 7.6, 10.2, 10.7.
"""

from __future__ import annotations

__all__ = [
    "PatchGenerationError",
    "ConfigParseError",
    "ConfigValidationError",
    "InputLoadError",
    "LayoutError",
    "WriteError",
    "VisualizationError",
]


class PatchGenerationError(Exception):
    """Base class for all typed patch-generation errors.

    Stores the offending path or identifier on ``self.target`` and renders it
    as part of the error message so every subclass surfaces what went wrong.
    """

    #: Human-readable prefix describing the failure category.
    _prefix = "Patch generation error"

    #: Label describing what ``target`` represents (e.g. "path", "parameter").
    _target_label = "target"

    def __init__(self, target: object, message: str | None = None) -> None:
        self.target = target
        self.message = message
        super().__init__(self._render())

    def _render(self) -> str:
        detail = f" ({self.message})" if self.message else ""
        return f"{self._prefix} [{self._target_label}={self.target!s}]{detail}"


class ConfigParseError(PatchGenerationError):
    """The configuration file could not be parsed. Names the file path (Req 10.7)."""

    _prefix = "Failed to parse configuration file"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class ConfigValidationError(PatchGenerationError):
    """A configuration parameter is missing or invalid. Names the parameter (Req 10.2)."""

    _prefix = "Invalid configuration"
    _target_label = "parameter"

    def __init__(self, parameter: object, message: str | None = None) -> None:
        self.parameter = parameter
        super().__init__(parameter, message)


class InputLoadError(PatchGenerationError):
    """A Phase 1 input artifact is missing or unreadable.

    Stores BOTH the offending path and the Fragment_ID and renders both so the
    failing input and the Fragment it belongs to are always named (Req 1.3).
    """

    _prefix = "Failed to load fragment input"
    _target_label = "path"

    def __init__(
        self,
        path: object,
        fragment_id: object,
        message: str | None = None,
    ) -> None:
        self.path = path
        self.fragment_id = fragment_id
        super().__init__(path, message)

    def _render(self) -> str:
        detail = f" ({self.message})" if self.message else ""
        return (
            f"{self._prefix} "
            f"[{self._target_label}={self.target!s}, "
            f"fragment_id={self.fragment_id!s}]{detail}"
        )


class LayoutError(PatchGenerationError):
    """A required subdirectory could not be created. Names the path (Req 7.5)."""

    _prefix = "Failed to initialize patch layout"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class WriteError(PatchGenerationError):
    """An output artifact could not be written. Names the target path (Req 7.6)."""

    _prefix = "Failed to write output"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class VisualizationError(PatchGenerationError):
    """A visualization image could not be saved. Names the target path (Req 8.5)."""

    _prefix = "Failed to render visualization"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)
