"""Typed exceptions for Phase 4 (Baseline Geometry).

Mirrors the Phase 1/2 error discipline: a common base plus specific subtypes
that store the offending path or identifier and render it in ``str()`` so every
failure names what went wrong. The pipeline decides whether a raised exception
is fatal (halt) or per-item (flag and continue).
"""

from __future__ import annotations

__all__ = [
    "BaselineGeometryError",
    "ConfigParseError",
    "ConfigValidationError",
    "InputLoadError",
    "DescriptorError",
    "WriteError",
]


class BaselineGeometryError(Exception):
    """Base class for all typed baseline-geometry errors."""

    _prefix = "Baseline geometry error"
    _target_label = "target"

    def __init__(self, target: object, message: str | None = None) -> None:
        self.target = target
        self.message = message
        super().__init__(self._render())

    def _render(self) -> str:
        detail = f" ({self.message})" if self.message else ""
        return f"{self._prefix} [{self._target_label}={self.target!s}]{detail}"


class ConfigParseError(BaselineGeometryError):
    """The configuration file could not be parsed. Names the file path."""

    _prefix = "Failed to parse configuration file"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class ConfigValidationError(BaselineGeometryError):
    """A configuration parameter is missing or invalid. Names the parameter."""

    _prefix = "Invalid configuration"
    _target_label = "parameter"

    def __init__(self, parameter: object, message: str | None = None) -> None:
        self.parameter = parameter
        super().__init__(parameter, message)


class InputLoadError(BaselineGeometryError):
    """A Phase 1-3 input artifact is missing or unreadable. Names the path."""

    _prefix = "Failed to load input artifact"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class DescriptorError(BaselineGeometryError):
    """A descriptor could not be computed. Names the fragment/patch identifier."""

    _prefix = "Failed to compute descriptor"
    _target_label = "id"

    def __init__(self, identifier: object, message: str | None = None) -> None:
        self.identifier = identifier
        super().__init__(identifier, message)


class WriteError(BaselineGeometryError):
    """An output artifact could not be written. Names the target path."""

    _prefix = "Failed to write output"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)
