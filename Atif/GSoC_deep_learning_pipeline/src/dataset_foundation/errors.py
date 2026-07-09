"""Shared typed exceptions for the dataset-foundation pipeline.

Every exception in this module stores the offending path or identifier and
renders it in ``str()`` so failures always name what went wrong. The pipeline
decides whether a raised exception is fatal (halt the run) or per-fragment
(flag and continue).

Requirements: 1.3, 1.8, 2.5, 5.7, 5.9, 7.6, 9.2, 9.6.
"""

from __future__ import annotations

__all__ = [
    "DatasetFoundationError",
    "ConfigParseError",
    "ConfigValidationError",
    "PlyReadError",
    "LayoutError",
    "WriteError",
    "NormalizationError",
    "AlignmentImportError",
]


class DatasetFoundationError(Exception):
    """Base class for all typed dataset-foundation errors.

    Stores the offending path or identifier on ``self.target`` and renders it
    as part of the error message so every subclass surfaces what went wrong.
    """

    #: Human-readable prefix describing the failure category.
    _prefix = "Dataset foundation error"

    #: Label describing what ``target`` represents (e.g. "path", "parameter").
    _target_label = "target"

    def __init__(self, target: object, message: str | None = None) -> None:
        self.target = target
        self.message = message
        super().__init__(self._render())

    def _render(self) -> str:
        detail = f" ({self.message})" if self.message else ""
        return f"{self._prefix} [{self._target_label}={self.target!s}]{detail}"


class ConfigParseError(DatasetFoundationError):
    """The configuration file could not be parsed. Names the file path (Req 9.6)."""

    _prefix = "Failed to parse configuration file"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class ConfigValidationError(DatasetFoundationError):
    """A configuration parameter is missing or invalid. Names the parameter (Req 9.2)."""

    _prefix = "Invalid configuration"
    _target_label = "parameter"

    def __init__(self, parameter: object, message: str | None = None) -> None:
        self.parameter = parameter
        super().__init__(parameter, message)


class PlyReadError(DatasetFoundationError):
    """A PLY file is missing or unreadable. Names the file path (Req 2.5)."""

    _prefix = "Failed to read PLY file"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class LayoutError(DatasetFoundationError):
    """A required subdirectory could not be created. Names the subdir (Req 1.3)."""

    _prefix = "Failed to initialize dataset layout"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class WriteError(DatasetFoundationError):
    """An output artifact could not be written. Names the target path (Req 1.8, 5.9, 7.6)."""

    _prefix = "Failed to write output"
    _target_label = "path"

    def __init__(self, path: object, message: str | None = None) -> None:
        self.path = path
        super().__init__(path, message)


class NormalizationError(DatasetFoundationError):
    """Standardization failed for a fragment (e.g. invalid scale factor).

    Names the Fragment_ID and, when relevant, the invalid scale factor (Req 5.7).
    """

    _prefix = "Failed to standardize fragment"
    _target_label = "fragment_id"

    def __init__(
        self,
        fragment_id: object,
        message: str | None = None,
        scale_factor: object | None = None,
    ) -> None:
        self.fragment_id = fragment_id
        self.scale_factor = scale_factor
        if scale_factor is not None:
            factor_note = f"invalid scale_factor={scale_factor!s}"
            message = f"{message}; {factor_note}" if message else factor_note
        super().__init__(fragment_id, message)


class AlignmentImportError(DatasetFoundationError):
    """A precomputed transform failed validation on import. Names the Fragment_ID (Req 3.8)."""

    _prefix = "Failed to import precomputed transform"
    _target_label = "fragment_id"

    def __init__(self, fragment_id: object, message: str | None = None) -> None:
        self.fragment_id = fragment_id
        super().__init__(fragment_id, message)
