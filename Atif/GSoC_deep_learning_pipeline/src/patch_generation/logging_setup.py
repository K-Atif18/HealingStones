"""Structured logging setup for the Patch Generation pipeline (Phase 2).

Self-contained helpers built on the standard-library :mod:`logging` module,
mirroring the Phase 1 (Dataset Foundation) pattern.

This module provides:

- :func:`configure_logging` to install a console handler with a consistent,
  structured formatter on the package logger (idempotent).
- :func:`get_logger` to obtain a namespaced logger for a component.
- :func:`log_stage` to emit a structured stage record carrying the stage name,
  the Fragment_ID (where applicable), and a success/failure status.

On each stage success the pipeline emits a record with the stage name, the
Fragment_ID (where applicable), and a success status (Req 10.4); on failure the
same fields are emitted with a failure status (Req 10.5).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

__all__ = [
    "StageStatus",
    "configure_logging",
    "get_logger",
    "log_stage",
]

# Root logger name for the package. Child loggers are namespaced beneath this
# so that a single ``configure_logging`` call governs the whole pipeline.
ROOT_LOGGER_NAME = "patch_generation"

# Format includes the timestamp, level, logger name, and message. Structured
# stage fields (stage / fragment_id / status) are rendered into the message by
# ``log_stage`` so they appear consistently regardless of handler.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class StageStatus(str, Enum):
    """Completion status for a pipeline stage record."""

    SUCCESS = "success"
    FAILURE = "failure"


def configure_logging(
    level: int = logging.INFO,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure and return the package root logger.

    Installs a single :class:`logging.StreamHandler` with the structured
    formatter on the ``patch_generation`` logger. The call is idempotent: a
    handler is only added if one is not already present (unless ``force`` is
    set, in which case existing handlers are replaced).

    Args:
        level: Logging level for the package logger.
        force: When ``True``, remove any existing handlers before adding the
            configured one.

    Returns:
        The configured package root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    # Avoid duplicate emission through the ancestor root logger.
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger under the package root.

    Args:
        name: Optional component name (e.g. ``"patch_extractor"``). When omitted
            the package root logger is returned. Fully-qualified names already
            under the package root are returned unchanged.

    Returns:
        A :class:`logging.Logger` namespaced beneath ``patch_generation``.
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def log_stage(
    logger: logging.Logger,
    stage: str,
    status: StageStatus | str,
    *,
    fragment_id: Optional[str] = None,
    level: Optional[int] = None,
    message: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emit a structured stage record.

    The record always carries the ``stage`` name and the ``status``. The
    Fragment_ID is included when applicable (i.e. when ``fragment_id`` is
    provided); dataset-level stages omit it. Additional structured fields may be
    supplied via keyword arguments and are appended to the record.

    Success records are emitted at ``INFO`` by default and failure records at
    ``ERROR`` by default (Req 10.4, 10.5); an explicit ``level`` overrides this.

    Args:
        logger: The logger to emit through (typically from :func:`get_logger`).
        stage: The pipeline stage name (e.g. ``"extract"``, ``"validate"``).
        status: Completion status, a :class:`StageStatus` or its string value
            (``"success"`` / ``"failure"``).
        fragment_id: The Fragment_ID this record pertains to, where applicable.
        level: Optional explicit logging level overriding the status default.
        message: Optional human-readable detail appended to the record.
        **fields: Arbitrary extra structured fields (e.g. ``patch_count``).
    """
    status_value = status.value if isinstance(status, StageStatus) else str(status)
    normalized_status = _normalize_status(status_value)

    if level is None:
        level = (
            logging.INFO
            if normalized_status is StageStatus.SUCCESS
            else logging.ERROR
        )

    parts = [f"stage={stage}", f"status={status_value}"]
    if fragment_id is not None:
        parts.append(f"fragment_id={fragment_id}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    if message:
        parts.append(f"message={message}")

    record_text = " ".join(parts)

    # Attach structured fields to the LogRecord via ``extra`` so downstream
    # handlers/formatters can consume them programmatically if desired.
    extra = {"stage": stage, "status": status_value}
    if fragment_id is not None:
        extra["fragment_id"] = fragment_id

    logger.log(level, record_text, extra=extra)


def _normalize_status(status_value: str) -> StageStatus:
    """Map a status string to a :class:`StageStatus`, defaulting to failure.

    Any value that is not explicitly the success status is treated as a failure
    for the purposes of selecting the default log level, so unexpected statuses
    are surfaced loudly rather than silently logged as informational.
    """
    try:
        return StageStatus(status_value)
    except ValueError:
        if status_value.lower() == StageStatus.SUCCESS.value:
            return StageStatus.SUCCESS
        return StageStatus.FAILURE
