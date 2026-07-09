"""Structured logging for the Baseline Geometry pipeline (Phase 4).

Self-contained helpers built on the standard-library :mod:`logging` module,
mirroring the Phase 1/2 pattern: a single console handler with a consistent
formatter on the package logger, namespaced child loggers, and a structured
``log_stage`` record carrying the stage name and a success/failure status.
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

ROOT_LOGGER_NAME = "baseline_geometry"

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class StageStatus(str, Enum):
    """Completion status for a pipeline stage record."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> logging.Logger:
    """Configure and return the package root logger (idempotent)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger under the package root."""
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
    """Emit a structured stage record (stage + status + optional fields)."""
    status_value = status.value if isinstance(status, StageStatus) else str(status)
    normalized = _normalize_status(status_value)

    if level is None:
        level = logging.INFO if normalized is not StageStatus.FAILURE else logging.ERROR

    parts = [f"stage={stage}", f"status={status_value}"]
    if fragment_id is not None:
        parts.append(f"fragment_id={fragment_id}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    if message:
        parts.append(f"message={message}")

    logger.log(level, " ".join(parts), extra={"stage": stage, "status": status_value})


def _normalize_status(status_value: str) -> StageStatus:
    try:
        return StageStatus(status_value)
    except ValueError:
        return StageStatus.FAILURE
