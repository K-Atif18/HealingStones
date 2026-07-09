"""Rigid-transform file persistence.

Write a 4x4 transform as a 16-value plain-text matrix (full float64 precision)
and read it back; round-trippable to 1e-6 per element. Implemented in task 8.3.

Responsibilities (design "Rigid_Transform" data model):
    * ``write_transform`` -- persist a 4x4 homogeneous matrix as a plain-text
                             file of 16 values at full float64 precision under
                             ``transforms/<fragment_id>.txt``. A failure to
                             write raises :class:`WriteError` naming the path
                             (Req 3.6).
    * ``read_transform``  -- read the persisted matrix back as a ``(4, 4)``
                             float64 array. Reading a written matrix reproduces
                             it within 1e-6 per element (Req 3.7). A missing or
                             malformed file raises a suitable error naming the
                             path.

Requirements: 3.6, 3.7.
"""

from __future__ import annotations

import os

import numpy as np

from .errors import WriteError

__all__ = ["write_transform", "read_transform"]

# Full float64 precision: 17 significant decimal digits round-trips a double
# exactly, so ``%.18e`` guarantees the written text reproduces every element.
_FLOAT_FMT = "%.18e"


def write_transform(path: str, matrix: np.ndarray) -> None:
    """Write a 4x4 rigid transform to ``path`` as 16 plain-text values.

    The matrix is written at full float64 precision so that reading it back
    reproduces every element within 1e-6 (Req 3.7).

    Args:
        path: Destination file path (e.g. ``transforms/<fragment_id>.txt``).
        matrix: A 4x4 homogeneous transformation matrix.

    Raises:
        WriteError: If ``matrix`` is not 4x4 or the file cannot be written;
            the error names ``path``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise WriteError(path, f"expected a 4x4 matrix, got shape {arr.shape}")
    try:
        np.savetxt(path, arr, fmt=_FLOAT_FMT)
    except OSError as exc:
        raise WriteError(path, str(exc)) from exc


def read_transform(path: str) -> np.ndarray:
    """Read a 4x4 rigid transform previously written by :func:`write_transform`.

    Args:
        path: Path to the plain-text transform file.

    Returns:
        The transform as a ``(4, 4)`` float64 :class:`numpy.ndarray`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain exactly 16 numeric values;
            the error names ``path``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"transform file not found: {path}")
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"malformed transform file: {path} ({exc})") from exc
    if arr.size != 16:
        raise ValueError(
            f"malformed transform file: {path} "
            f"(expected 16 values, got {arr.size})"
        )
    return arr.reshape(4, 4)
