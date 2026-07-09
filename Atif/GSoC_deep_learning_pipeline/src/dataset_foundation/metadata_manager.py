"""Metadata construction, persistence, and parsing.

Builds per-fragment and dataset-level Metadata records, writes them to disk as
JSON, and parses them back. The records follow the design "Metadata Schema"
exactly, and every value stored is a JSON-native type (``str``, ``int``,
``float``, ``bool``, ``None``, ``list``, ``dict``) so that

    read_metadata(write_metadata(path, record)) == record

holds field-for-field (Req 7.5). NumPy scalars/arrays and tuples are coerced to
plain Python ``int``/``float``/``list`` before serialization; consumers pass
Config objects (or their ``dataclasses.asdict`` form) which are normalized the
same way.

Writes are atomic: the JSON is first written to a sibling temporary file which
is then ``os.replace``'d onto the target path. A failure therefore names the
path (:class:`WriteError`) and never corrupts or deletes previously written
records (Req 1.8).

Requirements: 1.5, 1.6, 1.7, 1.8, 7.4, 7.5, 9.5.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import Any, Mapping, Sequence

from dataset_foundation.errors import WriteError

__all__ = [
    "build_fragment_metadata",
    "build_dataset_metadata",
    "write_metadata",
    "read_metadata",
    "to_json_native",
]


# ---------------------------------------------------------------------------
# JSON-native coercion
# ---------------------------------------------------------------------------


def to_json_native(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-native Python types.

    NumPy scalars become ``int``/``float``/``bool``, NumPy arrays and tuples
    become (nested) lists, mappings become plain ``dict`` with string keys, and
    dataclass instances are expanded via :func:`dataclasses.asdict`. The result
    contains only ``str``, ``int``, ``float``, ``bool``, ``None``, ``list`` and
    ``dict`` so it survives a JSON serialize/parse round-trip unchanged (Req 7.5).
    """
    # None / bool must be checked before the numeric branch (bool is an int).
    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        return float(value)

    # Dataclass instances (e.g. Config) -> dict, then normalize.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_json_native(dataclasses.asdict(value))

    # Mappings -> dict with stringified keys.
    if isinstance(value, Mapping):
        return {str(key): to_json_native(item) for key, item in value.items()}

    # NumPy support without importing numpy at module import time.
    item_method = getattr(value, "item", None)
    if item_method is not None and _is_numpy_scalar(value):
        return to_json_native(item_method())

    tolist_method = getattr(value, "tolist", None)
    if tolist_method is not None and not isinstance(value, (str, bytes)):
        # NumPy arrays and other array-likes exposing tolist().
        return to_json_native(tolist_method())

    # Generic sequences (list, tuple) -> list.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_json_native(item) for item in value]

    # Fall back to float for any remaining real-number-like objects.
    if isinstance(value, (int, float)):
        return float(value)

    return value


def _is_numpy_scalar(value: Any) -> bool:
    """Return True for a 0-d numpy scalar (has ``item`` and ``ndim == 0``)."""
    ndim = getattr(value, "ndim", None)
    return ndim == 0


# ---------------------------------------------------------------------------
# Per-fragment metadata
# ---------------------------------------------------------------------------


def build_fragment_metadata(
    *,
    fragment_id: str,
    source_filename: str,
    vertex_count: int,
    face_count: int,
    point_count_after_normalization: int,
    is_empty: bool,
    transform_file: str | None,
    alignment_method: str,
    voxel_size_mm: float,
    alignment_error: Mapping[str, Any] | None = None,
    flagged_for_review: bool = False,
    skipped: bool = False,
    failure_reason: str | None = None,
    degenerate_normal_count: int = 0,
    centering_offset: Sequence[float] | None = None,
    scale_factor: float = 1.0,
    artifact_paths: Mapping[str, Any] | None = None,
    units: str = "mm",
) -> dict[str, Any]:
    """Build a per-fragment Metadata record matching the design schema.

    Carries every design field, including for empty fragments (``is_empty`` /
    ``skipped`` true, zero counts). All values are coerced to JSON-native types
    so the record round-trips through :func:`write_metadata` / :func:`read_metadata`.

    ``alignment_error`` is a mapping with ``inlier_rmse_mm``, ``fitness`` and
    ``mean_surface_distance_mm`` (or ``None`` when unavailable). ``artifact_paths``
    maps ``aligned`` / ``normals`` / ``normalized`` to their relative paths.

    Requirements: 1.5, 1.6, 1.7, 7.4, 7.5.
    """
    record: dict[str, Any] = {
        "fragment_id": fragment_id,
        "source_filename": source_filename,
        "units": units,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "point_count_after_normalization": point_count_after_normalization,
        "is_empty": is_empty,
        "transform_file": transform_file,
        "alignment_method": alignment_method,
        "voxel_size_mm": voxel_size_mm,
        "alignment_error": _build_alignment_error(alignment_error),
        "flagged_for_review": flagged_for_review,
        "skipped": skipped,
        "failure_reason": failure_reason,
        "degenerate_normal_count": degenerate_normal_count,
        "centering_offset": list(centering_offset) if centering_offset is not None else [0.0, 0.0, 0.0],
        "scale_factor": scale_factor,
        "artifact_paths": {
            "aligned": None,
            "normals": None,
            "normalized": None,
            **(dict(artifact_paths) if artifact_paths is not None else {}),
        },
    }
    return to_json_native(record)


def _build_alignment_error(alignment_error: Mapping[str, Any] | None) -> Any:
    """Normalize the alignment-error sub-record (or ``None``)."""
    if alignment_error is None:
        return None
    return {
        "inlier_rmse_mm": alignment_error.get("inlier_rmse_mm"),
        "fitness": alignment_error.get("fitness"),
        "mean_surface_distance_mm": alignment_error.get("mean_surface_distance_mm"),
    }


# ---------------------------------------------------------------------------
# Dataset-level metadata
# ---------------------------------------------------------------------------


def build_dataset_metadata(
    *,
    run_timestamp: str,
    config_used: Any,
    fragment_ids: Sequence[str],
    id_to_filename: Mapping[str, str],
    fragments: Mapping[str, Mapping[str, Any]],
    full_model: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the dataset-level Metadata record matching the design schema.

    Carries the run timestamp, the serialized Config (provenance, Req 9.5), the
    ordered Fragment_IDs, the bidirectional id<->filename map (Req 1.5), per-
    fragment artifact paths plus alignment method (Req 7.4), the full-model
    counts, and the reconstruction-validation results (Req 8.x). ``config_used``
    may be a :class:`Config` dataclass, its ``asdict`` form, or any mapping; it
    is normalized to JSON-native types.

    Requirements: 1.5, 1.6, 1.7, 7.4, 9.5.
    """
    record: dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "config_used": config_used,
        "fragment_ids": list(fragment_ids),
        "id_to_filename": dict(id_to_filename),
        "fragments": {fid: dict(entry) for fid, entry in fragments.items()},
        "full_model": dict(full_model),
        "reconstruction": dict(reconstruction),
    }
    return to_json_native(record)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_metadata(path: str, record: Mapping[str, Any]) -> str:
    """Serialize ``record`` to JSON at ``path`` atomically.

    The JSON is written to a temporary file in the same directory and then
    atomically moved onto ``path`` via :func:`os.replace`. If anything fails the
    temporary file is removed and a :class:`WriteError` naming ``path`` is
    raised; any previously written file at ``path`` is left untouched (Req 1.8).

    Returns the path written to (for convenience in call chains).
    """
    native = to_json_native(record)
    directory = os.path.dirname(os.path.abspath(path))

    tmp_path: str | None = None
    try:
        os.makedirs(directory, exist_ok=True)
        # Serialize first so a serialization failure never touches the target.
        payload = json.dumps(native, indent=2, sort_keys=False, ensure_ascii=False)

        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)
        tmp_path = None  # ownership transferred to the target path
    except (OSError, TypeError, ValueError) as exc:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise WriteError(path, str(exc)) from exc

    return path


def read_metadata(path: str) -> dict[str, Any]:
    """Parse a JSON Metadata file at ``path`` into a dict.

    The returned dict contains only JSON-native types and equals the record
    originally passed to :func:`write_metadata` field-for-field (Req 7.5).
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
