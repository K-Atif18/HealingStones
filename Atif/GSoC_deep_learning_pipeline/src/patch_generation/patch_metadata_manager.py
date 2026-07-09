"""Per-Fragment and dataset-level Patch_Metadata management.

Builds per-Fragment and dataset-level Patch_Metadata records, writes them to
disk as JSON, and parses them back. The records follow the design
"Patch_Metadata" schema exactly, and every value stored is a JSON-native type
(``str``, ``int``, ``float``, ``bool``, ``None``, ``list``, ``dict``) so that

    read_metadata(write_metadata(path, record)) == to_json_native(record)

holds field-for-field (Req 7.4).

Phase 1 code is reused directly: :func:`dataset_foundation.metadata_manager.to_json_native`
performs the JSON-native coercion (NumPy scalars/arrays -> ``int``/``float``/``list``,
dataclasses such as :class:`~patch_generation.coverage_analyzer.CoverageResult`,
:class:`~patch_generation.overlap_analyzer.OverlapStats`, and ``SizeSummary`` ->
``dict`` via :func:`dataclasses.asdict`), and its atomic ``write_metadata`` /
``read_metadata`` provide the on-disk persistence (temp file + ``os.replace``).

The re-exported ``write_metadata``/``read_metadata`` therefore inherit the
Phase 1 atomic-write discipline. On write failure the Phase 1 implementation
raises :class:`dataset_foundation.errors.WriteError` naming the target path;
this module wraps that into :class:`patch_generation.errors.WriteError` so
callers can catch the Phase 2 error type while the offending path is still
named (Req 7.6).

Requirements: 1.5, 1.6, 5.5, 6.4, 6.6, 7.2, 7.4, 9.5, 10.6.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dataset_foundation.errors import WriteError as _DFWriteError
from dataset_foundation.metadata_manager import read_metadata as _read_metadata
from dataset_foundation.metadata_manager import to_json_native
from dataset_foundation.metadata_manager import write_metadata as _write_metadata

from patch_generation.errors import WriteError

__all__ = [
    "build_fragment_patch_metadata",
    "build_dataset_patch_metadata",
    "write_metadata",
    "read_metadata",
    "to_json_native",
]


# ---------------------------------------------------------------------------
# Field access helpers (accept dataclass instances OR plain mappings)
# ---------------------------------------------------------------------------


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from ``source`` whether it is a mapping or an object.

    Supports the analysis result dataclasses (``CoverageResult``,
    ``OverlapStats``, ``SizeSummary``) as well as plain ``dict`` records, so
    builders work regardless of whether callers pass the frozen dataclasses or
    already-serialized mappings.
    """
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _overlap_summary(overlap: Any, *, full: bool) -> dict[str, Any]:
    """Build the overlap sub-record.

    ``full`` includes ``analyzed_pair_count`` (per-Fragment record); the
    dataset-level per-fragment summary omits it (Req 5.5).
    """
    summary: dict[str, Any] = {
        "mean_overlap": _field(overlap, "mean_overlap", 0.0),
        "max_overlap": _field(overlap, "max_overlap", 0.0),
    }
    if full:
        summary = {
            "analyzed_pair_count": _field(overlap, "analyzed_pair_count", 0),
            **summary,
        }
    return summary


def _size_summary(size_summary: Any) -> dict[str, Any]:
    """Build the size-distribution sub-record (Req 9.5)."""
    return {
        "min_size": _field(size_summary, "min_size"),
        "max_size": _field(size_summary, "max_size"),
        "mean_size": _field(size_summary, "mean_size"),
        "median_size": _field(size_summary, "median_size"),
    }


# ---------------------------------------------------------------------------
# Per-Fragment metadata
# ---------------------------------------------------------------------------


def build_fragment_patch_metadata(
    *,
    fragment_id: str,
    point_count: int,
    skipped: bool = False,
    error: str | None = None,
    patch_count: int = 0,
    coverage: Any = None,
    overlap: Any = None,
    size_summary: Any = None,
    coverage_pass: bool = False,
    size_pass: bool = False,
    offending_bound: str | None = None,
    artifact_path: str | None = None,
) -> dict[str, Any]:
    """Build a per-Fragment Patch_Metadata record matching the design schema.

    Carries the loaded point count (Req 1.6) and skipped/error status for
    skipped or failed fragments (Req 1.5); the patch count; the coverage fields
    including the uncovered index list (Req 6.4, 6.6); the overlap summary
    (Req 5.5); the size-distribution summary (Req 9.5); the coverage/size pass
    flags and the offending bound on size failure; and the Patch_Record
    artifact path (Req 7.2).

    ``coverage`` may be a :class:`~patch_generation.coverage_analyzer.CoverageResult`
    (or mapping) exposing ``coverage_fraction``, ``covered_count``,
    ``uncovered_count``, ``uncovered_indices``. ``overlap`` may be an
    :class:`~patch_generation.overlap_analyzer.OverlapStats` (or mapping) and
    ``size_summary`` a ``SizeSummary`` (or mapping). All values are coerced to
    JSON-native types so the record round-trips through :func:`write_metadata`
    / :func:`read_metadata` field-for-field (Req 7.4).

    Requirements: 1.5, 1.6, 5.5, 6.4, 6.6, 7.2, 9.5.
    """
    uncovered_indices = _field(coverage, "uncovered_indices", [])

    record: dict[str, Any] = {
        "fragment_id": fragment_id,
        "point_count": point_count,
        "skipped": skipped,
        "error": error,
        "patch_count": patch_count,
        "coverage_fraction": _field(coverage, "coverage_fraction", 0.0),
        "covered_count": _field(coverage, "covered_count", 0),
        "uncovered_count": _field(coverage, "uncovered_count", 0),
        "uncovered_indices": uncovered_indices if uncovered_indices is not None else [],
        "overlap": _overlap_summary(overlap, full=True),
        "size_summary": _size_summary(size_summary),
        "coverage_pass": coverage_pass,
        "size_pass": size_pass,
        "offending_bound": offending_bound,
        "artifact_path": artifact_path,
    }
    return to_json_native(record)


# ---------------------------------------------------------------------------
# Dataset-level metadata
# ---------------------------------------------------------------------------


def build_dataset_patch_metadata(
    *,
    run_timestamp: str,
    config_used: Any,
    fragment_ids: Sequence[str],
    fragments: Mapping[str, Mapping[str, Any]],
    overall_validation_pass: bool,
) -> dict[str, Any]:
    """Build the dataset-level Patch_Metadata record matching the design schema.

    Carries the run timestamp and the serialized Config for provenance
    (Req 10.6), the ordered Fragment_IDs, the overall validation pass flag
    (Req 9.5), and a per-fragment summary map enumerating every processed
    Fragment with its patch count, coverage fraction (Req 6.6), overlap summary
    (Req 5.5), size summary, coverage/size pass flags, and artifact + metadata
    paths (Req 7.2).

    ``config_used`` may be a :class:`Config` dataclass, its ``asdict`` form, or
    any mapping; it is normalized to JSON-native types via
    :func:`to_json_native`. Each entry in ``fragments`` may carry the raw
    analysis dataclasses (``OverlapStats`` / ``SizeSummary``) under
    ``overlap`` / ``size_summary`` or already-built mappings; both are
    normalized to the dataset-level schema.

    Requirements: 5.5, 6.6, 7.2, 9.5, 10.6.
    """
    fragment_summaries: dict[str, Any] = {}
    for fid, entry in fragments.items():
        fragment_summaries[fid] = {
            "patch_count": _field(entry, "patch_count", 0),
            "coverage_fraction": _field(entry, "coverage_fraction", 0.0),
            "overlap": _overlap_summary(_field(entry, "overlap"), full=False),
            "size_summary": _size_summary(_field(entry, "size_summary")),
            "coverage_pass": _field(entry, "coverage_pass", False),
            "size_pass": _field(entry, "size_pass", False),
            "artifact_path": _field(entry, "artifact_path"),
            "metadata_path": _field(entry, "metadata_path"),
        }

    record: dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "config_used": config_used,
        "fragment_ids": list(fragment_ids),
        "overall_validation_pass": overall_validation_pass,
        "fragments": fragment_summaries,
    }
    return to_json_native(record)


# ---------------------------------------------------------------------------
# Persistence (reuses the Phase 1 atomic write/read)
# ---------------------------------------------------------------------------


def write_metadata(path: str, record: Mapping[str, Any]) -> str:
    """Serialize ``record`` to JSON at ``path`` atomically.

    Delegates to :func:`dataset_foundation.metadata_manager.write_metadata`,
    which coerces the record with :func:`to_json_native`, writes it to a sibling
    temporary file, and atomically ``os.replace``'s it onto ``path`` (any
    previously written file is left untouched on failure).

    A Phase 1 :class:`dataset_foundation.errors.WriteError` is re-raised as a
    Phase 2 :class:`patch_generation.errors.WriteError` naming ``path`` so
    callers catch the Phase 2 type while the offending path is still reported
    (Req 7.6).

    Returns the path written to (for convenience in call chains).
    """
    try:
        return _write_metadata(path, record)
    except _DFWriteError as exc:
        raise WriteError(path, exc.message) from exc


def read_metadata(path: str) -> dict[str, Any]:
    """Parse a JSON Patch_Metadata file at ``path`` into a dict.

    Delegates to :func:`dataset_foundation.metadata_manager.read_metadata`. The
    returned dict contains only JSON-native types and equals
    ``to_json_native(record)`` of the record originally written field-for-field
    (Req 7.4).
    """
    return _read_metadata(path)
