"""Input loading from Phase 1 outputs.

Reads the Fragment enumeration from ``dataset/metadata/dataset.json`` and, for
each Fragment_ID, the density-standardized Point_Cloud from
``dataset/normalized/<id>.ply`` (via
:func:`dataset_foundation.ply_io.read_point_cloud`). Points come from that
cloud's ``.points`` and the per-point Normals come from that same cloud's
``.normals``.

Normals source note (deviation from the design's stated file mapping):
    The design text says normals should be read from a separate
    ``dataset/normals/<id>.ply``. On the actual Phase 1 output, however, that
    directory holds the *pre-downsampling* (denser) cloud whose point count
    never matches the density-standardized ``normalized`` cloud (e.g. 4474 vs
    2973), which would violate Req 1.2/1.4 for every Fragment. Phase 1's
    voxel-downsampling preserves per-point normals, so the ``normalized`` cloud
    already carries exactly one normal per (downsampled) point. We therefore
    take both points and normals from the ``normalized`` cloud, which is the
    only self-consistent pairing (matched counts, one normal per source index)
    and satisfies Req 1.1's intent, Req 1.2, and Req 4.4. This choice was
    confirmed with the maintainer.

Error discipline (per design "Error Handling"): input problems for a single
Fragment are **per-Fragment and non-fatal**. Rather than raising,
:func:`load_fragment` returns a :class:`LoadedFragment` with its ``error``
field set (naming the offending path and Fragment_ID) so the pipeline can
record the failure, produce no patches for that Fragment, and continue with the
others. The Phase 1 reader raises :class:`dataset_foundation.errors.PlyReadError`
on a missing/unreadable file; that is caught here and converted into an
``error`` record.

Behavior summary:
    * Missing/unreadable ``normalized`` file -> ``error`` set (naming the path +
      Fragment_ID), ``skipped=False``, no usable points (Req 1.3).
    * Loaded normals count must equal loaded points count; a mismatch (including
      a non-empty cloud that carries no normals) -> ``error`` set naming the
      Fragment_ID, no usable points (Req 1.2, 1.4).
    * A zero-point normalized cloud -> ``skipped=True``, ``point_count=0``,
      ``error=None`` (vacuously covered; Req 1.5).
    * The loaded point count is always recorded on ``point_count`` (Req 1.6).

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from dataset_foundation.errors import PlyReadError
from dataset_foundation.metadata_manager import read_metadata
from dataset_foundation.ply_io import read_point_cloud

from patch_generation.errors import InputLoadError  # noqa: F401  (typed error re-export)

__all__ = [
    "LoadedFragment",
    "enumerate_fragment_ids",
    "load_fragment",
    "load_all",
]

# Subdirectory of the Dataset_Directory holding the density-standardized clouds
# consumed as Phase 2 inputs. These clouds carry their own per-point normals.
_NORMALIZED_SUBDIR = "normalized"
_DATASET_METADATA = os.path.join("metadata", "dataset.json")


def _empty_points() -> np.ndarray:
    """Canonical ``(0, 3)`` float64 array for fragments with no usable points."""
    return np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True)
class LoadedFragment:
    """An in-memory Fragment loaded from Phase 1 outputs.

    Attributes:
        fragment_id: The Phase 1 Fragment_ID this record describes.
        points: ``(N, 3)`` float64 coordinates in the Full_Model frame (mm).
            Empty ``(0, 3)`` when the Fragment was skipped or failed to load.
        normals: ``(N, 3)`` float64 unit normals aligned one-per-point with
            ``points``. Empty ``(0, 3)`` when skipped or failed.
        point_count: The number of loaded points (Req 1.6). Zero for skipped or
            failed loads.
        skipped: ``True`` when the normalized cloud has zero points and patch
            generation is skipped for this Fragment (Req 1.5).
        error: ``None`` on success/skip; otherwise a message naming the
            offending path and/or Fragment_ID (Req 1.3, 1.4). When set, no
            patches are produced for this Fragment.
    """

    fragment_id: str
    points: np.ndarray = field(default_factory=_empty_points)
    normals: np.ndarray = field(default_factory=_empty_points)
    point_count: int = 0
    skipped: bool = False
    error: str | None = None


def enumerate_fragment_ids(dataset_dir: str) -> list[str]:
    """Return the ordered Fragment_IDs enumerated by the Phase 1 dataset metadata.

    Reads ``<dataset_dir>/metadata/dataset.json`` and returns its
    ``fragment_ids`` list (Req 1.1).

    Args:
        dataset_dir: The Phase 1 output directory (``dataset/``).

    Returns:
        The list of Fragment_IDs in enumeration order (empty if none present).
    """
    metadata_path = os.path.join(dataset_dir, _DATASET_METADATA)
    record = read_metadata(metadata_path)
    fragment_ids = record.get("fragment_ids", [])
    return [str(fid) for fid in fragment_ids]


def load_fragment(dataset_dir: str, fragment_id: str) -> LoadedFragment:
    """Load one Fragment's points and normals from Phase 1 outputs.

    Both points and normals are read from
    ``<dataset_dir>/normalized/<fragment_id>.ply`` -- the density-standardized
    cloud, which carries one normal per point (see the module docstring for why
    the separate ``normals/`` directory is not used). This function does not
    raise for per-Fragment input problems; instead it returns a
    :class:`LoadedFragment` whose ``error`` names what failed (Req 1.3, 1.4), so
    the pipeline can skip the Fragment and continue.

    Args:
        dataset_dir: The Phase 1 output directory (``dataset/``).
        fragment_id: The Fragment_ID to load.

    Returns:
        A :class:`LoadedFragment`: populated with matched points/normals on
        success; ``skipped=True`` for a zero-point cloud (Req 1.5); or with
        ``error`` set for a missing/unreadable file or a point/normal count
        mismatch (Req 1.3, 1.4).
    """
    normalized_path = os.path.join(
        dataset_dir, _NORMALIZED_SUBDIR, f"{fragment_id}.ply"
    )

    # --- Read the normalized point cloud (points + normals). --------------
    try:
        normalized_pc = read_point_cloud(normalized_path)
    except PlyReadError as exc:
        return LoadedFragment(
            fragment_id=fragment_id,
            error=str(InputLoadError(normalized_path, fragment_id, str(exc))),
        )

    points = np.ascontiguousarray(normalized_pc.points, dtype=np.float64)
    point_count = int(points.shape[0])

    # A zero-point Fragment is skipped (vacuously fully covered) with no error
    # and produces zero patches downstream (Req 1.5).
    if point_count == 0:
        return LoadedFragment(fragment_id=fragment_id, point_count=0, skipped=True)

    # ``.normals`` is ``None`` when the cloud carries no normal vectors; for a
    # non-empty Fragment that is a count mismatch (0 != point_count).
    if normalized_pc.normals is None:
        normals = _empty_points()
    else:
        normals = np.ascontiguousarray(normalized_pc.normals, dtype=np.float64)
    normal_count = int(normals.shape[0])

    # The number of loaded normals must equal the number of loaded points
    # (Req 1.2); a mismatch is a per-Fragment error naming the Fragment_ID
    # (Req 1.4).
    if normal_count != point_count:
        message = (
            f"point/normal count mismatch: {point_count} points vs "
            f"{normal_count} normals in {normalized_path}"
        )
        return LoadedFragment(
            fragment_id=fragment_id,
            point_count=point_count,
            error=str(InputLoadError(normalized_path, fragment_id, message)),
        )

    return LoadedFragment(
        fragment_id=fragment_id,
        points=points,
        normals=normals,
        point_count=point_count,
        skipped=False,
        error=None,
    )


def load_all(dataset_dir: str) -> list[LoadedFragment]:
    """Load every Fragment enumerated by the Phase 1 dataset metadata.

    Enumerates Fragment_IDs via :func:`enumerate_fragment_ids`, then loads each
    with :func:`load_fragment`. Per-Fragment failures are captured on each
    returned record's ``error``/``skipped`` fields rather than raising, so the
    caller receives one :class:`LoadedFragment` per enumerated Fragment_ID
    regardless of individual load outcomes.

    Args:
        dataset_dir: The Phase 1 output directory (``dataset/``).

    Returns:
        One :class:`LoadedFragment` per enumerated Fragment_ID, in order.
    """
    return [
        load_fragment(dataset_dir, fragment_id)
        for fragment_id in enumerate_fragment_ids(dataset_dir)
    ]
