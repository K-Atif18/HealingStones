"""Density standardization and coordinate normalization.

Voxel downsampling for density standardization; optional centering/scaling.
Implemented in tasks 11.1/11.4. Requirements: 5.

Voxel downsampling (design "Density Standardization" section) partitions space
into cubic voxels of edge ``voxel_size_mm`` and replaces the points falling in
each occupied voxel with their centroid. Two guarantees matter downstream:

* **Non-increasing count (Req 5.2).** Each output point corresponds to exactly
  one occupied voxel, so the output count never exceeds the input count.
* **Exact idempotence (Req 5.3).** Re-downsampling an already-downsampled cloud
  at the same voxel size must leave the point count unchanged, to a tolerance of
  0 points.

Open3D's ``voxel_down_sample`` anchors its voxel grid on the *per-cloud* min
bound. That origin shifts after the first pass (the surviving centroids have a
different min bound than the original points), which can split or merge voxels
on a second pass and break exact idempotence. To avoid this we anchor the grid
to a fixed global origin at ``(0, 0, 0)`` and implement the centroid-per-voxel
reduction directly in NumPy. With a fixed origin the voxel index of a point is
``floor(point / voxel_size)``; the centroid of the points in voxel ``k`` lies
inside ``[k * voxel_size, (k + 1) * voxel_size)`` and therefore hashes back to
the same voxel ``k`` on a second pass -- one point in, one point out, so the
count (and the points themselves) are stable. Output points are sorted by voxel
index for deterministic, reproducible ordering.

Coordinate normalization (design "Coordinate normalization" bullet) runs after
density standardization. It applies an optional centering translation followed
by an optional positive scaling, computing each output point as
``(point - centering_offset) * scale_factor``. Both steps are opt-in via the
Configuration; the centering offset may be zero or negative (Req 5.5) while an
enabled scale factor must be strictly positive (Req 5.7). Millimeter units are
preserved throughout -- no implicit unit conversion is performed (Req 5.4).

``standardize`` is deliberately pure: it returns the normalized cloud together
with the offset and scale factor that were actually applied so the pipeline can
record them in per-fragment Metadata (Req 5.5). Writing the result to the
``normalized/`` subdirectory is delegated to :func:`write_normalized`, a thin
wrapper over :func:`dataset_foundation.ply_io.write_point_cloud`; on failure
that writer raises :class:`WriteError` naming the target path (Req 5.9), so the
pipeline (task 16.1) calls ``standardize`` then ``write_normalized``.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9.
"""

from __future__ import annotations

import numpy as np

from .config_loader import Config
from .errors import NormalizationError
from .geometry import PointCloud
from .ply_io import write_point_cloud

__all__ = ["voxel_downsample", "standardize", "write_normalized"]


def voxel_downsample(pc: PointCloud, voxel_size_mm: float) -> PointCloud:
    """Voxel-downsample ``pc`` to standardize point density.

    Points are grouped into cubic voxels of edge ``voxel_size_mm`` anchored to a
    fixed origin at ``(0, 0, 0)``, and each occupied voxel is replaced by the
    centroid of its points. Per-point colors and normals, when present, are
    averaged per voxel (averaged normals are renormalized to unit length).
    Output points are ordered deterministically by ascending voxel index.

    Args:
        pc: Input point cloud (coordinates in millimeters).
        voxel_size_mm: Positive voxel edge length in millimeters.

    Returns:
        A new :class:`PointCloud` whose point count is ``<=`` the input count.
        An empty input yields an empty cloud. Re-downsampling the result at the
        same voxel size is exactly idempotent (identical point count).

    Raises:
        ValueError: If ``voxel_size_mm`` is not strictly positive.
    """
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError(f"voxel_size_mm must be > 0, got {voxel_size_mm!r}")

    # Empty cloud -> empty cloud (Req 5.8 handled by standardize; here we keep
    # voxel_downsample total by returning an empty cloud for empty input).
    if pc.is_empty:
        return PointCloud()

    points = pc.points

    # Voxel index of every point with the grid origin fixed at (0, 0, 0). Using
    # int64 keeps large mm-scale coordinates exact. np.floor handles negative
    # coordinates correctly (rounds toward -inf).
    voxel_idx = np.floor(points / voxel_size_mm).astype(np.int64)

    # Deterministically group points by voxel. np.unique(axis=0) returns rows in
    # lexicographic order, giving a stable, reproducible output ordering. The
    # inverse maps each input point to its occupied-voxel slot.
    unique_vox, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_vox = unique_vox.shape[0]

    counts = np.bincount(inverse, minlength=n_vox).astype(np.float64)

    down_points = _mean_per_group(points, inverse, counts, n_vox)

    down_colors = None
    if pc.colors is not None:
        down_colors = _mean_per_group(pc.colors, inverse, counts, n_vox)

    down_normals = None
    if pc.normals is not None:
        averaged = _mean_per_group(pc.normals, inverse, counts, n_vox)
        norms = np.linalg.norm(averaged, axis=1, keepdims=True)
        # Guard against zero-length averages (opposing normals cancelling out):
        # fall back to a deterministic unit normal so every point stays a unit
        # vector.
        safe = norms[:, 0] > 0
        down_normals = np.empty_like(averaged)
        down_normals[safe] = averaged[safe] / norms[safe]
        down_normals[~safe] = np.array([0.0, 0.0, 1.0])

    return PointCloud(points=down_points, colors=down_colors, normals=down_normals)


def _mean_per_group(
    values: np.ndarray, inverse: np.ndarray, counts: np.ndarray, n_vox: int
) -> np.ndarray:
    """Return the per-group mean of ``(N, 3)`` ``values`` indexed by ``inverse``."""
    sums = np.zeros((n_vox, values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return sums / counts[:, None]


def standardize(
    pc: PointCloud, config: Config, fragment_id: str
) -> tuple[PointCloud, tuple[float, float, float], float]:
    """Standardize density and apply optional coordinate normalization.

    Pipeline of operations (design "Density Standardization and Coordinate
    Normalization" section):

    1. **Density standardization.** Voxel-downsample ``pc`` at
       ``config.voxel_size_mm`` (Req 5.1). Point count never increases (Req 5.2)
       and is exactly idempotent at a fixed voxel size (Req 5.3).
    2. **Coordinate normalization.** When ``config.centering_enabled`` subtract
       ``config.centering_offset`` (which may be zero or negative, Req 5.5);
       when ``config.scaling_enabled`` multiply by ``config.scale_factor``. With
       both enabled each output point equals ``(point - offset) * scale``.

    Millimeter units are preserved end to end -- no implicit unit conversion is
    performed (Req 5.4).

    The function is pure: it does not write anything. It returns the normalized
    cloud plus the offset and scale factor that were actually applied so the
    caller can record them in Metadata (Req 5.5). When centering/scaling are
    disabled the applied values are the identity ``(0.0, 0.0, 0.0)`` and
    ``1.0`` respectively. Use :func:`write_normalized` to persist the result to
    the ``normalized/`` subdirectory (Req 5.6, 5.9).

    Args:
        pc: Input point cloud (coordinates in millimeters).
        config: Pipeline configuration supplying the voxel size, centering and
            scaling toggles, centering offset, and scale factor.
        fragment_id: Fragment_ID used to name errors for this fragment.

    Returns:
        A 3-tuple ``(normalized_pc, applied_offset, applied_scale)`` where
        ``applied_offset`` is a ``(x, y, z)`` float tuple and ``applied_scale``
        is a float. An empty input yields an empty normalized cloud (Req 5.8).

    Raises:
        NormalizationError: If scaling is enabled and ``config.scale_factor`` is
            ``<= 0`` (or non-finite). The error names the Fragment_ID and the
            invalid scale factor, and no output is produced (Req 5.7).
    """
    # Validate the scale factor up front so an invalid configuration halts the
    # fragment before any work or output, naming the Fragment_ID and the
    # offending factor (Req 5.7).
    if config.scaling_enabled and not (
        np.isfinite(config.scale_factor) and config.scale_factor > 0
    ):
        raise NormalizationError(
            fragment_id,
            "scaling enabled but scale factor is not strictly positive",
            scale_factor=config.scale_factor,
        )

    applied_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    applied_scale: float = 1.0
    if config.centering_enabled:
        offset = config.centering_offset
        applied_offset = (float(offset[0]), float(offset[1]), float(offset[2]))
    if config.scaling_enabled:
        applied_scale = float(config.scale_factor)

    # Empty cloud -> empty normalized cloud (Req 5.8). The applied offset/factor
    # are still reported so metadata reflects the configured normalization.
    if pc.is_empty:
        return PointCloud(), applied_offset, applied_scale

    # 1. Density standardization via voxel downsampling (Req 5.1).
    downsampled = voxel_downsample(pc, config.voxel_size_mm)

    # 2. Coordinate normalization: (point - offset) * scale (Req 5.5). Units stay
    # in millimeters -- this is a pure affine transform, no conversion (Req 5.4).
    points = downsampled.points
    if config.centering_enabled:
        points = points - np.asarray(applied_offset, dtype=np.float64)
    if config.scaling_enabled:
        points = points * applied_scale

    # Centering is a translation and positive scaling is uniform, so neither
    # changes surface orientation: unit normals are preserved unchanged. Colors
    # are unaffected by geometry. Reuse the downsampled attributes as-is.
    normalized = PointCloud(
        points=points,
        colors=downsampled.colors,
        normals=downsampled.normals,
    )
    return normalized, applied_offset, applied_scale


def write_normalized(path: str, pc: PointCloud) -> None:
    """Write a normalized :class:`PointCloud` to the ``normalized/`` subdir.

    Thin wrapper over :func:`dataset_foundation.ply_io.write_point_cloud` so the
    pipeline persists the output of :func:`standardize`. An empty cloud is
    written as a valid empty binary little-endian PLY (Req 5.8).

    Args:
        path: Target filesystem path under the ``normalized/`` subdirectory.
        pc: The normalized point cloud to write.

    Raises:
        WriteError: If the file cannot be written, naming the target path
            (Req 5.9).
    """
    write_point_cloud(path, pc)
