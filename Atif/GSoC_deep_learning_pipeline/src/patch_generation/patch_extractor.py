"""Radius-based patch extraction and the Patch data model.

Defines the :class:`Patch` dataclass and :func:`extract_patch` /
:func:`extract_patches`: radius neighborhood extraction with Max_Patch_Points
nearest-point capping, local/global coordinates, per-point normals, and source
indices. Built on the shared KD-tree helpers in :mod:`patch_geometry`.

All geometry is expressed in millimeters in the Full_Model coordinate frame.

For a center at ``center_index`` with coordinates ``c = points[center_index]``:

1. **Radius membership.** All in-radius indices are queried inclusively via
   :func:`patch_geometry.radius_neighbors`; the center is always a member at
   distance 0 (Req 3.1, 3.4).
2. **Capping.** When the in-radius count exceeds ``max_patch_points``, keep the
   ``max_patch_points`` nearest to ``c`` (ties by lowest index) and drop the
   rest (Req 3.2); otherwise keep exactly the in-radius set (Req 3.5).
3. **Bounds.** The resulting size is in ``[1, max_patch_points]`` (Req 3.3).
4. **Record fields.** ``source_indices`` records each point's index in the
   Fragment cloud (Req 3.6); ``global_coords = points[source_indices]``,
   ``local_coords = global_coords - c`` (Req 4.3), and
   ``normals = fragment_normals[source_indices]`` (one per point, Req 4.4,
   4.5). Patch_IDs are assigned ``0..M-1`` per Fragment (Req 4.1, 4.2).

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from patch_generation.patch_geometry import (
    build_kdtree,
    nearest_k_within_radius,
    radius_neighbors,
)

__all__ = [
    "Patch",
    "extract_patch",
    "extract_patches",
]


@dataclass
class Patch:
    """A local geometric neighborhood around a Patch_Center.

    Attributes:
        patch_id: Identifier unique within the owning Fragment (Req 4.2).
        fragment_id: Owning Fragment_ID (Req 4.1).
        center: ``(3,)`` float64 Patch_Center coordinates (Full_Model frame).
        center_index: Source index of the center in the Fragment cloud.
        source_indices: ``(P,)`` int64 indices into the Fragment cloud
            (Req 3.6).
        global_coords: ``(P, 3)`` float64 coordinates in the Full_Model frame.
        local_coords: ``(P, 3)`` float64 coordinates relative to the center,
            i.e. ``global_coords - center`` (Req 4.3).
        normals: ``(P, 3)`` float64 per-point unit normals, one per point
            (Req 4.4, 4.5).
    """

    patch_id: int
    fragment_id: str
    center: np.ndarray
    center_index: int
    source_indices: np.ndarray
    global_coords: np.ndarray
    local_coords: np.ndarray
    normals: np.ndarray


def extract_patch(
    points: np.ndarray,
    normals: np.ndarray,
    center_index: int,
    patch_radius_mm: float,
    max_patch_points: int,
    patch_id: int,
    fragment_id: str,
    kdtree: cKDTree | None = None,
) -> Patch:
    """Extract a single radius-based Patch around ``center_index``.

    Args:
        points: ``(N, 3)`` array of Fragment point coordinates (mm).
        normals: ``(N, 3)`` array of per-point unit normals, one per point.
        center_index: Source index of the Patch_Center in ``points``.
        patch_radius_mm: Neighborhood radius (mm); membership is inclusive of
            the boundary (Req 3.1).
        max_patch_points: Maximum number of points a Patch may contain
            (``>= 1``); larger neighborhoods are capped to the nearest points
            (Req 3.2).
        patch_id: Identifier assigned to the Patch (unique within Fragment).
        fragment_id: Owning Fragment_ID.
        kdtree: Optional prebuilt KD-tree over ``points``; built on demand when
            not supplied.

    Returns:
        The extracted :class:`Patch`.

    Raises:
        ValueError: if ``points``/``normals`` are not ``(N, 3)`` with matching
            counts, if ``center_index`` is out of range, or if
            ``max_patch_points < 1``.
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    nrm = np.ascontiguousarray(np.asarray(normals, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if nrm.ndim != 2 or nrm.shape[1] != 3:
        raise ValueError(f"normals must have shape (N, 3), got {nrm.shape}")
    if nrm.shape[0] != pts.shape[0]:
        raise ValueError(
            "normals count must match points count: "
            f"{nrm.shape[0]} != {pts.shape[0]}"
        )
    n = pts.shape[0]
    if not 0 <= int(center_index) < n:
        raise ValueError(
            f"center_index {center_index} out of range for {n} points"
        )
    if int(max_patch_points) < 1:
        raise ValueError(
            f"max_patch_points must be >= 1, got {max_patch_points}"
        )

    tree = kdtree if kdtree is not None else build_kdtree(pts)
    center = np.ascontiguousarray(pts[int(center_index)], dtype=np.float64)

    in_radius = radius_neighbors(tree, center, patch_radius_mm)
    if in_radius.size > int(max_patch_points):
        # Keep the nearest max_patch_points (ties by lowest index, Req 3.2).
        kept = nearest_k_within_radius(
            tree, center, patch_radius_mm, int(max_patch_points)
        )
    else:
        # Keep exactly the in-radius set and nothing else (Req 3.5).
        kept = in_radius

    kept = np.ascontiguousarray(kept, dtype=np.int64)
    global_coords = np.ascontiguousarray(pts[kept], dtype=np.float64)
    local_coords = np.ascontiguousarray(global_coords - center, dtype=np.float64)
    patch_normals = np.ascontiguousarray(nrm[kept], dtype=np.float64)

    return Patch(
        patch_id=int(patch_id),
        fragment_id=str(fragment_id),
        center=center,
        center_index=int(center_index),
        source_indices=kept,
        global_coords=global_coords,
        local_coords=local_coords,
        normals=patch_normals,
    )


def extract_patches(
    points: np.ndarray,
    normals: np.ndarray,
    center_indices: np.ndarray,
    patch_radius_mm: float,
    max_patch_points: int,
    fragment_id: str,
) -> list[Patch]:
    """Extract patches for an ordered sequence of Patch_Centers.

    Builds the Fragment KD-tree once and reuses it across all centers, then
    assigns Patch_IDs ``0..M-1`` in center order (unique within the Fragment,
    Req 4.2).

    Args:
        points: ``(N, 3)`` array of Fragment point coordinates (mm).
        normals: ``(N, 3)`` array of per-point unit normals, one per point.
        center_indices: Ordered source indices of the Patch_Centers.
        patch_radius_mm: Neighborhood radius (mm).
        max_patch_points: Maximum number of points a Patch may contain.
        fragment_id: Owning Fragment_ID.

    Returns:
        A list of :class:`Patch` objects, one per center, in center order.
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    nrm = np.ascontiguousarray(np.asarray(normals, dtype=np.float64))
    tree = build_kdtree(pts)
    centers = np.asarray(center_indices, dtype=np.int64).reshape(-1)

    patches: list[Patch] = []
    for patch_id, center_index in enumerate(centers):
        patches.append(
            extract_patch(
                pts,
                nrm,
                int(center_index),
                patch_radius_mm,
                max_patch_points,
                patch_id,
                fragment_id,
                kdtree=tree,
            )
        )
    return patches
