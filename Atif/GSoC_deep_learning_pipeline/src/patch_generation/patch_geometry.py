"""Shared KD-tree neighborhood helpers.

Thin wrappers over :class:`scipy.spatial.cKDTree` shared by
``patch_extractor``, mirroring the Phase 1 ``geometry_metrics`` KD-tree
pattern. All geometry is expressed in millimeters in the Full_Model
coordinate frame.

Three helpers are provided:

* :func:`build_kdtree` builds a KD-tree over an ``(N, 3)`` point set.
* :func:`radius_neighbors` returns the source indices whose Euclidean
  distance to a center is ``<= radius`` (inclusive of the boundary, via
  ``query_ball_point``), sorted ascending for determinism (Req 3.1).
* :func:`nearest_k_within_radius` returns the ``min(k, len)`` in-radius
  indices nearest to a center, ordered by ``(distance asc, index asc)`` so
  ties break by lowest index deterministically (Req 3.2, 3.5).

Requirements: 3.1, 3.2, 3.5.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "build_kdtree",
    "radius_neighbors",
    "nearest_k_within_radius",
]


def build_kdtree(points: np.ndarray) -> cKDTree:
    """Build a KD-tree over an ``(N, 3)`` point set.

    Args:
        points: ``(N, 3)`` array-like of point coordinates (mm, Full_Model
            frame). Coerced to a contiguous float64 array.

    Returns:
        A :class:`scipy.spatial.cKDTree` built on the points.

    Raises:
        ValueError: if ``points`` does not have shape ``(N, 3)``.
    """
    arr = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    return cKDTree(arr)


def radius_neighbors(
    tree: cKDTree, center: np.ndarray, radius: float
) -> np.ndarray:
    """Source indices within ``radius`` (inclusive) of ``center``.

    Uses ``query_ball_point``, which is inclusive of the boundary: a point at
    Euclidean distance exactly equal to ``radius`` is included (Req 3.1).

    Args:
        tree: KD-tree built by :func:`build_kdtree`.
        center: ``(3,)`` array-like center coordinate.
        radius: Non-negative neighborhood radius (mm).

    Returns:
        An int64 array of the in-radius source indices, sorted ascending for
        deterministic ordering.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    idx = tree.query_ball_point(c, r=float(radius))
    result = np.asarray(idx, dtype=np.int64).reshape(-1)
    result.sort()
    return result


def nearest_k_within_radius(
    tree: cKDTree, center: np.ndarray, radius: float, k: int
) -> np.ndarray:
    """The ``min(k, len)`` in-radius indices nearest to ``center``.

    Gathers the inclusive in-radius indices (:func:`radius_neighbors`), orders
    them by ``(distance asc, index asc)`` so ties break by lowest index, and
    returns the first ``k`` (Req 3.2, 3.5).

    Args:
        tree: KD-tree built by :func:`build_kdtree`.
        center: ``(3,)`` array-like center coordinate.
        radius: Non-negative neighborhood radius (mm).
        k: Maximum number of indices to return; the result has
            ``min(k, number_of_in_radius_points)`` entries.

    Returns:
        An int64 array of the selected source indices, ordered nearest-first
        with lowest-index tie-breaking.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    in_radius = radius_neighbors(tree, c, radius)
    if in_radius.size == 0 or k <= 0:
        return np.empty((0,), dtype=np.int64)

    data = np.asarray(tree.data, dtype=np.float64)
    distances = np.linalg.norm(data[in_radius] - c, axis=1)
    # Sort by (distance asc, index asc). in_radius is already index-ascending,
    # so a stable sort on distance alone preserves lowest-index tie-breaking.
    order = np.argsort(distances, kind="stable")
    ordered = in_radius[order]
    take = min(int(k), ordered.size)
    return np.ascontiguousarray(ordered[:take], dtype=np.int64)
