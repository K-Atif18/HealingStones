"""Normal estimation and consistent orientation.

Estimate a unit normal for every point of a :class:`PointCloud`, orient them
consistently, and fall back deterministically for degenerate neighborhoods.

Pipeline (design "Normal Estimation and Orientation" section):

1. **PCA tangent-plane normals.** For each non-empty cloud, normals are the
   eigenvector of the smallest eigenvalue of the local covariance, computed by
   Open3D's ``estimate_normals`` with a *hybrid* KD-tree search
   (``KDTreeSearchParamHybrid(radius=search_radius_mm, max_nn=max_neighbors)``)
   (Req 4.1).
2. **Consistent orientation.** Open3D's
   ``orient_normals_consistent_tangent_plane(k=orientation_neighbors)`` performs
   minimum-spanning-tree propagation so neighboring normals agree in direction.
   Because raw MST propagation can leave isolated sign disagreements, a
   deterministic post-pass then flips any normal whose summed dot product with
   its ``orientation_neighbors`` nearest neighbors is negative, iterating to a
   fixed point (bounded iterations) so neighbor dot products trend ``>= 0``
   (Req 4.2).
3. **Unit magnitude.** Every output normal is renormalized to unit length,
   satisfying magnitude within 1.0 +/- 0.001 (Req 4.3) with exactly one normal
   per point (Req 4.4).
4. **Degenerate neighborhoods.** When a point has insufficient (fewer than ~3
   non-collinear) neighbors the covariance is rank-deficient and the raw normal
   is zero / NaN. Such points are assigned a deterministic fallback unit normal
   ``[0, 0, 1]`` (reproducible across runs) and counted (Req 4.5).
5. **Empty clouds** produce zero points and zero normals (Req 4.6).

The public entry point keeps the returned :class:`PointCloud` pure (a new cloud
carrying normals); writing the normals PLY to the ``normals`` subdirectory is
the pipeline's responsibility (task 16.1, Req 4.7).

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from dataset_foundation.config_loader import NormalEstimationParams
from dataset_foundation.geometry import PointCloud

__all__ = ["estimate_normals", "FALLBACK_NORMAL"]

# Deterministic fallback assigned to degenerate (rank-deficient) neighborhoods.
FALLBACK_NORMAL: tuple[float, float, float] = (0.0, 0.0, 1.0)

# Tolerance below which a raw normal magnitude is treated as degenerate.
_DEGENERATE_MAGNITUDE = 1e-6

# Upper bound on deterministic flip-pass sweeps toward a fixed point.
_MAX_FLIP_ITERATIONS = 50


def estimate_normals(
    pc: PointCloud, params: NormalEstimationParams
) -> tuple[PointCloud, int]:
    """Estimate consistently oriented unit normals for every point.

    Args:
        pc: The input :class:`PointCloud`. Colors, when present, are carried
            through to the output unchanged.
        params: Neighborhood parameters (``search_radius_mm``, ``max_neighbors``,
            ``orientation_neighbors``) controlling estimation and orientation.

    Returns:
        A ``(PointCloud, int)`` tuple. The first element is a *new* point cloud
        with the same points (and colors) carrying one unit normal per point;
        the second is ``degenerate_count``, the number of points that received
        the deterministic ``[0, 0, 1]`` fallback normal because their
        neighborhood was rank-deficient.

    Empty-input behavior:
        An empty cloud yields an empty cloud whose ``normals`` is a canonical
        ``(0, 3)`` array and a ``degenerate_count`` of ``0`` (Req 4.6).
    """
    points = np.ascontiguousarray(pc.points, dtype=np.float64)
    n_points = points.shape[0]

    if n_points == 0:
        empty = PointCloud(
            points=np.empty((0, 3), dtype=np.float64),
            colors=pc.colors,
            normals=np.empty((0, 3), dtype=np.float64),
        )
        return empty, 0

    # --- 1. PCA tangent-plane normals via hybrid KD-tree search (Req 4.1) ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=float(params.search_radius_mm),
        max_nn=int(params.max_neighbors),
    )
    pcd.estimate_normals(search_param=search)

    raw_normals = np.asarray(pcd.normals, dtype=np.float64)
    if raw_normals.shape != (n_points, 3):
        # Defensive: guarantee one row per point before any further processing.
        raw_normals = np.zeros((n_points, 3), dtype=np.float64)

    # --- Identify degenerate neighborhoods (rank-deficient covariance) -------
    # Open3D's estimate_normals returns a *unit* default normal (e.g. [0, 0, 1])
    # for isolated / rank-deficient neighborhoods, so a raw-magnitude test never
    # fires for genuinely under-sampled points. Detect degeneracy primarily by
    # the actual neighbor count within the configured search radius: a stable
    # PCA tangent plane needs at least 3 neighbors (excluding the point itself).
    # The magnitude / non-finite test is retained as an additional signal so
    # both causes of degeneracy are counted (Req 4.5).
    neighbor_counts = _radius_neighbor_counts(
        points, float(params.search_radius_mm)
    )
    insufficient_neighbors = neighbor_counts < 3

    magnitudes = np.linalg.norm(raw_normals, axis=1)
    invalid_normal = ~np.isfinite(magnitudes) | (magnitudes < _DEGENERATE_MAGNITUDE)

    degenerate_mask = insufficient_neighbors | invalid_normal

    # Seed degenerate rows with the fallback so orientation runs on valid unit
    # vectors; degenerate rows are forced back to the fallback after the flip
    # pass so the result stays deterministic regardless of orientation choices.
    normals = raw_normals.copy()
    normals[degenerate_mask] = FALLBACK_NORMAL
    pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals))

    # --- 2. Consistent orientation: MST propagation (Req 4.2) ----------------
    k = max(1, int(params.orientation_neighbors))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=k)
        normals = np.asarray(pcd.normals, dtype=np.float64).copy()
    except RuntimeError:
        # Tangent-plane propagation can fail on tiny/degenerate clouds; fall
        # back to the seeded normals and let the deterministic flip pass run.
        normals = np.asarray(pcd.normals, dtype=np.float64).copy()

    # --- 2b. Deterministic neighbor-dot-product flip pass to a fixed point ---
    normals = _consistent_flip_pass(points, normals, k)

    # --- 3. Renormalize every normal to unit length (Req 4.3) ----------------
    normals = _renormalize(normals)

    # --- 4. Deterministic fallback for degenerate neighborhoods (Req 4.5) ----
    normals[degenerate_mask] = FALLBACK_NORMAL
    degenerate_count = int(np.count_nonzero(degenerate_mask))

    result = PointCloud(
        points=points,
        colors=pc.colors,
        normals=np.ascontiguousarray(normals, dtype=np.float64),
    )
    return result, degenerate_count


def _radius_neighbor_counts(points: np.ndarray, radius: float) -> np.ndarray:
    """Count neighbors within ``radius`` of each point, excluding the point.

    Builds a single KD-tree and issues one radius query. Returns an integer
    array of per-point neighbor counts (self excluded). A non-positive radius
    yields all-zero counts, so every point is flagged degenerate.
    """
    n_points = points.shape[0]
    if n_points == 0:
        return np.zeros((0,), dtype=np.int64)
    if not np.isfinite(radius) or radius <= 0.0:
        return np.zeros(n_points, dtype=np.int64)

    tree = cKDTree(points)
    # query_ball_point returns, for each point, the indices within the radius
    # (including the point itself); subtract one to exclude self.
    neighbor_lists = tree.query_ball_point(points, r=radius)
    counts = np.fromiter(
        (len(lst) - 1 for lst in neighbor_lists), dtype=np.int64, count=n_points
    )
    return counts


def _consistent_flip_pass(
    points: np.ndarray, normals: np.ndarray, k: int
) -> np.ndarray:
    """Flip normals to drive neighbor dot products non-negative (Req 4.2).

    Builds the ``k``-nearest-neighbor graph once and repeatedly flips, in a
    single synchronous sweep, any normal whose summed dot product with its
    neighbors is negative. Sweeps iterate to a fixed point or until a bounded
    iteration cap, keeping the pass deterministic and terminating.
    """
    n_points = points.shape[0]
    if n_points < 2:
        return normals

    tree = cKDTree(points)
    # Query k neighbors plus self; drop the self column deterministically.
    query_k = min(k + 1, n_points)
    _, neighbor_idx = tree.query(points, k=query_k)
    neighbor_idx = np.atleast_2d(neighbor_idx)
    if neighbor_idx.shape[1] > 1:
        neighbor_idx = neighbor_idx[:, 1:]  # exclude self (first column)

    current = normals.copy()
    for _ in range(_MAX_FLIP_ITERATIONS):
        # Sum of neighbor normals for each point, then dot with own normal.
        neighbor_sum = current[neighbor_idx].sum(axis=1)
        alignment = np.einsum("ij,ij->i", current, neighbor_sum)
        flip_mask = alignment < 0.0
        if not np.any(flip_mask):
            break
        current[flip_mask] = -current[flip_mask]
    return current


def _renormalize(normals: np.ndarray) -> np.ndarray:
    """Scale each normal to unit length; degenerate rows are handled by caller.

    Rows whose magnitude is effectively zero are left as the fallback normal so
    the invariant "one unit normal per point" (Req 4.3, 4.4) always holds.
    """
    magnitudes = np.linalg.norm(normals, axis=1)
    safe = magnitudes > _DEGENERATE_MAGNITUDE
    out = np.array(normals, dtype=np.float64, copy=True)
    out[safe] = out[safe] / magnitudes[safe][:, None]
    out[~safe] = FALLBACK_NORMAL
    return out
