"""Seeded, deterministic Farthest Point Sampling of Patch_Centers.

Provides ``select_centers`` returning ordered source indices into the Fragment
point cloud, seeded for determinism with lowest-index tie-breaking.

See the design's "FPS Algorithm (Detail)" section. All randomness is confined
to the seeded first pick; every subsequent step is a deterministic function of
the point coordinates and the lowest-index tie-break rule, so repeated runs on
the same ``(points, count, seed)`` produce the identical ordered sequence.
"""

from __future__ import annotations

import numpy as np


def select_centers(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Select Patch_Center source indices via seeded Farthest Point Sampling.

    Args:
        points: ``(N, 3)`` float64 Fragment coordinates (Full_Model frame, mm).
        count: Requested number of centers (pre-cap).
        seed: Seed for the deterministic first-center pick.

    Returns:
        ``(K,)`` int64 array of ordered source indices into ``points`` with
        ``K = min(count, N)``. For ``K > 1`` the indices are pairwise distinct;
        when ``count >= N`` every point is selected exactly once. Returns an
        empty ``(0,)`` int64 array when ``N == 0``.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    if n == 0:
        return np.empty((0,), dtype=np.int64)

    k = min(int(count), n)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)

    # Seed the first center deterministically from the PRNG.
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))

    selected = np.empty((k,), dtype=np.int64)
    selected[0] = first

    # min_dist[i] = distance from point i to the nearest already-selected center.
    # Initialize to the distance from the first center.
    min_dist = np.linalg.norm(pts - pts[first], axis=1)
    # Selected points can never be reselected: force their min_dist to -inf so
    # argmax always prefers an unselected point (until all are selected).
    min_dist[first] = -np.inf

    for i in range(1, k):
        # np.argmax returns the first occurrence of the maximum -> lowest-index
        # tie-break for determinism.
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        dist_to_next = np.linalg.norm(pts - pts[nxt], axis=1)
        min_dist = np.minimum(min_dist, dist_to_next)
        min_dist[nxt] = -np.inf

    return selected
