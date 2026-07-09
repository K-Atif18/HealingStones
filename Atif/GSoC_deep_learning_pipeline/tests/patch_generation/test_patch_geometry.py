"""Unit and model-based tests for the shared KD-tree geometry helpers.

Cross-checks :mod:`patch_generation.patch_geometry` against brute-force
reference computations on small point clouds and pins down the boundary and
tie-breaking behavior the extractor depends on:

* :func:`radius_neighbors` returns exactly ``{i : ||p_i - c|| <= radius}``,
  sorted ascending, matching a direct pairwise computation (Req 3.1).
* :func:`nearest_k_within_radius` returns the ``min(k, len)`` in-radius
  indices nearest the center, ordered by ``(distance, index)`` so ties break
  by lowest index (Req 3.2, 3.5).
* A point at Euclidean distance exactly equal to ``radius`` is included
  (inclusive boundary, Req 3.1).
* ``k`` larger than the available in-radius count returns all in-radius
  indices; an empty neighborhood returns an empty int64 array.

Validates: Requirements 3.1, 3.2, 3.5
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from patch_generation.patch_geometry import (
    build_kdtree,
    nearest_k_within_radius,
    radius_neighbors,
)


# ---------------------------------------------------------------------------
# Brute-force reference implementations.
# ---------------------------------------------------------------------------


def _bf_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Indices within (inclusive) ``radius`` of ``center``, sorted ascending."""
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.nonzero(dists <= radius)[0]
    return np.sort(idx).astype(np.int64)


def _bf_nearest_k(
    points: np.ndarray, center: np.ndarray, radius: float, k: int
) -> np.ndarray:
    """The ``min(k, len)`` nearest in-radius indices by ``(distance, index)``."""
    in_radius = _bf_radius(points, center, radius)
    if in_radius.size == 0 or k <= 0:
        return np.empty((0,), dtype=np.int64)
    dists = np.linalg.norm(points[in_radius] - center, axis=1)
    # in_radius is index-ascending; a stable sort on distance breaks ties by
    # lowest index, matching the helper's contract.
    order = np.argsort(dists, kind="stable")
    ordered = in_radius[order]
    return ordered[: min(int(k), ordered.size)].astype(np.int64)


# ---------------------------------------------------------------------------
# Brute-force cross-check on fixed-seed random clouds.
# ---------------------------------------------------------------------------


def test_radius_neighbors_matches_brute_force():
    """radius_neighbors equals the direct pairwise membership set (Req 3.1)."""
    rng = np.random.default_rng(20240607)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        points = rng.uniform(-50.0, 50.0, size=(n, 3))
        center = points[int(rng.integers(n))]  # center on a real point
        radius = float(rng.uniform(1.0, 60.0))

        tree = build_kdtree(points)
        result = radius_neighbors(tree, center, radius)
        expected = _bf_radius(points, center, radius)

        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, expected)
        # Sorted ascending.
        assert np.all(np.diff(result) > 0) or result.size <= 1


def test_nearest_k_within_radius_matches_brute_force():
    """nearest_k results match the brute-force (distance, index) order (Req 3.2, 3.5)."""
    rng = np.random.default_rng(918273)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        points = rng.uniform(-50.0, 50.0, size=(n, 3))
        center = points[int(rng.integers(n))]
        radius = float(rng.uniform(1.0, 60.0))
        k = int(rng.integers(1, n + 3))  # sometimes larger than available

        tree = build_kdtree(points)
        result = nearest_k_within_radius(tree, center, radius, k)
        expected = _bf_nearest_k(points, center, radius, k)

        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, expected)
        # Never returns more than the in-radius count nor more than k.
        in_radius = _bf_radius(points, center, radius)
        assert result.size == min(k, in_radius.size)
        # Distances are non-decreasing (nearest-first ordering).
        dists = np.linalg.norm(points[result] - center, axis=1)
        assert np.all(np.diff(dists) >= -1e-9)


# ---------------------------------------------------------------------------
# Boundary inclusivity: a point at exactly distance == radius is included.
# ---------------------------------------------------------------------------


def test_radius_boundary_is_inclusive():
    """A point at Euclidean distance exactly == radius is a member (Req 3.1)."""
    radius = 5.0
    points = np.array(
        [
            [0.0, 0.0, 0.0],   # center, distance 0
            [radius, 0.0, 0.0],  # exactly on the boundary
            [0.0, radius, 0.0],  # exactly on the boundary
            [radius + 1e-6, 0.0, 0.0],  # just outside
        ],
        dtype=np.float64,
    )
    tree = build_kdtree(points)
    result = radius_neighbors(tree, points[0], radius)

    np.testing.assert_array_equal(result, np.array([0, 1, 2], dtype=np.int64))
    assert 3 not in result.tolist()


def test_nearest_k_includes_boundary_point():
    """nearest_k_within_radius includes a point sitting exactly on the radius."""
    radius = 3.0
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [radius, 0.0, 0.0],  # exactly on the boundary
        ],
        dtype=np.float64,
    )
    tree = build_kdtree(points)
    result = nearest_k_within_radius(tree, points[0], radius, k=3)
    np.testing.assert_array_equal(result, np.array([0, 1, 2], dtype=np.int64))


# ---------------------------------------------------------------------------
# Deterministic lowest-index tie-breaking on equidistant points.
# ---------------------------------------------------------------------------


def test_nearest_k_ties_break_by_lowest_index():
    """Equidistant points are returned lowest-index first (Req 3.2, 3.5)."""
    # Six points all at distance 10 from the origin center, plus the center.
    d = 10.0
    points = np.array(
        [
            [0.0, 0.0, 0.0],   # index 0: center, distance 0
            [d, 0.0, 0.0],     # index 1: distance d
            [-d, 0.0, 0.0],    # index 2: distance d
            [0.0, d, 0.0],     # index 3: distance d
            [0.0, -d, 0.0],    # index 4: distance d
            [0.0, 0.0, d],     # index 5: distance d
            [0.0, 0.0, -d],    # index 6: distance d
        ],
        dtype=np.float64,
    )
    tree = build_kdtree(points)

    # Ask for 4: center (nearest) then the three lowest-index equidistant ones.
    result = nearest_k_within_radius(tree, points[0], radius=d, k=4)
    np.testing.assert_array_equal(result, np.array([0, 1, 2, 3], dtype=np.int64))

    # Determinism: repeated calls give the identical ordering.
    again = nearest_k_within_radius(tree, points[0], radius=d, k=4)
    np.testing.assert_array_equal(result, again)


def test_nearest_k_full_tie_returns_ascending_indices():
    """When every candidate is equidistant, results are ascending indices."""
    d = 7.5
    # Six axis-aligned points, all at EXACTLY distance d from the origin
    # (exact in float64, unlike a trig-based ring), so the only tie-break is
    # the lowest-index rule.
    points = np.array(
        [
            [d, 0.0, 0.0],
            [-d, 0.0, 0.0],
            [0.0, d, 0.0],
            [0.0, -d, 0.0],
            [0.0, 0.0, d],
            [0.0, 0.0, -d],
        ],
        dtype=np.float64,
    )
    tree = build_kdtree(points)

    result = nearest_k_within_radius(tree, np.zeros(3), radius=d, k=5)
    np.testing.assert_array_equal(result, np.array([0, 1, 2, 3, 4], dtype=np.int64))


# ---------------------------------------------------------------------------
# k larger than available, empty neighborhood, dtypes.
# ---------------------------------------------------------------------------


def test_k_larger_than_available_returns_all_in_radius():
    """k exceeding the in-radius count returns every in-radius index (Req 3.2)."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],  # far outside the radius
        ],
        dtype=np.float64,
    )
    tree = build_kdtree(points)
    result = nearest_k_within_radius(tree, points[0], radius=5.0, k=100)

    np.testing.assert_array_equal(result, np.array([0, 1, 2], dtype=np.int64))
    assert result.dtype == np.int64


def test_empty_neighborhood_returns_empty_int64():
    """A center with no in-radius points yields an empty int64 array."""
    points = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]], dtype=np.float64)
    tree = build_kdtree(points)
    far = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)

    neighbors = radius_neighbors(tree, far, radius=1.0)
    assert neighbors.dtype == np.int64
    assert neighbors.size == 0

    nearest = nearest_k_within_radius(tree, far, radius=1.0, k=5)
    assert nearest.dtype == np.int64
    assert nearest.size == 0


def test_radius_and_nearest_dtypes_are_int64_on_nonempty():
    """Both helpers return int64 arrays for non-empty results (dtype contract)."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    tree = build_kdtree(points)
    assert radius_neighbors(tree, points[0], radius=5.0).dtype == np.int64
    assert nearest_k_within_radius(tree, points[0], radius=5.0, k=2).dtype == np.int64


# ---------------------------------------------------------------------------
# Property-based cross-check: helpers agree with brute force across inputs.
# ---------------------------------------------------------------------------


_coord = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def _cloud_center_radius(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    pts = draw(
        st.lists(
            st.tuples(_coord, _coord, _coord), min_size=n, max_size=n
        ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))
    )
    center_idx = draw(st.integers(min_value=0, max_value=n - 1))
    radius = draw(
        st.floats(min_value=0.5, max_value=250.0, allow_nan=False, allow_infinity=False)
    )
    k = draw(st.integers(min_value=1, max_value=n + 2))
    return pts, center_idx, radius, k


@given(_cloud_center_radius())
def test_property_helpers_match_brute_force(data):
    """radius/nearest helpers match brute force over many inputs (Req 3.1, 3.2, 3.5)."""
    points, center_idx, radius, k = data
    center = points[center_idx]
    tree = build_kdtree(points)

    np.testing.assert_array_equal(
        radius_neighbors(tree, center, radius), _bf_radius(points, center, radius)
    )
    np.testing.assert_array_equal(
        nearest_k_within_radius(tree, center, radius, k),
        _bf_nearest_k(points, center, radius, k),
    )
