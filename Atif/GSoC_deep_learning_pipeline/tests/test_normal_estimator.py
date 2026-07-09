"""Property-based and unit tests for :mod:`normal_estimator`.

Covers the normal estimation / orientation component:

* Property 11 - normals are exactly one unit vector per point (Req 4.1, 4.3,
  4.4): every non-empty cloud yields ``point_count`` normals, each of unit
  magnitude within 1.0 +/- 1e-3.
* Property 12 - neighboring normals are consistently oriented (Req 4.2): on a
  smooth (near-planar / gently curved) surface, every point's normal has a
  non-negative dot product with each of its nearest neighbors' normals.
* Property 13 - degenerate-neighborhood normals are deterministic (Req 4.5):
  a cloud whose neighborhoods are all empty (tiny search radius, far-apart
  points) collapses to the deterministic ``[0, 0, 1]`` fallback, is identical
  across repeated runs, and reports a stable degenerate count.
* Unit - empty-cloud normal estimation (Req 4.6, 4.7): an empty cloud yields an
  empty cloud with zero normals and a zero degenerate count.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial import cKDTree

from dataset_foundation.config_loader import NormalEstimationParams
from dataset_foundation.geometry import PointCloud
from dataset_foundation.normal_estimator import FALLBACK_NORMAL, estimate_normals


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _params(spacing: float, orientation_neighbors: int = 10) -> NormalEstimationParams:
    """Build params whose search radius comfortably exceeds the point spacing.

    A radius several times the grid spacing guarantees each interior point has a
    well-populated neighborhood, so PCA yields a robust tangent-plane normal.
    """
    search = min(20.0, max(5.0, spacing * 4.0))
    return NormalEstimationParams(
        search_radius_mm=search,
        max_neighbors=30,
        orientation_neighbors=orientation_neighbors,
    )


@st.composite
def _smooth_surface(draw, max_jitter_frac: float = 0.05, allow_sphere: bool = False):
    """Generate a well-sampled surface patch with mild jitter (N >= 16).

    Produces a plane, a gently curved paraboloid, or (optionally) a sphere in a
    realistic millimeter range. Returns ``(points, spacing)`` where ``spacing``
    is the nominal grid step used to size the search radius.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    kinds = ["plane", "paraboloid"]
    if allow_sphere:
        kinds = kinds + ["sphere"]
    kind = draw(st.sampled_from(kinds))

    side = draw(st.integers(min_value=4, max_value=7))  # 16..49 points
    spacing = draw(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    jitter = (
        draw(
            st.floats(
                min_value=0.0,
                max_value=max_jitter_frac,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        * spacing
    )

    if kind == "sphere":
        n = side * side
        radius = draw(
            st.floats(min_value=3.0, max_value=8.0, allow_nan=False, allow_infinity=False)
        )
        # Fibonacci sphere: evenly distributed points on a sphere surface.
        indices = np.arange(n, dtype=np.float64) + 0.5
        phi = np.arccos(1.0 - 2.0 * indices / n)
        theta = np.pi * (1.0 + 5.0**0.5) * indices
        pts = radius * np.stack(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ],
            axis=1,
        )
        # For a sphere, use its own nearest-neighbor spacing to size the radius.
        spacing = max(spacing, 3.5 * radius / np.sqrt(n))
    else:
        coords = (np.arange(side, dtype=np.float64) - (side - 1) / 2.0) * spacing
        gx, gy = np.meshgrid(coords, coords)
        gx = gx.ravel()
        gy = gy.ravel()
        if kind == "plane":
            gz = np.zeros_like(gx)
        else:  # paraboloid with small curvature -> stays smooth (C1 continuous)
            curvature = draw(
                st.floats(
                    min_value=0.0,
                    max_value=0.02,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            gz = curvature * (gx**2 + gy**2)
        pts = np.stack([gx, gy, gz], axis=1)

    pts = pts + rng.normal(scale=jitter, size=pts.shape)
    return np.ascontiguousarray(pts, dtype=np.float64), float(spacing)


# ---------------------------------------------------------------------------
# Property 11: Normals are exactly one unit vector per point (Req 4.1,4.3,4.4)
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 11: Normals are exactly one unit vector per point
@settings(max_examples=100)
@given(surface=_smooth_surface(max_jitter_frac=0.1, allow_sphere=True))
def test_property_11_one_unit_normal_per_point(surface):
    """Every point receives exactly one normal of unit magnitude.

    Validates: Requirements 4.1, 4.3, 4.4
    """
    points, spacing = surface
    n = points.shape[0]
    pc = PointCloud(points=points)

    result, _ = estimate_normals(pc, _params(spacing))

    # Exactly one normal per point (Req 4.4) and normals are present (Req 4.1).
    assert result.point_count == n
    assert result.has_normals
    assert result.normals.shape == (n, 3)

    # Unit magnitude within 1.0 +/- 1e-3 (Req 4.3).
    magnitudes = np.linalg.norm(result.normals, axis=1)
    assert np.all(np.abs(magnitudes - 1.0) <= 1e-3)


# ---------------------------------------------------------------------------
# Property 12: Neighboring normals are consistently oriented (Req 4.2)
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 12: Neighboring normals are consistently oriented
@settings(max_examples=100)
@given(surface=_smooth_surface(max_jitter_frac=0.03, allow_sphere=False))
def test_property_12_consistent_neighbor_orientation(surface):
    """Nearest-neighbor normal dot products are non-negative on a smooth surface.

    Validates: Requirements 4.2
    """
    points, spacing = surface
    n = points.shape[0]
    orientation_neighbors = 10
    pc = PointCloud(points=points)

    result, _ = estimate_normals(
        pc, _params(spacing, orientation_neighbors=orientation_neighbors)
    )
    normals = result.normals

    # Independent KD-tree over the output points; query k nearest neighbors.
    tree = cKDTree(result.points)
    query_k = min(orientation_neighbors + 1, n)
    _, neighbor_idx = tree.query(result.points, k=query_k)
    neighbor_idx = np.atleast_2d(neighbor_idx)
    if neighbor_idx.shape[1] > 1:
        neighbor_idx = neighbor_idx[:, 1:]  # drop self (first column)

    # dot(n_i, n_j) for every point i and each of its neighbors j.
    dots = np.einsum("ij,ikj->ik", normals, normals[neighbor_idx])
    assert np.all(dots >= -1e-6)


# ---------------------------------------------------------------------------
# Property 13: Degenerate-neighborhood normals are deterministic (Req 4.5)
# ---------------------------------------------------------------------------


@st.composite
def _degenerate_cloud(draw):
    """Generate a cloud whose neighborhoods are all empty.

    Points are placed far apart on a coarse grid and paired with a vanishingly
    small search radius, so every point's neighborhood is rank-deficient and
    must receive the deterministic fallback normal.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    n = draw(st.integers(min_value=3, max_value=12))
    spacing = draw(
        st.floats(min_value=500.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
    )
    # Points on a widely spaced line, jittered so no two coincide.
    base = np.arange(n, dtype=np.float64) * spacing
    points = np.stack(
        [base, rng.uniform(-1.0, 1.0, size=n), rng.uniform(-1.0, 1.0, size=n)],
        axis=1,
    )
    return np.ascontiguousarray(points, dtype=np.float64)


# Feature: dataset-foundation, Property 13: Degenerate-neighborhood normals are deterministic
@settings(max_examples=100)
@given(points=_degenerate_cloud())
def test_property_13_degenerate_normals_deterministic(points):
    """Empty-neighborhood normals collapse to a reproducible fallback.

    Validates: Requirements 4.5
    """
    n = points.shape[0]
    # Tiny search radius => no point has any neighbor within range.
    params = NormalEstimationParams(
        search_radius_mm=1e-9,
        max_neighbors=30,
        orientation_neighbors=10,
    )

    result_a, degenerate_a = estimate_normals(PointCloud(points=points), params)
    result_b, degenerate_b = estimate_normals(PointCloud(points=points), params)

    # Identical fallback across runs (Req 4.5).
    assert np.array_equal(result_a.normals, result_b.normals)
    assert degenerate_a == degenerate_b

    # These neighborhoods are all degenerate: count is recorded and rows are
    # the deterministic [0, 0, 1] fallback.
    assert degenerate_a > 0
    assert degenerate_a == n
    fallback = np.asarray(FALLBACK_NORMAL, dtype=np.float64)
    assert np.all(result_a.normals == fallback)


# ---------------------------------------------------------------------------
# Unit test 10.5: empty-cloud normal estimation (Req 4.6, 4.7)
# ---------------------------------------------------------------------------


def test_empty_cloud_yields_zero_normals():
    """An empty cloud produces zero points, zero normals, zero degenerate count.

    Validates: Requirements 4.6, 4.7
    """
    result, degenerate_count = estimate_normals(
        PointCloud(),
        NormalEstimationParams(
            search_radius_mm=5.0, max_neighbors=30, orientation_neighbors=10
        ),
    )

    assert result.point_count == 0
    assert degenerate_count == 0
    # No normals present on an empty cloud; the underlying array is empty.
    assert result.has_normals is False
    assert result.normals is None
