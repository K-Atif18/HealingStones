"""Tests for ``patch_generation.fps_sampler``.

Covers two spec tasks that share this file:

* Task 6.2 -- Property 4: FPS is deterministic under a fixed seed
  (Requirements 2.1, 2.3).
* Task 6.3 -- Property 5: FPS centers are a distinct subset, correctly counted
  and capped (Requirements 2.2, 2.4, 2.5, 2.6).

``select_centers(points, count, seed)`` returns a ``(K,)`` int64 array of
ordered source indices into ``points`` with ``K = min(count, N)``. Both
properties drive it with a shared point-cloud strategy that yields finite,
mm-scale ``(N, 3)`` float64 clouds. Duplicate points are allowed: FPS still
returns distinct *indices*, which is what these properties assert.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from patch_generation.fps_sampler import select_centers


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Finite mm-scale float coordinates (Full_Model frame). Bounded so distances
# stay well within float64 range and Hypothesis shrinks toward simple clouds.
_finite_coords = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@st.composite
def _point_clouds(draw, min_points: int = 1, max_points: int = 200):
    """Draw an ``(N, 3)`` float64 point cloud with ``N`` in ``[min, max]``.

    All-identical clouds are permitted; FPS is expected to still return
    pairwise-distinct source indices, so no degenerate case needs filtering.
    """
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    rows = draw(
        st.lists(
            st.tuples(_finite_coords, _finite_coords, _finite_coords),
            min_size=n,
            max_size=n,
        )
    )
    return np.asarray(rows, dtype=np.float64).reshape((n, 3))


@st.composite
def _cloud_count_seed(draw, allow_over_n: bool = True):
    """Draw ``(points, count, seed)`` for driving ``select_centers``.

    ``count`` ranges over ``1..N`` and, when ``allow_over_n`` is set, may also
    slightly exceed ``N`` to exercise the ``count >= N`` capping branch.
    """
    points = draw(_point_clouds())
    n = points.shape[0]
    upper = n + 5 if allow_over_n else n
    count = draw(st.integers(min_value=1, max_value=upper))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return points, count, seed


# ---------------------------------------------------------------------------
# Task 6.2 -- Property 4: FPS is deterministic under a fixed seed
# ---------------------------------------------------------------------------


# Feature: patch-generation, Property 4: FPS is deterministic under a fixed seed
# Validates: Requirements 2.1, 2.3
@settings(max_examples=100)
@given(data=_cloud_count_seed())
def test_fps_is_deterministic_under_fixed_seed(data):
    """Two calls with the same (points, count, seed) return identical arrays."""
    points, count, seed = data

    first = select_centers(points, count, seed)
    second = select_centers(points, count, seed)

    assert first.dtype == np.int64
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Task 6.3 -- Property 5: FPS centers are a distinct subset, correctly counted
# and capped
# ---------------------------------------------------------------------------


# Feature: patch-generation, Property 5: FPS centers are a distinct subset, correctly counted and capped
# Validates: Requirements 2.2, 2.4, 2.5, 2.6
@settings(max_examples=100)
@given(data=_cloud_count_seed())
def test_fps_centers_are_distinct_subset_counted_and_capped(data):
    """Returned indices are a distinct, in-range, correctly-sized subset."""
    points, count, seed = data
    n = points.shape[0]

    centers = select_centers(points, count, seed)

    # K = min(count, N) (Req 2.4).
    expected_k = min(count, n)
    assert len(centers) == expected_k

    # Every index is a valid point index in [0, N) (Req 2.2).
    assert np.all(centers >= 0)
    assert np.all(centers < n)

    # Pairwise distinct (Req 2.6).
    assert len(set(centers.tolist())) == len(centers)

    # Each returned index selects an actual point of the Fragment (Req 2.2).
    selected_points = points[centers]
    assert selected_points.shape == (expected_k, 3)

    # When count >= N, every point is selected exactly once (Req 2.5).
    if count >= n:
        assert set(centers.tolist()) == set(range(n))
