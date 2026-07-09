"""Property-based tests for ``patch_generation.patch_extractor``.

Covers Properties 6-9 of the Patch Generation design:

* Property 6 - Neighborhood exactness, center membership, source-index fidelity
  (Req 3.1, 3.4, 3.5, 3.6).
* Property 7 - Radius capping keeps the nearest points within size bounds
  (Req 3.2, 3.3).
* Property 8 - Local/global coordinate round-trip and one normal per point
  (Req 4.3, 4.4, 4.5).
* Property 9 - Patch_IDs are unique within a Fragment (Req 4.2).

All strategies yield finite mm-scale coordinates in the Full_Model frame and
build patches exclusively through ``extract_patch`` / ``extract_patches``.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from patch_generation.patch_extractor import extract_patch, extract_patches

FRAGMENT_ID = "fragment_test"

# Finite float coordinates at a realistic mm scale for the Caesar model.
_finite = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# Tightly packed coordinates (small mm-scale box) used to force radius capping.
_dense = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _rows(n: int, elem):
    """Strategy for an ``(n, 3)`` float64 array of coordinates from ``elem``."""

    return st.lists(
        st.tuples(elem, elem, elem),
        min_size=n,
        max_size=n,
    ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))


@st.composite
def _cloud_center_radius(draw, min_n: int = 1, max_n: int = 40):
    """A cloud with matched normals, a center index, and a positive radius."""

    n = draw(st.integers(min_value=min_n, max_value=max_n))
    points = draw(_rows(n, _finite))
    normals = draw(_rows(n, _finite))
    center_index = draw(st.integers(min_value=0, max_value=n - 1))
    radius = draw(
        st.floats(
            min_value=0.1,
            max_value=5.0e3,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return points, normals, int(center_index), float(radius)


@st.composite
def _dense_cloud_cap(draw, min_n: int = 4, max_n: int = 64):
    """A dense cloud, generous radius, a center, and a cap ``< n``.

    The radius (>= 4.0) exceeds the diameter of the ``[-1, 1]^3`` box (~3.46),
    so every point lies within the radius of any center; a cap strictly below
    ``n`` therefore always triggers the nearest-point capping path.
    """

    n = draw(st.integers(min_value=min_n, max_value=max_n))
    points = draw(_rows(n, _dense))
    normals = draw(_rows(n, _finite))
    center_index = draw(st.integers(min_value=0, max_value=n - 1))
    radius = draw(
        st.floats(
            min_value=4.0,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    cap = draw(st.integers(min_value=1, max_value=n - 1))
    return points, normals, int(center_index), float(radius), int(cap)


# Feature: patch-generation, Property 6: Neighborhood exactness, center
# membership, and source-index fidelity.
# Validates: Requirements 3.1, 3.4, 3.5, 3.6
@settings(max_examples=100, deadline=None)
@given(_cloud_center_radius())
def test_property_6_neighborhood_exactness_and_source_index_fidelity(data):
    points, normals, center_index, radius = data
    n = points.shape[0]

    center = points[center_index]
    dists = np.linalg.norm(points - center, axis=1)
    # Avoid ambiguity at the inclusive radius boundary so the brute-force set
    # and the KD-tree membership agree exactly.
    assume(np.all(np.abs(dists - radius) > 1e-6))

    # A cap of N guarantees the uncapped path: the in-radius count is <= N.
    patch = extract_patch(
        points,
        normals,
        center_index,
        radius,
        max_patch_points=n,
        patch_id=0,
        fragment_id=FRAGMENT_ID,
    )

    expected = set(np.nonzero(dists <= radius)[0].tolist())
    got = set(int(i) for i in patch.source_indices.tolist())

    # Neighborhood exactness: exactly the in-radius set, nothing else (3.1, 3.5).
    assert got == expected
    # Center membership: the center is always a member (3.4).
    assert center_index in got
    # Source-index fidelity: global coords echo the source points (3.6).
    for row, src in enumerate(patch.source_indices.tolist()):
        assert np.array_equal(patch.global_coords[row], points[src])


# Feature: patch-generation, Property 7: Radius capping keeps the nearest points
# within size bounds.
# Validates: Requirements 3.2, 3.3
@settings(max_examples=100, deadline=None)
@given(_dense_cloud_cap())
def test_property_7_radius_capping_keeps_nearest_within_bounds(data):
    points, normals, center_index, radius, cap = data
    n = points.shape[0]

    patch = extract_patch(
        points,
        normals,
        center_index,
        radius,
        max_patch_points=cap,
        patch_id=0,
        fragment_id=FRAGMENT_ID,
    )

    # Capping to exactly Max_Patch_Points when the neighborhood is larger (3.2).
    assert patch.source_indices.shape[0] == cap

    center = points[center_index]
    dists = np.linalg.norm(points - center, axis=1)
    # Every kept point is within the radius (3.1/3.2).
    kept = patch.source_indices.tolist()
    assert np.all(dists[kept] <= radius)

    # Kept set is the ``cap`` nearest, ties broken by lowest index to match the
    # deterministic extractor behavior.
    order = sorted(range(n), key=lambda i: (float(dists[i]), i))
    expected = set(order[:cap])
    assert set(int(i) for i in kept) == expected

    # Every extracted patch has a size in [1, Max_Patch_Points] (3.3).
    center_indices = list(range(min(n, 5)))
    patches = extract_patches(
        points,
        normals,
        center_indices,
        radius,
        max_patch_points=cap,
        fragment_id=FRAGMENT_ID,
    )
    for p in patches:
        size = p.source_indices.shape[0]
        assert 1 <= size <= cap


# Feature: patch-generation, Property 8: Local/global coordinate round-trip and
# one normal per point.
# Validates: Requirements 4.3, 4.4, 4.5
@settings(max_examples=100, deadline=None)
@given(_cloud_center_radius(), st.integers(min_value=1, max_value=256))
def test_property_8_local_global_roundtrip_and_one_normal_per_point(data, cap):
    points, normals, center_index, radius = data

    patch = extract_patch(
        points,
        normals,
        center_index,
        radius,
        max_patch_points=cap,
        patch_id=0,
        fragment_id=FRAGMENT_ID,
    )

    # local == global - center exactly, and local + center reproduces global
    # within 1e-4 per component (4.3).
    assert np.allclose(patch.local_coords, patch.global_coords - patch.center)
    reconstructed = patch.local_coords + patch.center
    assert np.allclose(reconstructed, patch.global_coords, atol=1e-4, rtol=0.0)

    count = patch.source_indices.shape[0]
    # Exactly one normal per point (4.4, 4.5).
    assert patch.normals.shape[0] == count
    assert patch.global_coords.shape[0] == count
    for row, src in enumerate(patch.source_indices.tolist()):
        assert np.array_equal(patch.normals[row], normals[src])


# Feature: patch-generation, Property 9: Patch_IDs are unique within a Fragment.
# Validates: Requirements 4.2
@settings(max_examples=100, deadline=None)
@given(
    _cloud_center_radius(),
    st.integers(min_value=1, max_value=256),
    st.data(),
)
def test_property_9_patch_ids_unique_within_fragment(data, cap, sampler):
    points, normals, _center_index, radius = data
    n = points.shape[0]

    # Draw an ordered, non-empty list of center indices into the cloud.
    center_indices = sampler.draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=1,
            max_size=min(n, 16),
        )
    )

    patches = extract_patches(
        points,
        normals,
        center_indices,
        radius,
        max_patch_points=cap,
        fragment_id=FRAGMENT_ID,
    )

    patch_ids = [p.patch_id for p in patches]
    # Pairwise distinct Patch_IDs within the Fragment (4.2).
    assert len(patch_ids) == len(patches)
    assert len(set(patch_ids)) == len(patch_ids)
