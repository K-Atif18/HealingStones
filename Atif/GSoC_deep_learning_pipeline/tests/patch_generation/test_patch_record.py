"""Property-based tests for ``patch_generation.patch_record``.

Covers Property 12 of the Patch Generation design:

* Property 12 - Patch_Record serialization round-trip: writing patches to a
  ``.npz`` archive then reading them back reproduces every Patch_Center,
  Global_Coordinate, Local_Coordinate and normal within 1e-4 per component, and
  reproduces every Patch_ID, owning Fragment_ID and source index exactly
  (Req 4.1, 7.1, 7.3).

Patches are produced through ``patch_generation.patch_extractor.extract_patches``
on a randomly generated point cloud + matched normals + a few center indices,
which guarantees internally consistent Patch records. The empty-Fragment
(zero-patch) case is exercised occasionally.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from patch_generation.patch_extractor import extract_patches
from patch_generation.patch_record import read_patch_records, write_patch_records

FRAGMENT_ID = "fragment_roundtrip_test"

# Finite float coordinates at a realistic mm scale for the Caesar model.
_finite = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _rows(n: int):
    """Strategy for an ``(n, 3)`` float64 array of finite coordinates."""

    return st.lists(
        st.tuples(_finite, _finite, _finite),
        min_size=n,
        max_size=n,
    ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))


@st.composite
def _patch_lists(draw):
    """A realistic list of Patch objects for a single Fragment.

    Generates a random point cloud with matched normals, draws a set of center
    indices, and runs the real extractor so every Patch is internally
    consistent. Occasionally yields the empty-list (zero-patch) case.
    """

    # Occasionally exercise the empty-Fragment (zero-patch) archive.
    if draw(st.integers(min_value=0, max_value=9)) == 0:
        return []

    n = draw(st.integers(min_value=1, max_value=40))
    points = draw(_rows(n))
    normals = draw(_rows(n))
    center_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=1,
            max_size=min(n, 12),
        )
    )
    radius = draw(
        st.floats(
            min_value=0.1,
            max_value=5.0e3,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    cap = draw(st.integers(min_value=1, max_value=64))

    return extract_patches(
        points,
        normals,
        center_indices,
        patch_radius_mm=radius,
        max_patch_points=cap,
        fragment_id=FRAGMENT_ID,
    )


# Feature: patch-generation, Property 12: Patch_Record serialization round-trip.
# Validates: Requirements 4.1, 7.1, 7.3
@settings(max_examples=100, deadline=None)
@given(_patch_lists())
def test_property_12_patch_record_serialization_roundtrip(patches):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "patches.npz")

        write_patch_records(path, patches, FRAGMENT_ID)
        restored = read_patch_records(path)

        # Same number of patches survives the round-trip.
        assert len(restored) == len(patches)

        # Match by Patch_ID (unique within a Fragment) so ordering is irrelevant.
        by_id = {p.patch_id: p for p in restored}
        assert len(by_id) == len(restored)

        for original in patches:
            assert original.patch_id in by_id
            got = by_id[original.patch_id]

            # Exact identity fields (7.3): Patch_ID, Fragment_ID, source indices.
            assert got.patch_id == original.patch_id
            assert got.fragment_id == original.fragment_id == FRAGMENT_ID
            assert np.array_equal(got.source_indices, original.source_indices)

            # Geometry reproduced within 1e-4 per component (4.1, 7.1).
            assert np.allclose(
                got.center, original.center, atol=1e-4, rtol=0.0
            )
            assert np.allclose(
                got.global_coords, original.global_coords, atol=1e-4, rtol=0.0
            )
            assert np.allclose(
                got.local_coords, original.local_coords, atol=1e-4, rtol=0.0
            )
            assert np.allclose(
                got.normals, original.normals, atol=1e-4, rtol=0.0
            )
