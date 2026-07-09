"""Property-based and unit tests for binary little-endian PLY I/O.

Covers three task-list items, all exercising ``dataset_foundation.ply_io``:

* Task 3.3 / Property 5  -- mesh load/derive round-trip and point-cloud count
  (Requirements 2.1, 2.2, 2.3, 2.7).
* Task 3.4 / Property 19 -- point cloud PLY round-trip (Requirements 7.2, 7.3).
* Task 3.5 (unit tests)  -- PLY error and empty-mesh paths
  (Requirements 2.4, 2.5, 2.6, 7.2).

Round-trips go through real Open3D readers/writers on real temporary files so
the tests validate genuine on-disk behavior rather than mocks.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.errors import PlyReadError
from dataset_foundation.geometry import Mesh, PointCloud
from dataset_foundation.ply_io import (
    load_mesh,
    mesh_to_point_cloud,
    read_point_cloud,
    write_mesh,
    write_point_cloud,
)

# ---------------------------------------------------------------------------
# Shared strategies.
# ---------------------------------------------------------------------------

# Finite mm-scale coordinates. Bounded to a realistic range and free of
# NaN/inf so on-disk round-trips are numerically well-defined.
_coord = st.floats(min_value=-1.0e4, max_value=1.0e4, allow_nan=False, allow_infinity=False)

# Per-element color components in [0, 1].
_color01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _rows(n: int, elem):
    """Strategy for an (n, 3) float64 array whose entries are drawn from ``elem``."""
    return st.lists(
        st.tuples(elem, elem, elem), min_size=n, max_size=n
    ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))


def _distinct_triple(n: int):
    """Strategy for a triangle of three distinct vertex indices in ``[0, n)``."""
    return st.lists(
        st.integers(min_value=0, max_value=n - 1), min_size=3, max_size=3, unique=True
    ).map(tuple)


@st.composite
def meshes(draw, min_vertices: int = 1, max_vertices: int = 30):
    """Strategy for a non-empty :class:`Mesh` with optional faces and colors.

    Vertices are finite mm-scale coordinates; faces (when present and the mesh
    has at least three vertices) reference distinct existing vertices; colors
    are present or absent so color-presence preservation can be checked.
    """
    n = draw(st.integers(min_value=min_vertices, max_value=max_vertices))
    vertices = draw(_rows(n, _coord))

    if n >= 3 and draw(st.booleans()):
        m = draw(st.integers(min_value=1, max_value=20))
        faces = draw(
            st.lists(_distinct_triple(n), min_size=m, max_size=m).map(
                lambda r: np.asarray(r, dtype=np.int32).reshape((len(r), 3))
            )
        )
    else:
        faces = np.empty((0, 3), dtype=np.int32)

    colors = draw(_rows(n, _color01)) if draw(st.booleans()) else None
    return Mesh(vertices=vertices, faces=faces, colors=colors)


@st.composite
def point_clouds(draw, min_points: int = 1, max_points: int = 40):
    """Strategy for a non-empty :class:`PointCloud` with optional normals/colors."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    points = draw(_rows(n, _coord))
    # Round-trip fidelity does not require unit normals, so arbitrary finite
    # vectors are sufficient to exercise the normal round-trip.
    normals = draw(_rows(n, _coord)) if draw(st.booleans()) else None
    colors = draw(_rows(n, _color01)) if draw(st.booleans()) else None
    return PointCloud(points=points, colors=colors, normals=normals)


# ---------------------------------------------------------------------------
# Task 3.3 / Property 5: Mesh load/derive round-trip and point-cloud count.
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 5: Mesh load/derive round-trip and point-cloud count
@settings(max_examples=100)
@given(mesh=meshes())
def test_mesh_load_derive_round_trip_and_point_count(mesh):
    """Validates: Requirements 2.1, 2.2, 2.3, 2.7"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mesh.ply")
        write_mesh(path, mesh)
        loaded = load_mesh(path)

    # Vertices preserved within tolerance (binary PLY carries float64 points).
    assert loaded.vertex_count == mesh.vertex_count
    assert np.allclose(loaded.vertices, mesh.vertices, atol=1e-4)

    # Faces preserved exactly (both count and index content, order-preserving).
    assert loaded.face_count == mesh.face_count
    assert np.array_equal(loaded.faces, mesh.faces)

    # Color *presence* is preserved: colors round-trip iff the source had them.
    assert loaded.has_colors == mesh.has_colors

    # Derived point cloud has one point per vertex (Req 2.3, 2.7) and carries
    # colors only when the mesh has per-vertex colors.
    pc = mesh_to_point_cloud(mesh)
    assert pc.point_count == mesh.vertex_count
    assert pc.has_colors == mesh.has_colors
    assert np.allclose(pc.points, mesh.vertices, atol=0.0)


# ---------------------------------------------------------------------------
# Task 3.4 / Property 19: Point cloud PLY round-trip.
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 19: Point cloud PLY round-trip
@settings(max_examples=100)
@given(pc=point_clouds())
def test_point_cloud_ply_round_trip(pc):
    """Validates: Requirements 7.2, 7.3"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cloud.ply")
        write_point_cloud(path, pc)
        loaded = read_point_cloud(path)

    # Point count and per-coordinate fidelity within 1e-4 mm (Req 7.3).
    assert loaded.point_count == pc.point_count
    assert np.max(np.abs(loaded.points - pc.points)) <= 1e-4

    # Normals are attached only when present, and each component round-trips
    # within 1e-4 (Req 7.3).
    assert loaded.has_normals == pc.has_normals
    if pc.has_normals:
        assert np.max(np.abs(loaded.normals - pc.normals)) <= 1e-4

    # Color presence is preserved (values are quantized on disk, so only
    # presence is asserted here).
    assert loaded.has_colors == pc.has_colors


# ---------------------------------------------------------------------------
# Task 3.5: unit tests for PLY error and empty-mesh paths.
# ---------------------------------------------------------------------------


def test_load_mesh_missing_path_raises_naming_path(tmp_path):
    """A non-existent PLY path raises PlyReadError naming the path (Req 2.5)."""
    missing = str(tmp_path / "does_not_exist.ply")
    with pytest.raises(PlyReadError) as exc_info:
        load_mesh(missing)
    assert missing in str(exc_info.value)


def test_load_mesh_garbage_file_raises(tmp_path):
    """A non-PLY / garbage file raises PlyReadError (Req 2.5)."""
    garbage = tmp_path / "garbage.ply"
    garbage.write_bytes(b"this is definitely not a PLY file\n\x00\x01\x02")
    with pytest.raises(PlyReadError) as exc_info:
        load_mesh(str(garbage))
    assert str(garbage) in str(exc_info.value)


def test_empty_mesh_round_trips_as_valid_empty_mesh(tmp_path):
    """An empty mesh written to disk loads back as a valid empty Mesh (Req 2.6)."""
    path = str(tmp_path / "empty_mesh.ply")
    write_mesh(path, Mesh())

    loaded = load_mesh(path)
    assert loaded.is_empty
    assert loaded.vertex_count == 0
    assert loaded.face_count == 0
    assert not loaded.has_colors


def test_written_point_cloud_header_declares_binary_little_endian(tmp_path):
    """A written point-cloud PLY declares binary_little_endian in its header (Req 7.2)."""
    path = str(tmp_path / "cloud.ply")
    pc = PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    )
    write_point_cloud(path, pc)

    with open(path, "rb") as handle:
        header = handle.read(256)
    assert b"binary_little_endian" in header
