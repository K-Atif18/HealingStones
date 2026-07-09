"""Shared geometry data models (Mesh, PointCloud).

Defines the in-memory :class:`Mesh` and :class:`PointCloud` dataclasses per the
design ("Data Models" section) together with conversions to/from Open3D types
(:class:`open3d.geometry.TriangleMesh` and :class:`open3d.geometry.PointCloud`).

Schema (all coordinates in millimeters):
    Mesh.vertices  -> (N, 3) float64
    Mesh.faces     -> (M, 3) int32 vertex indices
    Mesh.colors    -> (N, 3) float in [0, 1] or None
    PointCloud.points  -> (N, 3) float64
    PointCloud.colors  -> (N, 3) float in [0, 1] or None
    PointCloud.normals -> (N, 3) float64 unit vectors or None

Empty geometry (zero vertices / zero points) is handled gracefully: arrays are
normalized to correctly-shaped ``(0, 3)`` arrays so downstream code can rely on
the ``.shape[1] == 3`` invariant without special-casing emptiness.

Requirements: 2.1, 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

__all__ = ["Mesh", "PointCloud"]


def _as_float_array(data: np.ndarray | None, name: str) -> np.ndarray:
    """Coerce ``data`` to a contiguous ``(K, 3)`` float64 array.

    ``None`` and empty inputs both yield a canonical ``(0, 3)`` array so callers
    never have to special-case empty geometry.
    """
    if data is None:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (K, 3), got {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_int_array(data: np.ndarray | None, name: str) -> np.ndarray:
    """Coerce ``data`` to a contiguous ``(K, 3)`` int32 array."""
    if data is None:
        return np.empty((0, 3), dtype=np.int32)
    arr = np.asarray(data, dtype=np.int32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (K, 3), got {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_optional_float_array(
    data: np.ndarray | None, name: str, expected_rows: int
) -> np.ndarray | None:
    """Coerce optional attribute (colors/normals) to ``(N, 3)`` float64 or None.

    An input that is ``None`` or empty resolves to ``None`` (attribute absent).
    A present attribute must match ``expected_rows`` so per-element attributes
    stay aligned with the geometry.
    """
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if arr.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {arr.shape[0]} rows but geometry has {expected_rows}"
        )
    return np.ascontiguousarray(arr)


@dataclass
class Mesh:
    """A triangular mesh with optional per-vertex colors.

    Attributes:
        vertices: ``(N, 3)`` float64 vertex coordinates in millimeters.
        faces: ``(M, 3)`` int32 triangle vertex indices.
        colors: ``(N, 3)`` float per-vertex colors in ``[0, 1]``, or ``None``
            when the mesh has no color information.
    """

    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.int32))
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = _as_float_array(self.vertices, "vertices")
        self.faces = _as_int_array(self.faces, "faces")
        self.colors = _as_optional_float_array(
            self.colors, "colors", self.vertices.shape[0]
        )

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the mesh."""
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangular faces in the mesh."""
        return int(self.faces.shape[0])

    @property
    def is_empty(self) -> bool:
        """True when the mesh has no vertices and no faces."""
        return self.vertex_count == 0 and self.face_count == 0

    @property
    def has_colors(self) -> bool:
        """True when the mesh carries per-vertex colors."""
        return self.colors is not None

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        """Convert to an :class:`open3d.geometry.TriangleMesh`.

        Empty meshes produce a valid empty Open3D mesh. Colors are attached
        only when present.
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.ascontiguousarray(self.vertices, dtype=np.float64)
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(self.faces, dtype=np.int32)
        )
        if self.colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.colors, dtype=np.float64)
            )
        return mesh

    @classmethod
    def from_open3d(cls, mesh: o3d.geometry.TriangleMesh) -> "Mesh":
        """Build a :class:`Mesh` from an :class:`open3d.geometry.TriangleMesh`.

        Per-vertex colors are captured only when the source mesh has them
        (Req 2.1, 2.2).
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        colors = None
        if mesh.has_vertex_colors():
            candidate = np.asarray(mesh.vertex_colors, dtype=np.float64)
            if candidate.size > 0:
                colors = candidate
        return cls(vertices=vertices, faces=faces, colors=colors)


@dataclass
class PointCloud:
    """A point cloud with optional per-point colors and normals.

    Attributes:
        points: ``(N, 3)`` float64 point coordinates in millimeters.
        colors: ``(N, 3)`` float per-point colors in ``[0, 1]``, or ``None``.
        normals: ``(N, 3)`` float64 unit normal vectors, or ``None``.
    """

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float64))
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = _as_float_array(self.points, "points")
        self.colors = _as_optional_float_array(
            self.colors, "colors", self.points.shape[0]
        )
        self.normals = _as_optional_float_array(
            self.normals, "normals", self.points.shape[0]
        )

    @property
    def point_count(self) -> int:
        """Number of points in the cloud."""
        return int(self.points.shape[0])

    @property
    def is_empty(self) -> bool:
        """True when the cloud has no points."""
        return self.point_count == 0

    @property
    def has_colors(self) -> bool:
        """True when the cloud carries per-point colors."""
        return self.colors is not None

    @property
    def has_normals(self) -> bool:
        """True when the cloud carries per-point normals."""
        return self.normals is not None

    def to_open3d(self) -> o3d.geometry.PointCloud:
        """Convert to an :class:`open3d.geometry.PointCloud`.

        Empty clouds produce a valid empty Open3D point cloud. Colors and
        normals are attached only when present.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(self.points, dtype=np.float64)
        )
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.colors, dtype=np.float64)
            )
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.normals, dtype=np.float64)
            )
        return pcd

    @classmethod
    def from_open3d(cls, pcd: o3d.geometry.PointCloud) -> "PointCloud":
        """Build a :class:`PointCloud` from an :class:`open3d.geometry.PointCloud`.

        Colors and normals are captured only when the source cloud has them.
        """
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = None
        if pcd.has_colors():
            candidate = np.asarray(pcd.colors, dtype=np.float64)
            if candidate.size > 0:
                colors = candidate
        normals = None
        if pcd.has_normals():
            candidate = np.asarray(pcd.normals, dtype=np.float64)
            if candidate.size > 0:
                normals = candidate
        return cls(points=points, colors=colors, normals=normals)
