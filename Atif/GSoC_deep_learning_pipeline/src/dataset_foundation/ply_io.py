"""Binary little-endian PLY mesh/point-cloud I/O.

Read/write binary little-endian PLY meshes and point clouds and derive point
clouds from meshes. All geometry is in millimeters.

Responsibilities (design "ply_io" section):
    * ``load_mesh``          -- read a binary LE PLY mesh, detecting per-vertex
                                color presence; an empty PLY loads as a valid
                                empty :class:`Mesh`; a missing or unreadable
                                file raises :class:`PlyReadError` naming the path.
    * ``mesh_to_point_cloud``-- derive a :class:`PointCloud` from a mesh's
                                vertices; point count equals vertex count and
                                colors are carried only when present.
    * ``write_point_cloud`` / ``read_point_cloud`` -- round-trippable binary LE
                                PLY point-cloud I/O (coordinates within 1e-4 mm
                                and normals within 1e-4 per component, Req 7.3).
    * ``write_mesh``         -- write a mesh as a binary LE PLY file.

Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 7.2, 7.3.
"""

from __future__ import annotations

import os

import open3d as o3d

from .errors import PlyReadError, WriteError
from .geometry import Mesh, PointCloud

__all__ = [
    "load_mesh",
    "mesh_to_point_cloud",
    "write_point_cloud",
    "read_point_cloud",
    "write_mesh",
]

# Every well-formed PLY file (ASCII or binary) begins with this magic token.
_PLY_MAGIC = b"ply"

# Minimal binary little-endian PLY headers for empty geometry. Open3D refuses to
# write geometry with zero elements, but the pipeline must still emit empty
# artifacts for zero-point/zero-vertex fragments (Req 4.6, 5.8, 7.1) and empty
# meshes must load back as valid empty geometry (Req 2.6). We write these
# headers directly so the files remain valid, binary little-endian, and
# round-trippable to empty geometry.
_EMPTY_POINT_CLOUD_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex 0\n"
    "property double x\n"
    "property double y\n"
    "property double z\n"
    "end_header\n"
)

_EMPTY_MESH_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex 0\n"
    "property double x\n"
    "property double y\n"
    "property double z\n"
    "element face 0\n"
    "property list uchar int vertex_indices\n"
    "end_header\n"
)


def _write_header_only_ply(path: str, header: str) -> None:
    """Write a valid binary little-endian PLY containing only ``header``.

    Used for empty geometry, which Open3D refuses to write. Raises
    :class:`WriteError` naming the path on any I/O failure.
    """
    try:
        with open(path, "wb") as handle:
            handle.write(header.encode("ascii"))
    except OSError as exc:
        raise WriteError(path, str(exc)) from exc


def _ensure_readable_ply(path: str) -> None:
    """Validate that ``path`` names an existing, readable PLY file.

    Raises :class:`PlyReadError` (naming ``path``) when the file is missing, is
    not a regular file, is not readable, or does not begin with the PLY magic
    header. Catching the missing/unreadable cases explicitly is required because
    Open3D's readers do not raise on failure -- they silently return empty
    geometry (Req 2.5).
    """
    if not os.path.exists(path):
        raise PlyReadError(path, "file does not exist")
    if not os.path.isfile(path):
        raise PlyReadError(path, "path is not a regular file")
    if not os.access(path, os.R_OK):
        raise PlyReadError(path, "file is not readable")
    try:
        with open(path, "rb") as handle:
            header = handle.read(len(_PLY_MAGIC))
    except OSError as exc:
        raise PlyReadError(path, str(exc)) from exc
    if header[: len(_PLY_MAGIC)] != _PLY_MAGIC:
        raise PlyReadError(path, "file is not a valid PLY (missing 'ply' header)")


def load_mesh(path: str) -> Mesh:
    """Load a binary little-endian PLY mesh into a :class:`Mesh`.

    Per-vertex color presence is detected automatically: colors are attached
    only when the source file carries them (Req 2.1, 2.2). A valid PLY that
    declares zero vertices and zero faces loads as a valid empty ``Mesh``
    (Req 2.6). A missing or unreadable file raises :class:`PlyReadError`
    naming the offending path and produces no ``Mesh`` (Req 2.5).

    Args:
        path: Filesystem path to the PLY mesh file.

    Returns:
        The loaded :class:`Mesh` (possibly empty).

    Raises:
        PlyReadError: If the file is missing, is not readable, or cannot be
            parsed as a PLY mesh.
    """
    _ensure_readable_ply(path)
    try:
        o3d_mesh = o3d.io.read_triangle_mesh(path)
    except Exception as exc:  # pragma: no cover - Open3D rarely raises
        raise PlyReadError(path, str(exc)) from exc
    return Mesh.from_open3d(o3d_mesh)


def mesh_to_point_cloud(mesh: Mesh) -> PointCloud:
    """Derive a :class:`PointCloud` from a mesh's vertices.

    The derived cloud has exactly one point per mesh vertex (point count equals
    vertex count, Req 2.3, 2.7) and carries per-point colors only when the mesh
    has per-vertex colors. Normals are not derived here.

    Args:
        mesh: The source :class:`Mesh`.

    Returns:
        A :class:`PointCloud` whose points are the mesh vertices.
    """
    colors = mesh.colors if mesh.has_colors else None
    return PointCloud(points=mesh.vertices, colors=colors, normals=None)


def write_point_cloud(path: str, pc: PointCloud) -> None:
    """Write a :class:`PointCloud` to a binary little-endian PLY file.

    Reading the written file back reproduces point coordinates within 1e-4 mm
    and, where present, each normal component within 1e-4 (Req 7.2, 7.3).

    Args:
        path: Target filesystem path.
        pc: The point cloud to write.

    Raises:
        WriteError: If the file cannot be written, naming the target path.
    """
    if pc.point_count == 0:
        _write_header_only_ply(path, _EMPTY_POINT_CLOUD_HEADER)
        return
    pcd = pc.to_open3d()
    try:
        ok = o3d.io.write_point_cloud(
            path, pcd, write_ascii=False, compressed=False, print_progress=False
        )
    except Exception as exc:
        raise WriteError(path, str(exc)) from exc
    if not ok:
        raise WriteError(path, "Open3D failed to write point cloud")


def read_point_cloud(path: str) -> PointCloud:
    """Read a binary little-endian PLY point cloud into a :class:`PointCloud`.

    Per-point colors and normals are attached only when present in the file.

    Args:
        path: Filesystem path to the PLY point-cloud file.

    Returns:
        The loaded :class:`PointCloud` (possibly empty).

    Raises:
        PlyReadError: If the file is missing, is not readable, or cannot be
            parsed as a PLY point cloud.
    """
    _ensure_readable_ply(path)
    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception as exc:  # pragma: no cover - Open3D rarely raises
        raise PlyReadError(path, str(exc)) from exc
    return PointCloud.from_open3d(pcd)


def write_mesh(path: str, mesh: Mesh) -> None:
    """Write a :class:`Mesh` to a binary little-endian PLY file.

    Per-vertex colors are written only when present. Empty meshes produce a
    valid empty PLY file.

    Args:
        path: Target filesystem path.
        mesh: The mesh to write.

    Raises:
        WriteError: If the file cannot be written, naming the target path.
    """
    if mesh.vertex_count == 0:
        _write_header_only_ply(path, _EMPTY_MESH_HEADER)
        return
    o3d_mesh = mesh.to_open3d()
    try:
        ok = o3d.io.write_triangle_mesh(
            path,
            o3d_mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_colors=mesh.has_colors,
            print_progress=False,
        )
    except Exception as exc:
        raise WriteError(path, str(exc)) from exc
    if not ok:
        raise WriteError(path, "Open3D failed to write mesh")
