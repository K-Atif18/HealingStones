"""FPFH and SHOT descriptors for Phase 2 patches.

Design
------
Both descriptors are *point-wise* by construction: they summarise the surface
in a support region around a point. We compute them on the full Phase 1
fragment cloud (so each point sees its true geometric context, including points
that belong to neighbouring patches) and then **pool** the point-wise vectors
into a single per-patch signature using the patch's ``source_indices``:

* ``patch_pooling="mean"``   -> average of the descriptors over the patch points
  (a whole-patch signature; robust, what retrieval uses by default).
* ``patch_pooling="center"`` -> the descriptor at the patch centre point only
  (sharper, more sensitive to the exact centre).

FPFH
----
33-bin Fast Point Feature Histogram (Rusu et al. 2009), via Open3D. Pose
invariant; encodes the distribution of Darboux-frame angles between a point's
normal and its neighbours' normals.

SHOT
----
352-D Signature of Histograms of OrienTations (Tombari et al. 2010),
implemented here from scratch (Open3D has no SHOT):

1. A repeatable local reference frame (LRF) from the eigenvectors of the
   distance-weighted neighbour covariance, with SHOT sign disambiguation.
2. The spherical support is partitioned into 32 volumes
   (8 azimuth x 2 elevation x 2 radial).
3. In each volume an ``shot_cos_bins``-bin histogram of ``cos(theta)`` is
   accumulated, where ``theta`` is the angle between each neighbour's normal and
   the feature point's normal.
4. Concatenate (32 * bins dims) and L2-normalise.

This is a faithful-but-simplified SHOT: it uses hard spatial/angular binning
rather than SHOT's quadrilinear interpolation. That is an intentional baseline
choice (documented) -- the goal is a solid reference point, not a tuned
descriptor.

Both descriptors measure **similarity**. See the phase notes / report for why
this matters: complementary break surfaces are geometric opposites and are
expected to be *missed* by similarity descriptors.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from baseline_geometry.data_access import FragmentCloud, PatchTable
from baseline_geometry.errors import DescriptorError, WriteError

__all__ = [
    "PatchDescriptors",
    "ensure_normals",
    "compute_fpfh_points",
    "compute_shot_points",
    "pool_to_patches",
    "compute_patch_descriptors",
    "write_descriptors",
    "read_descriptors",
]


@dataclass(frozen=True)
class PatchDescriptors:
    """Per-patch descriptor matrix for one fragment."""

    fragment_id: str
    descriptor_type: str      # "fpfh" | "shot"
    patch_ids: np.ndarray     # (M,) int64
    descriptors: np.ndarray   # (M, D) float64, L2-normalised rows

    @property
    def dim(self) -> int:
        return int(self.descriptors.shape[1]) if self.descriptors.ndim == 2 else 0

    @property
    def patch_count(self) -> int:
        return int(self.patch_ids.shape[0])


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------
def ensure_normals(cloud: FragmentCloud, radius_mm: float, max_nn: int) -> np.ndarray:
    """Return per-point unit normals, estimating them if the cloud lacks them."""
    if cloud.normals.shape[0] == cloud.points.shape[0] and cloud.normals.shape[0] > 0:
        n = cloud.normals
    else:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_mm, max_nn=max_nn)
        )
        pcd.normalize_normals()
        n = np.asarray(pcd.normals, dtype=np.float64)
    # Guarantee unit length (guard against zero vectors).
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return n / norms


# ---------------------------------------------------------------------------
# FPFH
# ---------------------------------------------------------------------------
def compute_fpfh_points(cloud: FragmentCloud, normals: np.ndarray, radius_mm: float, max_nn: int) -> np.ndarray:
    """Compute (N, 33) FPFH descriptors for every point of the cloud."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    try:
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_mm, max_nn=max_nn),
        )
    except Exception as exc:  # pragma: no cover - Open3D internal failure
        raise DescriptorError(cloud.fragment_id, f"FPFH failed: {exc}") from exc
    # Open3D returns (33, N); transpose to (N, 33).
    return np.asarray(fpfh.data, dtype=np.float64).T


# ---------------------------------------------------------------------------
# SHOT (from scratch)
# ---------------------------------------------------------------------------
def _local_reference_frame(neighbors_rel: np.ndarray, dists: np.ndarray, radius: float) -> np.ndarray:
    """Compute a SHOT-style local reference frame (rows = x, y, z axes).

    ``neighbors_rel`` are neighbour coordinates relative to the feature point.
    Uses distance-weighted covariance + eigen-decomposition, then SHOT sign
    disambiguation so the frame is repeatable.
    """
    if neighbors_rel.shape[0] < 3:
        return np.eye(3, dtype=np.float64)

    weights = np.maximum(radius - dists, 0.0)
    wsum = weights.sum()
    if wsum < 1e-12:
        return np.eye(3, dtype=np.float64)

    # Distance-weighted scatter matrix.
    weighted = neighbors_rel * weights[:, None]
    cov = (weighted.T @ neighbors_rel) / wsum
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    # x axis = largest eigenvalue eigenvector, z axis = smallest.
    x_axis = eigvecs[:, 2].copy()
    z_axis = eigvecs[:, 0].copy()

    # Sign disambiguation (Tombari et al.): flip so the axis agrees with the
    # majority of neighbour vectors.
    sx = np.sum((neighbors_rel @ x_axis) >= 0)
    if sx < (neighbors_rel.shape[0] - sx):
        x_axis = -x_axis
    sz = np.sum((neighbors_rel @ z_axis) >= 0)
    if sz < (neighbors_rel.shape[0] - sz):
        z_axis = -z_axis

    y_axis = np.cross(z_axis, x_axis)
    ny = np.linalg.norm(y_axis)
    if ny < 1e-12:
        return np.eye(3, dtype=np.float64)
    y_axis /= ny
    # Re-orthogonalise x to guarantee a proper right-handed frame.
    x_axis = np.cross(y_axis, z_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def compute_shot_points(
    points: np.ndarray,
    normals: np.ndarray,
    query_indices: np.ndarray,
    radius_mm: float,
    cos_bins: int,
) -> np.ndarray:
    """Compute SHOT descriptors at ``query_indices`` of the cloud.

    Returns ``(len(query_indices), 32 * cos_bins)`` L2-normalised descriptors.
    """
    n_azimuth, n_elevation, n_radial = 8, 2, 2
    n_volumes = n_azimuth * n_elevation * n_radial  # 32
    desc_dim = n_volumes * cos_bins

    tree = cKDTree(points)
    out = np.zeros((query_indices.shape[0], desc_dim), dtype=np.float64)
    half_radius = radius_mm * 0.5

    for row, qi in enumerate(query_indices):
        qi = int(qi)
        p = points[qi]
        feat_normal = normals[qi]

        idx = tree.query_ball_point(p, radius_mm)
        idx = [j for j in idx if j != qi]
        if len(idx) < 3:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        rel = points[idx] - p
        dists = np.linalg.norm(rel, axis=1)

        lrf = _local_reference_frame(rel, dists, radius_mm)
        # Coordinates in the LRF.
        local = rel @ lrf.T  # (k, 3): columns are x, y, z in LRF
        lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

        # Radial bin: inner / outer shell.
        radial_bin = (dists >= half_radius).astype(np.int64)
        radial_bin = np.clip(radial_bin, 0, n_radial - 1)
        # Elevation bin: below / above the LRF xy-plane.
        elev_bin = (lz >= 0).astype(np.int64)
        # Azimuth bin: 8 sectors around z.
        azimuth = np.arctan2(ly, lx)  # [-pi, pi]
        az_bin = np.floor((azimuth + np.pi) / (2 * np.pi) * n_azimuth).astype(np.int64)
        az_bin = np.clip(az_bin, 0, n_azimuth - 1)

        volume = (radial_bin * n_elevation + elev_bin) * n_azimuth + az_bin

        # Cosine of angle between neighbour normals and the feature normal.
        cos = np.clip(normals[idx] @ feat_normal, -1.0, 1.0)
        cos_bin = np.floor((cos + 1.0) / 2.0 * cos_bins).astype(np.int64)
        cos_bin = np.clip(cos_bin, 0, cos_bins - 1)

        flat_bin = volume * cos_bins + cos_bin
        np.add.at(out[row], flat_bin, 1.0)

    # L2-normalise each descriptor row.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def pool_to_patches(
    point_descriptors: np.ndarray,
    table: PatchTable,
    pooling: str,
) -> np.ndarray:
    """Pool point-wise descriptors (indexed by fragment source index) into
    one descriptor per patch.

    ``point_descriptors`` has one row per fragment cloud point. For each patch
    we gather the rows for its ``source_indices`` (mean pooling) or its centre
    point (center pooling), then L2-normalise the result.
    """
    m = table.patch_count
    dim = point_descriptors.shape[1]
    out = np.zeros((m, dim), dtype=np.float64)

    for row in range(m):
        if pooling == "center":
            ci = int(table.center_indices[row])
            vec = point_descriptors[ci]
        else:  # mean
            start, end = int(table.offsets[row, 0]), int(table.offsets[row, 1])
            src = table.source_indices[start:end]
            if src.size == 0:
                continue
            vec = point_descriptors[src].mean(axis=0)
        out[row] = vec

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


def compute_patch_descriptors(
    descriptor_type: str,
    cloud: FragmentCloud,
    table: PatchTable,
    normals: np.ndarray,
    config,
) -> PatchDescriptors:
    """Compute per-patch descriptors of the requested type for one fragment."""
    if descriptor_type == "fpfh":
        point_desc = compute_fpfh_points(cloud, normals, config.fpfh_radius_mm, config.fpfh_max_nn)
    elif descriptor_type == "shot":
        # SHOT is computed at exactly the points we need for pooling.
        if config.patch_pooling == "center":
            query = np.unique(table.center_indices.astype(np.int64))
        else:
            query = np.unique(table.source_indices.astype(np.int64))
        shot_q = compute_shot_points(
            cloud.points, normals, query, config.shot_radius_mm, config.shot_cos_bins
        )
        # Scatter back to a full (N, D) array so pooling can index by source idx.
        point_desc = np.zeros((cloud.points.shape[0], shot_q.shape[1]), dtype=np.float64)
        point_desc[query] = shot_q
    else:
        raise DescriptorError(descriptor_type, "unknown descriptor type")

    pooled = pool_to_patches(point_desc, table, config.patch_pooling)
    return PatchDescriptors(
        fragment_id=cloud.fragment_id,
        descriptor_type=descriptor_type,
        patch_ids=table.patch_ids.copy(),
        descriptors=pooled,
    )


# ---------------------------------------------------------------------------
# Serialization (atomic .npz, mirroring Phase 2)
# ---------------------------------------------------------------------------
def write_descriptors(path: str, desc: PatchDescriptors) -> str:
    """Atomically write a :class:`PatchDescriptors` to a compressed ``.npz``."""
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path: str | None = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                fragment_id=np.asarray(str(desc.fragment_id)),
                descriptor_type=np.asarray(str(desc.descriptor_type)),
                patch_ids=desc.patch_ids.astype(np.int64),
                descriptors=desc.descriptors.astype(np.float64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except (OSError, ValueError, TypeError) as exc:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise WriteError(path, str(exc)) from exc
    return path


def read_descriptors(path: str) -> PatchDescriptors:
    """Read a :class:`PatchDescriptors` from a ``.npz`` written by this module."""
    if not os.path.isfile(path):
        raise DescriptorError(path, "descriptor archive not found")
    with np.load(path, allow_pickle=False) as data:
        return PatchDescriptors(
            fragment_id=str(data["fragment_id"].item()),
            descriptor_type=str(data["descriptor_type"].item()),
            patch_ids=np.asarray(data["patch_ids"], dtype=np.int64),
            descriptors=np.asarray(data["descriptors"], dtype=np.float64),
        )
