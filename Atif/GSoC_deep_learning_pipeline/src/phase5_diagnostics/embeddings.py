"""Embedding sources for the diagnostics.

Every diagnostic consumes ``dict[str, PatchDescriptors]`` (fragment_id ->
descriptors). This module builds the reference embedding sources used for
calibration, plus loaders for existing on-disk descriptors:

* ``random_embeddings``   : Gaussian noise, L2-normalised. The *null floor* --
                            it contains zero assembly information, so whatever
                            mAP it scores is pure harness generosity (driven by
                            large GT-set sizes per query).
* ``centroid_embeddings`` : trivial hand-features per patch -- point count,
                            local coordinate spread (bbox extents), mean normal.
                            A weak, information-poor baseline: any real method
                            must clear it comfortably.
* ``load_fpfh``           : the Phase 4 FPFH descriptors from disk.

The ``PatchDescriptors`` dataclass (fragment_id, descriptor_type, patch_ids,
descriptors[M,D] L2-normalised) is reused verbatim from Phase 4 so the retrieval
harness and every probe treat all sources identically.
"""

from __future__ import annotations

import os

import numpy as np

from baseline_geometry.data_access import PatchTable
from baseline_geometry.descriptors import PatchDescriptors, read_descriptors

from phase5_diagnostics.context import DiagnosticContext


__all__ = [
    "l2_normalize_rows",
    "random_embeddings",
    "centroid_embeddings",
    "load_fpfh",
]


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return mat / norms


def random_embeddings(
    ctx: DiagnosticContext, *, dim: int = 64, seed: int = 0
) -> dict[str, PatchDescriptors]:
    """Null floor: Gaussian random vectors, L2-normalised. No information."""
    rng = np.random.default_rng(seed)
    out: dict[str, PatchDescriptors] = {}
    for fid in ctx.fragment_ids:
        table = ctx.patch_tables[fid]
        m = table.patch_count
        mat = l2_normalize_rows(rng.standard_normal((m, dim)))
        out[fid] = PatchDescriptors(
            fragment_id=fid,
            descriptor_type="random",
            patch_ids=table.patch_ids.copy(),
            descriptors=mat.astype(np.float64),
        )
    return out


def _patch_centroid_features(table: PatchTable) -> np.ndarray:
    """Trivial per-patch features: [n_points, bbox extents (3), mean |normal| dirs (3)].

    Uses only *local* geometry (extents around the patch centre and normal
    orientation statistics). Deliberately weak and pose-dependent-free-ish; this
    is a floor, not a contender.
    """
    m = table.patch_count
    feats = np.zeros((m, 7), dtype=np.float64)
    for row in range(m):
        start, end = int(table.offsets[row, 0]), int(table.offsets[row, 1])
        pts = table.global_coords[start:end]
        nrm = table.normals[start:end]
        if pts.shape[0] == 0:
            continue
        centred = pts - pts.mean(axis=0, keepdims=True)
        extents = centred.max(axis=0) - centred.min(axis=0)
        mean_n = nrm.mean(axis=0)
        feats[row, 0] = pts.shape[0]
        feats[row, 1:4] = extents
        feats[row, 4:7] = mean_n
    return feats


def centroid_embeddings(ctx: DiagnosticContext) -> dict[str, PatchDescriptors]:
    """Weak baseline from trivial per-patch statistics (standardised + L2)."""
    # Collect all features first to standardise columns globally.
    raw: dict[str, np.ndarray] = {}
    for fid in ctx.fragment_ids:
        raw[fid] = _patch_centroid_features(ctx.patch_tables[fid])
    stacked = np.vstack(list(raw.values()))
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0

    out: dict[str, PatchDescriptors] = {}
    for fid in ctx.fragment_ids:
        z = (raw[fid] - mean) / std
        out[fid] = PatchDescriptors(
            fragment_id=fid,
            descriptor_type="centroid",
            patch_ids=ctx.patch_tables[fid].patch_ids.copy(),
            descriptors=l2_normalize_rows(z).astype(np.float64),
        )
    return out


def load_fpfh(
    ctx: DiagnosticContext, descriptors_root: str, dtype: str = "fpfh"
) -> dict[str, PatchDescriptors]:
    """Load the Phase 4 descriptors of a given type from disk."""
    out: dict[str, PatchDescriptors] = {}
    for fid in ctx.fragment_ids:
        path = os.path.join(descriptors_root, dtype, f"{fid}.npz")
        out[fid] = read_descriptors(path)
    return out
