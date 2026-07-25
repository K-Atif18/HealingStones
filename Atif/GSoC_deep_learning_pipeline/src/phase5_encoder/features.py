"""Rotation+translation-invariant per-point input features for the patch encoder.

Why not PCA-frame canonicalisation
-----------------------------------
Canonicalising a patch into its PCA local frame is only repeatable when the
covariance eigenvalues are well separated. On near-isotropic patches (the pose
gate stresses exactly these: ``standard_normal`` clouds) eigenvector ordering and
sign are ill-conditioned and flip under rotation, so the frame -- and therefore
any embedding built on it -- is *not* pose invariant. That is a design limitation
of canonicalisation, not a bug to patch.

Point Pair Features (PPF)
-------------------------
Instead we describe each patch point by four scalars that are algebraically
invariant under any rigid transform ``x -> R x + t`` (with normals ``n -> R n``),
computed relative to the patch centroid ``c`` and the (unit) mean normal ``n_c``:

    d_i   = || p_i - c ||                       (radial distance)
    a1_i  = angle( n_c ,  p_i - c )             (centroid-normal vs radial dir)
    a2_i  = angle( n_i ,  p_i - c )             (point-normal vs radial dir)
    a3_i  = angle( n_c ,  n_i )                 (normal-normal)

Angles are stored as cosines (bounded, smooth). Because ``p_i - c`` and both
normals rotate by the same ``R`` and the translation cancels in ``p_i - c``, all
four are unchanged by rigid motion. A network consuming only these is pose
invariant *by construction* -> the pose gate passes to machine precision for any
input, isotropic or not.

The feature is also permutation-equivariant per point (no ordering assumption);
permutation invariance of the whole patch comes from the max-pool in the encoder.
"""

from __future__ import annotations

import numpy as np


__all__ = ["ppf_features", "knn_ppf_features", "FEATURE_DIM", "PAIR_FEATURE_DIM"]

# d, cos(a1), cos(a2), cos(a3)  -- single-reference (centroid) variant
FEATURE_DIM = 4
# per (point, neighbour) pair: [||p_i-p_j||, cos(n_i, d_ij), cos(n_j, d_ij), cos(n_i, n_j)]
PAIR_FEATURE_DIM = 4


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def ppf_features(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Compute (N, 4) rigid-invariant PPF features for one patch.

    Parameters
    ----------
    points : (N, 3) float
        Patch point coordinates in *any* frame (invariance makes the frame
        irrelevant).
    normals : (N, 3) float
        Per-point unit normals (re-normalised defensively here).

    Returns
    -------
    (N, 4) float64 array: ``[d, cos(a1), cos(a2), cos(a3)]`` per point.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"normals must match points shape, got {normals.shape}")

    n = points.shape[0]
    if n == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64)

    normals = _unit(normals, axis=1)

    c = points.mean(axis=0)
    n_c = _unit(normals.mean(axis=0, keepdims=True), axis=1)[0]  # (3,)

    rel = points - c                       # (N,3), invariant up to shared R
    d = np.linalg.norm(rel, axis=1)        # (N,)
    rel_dir = _unit(rel, axis=1)           # (N,3)

    cos_a1 = np.clip(rel_dir @ n_c, -1.0, 1.0)                 # (N,)
    cos_a2 = np.clip(np.einsum("ij,ij->i", normals, rel_dir), -1.0, 1.0)
    cos_a3 = np.clip(normals @ n_c, -1.0, 1.0)                 # (N,)

    feats = np.stack([d, cos_a1, cos_a2, cos_a3], axis=1)
    return feats.astype(np.float64)


def knn_ppf_features(
    points: np.ndarray, normals: np.ndarray, k: int = 16
) -> np.ndarray:
    """Compute (N, k, 4) k-NN pairwise Point Pair Features for one patch.

    For each point ``i`` and each of its ``k`` nearest neighbours ``j`` (by
    Euclidean distance, self excluded), the classic PPF 4-tuple (Drost et al.
    2010; PPFNet/PPF-FoldNet input) is:

        f_ij = [ ||p_i - p_j||,
                 cos angle(n_i, p_j - p_i),
                 cos angle(n_j, p_j - p_i),
                 cos angle(n_i, n_j) ]

    Every term uses only vector norms and angles between vectors that all
    transform by the same rotation ``R`` (translation cancels in ``p_j - p_i``),
    so ``f_ij`` is **exactly** invariant under any rigid transform -- the pose
    gate still holds to machine precision (verified, not assumed).

    This retains *local pairwise* surface structure (comparable to FPFH's
    Darboux-angle histograms) instead of collapsing to a single-reference
    summary, so Phase 5A is not representationally handicapped below the FPFH
    baseline it must beat.

    If the patch has fewer than ``k+1`` points, neighbour rows are padded by
    repeating the nearest available neighbours (deterministic).
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"normals must match points shape, got {normals.shape}")

    n = points.shape[0]
    if n == 0:
        return np.zeros((0, k, PAIR_FEATURE_DIM), dtype=np.float64)

    normals = _unit(normals, axis=1)

    # Pairwise squared distances (N,N); patches are small (~64-128 pts) so this
    # dense computation is cheap and vectorised.
    diff = points[:, None, :] - points[None, :, :]      # (N,N,3) = p_i - p_j
    d2 = np.einsum("ijk,ijk->ij", diff, diff)           # (N,N)
    np.fill_diagonal(d2, np.inf)                        # exclude self

    kk = min(k, n - 1) if n > 1 else 1
    # k nearest neighbour indices per row.
    nn_idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]  # (N, kk)
    # sort those kk by actual distance for determinism
    for i in range(n):
        order = np.argsort(d2[i, nn_idx[i]], kind="mergesort")
        nn_idx[i] = nn_idx[i][order]

    # pad to k by repeating (deterministic) if kk < k
    if kk < k:
        pad = np.tile(nn_idx[:, :1], (1, k - kk)) if kk >= 1 else np.zeros((n, k - kk), np.int64)
        nn_idx = np.concatenate([nn_idx, pad], axis=1)
    nn_idx = nn_idx[:, :k]

    p_i = points[:, None, :]                            # (N,1,3)
    p_j = points[nn_idx]                                # (N,k,3)
    n_i = normals[:, None, :]                           # (N,1,3)
    n_j = normals[nn_idx]                               # (N,k,3)

    dvec = p_j - p_i                                    # (N,k,3), rotates by R
    dist = np.linalg.norm(dvec, axis=2)                 # (N,k)
    ddir = dvec / np.where(dist[..., None] < 1e-12, 1.0, dist[..., None])

    cos1 = np.clip(np.einsum("nkd,nkd->nk", np.broadcast_to(n_i, ddir.shape), ddir), -1, 1)
    cos2 = np.clip(np.einsum("nkd,nkd->nk", n_j, ddir), -1, 1)
    cos3 = np.clip(np.einsum("nkd,nkd->nk", np.broadcast_to(n_i, n_j.shape), n_j), -1, 1)

    feats = np.stack([dist, cos1, cos2, cos3], axis=2)  # (N,k,4)
    return feats.astype(np.float64)
