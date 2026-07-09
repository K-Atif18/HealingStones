#!/usr/bin/env python3
"""
Task 1: Patch Distinctiveness Analysis

For every patch computes:
  - Mean curvature (via eigenvalue method on local covariance)
  - Curvature variance
  - Normal variance (spread of normals = angular disorder)
  - Surface roughness (RMS deviation from local plane)
  - Local point density (points per mm^2)
  - PCA eigenvalues (lambda1 >= lambda2 >= lambda3)
  - Linearity   = (lambda1 - lambda2) / lambda1
  - Planarity   = (lambda2 - lambda3) / lambda1
  - Scattering  = lambda3 / lambda1

Then assigns a distinctiveness score and classifies every patch.

Outputs:
  phase3_final_review/patch_distinctiveness.json
  phase3_final_review/patch_distinctiveness.npz
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── repo root on path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from patch_generation.patch_record import read_patch_records

# ── constants ────────────────────────────────────────────────────────────────
PATCHES_DIR = ROOT / "patches"
OUT_DIR = ROOT / "phase3_final_review"
PATCH_RADIUS_MM = 8.0          # from config

# ────────────────────────────────────────────────────────────────────────────
def pca_features(local_coords: np.ndarray):
    """
    Compute PCA on local_coords (N×3, centred at patch centre = origin).

    Returns eigenvalues sorted descending, and derived shape features.
    lambda1 >= lambda2 >= lambda3.

    Linearity  = (lambda1 - lambda2) / lambda1
      → 1 for needle-like, 0 for isotropic
    Planarity  = (lambda2 - lambda3) / lambda1
      → 1 for disk-like, 0 for isotropic
    Scattering = lambda3 / lambda1
      → 1 for isotropic/rough, 0 for flat/linear
    """
    n = len(local_coords)
    if n < 3:
        return np.array([1e-9, 1e-9, 1e-9]), 0.0, 0.0, 0.0

    cov = np.cov(local_coords.T)          # 3×3
    eigvals = np.linalg.eigvalsh(cov)     # ascending order
    eigvals = np.sort(eigvals)[::-1]      # descending: lambda1 >= lambda2 >= lambda3
    eigvals = np.maximum(eigvals, 1e-12)  # numerical safety

    l1, l2, l3 = eigvals
    linearity  = (l1 - l2) / l1
    planarity  = (l2 - l3) / l1
    scattering = l3 / l1
    return eigvals, linearity, planarity, scattering


def normal_variance(normals: np.ndarray) -> float:
    """
    Mean squared angular deviation of normals from their mean direction.

    Measures: how much normals spread across the patch.
    High variance → curved or rough surface (assembly-informative).
    Low variance  → flat planar region (potentially ambiguous).

    Formula:  var = mean(1 - |n_i · n_mean|^2)
    Range: [0, 1].  0 = perfectly parallel normals (flat). 1 = random normals.
    """
    if len(normals) < 2:
        return 0.0
    mean_n = normals.mean(axis=0)
    norm = np.linalg.norm(mean_n)
    if norm < 1e-9:
        return 1.0
    mean_n /= norm
    dots = normals @ mean_n          # (N,)
    return float(np.mean(1.0 - dots**2))


def mean_curvature(local_coords: np.ndarray, normals: np.ndarray) -> tuple:
    """
    Approximate mean curvature via PCA on normals + position spread.

    Method: ratio of normal variation to spatial extent.
    κ ≈ std(normals) / std(positions)

    Returns (mean_curvature, curvature_variance).
    Higher values → more curved surface → more distinctive for matching.
    """
    if len(local_coords) < 3:
        return 0.0, 0.0

    pos_spread  = np.std(local_coords)
    if pos_spread < 1e-9:
        return 0.0, 0.0

    # Per-point curvature proxy: angle of normal from mean normal
    mean_n = normals.mean(axis=0)
    norm_mn = np.linalg.norm(mean_n)
    if norm_mn < 1e-9:
        return 1.0, 1.0
    mean_n /= norm_mn

    dots = np.clip(normals @ mean_n, -1.0, 1.0)
    angles = np.arccos(np.abs(dots))   # (N,) in [0, pi/2]
    kappa_per_point = angles / pos_spread

    return float(np.mean(kappa_per_point)), float(np.var(kappa_per_point))


def surface_roughness(local_coords: np.ndarray) -> float:
    """
    RMS distance of patch points from the best-fit plane.

    Best-fit plane found via PCA: plane normal = eigenvector of smallest eigenvalue.
    roughness = sqrt(mean(d_i^2))   where d_i = signed distance to plane.

    High roughness → fracture surface or carved detail → assembly-informative.
    Low roughness  → smooth polished surface → potentially ambiguous.
    """
    if len(local_coords) < 3:
        return 0.0
    cov = np.cov(local_coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    plane_normal = eigvecs[:, 0]   # eigenvector of smallest eigenvalue
    distances = local_coords @ plane_normal
    return float(np.sqrt(np.mean(distances**2)))


def local_density(n_points: int, radius_mm: float) -> float:
    """
    Points per mm^2 on the patch.

    Approximated as: n_points / (pi * r^2).
    High density → rich geometric detail available.
    """
    area = np.pi * radius_mm**2
    return n_points / area


def distinctiveness_score(nvar: float, curv_mean: float, curv_var: float,
                           roughness: float, scattering: float,
                           planarity: float) -> float:
    """
    Composite distinctiveness score in [0, 1].

    Weights are chosen based on assembly-relevance:
      - normal_variance:  0.30  (directly measures surface complexity)
      - curvature_mean:   0.25  (curved surfaces uniquely identify matching regions)
      - roughness:        0.20  (fracture surfaces are rough)
      - scattering:       0.15  (volumetric spread = complex 3D shape)
      - planarity:        0.10  (negative weight: high planarity = flat = bad)

    Each component is normalised against empirical maximum values appropriate
    for 8mm patches at 1.22mm spacing.
    """
    # Normalising constants estimated from Caesar fragment geometry
    MAX_NVAR       = 0.5
    MAX_CURV_MEAN  = 0.15
    MAX_CURV_VAR   = 0.05
    MAX_ROUGHNESS  = 2.5
    MAX_SCATTER    = 1.0
    MAX_PLANAR     = 1.0

    score = (
        0.30 * min(nvar / MAX_NVAR, 1.0) +
        0.25 * min(curv_mean / MAX_CURV_MEAN, 1.0) +
        0.20 * min(roughness / MAX_ROUGHNESS, 1.0) +
        0.15 * min(scattering / MAX_SCATTER, 1.0) +
        0.10 * (1.0 - min(planarity / MAX_PLANAR, 1.0))  # flat = low score
    )
    return float(np.clip(score, 0.0, 1.0))


def classify(score: float) -> str:
    """
    Classify patch by distinctiveness score.

    Thresholds (percentile-based, chosen to reflect assembly usefulness):
      score >= 0.55  →  "highly_distinctive"
      score >= 0.35  →  "moderately_distinctive"
      score >= 0.15  →  "ambiguous"
      score <  0.15  →  "flat"
    """
    if score >= 0.55:
        return "highly_distinctive"
    elif score >= 0.35:
        return "moderately_distinctive"
    elif score >= 0.15:
        return "ambiguous"
    else:
        return "flat"


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TASK 1: PATCH DISTINCTIVENESS ANALYSIS")
    print("=" * 60)

    fragment_dirs = sorted([d for d in PATCHES_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(fragment_dirs)} fragment directories")

    all_records = []   # list of dicts, one per patch
    fragment_summary = {}

    for frag_dir in fragment_dirs:
        frag_id = frag_dir.name
        npz_path = frag_dir / "patches.npz"
        if not npz_path.exists():
            print(f"  SKIP {frag_id}: no patches.npz")
            continue

        patches = read_patch_records(str(npz_path))
        print(f"  {frag_id}: {len(patches)} patches", end="", flush=True)

        frag_records = []
        scores = []

        for patch in patches:
            lc   = patch.local_coords        # (P, 3)
            nrm  = patch.normals             # (P, 3)
            n_pts = len(lc)

            # --- individual metrics ---
            nvar              = normal_variance(nrm)
            k_mean, k_var     = mean_curvature(lc, nrm)
            roughness         = surface_roughness(lc)
            density           = local_density(n_pts, PATCH_RADIUS_MM)
            eigvals, lin, pln, sct = pca_features(lc)

            score = distinctiveness_score(nvar, k_mean, k_var, roughness, sct, pln)
            label = classify(score)
            scores.append(score)

            rec = {
                "fragment_id":       frag_id,
                "patch_id":          int(patch.patch_id),
                "n_points":          n_pts,
                "normal_variance":   round(nvar, 6),
                "curvature_mean":    round(k_mean, 6),
                "curvature_variance":round(k_var, 6),
                "roughness_mm":      round(roughness, 6),
                "density_pts_mm2":   round(density, 4),
                "pca_lambda1":       round(float(eigvals[0]), 6),
                "pca_lambda2":       round(float(eigvals[1]), 6),
                "pca_lambda3":       round(float(eigvals[2]), 6),
                "linearity":         round(lin, 6),
                "planarity":         round(pln, 6),
                "scattering":        round(sct, 6),
                "distinctiveness":   round(score, 6),
                "class":             label,
            }
            frag_records.append(rec)

        all_records.extend(frag_records)

        scores_arr = np.array(scores)
        counts = {c: sum(1 for r in frag_records if r["class"] == c)
                  for c in ["highly_distinctive","moderately_distinctive","ambiguous","flat"]}
        fragment_summary[frag_id] = {
            "n_patches": len(patches),
            "mean_distinctiveness": round(float(scores_arr.mean()), 4),
            "std_distinctiveness":  round(float(scores_arr.std()), 4),
            "class_counts": counts,
        }
        print(f"  → mean score {scores_arr.mean():.3f}")

    # ── global statistics ────────────────────────────────────────────────────
    all_scores = np.array([r["distinctiveness"] for r in all_records])
    all_classes = [r["class"] for r in all_records]

    class_counts = defaultdict(int)
    for c in all_classes:
        class_counts[c] += 1

    total = len(all_records)
    print(f"\nTotal patches analysed: {total}")
    for cls in ["highly_distinctive","moderately_distinctive","ambiguous","flat"]:
        n = class_counts[cls]
        print(f"  {cls:30s}: {n:5d}  ({100*n/total:.1f}%)")

    print(f"\nDistinctiveness score statistics:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  p{p:2d}: {np.percentile(all_scores, p):.4f}")

    # ── save JSON ────────────────────────────────────────────────────────────
    output = {
        "total_patches": total,
        "global_statistics": {
            "mean":   round(float(all_scores.mean()), 5),
            "std":    round(float(all_scores.std()),  5),
            "min":    round(float(all_scores.min()),  5),
            "max":    round(float(all_scores.max()),  5),
            "p5":     round(float(np.percentile(all_scores, 5)),  5),
            "p25":    round(float(np.percentile(all_scores, 25)), 5),
            "p50":    round(float(np.percentile(all_scores, 50)), 5),
            "p75":    round(float(np.percentile(all_scores, 75)), 5),
            "p95":    round(float(np.percentile(all_scores, 95)), 5),
        },
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "class_percentages": {k: round(100*v/total, 2) for k,v in class_counts.items()},
        "fragment_summary": fragment_summary,
        "patch_records": all_records,
    }

    json_path = OUT_DIR / "patch_distinctiveness.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {json_path}")

    # ── save NPZ ─────────────────────────────────────────────────────────────
    frag_ids   = np.array([r["fragment_id"] for r in all_records])
    patch_ids  = np.array([r["patch_id"]    for r in all_records], dtype=np.int32)
    scores_out = np.array([r["distinctiveness"] for r in all_records], dtype=np.float32)
    classes    = np.array([r["class"] for r in all_records])
    nvar_arr   = np.array([r["normal_variance"]    for r in all_records], dtype=np.float32)
    curv_arr   = np.array([r["curvature_mean"]     for r in all_records], dtype=np.float32)
    rough_arr  = np.array([r["roughness_mm"]       for r in all_records], dtype=np.float32)
    sct_arr    = np.array([r["scattering"]         for r in all_records], dtype=np.float32)
    pln_arr    = np.array([r["planarity"]          for r in all_records], dtype=np.float32)

    npz_path = OUT_DIR / "patch_distinctiveness.npz"
    np.savez_compressed(
        npz_path,
        fragment_ids    = frag_ids,
        patch_ids       = patch_ids,
        distinctiveness = scores_out,
        classes         = classes,
        normal_variance = nvar_arr,
        curvature_mean  = curv_arr,
        roughness_mm    = rough_arr,
        scattering      = sct_arr,
        planarity       = pln_arr,
    )
    print(f"Saved → {npz_path}")
    print("\nTask 1 COMPLETE.")


if __name__ == "__main__":
    main()
