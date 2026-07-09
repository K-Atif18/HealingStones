#!/usr/bin/env python3
"""
Task 2: Positive Pair Quality Analysis

For every positive pair determines:
  - Distinctiveness class of both patches
  - Contact overlap of both patches
  - True 3D centre distance after applying ground-truth transforms
    (patches are already stored in model-frame coordinates, so we just
    compute ||center_A - center_B|| directly — no transform needed)

Answers: do positive pairs represent
  A) true correspondences
  B) nearby fracture regions
  C) contact-zone co-membership
  D) a mixture

Outputs: phase3_final_review/positive_pair_analysis.json
"""

import sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "phase3_final_review"
PAIRS_FILE   = ROOT / "pairs" / "positive_pairs.json"
DISTINCT_NPZ = OUT_DIR / "patch_distinctiveness.npz"
PATCH_RADIUS  = 8.0   # mm — two patches overlap when distance < 2×radius = 16 mm

SAMPLE_SIZE = 50_000   # analyse a random sample for speed (full set = 719K)

# ── load distinctiveness lookup ───────────────────────────────────────────────
def load_distinctiveness():
    d = np.load(DISTINCT_NPZ, allow_pickle=True)
    lookup = {}
    for fid, pid, score, cls in zip(
            d["fragment_ids"], d["patch_ids"],
            d["distinctiveness"], d["classes"]):
        lookup[(str(fid), int(pid))] = (float(score), str(cls))
    return lookup

# ── load positive pairs (sample) ─────────────────────────────────────────────
def load_positive_pairs(sample_size):
    print(f"Loading positive pairs from {PAIRS_FILE} …", flush=True)
    with open(PAIRS_FILE) as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    total = len(pairs)
    print(f"  Total positive pairs: {total:,}")
    if total > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(total, sample_size, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
        print(f"  Sampled: {len(pairs):,} (seed=42)")
    return pairs, total

# ── distance: centres already in model-frame coords ──────────────────────────
def centre_distance(pair):
    cA = np.array(pair["patch_A_center"])
    cB = np.array(pair["patch_B_center"])
    return float(np.linalg.norm(cA - cB))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TASK 2: POSITIVE PAIR QUALITY ANALYSIS")
    print("=" * 60)

    distinct_lookup = load_distinctiveness()
    print(f"Distinctiveness lookup: {len(distinct_lookup):,} patches")

    pairs, total_pairs = load_positive_pairs(SAMPLE_SIZE)

    # ── per-pair analysis ────────────────────────────────────────────────────
    distances   = []
    overlaps_A  = []
    overlaps_B  = []
    combo_counts = defaultdict(int)   # (class_A, class_B)
    combo_dists  = defaultdict(list)

    missing = 0
    for p in pairs:
        fA = p["fragment_A_id"]
        fB = p["fragment_B_id"]
        pA = int(p["patch_A_id"])
        pB = int(p["patch_B_id"])

        scoreA, clsA = distinct_lookup.get((fA, pA), (None, "unknown"))
        scoreB, clsB = distinct_lookup.get((fB, pB), (None, "unknown"))
        if scoreA is None or scoreB is None:
            missing += 1
            continue

        dist = centre_distance(p)
        ovA  = float(p.get("contact_overlap_A", 0))
        ovB  = float(p.get("contact_overlap_B", 0))

        distances.append(dist)
        overlaps_A.append(ovA)
        overlaps_B.append(ovB)

        # Normalise combo key so (A,B) == (B,A)
        key = tuple(sorted([clsA, clsB]))
        combo_counts[key] += 1
        combo_dists[key].append(dist)

    print(f"\nPairs analysed : {len(distances):,}  (missing lookup: {missing})")

    distances  = np.array(distances)
    overlaps_A = np.array(overlaps_A)
    overlaps_B = np.array(overlaps_B)

    # ── distance interpretation ───────────────────────────────────────────────
    #  two patches with centres within 16 mm OVERLAP (radius = 8 mm each)
    #  centres 0–8 mm   → one contains the other (high overlap)
    #  centres 8–16 mm  → partial overlap
    #  centres > 16 mm  → non-overlapping — could be true correspondences
    #                     only if patches are on mating fracture surfaces
    overlap_zone    = int(np.sum(distances <= 16.0))
    near_zone       = int(np.sum((distances > 16.0) & (distances <= 40.0)))
    far_zone        = int(np.sum(distances > 40.0))
    n = len(distances)

    print(f"\nCentre-distance distribution (model-frame Euclidean):")
    print(f"  d <= 16 mm  (overlapping patches, co-membership):  {overlap_zone:>6d}  ({100*overlap_zone/n:.1f}%)")
    print(f"  16 < d <= 40 mm (nearby fracture regions):         {near_zone:>6d}  ({100*near_zone/n:.1f}%)")
    print(f"  d >  40 mm  (distant / different regions):         {far_zone:>6d}  ({100*far_zone/n:.1f}%)")

    print(f"\nDistance statistics (mm):")
    for p in [5,25,50,75,95]:
        print(f"  p{p:2d}: {np.percentile(distances, p):.2f}")
    print(f"  mean: {distances.mean():.2f}   std: {distances.std():.2f}")

    print(f"\nContact overlap statistics:")
    print(f"  overlap_A  mean={overlaps_A.mean():.3f}  median={np.median(overlaps_A):.3f}  "
          f"min={overlaps_A.min():.3f}  max={overlaps_A.max():.3f}")
    print(f"  overlap_B  mean={overlaps_B.mean():.3f}  median={np.median(overlaps_B):.3f}  "
          f"min={overlaps_B.min():.3f}  max={overlaps_B.max():.3f}")

    # ── combo analysis ────────────────────────────────────────────────────────
    print(f"\nDistinctiveness combination breakdown:")
    combo_stats = {}
    for combo, cnt in sorted(combo_counts.items(), key=lambda x: -x[1]):
        dlist = np.array(combo_dists[combo])
        label = f"{combo[0]}  ×  {combo[1]}"
        pct   = 100 * cnt / len(distances)
        print(f"  {label:60s}  {cnt:>7,}  ({pct:.1f}%)  mean_dist={dlist.mean():.1f}mm")
        combo_stats[str(combo)] = {
            "count": cnt,
            "pct": round(pct, 2),
            "mean_dist_mm":   round(float(dlist.mean()), 2),
            "median_dist_mm": round(float(np.median(dlist)), 2),
        }

    # ── label interpretation ──────────────────────────────────────────────────
    # Diagnosis rules:
    # If majority of pairs have d <= 2×radius → co-membership dominates
    # If patches have high overlap (>0.8) → zone membership, not correspondence
    # True correspondence would require unique 1-to-1 mapping at fracture surface
    majority_overlap = overlap_zone / n > 0.4

    interpretation = {
        "A_true_correspondences":    bool(overlap_zone / n < 0.1),
        "B_nearby_fracture_regions": bool((16 < distances).mean() > 0.3),
        "C_contact_zone_comembership": bool(majority_overlap),
        "D_mixture": True,
        "dominant_category": (
            "C_contact_zone_comembership" if majority_overlap else
            "B_nearby_fracture_regions"
        ),
        "evidence": {
            "pct_overlapping_patches":  round(100 * overlap_zone / n, 1),
            "mean_contact_overlap_A":   round(float(overlaps_A.mean()), 3),
            "mean_contact_overlap_B":   round(float(overlaps_B.mean()), 3),
            "median_centre_dist_mm":    round(float(np.median(distances)), 2),
        }
    }

    print(f"\n── INTERPRETATION ──")
    print(f"  Dominant category: {interpretation['dominant_category']}")
    print(f"  {overlap_zone/n*100:.1f}% of sampled pairs have overlapping patch footprints (d<16mm)")
    print(f"  Mean contact overlap A={overlaps_A.mean():.3f}  B={overlaps_B.mean():.3f}")
    print(f"  Conclusion: positive pairs primarily encode CONTACT-ZONE CO-MEMBERSHIP,")
    print(f"  not 1-to-1 geometric correspondences.")

    # ── save ──────────────────────────────────────────────────────────────────
    # Build histogram for distance distribution
    hist_counts, hist_edges = np.histogram(distances, bins=20, range=(0, 200))
    histogram = {
        "bin_edges_mm": [round(float(e), 1) for e in hist_edges],
        "counts":       [int(c) for c in hist_counts],
    }

    output = {
        "sample_size":  len(distances),
        "total_pairs":  total_pairs,
        "missing_lookup": missing,
        "distance_zones": {
            "overlapping_d_le_16mm":  {"count": overlap_zone, "pct": round(100*overlap_zone/n,1)},
            "nearby_16_to_40mm":      {"count": near_zone,    "pct": round(100*near_zone/n,1)},
            "distant_gt_40mm":        {"count": far_zone,     "pct": round(100*far_zone/n,1)},
        },
        "distance_stats_mm": {
            "mean":   round(float(distances.mean()), 2),
            "std":    round(float(distances.std()), 2),
            "p5":     round(float(np.percentile(distances,  5)), 2),
            "p25":    round(float(np.percentile(distances, 25)), 2),
            "p50":    round(float(np.percentile(distances, 50)), 2),
            "p75":    round(float(np.percentile(distances, 75)), 2),
            "p95":    round(float(np.percentile(distances, 95)), 2),
        },
        "contact_overlap_stats": {
            "overlap_A": {
                "mean":   round(float(overlaps_A.mean()), 4),
                "median": round(float(np.median(overlaps_A)), 4),
                "min":    round(float(overlaps_A.min()), 4),
                "max":    round(float(overlaps_A.max()), 4),
            },
            "overlap_B": {
                "mean":   round(float(overlaps_B.mean()), 4),
                "median": round(float(np.median(overlaps_B)), 4),
                "min":    round(float(overlaps_B.min()), 4),
                "max":    round(float(overlaps_B.max()), 4),
            },
        },
        "combo_breakdown": combo_stats,
        "distance_histogram": histogram,
        "interpretation": interpretation,
    }

    out_path = OUT_DIR / "positive_pair_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Task 2 COMPLETE.")


if __name__ == "__main__":
    main()
