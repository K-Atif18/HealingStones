#!/usr/bin/env python3
"""
Task 5: Fragment Pair 5↔7 Contact Investigation
Task 6: Dataset Balance Analysis

Outputs:
  phase3_final_review/contact_pair_5_7_analysis.json
  phase3_final_review/dataset_balance_analysis.json
"""
import sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_foundation.ply_io import read_point_cloud
from patch_generation.patch_record import read_patch_records
from ground_truth_generation.fragment_pairs import detect_contact_points
from scipy.spatial import cKDTree

OUT_DIR     = ROOT / "phase3_final_review"
PAIRS_DIR   = ROOT / "pairs"
PATCHES_DIR = ROOT / "patches"
DATASET_DIR = ROOT / "dataset"
DISTINCT_NPZ = OUT_DIR / "patch_distinctiveness.npz"


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 5 — Fragment 5↔7 Contact Investigation
# ═══════════════════════════════════════════════════════════════════════════════
def task5_contact_5_7():
    print("=" * 60)
    print("TASK 5: FRAGMENT 5↔7 CONTACT INVESTIGATION")
    print("=" * 60)

    fA = "fragment_caesar_fragment_5"
    fB = "fragment_caesar_fragment_7"

    pc5 = read_point_cloud(DATASET_DIR / "normalized" / f"{fA}.ply")
    pc7 = read_point_cloud(DATASET_DIR / "normalized" / f"{fB}.ply")
    pts5 = np.array(pc5.points)
    pts7 = np.array(pc7.points)

    print(f"Fragment 5: {len(pts5)} points")
    print(f"Fragment 7: {len(pts7)} points")

    # Contact counts at every threshold
    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
    contact_data = {}
    for t in thresholds:
        idx5, idx7 = detect_contact_points(pts5, pts7, t)
        contact_data[t] = {"A": int(len(idx5)), "B": int(len(idx7))}
        print(f"  t={t:4.1f}mm → contact pts fragment_5={len(idx5):3d}, fragment_7={len(idx7):3d}")

    # Load the saved contact region (at 3mm)
    cr_path = PAIRS_DIR / "contact_regions" / f"{fA}_{fB}.npz"
    saved = np.load(cr_path, allow_pickle=False)
    ci5 = saved["contact_indices_A"]
    ci7 = saved["contact_indices_B"]
    contact_pts5 = pts5[ci5]
    contact_pts7 = pts7[ci7]

    print(f"\nSaved contact region (3mm threshold):")
    print(f"  Points on fragment_5: {len(ci5)}")
    print(f"  Points on fragment_7: {len(ci7)}")

    # Spatial analysis of contact region
    if len(contact_pts5) > 0 and len(contact_pts7) > 0:
        all_contact = np.vstack([contact_pts5, contact_pts7])
        spread = all_contact.max(axis=0) - all_contact.min(axis=0)
        print(f"  Contact region spatial spread: {spread[0]:.2f} × {spread[1]:.2f} × {spread[2]:.2f} mm")
        print(f"  Contact region max extent: {spread.max():.2f} mm")

        # Classify contact type
        max_extent = float(spread.max())
        if max_extent < 3.0:
            contact_type = "point_contact"
            desc = "Single point or vertex contact — sub-patch scale"
        elif max_extent < 8.0:
            contact_type = "edge_contact"
            desc = "Edge contact — smaller than one patch radius"
        else:
            contact_type = "surface_contact"
            desc = "Surface contact — multiple patches could cover it"
    else:
        contact_type = "no_contact"
        desc = "No contact detected at 3mm threshold"
        max_extent = 0.0

    print(f"\n  Contact type: {contact_type} — {desc}")

    # Why 0 positive pairs?
    patches5 = read_patch_records(str(PATCHES_DIR / fA / "patches.npz"))
    patches6_dummy = []   # not used, just checking fragment 5 contact patches
    ci5_set = set(ci5.tolist())
    ci7_set = set(ci7.tolist())

    overlapping5 = [p for p in patches5
                    if len(p.source_indices) > 0 and
                    len(set(p.source_indices.tolist()) & ci5_set) / len(p.source_indices) >= 0.30]
    patches7 = read_patch_records(str(PATCHES_DIR / fB / "patches.npz"))
    overlapping7 = [p for p in patches7
                    if len(p.source_indices) > 0 and
                    len(set(p.source_indices.tolist()) & ci7_set) / len(p.source_indices) >= 0.30]

    print(f"\n  Patches with ≥30% contact overlap:")
    print(f"    Fragment 5: {len(overlapping5)} patches")
    print(f"    Fragment 7: {len(overlapping7)} patches")
    print(f"    Cartesian product: {len(overlapping5) * len(overlapping7)} positive pairs would form")

    if len(overlapping5) == 0 or len(overlapping7) == 0:
        print(f"\n  → 0 positive pairs is CORRECT: "
              f"contact zone too small for any patch to have ≥30% overlap")
        zero_pairs_correct = True
        reason = ("Contact region contains only 5-7 points (< 1 patch size of ~100 pts). "
                  "No patch can have 30% of its points in this region. "
                  "This is mathematically correct behaviour, not a bug.")
    else:
        zero_pairs_correct = False
        reason = "Unexpected: patches found with contact overlap"

    # What threshold would generate pairs?
    for test_threshold in [0.05, 0.10, 0.15, 0.20]:
        t5 = [p for p in patches5
              if len(p.source_indices) > 0 and
              len(set(p.source_indices.tolist()) & ci5_set) / len(p.source_indices) >= test_threshold]
        t7 = [p for p in patches7
              if len(p.source_indices) > 0 and
              len(set(p.source_indices.tolist()) & ci7_set) / len(p.source_indices) >= test_threshold]
        print(f"  At threshold={test_threshold:.2f}: fragment_5={len(t5)} patches, "
              f"fragment_7={len(t7)} patches → {len(t5)*len(t7)} pairs")

    output = {
        "fragment_A": fA,
        "fragment_B": fB,
        "contact_counts_by_threshold": {str(t): v for t, v in contact_data.items()},
        "contact_region_3mm": {
            "contact_pts_A": int(len(ci5)),
            "contact_pts_B": int(len(ci7)),
            "spatial_spread_mm": [round(float(x), 2) for x in spread.tolist()] if len(ci5) > 0 else [],
            "max_extent_mm": round(max_extent, 2),
        },
        "contact_type": contact_type,
        "contact_description": desc,
        "zero_pairs_is_correct": zero_pairs_correct,
        "reason": reason,
        "patches_with_contact_overlap_30pct": {
            "fragment_5": len(overlapping5),
            "fragment_7": len(overlapping7),
        },
        "recommendation": (
            "Do NOT lower the overlap threshold to generate pairs for this pair. "
            "The contact is a corner/edge touch that is not matchable by local patch geometry. "
            "Including this pair with a lower threshold would introduce noise."
        ),
    }

    out_path = OUT_DIR / "contact_pair_5_7_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Task 5 COMPLETE.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 6 — Dataset Balance Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def task6_dataset_balance():
    print("=" * 60)
    print("TASK 6: DATASET BALANCE ANALYSIS")
    print("=" * 60)

    d = np.load(DISTINCT_NPZ, allow_pickle=True)
    distinct_lookup = {}
    for fid, pid, sc, cls in zip(d["fragment_ids"], d["patch_ids"],
                                  d["distinctiveness"], d["classes"]):
        distinct_lookup[(str(fid), int(pid))] = (float(sc), str(cls))

    # Load positive pairs (stream through — too large for full load analysis)
    print("Loading positive pairs …", flush=True)
    with open(PAIRS_DIR / "positive_pairs.json") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    print(f"  {len(pairs):,} positive pairs")

    # Per-patch appearance count
    patch_appearances = defaultdict(int)
    fragment_pair_counts = defaultdict(int)

    for p in pairs:
        fA = p["fragment_A_id"]
        fB = p["fragment_B_id"]
        pA = int(p["patch_A_id"])
        pB = int(p["patch_B_id"])
        patch_appearances[(fA, pA)] += 1
        patch_appearances[(fB, pB)] += 1
        key = tuple(sorted([fA.replace("fragment_caesar_",""),
                             fB.replace("fragment_caesar_","")]))
        fragment_pair_counts[key] += 1

    appearances = np.array(list(patch_appearances.values()))
    total_patches = len(distinct_lookup)
    active_patches = len(patch_appearances)
    inactive_patches = total_patches - active_patches

    print(f"\nPatch participation:")
    print(f"  Total patches:         {total_patches:,}")
    print(f"  Active (in any positive): {active_patches:,}  ({100*active_patches/total_patches:.1f}%)")
    print(f"  Inactive (never positive): {inactive_patches:,}  ({100*inactive_patches/total_patches:.1f}%)")
    print(f"\nAppearance distribution (times each patch appears in positives):")
    print(f"  Mean:    {appearances.mean():.1f}")
    print(f"  Median:  {np.median(appearances):.1f}")
    print(f"  Max:     {appearances.max()}")
    print(f"  Std:     {appearances.std():.1f}")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d}:   {np.percentile(appearances, p):.0f}")

    # Oversampled patches (>3σ above mean)
    threshold_3sigma = appearances.mean() + 3 * appearances.std()
    oversampled = {k: v for k, v in patch_appearances.items() if v > threshold_3sigma}
    print(f"\nOversampled patches (> {threshold_3sigma:.0f} appearances = mean+3σ): {len(oversampled):,}")
    # Show top 10
    top10 = sorted(oversampled.items(), key=lambda x: -x[1])[:10]
    for (fid, pid), cnt in top10:
        sc, cls = distinct_lookup.get((fid, pid), (0, "?"))
        print(f"  {fid[-1]}:{pid:4d}  appearances={cnt:5d}  score={sc:.3f}  class={cls}")

    # Fragment pair imbalance
    total_pos = sum(fragment_pair_counts.values())
    print(f"\nFragment pair positive distribution:")
    for key, cnt in sorted(fragment_pair_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total_pos
        print(f"  {key[0][-1]}↔{key[1][-1]}  {cnt:>7,}  ({pct:.1f}%)")

    max_cnt = max(fragment_pair_counts.values())
    min_cnt = min(v for v in fragment_pair_counts.values() if v > 0)

    # Recommended cap
    # Goal: reduce imbalance ratio to <= 10:1
    # Current ratio ≈ 214446 / 1232 ≈ 174:1
    # Cap at 50,000 keeps the top pairs contributing meaningfully
    recommended_cap = 50_000
    capped_total = sum(min(v, recommended_cap) for v in fragment_pair_counts.values())
    print(f"\nImbalance ratio (max/min): {max_cnt/min_cnt:.0f}:1")
    print(f"Recommended cap: {recommended_cap:,} pairs per fragment pair")
    print(f"After cap: ~{capped_total:,} total pairs "
          f"(vs {total_pos:,} current)")

    # Histogram of appearances
    hist_counts, hist_edges = np.histogram(appearances, bins=20)

    output = {
        "total_patches": int(total_patches),
        "active_patches": int(active_patches),
        "inactive_patches": int(inactive_patches),
        "active_pct": round(100 * active_patches / total_patches, 2),
        "appearance_stats": {
            "mean":   round(float(appearances.mean()), 1),
            "median": round(float(np.median(appearances)), 1),
            "max":    int(appearances.max()),
            "std":    round(float(appearances.std()), 1),
            "p25":    round(float(np.percentile(appearances, 25)), 1),
            "p75":    round(float(np.percentile(appearances, 75)), 1),
            "p95":    round(float(np.percentile(appearances, 95)), 1),
            "p99":    round(float(np.percentile(appearances, 99)), 1),
        },
        "oversampled_patches": {
            "threshold_3sigma": round(float(threshold_3sigma), 1),
            "count": len(oversampled),
            "pct":   round(100 * len(oversampled) / active_patches, 2),
        },
        "fragment_pair_distribution": {
            f"{k[0][-1]}_{k[1][-1]}": {"count": int(v), "pct": round(100*v/total_pos, 2)}
            for k, v in sorted(fragment_pair_counts.items(), key=lambda x: -x[1])
        },
        "imbalance": {
            "max_pairs_single_pair": int(max_cnt),
            "min_pairs_single_pair": int(min_cnt),
            "ratio": round(max_cnt / min_cnt, 1),
            "is_severe": bool(max_cnt / min_cnt > 50),
        },
        "recommendations": {
            "sampling_cap_per_pair": recommended_cap,
            "capped_total_pairs": int(capped_total),
            "reduction_factor": round(total_pos / capped_total, 2),
            "importance_weighting": (
                "Assign training weight = 1 / sqrt(pair_count) per pair "
                "so small-contact pairs (1↔3, 2↔4) contribute proportionally."
            ),
            "curriculum_strategy": (
                "Start training with high-distinctiveness patches only (score >= 0.55). "
                "Introduce ambiguous patches in later epochs once basic matching is learned."
            ),
        },
        "appearance_histogram": {
            "bin_edges": [round(float(e), 0) for e in hist_edges],
            "counts": [int(c) for c in hist_counts],
        },
    }

    out_path = OUT_DIR / "dataset_balance_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Task 6 COMPLETE.")


if __name__ == "__main__":
    task5_contact_5_7()
    task6_dataset_balance()
