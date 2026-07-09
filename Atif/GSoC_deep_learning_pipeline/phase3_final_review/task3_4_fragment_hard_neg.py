#!/usr/bin/env python3
"""
Tasks 3 & 4: Fragment Pair Contribution + Hard Negative Failure Analysis
Outputs:
  phase3_final_review/fragment_pair_analysis.json
  phase3_final_review/hard_negative_analysis.json
"""
import sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from patch_generation.patch_record import read_patch_records

OUT_DIR   = ROOT / "phase3_final_review"
PAIRS_DIR = ROOT / "pairs"
PATCHES_DIR = ROOT / "patches"
DISTINCT_NPZ = OUT_DIR / "patch_distinctiveness.npz"

# ── helpers ───────────────────────────────────────────────────────────────────
def load_distinct():
    d = np.load(DISTINCT_NPZ, allow_pickle=True)
    out = {}
    for fid, pid, sc, cls in zip(d["fragment_ids"], d["patch_ids"],
                                  d["distinctiveness"], d["classes"]):
        out[(str(fid), int(pid))] = (float(sc), str(cls))
    return out

def load_contact_region(pair_key):
    path = PAIRS_DIR / "contact_regions" / f"{pair_key}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {
            "contact_indices_A": d["contact_indices_A"],
            "contact_indices_B": d["contact_indices_B"],
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — Fragment Pair Contribution Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def task3_fragment_pair_contribution(distinct_lookup):
    print("=" * 60)
    print("TASK 3: FRAGMENT PAIR CONTRIBUTION ANALYSIS")
    print("=" * 60)

    with open(PAIRS_DIR / "dataset.json") as f:
        ds = json.load(f)

    adjacent_pairs = [tuple(p) for p in ds["adjacent_pair_ids"]]

    # Count positive pairs per fragment pair from pair distribution in validation
    with open(ROOT / "phase3_validation_report.json") as f:
        val = json.load(f)
    fp_dist_raw = val["audit_results"]["fragment_pair_dist"]
    # Keys look like "('fragment_caesar_fragment_6', 'fragment_caesar_fragment_7')"
    fp_dist = {}
    for k, v in fp_dist_raw.items():
        # parse tuple string
        parts = k.strip("()").replace("'","").split(", ")
        key = tuple(parts)
        fp_dist[key] = v

    results = []
    for fA, fB in adjacent_pairs:
        pair_key = f"{fA}_{fB}"
        cr = load_contact_region(pair_key)
        n_contact_A = int(len(cr["contact_indices_A"])) if cr else 0
        n_contact_B = int(len(cr["contact_indices_B"])) if cr else 0

        # Load patches for both fragments
        def load_p(fid):
            p = PATCHES_DIR / fid / "patches.npz"
            return read_patch_records(str(p)) if p.exists() else []

        patches_A = load_p(fA)
        patches_B = load_p(fB)

        # Patches with contact (>=30% overlap)
        def patches_in_contact(patches, contact_idx):
            cs = set(contact_idx) if contact_idx is not None and len(contact_idx) else set()
            out = []
            for p in patches:
                if len(p.source_indices) == 0:
                    continue
                ov = len(set(p.source_indices.tolist()) & cs) / len(p.source_indices)
                if ov >= 0.30:
                    out.append(p.patch_id)
            return out

        contact_patch_ids_A = patches_in_contact(patches_A, cr["contact_indices_A"] if cr else [])
        contact_patch_ids_B = patches_in_contact(patches_B, cr["contact_indices_B"] if cr else [])

        # Distinctiveness of contact patches
        def get_scores(fid, pids):
            return [distinct_lookup[(fid, pid)][0] for pid in pids if (fid, pid) in distinct_lookup]

        scores_A = get_scores(fA, contact_patch_ids_A)
        scores_B = get_scores(fB, contact_patch_ids_B)
        all_scores = scores_A + scores_B

        mean_dist = float(np.mean(all_scores)) if all_scores else 0.0
        def ambig_ratio(scores):
            if not scores: return 0.0
            return sum(1 for s in scores if s < 0.35) / len(scores)
        amb_ratio = ambig_ratio(all_scores)

        n_pos = fp_dist.get((fA, fB), fp_dist.get((fB, fA), 0))

        # Information content proxy: mean_distinctiveness × log(n_contact+1) / ambiguity_penalty
        info_content = mean_dist * np.log1p(n_contact_A + n_contact_B) * (1 - amb_ratio + 0.01)

        results.append({
            "fragment_A": fA.replace("fragment_caesar_", ""),
            "fragment_B": fB.replace("fragment_caesar_", ""),
            "contact_points_A": n_contact_A,
            "contact_points_B": n_contact_B,
            "contact_patches_A": len(contact_patch_ids_A),
            "contact_patches_B": len(contact_patch_ids_B),
            "positive_pairs": n_pos,
            "mean_distinctiveness": round(mean_dist, 4),
            "ambiguity_ratio": round(amb_ratio, 4),
            "information_content": round(float(info_content), 4),
        })

        print(f"  {fA[-1]}↔{fB[-1]}  pos={n_pos:>7,}  "
              f"contact_pts={n_contact_A+n_contact_B:>4}  "
              f"mean_dist={mean_dist:.3f}  amb={amb_ratio:.2f}  info={info_content:.2f}")

    results.sort(key=lambda x: -x["information_content"])
    print(f"\nRanked by information content:")
    for i, r in enumerate(results, 1):
        print(f"  #{i:2d}  {r['fragment_A']}↔{r['fragment_B']}  "
              f"info={r['information_content']:.2f}  pos={r['positive_pairs']:,}")

    total_pos = sum(r["positive_pairs"] for r in results)
    print(f"\nTotal positive pairs across all pairs: {total_pos:,}")
    dom = results[0]
    print(f"Most dominant pair: {dom['fragment_A']}↔{dom['fragment_B']} "
          f"({dom['positive_pairs']:,} pairs = "
          f"{100*dom['positive_pairs']/total_pos:.1f}% of total)")

    output = {
        "adjacent_pairs_count": len(results),
        "total_positive_pairs": total_pos,
        "pairs_ranked_by_info": results,
        "most_informative": results[0],
        "least_informative": results[-1],
        "imbalance_analysis": {
            "max_pairs": max(r["positive_pairs"] for r in results),
            "min_pairs": min(r["positive_pairs"] for r in results),
            "ratio_max_to_min": (
                round(max(r["positive_pairs"] for r in results) /
                      max(min(r["positive_pairs"] for r in results), 1), 1)
            ),
            "recommended_cap_per_pair": 50_000,
            "note": "Without a cap, training will overfit to pairs with large contact regions"
        }
    }

    out_path = OUT_DIR / "fragment_pair_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Task 3 COMPLETE.\n")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 4 — Hard Negative Failure Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def task4_hard_negative_analysis():
    print("=" * 60)
    print("TASK 4: HARD NEGATIVE FAILURE ANALYSIS")
    print("=" * 60)

    # Reproduce the exact pipeline on fragment_1 (1000 patches, mid-size)
    TARGET_FRAGMENT = "fragment_caesar_fragment_1"
    DISTANCE_THRESHOLD_MM = 16.0   # 2 × patch radius
    COUNT_REQUESTED = 100

    npz = PATCHES_DIR / TARGET_FRAGMENT / "patches.npz"
    patches = read_patch_records(str(npz))
    N = len(patches)
    print(f"Fragment: {TARGET_FRAGMENT}  ({N} patches)")

    centers = np.array([p.center for p in patches])
    tree = cKDTree(centers)

    # ── Stage 1: all candidate pairs ─────────────────────────────────────────
    # Total possible pairs
    stage1_count = N * (N - 1) // 2
    print(f"\nStage 1 — All candidate patch pairs:  {stage1_count:,}")

    # ── Stage 2: distance filter  (dist > 16 mm) ─────────────────────────────
    # Count pairs that pass distance filter
    stage2_count = 0
    dist_data = []
    for i in range(N):
        dists, _ = tree.query(centers[i], k=N)
        far = int(np.sum(dists[1:] > DISTANCE_THRESHOLD_MM))  # exclude self
        stage2_count += far
        dist_data.append(dists[1:])   # skip self
    stage2_count //= 2   # pairs counted twice

    dist_all = np.concatenate(dist_data)
    pct_far = 100 * np.sum(dist_all > DISTANCE_THRESHOLD_MM) / len(dist_all)
    print(f"Stage 2 — After distance > {DISTANCE_THRESHOLD_MM}mm filter: {stage2_count:,}  ({pct_far:.1f}% of directed edges)")

    # ── Stage 3: zero-overlap filter ─────────────────────────────────────────
    # Sample 500 far pairs and check source-index overlap
    rng = np.random.default_rng(0)
    stage3_pass = 0
    stage3_trials = 0
    overlap_fractions = []

    for _ in range(2000):  # sample up to 2000 candidates
        i = int(rng.integers(N))
        dists, idxs = tree.query(centers[i], k=N)
        far_mask = dists[1:] > DISTANCE_THRESHOLD_MM
        far_idxs = idxs[1:][far_mask]
        if len(far_idxs) == 0:
            continue
        j = int(rng.choice(far_idxs))
        stage3_trials += 1
        ov = len(set(patches[i].source_indices.tolist()) &
                 set(patches[j].source_indices.tolist()))
        overlap_fractions.append(ov / max(len(patches[i].source_indices), 1))
        if ov == 0:
            stage3_pass += 1

    if stage3_trials > 0:
        zero_ov_rate = stage3_pass / stage3_trials
        stage3_estimate = int(stage2_count * zero_ov_rate)
        print(f"Stage 3 — After zero source-overlap filter: ~{stage3_estimate:,}  "
              f"({100*zero_ov_rate:.1f}% pass)")
        print(f"  Mean overlap fraction: {np.mean(overlap_fractions):.3f}")
    else:
        zero_ov_rate = 0.0
        stage3_estimate = 0
        print("Stage 3 — No far pairs found during sampling!")

    # ── diagnosis ─────────────────────────────────────────────────────────────
    print(f"\n── ROOT CAUSE DIAGNOSIS ──")

    # Fragment extent vs threshold
    extents = centers.max(axis=0) - centers.min(axis=0)
    max_extent = float(extents.max())
    print(f"  Fragment extent: {extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} mm  (max={max_extent:.1f}mm)")
    print(f"  Hard-neg threshold: {DISTANCE_THRESHOLD_MM}mm  = {100*DISTANCE_THRESHOLD_MM/max_extent:.0f}% of max extent")

    # How many patches are within 16mm of any given patch?
    k_near = min(N, 50)
    dists_k, _ = tree.query(centers, k=k_near)
    near_counts = np.sum(dists_k[:, 1:] <= DISTANCE_THRESHOLD_MM, axis=1)
    print(f"  Mean patches within 16mm of any patch: {near_counts.mean():.0f}  "
          f"(= {100*near_counts.mean()/(N-1):.0f}% of all patches)")

    causes = []
    if max_extent < 3 * DISTANCE_THRESHOLD_MM:
        causes.append("FRAGMENT_TOO_SMALL: max extent < 3× threshold → few truly far pairs exist")
    if near_counts.mean() / (N - 1) > 0.6:
        causes.append("HIGH_PATCH_DENSITY: most patches are close together → distance filter eliminates most")
    if zero_ov_rate < 0.3:
        causes.append("HIGH_OVERLAP: far patches still share source indices due to FPS redundancy")

    for c in causes:
        print(f"  ❌ {c}")

    # ── alternative strategies ────────────────────────────────────────────────
    print(f"\n── ALTERNATIVE STRATEGIES ──")

    alts = [
        {
            "name": "Relaxed same-fragment (distance 10mm, allow 10% overlap)",
            "feasibility": "HIGH",
            "rationale": "Lowering threshold to 10mm (1.25× radius) significantly increases candidates. "
                         "Allowing 10% source-index overlap retains distinctiveness while being achievable.",
            "risk": "Partial overlap means some patch geometry is shared — mild label noise.",
            "recommended": True,
        },
        {
            "name": "Cross-fragment wrong-zone (adjacent fragment, non-contact patches)",
            "feasibility": "HIGH",
            "rationale": "Take patch from contact zone of fragment A, pair with non-contact patch of "
                         "adjacent fragment B. These are hard because they come from adjacent fragments "
                         "but are geometrically misaligned.",
            "risk": "Requires contact-zone labels to be accurate (which they are).",
            "recommended": True,
        },
        {
            "name": "Curvature-similar patches from non-adjacent fragments",
            "feasibility": "MEDIUM",
            "rationale": "Find patches with similar normal_variance/curvature across non-adjacent "
                         "fragments. These confuse the network because they look similar but never match.",
            "risk": "Requires computing per-patch similarity without deep features (use geometry stats).",
            "recommended": True,
        },
        {
            "name": "FPFH-similar patches",
            "feasibility": "LOW (requires Phase 4)",
            "rationale": "Best hard negatives but requires FPFH descriptors from Phase 4 first.",
            "risk": "Circular dependency on Phase 4.",
            "recommended": False,
        },
        {
            "name": "Symmetry-based (left/right face regions)",
            "feasibility": "LOW",
            "rationale": "Caesar bust may have near-symmetric regions. These are confusing.",
            "risk": "Requires detecting symmetric regions — non-trivial.",
            "recommended": False,
        },
    ]

    for a in alts:
        flag = "✅" if a["recommended"] else "⬜"
        print(f"  {flag} {a['name']}  [{a['feasibility']}]")

    output = {
        "target_fragment": TARGET_FRAGMENT,
        "n_patches": N,
        "pipeline_stages": {
            "stage1_all_pairs": stage1_count,
            "stage2_after_distance_filter": stage2_count,
            "stage3_after_overlap_filter_estimate": stage3_estimate,
            "final_hard_negatives": 0,
        },
        "diagnosis": {
            "fragment_max_extent_mm": round(max_extent, 1),
            "distance_threshold_mm": DISTANCE_THRESHOLD_MM,
            "threshold_pct_of_extent": round(100*DISTANCE_THRESHOLD_MM/max_extent, 1),
            "mean_patches_within_threshold": round(float(near_counts.mean()), 1),
            "pct_close_pairs": round(float(100*near_counts.mean()/(N-1)), 1),
            "mean_overlap_in_far_pairs": round(float(np.mean(overlap_fractions)), 4),
            "root_causes": causes,
        },
        "alternative_strategies": alts,
        "recommendation": "Use strategy 1 (relaxed same-fragment) + strategy 2 (cross-fragment wrong-zone). "
                          "These are achievable without Phase 4 and produce genuinely hard negatives.",
    }

    out_path = OUT_DIR / "hard_negative_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Task 4 COMPLETE.")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    distinct_lookup = load_distinct()
    task3_fragment_pair_contribution(distinct_lookup)
    task4_hard_negative_analysis()
