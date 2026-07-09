#!/usr/bin/env python3
"""
Phase 3 Interactive Visualizer

Lets you manually inspect:
  1. All contact pairs (both fragments side-by-side, contact region highlighted)
  2. Positive patch pairs from a selected fragment pair
  3. Random negative patch pairs from non-adjacent fragments

Usage:
    # Browse all 11 contact pairs one by one
    PYTHONPATH=src python3 scripts/visualize_phase3.py --mode contacts

    # View 5 random positive pairs for fragment pair 6<->7
    PYTHONPATH=src python3 scripts/visualize_phase3.py --mode positives --pair 6 7 --count 5

    # View a specific positive pair by index
    PYTHONPATH=src python3 scripts/visualize_phase3.py --mode positives --pair 1 4 --index 0

    # View 5 random negative pairs for fragment pair 1<->6
    PYTHONPATH=src python3 scripts/visualize_phase3.py --mode negatives --pair 1 6 --count 5

Adjacent pairs (positives exist):
  1-2  1-3  1-4  1-5  2-3  2-4  3-4  4-5  5-6  5-7  6-7

Non-adjacent pairs (negatives only):
  1-6  1-7  2-5  2-6  2-7  3-5  3-6  3-7  4-6  4-7
"""

import sys
import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset_foundation.ply_io import read_point_cloud
from patch_generation.patch_record import read_patch_records

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "normalized"
PATCHES_DIR = ROOT / "patches"
PAIRS_DIR   = ROOT / "pairs"
CR_DIR      = PAIRS_DIR / "contact_regions"

# ── colours ───────────────────────────────────────────────────────────────────
C_GRAY   = [0.75, 0.75, 0.75]
C_RED    = [0.95, 0.20, 0.20]
C_BLUE   = [0.15, 0.45, 0.90]
C_GREEN  = [0.10, 0.75, 0.30]
C_ORANGE = [1.00, 0.55, 0.10]
C_PURPLE = [0.70, 0.20, 0.85]
C_CYAN   = [0.10, 0.80, 0.80]
C_YELLOW = [0.95, 0.85, 0.10]

ADJACENT_PAIRS = [
    (1,2),(1,3),(1,4),(1,5),
    (2,3),(2,4),
    (3,4),(4,5),
    (5,6),(5,7),(6,7),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def fid(n):
    return f"fragment_caesar_fragment_{n}"

def load_fragment(n):
    pc = read_point_cloud(str(DATASET_DIR / f"{fid(n)}.ply"))
    return np.array(pc.points)

def load_patches(n):
    return read_patch_records(str(PATCHES_DIR / fid(n) / "patches.npz"))

def load_contact_region(a, b):
    for key in [f"{fid(a)}_{fid(b)}", f"{fid(b)}_{fid(a)}"]:
        p = CR_DIR / f"{key}.npz"
        if p.exists():
            d = np.load(p, allow_pickle=False)
            if key.startswith(fid(a)):
                return d["contact_indices_A"], d["contact_indices_B"]
            else:
                return d["contact_indices_B"], d["contact_indices_A"]
    return np.array([]), np.array([])

def show(geoms, title):
    o3d.visualization.draw_geometries(
        geoms, window_name=title,
        width=1400, height=900
    )

def make_sphere(center, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(center)
    s.paint_uniform_color(color)
    return s

def make_line(p1, p2, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([p1, p2])
    ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls

def colored_pcd(pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def side_by_side_offset(pts_a):
    extent = pts_a.max(axis=0) - pts_a.min(axis=0)
    return np.array([extent[0] + 20.0, 0.0, 0.0])

# ── load pairs (cached after first load) ──────────────────────────────────────
_pos_cache = None
_neg_cache = None

def get_positive_pairs(a, b):
    global _pos_cache
    if _pos_cache is None:
        print("  Loading positive_pairs.json …", flush=True)
        with open(PAIRS_DIR / "positive_pairs.json") as f:
            _pos_cache = json.load(f)["pairs"]
        print(f"  Loaded {len(_pos_cache):,} positive pairs.")
    fA, fB = fid(a), fid(b)
    out = []
    for p in _pos_cache:
        if p["fragment_A_id"] == fA and p["fragment_B_id"] == fB:
            out.append(p)
        elif p["fragment_A_id"] == fB and p["fragment_B_id"] == fA:
            # flip so A is always the lower-numbered fragment
            out.append({**p,
                "fragment_A_id": fA, "fragment_B_id": fB,
                "patch_A_id": p["patch_B_id"], "patch_B_id": p["patch_A_id"],
                "patch_A_center": p["patch_B_center"],
                "patch_B_center": p["patch_A_center"],
                "contact_overlap_A": p["contact_overlap_B"],
                "contact_overlap_B": p["contact_overlap_A"],
            })
    return out

def get_negative_pairs(a, b):
    global _neg_cache
    if _neg_cache is None:
        print("  Loading random_negative_pairs.json …", flush=True)
        with open(PAIRS_DIR / "random_negative_pairs.json") as f:
            _neg_cache = json.load(f)["pairs"]
        print(f"  Loaded {len(_neg_cache):,} negative pairs.")
    fA, fB = fid(a), fid(b)
    return [p for p in _neg_cache
            if (p["fragment_A_id"] == fA and p["fragment_B_id"] == fB) or
               (p["fragment_A_id"] == fB and p["fragment_B_id"] == fA)]


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1 — Contact pairs browser
# ══════════════════════════════════════════════════════════════════════════════
def mode_contacts():
    print("\n── CONTACT PAIRS BROWSER ──")
    print("Shows all 11 adjacent fragment pairs.")
    print("RED = contact region  |  GRAY = rest of fragment")
    print("YELLOW line = centroid-to-centroid connection")
    print("Close each window to advance.\n")

    for i, (a, b) in enumerate(ADJACENT_PAIRS):
        pts_a = load_fragment(a)
        pts_b = load_fragment(b)
        ci_a, ci_b = load_contact_region(a, b)

        offset = side_by_side_offset(pts_a)
        pts_b_shifted = pts_b + offset

        # Colour arrays
        col_a = np.tile(C_GRAY, (len(pts_a), 1))
        col_b = np.tile(C_GRAY, (len(pts_b), 1))
        if len(ci_a): col_a[ci_a] = C_RED
        if len(ci_b): col_b[ci_b] = C_RED

        geoms = [colored_pcd(pts_a, col_a),
                 colored_pcd(pts_b_shifted, col_b)]

        # Centroid connector
        if len(ci_a) and len(ci_b):
            cen_a = pts_a[ci_a].mean(axis=0)
            cen_b = pts_b_shifted[ci_b].mean(axis=0)
            geoms.append(make_line(cen_a, cen_b, C_YELLOW))
            geoms.append(make_sphere(cen_a, 1.5, C_YELLOW))
            geoms.append(make_sphere(cen_b, 1.5, C_YELLOW))

        is_edge = (a, b) == (5, 7)
        tag = " ⚠ EDGE CONTACT" if is_edge else ""

        print(f"  [{i+1}/11] Fragment {a} ↔ {b}{tag}  "
              f"contact pts: A={len(ci_a)}  B={len(ci_b)}")

        show(geoms,
             f"Contact Pair {i+1}/11: Fragment {a} ↔ {b}{tag}  "
             f"| contact A={len(ci_a)} B={len(ci_b)} pts")

    print("\nAll contact pairs viewed.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — Positive pairs viewer
# ══════════════════════════════════════════════════════════════════════════════
def mode_positives(a, b, count, pair_index):
    print(f"\n── POSITIVE PAIRS: Fragment {a} ↔ Fragment {b} ──")

    pairs = get_positive_pairs(a, b)
    if not pairs:
        print(f"No positive pairs found for {a}↔{b}. "
              f"Is this an adjacent pair? Adjacent: {ADJACENT_PAIRS}")
        return
    print(f"  {len(pairs):,} positive pairs available.")

    rng = np.random.default_rng(42)
    if pair_index is not None:
        selected = [pairs[pair_index % len(pairs)]]
    else:
        idx = rng.choice(len(pairs), min(count, len(pairs)), replace=False)
        selected = [pairs[i] for i in sorted(idx)]

    pts_a  = load_fragment(a)
    pts_b  = load_fragment(b)
    ci_a, ci_b = load_contact_region(a, b)
    ci_a_set = set(ci_a.tolist())
    ci_b_set = set(ci_b.tolist())

    pat_a = {p.patch_id: p for p in load_patches(a)}
    pat_b = {p.patch_id: p for p in load_patches(b)}

    offset = side_by_side_offset(pts_a)
    pts_b_shifted = pts_b + offset

    print("\nColour key:")
    print("  BLUE   = patch A points outside contact zone")
    print("  ORANGE = patch A points inside contact zone (the match evidence)")
    print("  GREEN  = patch B points outside contact zone")
    print("  ORANGE = patch B points inside contact zone")
    print("  RED    = contact zone (not in this patch)")
    print("  GRAY   = rest of fragment")
    print("  YELLOW line = patch centre connection  |  Close window for next.\n")

    for i, pair in enumerate(selected):
        pa_id = int(pair["patch_A_id"])
        pb_id = int(pair["patch_B_id"])
        pa = pat_a.get(pa_id)
        pb = pat_b.get(pb_id)
        if pa is None or pb is None:
            continue

        src_a = set(pa.source_indices.tolist())
        src_b = set(pb.source_indices.tolist())

        col_a = np.tile(C_GRAY, (len(pts_a), 1))
        col_b = np.tile(C_GRAY, (len(pts_b), 1))

        # Contact zone (dimmer red so patches stand out)
        for idx in ci_a_set: col_a[idx] = C_RED
        for idx in ci_b_set: col_b[idx] = C_RED

        # Patch points: orange if in contact, else blue/green
        for idx in src_a:
            col_a[idx] = C_ORANGE if idx in ci_a_set else C_BLUE
        for idx in src_b:
            col_b[idx] = C_ORANGE if idx in ci_b_set else C_GREEN

        cen_a = np.array(pair["patch_A_center"])
        cen_b = np.array(pair["patch_B_center"]) + offset

        geoms = [
            colored_pcd(pts_a, col_a),
            colored_pcd(pts_b_shifted, col_b),
            make_line(cen_a, cen_b, C_YELLOW),
            make_sphere(cen_a, 1.2, C_BLUE),
            make_sphere(cen_b, 1.2, C_GREEN),
        ]

        ov_a = pair.get("contact_overlap_A", 0)
        ov_b = pair.get("contact_overlap_B", 0)
        dist = pair.get("center_distance_mm", 0)

        print(f"  Pair {i+1}/{len(selected)}: "
              f"frag{a} patch{pa_id} ↔ frag{b} patch{pb_id}  "
              f"overlap_A={ov_a:.2f}  overlap_B={ov_b:.2f}  "
              f"centre_dist={dist:.1f}mm")

        show(geoms,
             f"Positive {i+1}/{len(selected)}: "
             f"frag{a} p{pa_id} ↔ frag{b} p{pb_id}  "
             f"| ovA={ov_a:.2f} ovB={ov_b:.2f} dist={dist:.1f}mm")

    print("\nDone.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 3 — Negative pairs viewer
# ══════════════════════════════════════════════════════════════════════════════
def mode_negatives(a, b, count):
    print(f"\n── RANDOM NEGATIVE PAIRS: Fragment {a} ↔ Fragment {b} ──")

    pairs = get_negative_pairs(a, b)
    if not pairs:
        print(f"No negative pairs found for {a}↔{b}. "
              f"Only non-adjacent pairs have random negatives.")
        return
    print(f"  {len(pairs):,} negative pairs available.")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), min(count, len(pairs)), replace=False)
    selected = [pairs[i] for i in sorted(idx)]

    # Resolve actual fragment numbers from pair data
    nums = {fid(n): n for n in range(1, 8)}
    fa_n = nums[selected[0]["fragment_A_id"]]
    fb_n = nums[selected[0]["fragment_B_id"]]

    pts_a = load_fragment(fa_n)
    pts_b = load_fragment(fb_n)
    pat_a = {p.patch_id: p for p in load_patches(fa_n)}
    pat_b = {p.patch_id: p for p in load_patches(fb_n)}

    offset = side_by_side_offset(pts_a)
    pts_b_shifted = pts_b + offset

    print("\nColour key:")
    print("  PURPLE = patch A  |  CYAN = patch B  |  GRAY = rest of fragment")
    print("  RED line = patch centre connection (these should NOT match)")
    print("  Close window for next.\n")

    for i, pair in enumerate(selected):
        pa_id = int(pair["patch_A_id"])
        pb_id = int(pair["patch_B_id"])
        pa = pat_a.get(pa_id)
        pb = pat_b.get(pb_id)
        if pa is None or pb is None:
            continue

        col_a = np.tile(C_GRAY, (len(pts_a), 1))
        col_b = np.tile(C_GRAY, (len(pts_b), 1))

        for si in pa.source_indices: col_a[si] = C_PURPLE
        for si in pb.source_indices: col_b[si] = C_CYAN

        cen_a = np.array(pair["patch_A_center"])
        cen_b = np.array(pair["patch_B_center"]) + offset

        geoms = [
            colored_pcd(pts_a, col_a),
            colored_pcd(pts_b_shifted, col_b),
            make_line(cen_a, cen_b, C_RED),
            make_sphere(cen_a, 1.2, C_PURPLE),
            make_sphere(cen_b, 1.2, C_CYAN),
        ]

        dist = pair.get("center_distance_mm", 0)
        print(f"  Pair {i+1}/{len(selected)}: "
              f"frag{fa_n} patch{pa_id} ↔ frag{fb_n} patch{pb_id}  "
              f"centre_dist={dist:.1f}mm  (should NOT match)")

        show(geoms,
             f"Negative {i+1}/{len(selected)}: "
             f"frag{fa_n} p{pa_id} ↔ frag{fb_n} p{pb_id}  "
             f"| dist={dist:.1f}mm — NON-MATCH")

    print("\nDone.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Interactive Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["contacts", "positives", "negatives"],
        help="contacts: all 11 contact pairs | positives: positive patch pairs | negatives: negative patch pairs"
    )
    parser.add_argument(
        "--pair", type=int, nargs=2, metavar=("A", "B"),
        help="Fragment numbers, e.g. --pair 6 7"
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of pairs to show (default: 5)"
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="Show a specific pair by index (positives mode only)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PHASE 3 VISUALIZER")
    print("  Controls: rotate=left-drag  zoom=scroll  pan=right-drag")
    print("  Close each window to advance to the next pair.")
    print("="*60)

    if args.mode == "contacts":
        mode_contacts()
    elif args.mode == "positives":
        if not args.pair:
            parser.error("--pair A B is required for positives mode")
        mode_positives(args.pair[0], args.pair[1], args.count, args.index)
    elif args.mode == "negatives":
        if not args.pair:
            parser.error("--pair A B is required for negatives mode")
        mode_negatives(args.pair[0], args.pair[1], args.count)


if __name__ == "__main__":
    main()
