#!/usr/bin/env python3
"""Interactive, fragment-centric Phase 4 explorer.

Opens each of the 7 Phase 4 diagnostics in its OWN window, focused on ONE
fragment at a time, and prints exactly what to look for so you can validate the
result by eye. The 3D views (Open3D) are fully rotatable/zoomable; the two
abstract plots (embedding, graph) open as matplotlib windows.

Windows open one after another -- CLOSE a window to advance to the next.

Usage:
    # All 7 windows for fragment 1
    PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 1

    # Just the 3D contact + registration windows for fragment 5
    PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 5 --viz 4 5

    # Pick which descriptor drives retrieval / hard-negatives
    PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 4 --descriptor fpfh

Viz index:
    1 distinctiveness   3D  patch classes painted on the fragment
    2 contact           3D  contact zones to each neighbour (one colour each)
    3 retrieval         3D  query patches + their top-k FPFH neighbours (TP/FP)
    4 hardneg           3D  similar-but-non-adjacent patch pairs
    5 registration      3D  neighbours re-assembled around this fragment
    6 embedding         2D  PCA/t-SNE with this fragment highlighted
    7 graph             2D  adjacency graph with this fragment highlighted
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import open3d as o3d

from baseline_geometry import data_access as da
from baseline_geometry import descriptors as dm
from baseline_geometry import registration as reg
from baseline_geometry.config_loader import load_config

# ── paths ──────────────────────────────────────────────────────────────────
CFG_PATH = ROOT / "config" / "baseline_geometry.yaml"
DESC_DIR = ROOT / "baseline_results" / "descriptors"
DISTINCT_NPZ = ROOT / "phase3_final_review" / "patch_distinctiveness.npz"
PAIR_ANALYSIS = ROOT / "phase3_final_review" / "fragment_pair_analysis.json"

ADJACENT_PAIRS = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
                  (3, 4), (4, 5), (5, 6), (5, 7), (6, 7)]

# ── colours (RGB 0-1) ────────────────────────────────────────────────────────
CLASS_RGB = {
    "highly_distinctive": (0.84, 0.15, 0.16),   # red
    "moderately_distinctive": (0.94, 0.75, 0.13),  # yellow
    "ambiguous": (0.12, 0.44, 0.88),            # blue
    "flat": (0.62, 0.62, 0.62),                 # gray
}
NEIGHBOR_PALETTE = [
    (0.84, 0.15, 0.16), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17),
    (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.09, 0.75, 0.81), (0.89, 0.47, 0.76),
]
GRAY = (0.75, 0.75, 0.75)
GREEN = (0.17, 0.63, 0.17)
RED = (0.84, 0.15, 0.16)
BLACK = (0.05, 0.05, 0.05)
BLUE = (0.12, 0.44, 0.88)
ORANGE = (1.0, 0.5, 0.05)


def fid(n: int) -> str:
    return f"fragment_caesar_fragment_{n}"


def frag_num(fragment_id: str) -> int:
    return int(fragment_id.rsplit("_", 1)[-1])


def rule(title: str):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def guide(lines):
    print("\n  WHAT TO LOOK FOR (validate):")
    for ln in lines:
        print(f"   • {ln}")
    print("\n  → Rotate/zoom the window. CLOSE it to open the next.\n")


# ═══════════════════════════════════════════════════════════════════════════
# Shared lazy-loaded context
# ═══════════════════════════════════════════════════════════════════════════
class Ctx:
    def __init__(self, descriptor="fpfh"):
        self.cfg = load_config(str(CFG_PATH))
        self.descriptor = descriptor
        self.fragment_ids = da.load_fragment_ids(self.cfg.dataset_dir)
        self._cloud, self._table, self._desc = {}, {}, {}
        self._pairs = None
        self._distinct = None

    def cloud(self, fnum):
        k = fid(fnum)
        if k not in self._cloud:
            self._cloud[k] = da.load_fragment_cloud(self.cfg.dataset_dir, k)
        return self._cloud[k]

    def table(self, fnum):
        k = fid(fnum)
        if k not in self._table:
            self._table[k] = da.load_patch_table(self.cfg.patches_dir, k)
        return self._table[k]

    def desc(self, fnum):
        k = fid(fnum)
        if k not in self._desc:
            self._desc[k] = dm.read_descriptors(str(DESC_DIR / self.descriptor / f"{k}.npz"))
        return self._desc[k]

    @property
    def pairs(self):
        if self._pairs is None:
            self._pairs = da.load_pairs(self.cfg.pairs_npz)
        return self._pairs

    @property
    def distinct(self):
        if self._distinct is None:
            d = np.load(DISTINCT_NPZ, allow_pickle=True)
            self._distinct = {k: d[k] for k in d.files}
        return self._distinct

    def neighbors(self, fnum):
        out = []
        for a, b in ADJACENT_PAIRS:
            if a == fnum:
                out.append(b)
            elif b == fnum:
                out.append(a)
        return sorted(out)

    def patch_points(self, fnum, patch_id):
        t = self.table(fnum)
        return t.patch_points(t.row_for_patch_id(int(patch_id)))

    def positive_partners(self):
        p = self.pairs
        vocab = p.fragment_vocab
        pos = np.where(p.labels == 1)[0]
        out = {}
        for i in pos:
            a = (str(vocab[p.fragment_A_idx[i]]), int(p.patch_A_ids[i]))
            b = (str(vocab[p.fragment_B_idx[i]]), int(p.patch_B_ids[i]))
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
        return out

    def gallery(self):
        mats, frags, pids = [], [], []
        for k in self.fragment_ids:
            pd = self._desc.get(k) or dm.read_descriptors(str(DESC_DIR / self.descriptor / f"{k}.npz"))
            self._desc[k] = pd
            mats.append(pd.descriptors)
            frags.extend([k] * pd.patch_count)
            pids.extend(int(x) for x in pd.patch_ids)
        return np.vstack(mats), np.asarray(frags, dtype=object), np.asarray(pids, np.int64)


# ── Open3D helpers ───────────────────────────────────────────────────────────
def make_pcd(points, color=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, np.float64))
    elif color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def show(geoms, title):
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1100, height=800)


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 1 — Distinctiveness painted on the fragment
# ═══════════════════════════════════════════════════════════════════════════
def viz1_distinctiveness(ctx: Ctx, fnum: int):
    rule(f"[1/7] DISTINCTIVENESS — Fragment {fnum}")
    d = ctx.distinct
    mask = d["fragment_ids"] == fid(fnum)
    pid_arr = d["patch_ids"][mask]
    cls_arr = d["classes"][mask]
    cls_by_pid = {int(p): c for p, c in zip(pid_arr, cls_arr)}

    table = ctx.table(fnum)
    centers = table.centers
    center_cls = [cls_by_pid.get(int(pid), "flat") for pid in table.patch_ids]

    cloud = ctx.cloud(fnum)
    pts = cloud.points
    # colour each surface point by the class of its nearest patch centre
    tree = cKDTree(centers)
    _, nn = tree.query(pts, k=1)
    colors = np.array([CLASS_RGB[center_cls[i]] for i in nn])

    counts = {c: int(sum(1 for x in center_cls if x == c)) for c in CLASS_RGB}
    print(f"  patch classes: {counts}")
    guide([
        "RED = highly distinctive (complex curvature), YELLOW = moderate,",
        "  BLUE = ambiguous, GRAY = flat.",
        "Distinctive (red) should sit on EDGES, ridges, and irregular features.",
        "Flat/ambiguous (gray/blue) should fill SMOOTH interior areas.",
        "If red is scattered randomly with no relation to shape, something is off.",
    ])
    show([make_pcd(pts, colors=colors)], f"[1] Distinctiveness — Fragment {fnum}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 2 — Contact regions to each neighbour
# ═══════════════════════════════════════════════════════════════════════════
def viz2_contact(ctx: Ctx, fnum: int):
    rule(f"[2/7] CONTACT REGIONS — Fragment {fnum}")
    contact_dir = str(Path(ctx.cfg.pairs_dataset_json).parent / "contact_regions")
    cloud = ctx.cloud(fnum)
    base = make_pcd(cloud.points, color=GRAY)
    geoms = [base]
    print(f"  Fragment {fnum} neighbours: {ctx.neighbors(fnum)}")
    print("  Contact colour legend:")
    for i, nb in enumerate(ctx.neighbors(fnum)):
        region = da.load_contact_region(contact_dir, fid(fnum), fid(nb))
        if region is None:
            continue
        # which side of the archive is THIS fragment?
        if region["fragment_A_id"] == fid(fnum):
            idx = region["contact_indices_A"]
        else:
            idx = region["contact_indices_B"]
        col = NEIGHBOR_PALETTE[i % len(NEIGHBOR_PALETTE)]
        flag = "  <-- TINY / suspect" if len(idx) < 20 else ""
        print(f"    F{nb}: {len(idx)} contact points  RGB{tuple(round(c,2) for c in col)}{flag}")
        if len(idx) > 0:
            cp = make_pcd(cloud.points[idx], color=col)
            geoms.append(cp)
    guide([
        "Each coloured patch = where this fragment touches ONE neighbour.",
        "Contact zones must lie on the BREAK boundary, not the outer surface.",
        "A big solid patch = strong planar interface (registers well).",
        "A thin line / tiny speck = weak edge contact (registration will struggle).",
        f"For F{fnum}, check every neighbour region is a plausible fracture face.",
    ])
    show(geoms, f"[2] Contact regions — Fragment {fnum}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 3 — Retrieval (query patches from this fragment + top-k neighbours)
# ═══════════════════════════════════════════════════════════════════════════
def viz3_retrieval(ctx: Ctx, fnum: int, n_queries=4, top_k=8):
    rule(f"[3/7] RETRIEVAL — Fragment {fnum}  (descriptor={ctx.descriptor.upper()})")
    gal, gfrag, gpid = ctx.gallery()
    grow = {(gfrag[i], int(gpid[i])): i for i in range(len(gpid))}
    partners = ctx.positive_partners()

    # queries = patches on THIS fragment that have >=1 positive partner
    cand = [pid for pid in ctx.table(fnum).patch_ids
            if (fid(fnum), int(pid)) in partners and (fid(fnum), int(pid)) in grow]
    if not cand:
        print("  (no contact patches on this fragment to query)")
        return
    rng = np.random.default_rng(fnum)
    q_pids = rng.choice(cand, size=min(n_queries, len(cand)), replace=False)

    geoms = []
    step = 26.0
    print(f"  Showing {len(q_pids)} query patches (col 0, BLACK) + top-{top_k} matches per row.")
    print("  GREEN = true positive (really assembles), RED = false positive (looks similar only).")
    n_tp_total = 0
    for r, qp in enumerate(q_pids):
        qkey = (fid(fnum), int(qp))
        qvec = gal[grow[qkey]]
        cross = gfrag != fid(fnum)
        ci = np.where(cross)[0]
        dists = np.linalg.norm(gal[ci] - qvec, axis=1)
        order = ci[np.argsort(dists)][:top_k]
        gt = partners.get(qkey, set())

        y = -r * step
        # query patch (black), centred
        qpts = ctx.patch_points(fnum, qp)
        qpts = qpts - qpts.mean(0) + np.array([0, y, 0])
        geoms.append(make_pcd(qpts, color=BLACK))

        row_tp = 0
        for c, gi in enumerate(order):
            rkey = (gfrag[gi], int(gpid[gi]))
            is_tp = rkey in gt
            row_tp += int(is_tp)
            col = GREEN if is_tp else RED
            pp = ctx.patch_points(frag_num(rkey[0]), rkey[1])
            pp = pp - pp.mean(0) + np.array([(c + 1) * step, y, 0])
            geoms.append(make_pcd(pp, color=col))
        n_tp_total += row_tp
        print(f"    query F{fnum}#{qp}: {row_tp}/{top_k} of top-{top_k} are true positives")
    print(f"  TOTAL true positives in top-{top_k}: {n_tp_total}/{len(q_pids)*top_k}")
    guide([
        "Left-most BLACK patch in each row is the query (from this fragment).",
        "To its right: the 8 nearest patches by descriptor distance, left=closest.",
        "GREEN among the closest = descriptor found a real assembly partner (good).",
        "Many RED (esp. flat blobs / thin lines) = it matched by LOOK, not fit.",
        "Expected for a similarity baseline: mostly RED, few GREEN, GREEN not first.",
        "This is the visual meaning of 'similarity != assembly compatibility'.",
    ])
    show(geoms, f"[3] Retrieval — Fragment {fnum} ({ctx.descriptor})")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 4 — Hard negatives (similar but non-adjacent) involving this fragment
# ═══════════════════════════════════════════════════════════════════════════
def viz4_hardneg(ctx: Ctx, fnum: int, n_show=6):
    rule(f"[4/7] HARD NEGATIVES — Fragment {fnum}  (descriptor={ctx.descriptor.upper()})")
    non_adj = [n for n in range(1, 8) if n != fnum and n not in ctx.neighbors(fnum)]
    print(f"  Fragment {fnum} is NON-adjacent to: {non_adj}")
    if not non_adj:
        print("  (this fragment is adjacent to everything — no hard negatives)")
        return

    da_ = ctx.desc(fnum)
    Da = da_.descriptors
    rng = np.random.default_rng(fnum)
    ia = rng.choice(Da.shape[0], min(300, Da.shape[0]), replace=False)

    cands = []
    for nb in non_adj:
        db_ = ctx.desc(nb)
        Db = db_.descriptors
        ib = rng.choice(Db.shape[0], min(300, Db.shape[0]), replace=False)
        d = np.linalg.norm(Da[ia][:, None, :] - Db[ib][None, :, :], axis=2)
        for f in np.argsort(d, axis=None)[:4]:
            r, c = np.unravel_index(f, d.shape)
            cands.append((float(d[r, c]), int(da_.patch_ids[ia[r]]), nb, int(db_.patch_ids[ib[c]])))
    cands.sort(key=lambda x: x[0])
    top = cands[:n_show]

    geoms = []
    step = 26.0
    print(f"  Most-similar patches on F{fnum} (BLUE) vs a NON-adjacent fragment (ORANGE):")
    for r, (dist, pa, nb, pb) in enumerate(top):
        y = -r * step
        a = ctx.patch_points(fnum, pa); a = a - a.mean(0) + np.array([0, y, 0])
        b = ctx.patch_points(nb, pb); b = b - b.mean(0) + np.array([step, y, 0])
        geoms.append(make_pcd(a, color=BLUE))
        geoms.append(make_pcd(b, color=ORANGE))
        print(f"    F{fnum}#{pa}  vs  F{nb}#{pb}   {ctx.descriptor.upper()} d={dist:.3f}")
    guide([
        "Each row: LEFT (blue) = a patch on this fragment; RIGHT (orange) = its",
        "  most-similar patch on a fragment it does NOT touch.",
        "They should look ALMOST IDENTICAL (both flat, or both same curve)...",
        "  ...yet they can never assemble. That contradiction is the point.",
        "These are exactly the 'hard negatives' Phase 5 needs to learn from.",
    ])
    show(geoms, f"[4] Hard negatives — Fragment {fnum} ({ctx.descriptor})")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 5 — Registration: re-assemble neighbours around this fragment
# ═══════════════════════════════════════════════════════════════════════════
def viz5_registration(ctx: Ctx, fnum: int):
    rule(f"[5/7] REGISTRATION — neighbours re-assembled around Fragment {fnum}")
    cfg = ctx.cfg
    contact_dir = str(Path(cfg.pairs_dataset_json).parent / "contact_regions")
    anchor = ctx.cloud(fnum)
    geoms = [make_pcd(anchor.points, color=GRAY)]
    rng = np.random.default_rng(cfg.perturb_seed)

    print(f"  Anchor = Fragment {fnum} (GRAY, fixed in ground-truth frame).")
    print("  Each neighbour is PERTURBED by a known transform, then recovered by")
    print("  CONTACT-SEEDED registration and drawn in colour. Errors printed below.\n")
    for i, nb in enumerate(ctx.neighbors(fnum)):
        cb = ctx.cloud(nb)
        region = da.load_contact_region(contact_dir, fid(fnum), fid(nb))
        T = reg.random_rigid_transform(cfg.perturb_max_rotation_deg, cfg.perturb_max_translation_mm, rng)
        col = NEIGHBOR_PALETTE[i % len(NEIGHBOR_PALETTE)]
        if region is None:
            continue
        # orient region so A = this fragment
        if region["fragment_A_id"] == fid(fnum):
            idx_a, idx_b = region["contact_indices_A"], region["contact_indices_B"]
            ca_pts, cb_pts = anchor.points[idx_a], cb.points[idx_b]
        else:
            idx_a, idx_b = region["contact_indices_B"], region["contact_indices_A"]
            ca_pts, cb_pts = anchor.points[idx_a], cb.points[idx_b]

        status = ""
        if len(ca_pts) >= 3 and len(cb_pts) >= 3:
            _, nn = cKDTree(ca_pts).query(cb_pts, k=1)
            matched_a = ca_pts[nn]
            matched_b_pert = (T[:3, :3] @ cb_pts.T).T + T[:3, 3]
            T_est = reg._kabsch(matched_b_pert, matched_a)
            pts_b_pert = (T[:3, :3] @ cb.points.T).T + T[:3, 3]
            # Refine on the CONTACT INTERFACE ONLY (not full clouds): abutting
            # fragments don't overlap, so full-cloud ICP would drag B off-pose.
            contact_b_pert = (T[:3, :3] @ cb_pts.T).T + T[:3, 3]
            src = reg._to_o3d(contact_b_pert, None)
            tgt = reg._to_o3d(ca_pts, None)
            tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=cfg.normal_radius_mm, max_nn=cfg.normal_max_nn))
            icp = o3d.pipelines.registration.registration_icp(
                src, tgt, cfg.icp_max_corr_dist_mm, T_est,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iterations))
            T_final = np.asarray(icp.transformation)
            recovered = (T_final[:3, :3] @ pts_b_pert.T).T + T_final[:3, 3]
            res = T_final @ T
            rot = reg.rotation_error_deg(res); trans = reg.translation_error_mm(res)
            ok = rot <= cfg.success_rotation_deg and trans <= cfg.success_translation_mm
            status = f"rot={rot:5.1f}°  trans={trans:5.1f}mm  {'OK ✓' if ok else 'FAIL ✗'}"
            geoms.append(make_pcd(recovered, color=col))
        else:
            status = f"only {len(ca_pts)}/{len(cb_pts)} contact pts — UNREGISTERABLE"
        print(f"    F{nb}: {status}   (colour RGB{tuple(round(c,2) for c in col)})")

    guide([
        "GRAY = this fragment (fixed). Each colour = a neighbour recovered from a",
        "  random perturbation using its ground-truth contact interface.",
        "A well-recovered neighbour should CLICK into its correct assembled spot,",
        "  hugging the gray fragment along the shared break (small rot/trans error).",
        "A neighbour floating away / mis-oriented = thin-contact failure (check the",
        "  printed error and cross-reference its contact size in window [2]).",
    ])
    show(geoms, f"[5] Registration — assembled around Fragment {fnum}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 6 — Embedding (2D, this fragment highlighted)
# ═══════════════════════════════════════════════════════════════════════════
def viz6_embedding(ctx: Ctx, fnum: int, tsne_sample=2500):
    rule(f"[6/7] EMBEDDING SPACE — Fragment {fnum} highlighted  ({ctx.descriptor.upper()})")
    import matplotlib
    matplotlib.use("GTK3Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    gal, gfrag, gpid = ctx.gallery()
    is_this = np.array([f == fid(fnum) for f in gfrag])
    n = gal.shape[0]

    pca = PCA(n_components=2, random_state=0).fit_transform(gal)
    rng = np.random.default_rng(0)
    # ensure this fragment is represented in the t-SNE subsample
    this_idx = np.where(is_this)[0]
    other_idx = np.where(~is_this)[0]
    take_other = min(tsne_sample - len(this_idx), len(other_idx))
    sub = np.concatenate([this_idx, rng.choice(other_idx, take_other, replace=False)])
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, init="pca",
                learning_rate="auto").fit_transform(gal[sub])
    sub_is_this = is_this[sub]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Embedding ({ctx.descriptor.upper()}) — Fragment {fnum} in RED vs all others in gray",
                 fontsize=13)
    axes[0].scatter(pca[~is_this, 0], pca[~is_this, 1], s=5, c="#cccccc", alpha=0.5, label="other fragments")
    axes[0].scatter(pca[is_this, 0], pca[is_this, 1], s=10, c="#d62728", alpha=0.8, label=f"Fragment {fnum}")
    axes[0].set_title("PCA (all patches)"); axes[0].legend()
    axes[1].scatter(tsne[~sub_is_this, 0], tsne[~sub_is_this, 1], s=6, c="#cccccc", alpha=0.5, label="other")
    axes[1].scatter(tsne[sub_is_this, 0], tsne[sub_is_this, 1], s=12, c="#d62728", alpha=0.8, label=f"F{fnum}")
    axes[1].set_title("t-SNE (subsample)"); axes[1].legend()
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    guide([
        "RED = this fragment's patches; GRAY = all other fragments' patches.",
        "GOOD (transferable descriptor): red is SPREAD THROUGHOUT the gray cloud —",
        "  local geometry is shared across fragments, so matching can generalise.",
        "WARNING (identity leakage): red forms its OWN isolated island — the",
        "  descriptor is encoding 'which fragment' instead of transferable shape,",
        "  which hurts retrieval to unseen objects. (F6/F7 tend to do this.)",
    ])
    print("  (matplotlib window open — close it to continue)")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 7 — Adjacency graph (2D, this fragment highlighted)
# ═══════════════════════════════════════════════════════════════════════════
def viz7_graph(ctx: Ctx, fnum: int):
    rule(f"[7/7] ADJACENCY GRAPH — Fragment {fnum} highlighted")
    import matplotlib
    matplotlib.use("GTK3Agg", force=True)
    import matplotlib.pyplot as plt

    analysis = json.loads(PAIR_ANALYSIS.read_text())
    edges = {}
    for e in analysis["pairs_ranked_by_info"]:
        a, b = frag_num(e["fragment_A"]), frag_num(e["fragment_B"])
        edges[tuple(sorted((a, b)))] = {
            "contact": e["contact_points_A"] + e["contact_points_B"],
            "positives": e["positive_pairs"],
        }
    pos = {1: (0, 1), 2: (1, 1), 3: (0, 0), 4: (1, 0), 5: (2.2, 0), 6: (3.4, 0.4), 7: (3.4, -0.6)}
    max_pos = max(v["positives"] for v in edges.values()) or 1

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle(f"Adjacency graph — Fragment {fnum} (gold) and its neighbours (orange edges)", fontsize=13)
    nbrs = set(ctx.neighbors(fnum))
    for (a, b), st in edges.items():
        touches = fnum in (a, b)
        lw = 0.6 + 8 * (st["positives"] / max_pos)
        if st["positives"] == 0:
            col = "#d62728"
        elif touches:
            col = "#ff7f0e"
        else:
            col = "#cccccc"
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "-", lw=lw, color=col,
                alpha=0.9 if touches else 0.4, zorder=1)
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        ax.text(mx, my, f"{st['positives']:,}\n({st['contact']}pts)", fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    for f, (x, y) in pos.items():
        if f == fnum:
            c, sz = "#f0c020", 2200
        elif f in nbrs:
            c, sz = "#ff7f0e", 1500
        else:
            c, sz = "#bbbbbb", 1100
        ax.scatter([x], [y], s=sz, color=c, edgecolors="k", zorder=2)
        ax.text(x, y, f"F{f}", fontsize=13, ha="center", va="center", fontweight="bold", zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.6, 4.2); ax.set_ylim(-1.3, 1.6)

    print(f"  Fragment {fnum} neighbours (from ground truth): {sorted(nbrs)}")
    guide([
        "GOLD node = this fragment; ORANGE nodes = its neighbours; ORANGE edges =",
        "  its contacts. Edge label = #positive pairs and (contact points).",
        "Confirm the orange neighbours match the contact windows you saw in [2].",
        "A RED edge = 0 positive pairs (e.g. 5-7): a real adjacency too small to use.",
        "Thicker edge = more training signal from that interface.",
    ])
    print("  (matplotlib window open — close it to finish)")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
VIZZES = {
    1: viz1_distinctiveness, 2: viz2_contact, 3: viz3_retrieval, 4: viz4_hardneg,
    5: viz5_registration, 6: viz6_embedding, 7: viz7_graph,
}


def main():
    ap = argparse.ArgumentParser(description="Interactive per-fragment Phase 4 explorer")
    ap.add_argument("--fragment", type=int, default=1, choices=range(1, 8),
                    help="Fragment number 1-7")
    ap.add_argument("--viz", nargs="+", type=int, default=list(range(1, 8)),
                    help="Which windows to open (1-7), in order")
    ap.add_argument("--descriptor", default="fpfh", choices=["fpfh", "shot"])
    args = ap.parse_args()

    ctx = Ctx(descriptor=args.descriptor)
    print(f"\n### Interactive explorer — Fragment {args.fragment} — "
          f"descriptor={args.descriptor.upper()} ###")
    print("Windows open ONE AT A TIME. Close each window to advance to the next.")
    for v in args.viz:
        if v not in VIZZES:
            print(f"  (skipping unknown viz {v})")
            continue
        try:
            VIZZES[v](ctx, args.fragment)
        except Exception as exc:  # keep going through the sequence
            print(f"  ⚠ viz {v} failed: {exc}")
    print("\nDone. Re-run with a different --fragment to inspect another one.")


if __name__ == "__main__":
    main()
