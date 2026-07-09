#!/usr/bin/env python3
"""Phase 4 diagnostic visualizations.

Seven figures that make the *similarity vs assembly-compatibility* story visible
and set up Phase 5. All outputs are PNGs written to
``baseline_results/visualizations/`` (matplotlib Agg backend -- no display
required).

Usage:
    PYTHONPATH=src python3 scripts/visualize_phase4.py --viz all
    PYTHONPATH=src python3 scripts/visualize_phase4.py --viz 1        # retrieval
    PYTHONPATH=src python3 scripts/visualize_phase4.py --viz 1 --descriptor fpfh --n-queries 6

Vizzes:
    1  retrieval    Query patch + top-10 FPFH neighbours (TP green / FP red)
    2  embedding    PCA + t-SNE of descriptors, coloured 4 ways
    3  distinct     Per-fragment distinctiveness classes in 3D
    4  contact      Contact regions (red) per adjacent pair; 5-7 highlighted
    5  registration Ground-truth vs global-FPFH vs contact-seeded overlays
    6  hardneg      Most-similar patches across NON-adjacent fragments
    7  graph        Fragment adjacency graph (edge = contact size / #positives)
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baseline_geometry import data_access as da
from baseline_geometry import descriptors as dm
from baseline_geometry import registration as reg
from baseline_geometry.config_loader import load_config

# ── paths ──────────────────────────────────────────────────────────────────
CFG_PATH = ROOT / "config" / "baseline_geometry.yaml"
OUT_DIR = ROOT / "baseline_results" / "visualizations"
DESC_DIR = ROOT / "baseline_results" / "descriptors"
DISTINCT_NPZ = ROOT / "phase3_final_review" / "patch_distinctiveness.npz"
PAIR_ANALYSIS = ROOT / "phase3_final_review" / "fragment_pair_analysis.json"

# ── colours ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "highly_distinctive": "#d62728",   # red
    "moderately_distinctive": "#f0c020",  # yellow
    "ambiguous": "#1f6fe0",            # blue
    "flat": "#9a9a9a",                 # gray
}
FRAG_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

ADJACENT_PAIRS = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
                  (3, 4), (4, 5), (5, 6), (5, 7), (6, 7)]


def fid(n: int) -> str:
    return f"fragment_caesar_fragment_{n}"


def frag_num(fragment_id: str) -> int:
    return int(fragment_id.rsplit("_", 1)[-1])


# ═══════════════════════════════════════════════════════════════════════════
# Shared loaders (cached in module-level dicts)
# ═══════════════════════════════════════════════════════════════════════════
class Context:
    """Lazy-loaded shared data for all visualizations."""

    def __init__(self, descriptor: str = "fpfh"):
        self.cfg = load_config(str(CFG_PATH))
        self.fragment_ids = da.load_fragment_ids(self.cfg.dataset_dir)
        self.descriptor = descriptor
        self._clouds: dict[str, da.FragmentCloud] = {}
        self._tables: dict[str, da.PatchTable] = {}
        self._desc: dict[str, dm.PatchDescriptors] = {}
        self._pairs = None
        self._distinct = None

    # --- fragments / patches ---
    def cloud(self, fragment_id: str) -> da.FragmentCloud:
        if fragment_id not in self._clouds:
            self._clouds[fragment_id] = da.load_fragment_cloud(self.cfg.dataset_dir, fragment_id)
        return self._clouds[fragment_id]

    def table(self, fragment_id: str) -> da.PatchTable:
        if fragment_id not in self._tables:
            self._tables[fragment_id] = da.load_patch_table(self.cfg.patches_dir, fragment_id)
        return self._tables[fragment_id]

    def desc(self, fragment_id: str) -> dm.PatchDescriptors:
        if fragment_id not in self._desc:
            path = DESC_DIR / self.descriptor / f"{fragment_id}.npz"
            self._desc[fragment_id] = dm.read_descriptors(str(path))
        return self._desc[fragment_id]

    @property
    def pairs(self) -> da.PairTable:
        if self._pairs is None:
            self._pairs = da.load_pairs(self.cfg.pairs_npz)
        return self._pairs

    @property
    def distinct(self) -> dict:
        if self._distinct is None:
            d = np.load(DISTINCT_NPZ, allow_pickle=True)
            self._distinct = {k: d[k] for k in d.files}
        return self._distinct

    # --- global descriptor gallery ---
    def gallery(self):
        """Return (D matrix, frag_ids array, patch_ids array) for all patches."""
        mats, frags, pids = [], [], []
        for fidx in self.fragment_ids:
            pd = self.desc(fidx)
            mats.append(pd.descriptors)
            frags.extend([fidx] * pd.patch_count)
            pids.extend(int(x) for x in pd.patch_ids)
        return np.vstack(mats), np.asarray(frags, dtype=object), np.asarray(pids, dtype=np.int64)

    def positive_partners(self) -> dict:
        """(frag, patch) -> set of (frag, patch) linked by a positive label."""
        p = self.pairs
        vocab = p.fragment_vocab
        pos = np.where(p.labels == 1)[0]
        out: dict[tuple, set] = {}
        for i in pos:
            a = (str(vocab[p.fragment_A_idx[i]]), int(p.patch_A_ids[i]))
            b = (str(vocab[p.fragment_B_idx[i]]), int(p.patch_B_ids[i]))
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
        return out

    def contact_patch_set(self) -> set:
        """Set of (frag, patch) that appear in >=1 positive pair (contact-zone)."""
        p = self.pairs
        vocab = p.fragment_vocab
        pos = np.where(p.labels == 1)[0]
        s = set()
        for i in pos:
            s.add((str(vocab[p.fragment_A_idx[i]]), int(p.patch_A_ids[i])))
            s.add((str(vocab[p.fragment_B_idx[i]]), int(p.patch_B_ids[i])))
        return s


def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_equal_3d(ax, pts):
    """Equal aspect for a 3D axis given a point set."""
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 1 — Retrieval successes and failures
# ═══════════════════════════════════════════════════════════════════════════
def viz_retrieval(ctx: Context, n_queries: int = 6, top_k: int = 10, seed: int = 0):
    _ensure_out()
    gal_desc, gal_frag, gal_pid = ctx.gallery()
    global_row = {(gal_frag[i], int(gal_pid[i])): i for i in range(len(gal_pid))}
    partners = ctx.positive_partners()
    distinct = ctx.distinct
    # class lookup: (frag, patch) -> class
    cls_lookup = {
        (str(distinct["fragment_ids"][i]), int(distinct["patch_ids"][i])): str(distinct["classes"][i])
        for i in range(len(distinct["patch_ids"]))
    }

    # Query candidates = patches that HAVE positive partners (so TP is possible).
    query_keys = [k for k in partners.keys() if k in global_row]
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(query_keys), size=min(n_queries, len(query_keys)), replace=False)
    query_keys = [query_keys[i] for i in sel]

    ncol = top_k + 1
    fig = plt.figure(figsize=(2.0 * ncol, 2.3 * len(query_keys)))
    fig.suptitle(
        f"Viz 1 — {ctx.descriptor.upper()} retrieval: query + top-{top_k} nearest "
        f"cross-fragment patches\n(green=true positive / assembles, red=false positive)",
        fontsize=12,
    )

    for r, qkey in enumerate(query_keys):
        qrow = global_row[qkey]
        qvec = gal_desc[qrow]
        qfrag, qpid = qkey
        # cross-fragment ranking
        cross = gal_frag != qfrag
        cand = np.where(cross)[0]
        dists = np.linalg.norm(gal_desc[cand] - qvec, axis=1)
        order = cand[np.argsort(dists, kind="mergesort")][:top_k]
        gt = partners.get(qkey, set())

        # query panel
        ax = fig.add_subplot(len(query_keys), ncol, r * ncol + 1, projection="3d")
        _draw_patch(ax, ctx, qfrag, qpid, color="#000000")
        ax.set_title(f"Q: F{frag_num(qfrag)}#{qpid}\n{cls_lookup.get(qkey,'?')[:10]}", fontsize=7)
        _strip_3d(ax)

        for c, grow in enumerate(order):
            rkey = (gal_frag[grow], int(gal_pid[grow]))
            is_tp = rkey in gt
            edge = "#2ca02c" if is_tp else "#d62728"
            d = float(np.linalg.norm(gal_desc[grow] - qvec))
            ax = fig.add_subplot(len(query_keys), ncol, r * ncol + 2 + c, projection="3d")
            _draw_patch(ax, ctx, rkey[0], rkey[1], color=edge)
            ax.set_title(f"F{frag_num(rkey[0])}#{rkey[1]}\nd={d:.2f}", fontsize=7, color=edge)
            _strip_3d(ax)
            for spine_pos in ("bottom", "top", "left", "right"):
                pass
            # colour the panel border
            ax.patch.set_edgecolor(edge)
            ax.patch.set_linewidth(2)

    legend = [
        Line2D([0], [0], color="#2ca02c", lw=3, label="True positive (assembles)"),
        Line2D([0], [0], color="#d62728", lw=3, label="False positive (looks similar only)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUT_DIR / f"viz1_retrieval_{ctx.descriptor}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"✓ Viz 1 written: {out}")


def _draw_patch(ax, ctx: Context, fragment_id: str, patch_id: int, color: str):
    table = ctx.table(fragment_id)
    row = table.row_for_patch_id(patch_id)
    pts = table.patch_points(row)
    c = pts - pts.mean(axis=0)
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=4, c=color, alpha=0.7)
    _set_equal_3d(ax, c)


def _strip_3d(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 2 — Embedding space (PCA + t-SNE)
# ═══════════════════════════════════════════════════════════════════════════
def viz_embedding(ctx: Context, tsne_sample: int = 2500, seed: int = 0):
    _ensure_out()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    gal_desc, gal_frag, gal_pid = ctx.gallery()
    n = gal_desc.shape[0]

    # colour maps
    distinct = ctx.distinct
    cls_lookup = {
        (str(distinct["fragment_ids"][i]), int(distinct["patch_ids"][i])): str(distinct["classes"][i])
        for i in range(len(distinct["patch_ids"]))
    }
    contact = ctx.contact_patch_set()

    keys = [(gal_frag[i], int(gal_pid[i])) for i in range(n)]
    frag_idx = np.array([frag_num(k[0]) for k in keys])
    is_contact = np.array([k in contact for k in keys])
    classes = np.array([cls_lookup.get(k, "flat") for k in keys])

    # PCA (all points) + t-SNE (subsample for speed)
    pca = PCA(n_components=2, random_state=seed).fit_transform(gal_desc)
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, size=min(tsne_sample, n), replace=False)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca",
                learning_rate="auto").fit_transform(gal_desc[sub])

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(
        f"Viz 2 — {ctx.descriptor.upper()} embedding space (top: PCA all {n} patches; "
        f"bottom: t-SNE {len(sub)} sample)", fontsize=13)

    def scatter_by_fragment(ax, xy, fnums, title):
        for f in range(1, 8):
            m = fnums == f
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=FRAG_COLORS[f - 1], label=f"F{f}", alpha=0.6)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7, markerscale=1.5, ncol=2)

    def scatter_by_contact(ax, xy, cmask, title):
        ax.scatter(xy[~cmask, 0], xy[~cmask, 1], s=5, color="#cccccc", alpha=0.5, label="non-contact")
        ax.scatter(xy[cmask, 0], xy[cmask, 1], s=8, color="#d62728", alpha=0.7, label="contact-zone")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8)

    def scatter_by_class(ax, xy, cls, title):
        for name, col in CLASS_COLORS.items():
            m = cls == name
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=col, alpha=0.6, label=name)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7)

    scatter_by_fragment(axes[0, 0], pca, frag_idx, "PCA — coloured by fragment")
    scatter_by_contact(axes[0, 1], pca, is_contact, "PCA — contact vs non-contact")
    scatter_by_class(axes[0, 2], pca, classes, "PCA — distinctiveness class")
    scatter_by_fragment(axes[1, 0], tsne, frag_idx[sub], "t-SNE — coloured by fragment")
    scatter_by_contact(axes[1, 1], tsne, is_contact[sub], "t-SNE — contact vs non-contact")
    scatter_by_class(axes[1, 2], tsne, classes[sub], "t-SNE — distinctiveness class")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / f"viz2_embedding_{ctx.descriptor}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"✓ Viz 2 written: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 3 — Distinctiveness scores in 3D
# ═══════════════════════════════════════════════════════════════════════════
def viz_distinctiveness(ctx: Context):
    _ensure_out()
    distinct = ctx.distinct
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle("Viz 3 — Patch distinctiveness class per fragment "
                 "(red=distinctive, yellow=moderate, blue=ambiguous, gray=flat)", fontsize=13)

    for i, fnum in enumerate(range(1, 8)):
        fragment_id = fid(fnum)
        table = ctx.table(fragment_id)
        centers = table.centers  # (M,3)
        # class per patch id
        mask = distinct["fragment_ids"] == fragment_id
        pid_arr = distinct["patch_ids"][mask]
        cls_arr = distinct["classes"][mask]
        cls_by_pid = {int(p): c for p, c in zip(pid_arr, cls_arr)}
        colors = [CLASS_COLORS.get(cls_by_pid.get(int(pid), "flat"), "#9a9a9a")
                  for pid in table.patch_ids]

        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=6, c=colors, alpha=0.75)
        ax.set_title(f"Fragment {fnum}  ({table.patch_count} patches)", fontsize=10)
        _set_equal_3d(ax, centers)
        _strip_3d(ax)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=n)
              for n, c in CLASS_COLORS.items()]
    fig.legend(handles=legend, loc="lower right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "viz3_distinctiveness.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"✓ Viz 3 written: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 4 — Contact regions
# ═══════════════════════════════════════════════════════════════════════════
def viz_contact(ctx: Context, max_points: int = 6000):
    _ensure_out()
    contact_dir = Path(ctx.cfg.pairs_dataset_json).parent / "contact_regions"
    rng = np.random.default_rng(0)

    n = len(ADJACENT_PAIRS)
    fig = plt.figure(figsize=(20, 2.6 * n))
    fig.suptitle("Viz 4 — Contact regions (red) per adjacent pair. "
                 "Left=Fragment A, Right=Fragment B. Note the tiny 5↔7 contact.", fontsize=13)

    for r, (a, b) in enumerate(ADJACENT_PAIRS):
        fa, fb = fid(a), fid(b)
        region = da.load_contact_region(str(contact_dir), fa, fb)
        ca = ctx.cloud(fa); cb = ctx.cloud(fb)

        for col, (fnum, cloud, idx_key) in enumerate(
            [(a, ca, "contact_indices_A"), (b, cb, "contact_indices_B")]
        ):
            ax = fig.add_subplot(n, 2, r * 2 + col + 1, projection="3d")
            pts = cloud.points
            if len(pts) > max_points:
                s = rng.choice(len(pts), max_points, replace=False)
                pts_plot = pts[s]
            else:
                pts_plot = pts
            ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
                       s=2, c="#cccccc", alpha=0.3)
            ncontact = 0
            if region is not None:
                cidx = region[idx_key]
                ncontact = len(cidx)
                if ncontact > 0:
                    cp = pts[cidx]
                    ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=10, c="#d62728", alpha=0.9)
            flag = "  ⚠ TINY" if ncontact < 20 else ""
            ax.set_title(f"F{a}↔F{b}: Fragment {fnum}  (contact={ncontact}){flag}", fontsize=9)
            _set_equal_3d(ax, pts_plot)
            _strip_3d(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "viz4_contact_regions.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"✓ Viz 4 written: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 5 — Registration failures (GT vs global vs contact)
# ═══════════════════════════════════════════════════════════════════════════
def _global_transform(ctx, ca, cb, T_perturb):
    """Reproduce global FPFH+RANSAC+ICP; return the estimated 4x4 transform."""
    import open3d as o3d
    cfg = ctx.cfg
    pts_b = (T_perturb[:3, :3] @ cb.points.T).T + T_perturb[:3, 3]
    nrm_b = (T_perturb[:3, :3] @ cb.normals.T).T if cb.normals.shape[0] == cb.points.shape[0] else np.empty((0, 3))
    src_down, src_f = reg._prepare(pts_b, nrm_b, cfg.registration_voxel_mm, cfg.fpfh_radius_mm,
                                   cfg.fpfh_max_nn, cfg.normal_radius_mm, cfg.normal_max_nn)
    tgt_down, tgt_f = reg._prepare(ca.points, ca.normals, cfg.registration_voxel_mm, cfg.fpfh_radius_mm,
                                   cfg.fpfh_max_nn, cfg.normal_radius_mm, cfg.normal_max_nn)
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_f, tgt_f, True, cfg.ransac_max_corr_dist_mm,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), cfg.ransac_n,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(cfg.ransac_max_corr_dist_mm)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(cfg.ransac_max_iterations, cfg.ransac_confidence))
    icp = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down, cfg.icp_max_corr_dist_mm, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iterations))
    return np.asarray(icp.transformation), pts_b


def _contact_transform(ctx, ca, cb, T_perturb, region):
    """Reproduce contact-seeded Kabsch+ICP; return estimated 4x4 transform."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    cfg = ctx.cfg
    idx_a, idx_b = region["contact_indices_A"], region["contact_indices_B"]
    pa, pb = ca.points[idx_a], cb.points[idx_b]
    if pa.shape[0] < 3 or pb.shape[0] < 3:
        return None, None
    _, nn = cKDTree(pa).query(pb, k=1)
    matched_a = pa[nn]
    matched_b_pert = (T_perturb[:3, :3] @ pb.T).T + T_perturb[:3, 3]
    T_est = reg._kabsch(matched_b_pert, matched_a)
    pts_b_pert = (T_perturb[:3, :3] @ cb.points.T).T + T_perturb[:3, 3]
    # Refine on the CONTACT INTERFACE ONLY (abutting fragments don't overlap).
    src = reg._to_o3d(matched_b_pert, None)
    tgt = reg._to_o3d(pa, None)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg.normal_radius_mm, max_nn=cfg.normal_max_nn))
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, cfg.icp_max_corr_dist_mm, T_est,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iterations))
    return np.asarray(icp.transformation), pts_b_pert


def _apply(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def viz_registration(ctx: Context, pairs=None, max_points: int = 4000):
    _ensure_out()
    cfg = ctx.cfg
    contact_dir = Path(cfg.pairs_dataset_json).parent / "contact_regions"
    rng = np.random.default_rng(cfg.perturb_seed)
    if pairs is None:
        pairs = ADJACENT_PAIRS

    n = len(pairs)
    fig = plt.figure(figsize=(15, 3.6 * n))
    fig.suptitle("Viz 5 — Registration: A(gray) + B recovered(orange). "
                 "Left=ground truth, Middle=global FPFH, Right=contact-seeded.\n"
                 "rot/trans = residual error vs identity (lower is better).", fontsize=12)

    def _sub(pts, m):
        if len(pts) > m:
            s = rng.choice(len(pts), m, replace=False)
            return pts[s]
        return pts

    for r, (a, b) in enumerate(pairs):
        fa, fb = fid(a), fid(b)
        ca, cb = ctx.cloud(fa), ctx.cloud(fb)
        region = da.load_contact_region(str(contact_dir), fa, fb)
        T_perturb = reg.random_rigid_transform(cfg.perturb_max_rotation_deg,
                                               cfg.perturb_max_translation_mm, rng)

        # Ground truth: B already in assembled frame (identity recovery).
        panels = []
        panels.append(("Ground truth", cb.points, 0.0, 0.0))

        T_glob, _ = _global_transform(ctx, ca, cb, T_perturb)
        pts_b_pert = _apply(T_perturb, cb.points)
        b_glob = _apply(T_glob, pts_b_pert)
        res_g = T_glob @ T_perturb
        panels.append(("Global FPFH", b_glob,
                       reg.rotation_error_deg(res_g), reg.translation_error_mm(res_g)))

        if region is not None:
            T_con, _ = _contact_transform(ctx, ca, cb, T_perturb, region)
            if T_con is not None:
                b_con = _apply(T_con, pts_b_pert)
                res_c = T_con @ T_perturb
                panels.append(("Contact-seeded", b_con,
                               reg.rotation_error_deg(res_c), reg.translation_error_mm(res_c)))
        while len(panels) < 3:
            panels.append(("(no contact)", None, float("nan"), float("nan")))

        a_pts = _sub(ca.points, max_points)
        for c, (title, b_pts, rot, trans) in enumerate(panels):
            ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
            ax.scatter(a_pts[:, 0], a_pts[:, 1], a_pts[:, 2], s=2, c="#bbbbbb", alpha=0.35)
            allpts = a_pts
            if b_pts is not None:
                bp = _sub(b_pts, max_points)
                ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2], s=2, c="#ff7f0e", alpha=0.45)
                allpts = np.vstack([a_pts, bp])
            good = (rot <= cfg.success_rotation_deg and trans <= cfg.success_translation_mm)
            tcol = "#2ca02c" if (good and c > 0) else ("#000000" if c == 0 else "#d62728")
            label = f"F{a}↔F{b} {title}"
            if c > 0 and np.isfinite(rot):
                label += f"\nrot={rot:.0f}° trans={trans:.0f}mm"
            ax.set_title(label, fontsize=8, color=tcol)
            _set_equal_3d(ax, allpts)
            _strip_3d(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "viz5_registration.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"✓ Viz 5 written: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 6 — Hard negative candidates (similar but non-adjacent)
# ═══════════════════════════════════════════════════════════════════════════
def viz_hard_negatives(ctx: Context, n_show: int = 8, per_pair_cap: int = 20000, seed: int = 0):
    _ensure_out()
    adjacent_set = {tuple(sorted(p)) for p in ADJACENT_PAIRS}
    frag_nums = list(range(1, 8))
    non_adjacent = [(a, b) for a, b in combinations(frag_nums, 2)
                    if (a, b) not in adjacent_set]

    rng = np.random.default_rng(seed)
    candidates = []  # (dist, fa, pa, fb, pb)
    for a, b in non_adjacent:
        da_ = ctx.desc(fid(a)); db_ = ctx.desc(fid(b))
        Da, Db = da_.descriptors, db_.descriptors
        # subsample to bound the O(Ma*Mb) search
        ia = rng.choice(Da.shape[0], min(400, Da.shape[0]), replace=False)
        ib = rng.choice(Db.shape[0], min(400, Db.shape[0]), replace=False)
        sub_a, sub_b = Da[ia], Db[ib]
        # pairwise distances
        d = np.linalg.norm(sub_a[:, None, :] - sub_b[None, :, :], axis=2)
        # take the few smallest from this pair
        flat = np.argsort(d, axis=None)[:5]
        for f in flat:
            r, c = np.unravel_index(f, d.shape)
            candidates.append((float(d[r, c]), fid(a), int(da_.patch_ids[ia[r]]),
                               fid(b), int(db_.patch_ids[ib[c]])))

    candidates.sort(key=lambda x: x[0])
    top = candidates[:n_show]

    fig = plt.figure(figsize=(4.2 * n_show / 2, 9))
    fig.suptitle(f"Viz 6 — Hard-negative candidates ({ctx.descriptor.upper()}): most-similar "
                 f"patches from NON-adjacent fragments.\nThey look alike but do NOT assemble — "
                 f"gold for Phase 5 contrastive training.", fontsize=12)
    cols = n_show
    for i, (dist, fa, pa, fb, pb) in enumerate(top):
        ax = fig.add_subplot(2, cols, i + 1, projection="3d")
        _draw_patch(ax, ctx, fa, pa, "#1f6fe0")
        ax.set_title(f"A: F{frag_num(fa)}#{pa}", fontsize=8); _strip_3d(ax)
        ax = fig.add_subplot(2, cols, cols + i + 1, projection="3d")
        _draw_patch(ax, ctx, fb, pb, "#ff7f0e")
        ax.set_title(f"B: F{frag_num(fb)}#{pb}\nFPFH d={dist:.2f}", fontsize=8); _strip_3d(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / f"viz6_hard_negatives_{ctx.descriptor}.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    # also dump the list as JSON for Phase 5 use
    dump = [{"fpfh_distance": d, "fragment_A": fa, "patch_A": pa,
             "fragment_B": fb, "patch_B": pb} for d, fa, pa, fb, pb in candidates[:200]]
    (OUT_DIR / f"viz6_hard_negative_candidates_{ctx.descriptor}.json").write_text(
        json.dumps(dump, indent=2))
    print(f"✓ Viz 6 written: {out} (+ candidate list JSON)")


# ═══════════════════════════════════════════════════════════════════════════
# VIZ 7 — Fragment adjacency graph
# ═══════════════════════════════════════════════════════════════════════════
def viz_graph(ctx: Context):
    _ensure_out()
    analysis = json.loads(PAIR_ANALYSIS.read_text())
    # edge stats keyed by (a,b)
    edges = {}
    for e in analysis["pairs_ranked_by_info"]:
        a = frag_num(e["fragment_A"]); b = frag_num(e["fragment_B"])
        edges[tuple(sorted((a, b)))] = {
            "contact": e["contact_points_A"] + e["contact_points_B"],
            "positives": e["positive_pairs"],
        }

    # Fixed layout reflecting the discovered topology.
    pos = {
        1: (0.0, 1.0), 2: (1.0, 1.0), 3: (0.0, 0.0), 4: (1.0, 0.0),
        5: (2.2, 0.0), 6: (3.4, 0.4), 7: (3.4, -0.6),
    }

    max_pos = max((v["positives"] for v in edges.values()), default=1) or 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Viz 7 — Fragment adjacency graph  "
                 "(edge width/label: left=#positive pairs, right=contact points)", fontsize=13)

    for ax, mode in zip(axes, ("positives", "contact")):
        for (a, b), st in edges.items():
            x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
            val = st[mode]
            if mode == "positives":
                lw = 0.5 + 8.0 * (val / max_pos)
                col = "#d62728" if val == 0 else plt.cm.viridis(val / max_pos)
                lbl = f"{val:,}"
            else:
                mx = max(v["contact"] for v in edges.values())
                lw = 0.5 + 8.0 * (val / mx)
                col = "#d62728" if val < 20 else plt.cm.viridis(val / mx)
                lbl = f"{val}"
            ax.plot(x, y, "-", lw=lw, color=col, alpha=0.7, zorder=1)
            ax.text((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, lbl, fontsize=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        # nodes
        for f, (x, y) in pos.items():
            deg = sum(1 for (a, b) in edges if f in (a, b))
            size = 800 + 400 * deg
            ax.scatter([x], [y], s=size, color=FRAG_COLORS[f - 1], zorder=2, edgecolors="k")
            ax.text(x, y, f"F{f}", fontsize=13, ha="center", va="center",
                    fontweight="bold", zorder=3)
        ax.set_title(f"Edge weight = {mode}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.6, 4.2); ax.set_ylim(-1.3, 1.6)

    # annotate hubs / peripherals
    axes[0].text(-0.5, -1.15, "F1,F4,F5 = hubs (4 neighbours) · F6,F7 = peripheral · "
                 "5↔7 edge ≈ 0 positives (tiny contact)", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "viz7_adjacency_graph.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"✓ Viz 7 written: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
VIZ_BY_NAME = {
    "1": "retrieval", "retrieval": "retrieval",
    "2": "embedding", "embedding": "embedding",
    "3": "distinct", "distinctiveness": "distinct", "distinct": "distinct",
    "4": "contact", "contact": "contact",
    "5": "registration", "registration": "registration",
    "6": "hardneg", "hard_negatives": "hardneg", "hardneg": "hardneg",
    "7": "graph", "adjacency": "graph", "graph": "graph",
}


def main():
    ap = argparse.ArgumentParser(description="Phase 4 diagnostic visualizations")
    ap.add_argument("--viz", nargs="+", default=["all"],
                    help="Which viz to run: all, or any of 1-7 / names")
    ap.add_argument("--descriptor", default="fpfh", choices=["fpfh", "shot"])
    ap.add_argument("--n-queries", type=int, default=6, help="viz1 query count")
    ap.add_argument("--pairs", nargs="*", type=int, default=None,
                    help="viz5 fragment numbers as flat pairs, e.g. 5 7 1 4")
    args = ap.parse_args()

    requested = args.viz
    if "all" in requested:
        order = ["graph", "distinct", "contact", "retrieval", "hardneg", "embedding", "registration"]
    else:
        order = []
        for r in requested:
            key = VIZ_BY_NAME.get(str(r).lower())
            if key is None:
                print(f"⚠ unknown viz '{r}', skipping"); continue
            order.append(key)

    ctx = Context(descriptor=args.descriptor)

    reg_pairs = None
    if args.pairs:
        it = iter(args.pairs)
        reg_pairs = list(zip(it, it))

    for key in order:
        if key == "retrieval":
            viz_retrieval(ctx, n_queries=args.n_queries)
        elif key == "embedding":
            viz_embedding(ctx)
        elif key == "distinct":
            viz_distinctiveness(ctx)
        elif key == "contact":
            viz_contact(ctx)
        elif key == "registration":
            viz_registration(ctx, pairs=reg_pairs)
        elif key == "hardneg":
            viz_hard_negatives(ctx)
        elif key == "graph":
            viz_graph(ctx)

    print(f"\nAll requested visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
