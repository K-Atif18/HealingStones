#!/usr/bin/env python3
"""Generate static PNG figures for the README (Phases 1-3).

Phase 4 already has figures in baseline_results/visualizations/. This script
produces matplotlib (Agg, no display) figures for Phases 1-3 and copies/refits
the Phase 1 render, writing everything into assets/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baseline_geometry import data_access as da
from baseline_geometry.config_loader import load_config

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

FRAG_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def fid(n):
    return f"fragment_caesar_fragment_{n}"


def _equal_3d(ax, pts):
    mins, maxs = pts.min(0), pts.max(0)
    c = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)


def phase1_assembly(ctx):
    """7 fragments coloured, assembled in the ground-truth frame."""
    fig = plt.figure(figsize=(12, 6))
    for j, view in enumerate([(20, -60), (20, 120)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        allpts = []
        for i in range(1, 8):
            pts = ctx.cloud(i).points
            s = np.random.default_rng(i).choice(len(pts), min(4000, len(pts)), replace=False)
            p = pts[s]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, color=FRAG_COLORS[i - 1], alpha=0.7,
                       label=f"F{i}")
            allpts.append(p)
        allpts = np.vstack(allpts)
        _equal_3d(ax, allpts)
        ax.view_init(elev=view[0], azim=view[1])
        if j == 0:
            ax.legend(fontsize=8, markerscale=2, ncol=2, loc="upper left")
    fig.suptitle("Phase 1 — 7 fragments aligned into the assembled Caesar statue "
                 "(reconstruction RMSE 4.6 mm)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ASSETS / "phase1_assembly.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def phase2_patches(ctx):
    """Patch centres coloured, + one fragment's patch-density heatmap."""
    fig = plt.figure(figsize=(14, 6))
    # left: all fragments' patch centres coloured by fragment
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    allc = []
    for i in range(1, 8):
        centers = ctx.table(i).centers
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=4,
                   color=FRAG_COLORS[i - 1], alpha=0.7, label=f"F{i}")
        allc.append(centers)
    _equal_3d(ax, np.vstack(allc))
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=8, markerscale=2, ncol=2, loc="upper left")
    ax.set_title("6,822 patch centres (FPS) across 7 fragments", fontsize=10)

    # right: fragment 4 points coloured by local patch density (coverage)
    fnum = 4
    cloud = ctx.cloud(fnum)
    table = ctx.table(fnum)
    from scipy.spatial import cKDTree
    tree = cKDTree(table.centers)
    # count patch centres within one radius of each point
    counts = tree.query_ball_point(cloud.points, r=ctx.cfg.fpfh_radius_mm, return_length=True)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    p = ax2.scatter(cloud.points[:, 0], cloud.points[:, 1], cloud.points[:, 2],
                    s=3, c=counts, cmap="viridis")
    _equal_3d(ax2, cloud.points)
    ax2.view_init(elev=20, azim=-60)
    ax2.set_title(f"Fragment {fnum}: patch coverage density (100% covered)", fontsize=10)
    fig.colorbar(p, ax=ax2, shrink=0.5, label="# overlapping patches")

    fig.suptitle("Phase 2 — Overlapping local patches (8 mm radius, ~100 pts each, 100% coverage)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ASSETS / "phase2_patches.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def phase3_contacts(ctx):
    """Contact regions for a few representative adjacent pairs."""
    contact_dir = str(Path(ctx.cfg.pairs_dataset_json).parent / "contact_regions")
    pairs = [(1, 4), (4, 5), (2, 3), (5, 7)]  # large, thin, medium, tiny
    fig = plt.figure(figsize=(16, 4.2))
    for k, (a, b) in enumerate(pairs):
        region = da.load_contact_region(contact_dir, fid(a), fid(b))
        ca, cb = ctx.cloud(a), ctx.cloud(b)
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        for cloud, key, base in [(ca, "contact_indices_A", "#8fb8e0"),
                                 (cb, "contact_indices_B", "#f0b080")]:
            pts = cloud.points
            s = np.random.default_rng(1).choice(len(pts), min(3000, len(pts)), replace=False)
            ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], s=2, color="#dddddd", alpha=0.3)
        ncontact = 0
        if region is not None:
            for cloud, key in [(ca, "contact_indices_A"), (cb, "contact_indices_B")]:
                if region["fragment_A_id"] == fid(a):
                    idx = region[key]
                else:
                    idx = region["contact_indices_B" if key == "contact_indices_A" else "contact_indices_A"]
                if len(idx):
                    cp = cloud.points[idx]
                    ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=8, color="#d62728", alpha=0.9)
                    ncontact += len(idx)
        allp = np.vstack([ca.points, cb.points])
        _equal_3d(ax, allp)
        ax.view_init(elev=20, azim=-60)
        flag = "  ⚠ TINY" if ncontact < 30 else ""
        ax.set_title(f"F{a} ↔ F{b}  (contact≈{ncontact} pts){flag}", fontsize=9)
    fig.suptitle("Phase 3 — Ground-truth contact regions (red) drive positive-pair generation",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS / "phase3_contacts.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


class Ctx:
    def __init__(self):
        self.cfg = load_config(str(ROOT / "config" / "baseline_geometry.yaml"))
        self._cloud, self._table = {}, {}

    def cloud(self, n):
        if n not in self._cloud:
            self._cloud[n] = da.load_fragment_cloud(self.cfg.dataset_dir, fid(n))
        return self._cloud[n]

    def table(self, n):
        if n not in self._table:
            self._table[n] = da.load_patch_table(self.cfg.patches_dir, fid(n))
        return self._table[n]


def main():
    ctx = Ctx()
    phase1_assembly(ctx)
    phase2_patches(ctx)
    phase3_contacts(ctx)
    # copy the 7 phase-4 figures into assets for a stable README path
    import shutil
    src = ROOT / "baseline_results" / "visualizations"
    for f in sorted(src.glob("viz*.png")):
        shutil.copy(f, ASSETS / f.name)
        print(f"✓ copied {f.name}")
    print(f"\nAll README assets in: {ASSETS}")


if __name__ == "__main__":
    main()
