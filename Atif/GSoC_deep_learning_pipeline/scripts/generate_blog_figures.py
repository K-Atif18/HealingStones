#!/usr/bin/env python3
"""Generate the missing blog figures from existing JSON data.

Produces:
  1. Training loss curve (train + val) from fold 1 history
  2. Per-fold encoder vs FPFH mAP comparison bar chart with CI error bars
  3. Hard-negative AUC comparison across folds

Usage:
    PYTHONPATH=src python3 scripts/generate_blog_figures.py

Outputs written to: assets/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training Loss Curve (Fold 1, default run)
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curve():
    """Plot training and validation loss over epochs."""
    hist_path = ROOT / "phase5a_runs" / "fold_fragment_caesar_fragment_1_history.json"
    if not hist_path.exists():
        print(f"⚠ Skipping training curve: {hist_path} not found")
        return

    hist = load_json(hist_path)
    epochs_data = hist["epochs"]

    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss_mean"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]

    # Find best validation epoch
    best_epoch_idx = np.argmin(val_loss)
    best_val = val_loss[best_epoch_idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, train_loss, "b-", linewidth=1.8, label="Training loss", alpha=0.9)
    ax.plot(epochs, val_loss, "r-", linewidth=1.8, label="Validation loss", alpha=0.9)

    # Mark best validation
    ax.axvline(x=epochs[best_epoch_idx], color="gray", linestyle="--", alpha=0.5)
    ax.scatter([epochs[best_epoch_idx]], [best_val], color="red", s=80, zorder=5,
               marker="*", label=f"Best val (epoch {epochs[best_epoch_idx]}, loss={best_val:.3f})")

    # Annotate the gap
    ax.annotate(
        "Train-val gap widens\n(overfitting to similarity)",
        xy=(35, (train_loss[35] + val_loss[35]) / 2),
        xytext=(38, 5.8),
        fontsize=9, color="purple",
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("InfoNCE Loss", fontsize=12)
    ax.set_title("Phase 5 Training Dynamics — Fold 1 (Held-out: Fragment 1)\n"
                 "Loss decreases, but model learns similarity, not complementarity",
                 fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 51)

    # Add secondary annotation
    ax.text(0.02, 0.05, f"Stopped by: epoch_cap (50)\nBest checkpoint: epoch {epochs[best_epoch_idx]}",
            transform=ax.transAxes, fontsize=9, color="gray",
            verticalalignment="bottom", fontfamily="monospace")

    fig.tight_layout()
    out = ASSETS / "blog_training_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-Fold Encoder vs FPFH mAP Comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_fold_comparison():
    """Grouped bar chart: encoder vs FPFH mAP per fold with CI error bars."""
    # Collect data from eval JSONs
    folds = []

    # Fold 1 (from phase5a_runs)
    eval1_path = ROOT / "phase5a_runs" / "eval_encoder.json"
    if eval1_path.exists():
        e1 = load_json(eval1_path)
        folds.append({
            "fold": "F1",
            "encoder_mAP": e1["heldout_fold_retrieval"]["encoder"]["mAP"],
            "fpfh_mAP": e1["heldout_fold_retrieval"]["fpfh"]["mAP"],
            "ci_lo": e1["paired_mAP_diff_encoder_minus_fpfh"]["ci95_lo"],
            "ci_hi": e1["paired_mAP_diff_encoder_minus_fpfh"]["ci95_hi"],
            "diff": e1["paired_mAP_diff_encoder_minus_fpfh"]["mean"],
        })

    # Folds 2, 3, 4 (from phase5a_runs_6fold)
    for fold_num in [2, 3, 4]:
        eval_path = ROOT / "phase5a_runs_6fold" / f"fold{fold_num}" / "eval.json"
        if eval_path.exists():
            e = load_json(eval_path)
            folds.append({
                "fold": f"F{fold_num}",
                "encoder_mAP": e["heldout_fold_retrieval"]["encoder"]["mAP"],
                "fpfh_mAP": e["heldout_fold_retrieval"]["fpfh"]["mAP"],
                "ci_lo": e["paired_mAP_diff_encoder_minus_fpfh"]["ci95_lo"],
                "ci_hi": e["paired_mAP_diff_encoder_minus_fpfh"]["ci95_hi"],
                "diff": e["paired_mAP_diff_encoder_minus_fpfh"]["mean"],
            })

    if not folds:
        print("⚠ Skipping fold comparison: no eval JSONs found")
        return

    fold_labels = [f["fold"] for f in folds]
    encoder_maps = [f["encoder_mAP"] for f in folds]
    fpfh_maps = [f["fpfh_mAP"] for f in folds]

    x = np.arange(len(fold_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1.2]})

    # Left panel: grouped bar chart
    bars1 = ax1.bar(x - width / 2, encoder_maps, width, label="Learned Encoder",
                    color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, fpfh_maps, width, label="FPFH Baseline",
                    color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Random baseline
    ax1.axhline(y=0.061, color="gray", linestyle=":", linewidth=1.5, label="Random baseline (0.061)")

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, color="#c0392b")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, color="#2980b9")

    ax1.set_xlabel("LOFO Fold (Held-out Fragment)", fontsize=11)
    ax1.set_ylabel("Retrieval mAP", fontsize=11)
    ax1.set_title("Encoder vs FPFH: Per-Fold mAP\n(Encoder loses on all 4 well-supported folds)",
                  fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels, fontsize=11)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_ylim(0, max(fpfh_maps) * 1.25)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right panel: difference with CI
    diffs = [f["diff"] for f in folds]
    ci_lo = [f["ci_lo"] for f in folds]
    ci_hi = [f["ci_hi"] for f in folds]
    errors = [[d - lo for d, lo in zip(diffs, ci_lo)],
              [hi - d for d, hi in zip(diffs, ci_hi)]]

    colors = ["#c0392b" if d < 0 else "#27ae60" for d in diffs]
    ax2.barh(x, diffs, xerr=errors, color=colors, alpha=0.8,
             edgecolor="black", linewidth=0.5, capsize=4)
    ax2.axvline(x=0, color="black", linewidth=1.2)
    ax2.set_yticks(x)
    ax2.set_yticklabels(fold_labels, fontsize=11)
    ax2.set_xlabel("mAP difference (Encoder − FPFH)", fontsize=10)
    ax2.set_title("Bootstrap 95% CI\n(all exclude zero → significant)", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")

    for i, (d, lo, hi) in enumerate(zip(diffs, ci_lo, ci_hi)):
        ax2.text(min(d, lo) - 0.003, i, f"[{lo:.4f}, {hi:.4f}]",
                 va="center", ha="right", fontsize=7.5, color="gray")

    fig.suptitle("Phase 5 Negative Result: Learned Encoder Significantly Worse Than FPFH",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ASSETS / "blog_fold_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Hard-Negative AUC Across Folds
# ─────────────────────────────────────────────────────────────────────────────

def fig_hard_negative_auc():
    """Bar chart showing hard-negative AUC per fold — the smoking gun."""
    fold_data = []

    for fold_num in [2, 3, 4]:
        eval_path = ROOT / "phase5a_runs_6fold" / f"fold{fold_num}" / "eval.json"
        if eval_path.exists():
            e = load_json(eval_path)
            cond3 = e.get("condition3_encoder_mined_VALID", {}).get("encoder", {})
            if cond3:
                fold_data.append({
                    "fold": f"F{fold_num}",
                    "hard_auc": cond3["auc_pos_vs_hard"],
                    "easy_auc": cond3["auc_pos_vs_easy"],
                })

    if not fold_data:
        print("⚠ Skipping hard-neg AUC: no eval data found")
        return

    fold_labels = [f["fold"] for f in fold_data]
    hard_aucs = [f["hard_auc"] for f in fold_data]
    easy_aucs = [f["easy_auc"] for f in fold_data]

    x = np.arange(len(fold_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_easy = ax.bar(x - width / 2, easy_aucs, width, label="AUC pos-vs-easy",
                       color="#27ae60", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars_hard = ax.bar(x + width / 2, hard_aucs, width, label="AUC pos-vs-hard (encoder-mined)",
                       color="#c0392b", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Chance line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (0.5)")

    # Value labels on hard bars
    for bar in bars_hard:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#c0392b")
    for bar in bars_easy:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color="#27ae60")

    # Annotation
    ax.annotate(
        "BELOW CHANCE\nEncoder actively prefers\nlook-alikes over true partners",
        xy=(x[1] + width / 2, hard_aucs[1]),
        xytext=(x[1] + 0.6, 0.25),
        fontsize=9, color="#c0392b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
        ha="center",
    )

    ax.set_xlabel("LOFO Fold (Held-out Fragment)", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("The Smoking Gun: Hard-Negative AUC (Encoder-Mined)\n"
                 "The encoder cannot resist its own look-alikes → learned similarity, not complementarity",
                 fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = ASSETS / "blog_hard_negative_auc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Mining Distance Over Training (embedding collapse diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def fig_mining_distance():
    """Plot per-epoch mean mining distance — shows embeddings spreading apart over training."""
    hist_path = ROOT / "phase5a_runs" / "fold_fragment_caesar_fragment_1_history.json"
    if not hist_path.exists():
        print(f"⚠ Skipping mining distance: {hist_path} not found")
        return

    hist = load_json(hist_path)
    epochs_data = hist["epochs"]

    epochs = []
    mean_dists = []

    for e in epochs_data:
        frag_dists = e.get("per_fragment_mined_distance_mean", {})
        if frag_dists:
            epochs.append(e["epoch"])
            mean_dists.append(np.mean(list(frag_dists.values())))

    if not epochs:
        print("⚠ Skipping mining distance: no per_fragment_mined_distance_mean in history")
        return

    # Also plot deduped hard neg count
    deduped = [e["deduped_hard_negatives_per_step_mean"] for e in epochs_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(epochs, mean_dists, "o-", color="#8e44ad", linewidth=1.5, markersize=4, alpha=0.8)
    ax1.set_ylabel("Mean mining distance\n(to nearest non-adjacent look-alike)", fontsize=10)
    ax1.set_title("Hard-Negative Mining Dynamics Over Training\n"
                  "Distance increases → embeddings spread apart → fewer unique hard negatives",
                  fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=mean_dists[0], color="gray", linestyle=":", alpha=0.5)
    ax1.text(2, mean_dists[0] + 0.005, f"Epoch 0: {mean_dists[0]:.3f}", fontsize=8, color="gray")

    ax2.plot(range(len(deduped)), deduped, "s-", color="#e67e22", linewidth=1.5, markersize=4, alpha=0.8)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Unique hard negatives\nper batch (deduped)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = ASSETS / "blog_mining_dynamics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Similarity vs Complementarity Conceptual Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_similarity_vs_complementarity():
    """Conceptual diagram showing why similarity != complementarity for assembly."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Similarity (what InfoNCE optimizes)
    ax = axes[0]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Similarity\n(what InfoNCE optimizes)", fontsize=13, fontweight="bold",
                 color="#c0392b")

    # Draw two similar convex shapes
    theta = np.linspace(0, np.pi, 50)
    # Shape A
    x1 = 2 + 1.5 * np.cos(theta)
    y1 = 4 + 1.2 * np.sin(theta)
    ax.fill(x1, y1, color="#3498db", alpha=0.6)
    ax.text(2, 2.5, "Patch A\n(convex)", ha="center", fontsize=10, color="#2c3e50")

    # Shape B (similar convex)
    x2 = 8 + 1.5 * np.cos(theta)
    y2 = 4 + 1.2 * np.sin(theta)
    ax.fill(x2, y2, color="#e74c3c", alpha=0.6)
    ax.text(8, 2.5, "Patch B\n(also convex)", ha="center", fontsize=10, color="#2c3e50")

    # Arrow between them
    ax.annotate("", xy=(6.2, 4), xytext=(3.8, 4),
                arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=2.5))
    ax.text(5, 4.5, "sim(A,B) ↑", ha="center", fontsize=11, color="#27ae60", fontweight="bold")
    ax.text(5, 0.5, "X  Cannot assemble!\n(convex cannot fit convex)", ha="center",
            fontsize=10, color="#c0392b", fontweight="bold")

    # Right: Complementarity (what assembly needs)
    ax = axes[1]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Complementarity\n(what assembly needs)", fontsize=13, fontweight="bold",
                 color="#27ae60")

    # Shape A (convex)
    x1 = 2 + 1.5 * np.cos(theta)
    y1 = 4 + 1.2 * np.sin(theta)
    ax.fill(x1, y1, color="#3498db", alpha=0.6)
    ax.text(2, 2.5, "Patch A\n(convex)", ha="center", fontsize=10, color="#2c3e50")

    # Shape C (concave - inverse)
    theta2 = np.linspace(0, np.pi, 50)
    x3 = 8 + 1.5 * np.cos(theta2)
    y3 = 4 - 1.2 * np.sin(theta2)  # flipped
    ax.fill(x3, y3, color="#f39c12", alpha=0.6)
    ax.text(8, 2.5, "Patch C\n(concave)", ha="center", fontsize=10, color="#2c3e50")

    # Arrow between them
    ax.annotate("", xy=(6.2, 4), xytext=(3.8, 4),
                arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=2.5))
    ax.text(5, 4.5, "fit(A,C) ↑", ha="center", fontsize=11, color="#27ae60", fontweight="bold")
    ax.text(5, 0.5, ">> These assemble!\n(convex interlocks with concave)", ha="center",
            fontsize=10, color="#27ae60", fontweight="bold")

    # Overall caption
    fig.suptitle("The Core Insight: Assembly Requires Complementarity, Not Similarity",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = ASSETS / "blog_similarity_vs_complementarity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Generating blog figures from existing experiment data...")
    print("=" * 60)

    fig_training_curve()
    fig_fold_comparison()
    fig_hard_negative_auc()
    fig_mining_distance()
    fig_similarity_vs_complementarity()

    print("\n" + "=" * 60)
    print("Done! All figures saved to assets/")
    print("=" * 60)
