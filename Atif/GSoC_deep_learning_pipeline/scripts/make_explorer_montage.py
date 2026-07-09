#!/usr/bin/env python3
"""Tile the 7 interactive per-fragment window captures into one labelled figure.

These captures come from ``scripts/explore_fragment.py`` (Open3D / matplotlib
windows). They are user-provided screenshots, not regenerated here; this script
just arranges the existing PNGs in ``assets/`` into a single montage that the
README references as Fig 1..7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# (filename, "Fig N", short caption)
PANELS = [
    ("4a_viz1.png", "Fig 1", "Distinctiveness — Fragment 1"),
    ("4b_viz2.png", "Fig 2", "Contact regions — Fragment 1"),
    ("4c_viz3.png", "Fig 3", "Retrieval (FPFH) — Fragment 1"),
    ("4d_viz4.png", "Fig 4", "Hard negatives (FPFH) — Fragment 1"),
    ("4e_viz5.png", "Fig 5", "Registration assembled around Fragment 1"),
    ("4e_viz5_different_frag.png", "Fig 6", "Registration assembled around Fragment 2"),
    ("4f_viz7.png", "Fig 7", "Adjacency graph — Fragment 1"),
]


def main():
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.ravel()
    for ax, (fname, tag, cap) in zip(axes, PANELS):
        path = ASSETS / fname
        if not path.exists():
            ax.text(0.5, 0.5, f"missing:\n{fname}", ha="center", va="center")
            ax.axis("off")
            continue
        ax.imshow(mpimg.imread(path))
        ax.set_title(f"{tag} — {cap}", fontsize=13, fontweight="bold")
        ax.axis("off")
    # last (8th) cell: legend / how-to
    axes[-1].axis("off")
    axes[-1].text(
        0.02, 0.95,
        "Interactive per-fragment explorer\n"
        "scripts/explore_fragment.py --fragment N\n\n"
        "Each window opens on its own (rotatable 3D for Fig 1-6,\n"
        "2D graph for Fig 7) and prints a 'what to look for'\n"
        "checklist to the terminal. Colours:\n\n"
        "Fig 1  red=distinctive, yellow=moderate,\n"
        "        blue=ambiguous, gray=flat\n"
        "Fig 2  each colour = contact with one neighbour\n"
        "Fig 3  black=query, green=true partner, red=false positive\n"
        "Fig 4  blue=this fragment, orange=non-adjacent look-alike\n"
        "Fig 5-6 gray=anchor, colours=neighbours re-assembled\n"
        "Fig 7  gold=this fragment, orange=its neighbours",
        fontsize=11, va="top", family="monospace",
    )
    fig.suptitle("Phase 4 — interactive per-fragment validation windows (example: Fragment 1)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = ASSETS / "phase4_explorer_windows.png"
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
