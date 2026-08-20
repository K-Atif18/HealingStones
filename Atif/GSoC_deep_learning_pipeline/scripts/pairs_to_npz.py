#!/usr/bin/env python3
"""Convert Phase 3 JSON pair files into a compact pairs.npz (parameterized).

Generalizes phase3_final_review/task7_storage_conversion.py so it works for any
Phase 3 output directory (e.g. the hand dataset), rather than the hardcoded
caesar 'pairs/' -> 'phase3_final_review/' paths.

NPZ schema (identical to the caesar pairs.npz consumed by Phase 4):
  fragment_vocab    : (V,) str      — fragment ID vocabulary
  fragment_A_idx    : (N,) int16    — index into fragment_vocab
  fragment_B_idx    : (N,) int16
  patch_A_ids       : (N,) int32
  patch_B_ids       : (N,) int32
  labels            : (N,) int8     — 1=positive, -1=random_neg, -2=hard_neg
  contact_overlap_A : (N,) float32
  contact_overlap_B : (N,) float32
  center_dist_mm    : (N,) float32

Usage:
  PYTHONPATH=src python3 scripts/pairs_to_npz.py \
      --pairs-dir pairs_hand --out pairs_hand/pairs.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

LABEL_POSITIVE = 1
LABEL_RANDOM_NEGATIVE = -1
LABEL_HARD_NEGATIVE = -2


def load_json_pairs(path: Path):
    if not path.exists():
        print(f"  {path.name}: missing, treated as empty")
        return []
    t0 = time.perf_counter()
    with open(path) as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    print(f"  {path.name}: {len(pairs):,} pairs ({time.perf_counter()-t0:.1f}s)")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Phase 3 JSON pairs to pairs.npz")
    ap.add_argument("--pairs-dir", required=True, help="Phase 3 output_dir containing *_pairs.json")
    ap.add_argument("--out", default=None, help="Output npz path (default: <pairs-dir>/pairs.npz)")
    args = ap.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_path = Path(args.out) if args.out else pairs_dir / "pairs.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = load_json_pairs(pairs_dir / "positive_pairs.json")
    neg = load_json_pairs(pairs_dir / "random_negative_pairs.json")
    hard = load_json_pairs(pairs_dir / "hard_negative_pairs.json")

    # Fragment vocabulary from all pairs.
    vocab = sorted(set(
        [p["fragment_A_id"] for p in pos] + [p["fragment_B_id"] for p in pos] +
        [p["fragment_A_id"] for p in neg] + [p["fragment_B_id"] for p in neg] +
        [p["fragment_A_id"] for p in hard] + [p["fragment_B_id"] for p in hard]
    ))
    frag_to_idx = {f: i for i, f in enumerate(vocab)}
    print(f"\nFragment vocabulary ({len(vocab)}):")
    for i, f in enumerate(vocab):
        print(f"  {i}: {f}")

    all_pairs, all_labels = [], []
    for pairs, lbl in [(pos, LABEL_POSITIVE), (neg, LABEL_RANDOM_NEGATIVE), (hard, LABEL_HARD_NEGATIVE)]:
        for p in pairs:
            all_pairs.append(p)
            all_labels.append(lbl)

    n = len(all_pairs)
    fA = np.empty(n, dtype=np.int16)
    fB = np.empty(n, dtype=np.int16)
    pA = np.empty(n, dtype=np.int32)
    pB = np.empty(n, dtype=np.int32)
    labels = np.array(all_labels, dtype=np.int8)
    ovA = np.zeros(n, dtype=np.float32)
    ovB = np.zeros(n, dtype=np.float32)
    dists = np.zeros(n, dtype=np.float32)

    for i, p in enumerate(all_pairs):
        fA[i] = frag_to_idx[p["fragment_A_id"]]
        fB[i] = frag_to_idx[p["fragment_B_id"]]
        pA[i] = int(p["patch_A_id"])
        pB[i] = int(p["patch_B_id"])
        ovA[i] = float(p.get("contact_overlap_A", 0.0))
        ovB[i] = float(p.get("contact_overlap_B", 0.0))
        cA = p.get("patch_A_center")
        cB = p.get("patch_B_center")
        if cA is not None and cB is not None:
            dists[i] = float(np.linalg.norm(np.asarray(cA) - np.asarray(cB)))

    print(f"\nTotal pairs: {n:,}  "
          f"(pos={int((labels==1).sum()):,}, "
          f"random_neg={int((labels==-1).sum()):,}, "
          f"hard_neg={int((labels==-2).sum()):,})")

    np.savez_compressed(
        out_path,
        fragment_vocab=np.array(vocab),
        fragment_A_idx=fA,
        fragment_B_idx=fB,
        patch_A_ids=pA,
        patch_B_ids=pB,
        labels=labels,
        contact_overlap_A=ovA,
        contact_overlap_B=ovB,
        center_dist_mm=dists,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB)")

    # Integrity re-load.
    loaded = np.load(out_path, allow_pickle=True)
    assert len(loaded["labels"]) == n
    print("Integrity check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
