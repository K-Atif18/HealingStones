#!/usr/bin/env python3
"""Measure real per-training-step GPU memory (PHASE5A_TRAINING_DESIGN_REVISED.md item 3).

PHASE5_RESULTS_LOG.md:155 measured memory for B independently *encoded* patches
(no positives/negatives structure). A real training step forwards
``B_a anchors + B_a positives + (deduped hard negatives)`` through the same
encoder before computing the InfoNCE loss and backpropagating. This script
measures that composition directly, with hard negatives deduplicated within
the batch (the mined pool is small -- see PHASE5A_TRAINING_DESIGN_REVISED.md
item 2b -- so within-batch sharing is expected to be nearly free), at a range
of candidate anchor batch sizes.

Run:
    cd <repo>
    PYTHONPATH=src python3 scripts/measure_step_memory.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from phase5_encoder.model import PointNetEncoder
from phase5_encoder.features import PAIR_FEATURE_DIM


def make_batch(n_anchors: int, n_points: int, k: int, n_hard_distinct: int, device: str):
    """Synthetic (N,k,4) feature batch mimicking real PPF-feature shape.

    ``n_hard_distinct`` is the number of *distinct* hard-negative patches
    encoded once and shared across all anchors in the step (dedup), rather
    than one hard negative per anchor.
    """
    anchors = torch.randn(n_anchors, n_points, k, PAIR_FEATURE_DIM, device=device)
    positives = torch.randn(n_anchors, n_points, k, PAIR_FEATURE_DIM, device=device)
    hard_neg = torch.randn(n_hard_distinct, n_points, k, PAIR_FEATURE_DIM, device=device)
    return anchors, positives, hard_neg


def info_nce_like_loss(anc: torch.Tensor, pos: torch.Tensor, hard: torch.Tensor,
                        temperature: float = 0.07) -> torch.Tensor:
    """A representative (not final) InfoNCE-shaped loss: enough to trigger a
    real backward pass with the right memory shape (full similarity matrix
    against positives + all distinct hard negatives + in-batch randoms)."""
    # in-batch randoms: every other anchor's positive acts as a random negative.
    candidates = torch.cat([pos, hard], dim=0)          # (B_a + H_distinct, D)
    sim = anc @ candidates.t() / temperature            # (B_a, B_a + H_distinct)
    targets = torch.arange(anc.shape[0], device=anc.device)
    return torch.nn.functional.cross_entropy(sim, targets)


def measure(n_anchors: int, n_points: int, k: int, n_hard_distinct: int,
            device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None

    model = PointNetEncoder(out_dim=64, pair_input=True).to(device)
    model.train()

    anc_feat, pos_feat, hard_feat = make_batch(n_anchors, n_points, k, n_hard_distinct, device)

    t0 = time.time()
    anc_emb = model(anc_feat)
    pos_emb = model(pos_feat)
    hard_emb = model(hard_feat) if n_hard_distinct > 0 else torch.empty(0, model.out_dim, device=device)

    loss = info_nce_like_loss(anc_emb, pos_emb, hard_emb)
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.time() - t0) * 1000.0

    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device == "cuda" else float("nan")
    total_patch_forwards = n_anchors * 2 + n_hard_distinct
    return {
        "n_anchors": n_anchors, "n_points": n_points, "k": k,
        "n_hard_distinct": n_hard_distinct,
        "total_patch_forwards": total_patch_forwards,
        "peak_mb": peak_mb, "step_ms": dt_ms,
    }


def measure_mining_pass(pool_size: int, n_points: int, k: int, device: str) -> dict:
    """Measure a single no_grad mining-pass encode over ``pool_size`` candidate
    patches (PHASE5A_TRAINING_DESIGN_REVISED.md §2b item 3). This mirrors the
    once-per-epoch mining pass: encode the fold's non-adjacent candidate pool
    once under no_grad/eval, cache the embedding matrix, no backward pass."""
    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = PointNetEncoder(out_dim=64, pair_input=True).to(device)
    model.eval()
    feats = torch.randn(pool_size, n_points, k, PAIR_FEATURE_DIM, device=device)
    t0 = time.time()
    with torch.no_grad():
        emb = model(feats)
    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.time() - t0) * 1000.0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device == "cuda" else float("nan")
    return {"pool_size": pool_size, "peak_mb": peak_mb, "time_ms": dt_ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"device={args.device}  n_points={args.n_points}  k={args.k}")
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}  total_mem={props.total_memory/1e6:.0f} MB")

    # Sweep anchor batch size x deduped hard-negative pool size, mirroring
    # the composition the training loop will actually forward per step:
    #   B_a anchors + B_a positives + n_hard_distinct deduped hard negatives.
    # See PHASE5A_TRAINING_DESIGN_REVISED.md §3 for the measured table this
    # sweep reproduces (B_a=512 with deduped hard negatives is the chosen
    # operating point: 1436.4 MB peak, ~96 ms/step).
    configs = [
        (128, 0), (256, 0),
        (512, 0), (512, 256), (512, 512),
        (1024, 0), (1024, 256),
        # Non-deduped comparison: h=4 hard negatives PER anchor (the
        # rejected design's ratio), i.e. n_hard_distinct = n_anchors * 4.
        (512, 512 * 4), (1024, 1024),
        # Boundary probes.
        (2048, 0), (2048, 2048), (3072, 0),
    ]
    print(f"\n{'B_a':>6} {'H_distinct':>10} {'total_fwd':>10} {'peak_MB':>10} {'step_ms':>10}")
    for n_anchors, n_hard in configs:
        try:
            r = measure(n_anchors, args.n_points, args.k, n_hard, args.device)
        except RuntimeError as e:
            print(f"{n_anchors:>6} {n_hard:>10} {'OOM':>10} {'--':>10} {'--':>10}   ({e})")
            torch.cuda.empty_cache()
            continue
        print(f"{r['n_anchors']:>6} {r['n_hard_distinct']:>10} "
              f"{r['total_patch_forwards']:>10} {r['peak_mb']:>10.1f} {r['step_ms']:>10.1f}")

    # Mining-pass measurement: one no_grad forward over each fragment's real
    # non-adjacent candidate pool size (PHASE5A_TRAINING_DESIGN_REVISED.md
    # §2b table: F1/F4=1822, F5=2000, F2/F3=2822, F6/F7=4000).
    print(f"\n{'pool_size':>10} {'peak_MB':>10} {'time_ms':>10}   (no_grad mining pass)")
    for pool_size in (1822, 2000, 2822, 4000):
        try:
            m = measure_mining_pass(pool_size, args.n_points, args.k, args.device)
        except RuntimeError as e:
            print(f"{pool_size:>10} {'OOM':>10} {'--':>10}   ({e})")
            torch.cuda.empty_cache()
            continue
        print(f"{m['pool_size']:>10} {m['peak_mb']:>10.1f} {m['time_ms']:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
