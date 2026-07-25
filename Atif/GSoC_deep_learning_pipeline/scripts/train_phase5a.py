#!/usr/bin/env python3
"""CLI: train the Phase 5A PointNet encoder for one (or all) LOFO folds.

DO NOT RUN THIS AS PART OF ANY AUTOMATED SESSION. Per the project's current
instruction, the human runs training, not the coding AI. This script and
``phase5_encoder.train`` are written and unit-tested (plumbing only, on a
toy fixture -- see tests/phase5_encoder/test_sampler_mining.py::
test_train_one_fold_smoke) but have NOT been run on the real dataset.

------------------------------------------------------------------------
EXACT COMMAND (fold 1 first, per the project's existing pattern of gating
a full 7-fold commitment on one fold's measured wall-clock --
PHASE5_RESULTS_LOG.md:160, "Real epoch wall-clock measured in the fold-1
timing gate before committing to 7 retrains"):

    cd /home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline
    PYTHONPATH=src python3 scripts/train_phase5a.py \\
        --held-out fragment_caesar_fragment_1 \\
        --out-dir phase5a_runs

No ulimit is needed for this script specifically (it trains on the GPU;
host RAM usage is patch-table loading only, not the dense-cloud steps the
project's general ulimit warning targets). If CUDA is not available it
falls back to CPU automatically (see ``TrainConfig.device``) but that has
NOT been timed and will be far slower than the GPU estimate below --
confirm ``torch.cuda.is_available()`` prints True before trusting any
wall-clock estimate.

------------------------------------------------------------------------
EXPECTED FOLD-1 WALL-CLOCK (estimate, not yet measured on real data --
run the command above and compare against this before committing to the
other 6 folds):

Per epoch: 1 mining pass + ``steps_per_epoch`` (default 12, per
PHASE5_RESULTS_LOG.md:160's "~6-11 minibatches/epoch" for a ~5,800-patch
LOFO fold) training steps.
  * Mining pass: largest fold-1-relevant non-adjacent pool is F6/F7 at
    4,000 patches -> measured 3221.9 MB / ~125 ms
    (PHASE5A_TRAINING_DESIGN_REVISED.md §2b item 3; reproduce via
    ``scripts/measure_step_memory.py``). Fold 1 itself (holding out F1)
    draws from smaller per-anchor-fragment pools since F1 is a hub with 4
    neighbours -- expect faster than the F6/F7 worst case measured.
  * Training step: B_a=512, deduped hard negatives -> measured 1436.4 MB /
    ~96 ms/step (PHASE5A_TRAINING_DESIGN_REVISED.md §3).
  * Rough per-epoch estimate: 125 ms (mining) + 12 * 96 ms (steps) + one
    validation pass (small, pair-level val split is a few hundred pairs at
    most) ~= 1.3-1.5 seconds/epoch. Real measurement WILL differ -- this
    encoder has never been run against real patch data end to end, only
    against the toy fixture and synthetic memory-probe tensors. Treat the
    number above as a sanity check for "is this wildly wrong" (e.g. if
    fold 1 takes 60s/epoch, something is not matching the measured
    building blocks and should be investigated before running all 7).
  * At epoch_cap=50: rough estimate ~65-75 seconds total for fold 1 if
    patience does not fire earlier. Report the REAL number back before
    running folds 2-7.

------------------------------------------------------------------------
WHAT IT WRITES, WHERE:

  <out-dir>/fold_<held_out>_history.json
      Per-epoch: train_loss_mean, val_loss, improved (bool),
      epochs_without_improvement, mining_seconds, training_seconds,
      interface_draw_counts (per-interface draw count that epoch -- this
      is how to check whether the capped-inverse-frequency weighting
      (design §4) lifted F2-F4's ~72 contact patches / 1,232 positives into
      a useful draw rate or overshot into overfitting it: compare
      interface_draw_counts["F2-F4"] against interface_draw_counts for a
      large interface like "F6-F7" across epochs -- if F2-F4's count is
      persistently near-equal to or exceeding F6-F7's despite having ~174x
      fewer positives backing it, the cap is too permissive and should be
      tightened; if it stays near zero, the cap is too tight),
      per_fragment_mined_distance_mean (per-fragment mean mined
      hard-negative distance that epoch -- watch for F6/F7 anchors
      systematically mining closer/farther than F1-F4 anchors purely from
      the candidate-pool-size asymmetry in design §2b's table, not a
      geometric difference).
      Top level: "stopped_by" is EXACTLY "patience" or "epoch_cap" --
      never ambiguous. "stopped_at_epoch" is the 0-indexed epoch at which
      training stopped.

  <out-dir>/fold_<held_out>_best.pt
      The model state_dict at the LOWEST validation loss epoch (not
      necessarily the last epoch) -- this is the checkpoint to export
      through the diagnostic battery (item 5 of the resume plan;
      NOT done by this script).

------------------------------------------------------------------------
HOW TO TELL WHETHER PATIENCE FIRED OR THE CAP DID:

    python3 -c "
import json
h = json.load(open('phase5a_runs/fold_fragment_caesar_fragment_1_history.json'))
print('stopped_by:', h['stopped_by'])
print('stopped_at_epoch:', h['stopped_at_epoch'], '/', <epoch_cap>)
"

If ``stopped_by == 'patience'``, the pair-level validation loss actually
improved-then-plateaued -- report the run as "patience-stopped at epoch N."
If ``stopped_by == 'epoch_cap'``, the epoch cap did the actual stopping and
validation loss may still have been improving -- report the run as
"epoch-cap-stopped at epoch 49 (patience did not fire)," NOT as
"early stopped," per PHASE5A_TRAINING_DESIGN_REVISED.md §7's explicit
reporting requirement (the pair-level split's stated weakness is that
patience may never fire because held-out pairs' endpoints still appear in
many training pairs -- see the measured positive-pairs-per-patch
distribution in §7: median 347 other positives per contact patch).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from phase5_diagnostics.context import load_context
from phase5_diagnostics.probes import pose_invariance_check
from phase5_encoder.model import PointNetEncoder, build_encode_patch
from phase5_encoder.train import TrainConfig, train_one_fold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--held-out", required=True,
                     help="Fragment id to hold out for this LOFO fold, e.g. "
                          "fragment_caesar_fragment_1")
    ap.add_argument("--dataset-dir", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--patches-dir", default=os.path.join(ROOT, "patches"))
    ap.add_argument("--pairs-npz", default=os.path.join(ROOT, "phase3_final_review", "pairs.npz"))
    ap.add_argument("--pairs-dataset-json", default=os.path.join(ROOT, "pairs", "dataset.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "phase5a_runs"))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epoch-cap", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. Falling back to CPU. The "
              "wall-clock estimates in this script's docstring assume the "
              "measured RTX 4050 numbers and will NOT hold on CPU.",
              flush=True)

    print("Loading diagnostic context (Phase 1-3 artifacts)...", flush=True)
    ctx = load_context(
        dataset_dir=args.dataset_dir, patches_dir=args.patches_dir,
        pairs_npz=args.pairs_npz, pairs_dataset_json=args.pairs_dataset_json,
        repo_root=ROOT,
    )
    if args.held_out not in ctx.fragment_ids:
        print(f"ERROR: --held-out {args.held_out!r} not in {ctx.fragment_ids}",
              file=sys.stderr)
        return 1

    cfg = TrainConfig(
        batch_size=args.batch_size, epoch_cap=args.epoch_cap,
        patience=args.patience, seed=args.seed, out_dir=args.out_dir,
    )

    # HARD GATE (PHASE5_PREREGISTRATION.md condition 6): training must not
    # proceed if the untrained pipeline fails the pose-invariance check.
    print("Checking pose-invariance gate before training (hard blocker)...", flush=True)
    model = PointNetEncoder(out_dim=cfg.out_dim, pair_input=True)
    encode_patch = build_encode_patch(model, n_points=cfg.n_points, k=cfg.k, device="cpu")
    gate = pose_invariance_check(encode_patch, n_trials=50, n_points=100, tol=1e-3, seed=0)
    print(f"  pose gate: passed={gate['passed']} max_deviation={gate['max_deviation']:.3e}",
          flush=True)
    if not gate["passed"]:
        print("POSE GATE FAILED. Training must not proceed. Do NOT loosen the "
              "tolerance to force a pass -- escalate to the human.", file=sys.stderr)
        return 1

    print(f"Training fold held_out={args.held_out} "
          f"(device={cfg.device}, batch_size={cfg.batch_size}, "
          f"epoch_cap={cfg.epoch_cap}, patience={cfg.patience})...", flush=True)
    history = train_one_fold(ctx, args.held_out, cfg)

    print(f"\nDone. stopped_by={history['stopped_by']} "
          f"stopped_at_epoch={history['stopped_at_epoch']}", flush=True)
    print(f"History: {history['history_path']}")
    print(f"Checkpoint (best val): {history['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
