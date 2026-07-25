#!/usr/bin/env python3
"""CLI: train the Phase 5A PointNet encoder for one (or all) LOFO folds.

DO NOT RUN THIS AS PART OF ANY AUTOMATED SESSION. Per the project's current
instruction, the human runs training, not the coding AI. This script and
``phase5_encoder.train`` are written and unit-tested (plumbing only, on a
toy fixture -- see tests/phase5_encoder/test_sampler_mining.py::
test_train_one_fold_smoke) but have NOT been run end-to-end on the real
dataset (the individual building blocks -- feature extraction, forward,
backward, mining -- HAVE been measured on real timing/memory probes; see
the wall-clock section below).

------------------------------------------------------------------------
REQUIRED ENVIRONMENT VARIABLE -- measured necessary this session, not
optional: a fresh `python3` process on this machine's RTX 4050 hit spurious
CUDA OOM errors at batch sizes and pool sizes that were previously measured
safe (e.g. B_a=256 with zero hard negatives OOM'd, despite `nvidia-smi`
showing only 15 MiB used out of 6141 MiB total) -- this was PyTorch
allocator fragmentation, not a real capacity limit; confirmed because
setting the flag below made the exact same code succeed at the exact same
previously-measured numbers (1436.4 MB @ B_a=512, etc.). Always export this
before running:

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

------------------------------------------------------------------------
EXACT COMMAND (fold 1 first, per the project's existing pattern of gating
a full 7-fold commitment on one fold's measured wall-clock --
PHASE5_RESULTS_LOG.md:160, "Real epoch wall-clock measured in the fold-1
timing gate before committing to 7 retrains"). Use --max-epochs 1 for a
first real-data smoke run before committing to the full epoch_cap:

    cd /home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONPATH=src python3 scripts/train_phase5a.py \\
        --held-out fragment_caesar_fragment_1 \\
        --out-dir phase5a_runs \\
        --max-epochs 1

If that one-epoch run's wall-clock and per_fragment_mined_distance_mean /
interface_draw_counts fields look sane (compare against the estimate
below), drop --max-epochs and rerun for the real epoch_cap=50.

No ulimit is needed for this script specifically (it trains on the GPU;
host RAM usage is patch-table loading only, not the dense-cloud steps the
project's general ulimit warning targets) -- and using `ulimit -v` around
a CUDA-initializing process is actively harmful: it was observed this
session to break CUDA driver initialization outright ("CUDA driver error:
out of memory" on `.to('cuda')` with a virtual-memory cap that had nothing
to do with GPU memory). If CUDA is not available it falls back to CPU
automatically (see ``TrainConfig.device``) but that has NOT been timed and
will be far slower than the GPU estimate below -- confirm
``torch.cuda.is_available()`` prints True before trusting any wall-clock
estimate.

------------------------------------------------------------------------
EXPECTED FOLD-1 WALL-CLOCK -- corrected estimate, building blocks measured
directly on real GPU timing (not the toy fixture), full derivation below.
An EARLIER version of this estimate (1.3-1.5 s/epoch) was WRONG: it omitted
the validation pass's cost, assumed feature-extraction time was
negligible, and used the wrong per-call encode cost. Corrected numbers,
each measured directly in a real Python process on this machine's RTX 4050
with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` set (measurement
included both the Python-loop PPF feature extraction AND the GPU forward/
backward, since the ORIGINAL estimate's blind spot was treating feature
extraction as free):

  * Training step (B_a=512 anchors + 512 positives, InfoNCE forward+
    backward): feature extraction ~873 ms + forward+backward ~250 ms =
    **~1123 ms/step**. At ``steps_per_epoch=12``: **~13.5 s** of the epoch.
    (Feature extraction dominates -- it is a per-patch Python loop calling
    ``knn_ppf_features``, not a batched GPU op; this is the same
    Python-loop cost the earlier draft flagged as a risk for the
    validation pass, but it applies equally to every training step and
    was NOT counted for training steps in the original estimate either.)
  * Mining pass (largest fold-1-relevant non-adjacent pool, F6/F7-sized at
    4,000 patches, no_grad): feature extraction ~3252 ms + forward
    ~136 ms = **~3.4 s**, once per epoch.
  * Validation pass (499 pairs, capped -- see the Deviation-1/2 fixes
    below -- chunked into ceil(998/256)=4 minibatches of 256): ~211 ms per
    minibatch x 4 = **~0.84 s**, once per epoch. (This is now the
    SMALLEST of the three terms, not the dominant one -- the original
    concern was the validation set being 49,569 pairs unc capped, which
    WOULD have dominated; capped at 500 it does not.)
  * **Per-epoch total: 3.4s (mining) + 13.5s (12 steps) + 0.84s (val) =
    ~17.7 seconds/epoch.**
  * At ``epoch_cap=50`` (if patience never fires): **~885 seconds
    (~14.8 minutes)** for fold 1 alone. All 7 folds at the cap: ~1.7
    hours. If patience fires around epoch 10 (a guess, not a measurement):
    ~3 minutes for fold 1.
  * This is a real, direct measurement of each component, not an
    extrapolation from a different quantity -- but it has NOT been run as
    the actual `train_one_fold` loop against real patch data end-to-end
    (only the toy-fixture smoke test has). Treat ~17.7s/epoch as the
    number to sanity-check the real `--max-epochs 1` run against; if the
    real run differs by more than ~2x, investigate before running longer.

------------------------------------------------------------------------
WHAT IT WRITES, WHERE:

  <out-dir>/fold_<held_out>_history.json
      Per-epoch: train_loss_mean, val_loss, improved (bool),
      epochs_without_improvement, mining_seconds, training_seconds,
      interface_draw_counts (per-interface draw count that epoch -- this
      is how to check whether the corrected capped-inverse-frequency
      weighting (design §4, semantics fixed this session -- see
      ``sampler.InterfaceWeighting``) lifted F2-F4's ~72 contact patches /
      1,232 positives into a useful draw rate without letting it dominate:
      compare interface_draw_counts["F2-F4"] against interface_draw_counts
      for a large interface like "F6-F7" across epochs -- F2-F4 should be
      MORE represented than its raw 0.17% pair share, but should NOT be
      the majority of draws; the corrected default (max_ratio=5.0, bounded
      relative to uniform) measured 24.65% of draws for F2-F4 on real
      fold-1 data (1.48x uniform), not the 55.4% the pre-fix code
      produced),
      deduped_hard_negatives_per_step_mean/min/max (Deviation 2's second
      requirement: the ACTUAL number of distinct hard negatives per step
      after dedup, since mining one negative per CONTACT PATCH rather than
      per DRAW means most of a 512-draw batch resolves to a much smaller
      distinct set -- this is now a reported number, not an assumed one;
      check it against the batch size to see the real ratio),
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

------------------------------------------------------------------------
BUGS FOUND AND FIXED BEFORE ANY RUN (context for reviewers, not required
reading to operate the script):

1. Validation set was uncapped: applying val_fraction=0.10 to raw
   per-interface pair-row counts gave 49,569 validation pairs on real fold
   1 (F6-F7 alone contributing 21,445), because the fraction was applied
   to counts dominated by a 174x interface-size spread, not to a bounded
   patch/pair budget. This would have OOM'd (a ~9.8 GB fp32 similarity
   matrix) or, had it not, dominated the entire epoch wall-clock. Fixed:
   ``build_fold_sampler`` now takes ``val_pair_cap`` (default 500) and
   downsamples proportionally across interfaces; ``_validation_loss`` also
   chunks into ``val_minibatch_size`` (default 256) minibatches so no
   single validation step's cost scales with the total.
2. Interface weighting inverted: the original ``max_ratio`` clamp bounded
   weight against the WRONG reference point (the largest interface's
   inverse weight, not uniform), so on fold 1's real 174x count spread the
   rarest interface (F2-F4) received 55.4% of ALL draws -- worse than the
   overfitting failure §4 was written to prevent. Fixed: see
   ``sampler.InterfaceWeighting``'s corrected semantics (clamps effective
   COUNTS, not weights, relative to the largest interface's real count);
   measured on real fold-1 data at the new default (max_ratio=5.0): F2-F4
   now gets 24.65% (1.48x uniform), not 55.4%.
3. Hard-negative dedup count was invisible: mining produces one negative
   per CONTACT PATCH, not per draw, so a 512-draw batch typically dedupes
   to far fewer than 512 distinct hard negatives -- now logged per step
   (``deduped_hard_negatives_per_step_mean/min/max``) instead of assumed.
4. In-batch positives were unmasked: cross_entropy over raw similarities
   treated every non-target candidate as a negative, including genuine
   positive partners that land in the same batch (median 347 positives
   per contact patch -- common at these draw rates, not rare). Fixed:
   ``_info_nce_step`` now masks any candidate that is a true positive
   partner of a given anchor (per ``_build_positive_partner_lookup``) out
   of that anchor's denominator, except the actual target column.

All four were caught by direct measurement against real fold-1 data BEFORE
any training run, per this project's standing rule that every design
number must be measured, not assumed.
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
    ap.add_argument("--max-epochs", type=int, default=None,
                     help="Override --epoch-cap for a quick first real-data "
                          "run (e.g. --max-epochs 1) to sanity-check "
                          "wall-clock against the estimate in this script's "
                          "docstring before committing to the full cap.")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-pair-cap", type=int, default=500,
                     help="Hard cap on total validation pairs (see the "
                          "'BUGS FOUND AND FIXED' section above -- do not "
                          "raise this casually, it was 49,569 uncapped).")
    ap.add_argument("--interface-weight-max-ratio", type=float, default=5.0,
                     help="See sampler.InterfaceWeighting's corrected "
                          "semantics: bounds a rare interface's weight to "
                          "at most this many times uniform, not relative "
                          "to the largest interface's raw inverse weight.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if torch.cuda.is_available() and "expandable_segments" not in alloc_conf:
        print("WARNING: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is "
              "not set. This was measured necessary this session to avoid "
              "spurious CUDA OOM errors from allocator fragmentation at "
              "batch sizes that are otherwise safe (see this script's "
              "docstring). Strongly recommend: "
              "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
              flush=True)

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

    epoch_cap = args.max_epochs if args.max_epochs is not None else args.epoch_cap
    cfg = TrainConfig(
        batch_size=args.batch_size, epoch_cap=epoch_cap,
        patience=args.patience, seed=args.seed, out_dir=args.out_dir,
        val_pair_cap=args.val_pair_cap,
        interface_weight_max_ratio=args.interface_weight_max_ratio,
    )
    if args.max_epochs is not None:
        print(f"NOTE: --max-epochs {args.max_epochs} overrides --epoch-cap "
              f"{args.epoch_cap} for this run (dry-run-style first "
              f"execution).", flush=True)

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
