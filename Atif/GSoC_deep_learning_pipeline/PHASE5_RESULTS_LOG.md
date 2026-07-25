# Phase 5 Results Log

**Single source of truth** for Phase 5 measurements. The numbers sections are
**generated** from `phase5_diagnostics_results/baseline_diagnostics.json` by
`scripts/generate_results_log.py` -- do not hand-edit numbers here; regenerate.
Prose (deviations, decisions) is maintained in the generator template.

- Generated: `2026-07-25T15:49:31.170129+00:00`
- Source JSON `run_timestamp`: `2026-07-23T16:53:22.637596+00:00`
- Git commit at generation: `d166346`
- Fragments: F1, F2, F3, F4, F5, F6, F7
- Total patches: 6822 | Adjacent pairs: 11

---

## (a) Pre-registration thresholds (frozen)

The pass/fail thresholds are fixed in `PHASE5_PREREGISTRATION.md` **before** any
encoder was measured and are **not** duplicated or weakened here. Summary of the
six-condition Level-3 success definition (see that file for exact wording):

1. Beat FPFH on the honest headline (macro mAP, contact-only gallery) by the §1 margin rule.
2. Win on the majority of well-supported LOFO folds **{F1,F2,F3,F4}**, both P@1 and mAP,
   via a **paired bootstrap** on per-query differences (encoder - FPFH) whose CI excludes zero.
3. **Shrink** the easy-vs-hard AUC gap vs FPFH.
4. Show a **positive** neighbour contact-vs-noncontact ranking gap.
5. Pass the **pose-invariance gate** (`passed == True`, tol=1e-3).
6. Not exhibit a dominant fragment-ID / boundary-openness signal that fully explains the gain.

Well-supported fold set for pass/fail = **F1,F2,F3,F4**. F5 = signal-ambiguous
(reported, not counted). F6/F7 = low-confidence (F7 shows genuine degeneracy).

---

## (b) FPFH baseline numbers (pre-encoder, dated above)

Provenance: `scripts/run_phase5_diagnostics.py` -> `baseline_diagnostics.json`. All numbers below are read from that JSON.

### Ranking (raw + contact-only gallery)

| source | raw mAP | raw P@1 | contact mAP | contact P@1 | macro mAP |
| --- | --- | --- | --- | --- | --- |
| random | 0.0615 | 0.0488 | 0.1022 | 0.0800 | 0.0668 |
| centroid | 0.0512 | 0.0025 | 0.0868 | 0.0075 | 0.0538 |
| fpfh | 0.1092 | 0.1187 | 0.1490 | 0.1400 | 0.1372 |

> Honest headline metric = **macro mAP, contact-only gallery**, read relative to the random null floor.

### Null floor (harness generosity)

| source | mAP | P@1 |
| --- | --- | --- |
| random | 0.0615 | 0.0488 |
| centroid | 0.0512 | 0.0025 |

### Shortcut battery (FPFH)

| probe | value | note |
| --- | --- | --- |
| fragment-ID kNN acc (pooled) | 0.8920 | chance 0.1429 |
| contact-vs-noncontact AUC | 0.7311 |  |
| neighbour contact gap | 0.0734 | positive = real interface signal |
| distinctiveness-sim Pearson r | -0.4004 |  |
| boundary-openness R^2 | 0.0021 |  |

### Hard-negative stratification (FPFH)

| metric | value |
| --- | --- |
| AUC pos-vs-easy | 0.7063 |
| AUC pos-vs-hard | 0.0922 |
| easy-minus-hard gap | 0.6141 |

> The 0.61 gap is FPFH's sharpest pathology: it separates positives from random negatives but collapses on look-alike non-adjacent patches. The encoder must **shrink** this (condition 3).

### Acquisition fingerprint (source-independent)

- Fragment-ID from density/spacing alone: **0.3555** (chance 0.1429). Mandates density normalisation + jitter in the encoder input pipeline.

### LOFO per fold with 95% bootstrap CIs (FPFH vs random)

| fold | queries | neighbors | FPFH P@1 | FPFH CI | RND P@1 | RND CI |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | 686 | 4 | 0.168 | [0.140, 0.195] | 0.071 | [0.054, 0.090] |
| F2 | 516 | 3 | 0.194 | [0.161, 0.229] | 0.064 | [0.045, 0.085] |
| F3 | 573 | 3 | 0.169 | [0.140, 0.199] | 0.035 | [0.021, 0.051] |
| F4 | 544 | 4 | 0.184 | [0.151, 0.219] | 0.077 | [0.057, 0.101] |
| F5 | 661 | 4 | 0.074 | [0.056, 0.094] | 0.054 | [0.038, 0.073] |
| F6 | 780 | 2 | 0.032 | [0.021, 0.045] | 0.064 | [0.047, 0.082] |
| F7 | 347 | 2 | 0.000 | [0.000, 0.000] | 0.081 | [0.055, 0.112] |

### Held-out fragment-ID probe -- sharper instrument for **condition 6**

Provenance: `phase5_diagnostics.probes.heldout_fragment_id_probe`, field `per_source.<src>.heldout_fragment_id_probe`. **Not a new condition** -- a precise lens on condition 6 (generalized shape-signature leakage vs literal memorization). Baselined on FPFH *before* any encoder exists.

| held-out | base rate | FPFH self-retr. | ×base | RND self-retr. |
| --- | --- | --- | --- | --- |
| F1 | 0.147 | 0.895 | 6.11 | 0.168 |
| F2 | 0.147 | 0.876 | 5.98 | 0.165 |
| F3 | 0.147 | 0.935 | 6.38 | 0.142 |
| F4 | 0.147 | 0.850 | 5.80 | 0.126 |
| F5 | 0.147 | 0.979 | 6.68 | 0.138 |
| F6 | 0.120 | 1.000 | 8.30 | 0.122 |
| F7 | 0.147 | 0.997 | 6.80 | 0.157 |

> FPFH isolates an unseen fragment at 6-8× base rate; random sits at base rate (probe validity check). The encoder must drive this **down** toward base rate if it learns interface association, not identity.


---

## (c) Architecture deviations (with measurements)

Deviations from the literal `CODING_AI_RESUME_CONTEXT.md` "Immediate next step"
wording. Each was flagged for sign-off before implementation, with a measurement.

### Deviation 1 -- fragment-balanced sampling; adversarial term deferred to 5B
- **Decision:** adopt fragment-balanced sampling in 5A training; do **not** add a
  fragment-adversarial (gradient-reversal) loss term yet.
- **Why:** proper per-fold LOFO retraining (a fold-`f` model never sees `f`)
  structurally prevents *literal* identity memorization; density normalization
  closes the acquisition back-door; balanced sampling stops large fragments
  dominating the contrastive loss. An adversarial term adds unstable,
  hyperparameter-heavy machinery to what must be the simplest baseline first.
- **Pre-committed trigger to add it in 5B:** the trained encoder's **held-out
  fragment-ID probe stays high AND** well-supported LOFO folds fail to beat FPFH
  despite training. (The trigger's probe is the held-out variant, not the pooled one.)

### Deviation 2 (refined) -- PPF rotation-invariant features over PCA-frame canonicalization
- **Decision:** feed the PointNet rotation+translation-invariant Point Pair
  Features instead of canonicalizing each patch into its PCA/SHOT local frame.
- **Why (scientific justification):** to **avoid a confound in interpreting a
  condition-1 failure.** With PCA canonicalization, a Phase-5A failure could not
  be distinguished between "PointNet cannot learn interface association" (a real
  negative result) and "the input frame was unstable / discarded pairwise
  structure" (an uninteresting confound). PPF removes that ambiguity. This is
  **not** a generic "richer features are better" argument.
- **Measurements (same unmodified pose gate, tol=1e-3):**
  - PCA/SHOT-frame canonicalization: `passed=False`, **max_dev = 4.4991** (fails by ~4500x).
  - PPF features + untrained PointNet: `passed=True`, **max_dev = 0.000e+00** (exact).
- **Refinement (approved, two standing conditions):** upgrade single-reference PPF
  to **k-NN pairwise PPF** (per-point neighbourhood features, `(N,k,4)`) so the
  input is not representationally weaker than FPFH's local histograms. Conditions:
  (1) re-measure the pose gate and report `max_dev`; (2) report `N×k×4` cost and
  confirm it fits the RTX 4050 budget (folded into the fold-1 timing step).
- **Refinement status: LANDED and verified.**
  - Feature: `phase5_encoder.features.knn_ppf_features` -> `(N, k, 4)`, k=16.
    Classic Drost/PPFNet 4-tuple per (point, neighbour); exactly rigid-invariant
    (verified `tests/phase5_encoder/test_knn_ppf_features_are_rigid_invariant`,
    matches to atol 1e-9 under random SO(3)+t).
  - **Condition 1 (re-measured gate):** k-NN pairwise PPF + untrained PointNet
    (pair path) through the unmodified `pose_invariance_check`, tol=1e-3:
    `passed=True`, **max_dev = 0.000e+00** (exact). Test:
    `tests/phase5_encoder/test_pose_gate.py::test_pose_invariance_gate_untrained_encoder`.
  - **Condition 2 (cost on RTX 4050, N=64, k=16, D=4):** measured fwd+bwd peak GPU:
    B=256 -> 374 MB (195 ms); B=1024 -> 1444 MB (57 ms); B=4096 -> **OOM** (only
    ~1 GB free of 5.64 GB total due to background processes). Activation memory
    scales with `B·N·k` (the k=16 factor is the cost of the swap). Encoder itself
    is tiny: **83,200 params (0.33 MB)**. **Verdict:** fits with a **batch-size cap
    of 512-1024**; a LOFO fold (~5,800 patches) is ~6-11 minibatches/epoch. Training
    will use batch 512 for headroom. Real epoch wall-clock measured in the fold-1
    timing gate before committing to 7 retrains.

---

## (f) Training design decisions

Decisions finalised in `PHASE5A_TRAINING_DESIGN_REVISED.md` (round 2,
approved), implemented in `src/phase5_encoder/{sampler,mining,train}.py` and
`scripts/{measure_step_memory,train_phase5a}.py`. Training itself has **not**
been run yet -- this section records the *design*, not results; section (d)
remains the place encoder results land once a fold actually trains.

### Loss
- **InfoNCE-style contrastive loss** over (anchor, positive, hard negatives,
  in-batch randoms), retained unchanged from the original (accepted) design
  -- see `PHASE5A_TRAINING_DESIGN_REVISED.md` header note ("Points 1, 3, 4 of
  the *original* design ... are retained unchanged").

### Hard-negative mechanism
- **Primary:** on-the-fly, per-epoch, fold-restricted mining under the
  encoder's *current* embedding (`src/phase5_encoder/mining.py::
  mine_epoch_hard_negatives`), targeting the measured FPFH pathology
  directly (easy-vs-hard AUC gap = 0.6141, `PHASE5_RESULTS_LOG.md`
  hard-negative-stratification table above).
- **Measured supply of the RETIRED alternative** (why it was retired):
  the FPFH-mined evaluation-only set (`ranking.mine_hard_negatives`) yields
  2,100 mined pairs, 2,100 distinct anchors, **max 1 hard negative per
  anchor** (strict `argmin`, `ranking.py:381`), covering only 1,285/2,100
  contact anchors (29-36% of contact patches per fragment). This set is
  frozen and evaluation-only (condition 3), never training-visible.
- **Deduplication:** hard negatives are deduplicated within each training
  batch before encoding (`train.py::_info_nce_step`) -- verified cheap:
  `B_a=512` with up to 512 deduped distinct hard negatives peaks at
  **1436.4 MB**, see the memory table below.

### Mining frequency and candidate-pool policy
- **Frequency: N = 1 epoch** (not per-step). Justification: a LOFO fold is
  ~11-12 minibatches/epoch at `B_a=512`; per-step mining would re-encode the
  full non-adjacent pool ~11-12x more often than per-epoch mining, for a
  representation that only moves incrementally within an epoch.
- **Candidate pool: full non-adjacent pool per fragment, no subsampling.**
  Preserves the real per-fragment pool-size asymmetry (measured via
  `ctx.neighbors_of`/`ctx.patch_count`): F1/F4=1,822, F5=2,000, F2/F3=2,822,
  **F6/F7=4,000** -- subsampling to a common size would erase the asymmetry
  the per-fragment difficulty logging exists to surface.
- **`no_grad` + cached table:** confirmed mandatory and measured (not
  assumed): one no_grad forward per fragment's candidate pool, per epoch.
  Measured peak memory scales with pool size and EXCEEDS a training step's
  memory for the largest pools (F6/F7 at 4,000 -> 3221.9 MB vs a training
  step's 1436.4 MB) -- corrects an earlier assumption in the design doc that
  mining would be cheaper; safe only because mining and a training step
  never run concurrently (verified: `torch.cuda.empty_cache()` between the
  mining pass and the resumed training loop).
- **Per-fragment difficulty logging:** implemented
  (`MinedNegatives.per_fragment_distance`, surfaced per-epoch in
  `train_one_fold`'s history JSON as `per_fragment_mined_distance_mean`) --
  specifically to catch an F6/F7-driven mining-difficulty artifact before it
  is mistaken for a geometric finding in downstream LOFO results.

### Interface weighting rule
- **Sampler: fragment -> interface -> partner**, not fragment-uniform
  (`src/phase5_encoder/sampler.py::FoldSampler.sample_batch`).
- **Weighting: capped inverse-frequency** (`InterfaceWeighting`,
  `max_ratio=20.0` default), NOT pure inverse-frequency. Reason: pure
  inverse-frequency over the measured per-interface positive-pair counts
  (F6-F7: 214,446 ... F2-F4: 1,232 -- a 174x spread, `phase3_final_review/
  pairs.npz`, `labels==1` grouped by `fragment_A_idx`/`fragment_B_idx`) would
  give F2-F4 ~174x the weight of F6-F7, oversampling its small contact-patch
  set (~72 patches) into overfitting rather than merely correcting the
  imbalance. The cap is a decision, not a measurement -- 20x was chosen as a
  bound loose enough to still substantially favour small interfaces without
  the unbounded 174x extreme; this ratio is a candidate for later tuning
  based on the per-epoch `interface_draw_counts` logged during training.
- **Multi-interface rule:** a draw's positive comes ONLY from the interface
  selected that step, even for patches belonging to more than one interface
  (measured: F1 has 23.9% multi-interface contact patches, F7 has 0%,
  `ctx.interface_patch_ids`) -- implemented and unit-tested
  (`test_multi_interface_patches_detected`).
- **n_well_supported_interfaces = 10, not 11** (F5-F7 has 0 contact patches
  on both sides).

### Positive-distance policy
- **Uncapped -- all positives used**, faithful to the Level-3 co-membership
  label as declared in `PHASE5_PREREGISTRATION.md`. Acknowledged tension
  (not hidden): positive-pair centre distances (`center_dist_mm`,
  `phase3_final_review/pairs.npz`) have median 18.64 mm, p90 33.34 mm, max
  69.66 mm, against an 8 mm patch radius -- only 11.2% of positives have
  centres within one radius. The loss optimizes a **region-scale**
  objective while the pass bar is judged partly by **P@1**, a top-1 metric.
  A distance-capped positive variant is a free-to-try follow-up
  (`center_dist_mm` already exists) but out of scope for the first 5A run.

### Norm layer
- **LayerNorm**, replacing BatchNorm1d in both `pair_mlp` and `point_mlp`
  (`src/phase5_encoder/model.py`). Reason: BatchNorm's eval-mode running
  statistics under LOFO would be fit only on the six training fragments,
  making a held-out fragment's distribution shift indistinguishable from a
  genuine interface-association failure -- the same confound shape already
  used to justify PPF over PCA canonicalisation (Deviation 2 above).
  Re-measured (not assumed) after the change: pose gate still
  `passed=True, max_deviation=0.0` for both `pair_input=True/False`.

### Stopping rule
- **Pair-level validation split, held out from every interface** (not a
  held-out interface, not a held-out fragment) -- `val_fraction=0.10` of
  each training interface's positive pairs. Patience=5 / epoch_cap=50,
  retained from the accepted original design.
- **Stated weakness, measured (not inferred):** of 6,822 total patches,
  only 4,107 carry >=1 positive pair; among those, the positive-count-per-
  patch distribution is min 115, p10 187, **median 347**, p90 618, p99 815,
  max 815. A held-out pair's endpoints typically still appear in a few
  hundred other training pairs, so **patience may never fire** -- the
  training history's `stopped_by` field (exactly `"patience"` or
  `"epoch_cap"`, never ambiguous) must be reported plainly either way.

### Measured batch size (§3)
- **`B_a = 512`**, verified via `scripts/measure_step_memory.py` (real
  per-step composition: `B_a` anchors + `B_a` positives + deduped hard
  negatives, through the real `PointNetEncoder`, forward+loss+backward,
  peak GPU memory on the RTX 4050):

| B_a | H_distinct (deduped) | total_fwd | peak MB | step ms |
|---:|---:|---:|---:|---:|
| 512 | 0 | 1024 | 1085.7 | ~64 |
| 512 | 512 | 1536 | **1436.4** | ~96 |
| 1024 | 0 | 2048 | 2153.7 | ~129 |
| 3072 | 0 | 6144 | **OOM** | -- |

  `B_a=512` with full dedup headroom (up to 512 distinct hard negatives) is
  the chosen operating point: 1436.4 MB peak, ~3.9x headroom under the
  ~5.64 GB usable budget on this GPU.

### Four bugs found and fixed in a second review pass (before any run)

1. **Validation set was uncapped.** `val_fraction` applied to raw
   per-interface pair counts gave **49,569 validation pairs on real fold
   1** (dominated by F6-F7's 21,445) -- a ~9.8 GB fp32 similarity matrix
   and 99,138 sequential patch encodes, which would OOM or dominate the
   epoch wall-clock. Fixed: `build_fold_sampler(..., val_pair_cap=500)`
   downsamples proportionally across interfaces; `_validation_loss` now
   chunks into `val_minibatch_size=256` minibatches. Measured after the
   fix on real fold 1: **499 total validation pairs**.
2. **Interface weighting inverted (measured: 55.4% of draws to the rarest
   interface).** The original `max_ratio` clamp bounded weight against the
   wrong reference (the largest interface's inverse weight, not uniform).
   A first fix attempt (clamp the weight, renormalise) was ALSO measured
   wrong (90.8%, worse). Correct fix: clamp EFFECTIVE COUNTS before taking
   inverse-frequency, `max_ratio` lowered 20.0->5.0. Measured on real fold
   1 after the fix: rarest interface (F2-F4) at **24.65% (1.48x uniform)**
   -- lifted meaningfully, not dominant.
3. **Hard-negative dedup count was unmeasured.** Now returned from
   `_info_nce_step` and logged per epoch
   (`deduped_hard_negatives_per_step_mean/min/max`).
4. **In-batch positives were unmasked in the InfoNCE denominator.** Two
   draws can be genuine partners of each other (median 347 positives per
   contact patch) -- now masked via `_build_positive_partner_lookup`.

**Corrected fold-1 wall-clock** (an earlier estimate of 1.3-1.5 s/epoch was
wrong -- it omitted the validation term and undercounted feature-extraction
cost). Measured directly, Python-loop feature extraction included:

| Component | Measured | Frequency | Per-epoch |
|---|---:|---|---:|
| Training step (512+512) | ~1123 ms | 12/epoch | ~13.5 s |
| Mining pass (4,000-patch pool) | ~3388 ms | 1/epoch | ~3.4 s |
| Validation (499 pairs, 4 minibatches) | ~211 ms/mb | 1/epoch | ~0.84 s |
| **Total** | | | **~17.7 s** |

50-epoch cap (fold 1, if patience never fires): **~885 s (~14.8 min)**.

**Operational finding:** a fresh process hit spurious CUDA OOM at
previously-safe sizes (allocator fragmentation, not real capacity --
`nvidia-smi` showed the GPU idle). Fixed by
`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, now required in
the training command. `scripts/train_phase5a.py` gained `--max-epochs`
(for a quick first real-data run before committing to the full cap),
`--val-pair-cap`, and `--interface-weight-max-ratio` as CLI parameters.

All four fixes are covered by regression tests in
`tests/phase5_encoder/test_sampler_mining.py`. Full targeted suite:
**46 passed**. Full repo suite: **182 passed**.

---

## (g) Fold-1 dry run + mining optimization

The human ran the 1-epoch fold-1 dry run
(`--held-out fragment_caesar_fragment_1 --max-epochs 1`). This is a timing +
plumbing gate ONLY; nothing here touches the six pre-registered conditions.

### Measured vs the §9 estimate (the estimate was wrong; diagnosed, not retrofitted)

| Component | prior estimate | measured (real fold 1) |
|---|---:|---:|
| Mining pass | ~3.4 s | 18.17 s |
| Training (12 steps) | ~13.5 s | 20.92 s |
| Validation | ~0.84 s | 0.85 s |
| Per-epoch total | ~17.7 s | 39.95 s |
| Peak GPU memory | 1436/3222 MB | 956.9 MB |

**Root cause:** the mining estimate measured a SINGLE 4,000-patch pool
encode, but the code encodes each of 6 anchor fragments' non-adjacent pools
separately, and they overlap. Real work: 18,887 patch-encodes (15,466 pool
+ 3,421 anchor) vs only 5,822 distinct training patches -- ~3.2x redundant.

### Recurring failure class (named explicitly)

This is the FIFTH unit/scope mismatch in the project (a SIXTH and SEVENTH
were found later -- see sections (h)/(i)). The citation rule catches a wrong
*location*; it does not catch a number correctly copied from a real
measurement of the WRONG QUANTITY, nor a parameter that looks active but is
frozen (#7):

| # | Symptom | wrong unit/scope |
|---|---|---|
| 1 | "batch 512" | per-*patch* memory used as per-*step* |
| 2 | `val_fraction=0.10` | applied to *pairs*, reasoned as *patches* |
| 3 | H=4 hard-neg ratio | per-*anchor* asserted, code mines per-*patch* |
| 4 | mined-distance asymmetry | §2b pool sizes are *full-dataset*, no *fold* sees them |
| 5 | ~17.7 s/epoch | measured *one* pool, code does *six* overlapping |
| 6 | `steps_per_epoch=12` "epoch" | 12 steps is ~1.4% of pairs / ~90 anchor-passes-worth is 600 steps; "epoch" was never one pass, so patience=5 spanned noise not a real epoch (see section (h)) |
| 7 | jitter augmentation | `seed=cfg.seed` constant -> every patch got the SAME noise every epoch; jitter looked active but was inert (a parameter that looked on and wasn't -- a subtler cousin of the class: not a wrong unit, a frozen stochastic op) |

**Added rule:** every measurement states its UNIT and SCOPE (per patch / per
step / per epoch; per fold / full dataset; one pool / all pools), not just
its source location.

### Mining optimization (pure refactor, applied and proven identical)

Encode all 5,822 distinct training patches once per epoch into an eval-mode
cached table (`_encode_training_table`); pool and anchor accesses become
index lookups. Proven mathematically identical to the per-anchor-pool path
by `test_mining_cache_matches_uncached`; reproducible-within-epoch (no
jitter leak) by `test_mining_cache_uses_eval_mode_no_jitter`. **Measured:
mining 18.17 s -> 6.30 s** (2.9x). Corrected per-epoch ~28 s; 7 folds x 50
epochs ~2.7 h (upper bound).

### Deviation 2 closed (measured fact, not assumed)

268.1 mean distinct deduped hard negatives/step (258-280) vs 512 anchors --
~52% distinct. NOT dilution: the 268 are shared denominator candidates seen
by every anchor (~34% of each anchor's denominator is hard). Whether 34%
moves the easy-vs-hard AUC gap is condition 3's job, not a pre-tune.

### Sampler verified

Observed interface draw fractions matched the weights within sampling noise
at 6,144 draws (F2-F4 obs 0.236 vs weight 0.246, etc.; 5*SE ~ 0.028). No
sampler bug.

### §2b prediction: PREDICTED, NOT OBSERVED (recheck post-training)

§2b hypothesized larger pool -> closer mined negatives. Fold 1 showed the
opposite (F5=0.046, F6=0.049 farther than F2/F3/F4/F7=0.025-0.026; F6 has
the largest fold-1 pool yet mines farthest). On an untrained encoder this is
random-embedding geometry, not interface signal -- so the §2b mechanism is
unconfirmed and must be rechecked after real training. Also: F6's fold-1
pool is 3,000, not the 4,000 in the §2b full-dataset table -- every LOFO
fold shrinks the pool, so no fold sees the table's 2.2x asymmetry.

---

## (h) Fold-1 training runs (honest record; NOT the verdict)

Two fold-1 training runs exist. BOTH were run with the `steps_per_epoch=12`
unit bug present (failure-class #6, section c): an "epoch" was 12 gradient
steps = ~1.4% of the fold's pairs, so 50 epochs = ~600 gradient steps =
~90 anchor-passes over the 3,421 fold-1 anchors. Patience measured over such
"epochs" spans noise, not real passes.

### Run 1 -- uncapped, 50 epochs, patience disabled (BASELINE)
`phase5a_runs/`. Best-val checkpoint epoch 19. Held-out fold-1 eval:
- Retrieval mAP 0.1065 vs FPFH 0.1053; paired mAP diff +0.0012
  [-0.0019, +0.0042] -- **CI includes zero, NOT a win**.
- P@1 0.1545 vs FPFH 0.1676 -- below.
- Condition-3 easy-vs-hard gap 0.261 vs FPFH 0.610 -- **shrunk (MET on fold)**.
- Held-out fragment-ID self-retrieval 0.470 vs FPFH 0.895 -- **~half the leakage**.
- Train loss 6.48->4.64, val loss non-monotonic (best 5.51 @ ep19, drifts to 5.73).

### Run 2 -- 16 mm distance-capped positives, 50 epochs, patience disabled (CONFOUNDED SIDE-BRANCH)
`phase5a_runs_capped16/` (kept as provenance, NOT a baseline). Best-val
checkpoint epoch 27. Held-out fold-1 eval:
- Retrieval mAP 0.1173 vs FPFH 0.1053; paired mAP diff +0.0120
  [+0.0085, +0.0157] -- CI excludes zero (mAP win on this fold).
- P@1 0.1516 vs FPFH 0.1676 -- still below.
- Condition-3 gap 0.280 vs FPFH 0.610 -- stayed shrunk.
- Fragment-ID self-retrieval 0.463 -- ~half FPFH, unchanged.

**Honest caveats on Run 2 (why it is NOT evidence of the cap's effect):**
1. The steps_per_epoch unit bug is present in both runs.
2. Its validation curve is NON-MONOTONIC noise (5.5<->6.3, no trend, on 499
   val pairs / 12 steps); "best @ epoch 27" is the low-point of a noisy
   signal (ep15=5.48, ep35=5.50 are within noise of ep27=5.19). The evaluated
   checkpoint is effectively arbitrary within the run.
3. Train loss falls cleanly (6.49->4.49) while val wanders -- the signature of
   fitting too-few gradient steps of data that doesn't generalize (small-anchor
   overfitting), NOT a converged-then-overfit model.
4. Fold-1 only, and "wins mAP / loses P@1" is NOT a pass (pre-registration
   requires BOTH on the majority of {F1-F4}).

The mAP win is promising and not dismissed, but three of the four inputs to it
are compromised (arbitrary checkpoint, unfixed epoch bug, and -- corrected --
the two runs actually differ only by the cap, both 50-epoch/patience-off, so
the confound is the unit bug + noisy checkpoint, not duration). Not yet durable
evidence. **Do not tune the cap to chase P@1 until the training loop is sound.**

### Run 3 -- UNIT-FIXED: 7 steps/epoch, cap 85, patience 8, uncapped (CLEAN)
`phase5a_runs_unit7/`. The single-variable re-run vs Run 1 baseline: only the
epoch UNIT changed (steps_per_epoch 12->7 = one anchor-pass over 3,421 anchors;
epoch_cap 50->85 to HOLD the budget at ~595 steps; patience re-enabled now that
it spans coherent passes). Budget held constant -- NOT trained longer.

Result: **patience FIRED at epoch 10** (77 gradient steps; did NOT reach the
85-cap). Best val loss 5.5288 @ **epoch 2**, then no improvement for 8 epochs.
- val trajectory: 5.573 -> 5.529(ep2) -> 5.588 -> ... -> 5.594; **no downward trend**.
- train trajectory: 6.498 -> 5.522, falling monotonically, still descending at stop.

**Interpretation (held loosely):** the epoch-unit fix worked -- patience is now
a meaningful signal and it fired early. But validation does NOT trend: best val
(5.529) is 0.8% below val[0] (5.573), reached at epoch 2, flat/noisy thereafter,
while train falls cleanly. This is the small-anchor overfitting signature
pre-framed last turn -- the unit fix does not touch it, as predicted.

**BUT two caveats keep this from being a conclusion:**
1. The validation SIGNAL may be incapable of trending: with median 347 positives
   per contact patch (§7), a held-out val pair's endpoints appear in hundreds of
   training pairs, so "held-out" val loss is nearly the same quantity as train
   loss. A flat val curve may mean "doesn't generalize" OR "the val metric can't
   distinguish generalization from memorization." Val loss is NOT a pass
   criterion anyway -- the verdict is LOFO retrieval through the battery.
2. **The one regularizer in the pipeline (jitter) was INERT** during all three
   runs -- see section (i). We cannot claim a capacity/data-ratio wall until the
   one regularizer that exists has actually functioned once.

---

## (i) Deviation 4 -- jitter augmentation was inert (constant seed)

**Defect (failure-class #7):** `_encode_patch_ids` passed `seed=cfg.seed`
(constant 0) to `prepare_patch` for every patch every epoch. `jitter()` seeds
its RNG from that, so **every patch received the identical noise realization on
every pass** -- a fixed perturbation baked in once, not fresh stochastic
augmentation. As regularization this is inert: the network sees the same
"augmented" patch each epoch, so there is nothing to average over and no
diversity to resist memorization. The one regularizer in the pipeline had never
actually functioned across Runs 1-3.

This is a subtler cousin of the recurring unit/scope class: not a wrong unit,
but **a parameter that looked active and wasn't** -- jitter was nominally
`training=True` and running, yet frozen by a constant seed.

**Fix:** `prepare_patch` gains a separate `jitter_seed` (default None ->
falls back to `seed`, so eval/mining/pose-gate paths stay byte-identical). The
FPS resample seed stays deterministic (its main path is argmax-from-centroid,
RNG-free; the seed only drives the pad branch on under-full patches -- so FPS
has ~no diversity to gain and must stay deterministic for the pose gate).
`_encode_patch_ids` derives the jitter seed per (cfg.seed, epoch, frag_index,
pid) via `np.random.default_rng([...])` (reproducible integer composition, NOT
Python `hash()` which is per-process salted), so noise is fresh per epoch AND
per patch -- not per-epoch-only (which would give all patches in an epoch
correlated noise).

**Honesty note:** working jitter is a genuine regularization change. If val
trends after this fix, that is a real improvement -- and the re-run (Run 4)
differs from Run 3 by this ONE variable (jitter-on vs jitter-inert), nothing
else; budget and all other settings held constant. If val STILL does not trend
with working augmentation, THEN the capacity/data-ratio wall is the honest
finding and we stop chasing it.

**Deferred (deliberately, not overlooked):** making FPS resample stochastic
per epoch. Its main path is deterministic by construction (pose gate); only the
pad branch on under-full patches uses the seed, so there is little diversity to
gain. Revisit only if working jitter still doesn't make val trend.

---

## (d) Encoder results

*No encoder trained yet.* This section is generated from the JSON once encoder
sources are present. Pose-invariance gate status for the untrained pipeline:
`passed=True, max_dev=0.000e+00` (see `tests/phase5_encoder/test_pose_gate.py`).

---

## (e) Six-condition verdict

*Pending encoder results.* Will be generated against the frozen thresholds in
section (a). Per the pre-registered rule: if condition 1 passes but 2-4 fail, the
honest conclusion is "improved contact/fragment discrimination, not interface
association" -- reported as such, not upgraded.
