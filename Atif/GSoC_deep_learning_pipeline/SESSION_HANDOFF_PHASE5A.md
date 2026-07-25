# Session Handoff — Phase 5A, Post-Implementation, Pre-Training

Paste this whole file (or the "Prompt to paste" block at the bottom) as the
first message to a new AI session, alongside `MASTER_PROMPT.md`,
`CODING_AI_RESUME_CONTEXT.md`, `PHASE5_PREREGISTRATION.md`, and
`PHASE5A_TRAINING_DESIGN_REVISED.md` (all in the repo root). This file
records exactly what happened in the session that just ended, so the next
session does not have to re-derive it and does not accidentally re-open
questions that are already closed.

## Repository state

- Git root: `/home/kira/Desktop/Healing_Stones` (one level ABOVE the
  project folder). Project folder:
  `/home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline`.
- Branch: `deep_learning`.
- HEAD at end of session: `507ab93`.
- Commit chain from this session, oldest to newest:
  - `410ce8d` — Phase 5 diagnostics infra + encoder skeleton + pre-reg docs
    committed for the first time (previously untracked since inception).
  - `6e25c00` — results-log regen (stamps `410ce8d`).
  - `95734a4` — Design revision round 2 approved; items 1–4 implemented
    (condition-3 fold-restriction, BatchNorm→LayerNorm, real memory
    measurement, §2b mining design resolved); item 5 (training loop +
    entry point) written but explicitly NOT run.
  - `e57c903` — results-log regen (stamps `95734a4`).
  - `1d22e49` — four bugs found on a second review pass and fixed BEFORE
    any real-data run (validation-set OOM, interface-weight inversion,
    unlogged hard-negative dedup count, unmasked in-batch positives).
  - `507ab93` — results-log regen (stamps `1d22e49`, current HEAD).
- Working tree is clean as of `507ab93`. Nothing is staged or uncommitted.
- Also fixed this session, unrelated to Phase 5 code but worth knowing:
  `master_p` (the original 631-line master prompt, untracked at the git
  root, one `rm` away from being lost) was moved to
  `MASTER_PROMPT.md` in the project folder and committed in `410ce8d`. A
  stray transcript file `con` was deleted. An unrelated top-level
  `README.md` (the umbrella "Healing Stones" project README, last touched
  by a different author, staged as deleted) was restored, not deleted —
  it belongs to the parent repo, not this pipeline.

## What exists now (files created or changed this session)

New Phase 5A implementation (all under `src/phase5_encoder/`):
- `sampler.py` — `FoldSampler`, `build_fold_sampler`, `InterfaceWeighting`.
  Fragment→interface→partner sampling; capped-effective-count
  inverse-frequency interface weighting (NOT plain inverse-frequency, NOT
  the first "clamp the weight" attempt — see Bug 2 below for why both of
  those are wrong); pair-level validation split held out from every
  interface, with a **hard cap** (`val_pair_cap`, default 500) on total
  validation pairs.
- `mining.py` — `mine_epoch_hard_negatives`, `MinedNegatives`. On-the-fly,
  per-epoch, fold-restricted hard-negative mining under the CURRENT
  encoder state, `no_grad`/`eval`, full non-adjacent candidate pool (no
  subsampling), per-fragment mined-distance logging.
- `train.py` — `TrainConfig`, `train_one_fold`, `_info_nce_step`,
  `_validation_loss`, `_build_positive_partner_lookup`. The actual training
  loop: InfoNCE loss with same-interface/cross-interface positive masking,
  deduped hard negatives, patience/epoch-cap with an explicit
  `stopped_by` field, checkpointing at best validation loss.
- `model.py` — `PointNetEncoder` now uses `nn.LayerNorm` (was
  `nn.BatchNorm1d` — changed this session; see Deviation/fix log below).

Diagnostics infra changes (`src/phase5_diagnostics/`):
- `ranking.py` — `easy_vs_hard_separability` gained a `held_out` parameter;
  `lofo_per_fold` gained a `hard_negatives` parameter and wires
  `held_out=<fold's fragment>` through per fold.
- `runner.py` — passes `hard_neg` into the `lofo_per_fold` call so every
  fold's `hard_negative_strata` entry is now held-out-restricted (fixes the
  circularity where condition 3 was previously measured on the training
  set).

New scripts (`scripts/`):
- `measure_step_memory.py` — real per-training-step GPU memory
  measurement (anchors + positives + deduped hard negatives, real
  `PointNetEncoder`, forward+loss+backward) and a real mining-pass memory
  measurement (`no_grad`, full candidate pool). Reproduces the numbers
  cited in the design doc.
- `train_phase5a.py` — the CLI entry point. **NOT YET RUN on real data.**
  Has an extensive docstring with the exact command, environment variable
  requirement, wall-clock estimate (corrected, see below), output file
  locations, and how to read `stopped_by`. Flags: `--held-out` (required),
  `--max-epochs` (override for a quick first run), `--val-pair-cap`,
  `--interface-weight-max-ratio`, plus the usual paths/seed/batch-size.

New/changed docs:
- `PHASE5A_TRAINING_DESIGN_REVISED.md` — the full design document, now at
  "round 2" with an added §8 (implementation status) and §9 (four bugs
  found and fixed in a second review pass). This is the primary reference
  for every design decision's rationale and citation.
- `PHASE5_RESULTS_LOG.md` — regenerated (auto-generated file, do not
  hand-edit; edit `scripts/generate_results_log.py`'s templates instead).
  Now has section (f) "Training design decisions" covering loss, hard-neg
  mechanism, mining policy, interface weighting, positive-distance policy,
  norm layer, stopping rule, measured batch size, AND the four
  second-review-pass bug fixes with their measured before/after numbers.
- `scripts/generate_results_log.py` — gained `section_training_design()`.

New tests: `tests/phase5_encoder/test_sampler_mining.py` (12 tests: sampler
construction, validation-split disjointness, multi-interface detection,
weighting-cap correctness, an end-to-end `train_one_fold` smoke test on a
toy fixture, and 5 regression tests that each reproduce one of the four
measured bugs directly rather than just checking for no-crash). 4 new
tests in `tests/phase5_diagnostics/test_diagnostics.py` for the
`held_out`-restriction fix.

**Test status at HEAD:** targeted suite (`tests/phase5_diagnostics/
tests/phase5_encoder/ tests/baseline_geometry/`) = **46 passed**. Full
repo suite (`tests/`, no `ulimit` — the project's memory cap is tuned for
Phase 1–4 dense-cloud steps and starves the phase5+model tests) =
**182 passed**.

## The full arc of this session, in order

1. **Verified a detailed external review** of the Phase 5 diagnostics
   infrastructure (pre-existing from an earlier session) against the
   actual code. Every specific claim in that review checked out: a real
   circularity in condition 3 (hard negatives mined once from FPFH,
   handed unrestricted into every source's `easy_vs_hard_separability`
   call, no fold filter anywhere), a real 1-per-anchor limit in
   `mine_hard_negatives` (strict `argmin`, no `top_k`), a real
   memory/batch mismatch in the existing results log, and a real
   BatchNorm/LOFO confound. Also did housekeeping: recovered `master_p` →
   `MASTER_PROMPT.md`, deleted a stray `con` transcript file, restored an
   unrelated `README.md` that was wrongly staged for deletion, committed
   the previously-untracked Phase 5 tree by explicit path (never
   `git add .`, since the git root is one level above the project).

2. **Drafted `PHASE5A_TRAINING_DESIGN_REVISED.md` round 1** addressing 7
   review items (condition-3 circularity, 1-per-anchor hard negatives,
   GPU memory sizing, fragment→interface→partner sampling, positive
   distance policy, BatchNorm→LayerNorm, validation/stopping strategy)
   under a strict citation rule: every number must cite a `file:line`, a
   JSON field, or pasted command output, or be marked
   `UNVERIFIED — placeholder`.

3. **Round 1 was reviewed and returned with corrections**: 4 wrong line
   citations (caught and fixed by re-grepping/re-reading, not trusting the
   first pass), and a substantive gap — mining cost was listed as "measure
   later" when it should have been a design decision (mining frequency,
   candidate-pool policy, `no_grad`/caching requirement, per-fragment
   difficulty logging). Fixed all of this in round 2, with real measured
   per-fragment non-adjacent pool sizes (F1/F4=1,822 ... F6/F7=4,000) and a
   measured positive-pairs-per-patch distribution (4,107/6,822 patches
   carry ≥1 positive; median 347 per patch) replacing an inferred claim.

4. **Design approved. Implemented items 1–4, scoped item 5** (commit
   `95734a4`): wired the `held_out` filter through the diagnostics,
   swapped BatchNorm for LayerNorm (re-measured the pose gate afterward,
   didn't just assume it still passed — `max_deviation=0.0`, confirmed),
   measured real per-step GPU memory (`scripts/measure_step_memory.py`,
   `B_a=512` verified at 1436.4 MB / ~96 ms/step), and wrote the full
   training loop + CLI entry point (`sampler.py`, `mining.py`, `train.py`,
   `scripts/train_phase5a.py`) — explicitly NOT run, per instruction that
   the human runs training, not the AI. One self-caught correction during
   this pass: an earlier speculation that the mining pass would be
   *cheaper* than a training step (no backward) was measured and found
   **wrong** — mining pools (up to 4,000 patches) are larger than the
   training batch (512), so a mining pass peaks at 3221.9 MB, more than a
   training step's 1436.4 MB. Corrected in the doc rather than left
   standing.

5. **A second review pass, still before any real-data run, found four
   more bugs** — all confirmed by direct measurement against real fold-1
   data before touching any code, and all fixed (commit `1d22e49`):
   - **Validation OOM**: `val_fraction` applied to raw pair-row counts
     gave 49,569 validation pairs on real fold 1 (not "a few hundred" as
     the docstring claimed) — a ~9.8 GB similarity matrix that would OOM
     or dominate the epoch. Fixed with a hard `val_pair_cap` (500,
     proportional downsampling across interfaces) plus minibatched
     evaluation. Verified after fix: 499 real validation pairs.
   - **Interface-weighting inversion**: the original cap semantics let
     the rarest interface (F2-F4) receive 55.4% of all training draws —
     the exact overfitting failure the weighting was designed to prevent.
     **A first fix attempt was also wrong** (clamp the post-normalization
     weight against uniform, then renormalize) — measured at 90.8%,
     *worse*. This false start is documented in the design doc rather than
     silently discarded. The correct fix clamps effective counts before
     taking inverse-frequency; default `max_ratio` lowered from 20.0 to
     5.0. Verified after fix: F2-F4 at 24.65% (1.48× uniform) — lifted
     meaningfully without dominating.
   - **Unmeasured hard-negative dedup count**: mining produces one
     negative per contact patch, not per draw, so the real per-step
     distinct hard-negative count is far below the batch size — this was
     asserted as "every-step signal" without ever being measured. Now
     returned from `_info_nce_step` and logged per epoch.
   - **Unmasked in-batch positives**: `cross_entropy` treated every
     non-target candidate as a negative, including genuine positive
     partners that land in the same batch (median 347 positives per
     contact patch — common, not rare, at real draw rates). Fixed with a
     per-fold positive-partner lookup that masks true partners out of the
     denominator.
   - Also re-derived the fold-1 wall-clock estimate with real
     measurements (the earlier estimate of 1.3–1.5 s/epoch was wrong — it
     omitted the validation term's cost entirely and undercounted
     Python-loop feature-extraction cost for training steps too).
     Corrected: **~17.7 s/epoch** (training steps ~13.5s, mining pass
     ~3.4s, validation ~0.84s — validation is now the *smallest* term, not
     the dominant one it would have been uncapped), **~14.8 minutes** for
     the 50-epoch cap on fold 1 alone if patience never fires.
   - Also found and fixed a real, separate operational issue while
     re-measuring: a fresh process hit spurious CUDA OOM errors at
     batch/pool sizes previously measured safe, despite `nvidia-smi`
     showing the GPU nearly idle. This was PyTorch allocator
     fragmentation, confirmed by re-measuring with
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set, which
     reproduced the original safe numbers exactly. This environment
     variable is now required and stated as such in
     `scripts/train_phase5a.py`'s docstring and a runtime warning.

## What is verified vs. what is still unverified

**Verified by direct measurement this session** (not assumed): pose gate
`max_deviation=0.0` before and after LayerNorm swap; per-step GPU memory
at `B_a=512` (1436.4 MB); mining-pass memory at all four real
per-fragment pool sizes (1,822 / 2,000 / 2,822 / 4,000 patches); real
fold-1 validation-pair count before (49,569) and after (499) the cap fix;
real fold-1 interface weights before (55.4% max) and after (24.65% max)
the weighting fix; per-component training/mining/validation wall-clock
costs (feature extraction + GPU forward/backward, separately); the
allocator-fragmentation OOM and its fix.

**Explicitly NOT yet done — this is the actual next step:**
- `scripts/train_phase5a.py` has never been run against the real dataset
  end-to-end. Only `train_one_fold`'s plumbing has been smoke-tested on a
  tiny synthetic toy fixture (4 fragments, 4 patches each). The ~17.7
  s/epoch estimate is arithmetic from separately-measured real components,
  not a single measured end-to-end epoch.
- No encoder has been trained. `PHASE5_RESULTS_LOG.md` section (d)
  "Encoder results" is still a placeholder.
- The 6-condition Phase 5A pass/fail verdict (`PHASE5_PREREGISTRATION.md`)
  has not been evaluated against anything, because there is no encoder
  output yet.

## Immediate next step

Run the fold-1 dry run exactly as documented in
`scripts/train_phase5a.py`'s docstring:

```bash
cd /home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=src python3 scripts/train_phase5a.py \
    --held-out fragment_caesar_fragment_1 \
    --out-dir phase5a_runs \
    --max-epochs 1
```

Then compare the real wall-clock and the history JSON's
`interface_draw_counts` / `deduped_hard_negatives_per_step_mean` /
`per_fragment_mined_distance_mean` fields against the estimates in this
file and the design doc. If the real numbers are within ~2× of the
~17.7 s/epoch estimate and the interface draw counts look like the
measured 24.65%-for-F2-F4 pattern (not dominated by one interface), drop
`--max-epochs` and run the full `epoch_cap=50` for fold 1. Only after
fold 1's full run completes and its wall-clock is reported should folds
2–7 be run. After all 7 folds: export each fold's checkpoint as
`PatchDescriptors` (the existing, unmodified contract in
`src/phase5_diagnostics/embeddings.py`) and run the **existing, unmodified**
diagnostic battery (`scripts/run_phase5_diagnostics.py` / `runner.py`) to
judge against all 6 pre-registered conditions — not just raw mAP.

## Standing rules that must continue to apply

- **Every design number must cite a file:line, a JSON field, or pasted
  command output.** This rule caught 4 wrong citations in round 1 and
  found 4 real bugs (including one false-start fix that was itself wrong)
  in the second review pass. Do not relax it now that training is closer.
- **The human runs training, not the AI.** Do not invoke
  `scripts/train_phase5a.py` without explicit instruction to do so, even
  with `--max-epochs 1`.
- **`git add` by explicit path, never `git add .`** — the git root is one
  level above the project folder and contains an unrelated umbrella
  README that has already been accidentally staged for deletion once this
  project's history.
- **Regenerate `PHASE5_RESULTS_LOG.md` via `scripts/generate_results_log.py`
  after any commit that changes Phase 5 code, then commit the regenerated
  log as a small follow-up**, so its "Git commit at generation" line is
  never decorative (stale/wrong). This project has now done this pattern
  three times in a row (`410ce8d`→`6e25c00`, `95734a4`→`e57c903`,
  `1d22e49`→`507ab93`) — keep doing it.
- **State explicitly whether `stopped_by` was `"patience"` or
  `"epoch_cap"`.** Never describe a training run as "early stopped"
  without specifying which. The pair-level validation split was chosen
  knowing patience may rarely fire (median 347 other positives per
  contact patch means held-out pairs' endpoints are rarely truly novel to
  the model) — this is a stated, accepted weakness, not a bug to "fix" by
  changing the validation strategy without re-discussing it.
- **A "genuine Level-3 success" requires all 6 conditions in
  `PHASE5_PREREGISTRATION.md`, not just beating FPFH on raw mAP.** If
  condition 1 passes but 2–4 fail, the required honest report is "improved
  contact/fragment discrimination, not interface association" — do not
  upgrade that claim.

---

## Prompt to paste verbatim into the new session

> Resuming this project. Read `SESSION_HANDOFF_PHASE5A.md` in the repo root
> fully before responding — it records everything from the immediately
> preceding session in full detail: what was implemented, what was
> reviewed and found wrong (including one fix attempt that was itself
> wrong and had to be fixed again — documented, not hidden), what is
> measured-verified vs. still unverified, and the exact next command to
> run. Also read `MASTER_PROMPT.md` (original governing instructions),
> `PHASE5_PREREGISTRATION.md` (frozen pass/fail thresholds — do not weaken
> them), and `PHASE5A_TRAINING_DESIGN_REVISED.md` (the full design
> document, now at "round 2" with a §9 documenting four bugs found and
> fixed on a second review pass before any training run).
>
> Current state, verified against the actual repo, not just told to you:
> HEAD is `507ab93` on branch `deep_learning`, working tree clean. 46
> tests pass in the targeted Phase 5 suite, 182 pass in the full repo
> suite. No encoder has been trained yet — `scripts/train_phase5a.py`
> exists, is fully implemented, and has been smoke-tested only on a toy
> fixture, never against real data end-to-end.
>
> The immediate next step is the fold-1 dry run documented in
> `SESSION_HANDOFF_PHASE5A.md`'s "Immediate next step" section:
> `--held-out fragment_caesar_fragment_1 --max-epochs 1`, with
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set (this was found
> necessary this session to avoid spurious allocator-fragmentation OOM
> errors — real finding, not optional). Compare the real wall-clock and
> the history JSON's per-interface/per-fragment fields against the
> estimates already measured and documented before deciding whether to
> proceed to the full 50-epoch run.
>
> Continue applying the same discipline this project has used throughout:
> every design number needs a `file:line`, JSON field, or pasted command
> output; I will independently re-verify any "done" or "fixed" claim
> against the actual repository state before accepting it; do not run
> `scripts/train_phase5a.py` without my explicit instruction, even for a
> 1-epoch dry run; regenerate and commit `PHASE5_RESULTS_LOG.md` as a
> follow-up after any Phase 5 code commit so its provenance line stays
> real.
