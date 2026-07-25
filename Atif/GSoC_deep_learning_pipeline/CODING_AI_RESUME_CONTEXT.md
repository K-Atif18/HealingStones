# Resume Context for the Coding AI — Phase 5, Start of Phase 5A

Read this file first in any new session, alongside `PHASE5_PREREGISTRATION.md`
(the fixed pass/fail thresholds) and `PHASE5_RESEARCH_CONTEXT.md` (the research
narrative / why, for the human). This file is the operational state: what
exists, what's verified, what command to run next. Original governing
instructions (role, phase structure, required reasoning-before-code workflow,
golden rule) are the master prompt already given at the start of this project
— continue operating under those rules; they are not repeated in full here,
only the state that has accumulated since.

## Repository root
`/home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline`

## Where things are

- `PHASE5_PREREGISTRATION.md` — pass/fail thresholds, fixed before results,
  now includes the bootstrap-CI refinement (section 4b) with the corrected
  well-supported LOFO fold set.
- `src/phase5_diagnostics/` — reusable diagnostic infrastructure:
  - `context.py` — derives contact/interface labels from Phase 1–3 outputs.
  - `embeddings.py` — `PatchDescriptors` contract + random/centroid/FPFH
    sources (this is the interface the future encoder's output must match).
  - `ranking.py` — `build_gallery`, `positive_partner_map`, `evaluate_ranking`
    (core harness), `calibrate_null`, `lofo_per_fold`, hard-negative mining,
    per-interface macro/micro breakdown, bootstrap CI helper.
  - `probes.py` — shortcut battery (fragment-ID probe, contact-vs-noncontact
    AUC, neighbor contact/non-contact ranking gap, distinctiveness-similarity
    correlation, boundary-openness R²) + pose-invariance gate.
  - `runner.py` — orchestrator, called by `scripts/run_phase5_diagnostics.py`.
- `scripts/run_phase5_diagnostics.py` — regenerates
  `phase5_diagnostics_results/baseline_diagnostics.json`. Run this any time
  diagnostics code changes or a new embedding source (the encoder) needs to be
  measured.
- `phase5_diagnostics_results/baseline_diagnostics.json` — current measured
  results for `random`, `centroid`, `fpfh` sources. Treat as regenerable, not
  hand-edited.
- `tests/phase5_diagnostics/test_diagnostics.py` — 8 tests, all passing,
  including `test_lofo_folds_are_nonempty` (regression guard, see "bug fixed"
  below) and 2 CI tests (CI brackets the mean; half-width shrinks with n).
- Upstream data this all depends on (Phases 1-4, already complete, do not
  regenerate unless asked): `dataset/` (aligned fragments, transforms,
  normals), `patches/` (6,822 patches), `pairs/` (positive/negative pair
  JSONs — large files, ~400MB each), `phase3_final_review/` (distinctiveness,
  contact analysis), `baseline_results/` (Phase 4 FPFH/SHOT/retrieval/
  registration numbers), `src/baseline_geometry/`.

## Command reference

```bash
# Run the phase5 diagnostics test suite
cd /home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline
python3 -m pytest tests/phase5_diagnostics/ tests/baseline_geometry/ -q

# Regenerate the diagnostics JSON (memory-bounded, ~2 min)
cd /home/kira/Desktop/Healing_Stones/Atif/GSoC_deep_learning_pipeline
ulimit -v 8000000
PYTHONPATH=src timeout 2400 python3 scripts/run_phase5_diagnostics.py
```

## Status as of last session (verified directly against repo, not just told)

- Phases 1–4 complete (dataset, patches, ground truth, geometric baselines).
  FPFH: raw mAP 0.1092, P@1 varies by fold; SHOT weaker (ROC-AUC 0.536).
- Phase 5 pre-registration written **before** measuring anything.
- Diagnostic infrastructure built and validated:
  - Null floor calibration (random/centroid) — done.
  - Shortcut battery — done, run on FPFH baseline:
    - fragment-ID kNN probe = 0.892 (chance 0.143) — **high, primary risk**.
    - contact-vs-noncontact AUC = 0.731.
    - neighbor contact-vs-noncontact ranking gap = +0.073 (positive, weak real
      signal).
    - distinctiveness–similarity Pearson r = -0.400 (FPFH tracks roughness).
    - boundary-openness R² = 0.002 (negligible, not just an edge detector).
  - Hard-negative stratification — done: `auc_pos_vs_easy` = 0.706,
    `auc_pos_vs_hard` = 0.092 (worse than chance), gap = 0.614. This is the
    sharpest FPFH pathology to beat.
  - Density/acquisition fingerprint probe — done: 0.356 (chance 0.143) →
    density normalization + jitter is a **mandatory** input-pipeline step for
    the encoder, independent of anything else.
  - Pose-invariance / leakage gate — implemented and validated: correctly
    fails a global-frame encoder, correctly passes a distance-only one. This
    gate **must block training** if it fails on the real Phase 5A pipeline.
  - LOFO per fold — **bug found and fixed** (see below), now produces real
    numbers with bootstrap CIs.
- All instrumentation is source-agnostic: it consumes `PatchDescriptors` and
  works identically for `random`, `centroid`, `fpfh` today and the trained
  encoder later. Do not fork this logic per-source.

## Bug that was found and fixed (context for why LOFO code looks the way it does)

Original `lofo_per_fold` produced `n_queries: 0, mAP: NaN` on every one of the
7 folds — silently. Root cause: `positive_partner_map(ctx, gallery)` had a
check `if key not in gallery.row_of: return` that required the *query*
patch itself to already be inside the (possibly filtered) gallery before it
could become a valid key. LOFO's gallery filter explicitly excludes the
held-out fragment (`g_keep: fid != held`), so the held-out fragment's own
patches could never satisfy that check — every fold's query set was empty by
construction, not by data sparsity. Fix: query eligibility was decoupled from
gallery membership — only the *candidate/partner* row (`gb`) must be in
`gallery.row_of`; the query key itself no longer needs to be. Guarded going
forward by `test_lofo_folds_are_nonempty`. **Lesson for future work in this
repo: always inspect the actual JSON output before trusting a "diagnostics
complete" claim — a prose summary said "computed for all 7 folds" once when
it factually was not.**

## Current LOFO baseline (FPFH), with 95% bootstrap CIs — the Phase 5A target

| Fold | queries | neighbors | FPFH P@1 [CI] | Random P@1 [CI] | Separated? |
|---|---:|---:|---|---|---|
| F1 | 686 | 4 | 0.168 [0.140, 0.195] | 0.071 [0.054, 0.090] | yes |
| F2 | 516 | 3 | 0.194 [0.161, 0.229] | 0.064 [0.045, 0.085] | yes |
| F3 | 573 | 3 | 0.169 [0.140, 0.199] | 0.035 [0.021, 0.051] | yes |
| F4 | 544 | 4 | 0.184 [0.151, 0.219] | 0.077 [0.057, 0.101] | yes |
| F5 | 661 | 4 | 0.074 [0.056, 0.094] | 0.054 [0.038, 0.073] | **no (overlap)** |
| F6 | 780 | 2 | 0.032 [0.021, 0.045] | 0.064 [0.047, 0.082] | no (FPFH worse, low-conf) |
| F7 | 347 | 2 | 0.000 [0.000, 0.000] | 0.081 [0.055, 0.112] | no (FPFH worse, low-conf) |

mAP CIs are tighter and separate on F1–F5(+F6); P@1 CIs are noisier (Bernoulli
variance) and only separate on F1–F4. **Refined well-supported fold set for
pass/fail: F1–F4** (F5 demoted to "signal-ambiguous," reported but not counted;
F6/F7 low-confidence, reported separately, F7 shows genuine degeneracy —
consistent with the known ~5-7 point F5↔F7 tiny-contact case from Phase 3).

## Phase 5A pass condition (from `PHASE5_PREREGISTRATION.md`, do not weaken)

The encoder must, on LOFO specifically: beat FPFH with **non-overlapping 95%
CIs on the majority of {F1, F2, F3, F4}**, on **both P@1 and mAP**. At
evaluation time use a **paired bootstrap on the per-query metric difference**
(encoder − FPFH, same query/gallery), not independent-CI overlap checking —
strictly more powerful. The plumbing for this (`return_per_query`) already
exists in `evaluate_ranking` and should be reused, not reimplemented.

Full 6-condition Phase 5 success definition lives in
`PHASE5_PREREGISTRATION.md`; do not treat a single beaten metric (e.g. raw mAP)
as sufficient — see the "genuine Level-3 success" checklist there.

## Immediate next step (what to do when this session resumes)

Nothing has been trained yet. Proceed to **Phase 5A**, in this order, per the
master prompt's required-workflow rule (explain objective/theory/inputs/
outputs/failure modes/validation before implementation, every phase):

1. Torch install/environment setup.
2. Input pipeline: mandatory density normalization + resampling + jitter
   (justified by the 0.356 density-fingerprint measurement) + pose
   canonicalization. Wire the existing `pose_invariance_check` in as a hard
   gate — training must not proceed if it fails.
3. Small PointNet encoder — deliberately the simplest option first. Planned
   progression is PointNet (5A) → DGCNN (5B) → Point Transformer V3 (5C); do
   not jump straight to PTv3.
4. Train on current Phase 3 interface co-membership labels **plus hard
   negatives** (FPFH-mined look-alike cross-fragment patches, not just random
   negatives) — the easy-vs-hard AUC gap (0.706 → 0.092) is the specific,
   measured pathology the encoder needs to close, and random negatives alone
   won't test that.
5. Export encoder output as `PatchDescriptors` (same contract as
   `embeddings.py`) and run it through the **existing, unmodified** diagnostic
   battery (`runner.py` / `scripts/run_phase5_diagnostics.py`) — this
   infrastructure was built to be reused as-is for exactly this purpose.
6. Judge the result against all 6 conditions in
   `PHASE5_PREREGISTRATION.md`, not against mAP alone. If condition 1 passes
   but 2–4 fail, report it honestly as "improved contact/fragment
   discrimination, not interface association" per the pre-registered rule —
   do not upgrade the claim.

Open, undecided design note (raised, not resolved): the fragment-ID probe is
high enough (0.892 on FPFH) that LOFO alone may be catching the shortcut too
late (after training). Consider whether Phase 5A training itself needs
fragment-balanced sampling or a fragment-adversarial loss term to actively
discourage identity memorization during training, not just detect it
afterward. Decide this before or during 5A implementation, flag the decision
explicitly when made.

---

## Improved resume prompt — paste this to start the next coding session

Use this verbatim (or lightly edited) as the first message to the coding AI in
a new session, with the original master prompt (role/phase-structure/golden
rule) and this file both attached/pasted as context:

> Resuming this project. Do not restart from Phase 1 reasoning — Phases 1-4
> are complete and verified, and Phase 5 pre-registration + diagnostic
> infrastructure are already built and tested. Read
> `CODING_AI_RESUME_CONTEXT.md` in the repo root fully before responding, then
> `PHASE5_PREREGISTRATION.md` for the exact thresholds you must not weaken.
>
> Current verified state (I re-checked this myself against the actual repo,
> not just your prior summary): 8 tests pass in
> `tests/phase5_diagnostics/test_diagnostics.py`, `baseline_diagnostics.json`
> contains real per-fold LOFO numbers with 95% bootstrap CIs for FPFH and
> random baselines, and the well-supported fold set for the encoder pass bar
> is F1-F4 (F5 is signal-ambiguous, F6/F7 are low-confidence with F7 showing
> FPFH performing below the null floor).
>
> Continue to Phase 5A exactly as scoped in `CODING_AI_RESUME_CONTEXT.md`
> section "Immediate next step": environment setup, then the density-
> normalizing + pose-canonicalizing input pipeline gated by the existing
> `pose_invariance_check` (training must not run if this gate fails), then a
> small PointNet encoder (not DGCNN, not PTv3 yet — that's 5B/5C), trained
> with hard negatives from the existing FPFH-mined set, not just random
> negatives.
>
> Before writing training code: follow the master prompt's required workflow
> (objective, theory, inputs/outputs, failure modes, validation, THEN
> implementation) for Phase 5A specifically. Explicitly address the open
> design note at the end of this file — whether fragment-balanced sampling or
> a fragment-adversarial term is needed during training given the 0.892
> fragment-ID probe result, or whether post-hoc LOFO gating is sufficient —
> before deciding the training loop's sampling strategy.
>
> When you report any diagnostic or training result back to me, cite the
> exact file/line or JSON field you're reading it from. I will independently
> re-verify every "complete" or "passing" claim against the actual repository
> output before accepting it — this project has already caught one false
> "complete" claim (the LOFO empty-fold bug) this way, so treat that
> verification step as expected, not adversarial.
