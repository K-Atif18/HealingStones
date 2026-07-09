# Phase 4: Baseline Geometry — COMPLETE ✅

**Date Completed:** 2026-07-09
**Runtime:** ~75 seconds (full pipeline, all stages)
**Status:** All deliverables met, validation gate PASSED (4/4 checks)

---

## Executive Summary

Phase 4 establishes **non-learning geometric baselines** and measures them
against the Phase 3 ground truth *before* any deep learning is introduced. This
is the reference point every later (learned) phase must justify itself against,
and it quantifies the **similarity-vs-complementarity gap** that motivates the
rest of the project.

- ✅ **FPFH (33-D) + SHOT (352-D) descriptors** computed for all **6,822 patches**
- ✅ **Pair separability:** FPFH ROC-AUC **0.708**, SHOT ROC-AUC **0.536** (both beat chance 0.5)
- ✅ **Ranking retrieval:** FPFH mAP **0.109**, P@1 **0.13** (cross-fragment)
- ✅ **Registration backend validated:** contact-seeded recovery **90.9%** success vs **0%** on non-adjacent control
- ✅ **Global (unaided) registration fails** (0% adjacent) — a real, expected result that motivates Phases 6–7
- ✅ **14 unit/integration tests passing**

**Key research finding:** Hand-crafted similarity descriptors carry *some*
assembly signal (FPFH well above chance) but the signal is weak in absolute
terms (mAP ~0.11), and global surface matching cannot localise the fracture
interface unaided. The rigid-solver backend is sound (90.9% recovery when the
true interface is given). This cleanly separates the two problems: **interface
localisation is the hard part, not the rigid math** — exactly what the learned
phases target.

---

## Why This Phase Matters (Theory)

FPFH and SHOT are **similarity** descriptors: they map a local surface patch to
a vector so that geometrically *alike* neighbourhoods land close together.

- **FPFH** — 33-bin histogram of Darboux-frame angles between a point's normal
  and its neighbours' normals; pose-invariant (via Open3D).
- **SHOT** — 352-D signature built from a repeatable local reference frame (LRF)
  partitioning the support into 32 volumes (8 azimuth × 2 elevation × 2 radial),
  each holding an 11-bin cosine histogram of neighbour-vs-centre normals
  (implemented from scratch; Open3D has no SHOT).

**The central caveat:** at a true fracture interface the matching surfaces are
**complementary** (a convex bump on A fits a concave dish on B) — geometric
*opposites*, not look-alikes. A similarity descriptor assigns them a *large*
distance exactly where they match best. So the baseline is expected to have a
systematic blind spot on complementary break patches. Quantifying that gap is a
primary Phase 4 deliverable.

### The 8 mandated evaluation questions
1. **What is measured?** Similarity of local normal-geometry histograms.
2. **Why useful for assembly?** Adjacent fragments share original-surface
   context near the break; similarity partially recovers that.
3. **Ambiguities remaining?** Symmetric/repeated statue regions and flat patches
   produce near-identical descriptors.
4. **False matches reduced how?** RANSAC geometric consistency + mutual NN, not
   descriptor distance alone.
5. **Assumptions?** That matching regions are *similar*.
6. **If it fails?** Complementary break surfaces are missed — the expected,
   informative failure.
7. **Similarity or compatibility?** Strictly similarity — the defining limit.
8. **Generalisation to unseen objects?** Fully — hand-crafted descriptors have
   no training set, so Phase 4 is our object-agnostic reference point.

---

## Deliverables

### 1. Descriptors (Phase 4.1)

| Descriptor | Dim | Patches | Pooling | Notes |
|-----------|-----|---------|---------|-------|
| FPFH | 33 | 6,822 | mean | via Open3D |
| SHOT | 352 | 6,822 | mean | from-scratch LRF + spherical histogram |

Stored per fragment at `baseline_results/descriptors/{fpfh,shot}/<frag>.npz`
(L2-normalised rows, atomic writes).

### 2. Retrieval & Separability (Phase 4.2)

| Metric | FPFH | SHOT |
|--------|------|------|
| Pair ROC-AUC | **0.708** | 0.536 |
| Pair Average Precision | 0.711 | 0.568 |
| Ranking mAP | 0.109 | 0.086 |
| Precision@1 | 0.130 | 0.113 |
| High-overlap-positives AUC | 0.736 | 0.584 |
| Low-overlap-positives AUC | 0.707 | 0.558 |

Full metrics: `baseline_results/retrieval/{fpfh,shot}_retrieval.json`,
summary: `baseline_results/retrieval/retrieval_summary.json`.

**Interpretation.** FPFH clearly beats chance on pair separation but has low
absolute ranking power (mAP ~0.11, P@1 ~0.13): given a query patch, the true
cross-fragment partner is rarely the nearest neighbour. SHOT is barely above
chance here — the from-scratch hard-binned SHOT is less discriminative at this
scale. Both are the reference numbers a learned encoder (Phase 5) must beat.

### 3. Registration (Phase 4.3)

Fragment B is perturbed by a **known** random rigid transform (≤30°, ≤20 mm)
and recovered; error = residual rotation/translation of `T_est @ T_perturb`.

| Mode | Adjacent success | Median rot err | Median trans err | Control success |
|------|------------------|----------------|------------------|-----------------|
| **Global** (FPFH+RANSAC+ICP, unaided) | **0/11 (0%)** | 142° | 35 mm | 0/6 (0%) |
| **Contact** (GT-interface-seeded) | **10/11 (90.9%)** | 1.5° | 1.6 mm | — |

Per-pair details: `baseline_results/registration/per_pair/*.json`,
summary: `baseline_results/registration/fpfh_registration.json`.

**Interpretation.** Global matching *fails* because fractured fragments **abut**
at a thin interface rather than **overlapping** — RANSAC maximises body overlap
and snaps B into a wrong high-fitness pose (fitness 0.5–0.8 but ~142° error).
This is the Phase 1 "metrics ≠ correctness" lesson resurfacing at registration.
Contact-seeded recovery succeeds on 10/11 pairs (median 1.5°/1.6 mm). The **ICP
refinement must run on the contact interface only, not the full clouds** — a
full-cloud ICP re-introduces the same body-overlap bias and degrades every pair
(it flipped a clean 2° F5↔F6 solve into a 22° failure before the fix). The sole
remaining failure is **5-7 (only 7 contact points)** — a near-degenerate
interface that is physically too small to constrain a rotation.

---

## Validation Gate (all 4 PASSED ✅)

| Check | Value | Passed |
|-------|-------|--------|
| FPFH pair ROC-AUC ≥ 0.5 | 0.708 | ✅ |
| SHOT pair ROC-AUC ≥ 0.5 | 0.536 | ✅ |
| Registration adjacent(contact) > control | 0.909 > 0.0 | ✅ |
| Descriptors computed for all patches | 6,822 | ✅ |

Passing means the baseline is a **trustworthy reference point** (beats chance;
adjacent registration separable from control). It does **not** mean the baseline
is sufficient for assembly — the weak mAP and the global-registration failure
are the expected limitations that motivate the learned phases.

---

## Similarity vs Complementarity vs Distinctiveness vs Assembly Usefulness

- **Similarity** (what FPFH/SHOT measure): partial signal — pair AUC 0.71, but
  low ranking power.
- **Complementarity** (what assembly needs at the break): *not* captured by
  similarity descriptors; the near-flat stratified-AUC pattern is consistent
  with this blind spot.
- **Distinctiveness** (from Phase 3 review): ~11% flat + ~13% ambiguous patches
  give degenerate descriptors and drag down ranking metrics.
- **Assembly usefulness:** a patch being *similar* to another is neither
  necessary nor sufficient for it to be assembly-useful. Phase 4 makes this
  concrete: high descriptor similarity ≠ correct registration (global mode).

---

## Outputs

```
baseline_results/
├── baseline_report.json                     # top-level summary + validation gate
├── descriptors/
│   ├── fpfh/<frag>.npz   (×7)               # (M,33) L2-normalised
│   └── shot/<frag>.npz   (×7)               # (M,352) L2-normalised
├── retrieval/
│   ├── fpfh_retrieval.json                   # pair separability + ranking + strata
│   ├── shot_retrieval.json
│   └── retrieval_summary.json
└── registration/
    ├── fpfh_registration.json                # global + contact + control summaries
    └── per_pair/*_{global,contact}.json      # per-pair rot/trans err, fitness, ncorr
```
**Total size:** ~11 MB.

---

## Configuration Used

`config/baseline_geometry.yaml` — key parameters:
```yaml
descriptors: [fpfh, shot]
fpfh_radius_mm: 8.0          # matches Phase 2 patch radius
shot_radius_mm: 8.0
shot_cos_bins: 11             # -> 32*11 = 352 dims
patch_pooling: mean
registration_descriptor: fpfh
perturb_max_rotation_deg: 30.0
perturb_max_translation_mm: 20.0
registration_voxel_mm: 1.5
success_rotation_deg: 15.0
success_translation_mm: 10.0
min_pair_auc: 0.5             # validation gate
```

---

## Commands

### Run complete Phase 4 pipeline
```bash
cd /home/kira/Desktop/Healing_Stones
ulimit -v 8000000
PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml
```

### Run individual stages
```bash
PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml --stages descriptors
PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml --stages retrieval
PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml --stages registration
```

### Inspect results
```bash
cat baseline_results/retrieval/retrieval_summary.json | python3 -m json.tool
python3 -c "import json; d=json.load(open('baseline_results/baseline_report.json')); print('gate:', d['validation']['passed'])"
```

### Tests
```bash
PYTHONPATH=src python3 -m pytest tests/baseline_geometry/ -v
```

---

## Lessons Learned

1. **Fragments abut, they don't overlap.** Global FPFH+RANSAC — the textbook
   registration recipe — fails on fractured fragments because it maximises body
   overlap, not interface contact. Fitness/RMSE look great while the pose is
   ~142° wrong. Always score against a known transform, never fitness alone.
   **Corollary (found via visual inspection):** even when seeded from the correct
   contact estimate, refining ICP on the *full clouds* re-introduces the same
   body-overlap bias and degrades the pose — ICP refinement must be restricted
   to the **contact interface only**. Fixing this raised contact-mode recovery
   from 63.6% → 90.9% and median error from 6° to 1.5°.
2. **Similarity ≠ compatibility, empirically.** FPFH beats chance (AUC 0.71) yet
   its ranking mAP is only ~0.11 and it cannot register unaided. This is the
   concrete, measured motivation for a learned, complementarity-aware model.
3. **Contact size drives registration solvability.** With interface-only ICP,
   10/11 pairs recover to ≤5.3°. The sole failure (5-7, only 7 contact points)
   is a near-degenerate interface physically too small to constrain a rotation —
   a genuine data limit, not an algorithm bug.
4. **From-scratch SHOT is weak at this scale.** Hard binning (no quadrilinear
   interpolation) makes our SHOT barely above chance; a good baseline reference,
   but FPFH is the stronger classical descriptor here.
5. **This phase generates the Phase 5 hard-negative tool.** FPFH distance between
   non-adjacent patches can now mine the hard negatives Phase 3 could not (which
   produced 0). Highly-similar-but-non-adjacent pairs are exactly hard negatives.

---

## Readiness for Phase 5 (Patch Encoder)

Phase 5 must beat these baselines: **FPFH pair-AUC 0.708 and mAP 0.109**. The
learned encoder's value proposition is to (a) raise ranking power (mAP/P@1) and
(b) learn *complementarity* rather than similarity so that break-surface patches
match. Phase 4 also supplies FPFH descriptors that can seed hard-negative mining
for the contrastive/triplet training set.

**Phase 4 Status: ✅ COMPLETE — Ready for Phase 5: Patch Encoder**
