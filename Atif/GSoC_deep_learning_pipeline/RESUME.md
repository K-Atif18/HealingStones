# Healing Stones — Resume Notes

Last updated: Phase 4 COMPLETE (baseline geometry — FPFH/SHOT + registration, validated).

## Project
Research-grade 3D fragment assembly system for a fractured Caesar statue.
Goal: given only fractured fragments, identify neighbors, find matching regions,
estimate transforms, and reconstruct the original — without the original at inference.
The full model is used ONLY for training data / ground truth.

Source data in `data/`:
- `caesar_full_model.ply` (837,781 verts, mesh + RGB, units = mm, ~135 mm tall)
- `caesar_fragment_1.ply` ... `caesar_fragment_7.ply` (dense meshes, up to 1.12M verts)

## PHASE 1 — Dataset Foundation: COMPLETE ✅
Code: `src/dataset_foundation/` (16 modules). Tests: `tests/` — 85 passing
(27 Hypothesis property tests + unit/integration/smoke).

Final validated run:
- All 7 fragments aligned to the model, visually confirmed correct.
- Reconstruction checkpoint PASSED: RMSE 4.60 mm, coverage 1.0.
- Outputs in `dataset/` (fragments/ transforms/ normals/ normalized/ metadata/).

### How to run Phase 1
Full pipeline (memory-safe; ~0.5 GB, seconds):
    PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml
Labelled 3D viewer (shows fragment names):
    PYTHONPATH=src python3 scripts/visualize.py --interactive     # or --save for a PNG
Assisted manual alignment (point-picking + ICP, merges into config):
    PYTHONPATH=src python3 scripts/manual_align.py --fragments fragment_caesar_fragment_2,...

### Important lessons learned in Phase 1 (carry into later phases)
1. The full model + fragments are VERY dense (10^5–10^6 pts). The pipeline
   downsamples the model and each fragment to `voxel_size_mm` up front, or it
   OOMs a 16 GB machine. Keep per-point ops on downsampled clouds.
2. Automatic FPFH+RANSAC+ICP alignment of partial fragments to a symmetric
   whole is UNRELIABLE: fragments snap to geometrically-similar wrong regions and
   still score high fitness / low RMSE. Metrics ≠ correctness; visual check is
   the real gate. Aligner uses multi-start RANSAC + multi-scale ICP to help, but
   the reliable ground-truth path is assisted manual alignment (scripts/manual_align.py)
   writing `precomputed_transforms` into config/default.yaml, which the pipeline
   imports + validates. Fragments 1,2,3,6 are currently manually pinned.
3. Config knobs live in `config/default.yaml`. Backups saved as `config/default.yaml.bak.*`.

## PHASE 2 — Patch Generation: COMPLETE ✅
Code: `src/patch_generation/` (15 modules). Tests: `tests/patch_generation/` — 51 passing
(16 Hypothesis property tests + unit tests).

Final validated run:
- All 7 fragments converted to overlapping geometric patches.
- Total: 6,822 patches from 16,833 fragment points.
- Validation checkpoint PASSED: 100% coverage, all sizes within bounds.
- Outputs in `patches/` (fragment_*/patches.npz + metadata.json + dataset.json).

### How to run Phase 2
Full pipeline (~22 seconds):
    PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml

### Patch Statistics
| Fragment | Points | Patches | Coverage | Mean Size |
|----------|--------|---------|----------|-----------|
| fragment_1 | 2,973 | 1,000 | 100% | 99.2 pts |
| fragment_2 | 2,376 | 1,000 | 100% | 99.8 pts |
| fragment_3 | 1,926 | 1,000 | 100% | 100.0 pts |
| fragment_4 | 4,371 | 1,000 | 100% | 100.9 pts |
| fragment_5 | 2,808 | 1,000 | 100% | 100.0 pts |
| fragment_6 | 822 | 822 | 100% | 101.4 pts |
| fragment_7 | 1,557 | 1,000 | 100% | 101.1 pts |
| **TOTAL** | **16,833** | **6,822** | **100%** | **~100 pts** |

Configuration: 8mm radius, max 128 pts/patch, 1000 FPS centers, full coverage required.

### Important lessons learned in Phase 2
1. Farthest Point Sampling (FPS) provides excellent coverage — all fragments reach 100%
   with just 1000 centers (or fewer for small fragments like fragment_6 with 822 points).
2. 8mm radius at 1.5mm point spacing yields ~100 points/patch, manageable for PointNet++.
3. Controlled overlap (27-35% mean) provides multiple viewpoints for robust learning.
4. Size capping (max 128 pts) handles dense regions without losing geometry.
5. Patches stored in compressed .npz format (~12 MB total) with full metadata for Phase 3.

## PHASE 3 — Ground Truth Generation: COMPLETE ✅
See `PHASE3_COMPLETE.md`. 1.44M labelled pairs (719K positive + 719K random
negative), 11 adjacent fragment pairs, contact regions for each. Compact
`phase3_final_review/pairs.npz` (7.6 MB, 104× smaller than JSON) is the
canonical pair table for downstream phases. Hard negatives = 0 (documented
limitation; to be mined via FPFH similarity in Phase 4/5).

### How to run Phase 3
    PYTHONPATH=src python3 scripts/phase3_pipeline.py --config config/ground_truth.yaml

## PHASE 4 — Baseline Geometry: COMPLETE ✅
Spec-less implementation package: `src/baseline_geometry/` (7 modules).
Tests: `tests/baseline_geometry/` — 14 passing. Config: `config/baseline_geometry.yaml`.

Final validated run (~75 s, validation gate PASSED 4/4):
- FPFH (33-D) + SHOT (352-D) descriptors for all 6,822 patches.
- Pair separability: FPFH ROC-AUC 0.708, SHOT 0.536 (both beat chance).
- Ranking retrieval: FPFH mAP 0.109, P@1 0.13 (cross-fragment).
- Registration backend: contact-seeded recovery 90.9% (10/11, median 1.5°/1.6mm) vs 0% non-adjacent control.
- Global (unaided) FPFH+RANSAC+ICP FAILS on adjacent pairs (0%, ~142° median) —
  a real, expected result (fragments abut, they don't overlap).
- Outputs in `baseline_results/` (descriptors/, retrieval/, registration/, baseline_report.json).

### How to run Phase 4
Full pipeline (~75 s):
    PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml
Individual stages:
    PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml --stages descriptors
    PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml --stages retrieval
    PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml --stages registration

### Important lessons learned in Phase 4 (carry into later phases)
1. **Fragments abut, they don't overlap.** Global FPFH+RANSAC maximises body
   overlap → snaps to wrong high-fitness poses (fitness 0.5–0.8, ~142° error).
   Score against a KNOWN transform, never fitness/RMSE alone. Interface
   localisation — not the rigid math — is the hard problem (Phases 6–7 target it).
2. **Similarity ≠ compatibility, measured.** FPFH beats chance (AUC 0.71) but
   ranking mAP is only ~0.11 and it can't register unaided. This is the concrete
   bar Phase 5's learned, complementarity-aware encoder must beat.
3. **Contact size drives solvability + interface-only ICP.** With ICP refinement
   restricted to the contact interface (NOT the full clouds — full-cloud ICP
   re-introduces body-overlap bias and degrades poses), 10/11 pairs recover to
   ≤5.3°. Only 5-7 (7 contact pts) fails — a genuine near-degenerate interface.
4. **Phase 4 supplies the Phase 5 hard-negative tool:** FPFH distance between
   non-adjacent patches mines the hard negatives Phase 3 could not generate.

## Remaining phases (roadmap)
5. Patch Encoder (PointNet++ + contrastive/triplet embeddings)  <-- NEXT
6. Fragment Retrieval (embedding DB, neighbor ranking)
7. Correspondence Learning (local matches + confidence)
8. Transformation Estimation (rigid registration, outlier rejection)
9. Assembly Graph (pose graph, global optimization, final reconstruction)
Each phase = its own spec + hard validation gate before advancing.

## Command references
- **QUICKSTART.md** — One-page cheat sheet for common commands
- **COMMANDS.md** — Complete reference for Phase 1 & 2 (all commands, testing, visualization)

## Env
Python 3.10.12. Installed: numpy(<2), scipy, hypothesis, pyyaml, pytest, open3d 0.19.
torch pinned in requirements.txt but NOT installed (not needed until a DL phase).
Run heavy commands under `ulimit -v 8000000` as a safety cap.
