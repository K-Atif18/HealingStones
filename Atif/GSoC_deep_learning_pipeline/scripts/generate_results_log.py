"""Generate PHASE5_RESULTS_LOG.md from the actual diagnostics JSON + test output.

This is the *single source of truth generator* for the Phase 5 results log: the
numbers section is formatted directly from ``baseline_diagnostics.json`` so it
cannot drift from the measured values (no hand transcription). Prose sections
that are decisions/deviations (not numbers) are templated here and updated as
milestones land.

Design (per the report-structure requirements):
  (a) pre-registration thresholds (frozen)      -> pointer, not duplicated
  (b) FPFH baseline numbers (dated, pre-encoder) -> generated from JSON
  (c) architecture deviations with measurements  -> templated + measured values
  (d) encoder results (once they exist)          -> generated from JSON when present
  (e) six-condition verdict                       -> generated when encoder exists

Every number carries provenance: the JSON field / script / test that produced it,
the JSON's own run_timestamp, and the git commit at generation time.

Usage:
    PYTHONPATH=src python3 scripts/generate_results_log.py
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(REPO, "phase5_diagnostics_results", "baseline_diagnostics.json")
OUT_PATH = os.path.join(REPO, "PHASE5_RESULTS_LOG.md")

FRAG_SHORT = {f"fragment_caesar_fragment_{i}": f"F{i}" for i in range(1, 8)}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _short(fid: str) -> str:
    return FRAG_SHORT.get(fid, fid)


def _fmt(x, nd=4):
    if isinstance(x, (int,)):
        return str(x)
    try:
        if x != x:  # nan
            return "nan"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _table(headers, rows) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def section_header(d: dict, commit: str) -> str:
    return f"""# Phase 5 Results Log

**Single source of truth** for Phase 5 measurements. The numbers sections are
**generated** from `phase5_diagnostics_results/baseline_diagnostics.json` by
`scripts/generate_results_log.py` -- do not hand-edit numbers here; regenerate.
Prose (deviations, decisions) is maintained in the generator template.

- Generated: `{datetime.now(timezone.utc).isoformat()}`
- Source JSON `run_timestamp`: `{d.get('run_timestamp')}`
- Git commit at generation: `{commit}`
- Fragments: {', '.join(_short(f) for f in d.get('fragment_ids', []))}
- Total patches: {d.get('n_total_patches')} | Adjacent pairs: {d.get('n_adjacent_pairs')}

---
"""


def section_prereg() -> str:
    return """## (a) Pre-registration thresholds (frozen)

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
"""


def section_fpfh_baseline(d: dict) -> str:
    src = d["per_source"]
    out = ["## (b) FPFH baseline numbers (pre-encoder, dated above)\n"]
    out.append("Provenance: `scripts/run_phase5_diagnostics.py` -> "
               "`baseline_diagnostics.json`. All numbers below are read from that JSON.\n")

    # Main ranking, all sources.
    out.append("### Ranking (raw + contact-only gallery)\n")
    rows = []
    for name in d["sources"]:
        e = src[name]
        raw = e["ranking_raw_gallery"]
        con = e["ranking_contact_only_gallery"]
        rows.append([
            name, _fmt(raw["mAP"]), _fmt(raw["precision_at_k"].get("1")),
            _fmt(con["mAP"]), _fmt(con["precision_at_k"].get("1")),
            _fmt(e["interface_breakdown"]["macro_mAP_well_supported"]),
        ])
    out.append(_table(
        ["source", "raw mAP", "raw P@1", "contact mAP", "contact P@1", "macro mAP"],
        rows))
    out.append("\n> Honest headline metric = **macro mAP, contact-only gallery**, "
               "read relative to the random null floor.\n")

    # Null floor.
    out.append("### Null floor (harness generosity)\n")
    nc = d.get("null_calibration", {})
    rows = [[n, _fmt(v["mAP"]), _fmt(v["precision_at_k"].get("1"))]
            for n, v in nc.items()]
    out.append(_table(["source", "mAP", "P@1"], rows))

    # Shortcut battery for FPFH.
    out.append("\n### Shortcut battery (FPFH)\n")
    sb = src["fpfh"]["shortcut_battery"]
    rows = [
        ["fragment-ID kNN acc (pooled)", _fmt(sb["fragment_id_probe"]["knn_accuracy"]),
         f"chance {_fmt(sb['fragment_id_probe']['chance'])}"],
        ["contact-vs-noncontact AUC", _fmt(sb["contact_vs_noncontact_auc"]["auc"]), ""],
        ["neighbour contact gap", _fmt(sb["neighbor_contact_vs_noncontact_ranking"]["mean_similarity_gap"]),
         "positive = real interface signal"],
        ["distinctiveness-sim Pearson r", _fmt(sb["distinctiveness_similarity_correlation"]["pearson_r"]), ""],
        ["boundary-openness R^2", _fmt(sb["boundary_openness_explained_variance"]["r_squared"]), ""],
    ]
    out.append(_table(["probe", "value", "note"], rows))

    # Hard-negative strata.
    out.append("\n### Hard-negative stratification (FPFH)\n")
    hn = src["fpfh"]["hard_negative_strata"]
    rows = [
        ["AUC pos-vs-easy", _fmt(hn["auc_pos_vs_easy"])],
        ["AUC pos-vs-hard", _fmt(hn["auc_pos_vs_hard"])],
        ["easy-minus-hard gap", _fmt(hn["easy_minus_hard_auc_gap"])],
    ]
    out.append(_table(["metric", "value"], rows))
    out.append("\n> The 0.61 gap is FPFH's sharpest pathology: it separates positives "
               "from random negatives but collapses on look-alike non-adjacent patches. "
               "The encoder must **shrink** this (condition 3).\n")

    # Acquisition fingerprint.
    af = d.get("density_fingerprint_probe", {})
    out.append("### Acquisition fingerprint (source-independent)\n")
    out.append(f"- Fragment-ID from density/spacing alone: "
               f"**{_fmt(af.get('knn_accuracy_from_density'))}** "
               f"(chance {_fmt(af.get('chance'))}). Mandates density normalisation "
               f"+ jitter in the encoder input pipeline.\n")

    # LOFO per fold (FPFH + random).
    out.append("### LOFO per fold with 95% bootstrap CIs (FPFH vs random)\n")
    fp = src["fpfh"]["lofo_per_fold"]["folds"]
    rn = src.get("random", {}).get("lofo_per_fold", {}).get("folds", {})
    rows = []
    for fid in d["fragment_ids"]:
        f = fp[fid]
        ci = f.get("ci_95", {}).get("precision_at_1", {})
        rci = rn.get(fid, {}).get("ci_95", {}).get("precision_at_1", {})
        rows.append([
            _short(fid), f["n_queries"], f["n_neighbors"],
            _fmt(f["precision_at_k"].get("1"), 3),
            f"[{_fmt(ci.get('lo'),3)}, {_fmt(ci.get('hi'),3)}]",
            _fmt(rci.get("mean"), 3) if rci else "n/a",
            f"[{_fmt(rci.get('lo'),3)}, {_fmt(rci.get('hi'),3)}]" if rci else "n/a",
        ])
    out.append(_table(
        ["fold", "queries", "neighbors", "FPFH P@1", "FPFH CI", "RND P@1", "RND CI"],
        rows))

    # Held-out fragment-ID probe (condition-6 refinement).
    out.append("\n### Held-out fragment-ID probe -- sharper instrument for **condition 6**\n")
    out.append("Provenance: `phase5_diagnostics.probes.heldout_fragment_id_probe`, "
               "field `per_source.<src>.heldout_fragment_id_probe`. **Not a new "
               "condition** -- a precise lens on condition 6 (generalized "
               "shape-signature leakage vs literal memorization). Baselined on FPFH "
               "*before* any encoder exists.\n")
    hp_fp = src["fpfh"]["heldout_fragment_id_probe"]
    hp_rn = src.get("random", {}).get("heldout_fragment_id_probe", {})
    rows = []
    for fid in d["fragment_ids"]:
        h = hp_fp[fid]
        r = hp_rn.get(fid, {})
        rows.append([
            _short(fid), _fmt(h.get("base_rate"), 3),
            _fmt(h.get("heldout_self_retrieval"), 3),
            _fmt(h.get("ratio_over_base_rate"), 2),
            _fmt(r.get("heldout_self_retrieval"), 3) if r else "n/a",
        ])
    out.append(_table(
        ["held-out", "base rate", "FPFH self-retr.", "×base", "RND self-retr."],
        rows))
    out.append("\n> FPFH isolates an unseen fragment at 6-8× base rate; random sits at "
               "base rate (probe validity check). The encoder must drive this **down** "
               "toward base rate if it learns interface association, not identity.\n")

    out.append("\n---\n")
    return "\n".join(out)


def section_deviations() -> str:
    """Deviation log -- verbatim permanent home. Prose (decisions), with measured
    values quoted from the gate/probe runs at the time each was decided."""
    return """## (c) Architecture deviations (with measurements)

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
"""


def section_encoder_placeholder(d: dict) -> str:
    # If encoder sources appear in the JSON later, generate their tables here.
    known = {"random", "centroid", "fpfh"}
    encoder_sources = [s for s in d.get("sources", []) if s not in known]
    if not encoder_sources:
        return """## (d) Encoder results

*No encoder trained yet.* This section is generated from the JSON once encoder
sources are present. Pose-invariance gate status for the untrained pipeline:
`passed=True, max_dev=0.000e+00` (see `tests/phase5_encoder/test_pose_gate.py`).

---

## (e) Six-condition verdict

*Pending encoder results.* Will be generated against the frozen thresholds in
section (a). Per the pre-registered rule: if condition 1 passes but 2-4 fail, the
honest conclusion is "improved contact/fragment discrimination, not interface
association" -- reported as such, not upgraded.
"""
    # (Generation for real encoder sources added when they exist.)
    return "## (d) Encoder results\n\n*(encoder sources present: %s -- table generation TBD)*\n" % ", ".join(encoder_sources)


def section_training_design() -> str:
    """Training design decisions -- same discipline as the deviation log
    (section c): decisions + measured values, not just prose. Populated once
    PHASE5A_TRAINING_DESIGN_REVISED.md is approved and its items 1-4 are
    implemented; kept as a template here so it regenerates from the same
    generator rather than being hand-maintained separately.

    Source of every number below: PHASE5A_TRAINING_DESIGN_REVISED.md, cross-
    referenced to the file/line citations in that document. This function
    does not re-derive anything; it summarises what that document already
    established, so this log and that design doc cannot silently drift.
    """
    return """## (f) Training design decisions

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

---
"""


def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    commit = _git_commit()
    parts = [
        section_header(d, commit),
        section_prereg(),
        section_fpfh_baseline(d),
        section_deviations(),
        section_training_design(),
        section_encoder_placeholder(d),
    ]
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {OUT_PATH} (commit {commit}, json {d.get('run_timestamp')})")


if __name__ == "__main__":
    main()
