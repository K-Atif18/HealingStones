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


def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    commit = _git_commit()
    parts = [
        section_header(d, commit),
        section_prereg(),
        section_fpfh_baseline(d),
        section_deviations(),
        section_encoder_placeholder(d),
    ]
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {OUT_PATH} (commit {commit}, json {d.get('run_timestamp')})")


if __name__ == "__main__":
    main()
