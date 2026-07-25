# Healing Stones — Phase 5 Research Context (Resume Point)

Last updated: 2026-07-19. Read this before doing anything else in a new session.
This file is the *research* memory (why we're doing things). See
`PHASE5_PREREGISTRATION.md` in the repo for the *pre-committed thresholds* (the
rules we agreed not to bend after seeing results). This file explains how we
got to those rules and what's next.

---

## 1. Project in one sentence

Research-grade 3D fragment assembly for fractured cultural-heritage artifacts
(currently: a real Caesar statue, 7 fragments), where the complete object is
available only for training/ground-truth, never at inference. The system must
end-to-end: retrieve neighboring fragments → localize matching regions →
estimate rigid pose → assemble.

## 2. The one sentence that governs every decision

**Similarity ≠ Complementarity.**

Phase 4 proved this experimentally: FPFH (a handcrafted geometric descriptor)
gets ROC-AUC 0.708 on pair classification (beats chance) but mAP only 0.109 on
retrieval ranking, and 0% success on global registration because fragments
*abut* rather than *overlap* (classical registration assumes overlap; ours
don't have it). Once the correct interface is known, contact-seeded
registration succeeds ~91% of the time. So: **registration is not the
bottleneck, interface/neighbor localization is.** Everything from Phase 5
onward is judged against this.

## 3. Four concepts that must never be collapsed into one

- **Similarity** — two patches look alike (e.g. two flat cheek regions).
  Does NOT imply assembly.
- **Complementarity** — two patches physically fit (convex ↔ concave at a
  fracture). This is what assembly actually requires. Similarity descriptors
  struggle here.
- **Distinctiveness** — how unique a patch's geometry is. High-distinctiveness
  patches (fracture ridges, sharp features) carry more assembly information
  than flat/symmetric/repeated regions.
- **Assembly usefulness** — the only quantity that actually matters: does this
  patch help retrieve the correct neighbor / localize the interface / recover
  the pose? A patch can be similar AND distinctive and still be assembly-
  useless.

## 4. What Phases 1–4 established (completed, don't redo)

- **Phase 1** (dataset foundation): Caesar statue + 7 fragments loaded, aligned
  to a common frame, normals computed, geometry normalized. Fragments
  reconstruct correctly under ground-truth transforms.
- **Phase 2** (patches): 6,822 overlapping patches via FPS + radius
  neighborhoods (radius 8mm, mean spacing ≈1.22mm). Full surface coverage
  verified.
- **Phase 3** (ground truth): 21 fragment pairs analyzed → 11 adjacent, 10
  separated (adjacency threshold 3.67mm, clean separation). Contact regions
  detected (5–938 points; F5↔F7 is a real but tiny ~5-7 point edge contact —
  a known hard case throughout). 719,279 positive pairs / 719,270 random
  negatives generated. **Critical caveat, never forget**: positives encode
  "both patches belong to the same contact/interface region" (contact-region
  co-membership), NOT direct patch-to-patch correspondence. This is enough to
  supervise retrieval (Level 3) but NOT correspondence (Level 4) or
  complementarity (Level 6) — see the supervision hierarchy in section 6.
  Distinctiveness distribution: 56.3% highly distinctive, 19.1% moderate,
  13.5% ambiguous, 11.2% flat.
- **Phase 4** (geometric baselines, the bar to beat): FPFH 33-dim (ROC-AUC
  0.708, mAP 0.109, P@1 0.13), SHOT 352-dim (ROC-AUC 0.536, weaker). Global
  registration (FPFH+RANSAC+ICP): 0% success — fragments abut, don't overlap.
  Contact-seeded registration: ~91% success, median rotation error ≈1.5°,
  translation ≈1.6mm. **Conclusion: the rigid solver works fine; finding the
  interface is the unsolved problem.**

## 5. What changed in the plan going into Phase 5 (the actual "tweaks")

Original master-prompt plan treated Phase 5 as "learn geometric embeddings,
build a PointNet++ baseline, do contrastive/triplet learning, evaluate
clustering." That was too coarse. Before writing any model code we:

1. **Named the real target precisely**: Phase 5 is not "learn similarity
   better" and not "learn correspondence" and not "learn complementarity." It
   targets **Level 3 — interface association / neighbor retrieval** on a
   supervision hierarchy (see section 6) derived directly from what Phase 3's
   labels can and cannot teach.
2. **Pre-registered thresholds before touching data** — wrote down in
   `PHASE5_PREREGISTRATION.md` what would count as pass/fail *before* running
   any diagnostic, specifically so a good-looking number couldn't be
   rationalized after the fact.
3. **Ran a full shortcut-risk analysis before training anything.** Enumerated
   every way a model could look successful under current labels without
   learning real interface association (fragment-ID memorization, contact
   detection, roughness/distinctiveness correlation, interface-size bias,
   acquisition/density fingerprinting, boundary/edge-topology detection,
   orientation leakage). Built a detector + ablation for each (see section 8).
4. **Built the diagnostic instrumentation as reusable infrastructure**
   (`src/phase5_diagnostics/`), not one-off scripts — it runs on *any*
   `PatchDescriptors` source (random / centroid / FPFH now, the trained
   encoder later) and is meant to run automatically on every future
   checkpoint.
5. **Measured the FPFH baseline through this exact instrumentation** so
   Phase 5A has a same-harness number to beat, not just the Phase 4 press
   release numbers.
6. **Caught and fixed a real bug in our own instrumentation** (see section 9)
   rather than trusting a "looks complete" report — LOFO initially returned
   `n_queries: 0` on every fold due to a gallery/query key mismatch; this was
   found, root-caused, fixed, tested, and regression-guarded before being
   trusted.
7. **Added bootstrap confidence intervals per LOFO fold** because point
   estimates at 2–4 true neighbors per query are too noisy to trust
   directly — this changed the pass bar itself (see section 10).

## 6. The supervision hierarchy (why Level 3 is the right target)

This is the single most important framing for Phase 5 and should be re-derived
if a new collaborator questions "why not just learn correspondence directly."

| Level | What it means | Supervised by current labels? |
|---|---|---|
| 0 | Surface recognition | Yes |
| 1 | Contact-region recognition | Yes, strongly |
| 2 | Interface identity | Partially |
| **3** | **Interface association / neighbor retrieval** | **Yes, strongly — THE PHASE 5 TARGET** |
| 4 | Patch correspondence | No — needs new GT |
| 5 | Point correspondence | No — needs dense correspondence GT |
| 6 | Complementarity | No — current labels are *symmetric*, complementarity is *asymmetric*; a symmetric contrastive loss could even actively reward similarity-based shortcuts instead |
| 7 | Transformation estimation | Phase 8 |
| 8 | Assembly reasoning | Phase 9 |

Consequence: **H5 (complementarity-aware learning beats similarity-aware
learning) cannot be tested in Phase 5.** It's the long-term thesis of the
project but requires new, asymmetric supervision that doesn't exist yet.
Phase 5's honest scientific claim, if successful, is bounded to: "learned
embeddings improve interface association / neighbor retrieval over
handcrafted descriptors." It cannot claim to have learned correspondence,
complementarity, or assembly reasoning.

## 7. Research hypotheses (pre-registered, Phase 5 relevant ones only)

- **H1**: learned embeddings beat FPFH on retrieval ranking (mAP/P@K/R@K).
  Baseline to beat: mAP 0.109 (raw gallery, per Phase 4). *Refined:* must beat
  FPFH on the honest/harder metrics too, see section 10.
- **H2**: distinctive patches carry more assembly info than ambiguous ones —
  test via retrieval stratified by distinctiveness class.
- **H3**: better retrieval → better downstream registration success (compare
  FPFH-seeded vs learned-seeded registration).
- **H4**: neighbor retrieval should be solved before correspondence
  estimation (justified directly by the Phase 4 bottleneck finding).
- **H5**: complementarity-aware learning beats similarity-aware learning —
  **out of scope for Phase 5**, deferred to a later phase requiring new
  supervision.

## 8. Shortcut taxonomy (must be checked before believing any Phase 5A result)

Each of these was reasoned through in detail (detector + ablation for each)
and most are now instrumented and measured against FPFH as the baseline
reading:

1. **Fragment-ID memorization** — the most dangerous one, because Caesar's
   topology is small and fixed (7 fragments, 11 adjacent edges); an
   identity→neighbor lookup table would look like a great retrieval result
   without learning anything about interfaces. **Measured**: fragment-ID kNN
   probe on FPFH embeddings = **0.892** accuracy (chance = 0.143, i.e. 6.2×
   chance). This is high enough to be a live risk for the encoder too, and it
   is why LOFO is the mandatory primary gate, not a nice-to-have.
2. **Contact-region detection masquerading as association** (Level 1 passed
   off as Level 3) — measured via contact-vs-noncontact AUC = 0.731 for FPFH,
   plus a contact-only-gallery retrieval re-test. This is real signal but
   necessary-not-sufficient; must not be the *whole* explanation for a good
   number.
3. **Roughness/distinctiveness-magnitude shortcut** — measured via Pearson r
   between embedding similarity and distinctiveness-score difference: **r =
   -0.40 for FPFH** (vs ≈0 for random/centroid). FPFH's notion of "similar" is
   substantially "both equally rough" — a direct, numeric confirmation of the
   similarity≠complementarity thesis, and a pathology the encoder must not
   inherit.
4. **Interface-size / co-occurrence bias** — large interfaces dominate
   positive counts, so micro-averaged mAP can look great from 2-3 big
   interfaces while small ones (esp. F5↔F7) silently fail. Countered by
   macro-averaging mAP across interfaces (equal weight per interface) instead
   of micro-averaging.
5. **Acquisition/density fingerprint** — per-fragment scan/density artifacts
   could leak identity even without global position. **Measured**: fragment-ID
   accuracy from density/spacing alone = **0.356** (2.5× chance). Real, and
   now a mandatory input-pipeline requirement (density normalization + jitter)
   for the encoder regardless of what FPFH shows.
6. **Boundary/open-edge topology shortcut** — contact patches sit at mesh
   boundaries; a model could detect "am I near an open edge" as a cheap proxy
   for contact. **Measured**: boundary-openness R² = **0.002** for FPFH —
   negligible, FPFH is not just an edge detector. Still must be re-checked for
   the learned encoder.
7. **Orientation/pose leakage through canonicalization** — a softer version of
   global-position leakage; if the local reference frame retains consistent
   orientation info, absolute pose sneaks back in. Guarded by a
   pose-invariance unit test (random SO(3)+translation must not change
   embeddings beyond tolerance) — **this gate blocks training if it fails**,
   and has already been validated to correctly fail a global-frame encoder and
   pass a distance-only one.

**Positive control that must hold for any future result to be meaningful**: a
true neighbor's *contact* patches should rank above that same neighbor's
*non-contact* patches. FPFH shows a small but real positive gap here (mean
similarity gap = 0.073 across 1,266 comparisons) — weak genuine interface
signal, not zero, not fragment-affinity-only.

## 9. A caught-and-corrected mistake (keep this honest in future summaries)

The diagnostics were first reported as "LOFO computed for all 7 folds." This
was **false** — every fold silently returned `n_queries: 0, mAP: NaN`. Root
cause: `positive_partner_map()` required the *query* patch to already be
present in the (filtered) gallery before creating a key, but LOFO's gallery
deliberately excludes the held-out fragment — so the held-out fragment's own
patches could never become valid query keys. Fixed by decoupling "query
eligibility" from "gallery membership" (only *candidate/partner* patches must
be in the gallery). Regression-tested (`test_lofo_folds_are_nonempty`). The
lesson generalized: **don't trust a diagnostics report until you've read the
actual JSON/test output, not just the prose summary** — this exact check
caught a real, otherwise-invisible false claim.

## 10. Current, CI-refined LOFO baseline (the actual pass bar for Phase 5A)

Per-fold, never averaged (n=7 is too small and too uneven to average
meaningfully — F6/F7 have only 2 known neighbors each and very thin contact
support). Measured through the fixed, tested LOFO harness with 95% percentile
bootstrap CIs (resampling queries) on both mAP and P@1:

| Fold | queries | neighbors | FPFH P@1 [95% CI] | Random P@1 [95% CI] | CI-separated? |
|---|---:|---:|---|---|---|
| F1 | 686 | 4 | 0.168 [0.140, 0.195] | 0.071 [0.054, 0.090] | **yes** |
| F2 | 516 | 3 | 0.194 [0.161, 0.229] | 0.064 [0.045, 0.085] | **yes** |
| F3 | 573 | 3 | 0.169 [0.140, 0.199] | 0.035 [0.021, 0.051] | **yes** |
| F4 | 544 | 4 | 0.184 [0.151, 0.219] | 0.077 [0.057, 0.101] | **yes** |
| F5 | 661 | 4 | 0.074 [0.056, 0.094] | 0.054 [0.038, 0.073] | no — overlap |
| F6 | 780 | 2 | 0.032 [0.021, 0.045] | 0.064 [0.047, 0.082] | no — FPFH *worse* (low-conf) |
| F7 | 347 | 2 | 0.000 [0.000, 0.000] | 0.081 [0.055, 0.112] | no — FPFH *worse* (low-conf) |

mAP CIs are much tighter (mAP averages the whole ranking; P@1 is a
hit/miss Bernoulli variable with more variance at these query counts) and
separate on F1–F5 (+F6), but F5's P@1 non-separation means FPFH ranks F5's
partners *somewhat* better on average while essentially never getting the
top-1 right — a real, specific nuance, not a contradiction.

**Refined well-supported set: F1–F4 only** (F5 demoted to "signal-ambiguous"
— even the baseline can't clear noise on P@1 there; it's reported but doesn't
count toward pass/fail). F6/F7 remain low-confidence, reported separately, and
F7 shows FPFH performing *below* the null floor (genuine degeneracy, matches
the known ~5-7 point F5↔F7 contact problem going back to Phase 3).

**Phase 5A encoder pass condition on LOFO (pre-registered, in
`PHASE5_PREREGISTRATION.md`)**: beat FPFH with non-overlapping 95% CIs on the
majority of {F1, F2, F3, F4}, on both P@1 and mAP. At encoder-evaluation time,
use a **paired bootstrap on the per-query metric difference** (encoder −
FPFH, same queries/gallery) rather than comparing two independent CIs — this
is strictly more statistically powerful and the plumbing for it
(`return_per_query`) already exists in `evaluate_ranking`.

## 11. Full pre-registered Phase 5 success definition (do not weaken this later)

A Phase 5 encoder result counts as genuine Level-3 success only if **all**
hold:
1. Beats FPFH on the honest headline metric (macro-averaged mAP over
   interfaces, contact-only gallery) by a margin at least as large as FPFH's
   own margin over the centroid baseline.
2. Wins on the majority of well-supported LOFO folds — now F1–F4 — via paired
   bootstrap, not raw point-estimate comparison.
3. Shrinks the easy-vs-hard negative AUC gap relative to FPFH (FPFH's gap:
   AUC 0.706 on easy/random negatives → 0.092, i.e. worse than chance, on
   FPFH-mined hard/look-alike negatives; gap = 0.614 — this is the sharpest
   existing evidence that FPFH separates positives from *random* junk but not
   from *plausible* look-alikes).
4. Shows a positive neighbor contact-vs-noncontact ranking gap (the positive
   control from section 8).
5. Passes the pose-invariance / no-global-leakage gate (hard blocker, training
   does not run if this fails).
6. Does not show a dominant fragment-ID probe or boundary-openness R² that
   fully explains the ranking gain.

If (1) holds but (2)-(4) fail, the honest, pre-committed conclusion is: *"the
encoder improved contact detection and/or fragment discrimination, not
interface association"* — a real but weaker finding, to be reported as such,
not spun.

Permanently out of scope for Phase 5, regardless of results: patch/point
correspondence (Level 4-5), complementarity (Level 6), generalization to
unseen objects (would need a deferred multi-object synthetic dataset — we
only have one real artifact right now).

## 12. Immediate next objective (where to resume)

**Phase 5A**: build a small PointNet retrieval encoder. Concretely, in order:
1. Torch install / environment setup.
2. Input pipeline: density normalization + resampling + jitter (mandatory,
   given the 0.356 density-fingerprint measurement) + pose canonicalization,
   with the pose-invariance gate wired in as a hard pre-training check.
3. Small PointNet encoder (deliberately simple first — Phase 5A/5B/5C is a
   progressive baseline strategy: PointNet → DGCNN → Point Transformer V3,
   not jumping straight to the most powerful architecture).
4. Train using current interface co-membership labels + hard negatives
   (FPFH-mined look-alikes, not just random negatives — the easy-vs-hard gap
   is the specific pathology to beat).
5. Export embeddings as `PatchDescriptors` and run through the *exact same*
   diagnostic battery already built (null calibration, shortcut battery,
   hard-negative strata, LOFO-with-CIs) — this infrastructure is reusable by
   construction, that was the point of building it first.
6. Judge against the full 6-condition success definition in section 11, not
   against mAP alone.

One open design note not yet resolved: given the fragment-ID probe is so high
(0.892) for FPFH, consider whether the encoder training needs
fragment-balanced sampling or a fragment-adversarial term to actively
discourage the identity shortcut, rather than relying solely on LOFO to catch
it after the fact. Flagged, not yet decided.

## 13. Key lessons to never re-derive from scratch

1. Similarity is not assembly compatibility — measured, not just argued
   (r = -0.40, easy/hard gap = 0.614).
2. Retrieval (Level 3) and correspondence (Level 4+) are different tasks with
   different supervision requirements; current labels cap us at Level 3.
3. Complementarity needs asymmetric supervision that doesn't exist yet — H5
   is deferred, not disproven.
4. Registration is not the bottleneck (91% success when interface is known);
   interface localization is.
5. A model can look successful on this dataset for at least 7 distinct wrong
   reasons (section 8) — every positive result must be checked against all of
   them before being believed.
6. n=7 fragments makes LOFO noisy — always report per-fold with CIs, never a
   single averaged LOFO number.
7. Trust the JSON/test output over the prose summary — we caught a real false
   claim ("LOFO computed for all 7 folds") this way once already.
8. Every future phase is judged against the Phase 4 baseline and the Phase 5
   pre-registration, not against "does the loss go down."

---

## 14. Improved resume prompt — paste this to start the next research session

Use this verbatim (or lightly edited) as the first message to a research-
partner AI (no-code, analysis-only role) in a new session, with this file
attached/pasted as context:

> I'm resuming a research project on learned 3D fragment assembly for
> fractured cultural-heritage artifacts (currently a real Caesar statue, 7
> fragments). Act as my research partner only — no code, no implementation.
> Your job is critique, hypothesis-checking, and identifying gaps in reasoning
> or measurement, the same way a skeptical PhD committee member would.
>
> Context attached: `PHASE5_RESEARCH_CONTEXT.md`. Read all of it before
> responding, especially section 5 (what changed going into Phase 5), section
> 8 (the shortcut taxonomy, all already measured against the FPFH baseline),
> and section 10-11 (the CI-refined LOFO baseline and the full pre-registered
> pass/fail definition).
>
> Governing rule for this whole project: similarity ≠ complementarity. Every
> claim must be evaluated against the four-concept framework (similarity /
> complementarity / distinctiveness / assembly usefulness) and against the
> six-condition Phase 5 success definition — never against a single metric
> like mAP or loss in isolation.
>
> Current state: Phases 1-4 complete and unchanged. Phase 5 pre-registration
> is written and locked (`PHASE5_PREREGISTRATION.md`). Diagnostic
> infrastructure is built, tested, and has already caught one real bug in
> itself (the LOFO empty-fold issue, section 9) — treat any future "diagnostics
> complete" claim from the coding AI with the same skepticism until verified
> against actual file output. We have not yet started Phase 5A (PointNet
> encoder) training.
>
> Do not let me (or the coding AI) relax the pre-registered thresholds after
> seeing a result. Do not accept "loss went down" or "AUC improved" as
> evidence of assembly-relevant learning on its own. If I report a result from
> the coding AI, your first job is to ask what shortcut from section 8 could
> explain it before accepting it as a real finding.
>
> Open item to help me think through first: whether Phase 5A training itself
> needs fragment-balanced sampling or a fragment-adversarial loss term to
> actively discourage the 0.892-probe fragment-ID shortcut *during* training,
> rather than relying on LOFO to catch it only after the fact. Help me reason
> through whether that's necessary before training, or whether it's
> premature and LOFO-as-a-post-hoc-gate is sufficient.
