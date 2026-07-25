# Phase 5 Pre-Registration

**Status:** written *before* running the diagnostics and *before* training any
encoder. Purpose: fix the interpretation rules in advance so a future positive
result cannot be rationalised after the fact. Numbers below are decision
thresholds, not observations.

Phase 5 target level (from the supervision-hierarchy analysis): **Level 3 —
interface association / neighbour retrieval.** Phase 5 is a *retrieval* phase,
not a correspondence phase. The labels cannot supervise correspondence (Level 4)
or complementarity (Level 6), so those are explicitly out of scope and out of
claim.

The bar to beat (Phase 4 FPFH, from `baseline_results/retrieval/retrieval_summary.json`):
- ranking **mAP = 0.109**, **P@1 = 0.13**, pair ROC-AUC = 0.708.

---

## Reporting rule (applies to every result, including FPFH and the encoder)

Every retrieval number is reported **relative to the null floor**, never in
isolation. The null floor is the mAP/P@1 achieved by *random* embeddings on the
same harness; it is non-zero purely because each query has a large ground-truth
set (Cartesian-product positives). "Beating FPFH" only counts if the margin over
the **null floor** also grows — a method that barely clears the floor has learnt
little regardless of its absolute mAP.

The **headline honest metric** is:
> macro-averaged mAP over well-supported interfaces, on the **contact-only
> gallery**, under **leave-one-fragment-out**.
Raw-gallery micro mAP is reported too, but is treated as optimistic.

---

## Diagnostic thresholds and verdicts

Thresholds are expressed relative to the FPFH baseline once measured; the
directional rules below are fixed now.

### 1. Null / weak-baseline calibration
- **Expectation:** random < centroid < FPFH on mAP. If random ≈ FPFH, the
  harness is saturated and mAP is not a usable discriminator → escalate to
  contact-only + macro metrics only.
- **Encoder pass condition (later):** encoder mAP must exceed FPFH by a margin
  **at least as large as** FPFH's own margin over the centroid baseline.

### 2. Shortcut battery (per source)

| Probe | "Genuine interface learning" | "Shortcut / memorization" |
|---|---|---|
| **Fragment-ID kNN accuracy** | near chance (≈ 1/7 = 0.143), or at least not the dominant signal | ≫ chance (e.g. > 0.5) → identity is encoded → **Shortcut 1 flag** |
| **Contact-vs-noncontact AUC** | moderate (contact-ness is a needed substrate) | if this is high **and** ranking collapses on contact-only gallery → **Shortcut 2** (contact detection masquerading as association) |
| **Neighbour contact-vs-noncontact gap** | clearly **positive** (neighbour's break-band ranked above its sculpted surface) | ≈ 0 → **fragment-affinity** signature (Shortcut 1) |
| **Distinctiveness–similarity Pearson r** | weak | strongly negative (e.g. \|r\| > 0.5) → similarity tracks roughness → **Shortcut 3** |
| **Boundary-openness R²** | low–moderate | high (e.g. > 0.5) → embedding ≈ edge detector → **Shortcut 6** |

### 3. Hard-negative stratification
- **`auc_pos_vs_easy` vs `auc_pos_vs_hard`:** a large positive gap
  (easy ≫ hard) means the representation separates positives from *random*
  negatives but not from *look-alike non-adjacent* patches → similarity, not
  compatibility.
- **Encoder pass condition (later):** the encoder must **shrink** the
  easy-minus-hard AUC gap relative to FPFH. Beating FPFH on easy negatives while
  the hard gap stays the same is *not* progress on assembly.

### 4. Leave-one-fragment-out (per fold, never averaged, n=7)
- Report every fold separately. Folds for peripheral/tiny-contact fragments
  (**F6, F7**; F7 tied to the ~7-point F5↔F7 contact) are **low-confidence** and
  interpreted apart from well-supported folds (F1/F4/F5 hubs, F2, F3).
- **Memorization verdict:** if raw-gallery mAP is high but **well-supported LOFO
  folds collapse toward the null floor**, the result is fragment memorization,
  not interface learning. LOFO is the primary defence against Shortcut 1/5.
- **Encoder pass condition (later):** the encoder must beat FPFH on the
  **majority of well-supported folds**, not just in aggregate.

#### 4b. Bootstrap-CI refinement (added after measuring the FPFH baseline)
Per-fold estimates are noisy at these query counts (2–4 true neighbours per
query), so point-estimate margins are not trusted. Each fold now carries a 95%
percentile bootstrap CI (resampling queries) for mAP and P@1.

Measured FPFH-vs-random-null separation (95% CIs):
- **P@1 CIs separate from the null on F1–F4 only.** F5's FPFH P@1 CI
  [0.056, 0.094] **overlaps** random's [0.038, 0.073] → F5 is *not* a
  demonstrated win even for the baseline on P@1.
- mAP CIs are tighter and separate on F1–F5 (+F6), but F5's P@1 ambiguity means
  FPFH ranks F5 partners only marginally and essentially never top-1.
- F7: FPFH P@1 = 0.000 [0,0], mAP CI *below* random → genuine degeneracy.

**Refined well-supported set for the pass decision: F1, F2, F3, F4.** F5 is
reported but reclassified "signal-ambiguous" (baseline itself can't clear noise
on P@1) and does **not** count toward pass/fail. F6/F7 remain low-confidence.

**Refined encoder pass condition (LOFO):** beat FPFH with **non-overlapping 95%
CIs on the majority of {F1,F2,F3,F4}**, on **both P@1 and mAP**. At encoder-eval
time the decisive test is a **paired bootstrap** on the per-query metric
*difference* (encoder − FPFH) over the identical query/gallery set — strictly
more powerful than comparing two independent CIs — with the pass requiring the
paired-difference CI to exclude zero.

### 5. Acquisition fingerprint
- **Fragment-ID accuracy from density/spacing alone:** if ≫ chance, per-fragment
  capture artifacts leak identity. Mitigation is mandatory in the encoder input
  pipeline (resampling + density normalisation + jitter) regardless of the FPFH
  reading.

### 6. Pose-invariance (encoder only; global-leakage gate)
- The encoder's full input+forward pipeline must satisfy
  `pose_invariance_check.passed == True` (max embedding deviation ≤ tol under
  random SO(3)+translation). **If this fails, training does not run.** This is
  the mechanical enforcement of the "no global-position leakage" rule.

---

## Overall Phase 5 success definition (pre-registered)

A Phase 5 encoder result will be called a **genuine Level-3 success** only if
**all** of the following hold:

1. Beats FPFH on the **honest headline metric** (macro mAP, contact-only
   gallery) by the margin rule in §1.
2. Wins on the **majority of well-supported LOFO folds** (§4).
3. **Shrinks** the easy-vs-hard AUC gap vs FPFH (§3).
4. Shows a **positive neighbour contact-vs-noncontact gap** (§2 positive
   control).
5. Passes the **pose-invariance gate** (§6).
6. Does **not** exhibit a dominant fragment-ID probe or boundary-openness R²
   that fully explains the ranking gain (§2).

If (1) holds but (2)–(4) fail, the honest conclusion is: *"the encoder improved
contact detection and/or fragment discrimination, not interface association."*
That is a real but weaker finding and must be reported as such.

Claims that remain **out of bounds** for Phase 5 regardless of results:
point/patch correspondence (Level 4–5), complementarity (Level 6), and
generalisation to unseen objects (needs the deferred synthetic multi-object
dataset).
