# Phase 5A Training Design — Revised, round 2 (reasoning only, no training code)

**Status:** draft for review, incorporating a second review pass. Supersedes
the prior (rejected) design on the 7 points below, and corrects four citation
errors found on re-check of the first revision (§1, §2, §6 evidence blocks —
see the note at the end of each corrected line). Points 1, 3, 4 of the
*original* design (InfoNCE rationale, fold-aware sampler,
patience-5/cap-50/best-checkpoint) are retained unchanged and are not
re-argued here.

**Citation rule in force from this document onward:** every numeric parameter
below cites a `file:line`, a JSON field path, or a command whose output is
pasted. If a number has none of those, it is marked `UNVERIFIED — placeholder`
and training must not start until it is measured. This rule now applies
recursively to this document's own citations — round 1 had three wrong line
numbers despite following the rule in spirit; round 2 re-verified every line
number against the file with a fresh read/grep rather than trusting round 1's
citations.

---

## 0. What was wrong with the rejected design (one line each)

| # | Rejected claim | Why it was wrong | Source of the false premise |
|---|---|---|---|
| 1 | Condition 3 (easy-vs-hard gap) is "computed on the held-out fold" | No fold filter exists anywhere in the call path | Asserted, not read from `runner.py` |
| 2 | Hard negatives are looked up "up to H=4 per anchor" | `mine_hard_negatives` is a strict `argmin`, max 1/anchor | `H=4` came from an unwritten prior conversation, not a file |
| 3 | Training batch is 512 | Memory was measured per single encoded patch, not per training step | `PHASE5_RESULTS_LOG.md:155-160` measured a different quantity than the one being sized |

All three are corrected below (§1, §2, §3). §4–§7 address the sampler,
positive-distance policy, norm layer, and stopping rule.

---

## 1. Condition 3 — fix the instrument, then fix the training/eval split

**Objective.** Condition 3 of the pre-registration (`PHASE5_PREREGISTRATION.md`,
§3: *"the encoder must **shrink** the easy-minus-hard AUC gap relative to
FPFH"*) must be measured on data the encoder did not train on. Currently it
is not.

**Evidence (verified this session, re-reading the code fresh; three line numbers
corrected after a second citation check against the actual file):**
- `src/phase5_diagnostics/runner.py:91` — `hard_neg = rk.mine_hard_negatives(ctx, sources[fpfh_source_name], seed=seed)`. Mined **once**, from FPFH, across **all 7 fragments**. No held-out argument. *(Corrected from an earlier draft's `:100`, which is inside the `calibrate_null` call, not the mine call.)*
- `src/phase5_diagnostics/runner.py:146` — `entry["hard_negative_strata"] = rk.easy_vs_hard_separability(ctx, emb, hard_neg, seed=seed)`. The same fixed `hard_neg` list is handed to every source (random/centroid/FPFH/future encoder) with no per-fold call.
- `src/phase5_diagnostics/ranking.py:404-497` (`easy_vs_hard_separability`, up to `_bootstrap_ci` at line 498) — positives come from `pos_idx = np.where(pairs.labels == 1)[0]` at **line 425** *(corrected from an earlier draft's "~424")*, unrestricted; negatives come from `hard_negatives` as passed in. No `held_out` parameter exists in the signature (`ranking.py:404-411`).

Conclusion: if the encoder is trained using any of these same 2,100 mined
pairs (or the same positives), condition 3 as currently instrumented is a
training-set number, not a held-out one.

**Fix (instrument):**
1. Add a `held_out: Optional[str] = None` parameter to `easy_vs_hard_separability`. When set, filter both `pos_idx` and the `hard_negatives` list to exclude any pair touching `held_out` before scoring — i.e. reuse the same fragment-membership filter pattern `runner.py` already applies via `_contact_only_keep` (`runner.py:59-60`), applied to the pair's `fragment_A_idx`/`fragment_B_idx` (via `pairs.fragment_vocab`, same lookup `easy_vs_hard_separability` already does at `ranking.py:415-423`).
2. Call this from `lofo_per_fold` (`ranking.py:522-587`) once per fold, mirroring the pattern already used for `evaluate_ranking(..., query_keep=...)` inside that function, so condition 3 gets a per-fold number exactly like conditions 1/2/4 already do.

**Fix (training/eval split — resolves this item together with §2):**
- Split the FPFH-mined 2,100 pairs (`mine_hard_negatives`, `per_fragment=300` × 7 fragments, `ranking.py:345-346`) into **frozen evaluation-only**, never shown to the training loop in any form.
- The training loop's hard negatives come from a **separate, on-the-fly, encoder-mined, fold-restricted** mechanism (promoted to primary per §2) — mined fresh each epoch from the current encoder's own embedding, restricted to the fold's training fragments only.
- This makes the split explicit and answers the review's demand directly: **training-visible pairs are (a) Phase 3 positives restricted to the training fold's fragments, and (b) on-the-fly encoder-mined hard negatives restricted to the same fold. Training never sees the frozen FPFH-mined 2,100, and never sees any pair touching the held-out fragment.**

---

## 2. Hard-negative mechanism — restated honestly, promoted on-the-fly path to primary

**Evidence:**
- `src/phase5_diagnostics/ranking.py:381` — `j = int(np.argmin(dists))` *(corrected from an earlier draft's `:373`, which is `non_adj_rows = np.where(non_adj_mask)[0]`; confirmed via `grep -n argmin ranking.py`, single match at line 381)*, a strict argmin over `dists = np.linalg.norm(gallery.desc[non_adj_rows] - qvec, axis=1)` (the line immediately preceding). One nearest non-adjacent patch per sampled anchor. No `top_k` argument exists anywhere in `mine_hard_negatives`'s signature (`ranking.py:345-350`).
- Measured directly (values as given in the review, reproducing the same computation `mine_hard_negatives` performs): 2,100 mined pairs total, 2,100 distinct anchors (1 negative each — consistent with the strict-argmin code), 1,285 of those anchors are contact patches, with per-fragment contact-anchor overlap in the 29–36% range (e.g. F1: 200/686 = 29.2%, F6: 285/780 = 36.5%, matching the earlier-session measurement).

**Restated mechanism (honest naming):**
The FPFH-mined set is **not** a "hard negative bank with padding fallback" — it
supplies at most 1 hard negative for ~30% of contact anchors and 0 for the
rest, so any design that called padding a "fallback" had padding doing ~93% of
the work. That mechanism is retired from training and kept as the frozen
evaluation-only pool (§1).

**Primary training mechanism (revised):** on-the-fly, per-epoch, encoder-mined
hard negatives:
1. Each epoch (or every N steps — see §3 for the batch-size interaction),
   embed a fold-restricted pool of candidate non-adjacent patches under the
   **current** encoder state.
2. For each training anchor, mine its nearest non-adjacent, non-fold-held-out
   patch under the current embedding (same `argmin` logic as
   `mine_hard_negatives`, reapplied every epoch instead of once with FPFH).
3. This directly targets the measured pathology (`easy_minus_hard_auc_gap =
   0.6141`, `PHASE5_RESULTS_LOG.md` per the earlier-session verification) by
   training against the representation's *current* confusions rather than a
   frozen, FPFH-defined notion of "hard."

**Ratio:** the previous design's "1:4:32" ratio is dropped — it was never
achievable from the mined set (§2 evidence) and there is no replacement fixed
ratio to assert without a measurement. The actual per-step hard-negative count
is whatever the batch construction in §3 settles on; that number will be
reported from the implemented sampler, not asserted here.

### 2b. Mining cost is a design decision, not a deferred measurement

Promoting on-the-fly mining to the primary mechanism (above) means its cost is
now part of the training-loop design, not a side measurement to fill in later
— it competes with the loss step (§3) for the same 5.64 GB, and it scales as
`anchors × candidate-pool size` per mining round, not as a fixed per-step
constant like a forward/backward pass.

**Evidence — the candidate pool is not fragment-symmetric.** Querying
`ctx.neighbors_of` (`context.py:80-88`) for each fragment and summing patch
counts (`ctx.patch_count`, `context.py:74-75`) over the non-adjacent set gives
the real non-adjacent candidate-pool size per fragment (measured directly,
`PYTHONPATH=src python3 -c "..."` using `load_context`, this session):

| Fragment | Neighbors | Non-adjacent fragments | Non-adjacent patch pool |
|---|---|---|---:|
| F1 | 2,3,4,5 | 6,7 | 1,822 |
| F2 | 1,3,4 | 5,6,7 | 2,822 |
| F3 | 1,2,4 | 5,6,7 | 2,822 |
| F4 | 1,2,3,5 | 6,7 | 1,822 |
| F5 | 1,4,6,7 | 2,3 | 2,000 |
| F6 | 5,7 | 1,2,3,4 | 4,000 |
| F7 | 5,6 | 1,2,3,4 | 4,000 |

F6/F7 anchors mine against a pool ~2.2× larger than F1/F4 anchors (4,000 vs
1,822). This is not just a cost asymmetry — it changes what "hard" means per
fragment: a larger candidate pool makes it statistically more likely to find
a closer look-alike, so F6/F7's mined negatives are systematically harder (in
raw distance terms) than F1/F4's, independent of any real difference in
geometric ambiguity. F6/F7 are also exactly the low-confidence fragments
(`PHASE5_PREREGISTRATION.md` §4b), so this asymmetry could produce an
F6/F7-driven training artifact that looks like a modeling result but is
actually a candidate-pool-size effect. Under LOFO the pool shifts again each
fold, since the held-out fragment's patches are removed from every other
fragment's candidate pool for that fold.

**Design questions — resolved:**

1. **Mining frequency: per epoch, not per-N-step.** A LOFO training fold has
   ~5,800 patches (`PHASE5_RESULTS_LOG.md:160`, "a LOFO fold (~5,800 patches)
   is ~6-11 minibatches/epoch" at the batch sizes considered there) and 6
   training fragments per fold. At `B_a=512` (§3, verified), a fold is
   ~11-12 minibatches/epoch. Re-mining every step would mean re-encoding up
   to a 4,000-patch candidate pool (§2b table, F6/F7's non-adjacent pool
   size) roughly once every ~96 ms (§3's `B_a=512` step time) — i.e. paying
   full-pool mining cost 11-12× more often than epoch-level mining, for a
   representation that only moves incrementally within an epoch. Per-epoch
   mining amortizes this: one mining pass costs one extra forward over the
   candidate pool (§3's dedup/no_grad path, cheap relative to a training step
   — see item 3 below) and stays valid for the ~11-12 steps until the next
   epoch. **N = 1 epoch.** If the fold-1 timing gate (item 5 below) shows the
   representation moving fast enough early in training that epoch-stale
   negatives are measurably too easy, revisit — but that is an empirical
   escalation trigger, not the default.
2. **Candidate-pool definition: full non-adjacent pool per fragment, not a
   fixed-count subsample.** Given `B_a=512` fits with 3.9× memory headroom
   (§3), and the largest per-fragment non-adjacent pool is 4,000 patches
   (F6/F7, §2b table) — well within a single encode-under-no_grad pass at the
   `B=1024 → 1444 MB` single-patch cost already measured in
   `PHASE5_RESULTS_LOG.md:155` — there is no memory pressure forcing a
   subsample. Using the **full** pool preserves the real per-fragment
   asymmetry (1,822 for F1/F4 vs 4,000 for F6/F7) rather than erasing it with
   a fixed-count or fixed-fraction subsample, which matters because §2b's
   per-fragment difficulty logging (item 4 below) is specifically designed to
   surface that asymmetry — subsampling to a common size would hide the
   effect the logging exists to catch.
3. **`no_grad` + cached embedding table: mandatory. Measured, not assumed —
   correcting an earlier speculation in this document.** Mining encodes each
   fold's training-visible candidate pool (≤4,000 patches, item 2 above) once
   per epoch under `torch.no_grad()` in `model.eval()` mode, caches the
   resulting embedding matrix, and reuses it for every anchor's argmin lookup
   that epoch — never re-encoded per anchor, never carrying autograd.
   Measured directly (single forward pass, `no_grad`, `model.eval()`, same
   RTX 4050, `(N=64,k=16,4)` features, real pool sizes from the §2b table):

   | Pool size (fragment) | peak MB | time ms |
   |---:|---:|---:|
   | 1,822 (F1/F4) | 1474.2 | ~109 |
   | 2,000 (F5) | 1616.8 | ~55 |
   | 2,822 (F2/F3) | 2277.5 | ~75 |
   | 4,000 (F6/F7) | **3221.9** | ~125 |

   **Correction:** an earlier draft of this document speculated the `no_grad`
   mining pass would cost *less* peak memory than the §3 training-step
   figure "since there's no backward pass." That is wrong for peak memory
   (though right for *not scaling with a second backward*) — encoding 4,000
   patches in one forward call peaks at 3221.9 MB, **more** than a full
   `B_a=512` training step (1436.4 MB), because the pool sizes here (up to
   4,000) are much larger than the training batch (512). This is still
   safe: the mining pass and a training step never run concurrently within
   the same forward/backward graph (mining happens once at epoch start,
   `torch.cuda.empty_cache()` is called before resuming the training loop),
   so the two costs do not add — but a training loop implementation must
   ensure they are not accidentally interleaved (e.g. mining triggered
   mid-batch) without an intervening cache clear, or the two peaks *would*
   stack and risk the OOM boundary found in §3 (which starts near 5,690 MB).
4. **Per-fragment difficulty logging: required every mining round.** Each
   epoch's mining pass logs, per training fragment, the mean and median mined
   hard-negative distance. Given the pool-size asymmetry in the §2b table, a
   systematic drop in F6/F7's (or whichever fragments are in the non-held-out
   six for a given fold) mined distances relative to F1-F4's is expected from
   pool size alone and must be visible in this log, so it is not mistaken for
   a geometric finding when it shows up in downstream LOFO results.

---

## 3. GPU memory — re-measure at the real per-step patch count

**Evidence:**
- `PHASE5_RESULTS_LOG.md:155` — "B=256 -> 374 MB (195 ms); B=1024 -> 1444 MB (57 ms); B=4096 -> OOM (only ~1 GB free of 5.64 GB)." This is memory for **B independently encoded patches**, not a training step.
- `PHASE5_RESULTS_LOG.md:160` — "will use batch 512 for headroom" — this conclusion does not state what quantity "batch 512" refers to (anchors? total forwards?).

**What a real training step forwards, given §1/§2:**
anchors + positives + on-the-fly hard negatives + any in-batch random
negatives. If the anchor batch is `B_a`, and each anchor carries 1 positive
(§5 pending) and up to `h` hard negatives, the per-step encoded-patch count is
approximately `B_a × (2 + h)`, before in-batch randoms. At `B_a = 512, h = 4`
(the rejected design's numbers) that is `512 × 6 = 3072` — inside the regime
`PHASE5_RESULTS_LOG.md:155` already reports failing (`B=4096 → OOM`) and well
above the largest confirmed-safe point (`B=1024 → 1444 MB`).

**Required before training code is written — now DONE, measured directly:**

`scripts/measure_step_memory.py` builds the real per-step composition
(`B_a` anchors + `B_a` positives + `n_hard_distinct` deduped hard negatives,
each `(N=64, k=16, 4)` PPF features through the real `PointNetEncoder`,
forward + InfoNCE-shaped loss + backward, peak GPU memory via
`torch.cuda.max_memory_allocated`) and was run on the same RTX 4050
(`total_mem=6051 MB` per `torch.cuda.get_device_properties`, consistent with
the "5.64 GB total" figure in `PHASE5_RESULTS_LOG.md:155`). Command:
`PYTHONPATH=src python3 scripts/measure_step_memory.py`. Measured output
(`B_a` = anchor batch, `H_distinct` = number of *distinct* hard-negative
patches encoded once and shared across the batch, `total_fwd` = total encoder
forward calls in the step = `2*B_a + H_distinct`):

| B_a | H_distinct | total_fwd | peak MB | step ms |
|---:|---:|---:|---:|---:|
| 128 | 0 | 256 | 293.1 | ~23 |
| 256 | 0 | 512 | 551.7 | ~33 |
| 512 | 0 | 1024 | 1085.7 | ~64 |
| 512 | 512 (deduped) | 1536 | 1436.4 | ~96 |
| 1024 | 0 | 2048 | 2153.7 | ~129 |
| 1024 | 256 (deduped) | 2304 | 2158.0 | ~147 |
| 512 | 2048 (non-deduped, h=4/anchor) | 3072 | 3588.9 | ~194 |
| 1024 | 1024 (non-deduped, h=1/anchor) | 3072 | 2855.2 | ~191 |
| 2048 | 0 | 4096 | 4289.6 | ~261 |
| 2048 | 2048 (deduped) | 6144 | **5692.6** | ~394 |
| 3072 | 0 | 6144 | **OOM** (tried to allocate 1.50 GiB, only 1.31 GiB free) | — |

**Correction to this document's own earlier assumption:** the draft above
speculated that `B_a=512` with a non-deduped 1:4 ratio (3072 total forwards)
would be "inside the regime that already OOM'd" — that speculation was
**wrong**. Measured directly: `B_a=512, H=2048` (non-deduped 1:4, 3072
forwards) peaks at 3588.9 MB and runs fine (~194 ms/step). The actual OOM
boundary is `total_fwd ≈ 6144` **only when `B_a` itself is large** (3072
anchors, 0 hard negatives, already OOMs) — `total_fwd` alone is not the right
proxy; `B_a` and `H_distinct` have different memory costs per unit (anchors
and positives both need gradients retained for the backward pass through two
full forward paths per anchor; deduped hard negatives are one extra forward
per distinct patch, shared). `B_a=2048, H=2048` deduped (also 6144 forwards)
peaks at 5692.6 MB — it fits, but at 94% of the 6051 MB device total, too
close to the ceiling to be a safe operating point given the log's own
observation that background processes already reduce free memory (measured
5.64 GB free of a 6.05 GB device in the earlier single-patch probe).

**Decision: `B_a = 512`, with hard negatives deduped within the batch
(mandatory, not optional — see below).** Verified-safe headroom: `B_a=512`
with up to 512 deduped distinct hard negatives peaks at 1436.4 MB — **3.9× headroom** under the ~5.64 GB usable budget, and comfortably below the
1444 MB the original log measured for a bare `B=1024` single-patch encode
with no gradient/loss structure at all. This resolves the batch-size
`UNVERIFIED` flag from the first draft: **`B_a = 512` is now VERIFIED at
1436.4 MB peak, ~96 ms/step, per the table above (command and full output
reproducible via `scripts/measure_step_memory.py`).**

**Dedup is mandatory, not a fallback.** The non-deduped comparison rows above
(`B_a=512,H=2048` and `B_a=1024,H=1024`, both representing "4 hard negatives
per anchor, undeduped") show 3588.9 MB and 2855.2 MB respectively — both fit
today, but neither has meaningful headroom left for other unavoidable
overhead (the on-the-fly mining pass in §2b runs *interleaved* with training
and needs its own memory), and neither is necessary: because the full mined
evaluation pool is only 2,100 vectors and the on-the-fly training pool is
bounded per fold (§2b table, ≤4,000 patches), the realistic number of
*distinct* hard negatives touched by a single 512-anchor batch is far smaller
than 512×4=2048 — dedup is both cheaper and the honest reflection of how much
information a batch of 512 anchors can actually draw from a pool that size.

---

## 4. Sampling — fragment→interface→partner, not fragment-uniform

**Objective.** Fix the real imbalance, which is per-interface, not per-fragment.

**Evidence — positive-pair counts per interface** (`phase3_final_review/pairs.npz`,
`labels == 1`, grouped by `fragment_A_idx`/`fragment_B_idx`, values as measured
and independently corroborated this session against the same field):

| Interface | Positive pairs | Share of 719,279 |
|---|---:|---:|
| F6-F7 | 214,446 | 29.8% |
| F1-F4 | 109,060 | 15.2% |
| F2-F3 | 104,067 | 14.5% |
| F4-F5 | 96,390 | 13.4% |
| F1-F2 | 73,264 | 10.2% |
| F5-F6 | 40,940 | 5.7% |
| F1-F5 | 38,896 | 5.4% |
| F3-F4 | 38,608 | 5.4% |
| F1-F3 | 2,376 | 0.33% |
| F2-F4 | 1,232 | 0.17% |
| F5-F7 | 0 | 0% |

174× spread between F6-F7 and F2-F4. F1-F3 and F2-F4 — both inside the
{F1,F2,F3,F4} fold set the pass bar (`PHASE5_PREREGISTRATION.md` §4b, "Refined
well-supported set for the pass decision: F1, F2, F3, F4") depends on — are
each under 0.4% of positives. Uniform fragment-level anchor draws still route
most gradient signal through F6-F7/F1-F4/F2-F3/F4-F5, which does not fix the
folds that matter for pass/fail.

**F5-F7 is not an 11th interface.** `ctx.interface_patch_ids` (`context.py:68`,
keyed `(fid_a, fid_b) -> {fid -> set[patch_id]}`) gives F5-F7 as 0 contact
patches on both sides — consistent with `F5-F7: 0` positive pairs above.
`n_well_supported_interfaces = 10`, not 11.

**Multi-interface patches need an explicit rule.** A contact patch can belong
to more than one interface (e.g. F1: 164/686 = 23.9% of F1's contact patches
sit on more than one interface boundary; F7: 0/347 — none of F7's contact
patches are multi-interface). `ctx.interface_patch_ids` makes this directly
queryable per patch (a patch id can appear in more than one interface's
patch-id sets for the same fragment).

**Revised sampler:**
1. Sample a fragment (fold-restricted to training fragments).
2. Sample one of that fragment's **interfaces** (not partner fragment
   directly), weighted to counteract the table above — e.g. inverse-frequency
   or capped-uniform over interfaces rather than over raw positive-pair
   counts, so F1-F3/F2-F4 are not starved relative to F6-F7/F1-F4.
   **Caution on the weighting choice itself:** pure inverse-frequency over the
   table above would give F2-F4 (1,232 pairs, backed by only a small number of
   contact patches) roughly 174× the per-pair sampling weight of F6-F7
   (214,446 pairs) — enough to oversample F2-F4's small contact-patch set hard
   enough to overfit it rather than merely correct the imbalance. **Default:
   capped-uniform, or inverse-frequency with a floor** (a minimum-weight clamp
   so no interface's weight multiplier exceeds a bounded ratio, e.g. capped at
   the ratio needed to equalize the two largest usable interfaces rather than
   the two most extreme ones). Whichever is chosen must be recorded in the
   results log with the reason, not left as the "e.g." placeholder above.
3. Sample an anchor patch belonging to that interface, then its partner.
4. **Multi-interface rule:** when an anchor patch belongs to more than one
   interface (queryable via `interface_patch_ids`), the interface chosen in
   step 2 determines *which partner set* is drawn from for that draw — i.e.
   the same patch can be legitimately drawn as an anchor for either of its
   interfaces on different sampling steps, but a single draw's positive
   partner must come from the interface selected in step 2, not the union of
   all interfaces the patch touches. This keeps each draw's positive
   semantically tied to one interface, and avoids implicitly upweighting
   multi-interface patches by letting one draw satisfy two interfaces'
   sampling quotas at once.

---

## 5. Positive-distance policy — state it, don't hide the P@1 tension

**Evidence — positive-pair centre distances** (`center_dist_mm` field,
`phase3_final_review/pairs.npz`, over positives):

| Statistic | Value |
|---|---:|
| min | 0.10 mm |
| p10 | 7.60 mm |
| median | 18.64 mm |
| p90 | 33.34 mm |
| max | 69.66 mm |
| mean | 19.76 mm |
| fraction ≤ 8 mm (one patch radius) | 11.2% |
| fraction ≤ 16 mm | 40.0% |

Patch radius is 8 mm (Phase 2, `README.md` "8 mm radius neighbourhood").
Only 11.2% of positives have centres within one radius of each other; the
median positive pair is 18.64 mm apart, and the loss will be pulling together
patches up to 69.66 mm apart that are both merely on "the same interface"
(Level-3 co-membership) rather than mutually nearest.

**Policy stated explicitly:** this design uses **all positives, uncapped**,
for training — faithful to the Level-3 co-membership label as declared in
`PHASE5_PREREGISTRATION.md` ("Level 3 — interface association / neighbour
retrieval"). This is a deliberate choice, not an oversight, and it has a
named cost: **the loss optimizes a region-scale objective (same interface)
while the pass bar is judged partly by a top-1 metric (P@1, per
`PHASE5_PREREGISTRATION.md` §4b: "beat FPFH with non-overlapping 95% CIs ...
on both P@1 and mAP").** A model that correctly pulls an anchor toward its
*entire* interface's positive set, rather than its single geometrically
nearest partner, can legitimately underperform on P@1 relative to what a
distance-capped variant might achieve, even while doing exactly what it was
trained to do.

**Because `center_dist_mm` already exists in the archive, a distance-capped
positive variant costs nothing to try as a follow-up ablation** (e.g. cap
positives at p50 or p90 distance) — but is out of scope for Phase 5A's first
run. Phase 5A reports both P@1 and mAP as pre-registered and states this
tension plainly rather than only reporting whichever metric looks better.

---

## 6. Norm layer — BatchNorm is a LOFO confound; switch to GroupNorm/LayerNorm

**Evidence:**
- `src/phase5_encoder/model.py:55` and `:59` — `PointNetEncoder.pair_mlp` and
  `.point_mlp` both use `nn.BatchNorm1d`.
- `src/phase5_encoder/model.py:114` (`build_encode_patch`) — calls
  `model.eval()` before the forward pass *(corrected from an earlier draft's
  `:107-109`, which is the feature-extraction branch — `knn_ppf_features`/
  `ppf_features` — not the `eval()` call)*. In eval mode, `BatchNorm1d` uses
  its running statistics, which under LOFO are fit **only on the six training
  fragments** (the held-out fragment contributes zero batches during
  training).

**Why this is a confound, not just a stylistic nit:** if the held-out
fragment's feature distribution differs from the training six (plausible —
fragments differ in size, curvature, and local density even after
normalization), LOFO performance for that fold degrades for a reason **unrelated to interface association** — purely a train/eval statistics mismatch. Condition 2 of the pre-registration ("wins on the majority of well-supported LOFO folds") would then fail for a reason the design cannot distinguish from a genuine negative result.

This is the same shape of argument already used and accepted in Deviation 2
(`PHASE5_RESULTS_LOG.md:128-140`): PCA-frame canonicalization was rejected
specifically because "a Phase-5A failure could not be distinguished between
'PointNet cannot learn interface association' ... and 'the input frame was
unstable'" — an uninteresting confound. BatchNorm-under-LOFO is the same
failure shape applied to normalization statistics instead of input framing.

**Decision:** replace `nn.BatchNorm1d` with `nn.GroupNorm` (or `nn.LayerNorm`,
equivalent here since these are per-point/per-pair feature vectors, not
spatial feature maps) in both `pair_mlp` and `point_mlp`. This has a second,
independent benefit: it removes the batch-size dependence entirely, which
interacts favorably with §3 — the memory/batch-size decision no longer needs
to also worry about BatchNorm's own preference for larger batches to get
stable running statistics.

---

## 7. Early stopping — pair-level split held out from every interface

**Decision:** validation uses a **pair-level split held out from every
interface** (a slice of positive pairs, drawn proportionally across all 10
well-supported interfaces, excluded from the training loss) — not a held-out
interface, not a held-out fragment.

**Reasoning (asymmetry argument):** early stopping is not itself part of the
pre-registered pass criteria (`PHASE5_PREREGISTRATION.md` §1–§6 define pass
conditions on LOFO/held-out-fragment evaluation, not on a validation curve).
A bad stop rule can therefore only make the reported encoder **worse than it
could be** (stopping too early or too late relative to the true optimum) — it
cannot manufacture a false pass, because the pass/fail decision is made on
LOFO folds computed independently of whatever the stop rule picked. This
asymmetry means the stop rule should be chosen to protect the **integrity** of
the metric (don't contaminate the pre-registered pass bar), not to maximize
signal quality.

**Why not a held-out interface:** costs 1 of only 10 well-supported interfaces
(§4 evidence: `n_well_supported_interfaces = 10`) — roughly 10% of the honest
headline macro mAP (`PHASE5_PREREGISTRATION.md`: "headline honest metric ...
macro-averaged mAP over well-supported interfaces"). Holding a *different*
interface out per LOFO fold makes folds incomparable to each other; holding
the *same* one out every fold systematically depresses that interface in
every reported number, which is worse.

**Why not a held-out fragment:** burns a second fragment per LOFO fold on top
of the one already held out for the fold itself, and with F6/F7 already
low-confidence (`PHASE5_PREREGISTRATION.md` §4b), the validation-fragment
choice would be drawn from only four usable fragments — a much smaller and
more biased pool than the pair-level alternative.

**Stated weakness (must be reported, not hidden):** a held-out pair's two
endpoint patches usually still appear in *other* training pairs — measured
directly this session, not inferred from the mean ratio: of the 6,822 total
patches, only **4,107 carry at least one positive** (the rest are non-contact
and carry zero, confirming positives are concentrated entirely on contact
patches). Among those 4,107, the positive-count-per-patch **distribution**
(not just the mean) is: min 115, p10 187, median 347, p90 618, p99 815, max
815, mean 350.3. Every contact patch that appears in any held-out validation
pair participates in a median of 347 *other* training positives — the
held-out signal is therefore close to the training signal almost everywhere
in the distribution, not just on average, which makes the caveat below load-bearing rather than a minor footnote: **patience may never fire** — the
epoch cap (50, retained from the accepted part of the original design) may be
doing the actual stopping in practice.

**Reporting requirement:** the training log must record the full validation
curve and explicitly state whether **patience fired or the cap did** for each
fold — never describe a run as "early stopped" without specifying which. If
interface-generalization evidence is wanted, it is measured as a **secondary,
post-hoc diagnostic** (e.g. re-running LOFO-style holdout per interface after
the fact), not used as the mechanism that decides when training stops.

---

## Summary of what is now fixed vs. still open

**Fixed by this revision:**
- Condition 3 will be fold-restricted; FPFH-mined 2,100 becomes eval-only.
  **Implemented:** `easy_vs_hard_separability(..., held_out=...)` added to
  `src/phase5_diagnostics/ranking.py`, wired into `lofo_per_fold(...,
  hard_negatives=...)`, wired into `runner.py`'s LOFO call. 4 new regression
  tests added to `tests/phase5_diagnostics/test_diagnostics.py`
  (`test_easy_vs_hard_separability_excludes_held_out_fragment`,
  `test_lofo_per_fold_wires_held_out_hard_negative_strata`,
  `test_runner_lofo_entry_differs_from_global_hard_negative_strata`, and the
  existing suite re-run). Full suite: **34 passed**
  (`tests/phase5_diagnostics/ tests/phase5_encoder/ tests/baseline_geometry/`).
- On-the-fly encoder-mined hard negatives become the primary training
  mechanism; no fixed ratio is asserted.
- Mining cost is treated as a design decision (§2b: frequency, candidate-pool
  definition, no_grad/cached-table requirement, per-fragment difficulty
  logging), not deferred to a later measurement — the F1/F4-vs-F6/F7
  candidate-pool asymmetry (1,822 vs 4,000 patches) is measured and named as
  a possible confound with LOFO's low-confidence fragments. **§2b's three
  open items are now resolved:** mining frequency N=1 epoch; candidate pool
  is the full non-adjacent pool (no subsampling, to preserve the asymmetry
  the logging is meant to surface); `no_grad`/cached-table mining pass
  measured directly (1474-3222 MB depending on pool size, `scripts/
  measure_step_memory.py`-adjacent measurement, does not stack with a
  training step's memory since they never run concurrently).
- Batch size measured directly, not asserted: **`B_a = 512` verified at
  1436.4 MB peak / ~96 ms per step** with deduped hard negatives
  (`scripts/measure_step_memory.py`), 3.9× headroom under budget.
- Sampler is fragment→interface→partner with an explicit multi-interface
  rule, and the naive inverse-frequency weighting is flagged as its own
  overfitting risk (174× on F2-F4) with capped-uniform as the stated default.
- Positive-distance policy is uncapped/all-positives, with the P@1 tension
  named explicitly rather than left implicit.
- Norm layer switched from BatchNorm to LayerNorm. **Implemented:**
  `src/phase5_encoder/model.py`'s `pair_mlp`/`point_mlp` now use
  `nn.LayerNorm`. Pose gate re-measured (not assumed): `max_deviation = 0.0`
  for both `pair_input=True` and `pair_input=False`, matching the
  pre-change baseline exactly, as expected since invariance is inherited
  from the PPF input, not the norm layer.
- Validation split is pair-level, held out from every interface, with the
  patience-vs-cap ambiguity to be reported honestly per fold — backed by a
  measured (not inferred) positive-pairs-per-patch distribution showing the
  held-out signal stays close to the training signal across the whole
  distribution, not just on average.
- Four citations corrected against a second line-by-line check
  (`ranking.py` argmin at 381 not 373; `runner.py` mine call at 91 not 100;
  `ranking.py` `pos_idx` at 425 not ~424; `model.py` `eval()` at 114 not
  107-109).
- One additional self-correction caught during this implementation pass:
  an earlier draft speculated the `no_grad` mining pass would cost less
  memory than a training step "since there's no backward." Measured
  directly and found wrong for peak memory (mining pools are larger than
  the training batch) — corrected in §2b item 3 above.

**Explicitly still open / requires the fold-1 timing gate before the full
7-fold retrain commitment (per the existing project pattern in
`PHASE5_RESULTS_LOG.md:160`, "Real epoch wall-clock measured in the fold-1
timing gate before committing to 7 retrains"):**
- Wall-clock for one full fold's epoch (training steps + one mining pass),
  end to end, on real (not synthetic) patch data.
- Confirmation that mining-pass memory and training-step memory, run back to
  back with a cache clear between them, do not leave any residual allocation
  that erodes the headroom measured in isolation above.

No training code is written in this document. Implementation of items 1
(condition-3 fix), 2 (BatchNorm->LayerNorm), 3/4 (mining design + memory
measurement) is complete as of this revision; the training loop itself
(item 5, the loop + entry point) is scoped next, to be handed off with an
exact command, expected fold-1 wall-clock, output locations, and the
patience-vs-cap reporting requirement -- per the human's instruction, the
human runs it, not this session.

---

## 8. Implementation status (post-approval, all 5 items)

Items 1-4 are implemented and verified (not just designed):

1. **Condition 3 fix — implemented.** `easy_vs_hard_separability(...,
   held_out=...)` added to `src/phase5_diagnostics/ranking.py`; wired into
   `lofo_per_fold(..., hard_negatives=...)`; wired into `runner.py`'s LOFO
   call site. 4 new tests added to
   `tests/phase5_diagnostics/test_diagnostics.py`. Full targeted suite:
   **34 passed** (`tests/phase5_diagnostics/ tests/phase5_encoder/
   tests/baseline_geometry/`). Full repo suite (no `ulimit`, since the cap
   is tuned for Phase 1-4 dense-cloud steps and starves the phase5+model
   tests): **177 passed** (`tests/`).
2. **BatchNorm -> LayerNorm — implemented.** `src/phase5_encoder/model.py`'s
   `pair_mlp`/`point_mlp`. Pose gate re-measured directly (not assumed):
   `pair_input=True` and `pair_input=False` both give
   `passed=True, max_deviation=0.0`, matching the pre-change baseline.
3. **§2b mining design — resolved.** N=1 epoch; full non-adjacent pool, no
   subsampling; `no_grad`/eval/cached-table mining implemented in
   `src/phase5_encoder/mining.py`; per-fragment mined-distance logging
   implemented (`MinedNegatives.per_fragment_distance`, surfaced per-epoch
   in the training history).
4. **§3 memory re-measurement — done, real output.** `scripts/
   measure_step_memory.py`. `B_a=512` with deduped hard negatives verified
   at 1436.4 MB peak / ~96 ms/step (3.9x headroom under the 5.64 GB usable
   budget). One self-correction made during this pass: an earlier
   speculation that the `no_grad` mining pass would cost less than a
   training step was wrong (measured 3221.9 MB for the largest, 4,000-patch
   pool -- more than a training step, because pool size exceeds batch size)
   -- corrected in §2b item 3 above rather than left standing.

New modules added: `src/phase5_encoder/sampler.py` (fragment->interface->
partner sampler + capped-inverse-frequency weighting + pair-level validation
split, §4/§7), `src/phase5_encoder/mining.py` (on-the-fly per-epoch mining,
§2/§2b), `src/phase5_encoder/train.py` (the training loop itself, §3/§5/§6),
`scripts/train_phase5a.py` (CLI entry point). All covered by
`tests/phase5_encoder/test_sampler_mining.py` (7 tests, including one
end-to-end plumbing smoke test of `train_one_fold` on a toy fixture).

**Item 5 (write the loop, do not run it) is complete as scoped.** The human
runs the actual training; see `scripts/train_phase5a.py`'s docstring for the
exact command, the fold-1 wall-clock estimate (not yet measured on real
data — the script has only been smoke-tested on a toy fixture), output file
locations, and the patience-vs-cap reporting requirement.

---

## 9. Four bugs found and fixed before any run (second review pass)

A second review of the implementation, before any real-data execution, found
four measured bugs — the same pattern as the round-1 citation errors: a
number or assumption stated for one quantity, applied in code to a different
one.

1. **Validation set was uncapped (the OOM blocker).** `val_fraction=0.10`
   was applied directly to raw per-interface pair-row counts. Measured on
   real fold 1: **49,569 validation pairs** (F6-F7 alone: 21,445), because
   `val_fraction` was applied to counts already dominated by the 174x
   interface-size spread (§4), not to a bounded pair/patch budget — the
   docstring's "a few hundred pairs at most" was wrong for the same reason
   the original "batch 512" claim was wrong: asserted for one quantity,
   applied to another. 49,569 pairs = 99,138 sequential per-patch encode
   calls (the un-batched Python-loop path) plus a ~9.8 GB fp32 similarity
   matrix on a 5.64 GB card. **Fixed:** `build_fold_sampler` now takes
   `val_pair_cap` (default 500), downsampling each interface's raw val
   slice proportionally to its own share of the uncapped total (so the
   capped set still reflects roughly the same per-interface mix). Measured
   on real fold 1 after the fix: **499 total validation pairs** (down from
   49,569), proportions preserved (F6-F7: 216, F2-F3: 105, ... F2-F4: 1).
   `_validation_loss` additionally chunks into `val_minibatch_size` (default
   256) minibatches so no single validation step's cost scales with the
   (still several-hundred-pair) total.

2. **Interface weighting inverted (measured: rarest interface got 55.4% of
   all draws).** The original `InterfaceWeighting.weights_for` clamped
   `inv` against `inv.min() * max_ratio` — but `inv.min()` is the inverse
   weight of the LARGEST interface, so "cap of 20" actually meant "allow
   the rarest interface up to 20x the largest interface's own inverse
   weight," which on fold 1's real 174x count spread resolved to **F2-F4
   receiving 55.4% of all draws** (measured directly) — exactly the
   overfitting failure §4 was written to prevent, not a mitigation of it.
   **First fix attempt (clamping the post-normalisation WEIGHT against
   uniform, then renormalising) was ALSO wrong, and this is worth stating
   plainly rather than hiding the false start:** with a 174x true spread,
   pure inverse-frequency already puts ~91.5% of weight on the single
   rarest interface before any cap; clamping that one value and
   renormalising the remainder redistributes mass through the SAME skewed
   proportions among the rest, so the clamped interface still ends up
   dominant after renormalisation (measured: 90.8% for F2-F4, WORSE than
   the bug it was meant to fix). **Correct fix:** clamp EFFECTIVE COUNTS,
   not weights — each interface's raw count is floored at
   `max_count / max_ratio` (never treated as rarer than `1/max_ratio` of
   the largest interface's real count) BEFORE taking inverse-frequency,
   which cannot let one interface dominate because the floor bounds the
   ratio between the two most extreme *counts* directly, not a weight that
   gets renormalised afterward. Default `max_ratio` also lowered from 20.0
   to 5.0 (20.0 still overshot to 3.33x uniform even under the correct
   method). Measured on real fold 1 with the corrected method and
   `max_ratio=5.0`: **F2-F4 at 24.65% (1.48x uniform)** — lifted
   meaningfully above its raw 0.17%-of-pairs share, without dominating.

3. **Hard-negative dedup count was unmeasured.** `mine_epoch_hard_negatives`
   mines one negative per *contact patch*, not per draw; with 512 draws
   funnelled through interface weighting into a small set of repeatedly-
   drawn patches, the deduped hard-negative count per step is typically far
   smaller than 512 — the original design's "every-step signal" framing
   asserted a ratio that was never measured. **Fixed:** `_info_nce_step`
   now returns `n_deduped_hard` alongside the loss;
   `train_one_fold` logs `deduped_hard_negatives_per_step_mean/min/max` per
   epoch, so the real ratio is a reported number, not an assumed one.

4. **In-batch positives were unmasked in the InfoNCE denominator.**
   `cross_entropy(sim, targets)` treated every non-target candidate column
   as a negative — but two draws from the same (or even a different)
   interface can be genuine positive partners of each other (median 347
   positives per contact patch, §7), and at the draw rates interface
   weighting produces this is common, not rare. **Fixed:**
   `_build_positive_partner_lookup` builds a `(fid,pid) -> {(fid,pid)}`
   true-partner lookup once per fold from `ctx.pairs` (a lighter-weight
   analogue of `phase5_diagnostics.ranking.positive_partner_map` that
   doesn't require a pre-built `Gallery`); `_info_nce_step` masks any
   candidate column that is a true positive of the row's anchor (except the
   actual target column) to `-inf` before `cross_entropy`.

**Corrected fold-1 wall-clock estimate**, with real per-component
measurements this time (Python-loop feature extraction included, not
assumed negligible — the earlier 1.3-1.5 s/epoch estimate omitted the
validation term's cost AND undercounted feature-extraction cost for the
training step itself):

| Component | Measured cost | Frequency |
|---|---:|---|
| Training step (512+512, feat+fwd+bwd) | ~1123 ms | 12/epoch → ~13.5 s |
| Mining pass (4,000-patch pool, feat+fwd) | ~3388 ms | 1/epoch → ~3.4 s |
| Validation (499 pairs, capped, 4 minibatches) | ~211 ms/minibatch | 1/epoch → ~0.84 s |
| **Per-epoch total** | | **~17.7 s** |
| **50-epoch cap (fold 1, if patience never fires)** | | **~885 s (~14.8 min)** |

An operational finding, also folded into `scripts/train_phase5a.py`'s
docstring: a fresh process on this machine's RTX 4050 hit spurious CUDA OOM
errors at batch/pool sizes previously measured safe, despite `nvidia-smi`
showing the GPU nearly idle — confirmed to be PyTorch allocator
fragmentation, fixed by `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(re-measuring with this set reproduced the original design-doc numbers
exactly). This is now a required environment variable in the training
command, not optional.

`scripts/train_phase5a.py` gained `--max-epochs` (overrides `--epoch-cap`
for a quick first real-data run, e.g. `--max-epochs 1`, to sanity-check
wall-clock against the table above before committing to the full cap),
`--val-pair-cap`, and `--interface-weight-max-ratio` as CLI-exposed
parameters rather than hardcoded.

All four fixes are covered by new regression tests in
`tests/phase5_encoder/test_sampler_mining.py`
(`test_validation_set_is_capped_not_proportional_to_raw_pair_counts`,
`test_interface_weighting_does_not_let_rarest_interface_dominate`,
`test_interface_weighting_uniform_when_all_counts_equal`,
`test_info_nce_step_masks_same_interface_true_positives`,
`test_deduped_hard_negative_count_is_returned`) — **46 passed**
(`tests/phase5_diagnostics/ tests/phase5_encoder/ tests/baseline_geometry/`).

No training has been run. The human runs it, per standing instruction.

---

## 10. Fold-1 dry run executed; the ~17.7 s estimate was wrong (5th building-block error) and is now measured

The human ran the 1-epoch fold-1 dry run
(`--held-out fragment_caesar_fragment_1 --max-epochs 1`). Results, from
`phase5a_runs/fold_fragment_caesar_fragment_1_history.json`:

| Component | §9 estimate | **Measured (real fold 1)** |
|---|---:|---:|
| Mining pass | ~3.4 s | **18.17 s** |
| Training (12 steps) | ~13.5 s | **20.92 s** |
| Validation | ~0.84 s | **0.85 s** |
| **Per-epoch total** | ~17.7 s | **39.95 s** |
| Peak GPU memory | (1436 step / 3222 mining) | **956.9 MB** |

**Diagnosis of the mining miss (NOT retrofitted — the building block was
wrong, per the standing rule):** §9/§2b measured mining as a *single*
4,000-patch pool encode (~3.4 s). But the code encodes each anchor
fragment's non-adjacent pool *separately*, and those pools overlap heavily
across the 6 anchor fragments. Measured real work for fold 1: **18,887
patch-encodes** (15,466 pool + 3,421 anchor), against only **5,822 distinct
training patches** — a ~3.2× redundancy. The estimate measured one pool;
the code encoded six overlapping ones.

**This is the FIFTH instance of one recurring failure class in this
project**, and it is worth naming explicitly because the citation rule does
not catch it:

| # | Symptom | The unit that was wrong |
|---|---|---|
| 1 | "batch 512" | measured per-*patch* memory, applied per-*step* |
| 2 | `val_fraction=0.10` | applied to *pairs*, reasoned about as *patches* |
| 3 | H=4 hard-neg ratio | asserted per-*anchor*, code mines per-*patch* |
| 4 | mined-distance asymmetry | §2b pool sizes are *full-dataset*, no *fold* sees them |
| 5 | ~17.7 s/epoch | measured *one* pool encode, code does *six* overlapping |

The citation rule ("cite a file:line / JSON field / pasted output") catches
a wrong *location*. It does not catch a wrong *unit* — a number correctly
copied from a real measurement of the wrong quantity. **Added rule: every
measurement must state its UNIT and the SCOPE it was measured over** (per
patch vs per step vs per epoch; per fold vs full dataset; one pool vs all
pools), not just where it came from. All five errors above are unit/scope
mismatches, not sourcing errors.

**Mining optimization (pure refactor, applied):** encode all 5,822 distinct
training-fragment patches once per epoch into an eval-mode cached table
(`_encode_training_table`), then both pool and anchor accesses are index
lookups (an anchor patch's embedding is the same vector whether mined *for*
or *against*). Proven mathematically identical to the per-anchor-pool
version by `test_mining_cache_matches_uncached` (same lookup dict, same
distances, cached vs uncached) and reproducible-within-epoch by
`test_mining_cache_uses_eval_mode_no_jitter`. **Measured on real fold 1:
mining 18.17 s → 6.30 s** (2.9×; slightly under the 3.2× encode-count
ratio due to fixed overhead). Corrected per-epoch ≈ **28 s**; 7 folds ×
50 epochs ≈ **2.7 hours** (upper bound, patience never firing).

**Deduped hard negatives — now a measured fact (Deviation 2 closed):**
268.1 mean distinct hard negatives per step (min 258, max 280) against 512
anchors — ~52% of anchors contribute a distinct negative, the rest collide.
This is **not dilution**: the 268 are *shared denominator candidates* seen
by every anchor, so each anchor is contrasted against all 268 hard + 511
in-batch-positive-as-negative = ~34% of the denominator is hard. Whether
34% is enough to move the easy-vs-hard AUC gap is exactly condition 3's
job to answer on real training — not something to pre-tune.

**Interface draw fractions matched the weights (sampler verified):**
observed F2-F4 = 0.236 vs weight 0.246; F3-F4 = 0.249 vs 0.246; F5-F6 =
0.251 vs 0.246; F6-F7 = 0.053 vs 0.049 — all within sampling noise at
6,144 draws (5·SE ≈ 0.028). The fragment→interface→partner sampler and the
corrected capped-count weighting behave as designed; no sampler bug.

**§2b mined-distance prediction — PREDICTED, NOT OBSERVED on the untrained
baseline; recheck post-training.** §2b hypothesized larger candidate pool →
closer mined negatives. Fold-1 measured the opposite: F5 (0.0458) and F6
(0.0485) mine ~1.8× *farther* than F2/F3/F4/F7 (0.0246–0.0264), and F6 has
the largest fold-1 pool yet mines farthest. On an untrained random-init
encoder this reflects the random embedding's geometry, not interface
signal, so it neither confirms nor refutes the hypothesis — but the §2b
mechanism is **unconfirmed** and must be re-checked after real training
rather than treated as established.

**Correction to the §2b pool-size table:** its figures (F1/F4=1,822 …
F6/F7=4,000, a 2.2× asymmetry) are computed over **all 7 fragments**. Every
LOFO fold removes the held-out fragment from every candidate pool, so no
fold ever sees those numbers. For fold 1 (F1 held out), F6's pool is
**3,000**, not 4,000. The full-dataset asymmetry figure is an upper bound
that no training fold actually experiences — this is failure-class #4 in
the table above.

**Loss moved within the epoch (plumbing OK, not a learning claim):** train
loss 6.524 → 6.455 over 12 steps (not flat, not NaN); val loss 5.518. One
epoch on an untrained encoder says nothing about interface association —
this is only a plumbing signal that forward/backward/optimizer are wired.

Nothing in §10 touches the six pre-registered conditions. It is a timing +
plumbing gate outcome plus the mining optimization it motivated. Still no
multi-epoch or multi-fold training run.
