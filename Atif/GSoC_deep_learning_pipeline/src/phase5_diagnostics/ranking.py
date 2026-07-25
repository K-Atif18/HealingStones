"""Retrieval diagnostics: the honest-metric variants of Phase 4 ranking.

All functions consume ``dict[str, PatchDescriptors]`` so they run identically on
the null floor, weak baseline, FPFH, and any future learned checkpoint.

Provided variants
-----------------
* ``build_gallery``          : concatenate embeddings (optionally filtered to a
                               patch subset, e.g. contact-only).
* ``positive_partner_map``   : the Level-3 ground truth as the harness sees it
                               ((fid,pid) -> set of partner gallery rows).
* ``evaluate_ranking``       : mAP / P@K / R@K over queries, with configurable
                               gallery filter and query restriction. This is the
                               generalisation of the Phase 4 harness.
* ``calibrate_null``         : run ranking on random + centroid sources to
                               establish the harness generosity floor.
* ``retrieval_by_interface`` : per-interface mAP (stratified) + macro-average
                               across interfaces (kills interface-size bias).
* ``mine_hard_negatives``    : FPFH-nearest cross-fragment, *non-adjacent*
                               patch pairs -- the evaluation-only hard-negative
                               set Phase 3 never produced.
* ``easy_vs_hard_separability``: pair-level AUC of positives vs easy (random)
                               negatives compared with positives vs hard
                               (mined) negatives. The gap is the headline
                               "does the embedding resist look-alikes" number.
* ``lofo_per_fold``          : leave-one-fragment-out retrieval, reported per
                               fold (never averaged given n=7).

The metric primitives (``roc_auc``, ``average_precision``) are imported from the
Phase 4 module so the numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from baseline_geometry.descriptors import PatchDescriptors
from baseline_geometry.retrieval import roc_auc, average_precision

from phase5_diagnostics.context import DiagnosticContext


__all__ = [
    "Gallery",
    "build_gallery",
    "positive_partner_map",
    "evaluate_ranking",
    "calibrate_null",
    "retrieval_by_interface",
    "mine_hard_negatives",
    "easy_vs_hard_separability",
    "lofo_per_fold",
    "_bootstrap_ci",
]


# ----------------------------------------------------------------------
# Gallery
# ----------------------------------------------------------------------
@dataclass
class Gallery:
    """A concatenated embedding gallery with row bookkeeping."""

    desc: np.ndarray            # (G, D)
    frag: np.ndarray            # (G,) object -> fragment_id
    patch: np.ndarray           # (G,) int64 -> patch id
    row_of: dict                # (fid, pid) -> row index

    @property
    def size(self) -> int:
        return int(self.desc.shape[0])


def build_gallery(
    embeddings: dict[str, PatchDescriptors],
    *,
    keep: Optional[Callable[[str, int], bool]] = None,
) -> Gallery:
    """Concatenate all embeddings into one gallery.

    ``keep(fid, pid)`` optionally filters which patches enter the gallery
    (e.g. contact-only). ``None`` keeps everything.
    """
    desc_chunks, frag_list, patch_list = [], [], []
    for fid, pd in embeddings.items():
        for row, pid in enumerate(pd.patch_ids):
            pid = int(pid)
            if keep is not None and not keep(fid, pid):
                continue
            desc_chunks.append(pd.descriptors[row])
            frag_list.append(fid)
            patch_list.append(pid)
    if not desc_chunks:
        return Gallery(np.empty((0, 0)), np.empty((0,), dtype=object),
                       np.empty((0,), dtype=np.int64), {})
    desc = np.vstack(desc_chunks)
    frag = np.asarray(frag_list, dtype=object)
    patch = np.asarray(patch_list, dtype=np.int64)
    row_of = {(frag[i], int(patch[i])): i for i in range(len(patch))}
    return Gallery(desc, frag, patch, row_of)


# ----------------------------------------------------------------------
# Level-3 ground truth (positive partners), from the Phase 3 pair table
# ----------------------------------------------------------------------
def positive_partner_map(ctx: DiagnosticContext, gallery: Gallery) -> dict[tuple, set]:
    """(fid, pid) -> set of gallery rows that are its positive partners.

    Symmetric: both directions of every positive pair are recorded. Rows absent
    from ``gallery`` (e.g. filtered out) are silently skipped, so this composes
    with a filtered gallery.
    """
    pairs = ctx.pairs
    vocab = pairs.fragment_vocab
    pos_idx = np.where(pairs.labels == 1)[0]
    out: dict[tuple, set] = {}

    def _add(fa, pa, fb, pb):
        key = (fa, int(pa))
        gb = gallery.row_of.get((fb, int(pb)))
        if gb is None:
            return  # candidate partner must be present in the (filtered) gallery
        # NOTE: we deliberately do NOT require the query key itself to be in the
        # gallery. Under LOFO the query patch lives on the held-out fragment,
        # which is excluded from the gallery by construction; its descriptor is
        # fetched from the full embeddings dict in evaluate_ranking. Requiring
        # the query in the gallery here made every LOFO fold empty.
        out.setdefault(key, set()).add(gb)

    for p in pos_idx:
        fa = str(vocab[pairs.fragment_A_idx[p]])
        fb = str(vocab[pairs.fragment_B_idx[p]])
        _add(fa, pairs.patch_A_ids[p], fb, pairs.patch_B_ids[p])
        _add(fb, pairs.patch_B_ids[p], fa, pairs.patch_A_ids[p])
    return out


# ----------------------------------------------------------------------
# General ranking evaluation
# ----------------------------------------------------------------------
def evaluate_ranking(
    ctx: DiagnosticContext,
    embeddings: dict[str, PatchDescriptors],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    query_sample: int = 800,
    seed: int = 0,
    gallery_keep: Optional[Callable[[str, int], bool]] = None,
    query_keep: Optional[Callable[[str, int], bool]] = None,
    query_keys: Optional[list[tuple]] = None,
    return_per_query: bool = False,
) -> dict:
    """Cross-fragment ranking retrieval (generalised Phase 4 harness).

    * ``gallery_keep`` filters gallery membership (e.g. contact-only gallery).
    * ``query_keep`` filters which queries are evaluated.
    * ``query_keys`` optionally fixes the exact query set (overrides sampling);
      used by per-interface and LOFO variants for determinism.

    A query never retrieves patches from its own fragment. Ground-truth partners
    are the Phase-3 positives restricted to the (possibly filtered) gallery.
    """
    gallery = build_gallery(embeddings, keep=gallery_keep)
    if gallery.size == 0:
        return {"n_queries": 0, "gallery_size": 0, "mAP": float("nan"),
                "precision_at_k": {}, "recall_at_k": {}}

    positives = positive_partner_map(ctx, gallery)

    # Row lookup for query descriptors from the FULL embedding universe, so a
    # query on a fragment excluded from the (filtered) gallery -- e.g. the
    # held-out fragment under LOFO -- can still be scored.
    full_row_of: dict[tuple, int] = {}
    full_desc_by_frag: dict[str, np.ndarray] = {}
    for fid, pd in embeddings.items():
        full_desc_by_frag[fid] = pd.descriptors
        for row, pid in enumerate(pd.patch_ids):
            full_row_of[(fid, int(pid))] = row

    keys = query_keys if query_keys is not None else list(positives.keys())
    if query_keep is not None:
        keys = [k for k in keys if query_keep(k[0], k[1])]
    # Keep only queries that still have >=1 partner in this gallery.
    keys = [k for k in keys if positives.get(k)]

    if query_keys is None and query_sample and len(keys) > query_sample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(keys), size=query_sample, replace=False)
        keys = [keys[i] for i in sel]

    p_at_k = {k: [] for k in k_values}
    r_at_k = {k: [] for k in k_values}
    ap_list = []

    for key in keys:
        fid_q, pid_q = key
        # Resolve the query descriptor: prefer the gallery row (same vector),
        # else fall back to the full embedding universe (LOFO held-out queries).
        qrow = gallery.row_of.get((fid_q, pid_q))
        if qrow is not None:
            qvec = gallery.desc[qrow]
        else:
            frow = full_row_of.get((fid_q, pid_q))
            if frow is None:
                continue
            qvec = full_desc_by_frag[fid_q][frow]
        cross = gallery.frag != fid_q
        if not cross.any():
            continue
        cand_rows = np.where(cross)[0]
        dists = np.linalg.norm(gallery.desc[cand_rows] - qvec, axis=1)
        order = np.argsort(dists, kind="mergesort")
        ranked_rows = cand_rows[order]

        gt = positives[key]
        n_gt = len(gt)
        rel = np.fromiter((1.0 if r in gt else 0.0 for r in ranked_rows),
                          dtype=np.float64, count=len(ranked_rows))
        for k in k_values:
            topk = rel[:k]
            p_at_k[k].append(topk.sum() / k)
            r_at_k[k].append(topk.sum() / n_gt)
        cum = np.cumsum(rel)
        precision = cum / (np.arange(len(rel)) + 1)
        ap_list.append((precision * rel).sum() / n_gt)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    result = {
        "n_queries": len(ap_list),
        "gallery_size": gallery.size,
        "mAP": _mean(ap_list),
        "precision_at_k": {str(k): _mean(p_at_k[k]) for k in k_values},
        "recall_at_k": {str(k): _mean(r_at_k[k]) for k in k_values},
    }
    if return_per_query:
        # Raw per-query arrays for bootstrap confidence intervals. Order is the
        # order queries were scored (parallel across all these arrays).
        result["per_query"] = {
            "ap": np.asarray(ap_list, dtype=np.float64),
            "precision_at_k": {str(k): np.asarray(p_at_k[k], dtype=np.float64)
                               for k in k_values},
            "recall_at_k": {str(k): np.asarray(r_at_k[k], dtype=np.float64)
                            for k in k_values},
        }
    return result


# ----------------------------------------------------------------------
# 1. Null / weak baseline calibration
# ----------------------------------------------------------------------
def calibrate_null(
    ctx: DiagnosticContext,
    sources: dict[str, dict[str, PatchDescriptors]],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    query_sample: int = 800,
    seed: int = 0,
) -> dict:
    """Run ranking on each provided source (e.g. random, centroid).

    Establishes the harness's inherent generosity floor: with large GT sets per
    query, even random embeddings score non-zero mAP. Every real result must be
    read *relative to this floor*.
    """
    out = {}
    for name, emb in sources.items():
        out[name] = evaluate_ranking(
            ctx, emb, k_values=k_values, query_sample=query_sample, seed=seed
        )
    return out


# ----------------------------------------------------------------------
# 2. Per-interface (macro vs micro) retrieval
# ----------------------------------------------------------------------
def retrieval_by_interface(
    ctx: DiagnosticContext,
    embeddings: dict[str, PatchDescriptors],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    seed: int = 0,
    gallery_keep: Optional[Callable[[str, int], bool]] = None,
    max_queries_per_interface: int = 200,
) -> dict:
    """Per-interface mAP + macro-average across interfaces.

    For each of the 11 adjacencies, queries are that interface's contact patches
    (on both sides), evaluated against the full cross-fragment gallery. Reporting
    per-interface mAP exposes interface-size bias that a single micro-averaged
    number hides; the macro-average weights every interface equally.
    """
    rng = np.random.default_rng(seed)
    per_interface = {}
    macro_maps = []

    for (fid_a, fid_b), sides in ctx.interface_patch_ids.items():
        keys = [(fid_a, pid) for pid in sorted(sides[fid_a])]
        keys += [(fid_b, pid) for pid in sorted(sides[fid_b])]
        if not keys:
            per_interface[f"{fid_a}__{fid_b}"] = {
                "n_contact_patches": 0, "mAP": float("nan"), "low_confidence": True,
            }
            continue
        if len(keys) > max_queries_per_interface:
            sel = rng.choice(len(keys), size=max_queries_per_interface, replace=False)
            keys = [keys[i] for i in sel]

        res = evaluate_ranking(
            ctx, embeddings, k_values=k_values, query_keys=keys,
            gallery_keep=gallery_keep, seed=seed,
        )
        n_contact = len(sides[fid_a]) + len(sides[fid_b])
        entry = {
            "n_contact_patches": int(n_contact),
            "n_queries": res["n_queries"],
            "mAP": res["mAP"],
            "precision_at_k": res["precision_at_k"],
            # Flag tiny interfaces (e.g. F5<->F7) as low-confidence.
            "low_confidence": bool(n_contact < 20),
        }
        per_interface[f"{fid_a}__{fid_b}"] = entry
        if not np.isnan(res["mAP"]) and not entry["low_confidence"]:
            macro_maps.append(res["mAP"])

    return {
        "per_interface": per_interface,
        "macro_mAP_well_supported": float(np.mean(macro_maps)) if macro_maps else float("nan"),
        "n_well_supported_interfaces": len(macro_maps),
        "note": (
            "macro_mAP averages well-supported interfaces equally (n_contact>=20). "
            "Compare with the micro-averaged mAP from the main ranking run: a large "
            "micro>macro gap indicates interface-size bias."
        ),
    }


# ----------------------------------------------------------------------
# 3. Hard-negative mining + easy-vs-hard separability
# ----------------------------------------------------------------------
def mine_hard_negatives(
    ctx: DiagnosticContext,
    embeddings: dict[str, PatchDescriptors],
    *,
    per_fragment: int = 300,
    seed: int = 0,
) -> list[tuple[str, int, str, int, float]]:
    """Mine nearest cross-fragment, *non-adjacent* patch pairs by embedding dist.

    Returns tuples ``(fid_a, pid_a, fid_b, pid_b, distance)``. These are
    "far-but-similar" pairs that provably do not assemble (their fragments are
    not neighbours), i.e. the evaluation-only hard negatives Phase 3 could not
    produce spatially. Mined with FPFH by default (pass FPFH embeddings).
    """
    rng = np.random.default_rng(seed)
    gallery = build_gallery(embeddings)
    out: list[tuple[str, int, str, int, float]] = []

    for fid in ctx.fragment_ids:
        neighbors = ctx.neighbors_of(fid)
        # Rows in the gallery that belong to non-adjacent (and different) fragments.
        non_adj_mask = np.array(
            [(gallery.frag[i] != fid and gallery.frag[i] not in neighbors)
             for i in range(gallery.size)]
        )
        if not non_adj_mask.any():
            continue
        non_adj_rows = np.where(non_adj_mask)[0]

        pd = embeddings[fid]
        m = pd.patch_count
        sample_rows = (rng.choice(m, size=min(per_fragment, m), replace=False)
                       if m > per_fragment else np.arange(m))
        for r in sample_rows:
            qvec = pd.descriptors[r]
            dists = np.linalg.norm(gallery.desc[non_adj_rows] - qvec, axis=1)
            j = int(np.argmin(dists))
            grow = non_adj_rows[j]
            out.append((fid, int(pd.patch_ids[r]),
                        str(gallery.frag[grow]), int(gallery.patch[grow]),
                        float(dists[j])))
    return out


def _pair_distance(
    embeddings: dict[str, PatchDescriptors], fa: str, pa: int, fb: str, pb: int
) -> Optional[float]:
    pda, pdb = embeddings.get(fa), embeddings.get(fb)
    if pda is None or pdb is None:
        return None
    # patch ids are 0..M-1 in row order.
    try:
        va = pda.descriptors[pa]
        vb = pdb.descriptors[pb]
    except IndexError:
        return None
    return float(np.linalg.norm(va - vb))


def easy_vs_hard_separability(
    ctx: DiagnosticContext,
    embeddings: dict[str, PatchDescriptors],
    hard_negatives: list[tuple[str, int, str, int, float]],
    *,
    pos_sample: int = 20000,
    easy_sample: int = 20000,
    seed: int = 0,
    held_out: Optional[str] = None,
) -> dict:
    """Pair-level AUC: positives vs easy negatives, and positives vs hard negatives.

    Score = negative embedding distance (higher = more similar = more
    positive-like). A method that only resists *easy* negatives (random,
    non-adjacent) but collapses on *hard* negatives (look-alike non-adjacent)
    is riding similarity, not assembly compatibility. The AUC gap quantifies it.

    ``held_out``, if given, restricts every pool (positives, easy negatives,
    hard negatives) to pairs where *neither* endpoint fragment is
    ``held_out``. This is what makes condition 3 (the easy-vs-hard gap) a
    held-out-fold measurement under LOFO instead of a training-set one: if
    the encoder trains on fold ``f``'s six visible fragments, condition 3 for
    fold ``f`` must not be scored using any pair touching fragment ``f``.
    Without this filter the instrument silently measures the training set
    (see PHASE5A_TRAINING_DESIGN_REVISED.md, item 1).
    """
    rng = np.random.default_rng(seed)
    pairs = ctx.pairs
    vocab = pairs.fragment_vocab

    def _fragment_mask(idx: np.ndarray) -> np.ndarray:
        """True where the pair at row ``idx`` does not touch ``held_out``."""
        if held_out is None:
            return np.ones(idx.shape, dtype=bool)
        fa = vocab[pairs.fragment_A_idx[idx]]
        fb = vocab[pairs.fragment_B_idx[idx]]
        return (fa != held_out) & (fb != held_out)

    # Positive distances.
    pos_idx = np.where(pairs.labels == 1)[0]
    pos_idx = pos_idx[_fragment_mask(pos_idx)]
    if pos_sample and pos_idx.size > pos_sample:
        pos_idx = rng.choice(pos_idx, size=pos_sample, replace=False)
    pos_d = []
    for p in pos_idx:
        d = _pair_distance(
            embeddings,
            str(vocab[pairs.fragment_A_idx[p]]), int(pairs.patch_A_ids[p]),
            str(vocab[pairs.fragment_B_idx[p]]), int(pairs.patch_B_ids[p]),
        )
        if d is not None:
            pos_d.append(d)
    pos_d = np.asarray(pos_d)

    # Easy negative distances (random label == -1).
    easy_idx = np.where(pairs.labels == -1)[0]
    easy_idx = easy_idx[_fragment_mask(easy_idx)]
    if easy_sample and easy_idx.size > easy_sample:
        easy_idx = rng.choice(easy_idx, size=easy_sample, replace=False)
    easy_d = []
    for p in easy_idx:
        d = _pair_distance(
            embeddings,
            str(vocab[pairs.fragment_A_idx[p]]), int(pairs.patch_A_ids[p]),
            str(vocab[pairs.fragment_B_idx[p]]), int(pairs.patch_B_ids[p]),
        )
        if d is not None:
            easy_d.append(d)
    easy_d = np.asarray(easy_d)

    # Hard negative distances: the hard-negative *pairs* are fixed (mined once
    # with FPFH for a fair, source-independent set), but the distance must be
    # recomputed under THIS source's embedding -- otherwise a non-FPFH source is
    # scored against FPFH distances, which is meaningless.
    hard_d = []
    for (fa, pa, fb, pb, _mined_dist) in hard_negatives:
        if held_out is not None and (fa == held_out or fb == held_out):
            continue
        d = _pair_distance(embeddings, fa, pa, fb, pb)
        if d is not None:
            hard_d.append(d)
    hard_d = np.asarray(hard_d) if hard_d else np.empty(0)

    def _auc(neg: np.ndarray) -> float:
        if pos_d.size == 0 or neg.size == 0:
            return float("nan")
        scores = np.concatenate([-pos_d, -neg])
        labels = np.concatenate([np.ones(pos_d.size), np.zeros(neg.size)])
        return roc_auc(scores, labels)

    auc_easy = _auc(easy_d)
    auc_hard = _auc(hard_d)
    return {
        "held_out_fragment": held_out,
        "n_positive": int(pos_d.size),
        "n_easy_negative": int(easy_d.size),
        "n_hard_negative": int(hard_d.size),
        "mean_distance_positive": float(pos_d.mean()) if pos_d.size else float("nan"),
        "mean_distance_easy_negative": float(easy_d.mean()) if easy_d.size else float("nan"),
        "mean_distance_hard_negative": float(hard_d.mean()) if hard_d.size else float("nan"),
        "auc_pos_vs_easy": auc_easy,
        "auc_pos_vs_hard": auc_hard,
        "easy_minus_hard_auc_gap": (
            float(auc_easy - auc_hard)
            if not (np.isnan(auc_easy) or np.isnan(auc_hard)) else float("nan")
        ),
        "note": (
            "A large easy-minus-hard gap means the embedding separates positives "
            "from random negatives but not from look-alike non-adjacent patches "
            "-> similarity, not compatibility."
        ),
    }


# ----------------------------------------------------------------------
# 4. Leave-one-fragment-out, per fold
# ----------------------------------------------------------------------
def _bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Percentile bootstrap CI for the mean of per-query metric ``values``.

    Resamples queries with replacement -- this captures the dominant source of
    per-fold variance here: few true neighbours per query and a modest query
    count make the mean noisy. Returns mean, lo/hi bounds and the CI half-width.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "half_width": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    mean = float(values.mean())
    return {"mean": mean, "lo": lo, "hi": hi,
            "half_width": float((hi - lo) / 2), "n": int(n)}


def lofo_per_fold(
    ctx: DiagnosticContext,
    embeddings: dict[str, PatchDescriptors],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    seed: int = 0,
    low_confidence_fragments: tuple[str, ...] = (),
    bootstrap: bool = True,
    n_boot: int = 2000,
    ci_metric_ks: tuple[int, ...] = (1,),
    hard_negatives: Optional[list[tuple[str, int, str, int, float]]] = None,
) -> dict:
    """Leave-one-fragment-out retrieval, reported per fold (never averaged).

    For held-out fragment ``f``: queries are ``f``'s contact patches; the gallery
    is all patches from the other six fragments. For a *fixed* embedding (random,
    FPFH) this is a per-fragment breakdown; for a *learned* checkpoint, the
    embeddings passed for fold ``f`` must come from a model trained without ``f``.
    The per-fold structure is identical either way, so the trained model reuses
    this function directly.

    When ``bootstrap`` is set, each fold also carries percentile bootstrap CIs
    for mAP and P@k (``ci_metric_ks``), computed by resampling queries. These
    quantify per-fold noise so a Phase-5A "beats FPFH" claim can be checked
    against overlapping CIs rather than eyeballed point estimates.

    When ``hard_negatives`` is given, each fold also carries a
    ``hard_negative_strata`` entry computed with ``held_out=held`` (see
    ``easy_vs_hard_separability``), so condition 3 (the easy-vs-hard AUC gap)
    is measured per fold, restricted to pairs that do not touch the held-out
    fragment -- the fix for the circularity described in
    PHASE5A_TRAINING_DESIGN_REVISED.md item 1.
    """
    folds = {}
    for held in ctx.fragment_ids:
        # Queries: contact patches on the held-out fragment.
        q_keep = lambda fid, pid, _h=held: (fid == _h and ctx.is_contact_patch(fid, pid))
        # Gallery: everything except the held-out fragment.
        g_keep = lambda fid, pid, _h=held: fid != _h
        res = evaluate_ranking(
            ctx, embeddings, k_values=k_values, seed=seed,
            gallery_keep=g_keep, query_keep=q_keep, query_sample=0,
            return_per_query=bootstrap,
        )
        n_neighbors = len(ctx.neighbors_of(held))
        entry = {
            "held_out_fragment": held,
            "n_queries": res["n_queries"],
            "gallery_size": res["gallery_size"],
            "n_neighbors": n_neighbors,
            "mAP": res["mAP"],
            "precision_at_k": res["precision_at_k"],
            "recall_at_k": res["recall_at_k"],
            "low_confidence": bool(held in low_confidence_fragments or n_neighbors <= 1),
        }
        if bootstrap and "per_query" in res:
            pq = res["per_query"]
            ci = {"mAP": _bootstrap_ci(pq["ap"], n_boot=n_boot, seed=seed)}
            for k in ci_metric_ks:
                ci[f"precision_at_{k}"] = _bootstrap_ci(
                    pq["precision_at_k"][str(k)], n_boot=n_boot, seed=seed
                )
            entry["ci_95"] = ci
        if hard_negatives is not None:
            entry["hard_negative_strata"] = easy_vs_hard_separability(
                ctx, embeddings, hard_negatives, seed=seed, held_out=held
            )
        folds[held] = entry
    return {
        "folds": folds,
        "note": (
            "Per-fold, not averaged (n=7). Folds flagged low_confidence have "
            "weak adjacency support (e.g. peripheral fragments / tiny contacts) "
            "and must be interpreted separately from well-supported folds. "
            "ci_95 are percentile bootstrap CIs over resampled queries: use "
            "them to check whether a fold's margin over baseline exceeds noise. "
            "hard_negative_strata (when present) is computed with held_out=<fold's "
            "fragment>, i.e. condition 3 for this fold excludes every pair that "
            "touches the held-out fragment."
        ),
    }
