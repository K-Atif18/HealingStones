"""Descriptor-based retrieval and separability metrics (Phase 4).

Two complementary evaluations against the Phase 3 ground truth:

1. **Pair-level separability.** For every labelled pair (positive vs random
   negative) compute the descriptor distance between patch A and patch B, then
   measure how well distance alone separates the classes: ROC-AUC and average
   precision. This is the headline "does similarity carry assembly signal?"
   number, and it feeds the Phase 4 validation gate.

2. **Ranking retrieval.** For a set of query patches (each of which has >=1
   true positive partner), rank *all cross-fragment patches* by descriptor
   distance and compute Precision@K, Recall@K, and mean Average Precision. A
   query never retrieves patches from its own fragment.

Both evaluations are additionally **stratified**:

* by contact overlap (a proxy for break-surface vs original-surface patches):
  high mutual overlap => the pair straddles the break, where similarity is
  expected to be weakest (complementarity). Reporting the AUC gap between the
  low-overlap and high-overlap strata quantifies the similarity-vs-
  complementarity blind spot that motivates the learned phases.

All metrics are implemented from scratch (no sklearn dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from baseline_geometry.data_access import PairTable
from baseline_geometry.descriptors import PatchDescriptors

__all__ = [
    "roc_auc",
    "average_precision",
    "evaluate_pairs",
    "evaluate_ranking",
]


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC where a *higher* score should indicate a positive.

    Computed via the rank-sum (Mann-Whitney U) identity, with tie handling.
    """
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = ranks[labels].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision where a *higher* score indicates a positive."""
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    cum_tp = np.cumsum(sorted_labels)
    precision = cum_tp / (np.arange(len(sorted_labels)) + 1)
    # AP = mean precision at the ranks where a true positive is retrieved.
    ap = (precision * sorted_labels).sum() / sorted_labels.sum()
    return float(ap)


def _descriptor_lookup(desc_by_fragment: dict[str, PatchDescriptors]) -> dict[str, dict[int, int]]:
    """Map fragment_id -> {patch_id -> row index in its descriptor matrix}."""
    lookup: dict[str, dict[int, int]] = {}
    for fid, pd in desc_by_fragment.items():
        lookup[fid] = {int(pid): row for row, pid in enumerate(pd.patch_ids)}
    return lookup


# ---------------------------------------------------------------------------
# Pair-level evaluation
# ---------------------------------------------------------------------------
def evaluate_pairs(
    pairs: PairTable,
    desc_by_fragment: dict[str, PatchDescriptors],
    *,
    sample: int = 0,
    seed: int = 0,
    high_overlap_threshold: float = 0.6,
    low_overlap_threshold: float = 0.35,
) -> dict:
    """Pair-level separability of positives vs random negatives.

    Score = negative descriptor distance (so higher = more similar = more
    likely positive). Returns ROC-AUC / AP overall and stratified by the
    minimum contact overlap of the pair.
    """
    vocab = pairs.fragment_vocab
    lookup = _descriptor_lookup(desc_by_fragment)

    mask = (pairs.labels == 1) | (pairs.labels == -1)
    idx = np.where(mask)[0]
    if sample and idx.size > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=sample, replace=False)

    scores = np.empty(idx.size, dtype=np.float64)
    labels = np.empty(idx.size, dtype=bool)
    min_overlap = np.empty(idx.size, dtype=np.float64)
    valid = np.zeros(idx.size, dtype=bool)

    for out_i, p in enumerate(idx):
        fid_a = str(vocab[pairs.fragment_A_idx[p]])
        fid_b = str(vocab[pairs.fragment_B_idx[p]])
        row_a = lookup.get(fid_a, {}).get(int(pairs.patch_A_ids[p]))
        row_b = lookup.get(fid_b, {}).get(int(pairs.patch_B_ids[p]))
        if row_a is None or row_b is None:
            continue
        va = desc_by_fragment[fid_a].descriptors[row_a]
        vb = desc_by_fragment[fid_b].descriptors[row_b]
        dist = float(np.linalg.norm(va - vb))
        scores[out_i] = -dist
        labels[out_i] = pairs.labels[p] == 1
        min_overlap[out_i] = min(pairs.contact_overlap_A[p], pairs.contact_overlap_B[p])
        valid[out_i] = True

    scores, labels, min_overlap = scores[valid], labels[valid], min_overlap[valid]

    result = {
        "n_pairs_evaluated": int(scores.size),
        "n_positive": int(labels.sum()),
        "n_negative": int((~labels).sum()),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "mean_distance_positive": float(-scores[labels].mean()) if labels.any() else float("nan"),
        "mean_distance_negative": float(-scores[~labels].mean()) if (~labels).any() else float("nan"),
    }

    # Stratify by contact overlap (positives only carry meaningful overlap; we
    # split the positive set and pair each stratum against ALL negatives).
    neg_mask = ~labels
    pos_mask = labels
    high_pos = pos_mask & (min_overlap >= high_overlap_threshold)
    low_pos = pos_mask & (min_overlap <= low_overlap_threshold)

    def _stratum_auc(pos_sel: np.ndarray) -> dict:
        sel = pos_sel | neg_mask
        if pos_sel.sum() == 0 or neg_mask.sum() == 0:
            return {"n_positive": int(pos_sel.sum()), "roc_auc": float("nan"),
                    "average_precision": float("nan")}
        return {
            "n_positive": int(pos_sel.sum()),
            "roc_auc": roc_auc(scores[sel], labels[sel]),
            "average_precision": average_precision(scores[sel], labels[sel]),
        }

    result["stratified"] = {
        "high_overlap_positives": _stratum_auc(high_pos),
        "low_overlap_positives": _stratum_auc(low_pos),
        "high_overlap_threshold": high_overlap_threshold,
        "low_overlap_threshold": low_overlap_threshold,
        "note": (
            "High-overlap positives straddle the break surface (complementary "
            "geometry); a lower AUC here than for low-overlap positives is the "
            "expected similarity-vs-complementarity blind spot."
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Ranking retrieval
# ---------------------------------------------------------------------------
def evaluate_ranking(
    pairs: PairTable,
    desc_by_fragment: dict[str, PatchDescriptors],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    query_sample: int = 800,
    seed: int = 0,
) -> dict:
    """Cross-fragment ranking retrieval.

    Builds a global gallery of all patches (concatenated descriptors). Each
    query is a patch that participates in >=1 positive pair; its ground-truth
    partners are all patches linked to it by a positive label. The query ranks
    all patches from *other* fragments by descriptor distance. Reports P@K,
    R@K, and mAP.
    """
    vocab = pairs.fragment_vocab
    fragment_ids = list(desc_by_fragment.keys())

    # Global gallery.
    gallery_desc = []
    gallery_frag = []
    gallery_patch = []
    for fid in fragment_ids:
        pd = desc_by_fragment[fid]
        gallery_desc.append(pd.descriptors)
        gallery_frag.extend([fid] * pd.patch_count)
        gallery_patch.extend(int(x) for x in pd.patch_ids)
    gallery_desc = np.vstack(gallery_desc)              # (G, D)
    gallery_frag = np.asarray(gallery_frag, dtype=object)
    gallery_patch = np.asarray(gallery_patch, dtype=np.int64)
    global_row = {
        (gallery_frag[i], int(gallery_patch[i])): i for i in range(len(gallery_patch))
    }

    # Build positive adjacency: (frag, patch) -> set of global gallery rows.
    pos_idx = np.where(pairs.labels == 1)[0]
    positives: dict[tuple, set] = {}

    def _add(fa, pa, fb, pb):
        key = (fa, int(pa))
        gb = global_row.get((fb, int(pb)))
        if gb is None:
            return
        positives.setdefault(key, set()).add(gb)

    for p in pos_idx:
        fa = str(vocab[pairs.fragment_A_idx[p]])
        fb = str(vocab[pairs.fragment_B_idx[p]])
        _add(fa, pairs.patch_A_ids[p], fb, pairs.patch_B_ids[p])
        _add(fb, pairs.patch_B_ids[p], fa, pairs.patch_A_ids[p])

    query_keys = list(positives.keys())
    if query_sample and len(query_keys) > query_sample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(query_keys), size=query_sample, replace=False)
        query_keys = [query_keys[i] for i in sel]

    max_k = max(k_values)
    p_at_k = {k: [] for k in k_values}
    r_at_k = {k: [] for k in k_values}
    ap_list = []

    for key in query_keys:
        fid_q, pid_q = key
        qrow = global_row.get((fid_q, pid_q))
        if qrow is None:
            continue
        qvec = gallery_desc[qrow]
        # Cross-fragment mask.
        cross = gallery_frag != fid_q
        if not cross.any():
            continue
        cand_rows = np.where(cross)[0]
        dists = np.linalg.norm(gallery_desc[cand_rows] - qvec, axis=1)
        order = np.argsort(dists, kind="mergesort")
        ranked_rows = cand_rows[order]

        gt = positives[key]
        n_gt = len(gt)
        if n_gt == 0:
            continue
        rel = np.array([1.0 if r in gt else 0.0 for r in ranked_rows], dtype=np.float64)

        for k in k_values:
            topk = rel[:k]
            p_at_k[k].append(topk.sum() / k)
            r_at_k[k].append(topk.sum() / n_gt)

        cum = np.cumsum(rel)
        precision = cum / (np.arange(len(rel)) + 1)
        ap_list.append((precision * rel).sum() / n_gt)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_queries": len(ap_list),
        "gallery_size": int(gallery_desc.shape[0]),
        "mAP": _mean(ap_list),
        "precision_at_k": {str(k): _mean(p_at_k[k]) for k in k_values},
        "recall_at_k": {str(k): _mean(r_at_k[k]) for k in k_values},
    }
