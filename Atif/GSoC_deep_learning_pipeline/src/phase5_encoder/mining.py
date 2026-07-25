"""On-the-fly, per-epoch, fold-restricted hard-negative mining.

Implements PHASE5A_TRAINING_DESIGN_REVISED.md §2/§2b: this is the PRIMARY
training hard-negative mechanism (the FPFH-mined 2,100 pairs are frozen and
evaluation-only, never imported here). Mining runs once per epoch, under
``torch.no_grad()`` in ``model.eval()``, against the fold's full non-adjacent
candidate pool (no subsampling -- see §2b item 2), and caches the resulting
embedding table for O(1) reuse by every anchor's lookup that epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from phase5_diagnostics.context import DiagnosticContext
from baseline_geometry.data_access import PatchTable
from phase5_encoder.preprocess import prepare_patch
from phase5_encoder.features import knn_ppf_features


__all__ = ["MinedNegatives", "mine_epoch_hard_negatives"]


@dataclass
class MinedNegatives:
    """Cached per-epoch mining result for one fold.

    ``lookup[(fid, pid)] -> (neg_fid, neg_pid, distance)``: the single
    nearest non-adjacent patch under the CURRENT encoder state, for every
    training anchor patch that had >=1 non-adjacent candidate this fold.
    ``per_fragment_distance``: fragment -> list of mined distances this
    epoch (§2b item 4, required per-fragment difficulty logging).
    """
    lookup: dict[tuple[str, int], tuple[str, int, float]]
    per_fragment_distance: dict[str, list]


def _encode_pool(
    model, patch_table: PatchTable, patch_ids: np.ndarray, *,
    n_points: int, k: int, device: str,
) -> np.ndarray:
    """Encode ``patch_ids`` from ``patch_table`` under no_grad/eval. Returns
    (len(patch_ids), out_dim) embeddings. This is the mining-pass encode
    measured in PHASE5A_TRAINING_DESIGN_REVISED.md §2b item 3
    (scripts/measure_step_memory.py's ``measure_mining_pass``)."""
    was_training = model.training
    model.eval()
    feats = []
    for pid in patch_ids.tolist():
        row = patch_table.row_for_patch_id(pid)
        pts, nrm = prepare_patch(
            patch_table.patch_points(row), patch_table.patch_normals(row),
            n_points, training=False,
        )
        feats.append(knn_ppf_features(pts, nrm, k=k))
    feats_t = torch.from_numpy(np.stack(feats)).float().to(device)
    with torch.no_grad():
        emb = model(feats_t).cpu().numpy()
    if was_training:
        model.train()
    return emb


def mine_epoch_hard_negatives(
    ctx: DiagnosticContext,
    model,
    held_out: str,
    train_anchor_ids: dict[str, np.ndarray],
    *,
    n_points: int,
    k: int = 16,
    device: str = "cpu",
    seed: int = 0,
) -> MinedNegatives:
    """One mining pass for the current epoch (§2b: N = 1 epoch, full pool,
    no subsampling). ``train_anchor_ids``: fragment -> patch ids to mine a
    negative for (typically each fragment's contact patches for this fold).

    Candidate pool per fragment ``fid``: every patch belonging to a fragment
    NOT in ``ctx.neighbors_of(fid) | {fid, held_out}`` -- i.e. the same
    non-adjacent definition ``mine_hard_negatives`` uses
    (src/phase5_diagnostics/ranking.py), with the held-out fragment removed
    from every candidate pool too, since its patches must not be visible to
    training in any form (§1).
    """
    per_fragment_distance: dict[str, list] = {}
    lookup: dict[tuple[str, int], tuple[str, int, float]] = {}

    train_fragments = [f for f in ctx.fragment_ids if f != held_out]

    # Cache one encoded pool per distinct non-adjacent-fragment-set, keyed by
    # the frozenset of fragments it spans (several anchor fragments can share
    # an identical candidate pool definition, but pool CONTENTS differ per
    # anchor fragment because the anchor's own fragment and its neighbours are
    # excluded -- so we cache per anchor fragment, not globally).
    for fid in train_fragments:
        if fid not in train_anchor_ids or train_anchor_ids[fid].size == 0:
            continue
        neighbors = ctx.neighbors_of(fid)
        non_adj_frags = [f for f in train_fragments if f != fid and f not in neighbors]
        if not non_adj_frags:
            continue

        # Encode each non-adjacent fragment's full patch pool once (§2b: no
        # subsampling, full pool, cached for reuse across all anchors below).
        pool_emb_chunks = []
        pool_fid_arr = []
        pool_pid_arr = []
        for nfid in non_adj_frags:
            pt = ctx.patch_tables[nfid]
            ids = np.arange(pt.patch_count, dtype=np.int64)
            emb = _encode_pool(model, pt, ids, n_points=n_points, k=k, device=device)
            pool_emb_chunks.append(emb)
            pool_fid_arr.extend([nfid] * pt.patch_count)
            pool_pid_arr.extend(ids.tolist())
        pool_emb = np.concatenate(pool_emb_chunks, axis=0)
        pool_fid_arr = np.array(pool_fid_arr, dtype=object)
        pool_pid_arr = np.array(pool_pid_arr, dtype=np.int64)

        anchor_pt = ctx.patch_tables[fid]
        anchor_emb = _encode_pool(
            model, anchor_pt, train_anchor_ids[fid], n_points=n_points, k=k, device=device
        )
        dists = np.linalg.norm(pool_emb[None, :, :] - anchor_emb[:, None, :], axis=2)
        j = np.argmin(dists, axis=1)
        for row, pid in enumerate(train_anchor_ids[fid].tolist()):
            neg_fid = str(pool_fid_arr[j[row]])
            neg_pid = int(pool_pid_arr[j[row]])
            dist = float(dists[row, j[row]])
            lookup[(fid, pid)] = (neg_fid, neg_pid, dist)
            per_fragment_distance.setdefault(fid, []).append(dist)

    return MinedNegatives(lookup=lookup, per_fragment_distance=per_fragment_distance)
