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
    (scripts/measure_step_memory.py's ``measure_mining_pass``).

    ALWAYS eval-mode + ``training=False`` in prepare_patch: jitter must never
    enter the mining path, or mined distances would stop being reproducible
    within an epoch and the mining cache would not be a pure function of the
    (frozen, this-epoch) weights. This is a hard invariant -- see
    ``_encode_training_table`` which reuses it.
    """
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


def _encode_training_table(
    ctx: DiagnosticContext, model, train_fragments: list[str], *,
    n_points: int, k: int, device: str,
) -> dict[str, np.ndarray]:
    """Encode EVERY patch of every training fragment exactly once this epoch,
    under the same eval-mode/``training=False`` path as ``_encode_pool``.

    Returns ``fid -> (patch_count, out_dim)`` embedding matrix, row i = patch
    id i (patch ids are 0..M-1 in row order per the PatchTable contract).

    This is the optimization that removes the 3.2x redundancy measured on
    fold 1: the previous code re-encoded each fragment's patches once per
    anchor fragment whose non-adjacent pool contained it (18,887 encodes for
    fold 1), but a patch's embedding is identical regardless of which anchor
    it is being mined for or against -- so encode all 5,822 distinct training
    patches once and make both pool AND anchor accesses index lookups.
    Mathematically identical to the per-anchor-pool version (proven by
    ``tests/phase5_encoder/test_sampler_mining.py::
    test_mining_cache_matches_uncached``); only faster.
    """
    table: dict[str, np.ndarray] = {}
    for fid in train_fragments:
        pt = ctx.patch_tables[fid]
        ids = np.arange(pt.patch_count, dtype=np.int64)
        table[fid] = _encode_pool(model, pt, ids, n_points=n_points, k=k, device=device)
    return table


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

    Implementation: encode every training-fragment patch once into a cached
    eval-mode embedding table (``_encode_training_table``), then compute each
    anchor fragment's pool and anchor embeddings by INDEX LOOKUP into that
    table. This is a pure refactor of the earlier per-anchor-pool encoding
    (which re-encoded overlapping pools ~3.2x); the mined lookup dict and
    distances are identical (proven by test), only ~3.2x faster.
    """
    per_fragment_distance: dict[str, list] = {}
    lookup: dict[tuple[str, int], tuple[str, int, float]] = {}

    train_fragments = [f for f in ctx.fragment_ids if f != held_out]

    # Encode all distinct training-fragment patches ONCE this epoch.
    table = _encode_training_table(
        ctx, model, train_fragments, n_points=n_points, k=k, device=device
    )

    for fid in train_fragments:
        if fid not in train_anchor_ids or train_anchor_ids[fid].size == 0:
            continue
        neighbors = ctx.neighbors_of(fid)
        non_adj_frags = [f for f in train_fragments if f != fid and f not in neighbors]
        if not non_adj_frags:
            continue

        # Build the candidate pool by concatenating cached embeddings, in the
        # SAME order as the previous implementation (per non-adjacent fragment
        # in `non_adj_frags` order, then ascending patch id within each) so
        # that np.argmin tie-breaking is byte-for-byte identical to the
        # uncached version.
        pool_emb_chunks = []
        pool_fid_arr = []
        pool_pid_arr = []
        for nfid in non_adj_frags:
            emb = table[nfid]                       # (M_nfid, D), cached
            pool_emb_chunks.append(emb)
            pool_fid_arr.extend([nfid] * emb.shape[0])
            pool_pid_arr.extend(range(emb.shape[0]))
        pool_emb = np.concatenate(pool_emb_chunks, axis=0)
        pool_fid_arr = np.array(pool_fid_arr, dtype=object)
        pool_pid_arr = np.array(pool_pid_arr, dtype=np.int64)

        # Anchor embeddings are ALSO index lookups into the same cached table
        # (an anchor patch's embedding is the same vector whether it is being
        # mined for or mined against).
        anchor_ids = train_anchor_ids[fid]
        anchor_emb = table[fid][anchor_ids]

        dists = np.linalg.norm(pool_emb[None, :, :] - anchor_emb[:, None, :], axis=2)
        j = np.argmin(dists, axis=1)
        for row, pid in enumerate(anchor_ids.tolist()):
            neg_fid = str(pool_fid_arr[j[row]])
            neg_pid = int(pool_pid_arr[j[row]])
            dist = float(dists[row, j[row]])
            lookup[(fid, pid)] = (neg_fid, neg_pid, dist)
            per_fragment_distance.setdefault(fid, []).append(dist)

    return MinedNegatives(lookup=lookup, per_fragment_distance=per_fragment_distance)
