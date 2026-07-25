"""Tests for the fragment->interface->partner sampler and on-the-fly mining
(PHASE5A_TRAINING_DESIGN_REVISED.md §§2, 2b, 4, 7). Reuses the toy 3-fragment
context/embeddings fixtures from test_diagnostics.py's pattern.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phase5_encoder.sampler import build_fold_sampler, InterfaceWeighting
from phase5_encoder.mining import mine_epoch_hard_negatives
from phase5_encoder.model import PointNetEncoder


def _toy_context_two_interfaces():
    """A,B,C,D with two interfaces sharing fragment B: (A,B) and (B,C). D is
    isolated (no adjacency) so it's a valid non-adjacent mining candidate for
    A and C but never a training anchor itself."""
    fids = ["A", "B", "C", "D"]

    def table(fid, m=4):
        return SimpleNamespace(
            fragment_id=fid, patch_count=m,
            patch_ids=np.arange(m, dtype=np.int64),
            offsets=np.array([[i, i + 1] for i in range(m)], dtype=np.int64),
            global_coords=np.random.default_rng(hash(fid) % (2**31)).standard_normal((m, 3)),
            normals=np.tile([0.0, 0.0, 1.0], (m, 1)),
        )
        # note: row_for_patch_id/patch_points/patch_normals are added below

    patch_tables = {}
    for fid in fids:
        t = table(fid)
        t.row_for_patch_id = lambda pid, _t=t: pid
        t.patch_points = lambda row, _t=t: _t.global_coords[row:row + 1].repeat(8, axis=0)
        t.patch_normals = lambda row, _t=t: _t.normals[row:row + 1].repeat(8, axis=0)
        patch_tables[fid] = t

    vocab = np.array(fids)
    # Positive pairs: (A,i)<->(B,i) for i in 0..2 (interface AB), (B,i)<->(C,i)
    # for i in 1..3 (interface BC). Some overlap on B (patch 1,2 multi-interface).
    fa_idx, fb_idx, pa, pb, labels = [], [], [], [], []
    for i in range(3):
        fa_idx.append(0); fb_idx.append(1); pa.append(i); pb.append(i); labels.append(1)
    for i in range(1, 4):
        fa_idx.append(1); fb_idx.append(2); pa.append(i); pb.append(i); labels.append(1)

    pairs = SimpleNamespace(
        fragment_vocab=vocab,
        fragment_A_idx=np.array(fa_idx, dtype=np.int64),
        fragment_B_idx=np.array(fb_idx, dtype=np.int64),
        patch_A_ids=np.array(pa, dtype=np.int64),
        patch_B_ids=np.array(pb, dtype=np.int64),
        labels=np.array(labels, dtype=np.int64),
    )

    ctx = SimpleNamespace(
        fragment_ids=fids,
        patch_tables=patch_tables,
        pairs=pairs,
        adjacent_pairs=[("A", "B"), ("B", "C")],
        contact_patch_ids={
            "A": {0, 1, 2}, "B": {0, 1, 2, 3}, "C": {1, 2, 3}, "D": set(),
        },
        interface_patch_ids={
            ("A", "B"): {"A": {0, 1, 2}, "B": {0, 1, 2}},
            ("B", "C"): {"B": {1, 2, 3}, "C": {1, 2, 3}},
        },
        distinctiveness={},
    )
    ctx.neighbors_of = lambda fid: {b for a, b in ctx.adjacent_pairs if a == fid} | \
                                   {a for a, b in ctx.adjacent_pairs if b == fid}
    ctx.is_contact_patch = lambda fid, pid: pid in ctx.contact_patch_ids.get(fid, set())
    return ctx


def test_fold_sampler_excludes_held_out_interfaces():
    ctx = _toy_context_two_interfaces()
    sampler = build_fold_sampler(ctx, held_out="C", val_fraction=0.0, seed=0)
    # (B,C) touches C -> excluded entirely. Only (A,B) remains.
    assert ("B", "C") not in sampler.train_interface_pair_rows
    assert ("A", "B") in sampler.train_interface_pair_rows
    assert "C" not in sampler.train_fragments


def test_fold_sampler_val_split_is_disjoint_from_train():
    ctx = _toy_context_two_interfaces()
    sampler = build_fold_sampler(ctx, held_out="D", val_fraction=0.34, seed=0)
    for iface in sampler.train_interface_pair_rows:
        train_rows = set(sampler.train_interface_pair_rows[iface].tolist())
        val_rows = set(sampler.val_interface_pair_rows.get(iface, np.array([])).tolist())
        assert train_rows.isdisjoint(val_rows), iface
    assert sampler.n_val_pairs() >= 1


def test_multi_interface_patches_detected():
    ctx = _toy_context_two_interfaces()
    sampler = build_fold_sampler(ctx, held_out="D", val_fraction=0.0, seed=0)
    # B's patches 1,2 belong to BOTH (A,B) and (B,C) -> multi-interface.
    assert 1 in sampler.multi_interface_patches["B"]
    assert 2 in sampler.multi_interface_patches["B"]
    # B's patch 0 only belongs to (A,B) -> not multi-interface.
    assert 0 not in sampler.multi_interface_patches["B"]


def test_interface_weighting_caps_extreme_ratio():
    w = InterfaceWeighting(max_ratio=20.0)
    counts = {("F2", "F4"): 1232, ("F6", "F7"): 214446}
    weights = w.weights_for(counts)
    ratio = max(weights.values()) / min(weights.values())
    assert ratio <= 20.0 + 1e-9, weights
    # Without a cap, pure inverse-frequency would give ~174x.
    uncapped_ratio = 214446 / 1232
    assert ratio < uncapped_ratio


def test_sample_batch_only_draws_from_train_split():
    ctx = _toy_context_two_interfaces()
    sampler = build_fold_sampler(ctx, held_out="D", val_fraction=0.34, seed=0)
    rng = np.random.default_rng(1)
    batch = sampler.sample_batch(ctx, batch_size=50, rng=rng)
    assert len(batch["anchor_fid"]) == 50
    # Every interface drawn must be a training-visible one.
    assert set(batch["interface"]) <= set(sampler.train_interface_pair_rows.keys())


def test_mine_epoch_hard_negatives_excludes_held_out_and_neighbors():
    """Mined negatives for A must come only from D (A's sole non-adjacent,
    non-held-out fragment) when C is held out and B is A's neighbour."""
    ctx = _toy_context_two_interfaces()
    torch.manual_seed(0)
    model = PointNetEncoder(out_dim=16, pair_input=True)
    anchor_ids = {"A": np.array([0, 1, 2], dtype=np.int64)}
    result = mine_epoch_hard_negatives(
        ctx, model, held_out="C", train_anchor_ids=anchor_ids,
        n_points=8, k=4, device="cpu", seed=0,
    )
    for (fid, pid), (neg_fid, neg_pid, dist) in result.lookup.items():
        assert fid == "A"
        assert neg_fid == "D"  # only non-adjacent, non-held-out fragment for A
    assert "A" in result.per_fragment_distance
    assert len(result.per_fragment_distance["A"]) == 3


def test_train_one_fold_smoke():
    """Mechanical smoke test: train_one_fold must run end to end on the toy
    context with tiny epoch/step counts and produce a history with an
    explicit stopped_by field -- this is NOT a real training run, only a
    plumbing check (per PHASE5A_TRAINING_DESIGN_REVISED.md item 5, the human
    runs the real training)."""
    import tempfile
    from phase5_encoder.train import TrainConfig, train_one_fold

    ctx = _toy_context_two_interfaces()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            n_points=8, k=4, out_dim=8, batch_size=4, epoch_cap=2, patience=1,
            steps_per_epoch=2, val_fraction=0.34, device="cpu", out_dir=tmp,
        )
        history = train_one_fold(ctx, held_out="D", cfg=cfg)
        assert history["stopped_by"] in ("patience", "epoch_cap")
        assert len(history["epochs"]) >= 1
        assert os.path.exists(history["history_path"])
        assert os.path.exists(history["checkpoint_path"])
        for ep in history["epochs"]:
            assert "interface_draw_counts" in ep
            assert "per_fragment_mined_distance_mean" in ep
