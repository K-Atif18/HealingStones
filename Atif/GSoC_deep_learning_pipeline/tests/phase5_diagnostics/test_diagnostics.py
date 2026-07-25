"""Tests for the Phase 5 diagnostic infrastructure.

These lock the behaviour of the *instrumentation* (not any model): the harness
metrics, the null-floor calibration, the shortcut probes, and -- crucially --
the pose-invariance gate used to prevent global-position leakage.
"""

from __future__ import annotations

import numpy as np
import pytest

from phase5_diagnostics import probes as pb
from phase5_diagnostics.ranking import roc_auc  # re-exported from baseline


def test_roc_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_chance():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(2000)
    labels = rng.integers(0, 2, size=2000)
    assert roc_auc(scores, labels) == pytest.approx(0.5, abs=0.05)


def test_pose_invariance_passes_for_invariant_encoder():
    """An encoder that uses only pairwise distances is pose-invariant."""
    def encode(points, normals):
        c = points.mean(axis=0)
        rel = points - c
        # rotation/translation-invariant summary: sorted pairwise-distance moments
        d = np.linalg.norm(rel, axis=1)
        return np.array([d.mean(), d.std(), d.max()])

    res = pb.pose_invariance_check(encode, n_trials=20, tol=1e-6, seed=1)
    assert res["passed"], res


def test_pose_invariance_fails_for_global_frame_encoder():
    """An encoder that reads absolute coordinates must FAIL the gate.

    This is the exact 'global-position leakage' bug the gate exists to catch.
    """
    def leaky_encode(points, normals):
        return points.mean(axis=0)  # absolute centroid -> pose sensitive

    res = pb.pose_invariance_check(leaky_encode, n_trials=20, tol=1e-3, seed=1)
    assert not res["passed"], res


def test_pose_invariance_detects_translation_only_leak():
    def translation_leak(points, normals):
        # rotation-invariant but translation-sensitive
        return np.array([np.linalg.norm(points.mean(axis=0))])

    res = pb.pose_invariance_check(translation_leak, n_trials=20, tol=1e-6, seed=2)
    assert not res["passed"], res


# ----------------------------------------------------------------------
# LOFO regression test: the held-out fragment's queries must be scored.
# This is the bug that made every fold report n_queries=0 / mAP=NaN.
# ----------------------------------------------------------------------
def _toy_context():
    """Build a minimal 3-fragment context with a hand-made positive pair table.

    Fragments A,B,C; A<->B adjacent (one positive pair), B<->C adjacent.
    Each fragment has 3 patches. Uses lightweight stand-ins matching the
    attributes the ranking code touches.
    """
    from types import SimpleNamespace
    import numpy as np

    fids = ["A", "B", "C"]

    def table(fid):
        m = 3
        return SimpleNamespace(
            fragment_id=fid,
            patch_count=m,
            patch_ids=np.arange(m, dtype=np.int64),
            offsets=np.array([[i, i + 1] for i in range(m)], dtype=np.int64),
            source_indices=np.arange(m, dtype=np.int64),
            global_coords=np.zeros((m, 3)),
            normals=np.tile([0.0, 0.0, 1.0], (m, 1)),
        )

    patch_tables = {f: table(f) for f in fids}

    # Positive pairs: (A,0)<->(B,0) and (B,1)<->(C,1). Labels +1.
    vocab = np.array(fids)
    fa_idx = np.array([0, 1], dtype=np.int64)   # A, B
    fb_idx = np.array([1, 2], dtype=np.int64)   # B, C
    pa = np.array([0, 1], dtype=np.int64)
    pb = np.array([0, 1], dtype=np.int64)
    labels = np.array([1, 1], dtype=np.int64)

    pairs = SimpleNamespace(
        fragment_vocab=vocab,
        fragment_A_idx=fa_idx, fragment_B_idx=fb_idx,
        patch_A_ids=pa, patch_B_ids=pb, labels=labels,
    )

    ctx = SimpleNamespace(
        fragment_ids=fids,
        patch_tables=patch_tables,
        pairs=pairs,
        adjacent_pairs=[("A", "B"), ("B", "C")],
        contact_patch_ids={"A": {0}, "B": {0, 1}, "C": {1}},
        interface_patch_ids={("A", "B"): {"A": {0}, "B": {0}},
                             ("B", "C"): {"B": {1}, "C": {1}}},
        distinctiveness={},
    )
    ctx.neighbors_of = lambda fid: {b for a, b in ctx.adjacent_pairs if a == fid} | \
                                   {a for a, b in ctx.adjacent_pairs if b == fid}
    ctx.is_contact_patch = lambda fid, pid: pid in ctx.contact_patch_ids.get(fid, set())
    ctx.total_patches = lambda: sum(t.patch_count for t in patch_tables.values())
    return ctx


def _toy_embeddings(ctx):
    """Embeddings where true partners are identical vectors (perfect retrieval)."""
    from baseline_geometry.descriptors import PatchDescriptors
    import numpy as np
    # Give (A,0),(B,0) the same vector; (B,1),(C,1) the same vector; others distinct.
    vec = {
        ("A", 0): [1, 0, 0, 0], ("B", 0): [1, 0, 0, 0],
        ("B", 1): [0, 1, 0, 0], ("C", 1): [0, 1, 0, 0],
        ("A", 1): [0, 0, 1, 0], ("A", 2): [0, 0, 0, 1],
        ("B", 2): [0.5, 0.5, 0, 0],
        ("C", 0): [0, 0, 1, 1], ("C", 2): [1, 1, 0, 0],
    }
    out = {}
    for fid in ctx.fragment_ids:
        M = ctx.patch_tables[fid].patch_count
        mat = np.array([vec[(fid, p)] for p in range(M)], dtype=np.float64)
        out[fid] = PatchDescriptors(fid, "toy", np.arange(M, dtype=np.int64), mat)
    return out


def test_lofo_folds_are_nonempty():
    """Every fold with adjacency support must produce a real query count."""
    from phase5_diagnostics import ranking as rk
    ctx = _toy_context()
    emb = _toy_embeddings(ctx)
    res = rk.lofo_per_fold(ctx, emb, k_values=(1,), seed=0)
    folds = res["folds"]
    # A holds out -> query (A,0), partner (B,0) is in gallery -> must score.
    assert folds["A"]["n_queries"] >= 1, folds["A"]
    # B holds out -> queries (B,0),(B,1); partners (A,0),(C,1) in gallery.
    assert folds["B"]["n_queries"] >= 1, folds["B"]
    # C holds out -> query (C,1), partner (B,1) in gallery.
    assert folds["C"]["n_queries"] >= 1, folds["C"]
    # With identical partner vectors, held-out retrieval should be perfect.
    assert folds["A"]["mAP"] == pytest.approx(1.0), folds["A"]


def test_lofo_has_bootstrap_cis():
    """Each fold carries a 95% bootstrap CI bracketing its point estimate."""
    from phase5_diagnostics import ranking as rk
    ctx = _toy_context()
    emb = _toy_embeddings(ctx)
    res = rk.lofo_per_fold(ctx, emb, k_values=(1,), seed=0, bootstrap=True, n_boot=500)
    for fid, f in res["folds"].items():
        if f["n_queries"] == 0:
            continue
        ci = f["ci_95"]["mAP"]
        assert ci["lo"] <= ci["mean"] <= ci["hi"], (fid, ci)
        assert ci["half_width"] >= 0.0


# ----------------------------------------------------------------------
# Condition-3 fix: easy_vs_hard_separability(held_out=...) must exclude every
# pair touching the held-out fragment, and lofo_per_fold must wire it through
# per fold. This is the regression test for
# PHASE5A_TRAINING_DESIGN_REVISED.md item 1 (the circularity bug).
# ----------------------------------------------------------------------
def test_easy_vs_hard_separability_excludes_held_out_fragment():
    """held_out must drop every positive/easy/hard pair touching that fragment."""
    from phase5_diagnostics import ranking as rk
    ctx = _toy_context()
    emb = _toy_embeddings(ctx)

    # Unrestricted: both positive pairs count (A,B) and (B,C).
    unrestricted = rk.easy_vs_hard_separability(ctx, emb, hard_negatives=[], seed=0)
    assert unrestricted["n_positive"] == 2, unrestricted

    # Holding out B must drop BOTH positive pairs, since both touch B.
    held_b = rk.easy_vs_hard_separability(
        ctx, emb, hard_negatives=[], seed=0, held_out="B"
    )
    assert held_b["n_positive"] == 0, held_b
    assert held_b["held_out_fragment"] == "B"

    # Holding out C must drop only the (B,C) pair, leaving (A,B).
    held_c = rk.easy_vs_hard_separability(
        ctx, emb, hard_negatives=[], seed=0, held_out="C"
    )
    assert held_c["n_positive"] == 1, held_c

    # Hard negatives touching the held-out fragment must also be dropped.
    hard_neg = [("A", 1, "C", 0, 0.5), ("A", 2, "B", 2, 0.3)]
    held_a = rk.easy_vs_hard_separability(
        ctx, emb, hard_negatives=hard_neg, seed=0, held_out="C"
    )
    # Only the second hard negative (A,B) survives; the first touches C.
    assert held_a["n_hard_negative"] == 1, held_a


def test_lofo_per_fold_wires_held_out_hard_negative_strata():
    """lofo_per_fold, given hard_negatives, must attach a per-fold
    hard_negative_strata restricted to that fold's held-out fragment --
    not the global, unrestricted numbers."""
    from phase5_diagnostics import ranking as rk
    ctx = _toy_context()
    emb = _toy_embeddings(ctx)
    hard_neg = [("A", 1, "C", 0, 0.5), ("A", 2, "B", 2, 0.3)]

    res = rk.lofo_per_fold(ctx, emb, k_values=(1,), seed=0, hard_negatives=hard_neg)
    folds = res["folds"]

    for fid, entry in folds.items():
        assert "hard_negative_strata" in entry, (fid, entry)
        strata = entry["hard_negative_strata"]
        assert strata["held_out_fragment"] == fid, (fid, strata)
        # No positive or hard-negative pair in this fold's strata may touch
        # the held-out fragment.
        vocab = ctx.pairs.fragment_vocab
        for p in range(len(ctx.pairs.labels)):
            fa, fb = str(vocab[ctx.pairs.fragment_A_idx[p]]), str(vocab[ctx.pairs.fragment_B_idx[p]])
            if fa == fid or fb == fid:
                # this pair must not have contributed -- checked indirectly via
                # counts already covered by test_easy_vs_hard_separability_*;
                # here we just confirm the strata dict carries the marker.
                pass

    # Fold B (holding out B) must report zero positives, since both toy
    # positive pairs touch B -- mirrors test_easy_vs_hard_separability_*.
    assert folds["B"]["hard_negative_strata"]["n_positive"] == 0, folds["B"]
    # Fold C must report exactly 1 positive (A,B) and exactly 1 hard negative
    # (the (A,B) one; the (A,C) one is dropped because it touches C).
    assert folds["C"]["hard_negative_strata"]["n_positive"] == 1, folds["C"]
    assert folds["C"]["hard_negative_strata"]["n_hard_negative"] == 1, folds["C"]


def test_runner_lofo_entry_differs_from_global_hard_negative_strata():
    """Sanity: the per-fold (held-out-restricted) strata must not simply
    equal the unrestricted, global entry -- otherwise the filter is a no-op."""
    from phase5_diagnostics import ranking as rk
    ctx = _toy_context()
    emb = _toy_embeddings(ctx)
    hard_neg = [("A", 1, "C", 0, 0.5), ("A", 2, "B", 2, 0.3)]

    global_strata = rk.easy_vs_hard_separability(ctx, emb, hard_neg, seed=0)
    lofo = rk.lofo_per_fold(ctx, emb, k_values=(1,), seed=0, hard_negatives=hard_neg)

    # Fold B's restricted n_positive (0) must differ from the global one (2).
    assert lofo["folds"]["B"]["hard_negative_strata"]["n_positive"] != global_strata["n_positive"]


def test_bootstrap_ci_shrinks_with_n():
    """CI half-width should shrink as the query sample grows (sanity)."""
    from phase5_diagnostics.ranking import _bootstrap_ci
    rng = np.random.default_rng(0)
    small = _bootstrap_ci(rng.random(20), n_boot=500, seed=0)
    large = _bootstrap_ci(rng.random(2000), n_boot=500, seed=0)
    assert large["half_width"] < small["half_width"]


# ----------------------------------------------------------------------
# Condition-6 refinement: held-out fragment-ID probe
# ----------------------------------------------------------------------
def test_heldout_fragment_id_probe_detects_island():
    """A source that puts a fragment on its own island => high self-retrieval.

    Build embeddings where fragment C's patches are all near-identical and far
    from A/B. Holding out C, its patches' nearest neighbours should be other C
    patches => self-retrieval well above the base rate.
    """
    from baseline_geometry.descriptors import PatchDescriptors
    ctx = _toy_context()
    # A,B spread out; C tightly clustered and displaced.
    vec = {
        ("A", 0): [1, 0, 0, 0], ("A", 1): [0, 1, 0, 0], ("A", 2): [0, 0, 1, 0],
        ("B", 0): [0, 0, 0, 1], ("B", 1): [1, 1, 0, 0], ("B", 2): [0, 1, 1, 0],
        ("C", 0): [5, 5, 5, 0], ("C", 1): [5, 5, 5, 0.01], ("C", 2): [5, 5, 5, 0.02],
    }
    emb = {}
    for fid in ctx.fragment_ids:
        M = ctx.patch_tables[fid].patch_count
        mat = np.array([vec[(fid, p)] for p in range(M)], dtype=np.float64)
        emb[fid] = PatchDescriptors(fid, "toy", np.arange(M, dtype=np.int64), mat)

    res = pb.heldout_fragment_id_probe(ctx, emb, held_out="C", k=1, seed=0)
    assert res["heldout_self_retrieval"] == pytest.approx(1.0), res
    assert res["ratio_over_base_rate"] > 1.0, res


def test_heldout_fragment_id_probe_no_island():
    """When a held-out fragment is interspersed, self-retrieval ~ base rate."""
    from baseline_geometry.descriptors import PatchDescriptors
    ctx = _toy_context()
    # All 9 patches on a shared line so each fragment's NN is often another frag.
    vec = {
        ("A", 0): [0, 0, 0, 0], ("B", 0): [1, 0, 0, 0], ("C", 0): [2, 0, 0, 0],
        ("A", 1): [3, 0, 0, 0], ("B", 1): [4, 0, 0, 0], ("C", 1): [5, 0, 0, 0],
        ("A", 2): [6, 0, 0, 0], ("B", 2): [7, 0, 0, 0], ("C", 2): [8, 0, 0, 0],
    }
    emb = {}
    for fid in ctx.fragment_ids:
        M = ctx.patch_tables[fid].patch_count
        mat = np.array([vec[(fid, p)] for p in range(M)], dtype=np.float64)
        emb[fid] = PatchDescriptors(fid, "toy", np.arange(M, dtype=np.int64), mat)

    res = pb.heldout_fragment_id_probe(ctx, emb, held_out="C", k=1, seed=0)
    # C patches at x=2,5,8; their nearest neighbours are A/B patches at x=1,3/4,6/7,9-ish
    # => not clustered as an island => self-retrieval below the island case.
    assert res["heldout_self_retrieval"] < 0.5, res
