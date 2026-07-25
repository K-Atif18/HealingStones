"""Shortcut-detection probes for the Phase 5 diagnostics.

Each probe targets one plausible shortcut from the pre-registration analysis and
is computed *independently* (not stacked). All operate on a generic embedding
source unless noted.

Probes
------
* ``fragment_id_probe``            : kNN accuracy of predicting fragment identity
                                     from embeddings (Shortcut 1: memorization).
* ``density_fingerprint_probe``    : fragment-id accuracy from point
                                     density/spacing alone (Shortcut 5:
                                     acquisition fingerprint).
* ``contact_vs_noncontact_auc``    : how separable contact patches are from
                                     non-contact patches under embedding
                                     distance to contact prototypes
                                     (Shortcut 2: contact detection).
* ``neighbor_contact_vs_noncontact_ranking`` : for a contact query, do the true
                                     neighbour's *contact* patches rank above its
                                     *non-contact* patches? (positive control /
                                     Shortcut 1 discriminator).
* ``distinctiveness_similarity_correlation`` : correlation between embedding
                                     similarity and distinctiveness-score
                                     difference (Shortcut 3: roughness).
* ``boundary_openness_explained_variance``   : how much of embedding structure a
                                     boundary/openness feature explains
                                     (Shortcut 6: boundary topology).
* ``pose_invariance_check``        : (callable-based) apply random SO(3) to a
                                     fragment's patches and confirm embeddings
                                     are unchanged within tolerance
                                     (Shortcut 7 / global-leakage gate).

Kept dependency-light: a small self-contained kNN (leave-one-out) and Pearson
correlation, no sklearn.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from baseline_geometry.descriptors import PatchDescriptors
from phase5_diagnostics.context import DiagnosticContext
from phase5_diagnostics.ranking import build_gallery


__all__ = [
    "fragment_id_probe",
    "density_fingerprint_probe",
    "contact_vs_noncontact_auc",
    "neighbor_contact_vs_noncontact_ranking",
    "distinctiveness_similarity_correlation",
    "boundary_openness_explained_variance",
    "pose_invariance_check",
    "heldout_fragment_id_probe",
]


# ----------------------------------------------------------------------
# kNN helper (leave-one-out, cosine/euclidean on L2-normalised vectors)
# ----------------------------------------------------------------------
def _knn_loo_accuracy(
    X: np.ndarray, y: np.ndarray, k: int = 5, sample: int = 2000, seed: int = 0
) -> float:
    """Leave-one-out kNN classification accuracy on (X, y).

    Subsamples query points for speed but ranks against the full set.
    """
    n = X.shape[0]
    if n <= k + 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    q_idx = (rng.choice(n, size=sample, replace=False) if n > sample else np.arange(n))
    correct = 0
    for qi in q_idx:
        d = np.linalg.norm(X - X[qi], axis=1)
        d[qi] = np.inf
        nn = np.argpartition(d, k)[:k]
        labels, counts = np.unique(y[nn], return_counts=True)
        pred = labels[np.argmax(counts)]
        correct += int(pred == y[qi])
    return correct / len(q_idx)


def _stack_embeddings(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, frag_label_index) stacked over all patches."""
    X, y = [], []
    fid_to_idx = {fid: i for i, fid in enumerate(ctx.fragment_ids)}
    for fid in ctx.fragment_ids:
        pd = embeddings[fid]
        X.append(pd.descriptors)
        y.extend([fid_to_idx[fid]] * pd.patch_count)
    return np.vstack(X), np.asarray(y, dtype=np.int64)


# ----------------------------------------------------------------------
# Shortcut 1: fragment-identity memorization
# ----------------------------------------------------------------------
def fragment_id_probe(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors],
    *, k: int = 5, sample: int = 2000, seed: int = 0,
) -> dict:
    """kNN accuracy of predicting which fragment a patch came from.

    High accuracy means the embedding encodes fragment identity, which -- under
    Caesar's fixed topology -- silently solves neighbour retrieval. Compare to
    chance = 1/#fragments.
    """
    X, y = _stack_embeddings(ctx, embeddings)
    acc = _knn_loo_accuracy(X, y, k=k, sample=sample, seed=seed)
    chance = 1.0 / len(ctx.fragment_ids)
    return {
        "knn_accuracy": acc,
        "chance": chance,
        "ratio_over_chance": (acc / chance) if (acc == acc) else float("nan"),
        "note": "High accuracy => fragment identity is encoded (memorization risk).",
    }


# ----------------------------------------------------------------------
# Condition-6 refinement: HELD-OUT fragment-ID separability
# ----------------------------------------------------------------------
def heldout_fragment_id_probe(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors],
    *, held_out: str, k: int = 5, sample: int = 2000, seed: int = 0,
) -> dict:
    """Can an *unseen* fragment's patches be told apart from the training six?

    This is a **sharper instrument for pre-registration condition 6** (fragment-ID
    shortcut), NOT a new condition. ``fragment_id_probe`` measures within-set
    *literal* memorization (all 7 fragments pooled, leave-one-patch-out): a fold
    model that has seen those patches can ace it by table-lookup. But the danger
    for a LOFO-evaluated encoder is subtler -- a *generalized shape-signature*
    strategy ("cluster by whichever coarse per-fragment geometry this piece
    produces") that transfers to an unseen fragment and is functionally
    fragment-ID by proxy. Literal memorization is impossible for a held-out
    fragment (never trained on), so any separability here is the generalized
    leakage that condition 6 is really about.

    Metric: binary kNN -- for each query patch (subsampled across all patches),
    is its nearest neighbour on the *same side* of the held-out/train split? We
    report accuracy for queries that live ON the held-out fragment specifically
    (``heldout_self_retrieval``): the fraction of held-out patches whose nearest
    cross-patch neighbour is also on the held-out fragment. High => the encoder
    isolates the unseen fragment as its own island (shape-signature leakage);
    near the base rate (held-out size / total) => no such island.

    Interpretation is directional only; it adds scrutiny to condition 6 and
    introduces no new pass/fail threshold.
    """
    if held_out not in ctx.fragment_ids:
        return {"held_out": held_out, "error": "unknown fragment"}

    # Stack all embeddings with a binary label: 1 == held-out fragment.
    X_list, is_held_list, frag_list = [], [], []
    for fid in ctx.fragment_ids:
        pd = embeddings[fid]
        X_list.append(pd.descriptors)
        is_held_list.extend([1 if fid == held_out else 0] * pd.patch_count)
        frag_list.extend([fid] * pd.patch_count)
    X = np.vstack(X_list)
    is_held = np.asarray(is_held_list, dtype=np.int64)

    n_total = X.shape[0]
    n_held = int(is_held.sum())
    base_rate = n_held / n_total if n_total else float("nan")
    if n_held < k + 1:
        return {"held_out": held_out, "n_held": n_held,
                "heldout_self_retrieval": float("nan"), "base_rate": base_rate,
                "note": "too few held-out patches for a stable probe."}

    rng = np.random.default_rng(seed)
    held_rows = np.where(is_held == 1)[0]
    q_idx = (rng.choice(held_rows, size=sample, replace=False)
             if held_rows.size > sample else held_rows)

    # For each held-out query, nearest OTHER patch: is it also held-out?
    correct_same = 0
    knn_same_frac = []
    for qi in q_idx:
        d = np.linalg.norm(X - X[qi], axis=1)
        d[qi] = np.inf
        nn = np.argpartition(d, k)[:k]
        # nearest single neighbour
        nn1 = nn[np.argmin(d[nn])]
        correct_same += int(is_held[nn1] == 1)
        knn_same_frac.append(float(np.mean(is_held[nn] == 1)))

    self_retrieval = correct_same / len(q_idx)
    return {
        "held_out": held_out,
        "n_held": n_held,
        "n_total": n_total,
        "base_rate": base_rate,                      # random expectation
        "heldout_self_retrieval": self_retrieval,     # P(NN of held-out is held-out)
        "heldout_knn_same_fraction": float(np.mean(knn_same_frac)),
        "ratio_over_base_rate": (self_retrieval / base_rate)
                                 if base_rate and base_rate == base_rate else float("nan"),
        "note": ("Sharper condition-6 instrument (NOT a new condition). High "
                 "self-retrieval for an UNSEEN fragment => generalized "
                 "shape-signature leakage, not literal memorization."),
    }


# ----------------------------------------------------------------------
# Shortcut 5: acquisition fingerprint (density/spacing)
# ----------------------------------------------------------------------
def _patch_density_features(ctx: DiagnosticContext) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch [n_points, mean nn-spacing, std nn-spacing] + fragment label."""
    X, y = [], []
    fid_to_idx = {fid: i for i, fid in enumerate(ctx.fragment_ids)}
    for fid in ctx.fragment_ids:
        table = ctx.patch_tables[fid]
        for row in range(table.patch_count):
            start, end = int(table.offsets[row, 0]), int(table.offsets[row, 1])
            pts = table.global_coords[start:end]
            n = pts.shape[0]
            if n >= 4:
                # mean nearest-neighbour spacing within the patch
                d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                nn = d.min(axis=1)
                feats = [n, float(nn.mean()), float(nn.std())]
            else:
                feats = [n, 0.0, 0.0]
            X.append(feats)
            y.append(fid_to_idx[fid])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def density_fingerprint_probe(
    ctx: DiagnosticContext, *, k: int = 5, sample: int = 2000, seed: int = 0
) -> dict:
    """Can fragment identity be predicted from density/spacing alone?

    If yes, a model can recover fragment identity via acquisition artifacts
    rather than shape -- a back-door to Shortcut 1 that survives shape controls.
    """
    X, y = _patch_density_features(ctx)
    # standardise columns
    mean, std = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    std[std < 1e-12] = 1.0
    Xn = (X - mean) / std
    acc = _knn_loo_accuracy(Xn, y, k=k, sample=sample, seed=seed)
    chance = 1.0 / len(ctx.fragment_ids)
    return {
        "knn_accuracy_from_density": acc,
        "chance": chance,
        "ratio_over_chance": (acc / chance) if (acc == acc) else float("nan"),
        "note": "High => per-fragment density/spacing leaks identity; normalise it away.",
    }


# ----------------------------------------------------------------------
# Shortcut 2: contact-region detection
# ----------------------------------------------------------------------
def contact_vs_noncontact_auc(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors],
    *, sample_per_class: int = 3000, seed: int = 0,
) -> dict:
    """How linearly separable are contact vs non-contact patches in embedding space?

    Uses a simple nearest-prototype score: distance to the contact-patch mean
    minus distance to the non-contact-patch mean, then AUC against the true
    contact label. High AUC => the embedding strongly encodes Level-1
    contact-ness (necessary substrate, but not sufficient for assembly).
    """
    from baseline_geometry.retrieval import roc_auc

    rng = np.random.default_rng(seed)
    contact_vecs, noncontact_vecs = [], []
    all_vecs, all_labels = [], []
    for fid in ctx.fragment_ids:
        pd = embeddings[fid]
        cset = ctx.contact_patch_ids.get(fid, set())
        for row, pid in enumerate(pd.patch_ids):
            v = pd.descriptors[row]
            is_c = int(pid) in cset
            (contact_vecs if is_c else noncontact_vecs).append(v)
            all_vecs.append(v)
            all_labels.append(1 if is_c else 0)
    if not contact_vecs or not noncontact_vecs:
        return {"auc": float("nan"), "n_contact": len(contact_vecs),
                "n_noncontact": len(noncontact_vecs)}

    contact_mean = np.mean(contact_vecs, axis=0)
    noncontact_mean = np.mean(noncontact_vecs, axis=0)
    all_vecs = np.asarray(all_vecs)
    all_labels = np.asarray(all_labels)

    # subsample for AUC
    idx = np.arange(all_vecs.shape[0])
    if idx.size > 2 * sample_per_class:
        idx = rng.choice(idx, size=2 * sample_per_class, replace=False)
    score = (np.linalg.norm(all_vecs[idx] - noncontact_mean, axis=1)
             - np.linalg.norm(all_vecs[idx] - contact_mean, axis=1))
    auc = roc_auc(score, all_labels[idx])
    return {
        "auc": auc,
        "n_contact": len(contact_vecs),
        "n_noncontact": len(noncontact_vecs),
        "note": "High AUC => embedding encodes contact-ness (Level 1). Necessary, not sufficient.",
    }


# ----------------------------------------------------------------------
# Positive control / Shortcut 1 discriminator
# ----------------------------------------------------------------------
def neighbor_contact_vs_noncontact_ranking(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors],
    *, query_sample: int = 400, seed: int = 0,
) -> dict:
    """For contact queries, compare mean rank-similarity to the true neighbour's
    contact vs non-contact patches.

    A genuine interface model ranks the neighbour's *break-band* patches high but
    its *sculpted-surface* patches low. A fragment-affinity shortcut ranks the
    whole neighbour high (both similar). Reports the mean similarity gap
    (contact - noncontact); a gap near zero is the fragment-affinity signature.
    """
    rng = np.random.default_rng(seed)
    gallery = build_gallery(embeddings)

    # Collect contact queries.
    queries = []
    for fid in ctx.fragment_ids:
        for pid in ctx.contact_patch_ids.get(fid, set()):
            queries.append((fid, pid))
    if not queries:
        return {"mean_similarity_gap": float("nan"), "n_queries": 0}
    if len(queries) > query_sample:
        sel = rng.choice(len(queries), size=query_sample, replace=False)
        queries = [queries[i] for i in sel]

    gaps = []
    for fid_q, pid_q in queries:
        qrow = gallery.row_of.get((fid_q, pid_q))
        if qrow is None:
            continue
        qvec = gallery.desc[qrow]
        for neigh in ctx.neighbors_of(fid_q):
            ncontact = ctx.contact_patch_ids.get(neigh, set())
            c_sims, nc_sims = [], []
            pd = embeddings[neigh]
            for row, pid in enumerate(pd.patch_ids):
                sim = -float(np.linalg.norm(pd.descriptors[row] - qvec))
                if int(pid) in ncontact:
                    c_sims.append(sim)
                else:
                    nc_sims.append(sim)
            if c_sims and nc_sims:
                gaps.append(np.mean(c_sims) - np.mean(nc_sims))
    return {
        "mean_similarity_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "n_comparisons": len(gaps),
        "note": ("Positive gap => neighbour's contact patches are ranked above its "
                 "non-contact patches (real interface signal). ~0 => fragment affinity."),
    }


# ----------------------------------------------------------------------
# Shortcut 3: roughness / distinctiveness
# ----------------------------------------------------------------------
def distinctiveness_similarity_correlation(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors],
    *, n_pairs: int = 50000, seed: int = 0,
) -> dict:
    """Correlate embedding similarity with distinctiveness-score *difference*.

    Sample random cross-fragment patch pairs; for each compute embedding
    similarity (negative distance) and |distinctiveness_A - distinctiveness_B|.
    A strong negative correlation (similar embeddings <=> similar distinctiveness)
    means similarity is largely explained by "both equally rough" -- Shortcut 3.
    """
    if not ctx.distinctiveness:
        return {"pearson_r": float("nan"), "note": "distinctiveness npz missing"}
    rng = np.random.default_rng(seed)
    fids = ctx.fragment_ids
    sims, ddiffs = [], []
    for _ in range(n_pairs):
        fa, fb = rng.choice(len(fids), size=2, replace=False)
        fa, fb = fids[fa], fids[fb]
        pda, pdb = embeddings[fa], embeddings[fb]
        ra = rng.integers(pda.patch_count)
        rb = rng.integers(pdb.patch_count)
        pia, pib = int(pda.patch_ids[ra]), int(pdb.patch_ids[rb])
        da_ = ctx.distinctiveness.get((fa, pia))
        db_ = ctx.distinctiveness.get((fb, pib))
        if da_ is None or db_ is None:
            continue
        sims.append(-float(np.linalg.norm(pda.descriptors[ra] - pdb.descriptors[rb])))
        ddiffs.append(abs(da_[0] - db_[0]))
    if len(sims) < 10:
        return {"pearson_r": float("nan"), "n": len(sims)}
    sims = np.asarray(sims); ddiffs = np.asarray(ddiffs)
    r = float(np.corrcoef(sims, ddiffs)[0, 1])
    return {
        "pearson_r": r,
        "n": len(sims),
        "note": ("Strong negative r => embedding similarity tracks 'both equally "
                 "distinctive/rough' (Shortcut 3), not assembly compatibility."),
    }


# ----------------------------------------------------------------------
# Shortcut 6: boundary / openness topology
# ----------------------------------------------------------------------
def _boundary_openness_feature(ctx: DiagnosticContext) -> dict[tuple[str, int], float]:
    """Per-patch openness: 1 - (angular coverage of neighbour directions).

    A patch on an open mesh boundary (a break edge) has neighbours only on one
    side => low angular coverage => high openness. Interior patches are
    surrounded => high coverage => low openness. Computed from the patch's own
    local_coords/global_coords directions relative to the centroid.
    """
    feat: dict[tuple[str, int], float] = {}
    for fid in ctx.fragment_ids:
        table = ctx.patch_tables[fid]
        for row in range(table.patch_count):
            start, end = int(table.offsets[row, 0]), int(table.offsets[row, 1])
            pts = table.global_coords[start:end]
            pid = int(table.patch_ids[row])
            if pts.shape[0] < 6:
                feat[(fid, pid)] = 1.0
                continue
            c = pts.mean(axis=0)
            rel = pts - c
            # project onto the 2 principal directions of the patch
            cov = np.cov(rel.T)
            _, vecs = np.linalg.eigh(cov)
            basis = vecs[:, 1:]  # two largest
            proj = rel @ basis
            ang = np.arctan2(proj[:, 1], proj[:, 0])
            hist, _ = np.histogram(ang, bins=12, range=(-np.pi, np.pi))
            coverage = np.count_nonzero(hist) / 12.0
            feat[(fid, pid)] = 1.0 - coverage
    return feat


def boundary_openness_explained_variance(
    ctx: DiagnosticContext, embeddings: dict[str, PatchDescriptors],
    *, n_components: int = 10, seed: int = 0,
) -> dict:
    """R^2 of predicting the openness feature from the top embedding PCs.

    High R^2 => embedding structure is largely explained by boundary/openness
    topology (Shortcut 6): the model may be detecting break *edges*, not fracture
    *geometry*.
    """
    feat_lookup = _boundary_openness_feature(ctx)
    X, target = [], []
    for fid in ctx.fragment_ids:
        pd = embeddings[fid]
        for row, pid in enumerate(pd.patch_ids):
            f = feat_lookup.get((fid, int(pid)))
            if f is None:
                continue
            X.append(pd.descriptors[row])
            target.append(f)
    X = np.asarray(X); target = np.asarray(target)
    if X.shape[0] < n_components + 2:
        return {"r_squared": float("nan")}
    # PCA via SVD
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:n_components].T
    # least-squares regress target on PCs (+ intercept)
    A = np.hstack([pcs, np.ones((pcs.shape[0], 1))])
    coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "r_squared": r2,
        "n_components": n_components,
        "note": "High R^2 => embedding largely explained by boundary/openness topology.",
    }


# ----------------------------------------------------------------------
# Shortcut 7 / global-leakage: pose invariance
# ----------------------------------------------------------------------
def pose_invariance_check(
    encode_patch: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_trials: int = 50,
    n_points: int = 100,
    tol: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Confirm an encoder is invariant to rigid pose (no global-frame leakage).

    ``encode_patch(points, normals) -> vector`` is the *full input pipeline +
    encoder*. For random patches we apply a random SO(3) rotation + translation
    to the points (and rotation to normals) and check the embedding is unchanged
    within ``tol``. This is the mechanical gate that must pass before any learned
    encoder is trusted; it works on any callable, so it applies to the future
    PointNet as a unit test.

    Returns the max/mean embedding deviation and a pass flag.
    """
    rng = np.random.default_rng(seed)

    def _rand_rotation() -> np.ndarray:
        # random rotation via QR of a Gaussian matrix
        A = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    max_dev = 0.0
    devs = []
    for _ in range(n_trials):
        pts = rng.standard_normal((n_points, 3)) * 4.0
        nrm = rng.standard_normal((n_points, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        e0 = np.asarray(encode_patch(pts, nrm), dtype=np.float64)

        Rt = _rand_rotation()
        t = rng.standard_normal(3) * 50.0
        pts2 = pts @ Rt.T + t
        nrm2 = nrm @ Rt.T
        e1 = np.asarray(encode_patch(pts2, nrm2), dtype=np.float64)

        dev = float(np.linalg.norm(e0 - e1))
        devs.append(dev)
        max_dev = max(max_dev, dev)

    return {
        "max_deviation": max_dev,
        "mean_deviation": float(np.mean(devs)) if devs else float("nan"),
        "tolerance": tol,
        "passed": bool(max_dev <= tol),
        "note": "If not passed, the encoder is pose-sensitive => global-frame leakage risk.",
    }
