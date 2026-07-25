"""Phase 5A training loop: one LOFO fold, per PHASE5A_TRAINING_DESIGN_REVISED.md.

This module is imported by ``scripts/train_phase5a.py``. It is written and
tested but NOT executed by this session -- per the human's instruction, the
human runs training. See the script's docstring for the exact command,
expected fold-1 wall-clock, output locations, and how to read
patience-vs-cap from the log.

Design points this loop implements (cross-referenced to the design doc):
  * Loss: InfoNCE-style, retained from the accepted original design (points
    1/3/4), operating on (anchor, positive, deduped hard negatives, in-batch
    randoms).
  * Sampler: fragment->interface->partner, capped-inverse-frequency
    weighting (design §4) via ``phase5_encoder.sampler``.
  * Hard negatives: on-the-fly, per-epoch, fold-restricted mining (design
    §2/§2b) via ``phase5_encoder.mining``, deduplicated within each batch.
  * Batch size: B_a=512 (design §3, measured 1436.4 MB peak / ~96 ms/step).
  * Norm layer: LayerNorm (design §6, already landed in ``model.py``).
  * Positive-distance policy: uncapped -- all positives used (design §5).
  * Validation / stopping: pair-level split held out from every interface
    (design §7), patience=5 / cap=50 epochs (retained from accepted design),
    with an explicit patience-vs-cap flag in the returned history so the
    caller never has to guess which one stopped the run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from phase5_diagnostics.context import DiagnosticContext
from phase5_encoder.model import PointNetEncoder, build_encode_patch
from phase5_encoder.sampler import build_fold_sampler, FoldSampler, InterfaceWeighting
from phase5_encoder.mining import mine_epoch_hard_negatives
from phase5_encoder.preprocess import prepare_patch
from phase5_encoder.features import knn_ppf_features


__all__ = ["TrainConfig", "train_one_fold"]


@dataclass
class TrainConfig:
    n_points: int = 64
    k: int = 16
    out_dim: int = 64
    batch_size: int = 512          # design §3, measured-safe operating point
    temperature: float = 0.07
    lr: float = 1e-3
    epoch_cap: int = 50            # retained from accepted original design
    patience: int = 5              # retained from accepted original design
    val_fraction: float = 0.10     # design §7 (per-interface proportion, pre-cap)
    val_pair_cap: int = 500        # HARD cap on total validation pairs -- fixes the
                                    # measured 49,569-pair OOM bug (see build_fold_sampler)
    val_minibatch_size: int = 256  # validation is chunked, never one shot (same bug)
    positive_max_center_dist_mm: float | None = None  # Deviation 3 (§11): distance-capped
                                    # positives; None = uncapped/all-positives (original)
    steps_per_epoch: int = 12      # ~ LOFO fold size / batch_size, per PHASE5_RESULTS_LOG.md:160
    interface_weight_max_ratio: float = 5.0  # design §4, corrected semantics -- see
                                              # sampler.InterfaceWeighting docstring
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    out_dir: str = "phase5a_runs"


def _encode_patch_ids(model, ctx: DiagnosticContext, fid: str, pids, cfg: TrainConfig,
                       training: bool) -> torch.Tensor:
    pt = ctx.patch_tables[fid]
    feats = []
    for pid in pids:
        row = pt.row_for_patch_id(int(pid))
        p, n = prepare_patch(pt.patch_points(row), pt.patch_normals(row),
                              cfg.n_points, training=training, seed=cfg.seed)
        feats.append(knn_ppf_features(p, n, k=cfg.k))
    t = torch.from_numpy(np.stack(feats)).float().to(cfg.device)
    return model(t)


def _build_positive_partner_lookup(ctx: DiagnosticContext) -> dict[tuple, set]:
    """(fid, pid) -> set of (fid, pid) that are its Phase-3 positive partners,
    built once per fold (not per step) directly from ``ctx.pairs`` -- a
    lighter-weight equivalent of ``phase5_diagnostics.ranking.
    positive_partner_map`` that doesn't require building a ``Gallery`` first
    (the training loop doesn't have a full-dataset embedding gallery on
    hand every step; it only needs pairwise partner lookups for masking).

    Used to fix a measured bug (Deviation 2): with in-batch cross_entropy,
    two draws from the SAME interface can be genuine positive partners of
    each other (median contact patch has 347 positives), so an unmasked
    denominator penalises true positives as negatives -- common, not rare,
    at the draw rates produced by interface weighting.
    """
    pairs = ctx.pairs
    vocab = pairs.fragment_vocab
    pos_idx = np.where(pairs.labels == 1)[0]
    out: dict[tuple, set] = {}
    for p in pos_idx.tolist():
        fa = str(vocab[pairs.fragment_A_idx[p]])
        fb = str(vocab[pairs.fragment_B_idx[p]])
        pa = int(pairs.patch_A_ids[p])
        pb = int(pairs.patch_B_ids[p])
        out.setdefault((fa, pa), set()).add((fb, pb))
        out.setdefault((fb, pb), set()).add((fa, pa))
    return out


def _info_nce_step(
    model, ctx, batch, mined, cfg: TrainConfig,
    positive_lookup: dict[tuple, set],
) -> tuple[torch.Tensor, int]:
    """One training step's loss. Hard negatives are deduplicated within the
    batch (design §3: 'dedup is mandatory, not a fallback') -- each DISTINCT
    (fid,pid) mined negative across the whole batch is encoded once.

    Same-interface in-batch positives are masked out of the InfoNCE
    denominator (Deviation 2 fix): a candidate column is excluded from row
    i's denominator if it is anchor i's designated positive (the target
    itself, kept) OR any candidate that is ALSO a true positive partner of
    anchor i per ``positive_lookup`` -- otherwise cross_entropy penalises a
    real positive as if it were a negative whenever two draws from the same
    (or another) interface happen to be partners, which is common (not
    rare) given a median of 347 positives per contact patch.

    Returns ``(loss, n_deduped_hard_negatives)`` so the caller can log the
    measured deduped count per step (Deviation 2's other requirement).
    """
    anchors = list(zip(batch["anchor_fid"], batch["anchor_pid"]))
    positives = list(zip(batch["partner_fid"], batch["partner_pid"]))

    def encode_list(pairs):
        by_frag: dict[str, list[int]] = {}
        order = []
        for fid, pid in pairs:
            by_frag.setdefault(fid, []).append(pid)
            order.append((fid, len(by_frag[fid]) - 1))
        embs_by_frag = {
            fid: _encode_patch_ids(model, ctx, fid, pids, cfg, training=True)
            for fid, pids in by_frag.items()
        }
        return torch.stack([embs_by_frag[fid][i] for fid, i in order])

    anc_emb = encode_list(anchors)
    pos_emb = encode_list(positives)

    # Deduplicate mined hard negatives across the batch.
    hard_keys = []
    seen = set()
    for fid, pid in anchors:
        neg = mined.lookup.get((fid, int(pid)))
        if neg is None:
            continue
        key = (neg[0], neg[1])
        if key not in seen:
            seen.add(key)
            hard_keys.append(key)
    n_deduped_hard = len(hard_keys)
    hard_emb = encode_list(hard_keys) if hard_keys else torch.empty(
        0, anc_emb.shape[1], device=cfg.device
    )

    candidates = torch.cat([pos_emb, hard_emb], dim=0)
    candidate_keys = positives + hard_keys  # parallel to `candidates` rows
    sim = anc_emb @ candidates.t() / cfg.temperature  # (B_a, B_a + n_deduped_hard)
    targets = torch.arange(anc_emb.shape[0], device=cfg.device)

    # Build the same-interface / cross-interface true-positive mask: for row
    # i (anchor_i), mask out every column j != target_i whose candidate is
    # ALSO a genuine positive partner of anchor_i.
    n_anc = len(anchors)
    n_cand = len(candidate_keys)
    mask = torch.zeros(n_anc, n_cand, dtype=torch.bool, device=cfg.device)
    for i, anchor_key in enumerate(anchors):
        true_partners = positive_lookup.get(anchor_key, set())
        if not true_partners:
            continue
        for j, cand_key in enumerate(candidate_keys):
            if j == i:
                continue  # never mask the actual target column
            if cand_key in true_partners:
                mask[i, j] = True
    sim = sim.masked_fill(mask, float("-inf"))

    loss = torch.nn.functional.cross_entropy(sim, targets)
    return loss, n_deduped_hard


def _validation_loss(model, ctx, sampler: FoldSampler, cfg: TrainConfig) -> float:
    """Pair-level validation loss (design §7): no mining, no hard negatives --
    just anchor/positive separation among in-batch randoms, on the FIXED
    held-out-from-every-interface pair set. Not part of the pre-registered
    pass criteria; used only to decide when to stop (or log that the cap did).

    Evaluated in minibatches of ``cfg.val_minibatch_size`` (default 256),
    never as one shot. This fixes a measured bug: an earlier version built
    ``all_rows`` from every validation pair and encoded them all in one
    matrix -- on fold 1 that was 49,569 pairs (99,138 patch encodes, a
    ~9.8 GB fp32 similarity matrix) because ``val_fraction`` was applied to
    raw pair-row counts dominated by F6-F7's 214k positives, not to a
    capped patch/pair budget. The sampler now caps the total validation set
    at ``val_pair_cap`` (default 500, see ``build_fold_sampler``), and this
    function additionally chunks that capped set into minibatches so no
    single validation step's similarity matrix or encode-call count scales
    with the (still potentially several-hundred-pair) total.
    """
    rows_by_iface = sampler.val_interface_pair_rows
    all_rows = [(iface, r) for iface, rs in rows_by_iface.items() for r in rs.tolist()]
    if not all_rows:
        return float("nan")
    model.eval()
    vocab = ctx.pairs.fragment_vocab

    def encode_list(pairs):
        by_frag: dict[str, list[int]] = {}
        order = []
        for fid, pid in pairs:
            by_frag.setdefault(fid, []).append(pid)
            order.append((fid, len(by_frag[fid]) - 1))
        embs_by_frag = {
            fid: _encode_patch_ids(model, ctx, fid, pids, cfg, training=False)
            for fid, pids in by_frag.items()
        }
        return torch.stack([embs_by_frag[fid][i] for fid, i in order])

    losses = []
    mb = max(1, cfg.val_minibatch_size)
    with torch.no_grad():
        for start in range(0, len(all_rows), mb):
            chunk = all_rows[start:start + mb]
            anchors, positives = [], []
            for _, row in chunk:
                fa = str(vocab[ctx.pairs.fragment_A_idx[row]])
                fb = str(vocab[ctx.pairs.fragment_B_idx[row]])
                pa = int(ctx.pairs.patch_A_ids[row])
                pb = int(ctx.pairs.patch_B_ids[row])
                anchors.append((fa, pa))
                positives.append((fb, pb))
            anc_emb = encode_list(anchors)
            pos_emb = encode_list(positives)
            sim = anc_emb @ pos_emb.t() / cfg.temperature
            targets = torch.arange(anc_emb.shape[0], device=cfg.device)
            loss = torch.nn.functional.cross_entropy(sim, targets)
            losses.append(float(loss.item()) * len(chunk))
    model.train()
    return float(sum(losses) / len(all_rows))


def train_one_fold(
    ctx: DiagnosticContext, held_out: str, cfg: TrainConfig,
) -> dict:
    """Train one LOFO fold. Returns a history dict (also written to disk as
    ``<out_dir>/fold_<held_out>_history.json``) with per-epoch train/val loss,
    per-interface draw counts, per-fragment mining distances, and an explicit
    ``stopped_by`` field ('patience' | 'epoch_cap') -- never a bare
    'early_stopped' claim (design §7 reporting requirement)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = PointNetEncoder(out_dim=cfg.out_dim, pair_input=True).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sampler = build_fold_sampler(
        ctx, held_out, val_fraction=cfg.val_fraction,
        val_pair_cap=cfg.val_pair_cap,
        weighting=InterfaceWeighting(max_ratio=cfg.interface_weight_max_ratio),
        positive_max_center_dist_mm=cfg.positive_max_center_dist_mm,
        seed=cfg.seed,
    )

    # Training anchors for mining = each training fragment's contact patches
    # (only contact patches can have a positive partner at all).
    train_anchor_ids = {
        fid: np.array(sorted(ctx.contact_patch_ids.get(fid, set())), dtype=np.int64)
        for fid in sampler.train_fragments
    }

    # Built once per fold (not per step): (fid,pid) -> set of true positive
    # partners, used to mask same-/cross-interface in-batch positives out of
    # the InfoNCE denominator (Deviation 2 fix).
    positive_lookup = _build_positive_partner_lookup(ctx)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {
        "held_out": held_out,
        "epochs": [],
        "stopped_by": None,
        "stopped_at_epoch": None,
        # Sampler config surfaced once so per-epoch interface_draw_counts can be
        # checked against the WEIGHTS that produced them: if observed draw
        # fractions diverge from these weights, that is a sampler bug, not a
        # tuning question (per the reviewer's fold-1 checklist).
        "interface_weights": {f"{k[0]}-{k[1]}": v
                              for k, v in sampler.interface_weights.items()},
        "n_train_interfaces": len(sampler.interface_weights),
        "n_val_pairs": sampler.n_val_pairs(),
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "steps_per_epoch": cfg.steps_per_epoch,
    }

    for epoch in range(cfg.epoch_cap):
        if cfg.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        mined = mine_epoch_hard_negatives(
            ctx, model, held_out, train_anchor_ids,
            n_points=cfg.n_points, k=cfg.k, device=cfg.device, seed=cfg.seed + epoch,
        )
        mine_dt = time.time() - t0

        interface_draw_counts: dict[str, int] = {}
        step_losses = []
        deduped_hard_counts = []
        t1 = time.time()
        for _ in range(cfg.steps_per_epoch):
            batch = sampler.sample_batch(ctx, cfg.batch_size, rng)
            for iface in batch["interface"]:
                key = f"{iface[0]}-{iface[1]}"
                interface_draw_counts[key] = interface_draw_counts.get(key, 0) + 1
            optim.zero_grad()
            loss, n_deduped_hard = _info_nce_step(
                model, ctx, batch, mined, cfg, positive_lookup
            )
            loss.backward()
            optim.step()
            step_losses.append(float(loss.item()))
            deduped_hard_counts.append(n_deduped_hard)
        train_dt = time.time() - t1

        # Validation timed SEPARATELY -- this is the term that was omitted from
        # the original (wrong) wall-clock estimate; report it on its own.
        t2 = time.time()
        val_loss = _validation_loss(model, ctx, sampler, cfg)
        val_dt = time.time() - t2

        peak_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if cfg.device == "cuda" else float("nan"))

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Observed per-interface draw FRACTIONS, for direct comparison against
        # history["interface_weights"] above.
        total_draws = sum(interface_draw_counts.values()) or 1
        interface_draw_fractions = {
            k: v / total_draws for k, v in interface_draw_counts.items()
        }

        history["epochs"].append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(step_losses)),
            # Full per-step trajectory so "did the loss move at all in one
            # epoch" is answerable (first vs last step), not just the mean.
            "train_loss_per_step": [float(x) for x in step_losses],
            "train_loss_first_step": float(step_losses[0]),
            "train_loss_last_step": float(step_losses[-1]),
            "val_loss": val_loss,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "mining_seconds": mine_dt,
            "training_seconds": train_dt,
            "validation_seconds": val_dt,
            "epoch_seconds_total": mine_dt + train_dt + val_dt,
            "peak_gpu_mb": peak_mb,
            "interface_draw_counts": interface_draw_counts,
            "interface_draw_fractions": interface_draw_fractions,
            "deduped_hard_negatives_per_step_mean": float(np.mean(deduped_hard_counts)),
            "deduped_hard_negatives_per_step_min": int(np.min(deduped_hard_counts)),
            "deduped_hard_negatives_per_step_max": int(np.max(deduped_hard_counts)),
            "per_fragment_mined_distance_mean": {
                fid: float(np.mean(ds)) for fid, ds in mined.per_fragment_distance.items()
            },
        })

        if epochs_without_improvement >= cfg.patience:
            history["stopped_by"] = "patience"
            history["stopped_at_epoch"] = epoch
            break
    else:
        history["stopped_by"] = "epoch_cap"
        history["stopped_at_epoch"] = cfg.epoch_cap - 1

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(cfg.out_dir, exist_ok=True)
    hist_path = os.path.join(cfg.out_dir, f"fold_{held_out}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    ckpt_path = os.path.join(cfg.out_dir, f"fold_{held_out}_best.pt")
    torch.save(model.state_dict(), ckpt_path)

    history["history_path"] = hist_path
    history["checkpoint_path"] = ckpt_path
    return history
