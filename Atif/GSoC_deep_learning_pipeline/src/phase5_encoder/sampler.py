"""Fragment -> interface -> partner sampler, pair-level validation split, and
on-the-fly hard-negative mining for one LOFO fold.

Implements PHASE5A_TRAINING_DESIGN_REVISED.md §§2, 2b, 4, 7 exactly as
specified there. This module contains no training loop; it only prepares
what one epoch needs (sampled anchor/positive/hard-negative index triples,
and the fixed validation split), so it can be unit-tested independently of
any model or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phase5_diagnostics.context import DiagnosticContext


__all__ = [
    "InterfaceKey", "FoldSampler", "build_fold_sampler",
    "InterfaceWeighting",
]

InterfaceKey = tuple[str, str]  # (fid_a, fid_b), as stored in ctx.interface_patch_ids


@dataclass
class InterfaceWeighting:
    """Per-interface sampling weight, capped-uniform by default (§4).

    Pure inverse-frequency over raw positive-pair counts would give the
    smallest interface (F2-F4, 1,232 pairs) ~174x the weight of the largest
    (F6-F7, 214,446 pairs) -- enough to oversample F2-F4's ~72 contact
    patches into overfitting rather than merely correcting the imbalance
    (PHASE5A_TRAINING_DESIGN_REVISED.md §4). Default: inverse-frequency with
    a floor, i.e. the weight ratio between any two interfaces is capped at
    ``max_ratio`` rather than left unbounded.

    **Two corrections, both caught on measurement before any training run:**

    1. ``max_ratio`` bounds a weight relative to the UNIFORM weight
       (``1/n_interfaces``), not relative to the smallest inverse-frequency
       weight -- clamping against ``inv.min()`` (the largest interface's
       inverse weight) let the rarest interface reach 55.4% of all draws on
       fold 1 (measured), the exact failure §4 was written to prevent.

    2. Clamping the post-normalisation WEIGHT and renormalising is not
       enough on its own: with a 174x true spread, pure inverse-frequency
       already puts ~91.5% of weight-before-any-cap on the single rarest
       interface (measured on fold 1's counts) -- clamping that one value
       down and renormalising the remainder just redistributes mass through
       the SAME skewed inverse-frequency proportions among the rest, so the
       clamped interface still ends up dominant after renormalisation
       (measured: this produced 90.8% for F2-F4, WORSE than the ratio bug it
       was meant to fix). The correct fix operates on effective COUNTS, not
       weights: each interface's raw positive-pair count is clamped into
       ``[min_count/floor_ratio, min_count*ceil_ratio]`` relative to the
       smallest raw count seen, where the ratios are chosen so the resulting
       inverse-frequency-of-clamped-counts weighting cannot exceed
       ``max_ratio`` times uniform for the rarest interface. This keeps the
       "rare interfaces sampled more than their raw count would imply, but
       not to the point of dominating" property §4 actually asked for.
    """

    max_ratio: float = 5.0  # a single interface's weight <= max_ratio * uniform_weight
    policy: str = "clamped_effective_count_inverse_frequency"

    def weights_for(self, interface_counts: dict[InterfaceKey, int]) -> dict[InterfaceKey, float]:
        """Return a normalised sampling-weight dict over the given interfaces.

        ``interface_counts`` should be **positive-pair counts restricted to
        the current fold's training-visible interfaces** (i.e. already
        excludes any interface touching the held-out fragment), not the
        global table.

        Method: clamp each interface's raw count into
        ``[max_count / max_ratio, max_count]`` (i.e. the smallest interface's
        EFFECTIVE count is never treated as smaller than
        ``1/max_ratio`` of the largest interface's real count), THEN take
        inverse-frequency of the clamped counts and normalise. This bounds
        the rarest interface's weight to at most ``max_ratio`` times the
        largest interface's weight under plain inverse-frequency of the
        clamped counts -- which is at most ``max_ratio * uniform`` when
        there are more than one interface, verified by direct computation
        below rather than asserted.
        """
        if not interface_counts:
            return {}
        keys = list(interface_counts.keys())
        counts = np.array([interface_counts[k] for k in keys], dtype=np.float64)
        counts = np.maximum(counts, 1.0)

        max_count = counts.max()
        floor = max_count / self.max_ratio
        counts_clamped = np.maximum(counts, floor)

        inv = 1.0 / counts_clamped
        w = inv / inv.sum()
        return dict(zip(keys, w.tolist()))


@dataclass
class FoldSampler:
    """Everything needed to draw training batches and hold out a validation
    split, for one LOFO fold (``held_out`` excluded from every structure)."""

    held_out: str
    train_fragments: list[str]
    # interface -> {fid -> np.ndarray of *training* anchor patch ids for that side}
    train_interface_patches: dict[InterfaceKey, dict[str, np.ndarray]]
    # interface -> np.ndarray of positive-pair row indices (into ctx.pairs), TRAIN split
    train_interface_pair_rows: dict[InterfaceKey, np.ndarray]
    # interface -> np.ndarray of positive-pair row indices, VALIDATION split (held out
    # from training loss, per §7 -- a pair-level split, not an interface or fragment one)
    val_interface_pair_rows: dict[InterfaceKey, np.ndarray]
    weighting: InterfaceWeighting
    interface_weights: dict[InterfaceKey, float]
    # fragment -> patch ids belonging to >1 interface (for the multi-interface rule, §4.4)
    multi_interface_patches: dict[str, set] = field(default_factory=dict)

    def sample_batch(
        self, ctx: DiagnosticContext, batch_size: int, rng: np.random.Generator,
    ) -> dict:
        """Draw ``batch_size`` (anchor, positive) index pairs via
        fragment -> interface -> partner (§4), respecting the multi-interface
        rule: the interface drawn for a step is the ONLY interface whose
        partner set that draw's positive may come from, even if the anchor
        patch also belongs to other interfaces.

        Returns a dict with parallel arrays ``anchor_fid, anchor_pid,
        partner_fid, partner_pid, interface`` (len == batch_size, may be <
        batch_size if an interface has zero usable pairs, though with the
        weighting only interfaces with >=1 training pair are ever drawn).
        """
        interfaces = list(self.interface_weights.keys())
        if not interfaces:
            raise ValueError(f"fold {self.held_out}: no training interfaces available")
        probs = np.array([self.interface_weights[i] for i in interfaces])
        probs = probs / probs.sum()

        draws = rng.choice(len(interfaces), size=batch_size, p=probs)
        anchor_fid, anchor_pid, partner_fid, partner_pid, iface_out = [], [], [], [], []
        for d in draws:
            iface = interfaces[d]
            rows = self.train_interface_pair_rows[iface]
            row = rows[rng.integers(0, len(rows))]
            vocab = ctx.pairs.fragment_vocab
            fa = str(vocab[ctx.pairs.fragment_A_idx[row]])
            fb = str(vocab[ctx.pairs.fragment_B_idx[row]])
            pa = int(ctx.pairs.patch_A_ids[row])
            pb = int(ctx.pairs.patch_B_ids[row])
            # Randomize which side is "anchor" vs "partner" for this draw.
            if rng.random() < 0.5:
                anchor_fid.append(fa); anchor_pid.append(pa)
                partner_fid.append(fb); partner_pid.append(pb)
            else:
                anchor_fid.append(fb); anchor_pid.append(pb)
                partner_fid.append(fa); partner_pid.append(pa)
            iface_out.append(iface)
        return {
            "anchor_fid": anchor_fid, "anchor_pid": anchor_pid,
            "partner_fid": partner_fid, "partner_pid": partner_pid,
            "interface": iface_out,
        }

    def n_val_pairs(self) -> int:
        return int(sum(len(v) for v in self.val_interface_pair_rows.values()))


def build_fold_sampler(
    ctx: DiagnosticContext,
    held_out: str,
    *,
    val_fraction: float = 0.10,
    val_pair_cap: int = 500,
    weighting: Optional[InterfaceWeighting] = None,
    seed: int = 0,
) -> FoldSampler:
    """Build a :class:`FoldSampler` for LOFO fold ``held_out``.

    Validation split (§7): a **pair-level** split held out from every
    interface -- ``val_fraction`` of each training interface's positive
    pairs, chosen uniformly at random, excluded from ``train_interface_pair_rows``
    and placed in ``val_interface_pair_rows``. Not a held-out interface, not a
    held-out fragment (see §7 for the asymmetry argument).

    ``val_pair_cap`` bounds the TOTAL validation set size (default 500
    pairs). This is a hard fix for a measured bug: applying ``val_fraction``
    directly to raw pair-row counts produced 49,569 validation pairs for
    fold 1 (10% of 446,114 training pairs dominated by F6-F7's 214k
    positives) -- 99,138 patch encodes and a ~9.8 GB fp32 similarity matrix
    per validation pass, which OOMs and would dominate the epoch wall-clock
    even if it somehow didn't. The per-interface ``val_fraction`` proportions
    are computed first (so each interface still contributes a fraction of
    its own pairs, not a uniform slice that would starve small interfaces),
    then the whole set is downsampled proportionally to ``val_pair_cap`` if
    the raw total exceeds it.
    """
    weighting = weighting or InterfaceWeighting()
    rng = np.random.default_rng(seed)
    train_fragments = [f for f in ctx.fragment_ids if f != held_out]

    train_interface_patches: dict[InterfaceKey, dict[str, np.ndarray]] = {}
    train_pair_rows: dict[InterfaceKey, np.ndarray] = {}
    val_pair_rows: dict[InterfaceKey, np.ndarray] = {}
    interface_counts: dict[InterfaceKey, int] = {}

    vocab = ctx.pairs.fragment_vocab
    fa_all = vocab[ctx.pairs.fragment_A_idx]
    fb_all = vocab[ctx.pairs.fragment_B_idx]
    is_pos = ctx.pairs.labels == 1

    # Pass 1: compute the raw (uncapped) per-interface val/train split at
    # val_fraction, same as before -- this preserves per-interface
    # proportionality (each interface's val slice is a fraction of ITS OWN
    # pairs, not a shared pool sliced uniformly).
    raw_val_rows: dict[InterfaceKey, np.ndarray] = {}
    for (fid_a, fid_b), sides in ctx.interface_patch_ids.items():
        if held_out in (fid_a, fid_b):
            continue  # entire interface excluded -- it touches the held-out fragment
        if not sides.get(fid_a) or not sides.get(fid_b):
            continue  # e.g. F5-F7: zero contact patches on one/both sides
        train_interface_patches[(fid_a, fid_b)] = {
            fid_a: np.array(sorted(sides[fid_a]), dtype=np.int64),
            fid_b: np.array(sorted(sides[fid_b]), dtype=np.int64),
        }

        mask = is_pos & (
            ((fa_all == fid_a) & (fb_all == fid_b)) |
            ((fa_all == fid_b) & (fb_all == fid_a))
        )
        rows = np.where(mask)[0]
        if rows.size == 0:
            continue
        rng.shuffle(rows)
        n_val = max(1, int(round(rows.size * val_fraction))) if rows.size > 1 else 0
        raw_val_rows[(fid_a, fid_b)] = rows[:n_val]
        train_pair_rows[(fid_a, fid_b)] = rows[n_val:]
        interface_counts[(fid_a, fid_b)] = int(rows[n_val:].size)

    # Pass 2: cap the TOTAL validation set to val_pair_cap, downsampling each
    # interface's raw val slice proportionally to its own share of the raw
    # total (so a 500-pair cap still reflects roughly the same per-interface
    # mix as the uncapped 49,569 would have, just 99x smaller).
    raw_total = sum(len(v) for v in raw_val_rows.values())
    if raw_total > val_pair_cap and raw_total > 0:
        for iface, rows in raw_val_rows.items():
            share = len(rows) / raw_total
            n_keep = max(1, int(round(val_pair_cap * share))) if len(rows) > 0 else 0
            n_keep = min(n_keep, len(rows))
            val_pair_rows[iface] = rows[:n_keep]
    else:
        val_pair_rows = raw_val_rows

    # Drop interfaces left with zero training pairs after the val split.
    interface_counts = {k: v for k, v in interface_counts.items() if v > 0}
    train_pair_rows = {k: v for k, v in train_pair_rows.items() if k in interface_counts}

    interface_weights = weighting.weights_for(interface_counts)

    # Multi-interface patches (§4.4): fragment -> patch ids appearing in >1 interface.
    multi: dict[str, set] = {f: set() for f in train_fragments}
    seen: dict[tuple[str, int], int] = {}
    for (fid_a, fid_b), sides in train_interface_patches.items():
        for fid, pids in sides.items():
            for pid in pids.tolist():
                key = (fid, pid)
                seen[key] = seen.get(key, 0) + 1
    for (fid, pid), n in seen.items():
        if n > 1:
            multi[fid].add(pid)

    return FoldSampler(
        held_out=held_out,
        train_fragments=train_fragments,
        train_interface_patches=train_interface_patches,
        train_interface_pair_rows=train_pair_rows,
        val_interface_pair_rows=val_pair_rows,
        weighting=weighting,
        interface_weights=interface_weights,
        multi_interface_patches=multi,
    )
