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
    """

    max_ratio: float = 20.0  # cap on (max_weight / min_weight) across interfaces
    policy: str = "capped_inverse_frequency"

    def weights_for(self, interface_counts: dict[InterfaceKey, int]) -> dict[InterfaceKey, float]:
        """Return a normalised sampling-weight dict over the given interfaces.

        ``interface_counts`` should be **positive-pair counts restricted to
        the current fold's training-visible interfaces** (i.e. already
        excludes any interface touching the held-out fragment), not the
        global table.
        """
        if not interface_counts:
            return {}
        counts = np.array(list(interface_counts.values()), dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        inv_capped = np.minimum(inv, inv.min() * self.max_ratio)
        w = inv_capped / inv_capped.sum()
        return dict(zip(interface_counts.keys(), w.tolist()))


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
    weighting: Optional[InterfaceWeighting] = None,
    seed: int = 0,
) -> FoldSampler:
    """Build a :class:`FoldSampler` for LOFO fold ``held_out``.

    Validation split (§7): a **pair-level** split held out from every
    interface -- ``val_fraction`` of each training interface's positive
    pairs, chosen uniformly at random, excluded from ``train_interface_pair_rows``
    and placed in ``val_interface_pair_rows``. Not a held-out interface, not a
    held-out fragment (see §7 for the asymmetry argument).
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
        val_pair_rows[(fid_a, fid_b)] = rows[:n_val]
        train_pair_rows[(fid_a, fid_b)] = rows[n_val:]
        interface_counts[(fid_a, fid_b)] = int(rows[n_val:].size)

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
