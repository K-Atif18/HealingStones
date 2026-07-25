"""Orchestrator for the Phase 5 pre-registration diagnostics.

``run_diagnostics`` runs the full battery on a set of named embedding sources and
returns one JSON-serialisable report. It is written so the identical call runs on
the reference sources now (random, centroid, FPFH) and on any future learned
checkpoint later -- just add the checkpoint to ``sources``.

The report is organised to mirror the pre-registration items:
  1. null_calibration        (random + centroid floor)
  2. shortcut_battery        (per source)
  3. hard_negative_strata    (per source; mined with FPFH)
  4. lofo_per_fold           (per source)
  + main_ranking             (raw + contact-only gallery, per source)
  + interface_breakdown      (macro vs micro, per source)

Thresholds live in ``PHASE5_PREREGISTRATION.md`` (not in code) and are applied
by ``apply_prereg_verdicts`` for convenience, but the raw numbers are always
reported so the verdict logic is auditable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from baseline_geometry.descriptors import PatchDescriptors
from phase5_diagnostics import ranking as rk
from phase5_diagnostics import probes as pb
from phase5_diagnostics.context import DiagnosticContext


__all__ = ["run_diagnostics", "write_report"]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set,)):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


def write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def _contact_only_keep(ctx: DiagnosticContext):
    return lambda fid, pid: ctx.is_contact_patch(fid, pid)


def run_diagnostics(
    ctx: DiagnosticContext,
    sources: dict[str, dict[str, PatchDescriptors]],
    *,
    fpfh_source_name: str = "fpfh",
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    query_sample: int = 800,
    seed: int = 0,
    low_confidence_fragments: tuple[str, ...] = (),
) -> dict:
    """Run the full diagnostic battery on all sources.

    ``sources`` maps a name -> embeddings dict. Hard negatives are mined once
    using the FPFH source (if present), then reused to score every source, so
    the hard set is fixed across sources for a fair comparison.
    """
    report: dict[str, Any] = {
        "phase": "Phase 5 pre-registration diagnostics",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "fragment_ids": ctx.fragment_ids,
        "n_total_patches": ctx.total_patches(),
        "n_adjacent_pairs": len(ctx.adjacent_pairs),
        "sources": list(sources.keys()),
    }

    # --- Mine hard negatives once, using FPFH (fixed set for all sources) ---
    hard_neg = []
    if fpfh_source_name in sources:
        hard_neg = rk.mine_hard_negatives(ctx, sources[fpfh_source_name], seed=seed)
    report["hard_negatives_mined"] = len(hard_neg)

    contact_keep = _contact_only_keep(ctx)

    # --- 1. Null calibration (floor) ---
    null_sources = {n: sources[n] for n in ("random", "centroid") if n in sources}
    report["null_calibration"] = rk.calibrate_null(
        ctx, null_sources, k_values=k_values, query_sample=query_sample, seed=seed
    )

    # --- Per-source diagnostics ---
    per_source: dict[str, Any] = {}
    for name, emb in sources.items():
        entry: dict[str, Any] = {}

        # Main ranking: raw gallery + contact-only gallery.
        entry["ranking_raw_gallery"] = rk.evaluate_ranking(
            ctx, emb, k_values=k_values, query_sample=query_sample, seed=seed
        )
        entry["ranking_contact_only_gallery"] = rk.evaluate_ranking(
            ctx, emb, k_values=k_values, query_sample=query_sample, seed=seed,
            gallery_keep=contact_keep,
        )

        # Interface breakdown (macro vs micro).
        entry["interface_breakdown"] = rk.retrieval_by_interface(
            ctx, emb, k_values=k_values, seed=seed
        )

        # Shortcut battery.
        entry["shortcut_battery"] = {
            "fragment_id_probe": pb.fragment_id_probe(ctx, emb, seed=seed),
            "contact_vs_noncontact_auc": pb.contact_vs_noncontact_auc(ctx, emb, seed=seed),
            "neighbor_contact_vs_noncontact_ranking":
                pb.neighbor_contact_vs_noncontact_ranking(ctx, emb, seed=seed),
            "distinctiveness_similarity_correlation":
                pb.distinctiveness_similarity_correlation(ctx, emb, seed=seed),
            "boundary_openness_explained_variance":
                pb.boundary_openness_explained_variance(ctx, emb, seed=seed),
        }

        # Condition-6 refinement: held-out fragment-ID separability, per fragment.
        # For a fixed source (FPFH/random/centroid) this is a reference baseline
        # ("what does an unseen fragment's identity separability look like for a
        # non-learned descriptor"); for a LOFO-trained encoder the held-out
        # fragment's embeddings come from the model that never trained on it,
        # which is the version that actually tests generalized shape-signature
        # leakage. Sharper instrument for condition 6, NOT a new condition.
        entry["heldout_fragment_id_probe"] = {
            held: pb.heldout_fragment_id_probe(ctx, emb, held_out=held, seed=seed)
            for held in ctx.fragment_ids
        }

        # Hard-negative stratification (unrestricted -- kept for backward
        # comparability with prior reports; NOT the held-out-safe number).
        entry["hard_negative_strata"] = rk.easy_vs_hard_separability(
            ctx, emb, hard_neg, seed=seed
        )

        # LOFO per fold. hard_neg is passed through so each fold also gets a
        # held_out-restricted hard_negative_strata entry (condition 3, fixed
        # per PHASE5A_TRAINING_DESIGN_REVISED.md item 1) -- this is the number
        # that should actually be compared against the pre-registered
        # condition-3 threshold, not the unrestricted entry above.
        entry["lofo_per_fold"] = rk.lofo_per_fold(
            ctx, emb, k_values=k_values, seed=seed,
            low_confidence_fragments=low_confidence_fragments,
            hard_negatives=hard_neg,
        )

        per_source[name] = entry

    report["per_source"] = per_source

    # --- Acquisition fingerprint is source-independent (raw geometry) ---
    report["density_fingerprint_probe"] = pb.density_fingerprint_probe(ctx, seed=seed)

    return report
