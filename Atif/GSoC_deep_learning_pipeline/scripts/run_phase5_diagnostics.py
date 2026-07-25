#!/usr/bin/env python3
"""CLI: run the Phase 5 pre-registration diagnostics on the reference sources.

Runs the full battery on:
  * random   -- null floor (no information)
  * centroid -- weak trivial-feature baseline
  * fpfh     -- the Phase 4 handcrafted descriptors (the bar to beat)

and writes a single report to phase5_diagnostics_results/baseline_diagnostics.json.

This is instrumentation, not training. Review the FPFH row against the
pre-registration thresholds in PHASE5_PREREGISTRATION.md before building the
Phase 5A encoder.

Run:
    PYTHONPATH=src python3 scripts/run_phase5_diagnostics.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from phase5_diagnostics.context import load_context
from phase5_diagnostics import embeddings as emb_mod
from phase5_diagnostics import runner


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 pre-registration diagnostics.")
    ap.add_argument("--dataset-dir", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--patches-dir", default=os.path.join(ROOT, "patches"))
    ap.add_argument("--pairs-npz", default=os.path.join(ROOT, "phase3_final_review", "pairs.npz"))
    ap.add_argument("--pairs-dataset-json", default=os.path.join(ROOT, "pairs", "dataset.json"))
    ap.add_argument("--descriptors-root", default=os.path.join(ROOT, "baseline_results", "descriptors"))
    ap.add_argument("--out", default=os.path.join(ROOT, "phase5_diagnostics_results", "baseline_diagnostics.json"))
    ap.add_argument("--query-sample", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading diagnostic context (Phase 1-3 artifacts)...", flush=True)
    ctx = load_context(
        dataset_dir=args.dataset_dir,
        patches_dir=args.patches_dir,
        pairs_npz=args.pairs_npz,
        pairs_dataset_json=args.pairs_dataset_json,
        repo_root=ROOT,
    )
    print(f"  fragments: {len(ctx.fragment_ids)}  patches: {ctx.total_patches()}  "
          f"adjacencies: {len(ctx.adjacent_pairs)}", flush=True)
    for fid in ctx.fragment_ids:
        print(f"    {fid}: {len(ctx.contact_patch_ids.get(fid, set()))} contact patches, "
              f"neighbors={sorted(n[-1] for n in ctx.neighbors_of(fid))}", flush=True)

    print("Building reference embedding sources (random, centroid, fpfh)...", flush=True)
    sources = {
        "random": emb_mod.random_embeddings(ctx, dim=64, seed=args.seed),
        "centroid": emb_mod.centroid_embeddings(ctx),
        "fpfh": emb_mod.load_fpfh(ctx, args.descriptors_root, "fpfh"),
    }

    # Peripheral / tiny-contact fragments flagged low-confidence for LOFO.
    low_conf = ("fragment_caesar_fragment_6", "fragment_caesar_fragment_7")

    print("Running diagnostics (this runs the full battery on each source)...", flush=True)
    report = runner.run_diagnostics(
        ctx, sources,
        fpfh_source_name="fpfh",
        query_sample=args.query_sample,
        seed=args.seed,
        low_confidence_fragments=low_conf,
    )

    runner.write_report(args.out, report)
    print(f"\nReport written -> {args.out}", flush=True)

    # Console summary of the headline numbers.
    _print_summary(report)
    return 0


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) and x == x else str(x)


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 72)
    print("HEADLINE SUMMARY (review against PHASE5_PREREGISTRATION.md)")
    print("=" * 72)

    nc = report["null_calibration"]
    for name, r in nc.items():
        print(f"[null floor] {name:9s}  mAP={_fmt(r['mAP'])}  P@1={_fmt(r['precision_at_k'].get('1'))}")

    for name, e in report["per_source"].items():
        raw = e["ranking_raw_gallery"]
        con = e["ranking_contact_only_gallery"]
        ib = e["interface_breakdown"]
        sb = e["shortcut_battery"]
        hn = e["hard_negative_strata"]
        print(f"\n--- source: {name} ---")
        print(f"  ranking raw gallery      : mAP={_fmt(raw['mAP'])}  P@1={_fmt(raw['precision_at_k'].get('1'))}")
        print(f"  ranking contact-only gal : mAP={_fmt(con['mAP'])}  P@1={_fmt(con['precision_at_k'].get('1'))}")
        print(f"  interface macro mAP      : {_fmt(ib['macro_mAP_well_supported'])} "
              f"(n_well_supported={ib['n_well_supported_interfaces']})")
        print(f"  fragment-id kNN acc      : {_fmt(sb['fragment_id_probe']['knn_accuracy'])} "
              f"(chance={_fmt(sb['fragment_id_probe']['chance'])})")
        print(f"  contact-vs-noncontact AUC: {_fmt(sb['contact_vs_noncontact_auc']['auc'])}")
        print(f"  neighbor contact gap     : {_fmt(sb['neighbor_contact_vs_noncontact_ranking']['mean_similarity_gap'])}")
        print(f"  distinct-sim pearson r   : {_fmt(sb['distinctiveness_similarity_correlation']['pearson_r'])}")
        print(f"  boundary openness R^2    : {_fmt(sb['boundary_openness_explained_variance']['r_squared'])}")
        print(f"  AUC pos-vs-easy / hard   : {_fmt(hn['auc_pos_vs_easy'])} / {_fmt(hn['auc_pos_vs_hard'])} "
              f"(gap={_fmt(hn['easy_minus_hard_auc_gap'])})")

    dfp = report["density_fingerprint_probe"]
    print(f"\n[acquisition] fragment-id from density alone: "
          f"{_fmt(dfp['knn_accuracy_from_density'])} (chance={_fmt(dfp['chance'])})")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
