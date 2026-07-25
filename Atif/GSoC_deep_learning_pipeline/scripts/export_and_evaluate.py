#!/usr/bin/env python3
"""Export a trained Phase 5A checkpoint as PatchDescriptors and evaluate it
through the UNMODIFIED diagnostic battery.

This is a DIAGNOSTIC, not training: it loads a frozen checkpoint, encodes every
patch once (eval-mode, training=False -> no jitter, deterministic), builds the
same ``dict[str, PatchDescriptors]`` contract the FPFH/random/centroid sources
use, and runs the existing ranking/LOFO/shortcut code on it.

CRITICAL LOFO CAVEAT: a checkpoint trained holding out fragment F gives a VALID
held-out evaluation ONLY for fold F (F's embeddings come from a model that never
saw F). The other fragments were in F's training set, so any fold != F computed
from this single checkpoint is NOT a held-out number. This script therefore
reports the held-out fold (the checkpoint's own held_out fragment) as the
trustworthy result and clearly separates it from the (non-held-out) rest.
The full six-condition verdict needs all 7 fold checkpoints.

Run:
    cd <repo>
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONPATH=src python3 scripts/export_and_evaluate.py \\
        --checkpoint phase5a_runs/fold_fragment_caesar_fragment_1_best.pt \\
        --held-out fragment_caesar_fragment_1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_geometry.descriptors import PatchDescriptors
from phase5_diagnostics.context import load_context
from phase5_diagnostics import ranking as rk
from phase5_diagnostics import probes as pb
from phase5_diagnostics import embeddings as emb_mod
from phase5_encoder.model import PointNetEncoder, build_encode_patch


def encode_all_patches(ctx, model, *, n_points, k, device):
    """Encode every patch of every fragment into PatchDescriptors, using the
    same eval-mode/training=False path as the pose gate and mining (no jitter,
    deterministic). Returns dict[fid -> PatchDescriptors]."""
    encode_patch = build_encode_patch(
        model, n_points=n_points, k=k, device=device, training=False
    )
    out = {}
    for fid in ctx.fragment_ids:
        pt = ctx.patch_tables[fid]
        vecs = []
        for row in range(pt.patch_count):
            v = encode_patch(pt.patch_points(row), pt.patch_normals(row))
            vecs.append(v)
        mat = np.asarray(vecs, dtype=np.float64)
        out[fid] = PatchDescriptors(
            fragment_id=fid,
            descriptor_type="encoder",
            patch_ids=pt.patch_ids.copy(),
            descriptors=mat,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--dataset-dir", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--patches-dir", default=os.path.join(ROOT, "patches"))
    ap.add_argument("--pairs-npz", default=os.path.join(ROOT, "phase3_final_review", "pairs.npz"))
    ap.add_argument("--pairs-dataset-json", default=os.path.join(ROOT, "pairs", "dataset.json"))
    ap.add_argument("--descriptors-root", default=os.path.join(ROOT, "baseline_results", "descriptors"))
    ap.add_argument("--n-points", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "phase5a_runs", "eval_encoder.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    print("Loading context...", flush=True)
    ctx = load_context(
        dataset_dir=args.dataset_dir, patches_dir=args.patches_dir,
        pairs_npz=args.pairs_npz, pairs_dataset_json=args.pairs_dataset_json,
        repo_root=ROOT,
    )

    print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
    model = PointNetEncoder(out_dim=args.out_dim, pair_input=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    t0 = time.time()
    print("Encoding all patches (eval-mode, no jitter)...", flush=True)
    enc = encode_all_patches(ctx, model, n_points=args.n_points, k=args.k, device=device)
    print(f"  encoded {sum(v.patch_count for v in enc.values())} patches "
          f"in {time.time()-t0:.1f}s", flush=True)

    # FPFH for side-by-side comparison on the same fold.
    fpfh = emb_mod.load_fpfh(ctx, args.descriptors_root, "fpfh")

    held = args.held_out
    report = {
        "checkpoint": args.checkpoint,
        "held_out": held,
        "VALIDITY_NOTE": (
            "Only the held-out fold (this checkpoint's held_out fragment) is a "
            "valid held-out LOFO number. Other folds require their own "
            "checkpoints. This is ONE fold, not the six-condition verdict."
        ),
    }

    # --- Held-out-fold retrieval: encoder vs FPFH, paired per-query ---
    def fold_metrics(emb):
        q_keep = lambda fid, pid, _h=held: (fid == _h and ctx.is_contact_patch(fid, pid))
        g_keep = lambda fid, pid, _h=held: fid != _h
        return rk.evaluate_ranking(
            ctx, emb, k_values=(1, 5, 10, 20), seed=args.seed,
            gallery_keep=g_keep, query_keep=q_keep, query_sample=0,
            return_per_query=True,
        )

    enc_res = fold_metrics(enc)
    fpfh_res = fold_metrics(fpfh)

    report["heldout_fold_retrieval"] = {
        "encoder": {"mAP": enc_res["mAP"],
                     "precision_at_k": enc_res["precision_at_k"],
                     "recall_at_k": enc_res["recall_at_k"],
                     "n_queries": enc_res["n_queries"]},
        "fpfh": {"mAP": fpfh_res["mAP"],
                  "precision_at_k": fpfh_res["precision_at_k"],
                  "recall_at_k": fpfh_res["recall_at_k"],
                  "n_queries": fpfh_res["n_queries"]},
    }

    # --- Paired bootstrap on per-query mAP difference (encoder - FPFH) ---
    # Only valid if the two runs scored the SAME query set in the same order.
    enc_ap = enc_res.get("per_query", {}).get("ap")
    fpfh_ap = fpfh_res.get("per_query", {}).get("ap")
    if enc_ap is not None and fpfh_ap is not None and len(enc_ap) == len(fpfh_ap) and len(enc_ap) > 0:
        diff = np.asarray(enc_ap) - np.asarray(fpfh_ap)
        rng = np.random.default_rng(args.seed)
        n = diff.shape[0]
        boot = rng.integers(0, n, size=(2000, n))
        boot_means = diff[boot].mean(axis=1)
        lo, hi = float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))
        report["paired_mAP_diff_encoder_minus_fpfh"] = {
            "mean": float(diff.mean()), "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_queries": int(n),
            "note": "positive & CI excludes zero => encoder beats FPFH on this fold's mAP",
        }

    # --- Held-out fragment-ID probe (condition 6 lens) ---
    report["heldout_fragment_id_probe"] = {
        "encoder": pb.heldout_fragment_id_probe(ctx, enc, held_out=held, seed=args.seed),
        "fpfh": pb.heldout_fragment_id_probe(ctx, fpfh, held_out=held, seed=args.seed),
    }

    # --- Condition 3: easy-vs-hard AUC gap, FOLD-RESTRICTED (held_out=held) ---
    # The thesis metric. Uses the frozen FPFH-mined hard-negative eval set,
    # scored under EACH source's own embedding, restricted to pairs not
    # touching the held-out fragment (the fix from the condition-3 session).
    hard_neg = rk.mine_hard_negatives(ctx, fpfh, seed=args.seed)
    report["condition3_easy_vs_hard_heldout"] = {
        "encoder": rk.easy_vs_hard_separability(ctx, enc, hard_neg, seed=args.seed, held_out=held),
        "fpfh": rk.easy_vs_hard_separability(ctx, fpfh, hard_neg, seed=args.seed, held_out=held),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))

    # Console summary.
    print("\n" + "=" * 72)
    print(f"HELD-OUT FOLD {held} — encoder vs FPFH (ONE fold, not the verdict)")
    print("=" * 72)
    er, fr = report["heldout_fold_retrieval"]["encoder"], report["heldout_fold_retrieval"]["fpfh"]
    print(f"  mAP:  encoder={er['mAP']:.4f}   fpfh={fr['mAP']:.4f}")
    print(f"  P@1:  encoder={er['precision_at_k'].get('1'):.4f}   fpfh={fr['precision_at_k'].get('1'):.4f}")
    print(f"  n_queries: encoder={er['n_queries']} fpfh={fr['n_queries']}")
    if "paired_mAP_diff_encoder_minus_fpfh" in report:
        p = report["paired_mAP_diff_encoder_minus_fpfh"]
        print(f"  paired mAP diff (enc-fpfh): {p['mean']:+.4f} "
              f"[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}] "
              f"excludes_zero={p['excludes_zero']}")
    hp = report["heldout_fragment_id_probe"]
    print(f"  held-out fragment-ID self-retrieval: encoder="
          f"{hp['encoder'].get('heldout_self_retrieval'):.3f} "
          f"fpfh={hp['fpfh'].get('heldout_self_retrieval'):.3f} "
          f"(base rate {hp['encoder'].get('base_rate'):.3f})")
    c3 = report["condition3_easy_vs_hard_heldout"]
    ce, cf = c3["encoder"], c3["fpfh"]
    print(f"  COND-3 easy-vs-hard gap (held-out; SMALLER is better):")
    print(f"    encoder: easy={ce['auc_pos_vs_easy']:.3f} hard={ce['auc_pos_vs_hard']:.3f} "
          f"gap={ce['easy_minus_hard_auc_gap']:.3f}")
    print(f"    fpfh:    easy={cf['auc_pos_vs_easy']:.3f} hard={cf['auc_pos_vs_hard']:.3f} "
          f"gap={cf['easy_minus_hard_auc_gap']:.3f}")
    _shrunk = ce['easy_minus_hard_auc_gap'] < cf['easy_minus_hard_auc_gap']
    print(f"    encoder shrinks the gap vs FPFH: {_shrunk}")
    print(f"\nWritten -> {args.out}")
    print("REMINDER: one held-out fold. NOT the six-condition verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
