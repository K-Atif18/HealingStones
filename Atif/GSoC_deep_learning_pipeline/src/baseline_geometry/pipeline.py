"""Phase 4 pipeline orchestration and CLI entry point.

Wires together: config loading, descriptor computation (FPFH + SHOT) for all
fragments, descriptor-based retrieval and pair separability, FPFH+RANSAC+ICP
registration, and the top-level baseline report with the validation gate.

Run:
    PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml

Selective stages (for iteration):
    ... --stages descriptors,retrieval
    ... --stages registration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from baseline_geometry.config_loader import load_config, dump_config, Config
from baseline_geometry import data_access as da
from baseline_geometry import descriptors as desc_mod
from baseline_geometry import retrieval as ret_mod
from baseline_geometry import registration as reg_mod
from baseline_geometry.errors import WriteError
from baseline_geometry.logging_setup import (
    configure_logging, get_logger, log_stage, StageStatus,
)

_LOG = get_logger("pipeline")

ALL_STAGES = ("descriptors", "retrieval", "registration")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, obj: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, default=_json_default)
        os.replace(tmp, path)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise WriteError(path, str(exc)) from exc


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Stage 1: descriptors
# ---------------------------------------------------------------------------
def stage_descriptors(config: Config, fragment_ids: list[str]) -> dict[str, dict[str, desc_mod.PatchDescriptors]]:
    """Compute + persist descriptors for every fragment and requested type.

    Returns nested dict: descriptor_type -> fragment_id -> PatchDescriptors.
    """
    out: dict[str, dict[str, desc_mod.PatchDescriptors]] = {d: {} for d in config.descriptors}

    for fid in fragment_ids:
        cloud = da.load_fragment_cloud(config.dataset_dir, fid)
        table = da.load_patch_table(config.patches_dir, fid)
        normals = desc_mod.ensure_normals(cloud, config.normal_radius_mm, config.normal_max_nn)

        for dtype in config.descriptors:
            t0 = time.time()
            pd = desc_mod.compute_patch_descriptors(dtype, cloud, table, normals, config)
            path = os.path.join(config.output_dir, "descriptors", dtype, f"{fid}.npz")
            desc_mod.write_descriptors(path, pd)
            out[dtype][fid] = pd
            log_stage(
                _LOG, "descriptors", StageStatus.SUCCESS, fragment_id=fid,
                descriptor=dtype, dim=pd.dim, n_patches=pd.patch_count,
                seconds=round(time.time() - t0, 2),
            )
    return out


def _load_descriptors(config: Config, fragment_ids: list[str]) -> dict[str, dict[str, desc_mod.PatchDescriptors]]:
    """Load previously written descriptors from disk (for selective stages)."""
    out: dict[str, dict[str, desc_mod.PatchDescriptors]] = {d: {} for d in config.descriptors}
    for dtype in config.descriptors:
        for fid in fragment_ids:
            path = os.path.join(config.output_dir, "descriptors", dtype, f"{fid}.npz")
            out[dtype][fid] = desc_mod.read_descriptors(path)
    return out


# ---------------------------------------------------------------------------
# Stage 2: retrieval
# ---------------------------------------------------------------------------
def stage_retrieval(config: Config, desc_all: dict) -> dict:
    pairs = da.load_pairs(config.pairs_npz)
    summary: dict[str, Any] = {}

    for dtype in config.descriptors:
        t0 = time.time()
        pair_metrics = ret_mod.evaluate_pairs(
            pairs, desc_all[dtype], sample=config.pair_auc_sample, seed=config.perturb_seed,
        )
        rank_metrics = ret_mod.evaluate_ranking(
            pairs, desc_all[dtype],
            k_values=config.retrieval_k_values,
            query_sample=config.retrieval_query_sample,
            seed=config.perturb_seed,
        )
        result = {"pair_separability": pair_metrics, "ranking": rank_metrics}
        _atomic_write_json(os.path.join(config.output_dir, "retrieval", f"{dtype}_retrieval.json"), result)
        summary[dtype] = {
            "roc_auc": pair_metrics["roc_auc"],
            "average_precision": pair_metrics["average_precision"],
            "mAP": rank_metrics["mAP"],
            "precision_at_1": rank_metrics["precision_at_k"].get("1"),
            "high_overlap_auc": pair_metrics["stratified"]["high_overlap_positives"]["roc_auc"],
            "low_overlap_auc": pair_metrics["stratified"]["low_overlap_positives"]["roc_auc"],
        }
        log_stage(
            _LOG, "retrieval", StageStatus.SUCCESS, descriptor=dtype,
            roc_auc=round(pair_metrics["roc_auc"], 4), mAP=round(rank_metrics["mAP"], 4),
            seconds=round(time.time() - t0, 2),
        )

    _atomic_write_json(os.path.join(config.output_dir, "retrieval", "retrieval_summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------
# Stage 3: registration
# ---------------------------------------------------------------------------
def stage_registration(config: Config, fragment_ids: list[str]) -> dict:
    clouds = {fid: da.load_fragment_cloud(config.dataset_dir, fid) for fid in fragment_ids}
    adjacent_pairs = da.load_adjacent_pairs(config.pairs_dataset_json)
    contact_dir = os.path.join(os.path.dirname(config.pairs_dataset_json), "contact_regions")

    t0 = time.time()
    result = reg_mod.run_registration_experiment(
        clouds, adjacent_pairs, config, contact_dir=contact_dir,
    )

    # Persist per-pair details and the summary.
    per_pair_dir = os.path.join(config.output_dir, "registration", "per_pair")
    for entry in result["per_pair"]:
        fname = f"{entry['fragment_A']}__{entry['fragment_B']}_global.json"
        _atomic_write_json(os.path.join(per_pair_dir, fname), entry)
    for entry in result["contact_per_pair"]:
        fname = f"{entry['fragment_A']}__{entry['fragment_B']}_contact.json"
        _atomic_write_json(os.path.join(per_pair_dir, fname), entry)
    _atomic_write_json(os.path.join(config.output_dir, "registration", "fpfh_registration.json"), result)

    log_stage(
        _LOG, "registration", StageStatus.SUCCESS,
        adjacent_global_success=result["adjacent_global_summary"].get("success_rate"),
        adjacent_contact_success=result["adjacent_contact_summary"].get("success_rate"),
        control_success=result["control_summary"].get("success_rate"),
        seconds=round(time.time() - t0, 2),
    )
    return {
        "adjacent_global_summary": result["adjacent_global_summary"],
        "adjacent_contact_summary": result["adjacent_contact_summary"],
        "control_summary": result["control_summary"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(config_path: str, stages: tuple[str, ...] = ALL_STAGES) -> dict:
    configure_logging()
    config = load_config(config_path)
    log_stage(_LOG, "load_config", StageStatus.SUCCESS, path=config_path, stages=",".join(stages))

    fragment_ids = da.load_fragment_ids(config.dataset_dir)
    log_stage(_LOG, "load_fragments", StageStatus.SUCCESS, n=len(fragment_ids))

    report: dict[str, Any] = {
        "phase": "Phase 4: Baseline Geometry",
        "run_timestamp": _utc_now(),
        "config_used": dump_config(config),
        "fragment_ids": fragment_ids,
        "stages_run": list(stages),
    }

    desc_all: dict | None = None
    if "descriptors" in stages:
        desc_all = stage_descriptors(config, fragment_ids)
        report["descriptors"] = {
            dtype: {fid: desc_all[dtype][fid].dim for fid in fragment_ids}
            for dtype in config.descriptors
        }
        report["total_patches_described"] = int(
            sum(desc_all[config.descriptors[0]][fid].patch_count for fid in fragment_ids)
        )

    if "retrieval" in stages:
        if desc_all is None:
            desc_all = _load_descriptors(config, fragment_ids)
        report["retrieval"] = stage_retrieval(config, desc_all)

    if "registration" in stages:
        report["registration"] = stage_registration(config, fragment_ids)

    # -----------------------------------------------------------------
    # Validation gate
    # -----------------------------------------------------------------
    gate = _evaluate_gate(config, report)
    report["validation"] = gate
    log_stage(
        _LOG, "validation_gate",
        StageStatus.SUCCESS if gate["passed"] else StageStatus.FAILURE,
        passed=gate["passed"],
    )

    _atomic_write_json(os.path.join(config.output_dir, "baseline_report.json"), report)
    log_stage(_LOG, "write_report", StageStatus.SUCCESS,
              path=os.path.join(config.output_dir, "baseline_report.json"))
    return report


def _evaluate_gate(config: Config, report: dict) -> dict:
    checks: list[dict] = []

    if "retrieval" in report:
        for dtype, m in report["retrieval"].items():
            auc = m.get("roc_auc")
            ok = auc is not None and not (isinstance(auc, float) and np.isnan(auc)) and auc >= config.min_pair_auc
            checks.append({
                "check": f"{dtype}_roc_auc>=?{config.min_pair_auc}",
                "value": auc, "passed": bool(ok),
            })

    if "registration" in report:
        adj = report["registration"]["adjacent_contact_summary"].get("success_rate")
        ctl = report["registration"]["control_summary"].get("success_rate")
        if adj is not None and ctl is not None:
            checks.append({
                "check": "registration_adjacent_contact>control",
                "value": {"adjacent_contact": adj, "control": ctl},
                "passed": bool(adj > ctl),
            })

    # Descriptors coverage check.
    if "total_patches_described" in report:
        checks.append({
            "check": "descriptors_computed_for_all_patches",
            "value": report["total_patches_described"],
            "passed": report["total_patches_described"] > 0,
        })

    passed = all(c["passed"] for c in checks) if checks else False
    return {
        "passed": passed,
        "checks": checks,
        "note": (
            "The gate confirms the baseline is a trustworthy reference point "
            "(beats chance; adjacent registration separable from control). It "
            "does NOT mean the baseline is sufficient for assembly -- the "
            "similarity-vs-complementarity gap in the stratified retrieval "
            "metrics is the expected limitation motivating later phases."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 (Baseline Geometry) pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline_geometry.yaml")
    parser.add_argument(
        "--stages", type=str, default=",".join(ALL_STAGES),
        help=f"Comma-separated subset of {ALL_STAGES}",
    )
    args = parser.parse_args()

    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        print(f"✗ Unknown stage(s): {unknown}. Valid: {ALL_STAGES}", file=sys.stderr)
        sys.exit(2)

    try:
        report = run(args.config, stages)
        passed = report["validation"]["passed"]
        print(f"{'✓' if passed else '✗'} Phase 4 pipeline completed. Validation gate: "
              f"{'PASSED' if passed else 'FAILED'}", file=sys.stderr)
        sys.exit(0 if passed else 1)
    except Exception as exc:
        print(f"✗ Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
