"""Pipeline orchestration.

Orchestrate every dataset-foundation stage, wiring the independently-tested
components together into a single ``run(config_path)`` entry point.

Flow (design "Pipeline Sequence"):

1. ``configure_logging`` then ``load_config`` -- a config parse/validation
   failure is *fatal* (Req 9.2, 9.6) and propagates out of :func:`run`.
2. ``ensure_layout`` -- a layout failure is *fatal* (Req 1.3).
3. Load the Full_Model mesh and derive its point cloud.
4. ``assign_fragment_ids`` -- deterministic Fragment_IDs (Req 1.4, 1.5).
5. For each Fragment, resiliently (Req 3.9): align -> estimate normals ->
   standardize -> write ``fragments``/``transforms``/``normals``/``normalized``
   artifacts -> build and write the per-fragment Metadata record. Per-fragment
   failures (invalid imported transform, invalid scale factor) flag the fragment
   and continue; any *output write* failure is fatal and halts the run without
   marking it completed (Req 7.6).
6. Run reconstruction validation (Req 8.7).
7. Write the dataset-level Metadata record with provenance -- run timestamp,
   the serialized Config, and per-fragment alignment method (Req 9.5).

Each stage emits a structured success/failure log record carrying the stage
name, the Fragment_ID where applicable, and a status (Req 9.4, 9.7).

Requirements: 1.6, 3.6, 4.7, 5.6, 7.1, 7.4, 7.6, 8.7, 9.4, 9.5, 9.7.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dataset_foundation.aligner import AlignmentResult, align
from dataset_foundation.config_loader import Config, load_config
from dataset_foundation.dataset_layout import DatasetPaths, ensure_layout
from dataset_foundation.errors import AlignmentImportError, NormalizationError
from dataset_foundation.fragment_registry import assign_fragment_ids
from dataset_foundation.geometry import PointCloud
from dataset_foundation.logging_setup import (
    StageStatus,
    configure_logging,
    get_logger,
    log_stage,
)
from dataset_foundation.metadata_manager import (
    build_dataset_metadata,
    build_fragment_metadata,
    write_metadata,
)
from dataset_foundation.normal_estimator import estimate_normals
from dataset_foundation.normalizer import standardize, voxel_downsample, write_normalized
from dataset_foundation.ply_io import load_mesh, mesh_to_point_cloud, write_point_cloud
from dataset_foundation.reconstruction_validator import (
    ReconstructionResult,
    validate_reconstruction,
)
from dataset_foundation.transforms_io import write_transform

__all__ = ["run", "main"]

_LOG = get_logger("pipeline")

#: Default configuration path used by the module entry point.
DEFAULT_CONFIG_PATH = "config/default.yaml"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now_iso8601() -> str:
    """Return the current UTC time as an ISO 8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relpath(root: Path, path: Path) -> str:
    """Return ``path`` relative to the dataset ``root`` using forward slashes."""
    return Path(path).relative_to(root).as_posix()


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map ``(N, 3)`` points through a 4x4 homogeneous transform.

    The transform maps the Fragment frame onto the Full_Model frame, so the
    result is ``points @ R.T + t`` with ``R = transform[:3, :3]`` and
    ``t = transform[:3, 3]`` (Req 3.1).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return pts @ rotation.T + translation


def _aligned_cloud(fragment_pc: PointCloud, result: AlignmentResult) -> PointCloud:
    """Build the model-frame aligned cloud for a Fragment.

    A skipped (zero-vertex) Fragment yields an empty cloud; otherwise the
    Fragment points are mapped through the resolved transform (identity when no
    valid transform is available). Colors are carried through; normals are
    dropped here and re-estimated on the aligned cloud.
    """
    if result.skipped or fragment_pc.is_empty:
        return PointCloud()
    transform = result.transform
    if transform is None:
        transform = np.eye(4, dtype=np.float64)
    transformed = _apply_transform(fragment_pc.points, transform)
    return PointCloud(points=transformed, colors=fragment_pc.colors, normals=None)


def _alignment_error_dict(result: AlignmentResult) -> dict[str, Any] | None:
    """Convert an :class:`AlignmentResult` error into a plain dict (or ``None``)."""
    error = result.error
    if error is None:
        return None
    return {
        "inlier_rmse_mm": float(error.inlier_rmse_mm),
        "fitness": float(error.fitness),
        "mean_surface_distance_mm": float(error.mean_surface_distance_mm),
    }


def _reconstruction_dict(result: ReconstructionResult) -> dict[str, Any]:
    """Serialize a :class:`ReconstructionResult` for the dataset metadata (Req 8.7)."""
    return {
        "mean_distance_mm": result.mean_distance_mm,
        "rmse_mm": result.rmse_mm,
        "coverage_fraction": result.coverage_fraction,
        "threshold_mm": result.threshold_mm,
        "passed": result.passed,
        "empty": result.empty,
        "offending_fragment_ids": list(result.offending_fragment_ids),
    }


# ---------------------------------------------------------------------------
# Per-fragment processing
# ---------------------------------------------------------------------------


def _process_fragment(
    *,
    fragment_id: str,
    fragment_path: str,
    full_model_pc: PointCloud,
    config: Config,
    paths: DatasetPaths,
) -> tuple[PointCloud, float | None, dict[str, Any]]:
    """Process a single Fragment end-to-end and write all of its artifacts.

    Returns ``(aligned_pc, inlier_rmse_mm, fragment_record)`` where ``aligned_pc``
    is the model-frame cloud accumulated for reconstruction, ``inlier_rmse_mm``
    is this fragment's Inlier_RMSE (or ``None``), and ``fragment_record`` is the
    written per-fragment Metadata dict.

    Per-fragment errors (invalid imported transform, invalid scale factor) flag
    the fragment and continue (Req 3.8, 5.7). Output write failures are *fatal*
    and propagate (Req 7.6).
    """
    source_filename = os.path.basename(fragment_path)

    # --- Load mesh + derived point cloud (Req 2.1-2.3) ----------------------
    mesh = load_mesh(fragment_path)
    fragment_pc = mesh_to_point_cloud(mesh)
    # Memory-safety: fragments are very dense (up to ~10^6 points). Downsample to
    # the configured voxel size before the per-point stages (alignment metrics,
    # normal estimation, reconstruction) so they never operate on the full-
    # resolution cloud. Registration downsamples further internally for FPFH, so
    # this does not degrade coarse alignment. The original vertex count is still
    # recorded in metadata from ``mesh`` below.
    fragment_pc = voxel_downsample(fragment_pc, config.voxel_size_mm)
    log_stage(
        _LOG,
        "load_fragment",
        StageStatus.SUCCESS,
        fragment_id=fragment_id,
        vertex_count=mesh.vertex_count,
        downsampled_point_count=fragment_pc.point_count,
    )

    flagged_for_review = False
    failure_reason: str | None = None

    # --- Alignment (resilient: import failure flags + continues) ------------
    try:
        result = align(fragment_pc, full_model_pc, fragment_id, config)
    except AlignmentImportError as exc:
        # Invalid precomputed transform: flag + record, do NOT halt (Req 3.8).
        result = AlignmentResult(
            transform=None,
            error=None,
            method="imported",
            flagged_for_review=True,
            skipped=False,
            failure_reason=str(exc),
        )
        log_stage(
            _LOG,
            "align",
            StageStatus.FAILURE,
            fragment_id=fragment_id,
            message=str(exc),
        )
    else:
        log_stage(
            _LOG,
            "align",
            StageStatus.SUCCESS if not result.flagged_for_review else StageStatus.FAILURE,
            fragment_id=fragment_id,
            method=result.method,
            flagged_for_review=result.flagged_for_review,
        )

    flagged_for_review = flagged_for_review or result.flagged_for_review
    failure_reason = result.failure_reason or failure_reason

    # Resolve the model-frame aligned cloud and the transform to persist.
    aligned_pc = _aligned_cloud(fragment_pc, result)
    transform_matrix = (
        result.transform if result.transform is not None else np.eye(4, dtype=np.float64)
    )

    # --- Normal estimation on the aligned cloud (Req 4.1-4.7) ---------------
    normals_pc, degenerate_count = estimate_normals(aligned_pc, config.normal_params)
    log_stage(
        _LOG,
        "estimate_normals",
        StageStatus.SUCCESS,
        fragment_id=fragment_id,
        degenerate_normal_count=degenerate_count,
    )

    # --- Density standardization + coordinate normalization (Req 5.x) -------
    normalized_pc: PointCloud | None = None
    applied_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    applied_scale: float = 1.0
    try:
        normalized_pc, applied_offset, applied_scale = standardize(
            normals_pc, config, fragment_id
        )
        log_stage(
            _LOG,
            "standardize",
            StageStatus.SUCCESS,
            fragment_id=fragment_id,
            point_count=normalized_pc.point_count,
        )
    except NormalizationError as exc:
        # Invalid scale factor: halt this fragment's standardization, flag +
        # record, and continue with the run (Req 5.7).
        flagged_for_review = True
        failure_reason = (
            f"{failure_reason}; {exc}" if failure_reason else str(exc)
        )
        log_stage(
            _LOG,
            "standardize",
            StageStatus.FAILURE,
            fragment_id=fragment_id,
            message=str(exc),
        )

    # --- Write artifacts. ANY write failure is FATAL (Req 7.1, 7.6) ---------
    # Every fragment gets an aligned cloud + a transform file + a normals cloud
    # (Req 7.1); the transform file is written even for skipped/failed fragments
    # using the identity matrix so the transforms/ directory stays complete.
    write_point_cloud(str(paths.fragment_pc(fragment_id)), aligned_pc)
    write_transform(str(paths.transform(fragment_id)), transform_matrix)
    write_point_cloud(str(paths.normals_pc(fragment_id)), normals_pc)

    normalized_rel: str | None = None
    point_count_after_normalization = 0
    if normalized_pc is not None:
        write_normalized(str(paths.normalized_pc(fragment_id)), normalized_pc)
        normalized_rel = _relpath(paths.root, paths.normalized_pc(fragment_id))
        point_count_after_normalization = normalized_pc.point_count

    log_stage(
        _LOG,
        "write_artifacts",
        StageStatus.SUCCESS,
        fragment_id=fragment_id,
    )

    # --- Build + write per-fragment Metadata (write failure fatal, Req 1.8) -
    artifact_paths = {
        "aligned": _relpath(paths.root, paths.fragment_pc(fragment_id)),
        "normals": _relpath(paths.root, paths.normals_pc(fragment_id)),
        "normalized": normalized_rel,
    }
    fragment_record = build_fragment_metadata(
        fragment_id=fragment_id,
        source_filename=source_filename,
        vertex_count=mesh.vertex_count,
        face_count=mesh.face_count,
        point_count_after_normalization=point_count_after_normalization,
        is_empty=fragment_pc.is_empty,
        transform_file=_relpath(paths.root, paths.transform(fragment_id)),
        alignment_method=result.method,
        voxel_size_mm=config.voxel_size_mm,
        alignment_error=_alignment_error_dict(result),
        flagged_for_review=flagged_for_review,
        skipped=result.skipped,
        failure_reason=failure_reason,
        degenerate_normal_count=degenerate_count,
        centering_offset=list(applied_offset),
        scale_factor=applied_scale,
        artifact_paths=artifact_paths,
    )
    write_metadata(str(paths.metadata_record(fragment_id)), fragment_record)
    log_stage(
        _LOG,
        "write_metadata",
        StageStatus.SUCCESS,
        fragment_id=fragment_id,
    )

    inlier_rmse = result.error.inlier_rmse_mm if result.error is not None else None
    return aligned_pc, inlier_rmse, fragment_record


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(config_path: str) -> dict[str, Any]:
    """Run the full dataset-foundation pipeline for a configuration file.

    Loads and validates the configuration, ensures the dataset layout, loads the
    Full_Model, assigns Fragment_IDs, processes every Fragment (align -> normals
    -> standardize -> write artifacts -> write metadata), runs the reconstruction
    validation checkpoint, and writes the dataset-level Metadata record with
    provenance.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The dataset-level Metadata dict that was written to disk.

    Raises:
        ConfigParseError, ConfigValidationError: On invalid configuration (fatal).
        LayoutError: On dataset-layout creation failure (fatal).
        PlyReadError: When a required PLY input is missing/unreadable (fatal).
        WriteError: On any output write failure; the run halts and is not marked
            completed (Req 7.6).
    """
    configure_logging()

    # (1) Configuration (parse/validation failures are fatal, Req 9.2, 9.6).
    config = load_config(config_path)
    log_stage(_LOG, "load_config", StageStatus.SUCCESS, path=config_path)

    # (2) Dataset layout (creation failure is fatal, Req 1.3).
    paths = ensure_layout(config.dataset_dir)
    log_stage(_LOG, "ensure_layout", StageStatus.SUCCESS, dataset_dir=str(paths.root))

    # (3) Load the Full_Model mesh and derive its point cloud (fatal on read).
    full_model_mesh = load_mesh(config.full_model_path)
    full_model_pc = mesh_to_point_cloud(full_model_mesh)
    # Memory-safety: the reference model is very dense (~10^6 points). Downsample
    # it once to the configured voxel size so every per-fragment alignment,
    # metric, and reconstruction query runs against a reduced reference cloud
    # instead of the full-resolution one. The mesh vertex/face counts recorded in
    # metadata below still reflect the original full-resolution geometry.
    full_model_pc = voxel_downsample(full_model_pc, config.voxel_size_mm)
    log_stage(
        _LOG,
        "load_full_model",
        StageStatus.SUCCESS,
        vertex_count=full_model_mesh.vertex_count,
        face_count=full_model_mesh.face_count,
        downsampled_point_count=full_model_pc.point_count,
    )

    # (4) Deterministic Fragment_IDs (Req 1.4, 1.5).
    registry = assign_fragment_ids(config.fragment_paths)
    fragment_ids = list(registry.ids())
    log_stage(
        _LOG,
        "assign_fragment_ids",
        StageStatus.SUCCESS,
        fragment_count=len(fragment_ids),
    )

    # (5) Per-fragment processing (resilient; write failures are fatal).
    aligned_by_id: dict[str, PointCloud] = {}
    per_fragment_inlier_rmse: dict[str, float] = {}
    fragments_meta: dict[str, dict[str, Any]] = {}

    for fragment_id in fragment_ids:
        fragment_path = registry.filename_for(fragment_id)
        aligned_pc, inlier_rmse, fragment_record = _process_fragment(
            fragment_id=fragment_id,
            fragment_path=fragment_path,
            full_model_pc=full_model_pc,
            config=config,
            paths=paths,
        )
        aligned_by_id[fragment_id] = aligned_pc
        if inlier_rmse is not None:
            per_fragment_inlier_rmse[fragment_id] = inlier_rmse
        fragments_meta[fragment_id] = {
            "aligned": _relpath(paths.root, paths.fragment_pc(fragment_id)),
            "transform": _relpath(paths.root, paths.transform(fragment_id)),
            "normals": _relpath(paths.root, paths.normals_pc(fragment_id)),
            "normalized": fragment_record["artifact_paths"].get("normalized"),
            "metadata": _relpath(paths.root, paths.metadata_record(fragment_id)),
            "alignment_method": fragment_record["alignment_method"],
        }

    # (6) Reconstruction validation checkpoint (Req 8.7).
    reconstruction = validate_reconstruction(
        aligned_by_id,
        full_model_pc,
        config,
        per_fragment_inlier_rmse=per_fragment_inlier_rmse,
    )
    log_stage(
        _LOG,
        "reconstruction_validation",
        StageStatus.SUCCESS if reconstruction.passed else StageStatus.FAILURE,
        rmse_mm=reconstruction.rmse_mm,
        coverage_fraction=reconstruction.coverage_fraction,
        passed=reconstruction.passed,
    )

    # (7) Dataset-level Metadata with provenance (Req 9.5); write is fatal.
    dataset_record = build_dataset_metadata(
        run_timestamp=_utc_now_iso8601(),
        config_used=config,
        fragment_ids=fragment_ids,
        id_to_filename={fid: os.path.basename(registry.filename_for(fid)) for fid in fragment_ids},
        fragments=fragments_meta,
        full_model={
            "vertex_count": full_model_mesh.vertex_count,
            "face_count": full_model_mesh.face_count,
        },
        reconstruction=_reconstruction_dict(reconstruction),
    )
    write_metadata(str(paths.dataset_metadata()), dataset_record)
    log_stage(
        _LOG,
        "write_dataset_metadata",
        StageStatus.SUCCESS,
        path=str(paths.dataset_metadata()),
    )

    return dataset_record


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m dataset_foundation.pipeline``.

    Parses ``--config`` (defaulting to ``config/default.yaml``) and runs the
    pipeline. Returns ``0`` on success.
    """
    parser = argparse.ArgumentParser(
        description="Run the dataset-foundation pipeline on a configuration file."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    args = parser.parse_args(argv)

    result = run(args.config)
    reconstruction = result.get("reconstruction", {})
    print(
        "Pipeline complete: "
        f"fragments={len(result.get('fragment_ids', []))} "
        f"reconstruction_passed={reconstruction.get('passed')} "
        f"rmse_mm={reconstruction.get('rmse_mm')}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
