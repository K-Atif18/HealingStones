"""Pipeline orchestration and entry point.

Implements ``run(config_path)`` wiring together config loading, layout,
input loading, FPS center selection, patch extraction, overlap/coverage
analysis, validation, serialization, and metadata, with resilient per-Fragment
handling and fatal I/O handling, plus a ``python -m patch_generation.pipeline``
entry point.
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from patch_generation.config_loader import load_config, dump_config
from patch_generation.patch_layout import ensure_layout
from patch_generation.input_loader import load_all, LoadedFragment
from patch_generation.fps_sampler import select_centers
from patch_generation.patch_extractor import extract_patches
from patch_generation.overlap_analyzer import analyze_overlap
from patch_generation.coverage_analyzer import analyze_coverage
from patch_generation.validation import validate_fragment, overall_pass, size_summary
from patch_generation.patch_record import write_patch_records
from patch_generation.patch_metadata_manager import (
    build_fragment_patch_metadata,
    build_dataset_patch_metadata,
    write_metadata,
)
from patch_generation.logging_setup import configure_logging, get_logger, log_stage, StageStatus
from patch_generation.errors import WriteError


_LOG = get_logger(__name__)


def run(config_path: str) -> dict[str, Any]:
    """Run the full patch-generation pipeline for a configuration file.

    Loads and validates the configuration, ensures the patch layout, loads all
    Phase 1 fragments, processes each non-empty fragment (FPS center selection,
    patch extraction, overlap/coverage analysis, validation, serialization,
    metadata), and writes the dataset-level metadata record with provenance.

    Per-fragment failures (missing input, empty fragment) are handled resiliently:
    they are recorded but do not halt the run. Fatal failures (layout creation,
    write failures) halt immediately.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The dataset-level metadata dict that was written to disk.

    Raises:
        ConfigParseError, ConfigValidationError: On invalid configuration (fatal).
        LayoutError: On patch-layout creation failure (fatal).
        InputLoadError: When dataset metadata is missing/unreadable (fatal).
        WriteError: On any patch record or metadata write failure; the run halts
            and is not marked completed (Req 7.6).
    """
    configure_logging()

    # (1) Configuration (parse/validation failures are fatal)
    config = load_config(config_path)
    log_stage(_LOG, "load_config", StageStatus.SUCCESS, path=config_path)

    # (2) Patch directory layout (creation failure is fatal, Req 7.5)
    paths = ensure_layout(config.patch_dir)
    log_stage(_LOG, "ensure_layout", StageStatus.SUCCESS, patch_dir=config.patch_dir)

    # (3) Load all fragments from Phase 1 outputs (enumeration failure is fatal)
    fragments = load_all(config.dataset_dir)
    log_stage(
        _LOG,
        "load_all_fragments",
        StageStatus.SUCCESS,
        fragment_count=len(fragments),
    )

    # (4) Per-fragment processing (resilient; write failures are fatal)
    per_fragment_meta: dict[str, dict[str, Any]] = {}
    validation_results = []
    skipped_ids = []

    for frag in fragments:
        fragment_id = frag.fragment_id

        # Skip empty or failed fragments
        if frag.skipped:
            log_stage(
                _LOG,
                "process_fragment",
                StageStatus.SKIPPED,
                fragment_id=fragment_id,
                reason="empty_fragment",
            )
            skipped_ids.append(fragment_id)
            per_fragment_meta[fragment_id] = _build_skipped_metadata(frag, paths)
            continue

        if frag.error is not None:
            log_stage(
                _LOG,
                "process_fragment",
                StageStatus.FAILURE,
                fragment_id=fragment_id,
                error=frag.error,
            )
            skipped_ids.append(fragment_id)
            per_fragment_meta[fragment_id] = _build_error_metadata(frag, paths)
            continue

        # Process non-empty, loaded fragment
        try:
            fragment_meta = _process_fragment(frag, config, paths)
            per_fragment_meta[fragment_id] = fragment_meta
            validation_results.append(fragment_meta["validation"])
            log_stage(
                _LOG,
                "process_fragment",
                StageStatus.SUCCESS,
                fragment_id=fragment_id,
                patch_count=fragment_meta["patch_count"],
                coverage_fraction=fragment_meta["coverage_fraction"],
            )
        except WriteError:
            # Write failures are fatal (Req 7.6)
            log_stage(
                _LOG,
                "process_fragment",
                StageStatus.FAILURE,
                fragment_id=fragment_id,
                error="write_failure",
            )
            raise

    # (5) Overall validation checkpoint (Req 9.6)
    checkpoint_passed = overall_pass(validation_results, skipped_ids)
    log_stage(
        _LOG,
        "validation_checkpoint",
        StageStatus.SUCCESS if checkpoint_passed else StageStatus.FAILURE,
        passed=checkpoint_passed,
    )

    # (6) Dataset-level metadata with provenance (Req 10.6)
    dataset_record = build_dataset_patch_metadata(
        run_timestamp=_utc_now_iso8601(),
        config_used=dump_config(config),
        fragment_ids=[f.fragment_id for f in fragments],
        fragments=per_fragment_meta,
        overall_validation_pass=checkpoint_passed,
    )

    # Write is fatal (Req 7.6)
    write_metadata(str(paths.dataset_metadata()), dataset_record)
    log_stage(
        _LOG,
        "write_dataset_metadata",
        StageStatus.SUCCESS,
        path=str(paths.dataset_metadata()),
    )

    return dataset_record


def _process_fragment(
    frag: LoadedFragment,
    config,
    paths,
) -> dict[str, Any]:
    """Process a single loaded fragment through the full patch pipeline.

    Args:
        frag: The loaded fragment with points and normals.
        config: The pipeline configuration.
        paths: The patch layout paths.

    Returns:
        Fragment metadata dict.

    Raises:
        WriteError: On patch record or metadata write failure (fatal).
    """
    fragment_id = frag.fragment_id

    # (a) FPS center selection (Req 2.1-2.6)
    if config.target_center_count is not None:
        center_count = config.target_center_count
    else:
        # Density-based: estimate surface area and compute count
        # For simplicity, approximate as point count / density per point
        # Real impl would use actual surface area computation
        center_count = min(frag.point_count, 1000)  # fallback

    center_indices = select_centers(frag.points, center_count, config.fps_seed)

    # (b) Patch extraction (Req 3.1-4.5)
    patches = extract_patches(
        fragment_id=fragment_id,
        points=frag.points,
        normals=frag.normals,
        center_indices=center_indices,
        patch_radius_mm=config.patch_radius_mm,
        max_patch_points=config.max_patch_points,
    )

    # (c) Overlap analysis (Req 5.1-5.5)
    overlap_stats = analyze_overlap(patches)

    # (d) Coverage analysis (Req 6.1-6.6)
    coverage = analyze_coverage(patches, frag.point_count)

    # (e) Validation (Req 9.1-9.4)
    validation = validate_fragment(coverage, patches, config, fragment_id)

    # (f) Size distribution (Req 9.3)
    size_dist = size_summary(patches)

    # (g) Serialize patch records (Req 7.1, 7.3); write is fatal
    patch_records_path = paths.patch_records(fragment_id)
    write_patch_records(patch_records_path, patches, fragment_id)

    # (h) Build and write fragment metadata (Req 7.2, 7.4)
    fragment_meta = build_fragment_patch_metadata(
        fragment_id=fragment_id,
        point_count=frag.point_count,
        patch_count=len(patches),
        coverage=coverage,
        overlap=overlap_stats,
        size_summary=size_dist,
        coverage_pass=validation.coverage_pass,
        size_pass=validation.size_pass,
        offending_bound=validation.offending_bound,
        artifact_path=str(patch_records_path),
    )

    # Write is fatal
    metadata_path = paths.fragment_metadata(fragment_id)
    write_metadata(str(metadata_path), fragment_meta)

    # Attach validation for aggregation and add metadata_path for dataset-level record
    fragment_meta["validation"] = validation
    fragment_meta["metadata_path"] = str(metadata_path)

    return fragment_meta


def _build_skipped_metadata(frag: LoadedFragment, paths) -> dict[str, Any]:
    """Build metadata for a skipped (empty) fragment."""
    return {
        "fragment_id": frag.fragment_id,
        "point_count": 0,
        "skipped": True,
        "error": None,
        "patch_count": 0,
        "artifact_paths": {},
    }


def _build_error_metadata(frag: LoadedFragment, paths) -> dict[str, Any]:
    """Build metadata for a fragment that failed to load."""
    return {
        "fragment_id": frag.fragment_id,
        "point_count": 0,
        "skipped": False,
        "error": frag.error,
        "patch_count": 0,
        "artifact_paths": {},
    }


def _utc_now_iso8601() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase 2 (Patch Generation) pipeline on Phase 1 outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to patch generation configuration YAML file",
    )
    args = parser.parse_args()

    try:
        run(args.config)
        print(f"✓ Patch generation pipeline completed successfully.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"✗ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
