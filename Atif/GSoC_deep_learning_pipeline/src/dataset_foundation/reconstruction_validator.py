"""Reconstruction validation checkpoint.

Build combined cloud; compute mean/RMSE surface distance and coverage;
pass/fail. Implemented in task 14. Requirements: 8.

This module implements combined-cloud construction (task 14.1), the
combined-to-model distance/coverage metrics (task 14.3), and the pass/fail
decision with offender identification (task 14.6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from dataset_foundation.config_loader import Config
from dataset_foundation.geometry import PointCloud
from dataset_foundation.geometry_metrics import (
    coverage_fraction,
    mean_and_rmse,
    nearest_neighbor_distances,
)

__all__ = [
    "build_combined_cloud",
    "compute_reconstruction_metrics",
    "ReconstructionResult",
    "validate_reconstruction",
]

# Aligned fragments may be provided either keyed by fragment id or as a plain
# sequence. In both cases every cloud is expected to already live in the model
# frame.
AlignedFragments = Union[Mapping[str, PointCloud], Sequence[PointCloud]]


def _iter_clouds(aligned_fragments: AlignedFragments):
    """Yield the :class:`PointCloud` values from a dict or sequence input."""
    if isinstance(aligned_fragments, Mapping):
        yield from aligned_fragments.values()
    else:
        yield from aligned_fragments


def build_combined_cloud(aligned_fragments: AlignedFragments) -> PointCloud:
    """Concatenate model-frame fragment clouds into a single combined cloud.

    Every fragment in ``aligned_fragments`` must already be expressed in the
    model frame. Fragments with zero points are excluded so the combined cloud
    contains exactly the union of the non-empty fragment points (Req 8.1).

    Per-point colors and normals are carried through only when *every*
    non-empty fragment provides them; otherwise the corresponding attribute is
    dropped so the combined arrays stay aligned with the points. Points alone
    are sufficient for downstream reconstruction distance/coverage metrics.

    Args:
        aligned_fragments: Either a mapping of fragment id to
            :class:`PointCloud` or a sequence of :class:`PointCloud`, all in the
            model frame.

    Returns:
        A :class:`PointCloud` whose point count equals the sum of the point
        counts of the non-empty input fragments. When every fragment is empty
        (or the input is empty), an empty :class:`PointCloud` is returned.
    """
    non_empty = [cloud for cloud in _iter_clouds(aligned_fragments) if not cloud.is_empty]

    if not non_empty:
        return PointCloud()

    points = np.concatenate([cloud.points for cloud in non_empty], axis=0)

    colors = None
    if all(cloud.has_colors for cloud in non_empty):
        colors = np.concatenate([cloud.colors for cloud in non_empty], axis=0)

    normals = None
    if all(cloud.has_normals for cloud in non_empty):
        normals = np.concatenate([cloud.normals for cloud in non_empty], axis=0)

    return PointCloud(points=points, colors=colors, normals=normals)


def compute_reconstruction_metrics(
    combined_pc: PointCloud,
    full_model_pc: PointCloud,
    correspondence_distance_mm: float,
) -> tuple[float, float, float]:
    """Combined-cloud reconstruction metrics against the full-model surface.

    Computes the mean and RMSE of the nearest-neighbor distances from every
    combined-cloud point to the full-model surface (Req 8.2), plus the coverage
    fraction of the model within ``correspondence_distance_mm`` (Req 8.3). This
    helper assumes a non-empty combined cloud; callers must handle the empty
    case separately (see :func:`validate_reconstruction`, Req 8.6).

    Args:
        combined_pc: The combined model-frame cloud (typically from
            :func:`build_combined_cloud`).
        full_model_pc: The full-model surface point cloud.
        correspondence_distance_mm: Coverage radius in millimeters.

    Returns:
        A ``(mean_distance_mm, rmse_mm, coverage_fraction)`` tuple of Python
        floats. Both distance values are ``>= 0`` with ``rmse_mm >= mean``, and
        the coverage fraction lies in ``[0.0, 1.0]``.
    """
    distances = nearest_neighbor_distances(combined_pc.points, full_model_pc.points)
    mean_distance_mm, rmse_mm = mean_and_rmse(distances)
    coverage = coverage_fraction(
        full_model_pc.points, combined_pc.points, correspondence_distance_mm
    )
    return mean_distance_mm, rmse_mm, coverage


@dataclass(frozen=True)
class ReconstructionResult:
    """Outcome of the reconstruction validation checkpoint (Req 8.2-8.6).

    Attributes:
        mean_distance_mm: Mean NN distance from the combined cloud to the model
            surface in mm, or ``None`` when the combined cloud is empty.
        rmse_mm: RMSE of those NN distances in mm, or ``None`` when empty.
        coverage_fraction: Fraction of the model covered within the
            correspondence distance, in ``[0, 1]``, or ``None`` when empty.
        threshold_mm: The reconstruction error threshold used for the pass/fail
            decision (``config.reconstruction_error_threshold_mm``).
        passed: ``True`` iff the combined cloud is non-empty and
            ``rmse_mm <= threshold_mm`` (Req 8.4).
        empty: ``True`` when the combined cloud had no points, in which case
            metrics were not computed and ``passed`` is ``False`` (Req 8.6).
        offending_fragment_ids: On failure, the ids of fragments whose
            individual Inlier_RMSE exceeds the alignment threshold (Req 8.5);
            empty on pass.
    """

    mean_distance_mm: Optional[float]
    rmse_mm: Optional[float]
    coverage_fraction: Optional[float]
    threshold_mm: float
    passed: bool
    empty: bool
    offending_fragment_ids: list[str] = field(default_factory=list)


def validate_reconstruction(
    aligned_fragments: AlignedFragments,
    full_model_pc: PointCloud,
    config: Config,
    per_fragment_inlier_rmse: Optional[Mapping[str, float]] = None,
) -> ReconstructionResult:
    """Run the reconstruction validation checkpoint (Req 8.2-8.6).

    Builds the combined model-frame cloud, computes distance/coverage metrics,
    and applies the pass/fail decision:

    * An empty combined cloud reports failure indicating emptiness and skips
      metric computation entirely (Req 8.6).
    * Otherwise, pass iff the RMSE of combined-to-model NN distances is within
      ``config.reconstruction_error_threshold_mm`` (Req 8.4).
    * On failure, the offending fragments are those whose individual Inlier_RMSE
      (from ``per_fragment_inlier_rmse``) exceeds
      ``config.alignment_error_threshold_mm`` (Req 8.5). When no per-fragment
      mapping is supplied, the offender list is empty. On pass the list is
      always empty.

    Args:
        aligned_fragments: Mapping of fragment id to :class:`PointCloud`, or a
            sequence of :class:`PointCloud`, all already in the model frame.
        full_model_pc: The full-model surface point cloud.
        config: The pipeline configuration providing the reconstruction and
            alignment thresholds and the coverage correspondence distance.
        per_fragment_inlier_rmse: Optional mapping of fragment id to that
            fragment's alignment Inlier_RMSE (mm), used to identify offenders on
            failure.

    Returns:
        A :class:`ReconstructionResult` describing the checkpoint outcome.
    """
    threshold_mm = config.reconstruction_error_threshold_mm

    combined = build_combined_cloud(aligned_fragments)
    if combined.is_empty:
        return ReconstructionResult(
            mean_distance_mm=None,
            rmse_mm=None,
            coverage_fraction=None,
            threshold_mm=threshold_mm,
            passed=False,
            empty=True,
            offending_fragment_ids=[],
        )

    mean_distance_mm, rmse_mm, coverage = compute_reconstruction_metrics(
        combined, full_model_pc, config.correspondence_distance_mm
    )

    passed = rmse_mm <= threshold_mm

    offending_fragment_ids: list[str] = []
    if not passed and per_fragment_inlier_rmse:
        alignment_threshold_mm = config.alignment_error_threshold_mm
        offending_fragment_ids = [
            fragment_id
            for fragment_id, inlier_rmse in per_fragment_inlier_rmse.items()
            if inlier_rmse > alignment_threshold_mm
        ]

    return ReconstructionResult(
        mean_distance_mm=mean_distance_mm,
        rmse_mm=rmse_mm,
        coverage_fraction=coverage,
        threshold_mm=threshold_mm,
        passed=passed,
        empty=False,
        offending_fragment_ids=offending_fragment_ids,
    )
