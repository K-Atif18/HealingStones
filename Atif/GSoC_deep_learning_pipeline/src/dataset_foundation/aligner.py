"""Fragment-to-model rigid alignment.

Compute a Rigid_Transform (global registration + ICP) or import/validate a
precomputed one; compute Alignment_Error.

The pipeline maps each Fragment's coordinate frame onto the Full_Model frame
(Req 3.1). A Fragment may instead carry a precomputed transform in the
Configuration, in which case that matrix is validated and imported rather than
recomputed (Req 3.2, 3.8). Alignment quality is summarized by an
:class:`AlignmentError` (Inlier_RMSE, Fitness, mean surface distance) in
millimeters (Req 3.4); Fragments with excessive error or zero fitness are
flagged for review (Req 3.5, 3.9) and zero-vertex Fragments are skipped
(Req 3.10).

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 3.9, 3.10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from dataset_foundation.config_loader import Config
from dataset_foundation.errors import AlignmentImportError
from dataset_foundation.geometry import PointCloud
from dataset_foundation.geometry_metrics import mean_and_rmse, nearest_neighbor_distances

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "align",
    "is_valid_rigid_transform",
]

_LOG = logging.getLogger(__name__)

# --- Registration robustness knobs (internal, deterministic) ----------------
# RANSAC global registration is stochastic and, on fragments of a symmetric
# whole, a single attempt often lands in a wrong-but-plausible basin. We run a
# handful of independently-seeded restarts and keep the best result (highest
# fitness, then lowest inlier RMSE). Seeding each restart from ``seed + i`` keeps
# the whole procedure reproducible.
_RANSAC_RESTARTS = 8

# Point-to-plane ICP is refined coarse-to-fine: a wide correspondence radius
# first pulls a mis-tilted fragment into place, then progressively tighter
# radii lock in the fine alignment. Multipliers are applied to
# ``icp_max_distance_mm``.
_ICP_SCALE_SCHEDULE = (8.0, 4.0, 2.0, 1.0)


def is_valid_rigid_transform(matrix, det_tol: float = 1e-3) -> bool:
    """Return True iff ``matrix`` is a valid 4x4 homogeneous rigid transform.

    A valid Rigid_Transform (Req 3.3) satisfies all of:

    * shape is exactly ``(4, 4)`` and all entries are finite;
    * the upper-left 3x3 block ``R`` is a rotation, i.e. orthonormal
      (``R @ R.T`` is close to the identity);
    * ``det(R)`` lies within ``1.0 ± det_tol`` (so proper rotations pass but
      reflections with determinant ``-1`` and scalings are rejected);
    * the bottom row is exactly ``[0, 0, 0, 1]`` (up to a tiny numerical
      epsilon).

    The input may be any array-like. Anything that cannot be interpreted as a
    4x4 numeric array, or that has the wrong shape, returns ``False`` rather
    than raising.

    Args:
        matrix: Array-like candidate transform.
        det_tol: Allowed absolute deviation of ``det(R)`` from ``1.0``.

    Returns:
        ``True`` if the matrix is a valid rigid transform, else ``False``.
    """
    try:
        m = np.asarray(matrix, dtype=np.float64)
    except (ValueError, TypeError):
        return False

    if m.shape != (4, 4):
        return False

    if not np.all(np.isfinite(m)):
        return False

    # Bottom row must be exactly [0, 0, 0, 1] (allow tiny numerical epsilon).
    bottom_eps = 1e-9
    if not np.allclose(m[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=bottom_eps, rtol=0.0):
        return False

    r = m[:3, :3]

    # Orthonormality: R @ R.T ~= I.
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0.0):
        return False

    # Proper rotation: determinant within 1.0 +/- det_tol (rejects reflections).
    det = float(np.linalg.det(r))
    if abs(det - 1.0) > det_tol:
        return False

    return True


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


@dataclass
class AlignmentError:
    """Quantitative summary of alignment quality (all in millimeters where applicable).

    Attributes:
        inlier_rmse_mm: RMSE of corresponding point pairs within the
            correspondence distance threshold, in mm (Req 3.4).
        fitness: Fraction of Fragment points with a Full_Model correspondence
            within the threshold, in ``[0, 1]`` (Req 3.4).
        mean_surface_distance_mm: Mean nearest-neighbor distance from the aligned
            Fragment points to the Full_Model points, in mm (Req 3.4).
    """

    inlier_rmse_mm: float
    fitness: float
    mean_surface_distance_mm: float


@dataclass
class AlignmentResult:
    """Outcome of aligning a single Fragment to the Full_Model.

    Attributes:
        transform: The final 4x4 Rigid_Transform (float64) mapping the Fragment
            frame onto the Full_Model frame, or ``None`` when skipped.
        error: The :class:`AlignmentError` metrics, or ``None`` when skipped.
        method: How the transform was obtained: ``"computed"``, ``"imported"``,
            or ``"skipped"``.
        flagged_for_review: ``True`` when the Fragment needs manual review
            (excessive Inlier_RMSE, zero fitness, or an invalid matrix).
        skipped: ``True`` when the Fragment had zero vertices and alignment was
            skipped entirely (Req 3.10).
        failure_reason: Human-readable reason when flagged/failed, else ``None``.
    """

    transform: np.ndarray | None
    error: AlignmentError | None
    method: str
    flagged_for_review: bool
    skipped: bool
    failure_reason: str | None


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def _preprocess_for_features(
    pcd: o3d.geometry.PointCloud, config: Config
) -> tuple[o3d.geometry.PointCloud, "o3d.pipelines.registration.Feature"]:
    """Voxel-downsample a cloud, estimate normals, and compute FPFH features.

    Returns the downsampled cloud (with normals) and its FPFH descriptors.
    """
    reg = config.registration_params
    down = pcd.voxel_down_sample(voxel_size=reg.feature_voxel_size_mm)

    down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=reg.normal_radius_mm, max_nn=30
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=reg.fpfh_radius_mm, max_nn=reg.fpfh_max_neighbors
        ),
    )
    return down, fpfh


def _global_registration(
    frag_down: o3d.geometry.PointCloud,
    model_down: o3d.geometry.PointCloud,
    frag_fpfh: "o3d.pipelines.registration.Feature",
    model_fpfh: "o3d.pipelines.registration.Feature",
    config: Config,
) -> np.ndarray:
    """Multi-start RANSAC feature-matching global registration.

    Runs ``_RANSAC_RESTARTS`` independently-seeded RANSAC attempts and keeps the
    best transform (highest fitness, then lowest inlier RMSE). Multiple restarts
    make coarse alignment far more reliable for fragments of a symmetric whole,
    while per-restart seeding keeps the result reproducible. Returns a coarse
    4x4 transform.
    """
    reg = config.registration_params
    distance = reg.ransac_distance_mm

    best_transform = np.eye(4, dtype=np.float64)
    best_key: tuple[float, float] | None = None

    for attempt in range(_RANSAC_RESTARTS):
        try:  # best-effort per-restart seeding for reproducibility
            o3d.utility.random.seed(reg.seed + attempt)
        except Exception:  # pragma: no cover - depends on Open3D build
            pass

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            frag_down,
            model_down,
            frag_fpfh,
            model_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=reg.ransac_max_iterations,
                confidence=reg.ransac_confidence,
            ),
        )
        # Maximize fitness, then minimize inlier RMSE (0 rmse with 0 fitness is
        # a degenerate "no correspondences" case, ranked worst by fitness).
        key = (float(result.fitness), -float(result.inlier_rmse))
        if best_key is None or key > best_key:
            best_key = key
            best_transform = np.asarray(result.transformation, dtype=np.float64)

    return best_transform


def _icp_refine(
    frag_down: o3d.geometry.PointCloud,
    model_down: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Multi-scale point-to-plane ICP refinement starting from ``init_transform``.

    Refines coarse-to-fine over ``_ICP_SCALE_SCHEDULE``: a wide correspondence
    radius first corrects a mis-tilted coarse pose, then tighter radii lock in
    the fine alignment. The model cloud must carry normals for point-to-plane
    estimation.
    """
    reg = config.registration_params
    transform = np.asarray(init_transform, dtype=np.float64)

    for multiplier in _ICP_SCALE_SCHEDULE:
        max_distance = reg.icp_max_distance_mm * multiplier
        result = o3d.pipelines.registration.registration_icp(
            frag_down,
            model_down,
            max_distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=reg.icp_max_iterations
            ),
        )
        transform = np.asarray(result.transformation, dtype=np.float64)

    return transform


def _compute_alignment_error(
    frag_o3d: o3d.geometry.PointCloud,
    model_o3d: o3d.geometry.PointCloud,
    fragment_points: np.ndarray,
    model_points: np.ndarray,
    transform: np.ndarray,
    config: Config,
) -> AlignmentError:
    """Evaluate the final transform on the FULL clouds and build an AlignmentError.

    Inlier_RMSE and Fitness come from Open3D's ``evaluate_registration`` at the
    configured correspondence distance; the mean surface distance is the mean of
    nearest-neighbor distances from the transformed fragment points to the model
    points (Req 3.4). All values are in millimeters where applicable.
    """
    evaluation = o3d.pipelines.registration.evaluate_registration(
        frag_o3d, model_o3d, config.correspondence_distance_mm, transform
    )
    inlier_rmse = float(evaluation.inlier_rmse)
    fitness = float(evaluation.fitness)

    transformed = _apply_transform(fragment_points, transform)
    if transformed.shape[0] == 0 or model_points.shape[0] == 0:
        mean_surface_distance = 0.0
    else:
        distances = nearest_neighbor_distances(transformed, model_points)
        mean_surface_distance, _ = mean_and_rmse(distances)

    return AlignmentError(
        inlier_rmse_mm=inlier_rmse,
        fitness=fitness,
        mean_surface_distance_mm=float(mean_surface_distance),
    )


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an ``(N, 3)`` point array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return pts @ rotation.T + translation


def _matrix_from_precomputed(matrix: list[list[float]]) -> np.ndarray:
    """Build a 4x4 float64 array from a nested-list precomputed transform."""
    return np.asarray(matrix, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align(
    fragment_pc: PointCloud,
    full_model_pc: PointCloud,
    fragment_id: str,
    config: Config,
) -> AlignmentResult:
    """Align a Fragment onto the Full_Model, returning an :class:`AlignmentResult`.

    Resolution order:

    1. **Zero-vertex Fragment** -> skip computation entirely (Req 3.10).
    2. **Precomputed transform present** for ``fragment_id`` -> validate and
       import; raise :class:`AlignmentImportError` on an invalid matrix
       (Req 3.2, 3.8).
    3. **Otherwise compute** via FPFH + RANSAC global registration followed by
       point-to-plane ICP (Req 3.1), then evaluate metrics (Req 3.4).

    Fragments with excessive Inlier_RMSE (Req 3.5), zero fitness (Req 3.9), or an
    invalid final matrix are flagged for manual review without halting.

    Args:
        fragment_pc: The Fragment Point_Cloud (model-frame target is the model).
        full_model_pc: The Full_Model Point_Cloud.
        fragment_id: Stable identifier used for imports and error reporting.
        config: The pipeline :class:`Config`.

    Returns:
        An :class:`AlignmentResult` describing the outcome.

    Raises:
        AlignmentImportError: if a precomputed transform for ``fragment_id`` is
            present but fails the rigid-transform validity check (Req 3.8).
    """
    # (1) Zero-vertex fragment: skip alignment entirely (Req 3.10).
    if fragment_pc is None or fragment_pc.point_count == 0:
        _LOG.info(
            "alignment skipped for zero-vertex fragment", extra={"fragment_id": fragment_id}
        )
        return AlignmentResult(
            transform=None,
            error=None,
            method="skipped",
            flagged_for_review=False,
            skipped=True,
            failure_reason=None,
        )

    # (2) Import a precomputed transform when one is provided (Req 3.2, 3.8).
    precomputed = config.precomputed_transforms.get(fragment_id)
    if precomputed is not None:
        return _import_precomputed(
            fragment_pc, full_model_pc, fragment_id, config, precomputed
        )

    # (3) Compute via global registration + ICP (Req 3.1).
    return _compute_alignment(fragment_pc, full_model_pc, fragment_id, config)


def _import_precomputed(
    fragment_pc: PointCloud,
    full_model_pc: PointCloud,
    fragment_id: str,
    config: Config,
    precomputed: list[list[float]],
) -> AlignmentResult:
    """Validate and import a precomputed transform (Req 3.2, 3.8)."""
    try:
        matrix = _matrix_from_precomputed(precomputed)
    except (ValueError, TypeError) as exc:
        raise AlignmentImportError(
            fragment_id, f"precomputed transform is not a numeric 4x4 matrix: {exc}"
        ) from exc

    if not is_valid_rigid_transform(matrix):
        raise AlignmentImportError(
            fragment_id,
            "precomputed transform failed 4x4 rigid-transform validity constraints",
        )

    frag_o3d = fragment_pc.to_open3d()
    model_o3d = full_model_pc.to_open3d()
    error = _compute_alignment_error(
        frag_o3d,
        model_o3d,
        fragment_pc.points,
        full_model_pc.points,
        matrix,
        config,
    )

    flagged, reason = _review_decision(error, config)
    _LOG.info(
        "imported precomputed transform",
        extra={"fragment_id": fragment_id, "flagged_for_review": flagged},
    )
    return AlignmentResult(
        transform=matrix,
        error=error,
        method="imported",
        flagged_for_review=flagged,
        skipped=False,
        failure_reason=reason,
    )


def _compute_alignment(
    fragment_pc: PointCloud,
    full_model_pc: PointCloud,
    fragment_id: str,
    config: Config,
) -> AlignmentResult:
    """Compute alignment via FPFH + RANSAC then point-to-plane ICP (Req 3.1, 3.4)."""
    frag_o3d = fragment_pc.to_open3d()
    model_o3d = full_model_pc.to_open3d()

    # Deterministic RANSAC where the API supports seeding (best-effort).
    try:
        o3d.utility.random.seed(config.registration_params.seed)
    except Exception:  # pragma: no cover - depends on Open3D build
        pass

    # Guard against degenerate/empty inputs: flag failure rather than crash.
    if full_model_pc.point_count == 0:
        return _failed_result(
            "full model point cloud is empty; cannot compute alignment"
        )

    try:
        frag_down, frag_fpfh = _preprocess_for_features(frag_o3d, config)
        model_down, model_fpfh = _preprocess_for_features(model_o3d, config)
    except Exception as exc:  # pragma: no cover - defensive
        return _failed_result(f"feature preprocessing failed: {exc}")

    frag_down_pts = np.asarray(frag_down.points, dtype=np.float64)
    model_down_pts = np.asarray(model_down.points, dtype=np.float64)
    if frag_down_pts.shape[0] < 3 or model_down_pts.shape[0] < 3:
        return _failed_result(
            "downsampled cloud too small for registration "
            f"(fragment={frag_down_pts.shape[0]}, model={model_down_pts.shape[0]})"
        )

    try:
        coarse = _global_registration(
            frag_down, model_down, frag_fpfh, model_fpfh, config
        )
        transform = _icp_refine(frag_down, model_down, coarse, config)
    except Exception as exc:  # pragma: no cover - defensive
        return _failed_result(f"registration failed: {exc}")

    transform = np.asarray(transform, dtype=np.float64)

    error = _compute_alignment_error(
        frag_o3d,
        model_o3d,
        fragment_pc.points,
        full_model_pc.points,
        transform,
        config,
    )

    flagged = False
    reason: str | None = None

    # Validate final matrix (Req 3.3); flag but do not raise on failure.
    if not is_valid_rigid_transform(transform):
        flagged = True
        reason = "computed transform failed 4x4 rigid-transform validity constraints"

    # Zero fitness => flag + record failure, continue (Req 3.9). Do NOT raise.
    if error.fitness == 0:
        flagged = True
        reason = _combine_reason(
            reason, "global registration + ICP produced zero fitness"
        )

    # Excessive Inlier_RMSE => flag for review (Req 3.5).
    if error.inlier_rmse_mm > config.alignment_error_threshold_mm:
        flagged = True
        reason = _combine_reason(
            reason,
            "inlier_rmse "
            f"{error.inlier_rmse_mm:.6g} mm exceeds threshold "
            f"{config.alignment_error_threshold_mm:.6g} mm",
        )

    _LOG.info(
        "computed alignment",
        extra={"fragment_id": fragment_id, "flagged_for_review": flagged},
    )
    return AlignmentResult(
        transform=transform,
        error=error,
        method="computed",
        flagged_for_review=flagged,
        skipped=False,
        failure_reason=reason,
    )


def _review_decision(error: AlignmentError, config: Config) -> tuple[bool, str | None]:
    """Flag decision for imported transforms based on Inlier_RMSE (Req 3.5)."""
    if error.inlier_rmse_mm > config.alignment_error_threshold_mm:
        return True, (
            "inlier_rmse "
            f"{error.inlier_rmse_mm:.6g} mm exceeds threshold "
            f"{config.alignment_error_threshold_mm:.6g} mm"
        )
    return False, None


def _failed_result(reason: str) -> AlignmentResult:
    """Build a flagged, computed-method result carrying an identity transform.

    Used when computation cannot proceed on degenerate/tiny inputs; the fragment
    is flagged for review rather than crashing the run.
    """
    identity = np.eye(4, dtype=np.float64)
    error = AlignmentError(
        inlier_rmse_mm=0.0, fitness=0.0, mean_surface_distance_mm=0.0
    )
    return AlignmentResult(
        transform=identity,
        error=error,
        method="computed",
        flagged_for_review=True,
        skipped=False,
        failure_reason=reason,
    )


def _combine_reason(existing: str | None, addition: str) -> str:
    """Join failure reasons with a separator, preserving earlier reasons."""
    if existing:
        return f"{existing}; {addition}"
    return addition
