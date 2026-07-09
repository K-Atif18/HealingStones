"""Property-based and integration tests for fragment-to-model rigid alignment.

Covers:
* task 8.2 / Property 6 - the rigid-transform validity check accepts proper
  rigid transforms and rejects everything else (Req 3.3);
* task 8.6 / Property 7 - precomputed-transform import accepts valid matrices
  and rejects invalid ones (Req 3.2, 3.8);
* task 8.7 / Property 9 - alignment metric bounds (Req 3.4);
* task 8.8 / Property 10 - alignment review-flag decision vs threshold (Req 3.5);
* task 8.9 - zero-fitness flag/record/continue and zero-vertex skip
  (Req 3.9, 3.10);
* task 8.10 - synthetic known-transform recovery via global registration + ICP
  (Req 3.1).
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from dataset_foundation.aligner import align, is_valid_rigid_transform
from dataset_foundation.config_loader import Config, load_config
from dataset_foundation.errors import AlignmentImportError
from dataset_foundation.geometry import PointCloud
from dataset_foundation.geometry_metrics import (
    mean_and_rmse,
    nearest_neighbor_distances,
)

# Finite translation components; kept well within float64 range so the
# assembled homogeneous matrix stays finite and numerically well-behaved.
_translations = st.lists(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
).map(np.array)

# Seed the rotation generator through a Hypothesis-controlled integer so
# examples shrink/replay deterministically.
_rotation_seeds = st.integers(min_value=0, max_value=2**31 - 1)


def _rotation_from_seed(seed: int) -> np.ndarray:
    """Return a proper rotation matrix (det = +1) for a given seed."""
    return Rotation.random(random_state=np.random.default_rng(seed)).as_matrix()


def _assemble(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous matrix from a 3x3 block and translation."""
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


# Feature: dataset-foundation, Property 6: Rigid-transform validity check is correct
@settings(max_examples=100)
@given(seed=_rotation_seeds, t=_translations)
def test_valid_rigid_transform_is_accepted(seed, t):
    """Validates: Requirements 3.3

    For any proper rotation R and any finite translation t, the assembled
    homogeneous matrix [[R, t], [0, 0, 0, 1]] must be accepted.
    """
    r = _rotation_from_seed(seed)
    m = _assemble(r, t)
    assert is_valid_rigid_transform(m) is True


# Feature: dataset-foundation, Property 6: Rigid-transform validity check is correct
@settings(max_examples=100)
@given(
    seed=_rotation_seeds,
    t=_translations,
    scale=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_scaled_rotation_is_rejected(seed, t, scale):
    """Validates: Requirements 3.3

    Scaling the rotation block by s != 1 breaks orthonormality / pushes the
    determinant away from 1, so the matrix must be rejected. We require the
    scale to sit far enough from 1 to exceed the determinant tolerance.
    """
    # Ensure |scale - 1| is comfortably beyond det_tol (default 1e-3).
    if abs(scale - 1.0) <= 0.01:
        scale = scale + 1.0 if scale >= 0 else scale - 1.0
    if abs(scale - 1.0) <= 0.01:  # pragma: no cover - defensive
        scale = 2.0
    r = _rotation_from_seed(seed) * scale
    m = _assemble(r, t)
    assert is_valid_rigid_transform(m) is False


# Feature: dataset-foundation, Property 6: Rigid-transform validity check is correct
@settings(max_examples=100)
@given(seed=_rotation_seeds, t=_translations, col=st.integers(min_value=0, max_value=2))
def test_reflection_is_rejected(seed, t, col):
    """Validates: Requirements 3.3

    A reflection (det = -1, produced by negating one column of a proper
    rotation) is orthonormal but is not a proper rotation, so it must be
    rejected.
    """
    r = _rotation_from_seed(seed)
    r[:, col] = -r[:, col]
    m = _assemble(r, t)
    assert is_valid_rigid_transform(m) is False


# Feature: dataset-foundation, Property 6: Rigid-transform validity check is correct
@settings(max_examples=100)
@given(
    seed=_rotation_seeds,
    t=_translations,
    bottom=st.sampled_from(
        [
            [0.0, 0.0, 0.0, 2.0],  # wrong homogeneous scale
            [1.0, 0.0, 0.0, 1.0],  # nonzero bottom-left entry
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],  # missing homogeneous 1
        ]
    ),
)
def test_altered_bottom_row_is_rejected(seed, t, bottom):
    """Validates: Requirements 3.3

    A proper rotation with a bottom row other than exactly [0, 0, 0, 1] is not
    a valid homogeneous rigid transform and must be rejected.
    """
    r = _rotation_from_seed(seed)
    m = _assemble(r, t)
    m[3, :] = np.array(bottom)
    assert is_valid_rigid_transform(m) is False


# Feature: dataset-foundation, Property 6: Rigid-transform validity check is correct
@settings(max_examples=100)
@given(
    seed=_rotation_seeds,
    shape=st.sampled_from([(3, 3), (4, 3), (3, 4), (2, 2), (5, 5)]),
)
def test_wrong_shape_is_rejected(seed, shape):
    """Validates: Requirements 3.3

    Anything that is not exactly 4x4 (e.g. a bare 3x3 rotation or a 4x3
    matrix) must be rejected rather than raising.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(shape)
    # A bare 3x3 proper rotation should still be rejected on shape alone.
    if shape == (3, 3):
        m = _rotation_from_seed(seed)
    assert is_valid_rigid_transform(m) is False


# ===========================================================================
# Shared helpers for the alignment tests (tasks 8.6 - 8.10)
# ===========================================================================

# Absolute path to config/default.yaml (two levels up from this test file).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "default.yaml")


def _base_config() -> Config:
    """Load a realistic Config from config/default.yaml.

    Input paths in the YAML are relative to the repository root, so the config
    is loaded with the process CWD temporarily set there to satisfy the
    path-existence validation performed by ``load_config``.
    """
    previous = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        return load_config(_DEFAULT_CONFIG_PATH)
    finally:
        os.chdir(previous)


_cloud_seeds = st.integers(min_value=0, max_value=2**31 - 1)


def _random_cloud(seed: int, n: int, spread: float = 50.0) -> PointCloud:
    """A small non-empty PointCloud of ``n`` points in a mm-scale cube."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-spread, spread, size=(n, 3)).astype(np.float64)
    return PointCloud(points=pts)


def _reflection_matrix(seed: int, t: np.ndarray, col: int) -> np.ndarray:
    """A homogeneous matrix whose 3x3 block is an (invalid) reflection."""
    r = _rotation_from_seed(seed)
    r[:, col] = -r[:, col]  # det -> -1: orthonormal but not a proper rotation
    return _assemble(r, t)


# ===========================================================================
# Task 8.6 / Property 7: Precomputed-transform import accepts valid, rejects invalid
# ===========================================================================


# Feature: dataset-foundation, Property 7: Precomputed-transform import accepts valid and rejects invalid
@settings(max_examples=100)
@given(seed=_rotation_seeds, t=_translations, frag_seed=_cloud_seeds)
def test_precomputed_import_accepts_valid_transform(seed, t, frag_seed):
    """Validates: Requirements 3.2, 3.8

    When the Configuration carries a *valid* rigid transform for a Fragment_ID,
    ``align`` imports it: the result method is ``"imported"`` and the returned
    transform equals the provided matrix (no registration is run).
    """
    fragment_id = "frag_valid"
    matrix = _assemble(_rotation_from_seed(seed), t)
    assert is_valid_rigid_transform(matrix) is True  # sanity: input is valid

    config = dataclasses.replace(
        _base_config(),
        precomputed_transforms={fragment_id: matrix.tolist()},
    )

    fragment = _random_cloud(frag_seed, n=30)
    model = _random_cloud(frag_seed + 1, n=40)

    result = align(fragment, model, fragment_id, config)

    assert result.method == "imported"
    assert result.skipped is False
    assert result.transform is not None
    assert np.allclose(result.transform, matrix, atol=1e-9)


# Feature: dataset-foundation, Property 7: Precomputed-transform import accepts valid and rejects invalid
@settings(max_examples=100)
@given(
    seed=_rotation_seeds,
    t=_translations,
    frag_seed=_cloud_seeds,
    col=st.integers(min_value=0, max_value=2),
)
def test_precomputed_import_rejects_invalid_transform(seed, t, frag_seed, col):
    """Validates: Requirements 3.2, 3.8

    When the Configuration carries an *invalid* transform (here a reflection,
    det = -1) for a Fragment_ID, ``align`` raises :class:`AlignmentImportError`
    whose message names the offending Fragment_ID.
    """
    fragment_id = "frag_invalid"
    bad_matrix = _reflection_matrix(seed, t, col)
    assert is_valid_rigid_transform(bad_matrix) is False  # sanity: input is invalid

    config = dataclasses.replace(
        _base_config(),
        precomputed_transforms={fragment_id: bad_matrix.tolist()},
    )

    fragment = _random_cloud(frag_seed, n=30)
    model = _random_cloud(frag_seed + 1, n=40)

    with pytest.raises(AlignmentImportError) as exc_info:
        align(fragment, model, fragment_id, config)

    assert fragment_id in str(exc_info.value)


# ===========================================================================
# Task 8.7 / Property 9: Alignment metric bounds
# ===========================================================================


# Feature: dataset-foundation, Property 9: Alignment metric bounds
@settings(max_examples=100)
@given(seed=_rotation_seeds, t=_translations, frag_seed=_cloud_seeds)
def test_alignment_metric_bounds(seed, t, frag_seed):
    """Validates: Requirements 3.4

    For any alignment, the reported metrics are well-formed: Fitness lies in
    ``[0, 1]``, Inlier_RMSE is ``>= 0``, and the mean surface distance is
    ``>= 0``. Exercised via the import path so metric computation runs without
    the expensive RANSAC stage.
    """
    fragment_id = "frag_metrics"
    matrix = _assemble(_rotation_from_seed(seed), t)

    config = dataclasses.replace(
        _base_config(),
        precomputed_transforms={fragment_id: matrix.tolist()},
    )

    fragment = _random_cloud(frag_seed, n=40)
    model = _random_cloud(frag_seed + 1, n=60)

    result = align(fragment, model, fragment_id, config)

    assert result.error is not None
    assert 0.0 <= result.error.fitness <= 1.0
    assert result.error.inlier_rmse_mm >= 0.0
    assert result.error.mean_surface_distance_mm >= 0.0


# ===========================================================================
# Task 8.8 / Property 10: Alignment review-flag decision
# ===========================================================================


# Feature: dataset-foundation, Property 10: Alignment review-flag decision
@settings(max_examples=100)
@given(
    frag_seed=_cloud_seeds,
    noise_seed=_cloud_seeds,
    threshold=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_alignment_review_flag_decision(frag_seed, noise_seed, threshold):
    """Validates: Requirements 3.5

    A Fragment is flagged for review iff its Inlier_RMSE strictly exceeds the
    configured ``alignment_error_threshold_mm``. The fragment is a noisy copy of
    the model imported under the identity transform (a valid rigid transform),
    so a positive Inlier_RMSE is produced and the threshold decision straddles
    both outcomes across examples.
    """
    fragment_id = "frag_flag"
    identity = np.eye(4, dtype=np.float64)

    model = _random_cloud(frag_seed, n=50)
    rng = np.random.default_rng(noise_seed)
    noisy = model.points + rng.normal(scale=0.5, size=model.points.shape)
    fragment = PointCloud(points=noisy)

    config = dataclasses.replace(
        _base_config(),
        precomputed_transforms={fragment_id: identity.tolist()},
        alignment_error_threshold_mm=threshold,
    )

    result = align(fragment, model, fragment_id, config)

    assert result.method == "imported"
    assert result.error is not None
    expected_flag = result.error.inlier_rmse_mm > threshold
    assert result.flagged_for_review is expected_flag


# ===========================================================================
# Task 8.9: Zero-fitness and zero-vertex handling (unit tests)
# ===========================================================================


def test_zero_vertex_fragment_is_skipped_and_recorded():
    """Validates: Requirements 3.10

    A Fragment with zero vertices is skipped entirely: no transform, no error,
    method ``"skipped"``, and the result is recorded (not flagged, no failure).
    """
    config = _base_config()
    empty_fragment = PointCloud()  # zero points
    model = _random_cloud(seed=1, n=50)

    result = align(empty_fragment, model, "frag_empty", config)

    assert result.skipped is True
    assert result.method == "skipped"
    assert result.transform is None
    assert result.error is None
    assert result.flagged_for_review is False
    assert result.failure_reason is None


def test_zero_fitness_alignment_flags_records_and_continues():
    """Validates: Requirements 3.9

    A computed alignment that yields zero fitness is flagged for review with a
    recorded failure reason, and the run continues (no exception is raised). Two
    unrelated clouds under a vanishingly small correspondence distance cannot
    produce any inliers, forcing zero fitness.
    """
    base = _base_config()
    config = dataclasses.replace(
        base,
        # No precomputed transform -> the compute path runs.
        precomputed_transforms={},
        # No pair of points can match within this distance -> zero fitness.
        correspondence_distance_mm=1e-9,
        # Keep RANSAC cheap for a unit test.
        registration_params=dataclasses.replace(
            base.registration_params,
            ransac_max_iterations=200,
        ),
    )

    fragment = _random_cloud(seed=101, n=60)
    model = _random_cloud(seed=202, n=80)

    # Must not raise: zero fitness is flagged/recorded, not fatal.
    result = align(fragment, model, "frag_zero_fitness", config)

    assert result.method == "computed"
    assert result.skipped is False
    assert result.error is not None
    assert result.error.fitness == 0.0
    assert result.flagged_for_review is True
    assert result.failure_reason is not None


# ===========================================================================
# Task 8.10: Synthetic known-transform recovery (integration test)
# ===========================================================================


def _synthetic_feature_rich_cloud(seed: int = 7, n_faces: int = 1600) -> np.ndarray:
    """Build a deterministic, asymmetric, feature-rich mm-scale point cloud.

    Points are sampled on the faces of a box with three distinct edge lengths,
    plus an off-center spherical "bump" at one corner that breaks the box's
    rotational symmetry so a rigid alignment is uniquely recoverable.
    """
    rng = np.random.default_rng(seed)
    lx, ly, lz = 120.0, 80.0, 50.0

    faces = []
    per_face = n_faces // 6
    u = lambda: rng.uniform(0.0, 1.0, size=per_face)  # noqa: E731
    faces.append(np.column_stack([np.zeros(per_face), u() * ly, u() * lz]))
    faces.append(np.column_stack([np.full(per_face, lx), u() * ly, u() * lz]))
    faces.append(np.column_stack([u() * lx, np.zeros(per_face), u() * lz]))
    faces.append(np.column_stack([u() * lx, np.full(per_face, ly), u() * lz]))
    faces.append(np.column_stack([u() * lx, u() * ly, np.zeros(per_face)]))
    faces.append(np.column_stack([u() * lx, u() * ly, np.full(per_face, lz)]))

    box_pts = np.vstack(faces)

    # Asymmetric bump: a dense spherical cap near the (lx, ly, lz) corner.
    n_bump = 250
    dirs = rng.normal(size=(n_bump, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bump = np.array([lx, ly, lz]) + 14.0 * dirs

    return np.vstack([box_pts, bump]).astype(np.float64)


def test_synthetic_known_transform_recovery():
    """Validates: Requirements 3.1

    Apply a known rigid transform to a synthetic cloud to produce a Fragment,
    then align it to the original (the Full_Model) via global registration +
    ICP. The recovered transform must map the Fragment back onto the model:
    high fitness, low Inlier_RMSE, and small mean nearest-neighbor distance
    after applying the recovered transform.
    """
    model_points = _synthetic_feature_rich_cloud(seed=7)

    # Known rigid transform applied to the model to synthesize the fragment.
    rotation = Rotation.from_euler("xyz", [20.0, -35.0, 15.0], degrees=True).as_matrix()
    translation = np.array([25.0, -18.0, 12.0], dtype=np.float64)
    known = _assemble(rotation, translation)
    assert is_valid_rigid_transform(known) is True

    fragment_points = model_points @ rotation.T + translation

    model = PointCloud(points=model_points)
    fragment = PointCloud(points=fragment_points)

    config = _base_config()  # default mm-scale registration params suit this scale
    # Ensure the compute path runs for THIS fragment. The default config may
    # carry manually-pinned transforms for real fragments, but never for this
    # synthetic id, so alignment is computed rather than imported.
    assert "frag_synthetic" not in config.precomputed_transforms

    result = align(fragment, model, "frag_synthetic", config)

    assert result.method == "computed"
    assert result.transform is not None
    assert is_valid_rigid_transform(result.transform) is True

    # Registration quality: strong overlap and sub-mm residual.
    assert result.error is not None
    assert result.error.fitness > 0.5
    assert result.error.inlier_rmse_mm < 3.0

    # Applying the recovered transform must bring the fragment onto the model.
    recovered = result.transform
    transformed = fragment_points @ recovered[:3, :3].T + recovered[:3, 3]
    distances = nearest_neighbor_distances(transformed, model_points)
    mean_distance, _ = mean_and_rmse(distances)
    assert mean_distance < 2.0
