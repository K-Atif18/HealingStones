"""Property-based and unit tests for the reconstruction validation checkpoint.

Covers the combined-cloud construction, the combined-to-model distance/coverage
metrics, and the pass/fail decision with offender identification implemented in
``dataset_foundation.reconstruction_validator``:

* Property 23 - the combined cloud excludes empty fragments so its point count
  equals the sum of the non-empty fragment point counts (Req 8.1).
* Property 24 - reconstruction distance metrics are well-formed: mean/RMSE are
  non-negative, ``rmse >= mean``, and they match a brute-force nearest-neighbor
  computation on small inputs (Req 8.2).
* Property 25 - the coverage fraction is bounded in ``[0, 1]`` (Req 8.3).
* Property 26 - the pass/fail decision matches ``rmse <= threshold`` and the
  offending fragments are exactly those whose Inlier_RMSE exceeds the alignment
  threshold (Req 8.4, 8.5).
* Unit - an empty combined cloud reports failure without computing metrics
  (Req 8.6).

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.config_loader import Config, load_config
from dataset_foundation.geometry import PointCloud
from dataset_foundation.reconstruction_validator import (
    ReconstructionResult,
    build_combined_cloud,
    compute_reconstruction_metrics,
    validate_reconstruction,
)

from .conftest import points_arrays

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = "config/default.yaml"


def _base_config() -> Config:
    """Load the default Configuration.

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


def _brute_force_nn(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Direct O(N*M) nearest-neighbor distance reference implementation."""
    diff = source[:, None, :] - target[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dists, axis=1)


# ---------------------------------------------------------------------------
# Feature: dataset-foundation, Property 23: Combined cloud excludes empty
# fragments
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    fragment_points=st.lists(
        points_arrays(min_points=0, max_points=12),
        min_size=0,
        max_size=8,
    )
)
def test_property_23_combined_excludes_empty_sequence(fragment_points):
    """Property 23: for a sequence input, the combined point count equals the
    sum of the non-empty fragment point counts (Validates: Requirements 8.1)."""
    fragments = [PointCloud(points=pts) for pts in fragment_points]
    expected = sum(frag.point_count for frag in fragments if not frag.is_empty)

    combined = build_combined_cloud(fragments)

    assert combined.point_count == expected


@settings(max_examples=100)
@given(
    fragment_points=st.lists(
        points_arrays(min_points=0, max_points=12),
        min_size=0,
        max_size=8,
    )
)
def test_property_23_combined_excludes_empty_mapping(fragment_points):
    """Property 23: the dict-keyed input excludes empty fragments identically
    to the sequence input (Validates: Requirements 8.1)."""
    fragments = {
        f"fragment_{i}": PointCloud(points=pts)
        for i, pts in enumerate(fragment_points)
    }
    expected = sum(
        frag.point_count for frag in fragments.values() if not frag.is_empty
    )

    combined = build_combined_cloud(fragments)

    assert combined.point_count == expected


# ---------------------------------------------------------------------------
# Feature: dataset-foundation, Property 24: Reconstruction distance metrics are
# well-formed
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    combined_points=points_arrays(min_points=1, max_points=16),
    model_points=points_arrays(min_points=1, max_points=16),
    correspondence_distance_mm=st.floats(
        min_value=1e-6, max_value=1.0e5, allow_nan=False, allow_infinity=False
    ),
)
def test_property_24_distance_metrics_well_formed(
    combined_points, model_points, correspondence_distance_mm
):
    """Property 24: mean/RMSE are non-negative with ``rmse >= mean`` and match a
    brute-force nearest-neighbor computation; coverage is bounded in [0, 1]
    (Validates: Requirements 8.2)."""
    combined = PointCloud(points=combined_points)
    model = PointCloud(points=model_points)

    mean, rmse, coverage = compute_reconstruction_metrics(
        combined, model, correspondence_distance_mm
    )

    assert mean >= 0.0
    assert rmse >= 0.0
    assert rmse >= mean - 1e-9
    assert 0.0 <= coverage <= 1.0

    # Brute-force cross-check on the small combined -> model distances.
    expected_distances = _brute_force_nn(combined_points, model_points)
    expected_mean = float(np.mean(expected_distances))
    expected_rmse = float(np.sqrt(np.mean(np.square(expected_distances))))

    assert mean == pytest.approx(expected_mean, rel=1e-6, abs=1e-6)
    assert rmse == pytest.approx(expected_rmse, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# Feature: dataset-foundation, Property 25: Coverage fraction is bounded in
# [0, 1]
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    combined_points=points_arrays(min_points=1, max_points=16),
    model_points=points_arrays(min_points=1, max_points=16),
    correspondence_distance_mm=st.floats(
        min_value=0.0, max_value=1.0e5, allow_nan=False, allow_infinity=False
    ),
)
def test_property_25_coverage_fraction_bounded(
    combined_points, model_points, correspondence_distance_mm
):
    """Property 25: the coverage fraction always lies in [0, 1]
    (Validates: Requirements 8.3)."""
    combined = PointCloud(points=combined_points)
    model = PointCloud(points=model_points)

    _, _, coverage = compute_reconstruction_metrics(
        combined, model, max(correspondence_distance_mm, 1e-6)
    )

    assert 0.0 <= coverage <= 1.0


# ---------------------------------------------------------------------------
# Feature: dataset-foundation, Property 26: Reconstruction pass/fail decision
# and offender identification
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    combined_points=points_arrays(min_points=1, max_points=16),
    model_points=points_arrays(min_points=1, max_points=16),
    threshold_scale=st.floats(
        min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_26_pass_iff_rmse_within_threshold(
    combined_points, model_points, threshold_scale
):
    """Property 26: for a non-empty combined cloud, the checkpoint passes iff the
    measured RMSE is within the reconstruction threshold. The threshold is swept
    around the measured RMSE via ``threshold_scale`` (Validates: Requirements
    8.4)."""
    base = _base_config()
    combined = PointCloud(points=combined_points)
    model = PointCloud(points=model_points)

    _, measured_rmse, _ = compute_reconstruction_metrics(
        combined, model, base.correspondence_distance_mm
    )
    # A strictly positive threshold that straddles the measured RMSE as the
    # scale moves through 1.0; config validation requires threshold > 0.
    threshold_mm = max(measured_rmse * threshold_scale, 1e-9)

    config = dataclasses.replace(
        base, reconstruction_error_threshold_mm=threshold_mm
    )

    result = validate_reconstruction(
        {"fragment_0": combined}, model, config
    )

    assert result.empty is False
    assert result.passed == (result.rmse_mm <= result.threshold_mm)


@settings(max_examples=100)
@given(
    inlier_rmses=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=7,
    ),
    alignment_threshold_mm=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_26_failure_offenders_exact(inlier_rmses, alignment_threshold_mm):
    """Property 26: on a guaranteed failure, the offending fragment ids are
    exactly those whose Inlier_RMSE exceeds the alignment threshold
    (Validates: Requirements 8.5)."""
    base = _base_config()

    # Combined cloud placed far from the model so RMSE is large; a tiny
    # reconstruction threshold guarantees a failure and exercises the offender
    # branch.
    combined = PointCloud(points=np.zeros((4, 3), dtype=np.float64))
    model = PointCloud(
        points=np.full((4, 3), 1.0e6, dtype=np.float64)
    )

    per_fragment_inlier_rmse = {
        f"fragment_{i}": rmse for i, rmse in enumerate(inlier_rmses)
    }
    config = dataclasses.replace(
        base,
        reconstruction_error_threshold_mm=1e-9,
        alignment_error_threshold_mm=alignment_threshold_mm,
    )

    result = validate_reconstruction(
        {"combined": combined},
        model,
        config,
        per_fragment_inlier_rmse=per_fragment_inlier_rmse,
    )

    assert result.passed is False
    expected_offenders = {
        fid
        for fid, rmse in per_fragment_inlier_rmse.items()
        if rmse > alignment_threshold_mm
    }
    assert set(result.offending_fragment_ids) == expected_offenders


@settings(max_examples=100)
@given(
    combined_points=points_arrays(min_points=1, max_points=16),
    inlier_rmses=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=7,
    ),
)
def test_property_26_pass_has_no_offenders(combined_points, inlier_rmses):
    """Property 26: on a pass the offender list is always empty, regardless of
    per-fragment Inlier_RMSE values (Validates: Requirements 8.5)."""
    base = _base_config()

    # Combined cloud coincident with the model so RMSE is ~0; a generous
    # threshold guarantees a pass.
    combined = PointCloud(points=combined_points)
    model = PointCloud(points=combined_points.copy())

    per_fragment_inlier_rmse = {
        f"fragment_{i}": rmse for i, rmse in enumerate(inlier_rmses)
    }
    config = dataclasses.replace(
        base,
        reconstruction_error_threshold_mm=1.0e6,
        alignment_error_threshold_mm=1e-9,
    )

    result = validate_reconstruction(
        {"combined": combined},
        model,
        config,
        per_fragment_inlier_rmse=per_fragment_inlier_rmse,
    )

    assert result.passed is True
    assert result.offending_fragment_ids == []


# ---------------------------------------------------------------------------
# Unit test (task 14.8): empty combined cloud fails without computing metrics
# ---------------------------------------------------------------------------


def test_empty_combined_cloud_reports_failure_without_metrics():
    """An all-empty (or empty) fragment set yields an empty-failure result with
    no metrics computed (Validates: Requirements 8.6)."""
    base = _base_config()
    model = PointCloud(points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    # All fragments empty -> combined cloud is empty.
    aligned = {"fragment_0": PointCloud(), "fragment_1": PointCloud()}

    result = validate_reconstruction(aligned, model, base)

    assert isinstance(result, ReconstructionResult)
    assert result.empty is True
    assert result.passed is False
    assert result.mean_distance_mm is None
    assert result.rmse_mm is None
    assert result.coverage_fraction is None
    assert result.offending_fragment_ids == []


def test_empty_fragment_list_reports_failure_without_metrics():
    """An empty fragment list is treated the same as an all-empty set
    (Validates: Requirements 8.6)."""
    base = _base_config()
    model = PointCloud(points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    result = validate_reconstruction([], model, base)

    assert result.empty is True
    assert result.passed is False
    assert result.mean_distance_mm is None
    assert result.rmse_mm is None
    assert result.coverage_fraction is None
