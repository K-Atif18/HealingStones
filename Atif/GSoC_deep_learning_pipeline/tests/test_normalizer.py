"""Property-based tests for voxel downsampling density standardization.

Covers the two metamorphic guarantees of
:func:`dataset_foundation.normalizer.voxel_downsample`:

* **Property 14 (Req 5.2).** Downsampling never increases the point count -- each
  output point corresponds to exactly one occupied voxel.
* **Property 15 (Req 5.3).** Downsampling is exactly idempotent: re-downsampling
  an already-downsampled cloud at the same voxel size leaves the point count
  unchanged (tolerance 0 points).

Validates: Requirements 5.2, 5.3
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.geometry import PointCloud
from dataset_foundation.normalizer import voxel_downsample

from .conftest import points_arrays

# Positive voxel edge lengths in millimeters, kept well away from zero to avoid
# degenerate grids while spanning a realistic range.
voxel_sizes = st.floats(
    min_value=0.1,
    max_value=50.0,
    allow_nan=False,
    allow_infinity=False,
)


# Feature: dataset-foundation, Property 14: Voxel downsampling does not increase point count (metamorphic)
@settings(max_examples=100)
@given(points=points_arrays(min_points=0, max_points=500), voxel_size_mm=voxel_sizes)
def test_voxel_downsample_does_not_increase_point_count(points, voxel_size_mm):
    """Property 14: output count <= input count (Validates: Requirements 5.2)."""
    pc = PointCloud(points=points)
    result = voxel_downsample(pc, voxel_size_mm)
    assert result.point_count <= pc.point_count


# Feature: dataset-foundation, Property 15: Voxel downsampling is idempotent
@settings(max_examples=100)
@given(points=points_arrays(min_points=0, max_points=500), voxel_size_mm=voxel_sizes)
def test_voxel_downsample_is_idempotent(points, voxel_size_mm):
    """Property 15: re-downsampling at the same voxel size is stable, tolerance 0
    (Validates: Requirements 5.3)."""
    pc = PointCloud(points=points)
    d1 = voxel_downsample(pc, voxel_size_mm)
    d2 = voxel_downsample(d1, voxel_size_mm)

    # Exact idempotence on count (tolerance 0 points).
    assert d2.point_count == d1.point_count

    # The points themselves are stable too (order-independent comparison).
    p1 = d1.points[np.lexsort(d1.points.T)]
    p2 = d2.points[np.lexsort(d2.points.T)]
    assert np.array_equal(p1, p2)


# ===========================================================================
# Coordinate normalization tests (tasks 11.5, 11.6, 11.7).
#
# These exercise dataset_foundation.normalizer.standardize and
# write_normalized. Configs are built by cloning the default Configuration and
# overriding only the normalization-relevant fields (the dataclasses are frozen
# so we use dataclasses.replace).
#
# Validates: Requirements 5.4, 5.5, 5.7, 5.8, 5.9
# ===========================================================================

import dataclasses
import os

import pytest
from hypothesis import example

from dataset_foundation.config_loader import load_config
from dataset_foundation.errors import NormalizationError, WriteError
from dataset_foundation.normalizer import standardize, write_normalized

# The default configuration lives at <repo>/config/default.yaml.
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "default.yaml",
)


@pytest.fixture(scope="module")
def base_config():
    """Load the default Configuration once for the module."""
    return load_config(_CONFIG_PATH)


# Centering offsets may be zero or negative (Req 5.5), so allow the full range.
offsets = st.tuples(
    st.floats(min_value=-1.0e4, max_value=1.0e4, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0e4, max_value=1.0e4, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0e4, max_value=1.0e4, allow_nan=False, allow_infinity=False),
)

# Enabled scaling requires a strictly-positive factor (Req 5.7).
positive_scales = st.floats(
    min_value=1.0e-3, max_value=1.0e3, allow_nan=False, allow_infinity=False
)

# Invalid scale factors: zero and negatives (Req 5.7).
invalid_scales = st.floats(
    max_value=0.0, allow_nan=False, allow_infinity=False
)

# Fragment identifiers used to name per-fragment errors.
fragment_ids = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=16,
)


# Feature: dataset-foundation, Property 16: Coordinate normalization applies and records the configured transform
@settings(max_examples=100)
@given(
    points=points_arrays(min_points=1, max_points=200),
    offset=offsets,
    scale=positive_scales,
    voxel_size_mm=voxel_sizes,
)
def test_coordinate_normalization_applies_and_records_transform(
    base_config, points, offset, scale, voxel_size_mm
):
    """Property 16: with centering + scaling enabled every output point equals
    ``(downsampled - offset) * scale`` and the applied offset/scale are recorded
    (Validates: Requirements 5.4, 5.5)."""
    pc = PointCloud(points=points)

    config = dataclasses.replace(
        base_config,
        centering_enabled=True,
        scaling_enabled=True,
        centering_offset=offset,
        scale_factor=scale,
        voxel_size_mm=voxel_size_mm,
    )

    result, applied_offset, applied_scale = standardize(pc, config, "frag")

    # Each output point == (input - offset) * scale, computed against the same
    # density-standardized cloud the implementation operates on.
    downsampled = voxel_downsample(pc, config.voxel_size_mm)
    expected = (downsampled.points - np.asarray(offset, dtype=np.float64)) * scale
    assert np.allclose(result.points, expected, rtol=1e-6, atol=1e-6)

    # The applied transform is reported exactly as configured (Req 5.5).
    assert np.allclose(applied_offset, offset)
    assert applied_scale == scale


# Feature: dataset-foundation, Property 16: Coordinate normalization applies and records the configured transform
@settings(max_examples=100)
@given(
    points=points_arrays(min_points=1, max_points=200),
    voxel_size_mm=voxel_sizes,
)
def test_coordinate_normalization_unscaled_preserves_extents(
    base_config, points, voxel_size_mm
):
    """Property 16 (identity case): with centering + scaling disabled the output
    preserves the density-standardized extents and records the identity
    transform (Validates: Requirements 5.4, 5.5)."""
    pc = PointCloud(points=points)

    config = dataclasses.replace(
        base_config,
        centering_enabled=False,
        scaling_enabled=False,
        voxel_size_mm=voxel_size_mm,
    )

    result, applied_offset, applied_scale = standardize(pc, config, "frag")

    downsampled = voxel_downsample(pc, config.voxel_size_mm)
    # Extents (min/max per axis) are unchanged when no transform is applied.
    assert np.allclose(result.points.min(axis=0), downsampled.points.min(axis=0))
    assert np.allclose(result.points.max(axis=0), downsampled.points.max(axis=0))

    # The identity transform is recorded.
    assert applied_offset == (0.0, 0.0, 0.0)
    assert applied_scale == 1.0


# Feature: dataset-foundation, Property 17: Invalid scale factor is rejected per fragment
@settings(max_examples=100)
@given(
    points=points_arrays(min_points=1, max_points=200),
    fragment_id=fragment_ids,
    scale=invalid_scales,
)
@example(
    points=np.array([[0.0, 0.0, 0.0]]),
    fragment_id="Fragment_1",
    scale=0.0,
)
@example(
    points=np.array([[1.0, 2.0, 3.0]]),
    fragment_id="Fragment_2",
    scale=-2.5,
)
def test_invalid_scale_factor_rejected_per_fragment(
    base_config, points, fragment_id, scale
):
    """Property 17: a non-positive scale factor with scaling enabled halts the
    fragment with a NormalizationError naming the Fragment_ID and the invalid
    factor; no normalized cloud is produced (Validates: Requirements 5.7)."""
    pc = PointCloud(points=points)

    config = dataclasses.replace(
        base_config,
        scaling_enabled=True,
        scale_factor=scale,
    )

    with pytest.raises(NormalizationError) as exc_info:
        standardize(pc, config, fragment_id)

    exc = exc_info.value
    # The error names the offending fragment and the invalid scale factor.
    assert exc.fragment_id == fragment_id
    assert str(fragment_id) in str(exc)
    assert exc.scale_factor == scale
    assert str(scale) in str(exc)


# --- Unit tests (task 11.7): empty-cloud and write-failure paths -----------


def test_empty_cloud_yields_empty_normalized_output(base_config):
    """Empty input cloud produces an empty normalized cloud while still
    reporting the configured transform (Validates: Requirements 5.8)."""
    config = dataclasses.replace(
        base_config,
        centering_enabled=True,
        scaling_enabled=True,
        centering_offset=(1.0, -2.0, 3.0),
        scale_factor=2.0,
    )

    result, applied_offset, applied_scale = standardize(PointCloud(), config, "frag")

    assert result.point_count == 0
    assert applied_offset == (1.0, -2.0, 3.0)
    assert applied_scale == 2.0


def test_normalized_write_failure_names_target_path(tmp_path):
    """A failed normalized write raises WriteError naming the target path
    (Validates: Requirements 5.9)."""
    # A valid, non-empty normalized cloud to write.
    pc = PointCloud(points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    # Make the parent of the target a regular file so the write cannot succeed.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    target = str(blocker / "normalized.ply")

    with pytest.raises(WriteError) as exc_info:
        write_normalized(target, pc)

    assert target in str(exc_info.value)
