"""Unit and model-based tests for the shared geometry-metrics helpers.

Covers the nearest-neighbor / surface-distance utilities used by the aligner
and reconstruction validator:

* :func:`nearest_neighbor_distances` matches a brute-force ``O(N*M)`` numpy
  computation on small inputs and is always non-negative.
* :func:`mean_and_rmse` satisfies ``rmse >= mean >= 0`` and returns
  ``(0.0, 0.0)`` for empty input.
* :func:`coverage_fraction` is bounded in ``[0, 1]``, handles full/zero
  coverage, rejects negative thresholds, and returns ``0.0`` for empty inputs.

Validates: Requirements 8.2
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dataset_foundation.geometry_metrics import (
    coverage_fraction,
    mean_and_rmse,
    nearest_neighbor_distances,
)

from .conftest import points_arrays


def _brute_force_nn(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Direct O(N*M) nearest-neighbor distance reference implementation."""
    # Pairwise differences: (N, M, 3) -> distances (N, M) -> min over M.
    diff = source[:, None, :] - target[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dists, axis=1)


# ---------------------------------------------------------------------------
# nearest_neighbor_distances: brute-force cross-check + non-negativity
# ---------------------------------------------------------------------------


def test_nn_matches_brute_force_fixed_examples():
    """NN distances match a direct pairwise computation on hand-built sets."""
    rng = np.random.default_rng(1234)
    for _ in range(20):
        n = int(rng.integers(1, 12))
        m = int(rng.integers(1, 12))
        source = rng.uniform(-50.0, 50.0, size=(n, 3))
        target = rng.uniform(-50.0, 50.0, size=(m, 3))

        result = nearest_neighbor_distances(source, target)
        expected = _brute_force_nn(source, target)

        assert result.shape == (n,)
        assert np.allclose(result, expected, atol=1e-9)
        assert np.all(result >= 0.0)


def test_nn_zero_distance_when_source_in_target():
    """Points present in the target set have zero nearest-neighbor distance."""
    target = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [10.0, 0.0, 0.0]])
    source = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    result = nearest_neighbor_distances(source, target)

    assert np.allclose(result, [0.0, 0.0])


def test_nn_empty_source_returns_empty():
    """An empty source yields an empty (0,) array regardless of the target."""
    target = np.array([[0.0, 0.0, 0.0]])
    result = nearest_neighbor_distances(np.empty((0, 3)), target)

    assert result.shape == (0,)


def test_nn_empty_target_with_source_raises():
    """A non-empty source against an empty target raises ValueError."""
    source = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        nearest_neighbor_distances(source, np.empty((0, 3)))


@given(source=points_arrays(min_points=1, max_points=16),
       target=points_arrays(min_points=1, max_points=16))
def test_nn_brute_force_property(source, target):
    """Property: NN distances equal the brute-force minimum and are >= 0."""
    result = nearest_neighbor_distances(source, target)
    expected = _brute_force_nn(source, target)

    assert result.shape == (source.shape[0],)
    assert np.all(result >= 0.0)
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# mean_and_rmse: rmse >= mean >= 0 and empty -> (0.0, 0.0)
# ---------------------------------------------------------------------------


def test_mean_and_rmse_empty_returns_zeros():
    """Empty input yields (0.0, 0.0)."""
    assert mean_and_rmse(np.empty((0,))) == (0.0, 0.0)


def test_mean_and_rmse_known_values():
    """RMSE and mean match a hand-computed example."""
    distances = np.array([0.0, 3.0, 4.0])
    mean, rmse = mean_and_rmse(distances)

    assert mean == pytest.approx(7.0 / 3.0)
    assert rmse == pytest.approx(np.sqrt((0.0 + 9.0 + 16.0) / 3.0))
    assert rmse >= mean >= 0.0


def test_mean_and_rmse_equal_when_uniform():
    """RMSE equals the mean when all distances are identical."""
    distances = np.full(5, 2.5)
    mean, rmse = mean_and_rmse(distances)

    assert mean == pytest.approx(2.5)
    assert rmse == pytest.approx(2.5)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0e6,
                  allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    )
)
def test_mean_and_rmse_property(values):
    """Property: rmse >= mean >= 0 for any non-negative distance array."""
    arr = np.asarray(values, dtype=np.float64)
    mean, rmse = mean_and_rmse(arr)

    assert mean >= 0.0
    assert rmse >= 0.0
    # Allow tiny floating-point slack on the invariant boundary.
    assert rmse >= mean - 1e-9


# ---------------------------------------------------------------------------
# coverage_fraction: bounds, full/zero coverage, errors, empty inputs
# ---------------------------------------------------------------------------


def test_coverage_all_points_covered_is_one():
    """Every model point coincident with a sample point yields coverage 1.0."""
    model = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    sample = model.copy()

    assert coverage_fraction(model, sample, threshold=1e-6) == 1.0


def test_coverage_none_covered_is_zero():
    """Distant samples with a tiny threshold cover nothing -> 0.0."""
    model = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sample = np.array([[1.0e6, 1.0e6, 1.0e6]])

    assert coverage_fraction(model, sample, threshold=1e-3) == 0.0


def test_coverage_partial_within_bounds():
    """Partial coverage lands strictly inside [0, 1]."""
    model = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    sample = np.array([[0.0, 0.0, 0.0]])

    frac = coverage_fraction(model, sample, threshold=1.0)

    assert frac == pytest.approx(0.5)
    assert 0.0 <= frac <= 1.0


def test_coverage_negative_threshold_raises():
    """A negative threshold raises ValueError."""
    model = np.array([[0.0, 0.0, 0.0]])
    sample = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        coverage_fraction(model, sample, threshold=-1.0)


def test_coverage_empty_model_is_zero():
    """No model points to cover -> 0.0."""
    sample = np.array([[0.0, 0.0, 0.0]])
    assert coverage_fraction(np.empty((0, 3)), sample, threshold=1.0) == 0.0


def test_coverage_empty_sample_is_zero():
    """No sample points to cover with -> 0.0."""
    model = np.array([[0.0, 0.0, 0.0]])
    assert coverage_fraction(model, np.empty((0, 3)), threshold=1.0) == 0.0


@given(
    model=points_arrays(min_points=0, max_points=16),
    sample=points_arrays(min_points=0, max_points=16),
    threshold=st.floats(min_value=0.0, max_value=1.0e5,
                        allow_nan=False, allow_infinity=False),
)
def test_coverage_bounded_property(model, sample, threshold):
    """Property: coverage_fraction always lies in [0, 1]."""
    frac = coverage_fraction(model, sample, threshold)
    assert 0.0 <= frac <= 1.0
