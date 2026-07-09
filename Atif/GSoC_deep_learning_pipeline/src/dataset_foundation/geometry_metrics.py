"""Shared nearest-neighbor / surface-distance utilities (KD-tree based).

These helpers underpin two design components:

* the **aligner**, which reports the *mean surface distance* between an aligned
  fragment and the full model (Req 3.4), and
* the **reconstruction validator**, which reports the mean and RMSE of
  nearest-neighbor distances from the combined cloud to the model surface
  (Req 8.2) plus the *coverage fraction* of the model within a threshold
  (Req 8.3).

All distances are expressed in millimeters, matching the mm-scale convention
used end-to-end across the dataset foundation. A single :class:`scipy.spatial`
KD-tree backs every query so the aligner and validator share one well-tested
implementation.

Empty-input handling is documented per function:

* :func:`nearest_neighbor_distances` returns an empty ``(0,)`` array whenever
  there are no source points, and raises :class:`ValueError` when the target
  set is empty but sources exist (no neighbor can be defined).
* :func:`mean_and_rmse` returns ``(0.0, 0.0)`` for empty input; the RMSE is
  always ``>= mean >= 0``.
* :func:`coverage_fraction` returns ``0.0`` when there are no model points
  (nothing to cover) or no sample points (nothing covers anything).

Requirements: 3.4, 8.2, 8.3.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "nearest_neighbor_distances",
    "mean_and_rmse",
    "coverage_fraction",
]


def _as_points(data: np.ndarray, name: str) -> np.ndarray:
    """Coerce an array-like point set to a contiguous ``(N, 3)`` float64 array.

    ``None`` and empty inputs normalize to a canonical ``(0, 3)`` array so
    callers never have to special-case empty geometry. A ``PointCloud`` (or any
    object exposing a ``points`` attribute) is accepted for convenience.
    """
    if data is None:
        return np.empty((0, 3), dtype=np.float64)
    if hasattr(data, "points"):
        data = data.points
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    return np.ascontiguousarray(arr)


def nearest_neighbor_distances(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Per-source nearest-neighbor distances to the target set (mm).

    For every point in ``source_points`` this returns the Euclidean distance to
    its closest point in ``target_points``, computed via a KD-tree built on the
    target set.

    Args:
        source_points: ``(N, 3)`` array-like (or ``PointCloud``) of query points.
        target_points: ``(M, 3)`` array-like (or ``PointCloud``) of reference
            points the distances are measured to.

    Returns:
        A ``(N,)`` float64 array of non-negative nearest-neighbor distances in
        millimeters, ordered to match ``source_points``.

    Raises:
        ValueError: if ``target_points`` is empty while ``source_points`` is
            non-empty (no nearest neighbor can be defined).

    Empty-input behavior:
        When ``source_points`` is empty the result is an empty ``(0,)`` array,
        regardless of the target set.
    """
    source = _as_points(source_points, "source_points")
    target = _as_points(target_points, "target_points")

    if source.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if target.shape[0] == 0:
        raise ValueError(
            "target_points is empty; cannot compute nearest-neighbor distances "
            "for non-empty source_points"
        )

    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    # Guard against tiny negative values from floating-point noise.
    return np.maximum(distances, 0.0)


def mean_and_rmse(distances: np.ndarray) -> tuple[float, float]:
    """Mean and root-mean-square of a set of distances.

    Args:
        distances: array-like of non-negative distances (mm), e.g. the output of
            :func:`nearest_neighbor_distances`.

    Returns:
        A ``(mean, rmse)`` tuple of Python floats. Both values are ``>= 0`` and
        the invariant ``rmse >= mean`` always holds (RMSE is the quadratic mean,
        which dominates the arithmetic mean).

    Empty-input behavior:
        Returns ``(0.0, 0.0)`` when ``distances`` is empty, so downstream metric
        records stay well-formed without special-casing.
    """
    arr = np.asarray(distances, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0

    mean = float(np.mean(arr))
    rmse = float(np.sqrt(np.mean(np.square(arr))))
    # RMSE >= mean mathematically; clamp any floating-point drift.
    if rmse < mean:
        rmse = mean
    return mean, rmse


def coverage_fraction(
    model_points: np.ndarray, sample_points: np.ndarray, threshold: float
) -> float:
    """Fraction of model points covered by a sample set within ``threshold``.

    A model point is *covered* when at least one sample point lies within
    ``threshold`` millimeters of it. Coverage is the number of covered model
    points divided by the total number of model points, and is guaranteed to lie
    in the closed interval ``[0.0, 1.0]`` (Req 8.3).

    Args:
        model_points: ``(N, 3)`` array-like (or ``PointCloud``) of surface points
            whose coverage is being measured.
        sample_points: ``(M, 3)`` array-like (or ``PointCloud``) of points that
            may cover the model surface (e.g. the combined aligned cloud).
        threshold: Non-negative coverage radius in millimeters.

    Returns:
        The covered fraction as a Python float in ``[0.0, 1.0]``.

    Raises:
        ValueError: if ``threshold`` is negative.

    Empty-input behavior:
        Returns ``0.0`` when there are no model points (nothing to cover) or no
        sample points (nothing can cover the model).
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    model = _as_points(model_points, "model_points")
    sample = _as_points(sample_points, "sample_points")

    if model.shape[0] == 0 or sample.shape[0] == 0:
        return 0.0

    tree = cKDTree(sample)
    # Distance from each model point to its nearest sample point.
    distances, _ = tree.query(model, k=1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    covered = int(np.count_nonzero(distances <= threshold))
    return covered / model.shape[0]
