"""Tests for :mod:`patch_generation.input_loader`.

Covers two tasks that share this file:

* Task 4.2 (property) -- **Property 3: Loaded normals count matches points and
  mismatches are rejected** (Validates Requirements 1.2, 1.4).
* Task 4.3 (unit) -- input error and empty-fragment paths
  (Validates Requirements 1.3, 1.5, 1.6).

The loader reads BOTH points and per-point normals from the density-standardized
``normalized/<id>.ply`` cloud (a documented deviation from the design's separate
``normals/`` directory -- see the ``input_loader`` module docstring). These tests
therefore stage a synthetic ``dataset/`` on disk (``metadata/dataset.json`` plus
``normalized/<id>.ply``) and exercise :func:`load_fragment` against it.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.geometry import PointCloud
from dataset_foundation.metadata_manager import write_metadata
from dataset_foundation.ply_io import write_point_cloud

from patch_generation.input_loader import load_fragment

_FRAGMENT_ID = "fragment_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Finite float coordinates at a realistic mm scale.
_coord = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _rows(n: int):
    """Strategy for an ``(n, 3)`` float64 array of finite coordinates."""
    return st.lists(
        st.tuples(_coord, _coord, _coord),
        min_size=n,
        max_size=n,
    ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))


def _stage_dataset(
    dataset_dir: str,
    fragment_id: str,
    point_cloud: PointCloud | None = None,
    *,
    write_ply: bool = True,
) -> str:
    """Write a minimal synthetic Phase 1 ``dataset/`` and return the .ply path.

    Always enumerates ``fragment_id`` in ``metadata/dataset.json``. Writes
    ``normalized/<id>.ply`` from ``point_cloud`` only when ``write_ply`` is True
    (set False to exercise the missing-input path).
    """
    normalized_path = os.path.join(dataset_dir, "normalized", f"{fragment_id}.ply")
    if write_ply:
        os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
        assert point_cloud is not None
        write_point_cloud(normalized_path, point_cloud)
    write_metadata(
        os.path.join(dataset_dir, "metadata", "dataset.json"),
        {"fragment_ids": [fragment_id]},
    )
    return normalized_path


# ---------------------------------------------------------------------------
# Task 4.2 -- Property 3
# ---------------------------------------------------------------------------

# Feature: patch-generation, Property 3: Loaded normals count matches points and mismatches are rejected
# Validates: Requirements 1.2, 1.4
@settings(max_examples=100)
@given(n=st.integers(min_value=1, max_value=64), data=st.data())
def test_property3_matched_normals_count(n: int, data) -> None:
    """A cloud with N points and N normals loads with matched counts (Req 1.2)."""
    points = data.draw(_rows(n))
    normals = data.draw(_rows(n))
    with tempfile.TemporaryDirectory() as dataset_dir:
        _stage_dataset(dataset_dir, _FRAGMENT_ID, PointCloud(points=points, normals=normals))

        loaded = load_fragment(dataset_dir, _FRAGMENT_ID)

        assert loaded.error is None
        assert loaded.skipped is False
        assert loaded.point_count == n
        # Req 1.2: number of loaded normals equals the number of loaded points.
        assert loaded.normals.shape[0] == loaded.points.shape[0] == n


# Feature: patch-generation, Property 3: Loaded normals count matches points and mismatches are rejected
# Validates: Requirements 1.2, 1.4
@settings(max_examples=100)
@given(n=st.integers(min_value=1, max_value=64), data=st.data())
def test_property3_count_mismatch_is_rejected(n: int, data) -> None:
    """A non-empty cloud carrying no normals is a 0 != N mismatch (Req 1.4).

    ``PointCloud`` enforces one-normal-per-point at construction, so a genuine
    count mismatch is simulated by writing a normalized cloud with NO normals
    for a non-empty point set: the loader sees 0 normals against N points and
    rejects the Fragment.
    """
    points = data.draw(_rows(n))
    with tempfile.TemporaryDirectory() as dataset_dir:
        _stage_dataset(dataset_dir, _FRAGMENT_ID, PointCloud(points=points, normals=None))

        loaded = load_fragment(dataset_dir, _FRAGMENT_ID)

        # Req 1.4: mismatch -> error naming the Fragment_ID, no usable points.
        assert loaded.error is not None
        assert _FRAGMENT_ID in loaded.error
        assert loaded.points.shape[0] == 0
        assert loaded.normals.shape[0] == 0
        assert loaded.skipped is False


# ---------------------------------------------------------------------------
# Task 4.3 -- unit tests for error and empty-fragment paths
# ---------------------------------------------------------------------------


def test_missing_input_file_records_error() -> None:
    """Missing normalized/<id>.ply -> error naming the path + Fragment_ID (Req 1.3)."""
    fragment_id = "fragment_missing"
    with tempfile.TemporaryDirectory() as dataset_dir:
        # Enumerate the id but deliberately do not write its .ply file.
        normalized_path = _stage_dataset(dataset_dir, fragment_id, write_ply=False)

        loaded = load_fragment(dataset_dir, fragment_id)

        assert loaded.error is not None
        assert fragment_id in loaded.error
        assert normalized_path in loaded.error
        # No patches producible: no usable points, not merely skipped.
        assert loaded.points.shape[0] == 0
        assert loaded.skipped is False


def test_empty_fragment_is_skipped() -> None:
    """A zero-point normalized cloud is skipped, not errored (Req 1.5)."""
    fragment_id = "fragment_empty"
    with tempfile.TemporaryDirectory() as dataset_dir:
        empty = PointCloud(points=np.empty((0, 3), dtype=np.float64))
        _stage_dataset(dataset_dir, fragment_id, empty)

        loaded = load_fragment(dataset_dir, fragment_id)

        assert loaded.skipped is True
        assert loaded.point_count == 0
        assert loaded.error is None
        assert loaded.points.shape[0] == 0


def test_loaded_point_count_is_recorded() -> None:
    """The loaded point count is recorded per Fragment (Req 1.6)."""
    fragment_id = "fragment_counted"
    n = 17
    rng = np.random.RandomState(0)
    points = rng.rand(n, 3).astype(np.float64)
    normals = rng.rand(n, 3).astype(np.float64)
    with tempfile.TemporaryDirectory() as dataset_dir:
        _stage_dataset(dataset_dir, fragment_id, PointCloud(points=points, normals=normals))

        loaded = load_fragment(dataset_dir, fragment_id)

        assert loaded.error is None
        assert loaded.point_count == n
        assert loaded.points.shape[0] == n
