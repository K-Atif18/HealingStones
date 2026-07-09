"""Tests for :mod:`dataset_foundation.metadata_manager`.

Covers:
- Property 20 (metadata round-trip): parsing a written Metadata file then
  serializing produces a record whose every field name and value equals the
  original (Req 7.5).
- Unit test 12.3: a metadata write failure names the target path and preserves
  previously written records (Req 1.8).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.errors import WriteError
from dataset_foundation.metadata_manager import (
    build_fragment_metadata,
    read_metadata,
    to_json_native,
    write_metadata,
)


# ---------------------------------------------------------------------------
# JSON-native value strategy for Property 20.
# ---------------------------------------------------------------------------

# JSON object keys are always strings, so restrict dict keys to text. Floats are
# kept finite because JSON has no representation for NaN/Infinity and Python's
# json round-trips finite floats exactly via repr().
_json_keys = st.text(min_size=0, max_size=8)

_json_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=32),
)


def _json_native(children):
    return st.one_of(
        _json_leaves,
        st.lists(children, max_size=6),
        st.dictionaries(_json_keys, children, max_size=6),
    )


# A JSON-native record is a dict whose values are arbitrarily nested JSON values.
json_records = st.dictionaries(
    _json_keys,
    st.recursive(_json_leaves, _json_native, max_leaves=25),
    max_size=8,
)


# ---------------------------------------------------------------------------
# Property 20: Metadata round-trip
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 20: Metadata round-trip
@settings(max_examples=100)
@given(record=json_records)
def test_metadata_round_trip(record):
    """Writing then reading a JSON-native record reproduces it field-for-field.

    **Validates: Requirements 7.5**
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "record.json")

        write_metadata(path, record)
        parsed = read_metadata(path)

    assert parsed == record


# Feature: dataset-foundation, Property 20: Metadata round-trip
@settings(max_examples=100)
@given(
    vertex_count=st.integers(min_value=0, max_value=10_000),
    face_count=st.integers(min_value=0, max_value=10_000),
    voxel_size_mm=st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False),
    scale_factor=st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False),
    inlier_rmse=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    fitness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_fragment_metadata_round_trip_with_numpy(
    vertex_count,
    face_count,
    voxel_size_mm,
    scale_factor,
    inlier_rmse,
    fitness,
):
    """A real fragment record built from numpy scalars/arrays/tuples round-trips.

    The record is built with numpy scalar and array inputs plus a tuple
    ``centering_offset``; after coercion via ``build_fragment_metadata`` the
    written-then-read record equals ``to_json_native(original)``.

    **Validates: Requirements 7.5**
    """
    record = build_fragment_metadata(
        fragment_id="frag_1",
        source_filename="caesar_fragment_1.ply",
        vertex_count=np.int64(vertex_count),
        face_count=np.int32(face_count),
        point_count_after_normalization=np.int64(vertex_count),
        is_empty=vertex_count == 0,
        transform_file="transforms/frag_1.txt",
        alignment_method="computed",
        voxel_size_mm=np.float64(voxel_size_mm),
        alignment_error={
            "inlier_rmse_mm": np.float64(inlier_rmse),
            "fitness": np.float32(fitness),
            "mean_surface_distance_mm": None,
        },
        centering_offset=(np.float64(1.0), np.float64(-2.0), np.float64(3.5)),
        scale_factor=np.float64(scale_factor),
        artifact_paths={"aligned": "fragments/frag_1.ply"},
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "frag_1.json")
        write_metadata(path, record)
        parsed = read_metadata(path)

    # ``record`` is already JSON-native (build_* coerces), so it equals its own
    # normalization and the parsed file.
    assert parsed == to_json_native(record)
    assert parsed == record


# ---------------------------------------------------------------------------
# Unit test 12.3: write failure names path and preserves prior records.
# ---------------------------------------------------------------------------


def test_write_failure_names_path_and_preserves_prior_record(tmp_path):
    """A failed write names the target path and leaves earlier records intact.

    First a valid record is written to path A. A second write is then forced to
    fail by passing a record containing a non-JSON-native object that
    ``json.dumps`` rejects. The resulting :class:`WriteError` must name the
    target path, and the previously written file at A must be unchanged (Req 1.8).
    """
    path_a = os.path.join(str(tmp_path), "record_a.json")
    original = {"fragment_id": "frag_1", "vertex_count": 42, "units": "mm"}

    write_metadata(path_a, original)
    assert read_metadata(path_a) == original
    before = _read_bytes(path_a)

    # Force a serialization failure: object() is returned as-is by
    # to_json_native and rejected by json.dumps with a TypeError.
    path_b = os.path.join(str(tmp_path), "record_b.json")
    bad_record = {"fragment_id": "frag_2", "payload": object()}

    with pytest.raises(WriteError) as excinfo:
        write_metadata(path_b, bad_record)

    # The error names the target path of the failed write.
    assert path_b in str(excinfo.value)
    assert excinfo.value.path == path_b

    # The failed write left no partial file at its own target.
    assert not os.path.exists(path_b)

    # The previously written record at path A is preserved byte-for-byte.
    assert os.path.exists(path_a)
    assert _read_bytes(path_a) == before
    assert read_metadata(path_a) == original


def test_write_failure_on_bad_target_path_preserves_prior_record(tmp_path):
    """A write whose target parent is a file names the path and preserves records.

    A different failure mode: the target's parent directory component is an
    existing regular file, so the atomic write cannot create the temp file /
    replace the target. The prior record at path A stays intact (Req 1.8).
    """
    path_a = os.path.join(str(tmp_path), "record_a.json")
    original = {"fragment_id": "frag_1", "vertex_count": 7}
    write_metadata(path_a, original)
    before = _read_bytes(path_a)

    # Create a regular file and try to write "underneath" it as if a directory.
    blocker = os.path.join(str(tmp_path), "not_a_dir")
    with open(blocker, "w", encoding="utf-8") as handle:
        handle.write("i am a file")

    bad_path = os.path.join(blocker, "nested", "record_b.json")
    with pytest.raises(WriteError) as excinfo:
        write_metadata(bad_path, original)

    assert bad_path in str(excinfo.value)

    # Prior record preserved.
    assert _read_bytes(path_a) == before
    assert read_metadata(path_a) == original


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()
