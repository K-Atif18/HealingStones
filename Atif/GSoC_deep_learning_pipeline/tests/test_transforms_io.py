"""Property-based tests for rigid-transform file persistence.

Exercises the round-trip guarantee of :mod:`dataset_foundation.transforms_io`
(task 8.4 / Property 8): writing a valid Rigid_Transform to a file and reading
it back reproduces every matrix element within 1e-6.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from dataset_foundation.transforms_io import read_transform, write_transform


def _rigid_transform(seed: int, translation) -> np.ndarray:
    """Build a valid 4x4 rigid transform from a proper rotation + translation.

    The rotation is a proper (det == +1) orthonormal matrix produced by
    ``scipy.spatial.transform.Rotation.random`` seeded deterministically from
    a Hypothesis-provided integer, giving reproducible coverage of SO(3).
    """
    rotation = Rotation.random(random_state=seed).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


# Feature: dataset-foundation, Property 8: Transform file round-trip
# Validates: Requirements 3.7
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    translation=st.tuples(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    ),
)
def test_transform_file_round_trip(tmp_path, seed, translation):
    """For any valid Rigid_Transform, write-then-read reproduces every element."""
    original = _rigid_transform(seed, translation)
    path = str(tmp_path / "transform.txt")

    write_transform(path, original)
    restored = read_transform(path)

    assert restored.shape == (4, 4)
    assert np.max(np.abs(restored - original)) <= 1e-6
