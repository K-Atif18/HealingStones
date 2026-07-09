"""Shared pytest fixtures and Hypothesis configuration for the test suite.

Registers a Hypothesis profile running a minimum of 100 examples per property
test (per the design's correctness-properties convention) and exposes a small
set of shared strategies that later test tasks build upon.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Hypothesis profile: all property-based tests run >= 100 examples.
# ---------------------------------------------------------------------------
settings.register_profile(
    "dataset_foundation",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dataset_foundation")


# ---------------------------------------------------------------------------
# Shared strategies (kept minimal; expanded by later test tasks).
# ---------------------------------------------------------------------------

# Finite float coordinates at a realistic mm scale for the Caesar model.
finite_coords = st.floats(
    min_value=-1.0e5,
    max_value=1.0e5,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def points_arrays(min_points: int = 0, max_points: int = 64):
    """Strategy for (N, 3) float64 point arrays in millimeters."""

    return st.integers(min_value=min_points, max_value=max_points).flatmap(
        lambda n: st.lists(
            st.tuples(finite_coords, finite_coords, finite_coords),
            min_size=n,
            max_size=n,
        ).map(lambda rows: np.asarray(rows, dtype=np.float64).reshape((n, 3)))
    )


def filenames(min_size: int = 1, max_size: int = 8):
    """Strategy for distinct, filesystem-safe PLY-style base filenames."""

    stem = st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-",
        ),
        min_size=1,
        max_size=16,
    )
    return st.lists(
        stem.map(lambda s: f"{s}.ply"),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )
