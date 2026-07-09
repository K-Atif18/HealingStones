"""Shared pytest fixtures and Hypothesis strategies for Phase 2 tests.

Registers a Hypothesis profile running a minimum of 100 examples per property
test (per the design's correctness-properties convention) and exposes a small
set of shared strategies that later test tasks build upon.

Strategies are intentionally minimal and dependency-free: they yield plain
NumPy arrays and JSON-native dicts rather than the (not-yet-implemented)
``Config``/``Patch`` dataclasses, so this module imports cleanly during
scaffolding. Later tasks extend these strategies and map them onto the real
dataclasses as those modules land.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Hypothesis profile: all property-based tests run >= 100 examples.
# ---------------------------------------------------------------------------
settings.register_profile(
    "patch_generation",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("patch_generation")


# ---------------------------------------------------------------------------
# Coordinate primitives (mm scale, Full_Model frame).
# ---------------------------------------------------------------------------

# Finite float coordinates at a realistic mm scale for the Caesar model.
finite_coords = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _rows(n: int):
    """Strategy for an ``(n, 3)`` float64 array of finite coordinates."""

    return st.lists(
        st.tuples(finite_coords, finite_coords, finite_coords),
        min_size=n,
        max_size=n,
    ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))


# ---------------------------------------------------------------------------
# Point clouds with matched normals (and the mismatched counterpart).
# ---------------------------------------------------------------------------


def point_clouds_with_normals(min_points: int = 1, max_points: int = 64):
    """Strategy for ``(points, normals)`` with matched ``(N, 3)`` shapes."""

    return st.integers(min_value=min_points, max_value=max_points).flatmap(
        lambda n: st.tuples(_rows(n), _rows(n))
    )


def mismatched_point_normal_counts(min_points: int = 1, max_points: int = 32):
    """Strategy for ``(points, normals)`` whose row counts deliberately differ."""

    counts = st.tuples(
        st.integers(min_value=min_points, max_value=max_points),
        st.integers(min_value=min_points, max_value=max_points),
    ).filter(lambda nm: nm[0] != nm[1])
    return counts.flatmap(lambda nm: st.tuples(_rows(nm[0]), _rows(nm[1])))


def dense_point_clouds(min_points: int = 8, max_points: int = 96):
    """Cap-triggering dense clouds: many points packed in a tiny mm-scale box.

    Useful for exercising Max_Patch_Points capping, since a small radius still
    captures more neighbors than the cap.
    """

    small = st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    return st.integers(min_value=min_points, max_value=max_points).flatmap(
        lambda n: st.lists(
            st.tuples(small, small, small),
            min_size=n,
            max_size=n,
        ).map(lambda r: np.asarray(r, dtype=np.float64).reshape((n, 3)))
    )


# ---------------------------------------------------------------------------
# Sampling / extraction parameters.
# ---------------------------------------------------------------------------


def center_indices(n: int, min_size: int = 1, max_size: int | None = None):
    """Strategy for a list of distinct source indices into an ``n``-point cloud."""

    upper = n if max_size is None else min(max_size, n)
    return st.lists(
        st.integers(min_value=0, max_value=max(0, n - 1)),
        min_size=min(min_size, n),
        max_size=max(min(min_size, n), upper),
        unique=True,
    )


def radii(min_value: float = 0.1, max_value: float = 100.0):
    """Strategy for a positive ``patch_radius_mm`` value."""

    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def max_patch_points(min_value: int = 1, max_value: int = 256):
    """Strategy for a valid ``max_patch_points`` cap (>= 1)."""

    return st.integers(min_value=min_value, max_value=max_value)


def source_index_sets(min_index: int = 0, max_index: int = 200, max_size: int = 32):
    """Strategy for a set of source indices (as used by overlap/coverage)."""

    return st.sets(
        st.integers(min_value=min_index, max_value=max_index),
        min_size=0,
        max_size=max_size,
    )


# ---------------------------------------------------------------------------
# Patch collections (dict form; mapped onto the Patch dataclass by later tasks).
# ---------------------------------------------------------------------------


def patches(fragment_id: str = "fragment_test", max_points: int = 16):
    """Strategy for a single patch represented as a JSON-native-ish dict.

    Includes the fields overlap/coverage/size analysis rely on
    (``source_indices``) plus identity fields; coordinate arrays are omitted
    here and added by tasks that need them.
    """

    return st.builds(
        lambda pid, cidx, src: {
            "patch_id": pid,
            "fragment_id": fragment_id,
            "center_index": cidx,
            "source_indices": sorted(set(src) | {cidx}),
        },
        st.integers(min_value=0, max_value=10_000),
        st.integers(min_value=0, max_value=200),
        st.sets(st.integers(min_value=0, max_value=200), min_size=0, max_size=max_points),
    )


def patch_collections(min_size: int = 0, max_size: int = 12):
    """Strategy for a collection (list) of patch dicts within one Fragment."""

    return st.lists(patches(), min_size=min_size, max_size=max_size)


def patch_size_lists(min_size: int = 0, max_size: int = 20):
    """Strategy for a list of per-patch point counts (>= 1 each)."""

    return st.lists(
        st.integers(min_value=1, max_value=256),
        min_size=min_size,
        max_size=max_size,
    )


# ---------------------------------------------------------------------------
# Configurations (valid + deliberately invalidated), as field dicts.
# ---------------------------------------------------------------------------


def valid_configurations():
    """Strategy for a valid Phase 2 Configuration as a field dict.

    Enforces the design constraints: positive radius, cap >= 1, coverage in
    [0, 1], min size >= 1, image dims >= 1024, and exactly one of
    ``target_center_count`` / ``target_center_density_per_mm2``.
    """

    base = st.fixed_dictionaries(
        {
            "dataset_dir": st.just("dataset/"),
            "patch_dir": st.just("patches/"),
            "patch_radius_mm": st.floats(
                min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
            "max_patch_points": st.integers(min_value=1, max_value=1024),
            "fps_seed": st.integers(min_value=0, max_value=2**31 - 1),
            "coverage_threshold": st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            "min_patch_size": st.integers(min_value=1, max_value=64),
            "image_output_path": st.just("patches/visualization.png"),
            "image_width": st.integers(min_value=1024, max_value=4096),
            "image_height": st.integers(min_value=1024, max_value=4096),
        }
    )

    def _add_center_spec(cfg, use_count, count, density):
        cfg = dict(cfg)
        if use_count:
            cfg["target_center_count"] = count
            cfg["target_center_density_per_mm2"] = None
        else:
            cfg["target_center_count"] = None
            cfg["target_center_density_per_mm2"] = density
        return cfg

    return st.builds(
        _add_center_spec,
        base,
        st.booleans(),
        st.integers(min_value=1, max_value=100_000),
        st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False),
    )


# (parameter name, mutator) pairs producing an invalid Configuration.
_INVALIDATORS = [
    ("patch_radius_mm", lambda c: {**c, "patch_radius_mm": 0.0}),
    ("patch_radius_mm", lambda c: {**c, "patch_radius_mm": -1.0}),
    ("max_patch_points", lambda c: {**c, "max_patch_points": 0}),
    ("coverage_threshold", lambda c: {**c, "coverage_threshold": 1.5}),
    ("coverage_threshold", lambda c: {**c, "coverage_threshold": -0.1}),
    ("min_patch_size", lambda c: {**c, "min_patch_size": 0}),
    ("image_width", lambda c: {**c, "image_width": 1023}),
    ("image_height", lambda c: {**c, "image_height": 1023}),
    (
        "target_center_count",
        lambda c: {**c, "target_center_count": 1000, "target_center_density_per_mm2": 0.5},
    ),
    (
        "target_center_count",
        lambda c: {**c, "target_center_count": None, "target_center_density_per_mm2": None},
    ),
]


def invalid_configurations():
    """Strategy for ``(offending_parameter, config_dict)`` invalid Configurations."""

    return st.tuples(valid_configurations(), st.sampled_from(_INVALIDATORS)).map(
        lambda vc: (vc[1][0], vc[1][1](vc[0]))
    )


# ---------------------------------------------------------------------------
# Validation outcomes.
# ---------------------------------------------------------------------------


def validation_outcomes():
    """Strategy for a per-Fragment validation outcome as a field dict."""

    return st.fixed_dictionaries(
        {
            "coverage_pass": st.booleans(),
            "size_pass": st.booleans(),
            "coverage_fraction": st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            "uncovered_count": st.integers(min_value=0, max_value=10_000),
            "offending_bound": st.sampled_from([None, "min_patch_size", "max_patch_points"]),
        }
    )
