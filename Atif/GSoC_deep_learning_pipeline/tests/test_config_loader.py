"""Property-based tests for :mod:`dataset_foundation.config_loader`.

Covers the Configuration round-trip property (Property 1) which validates that
``dump_config`` and ``load_config`` are exact inverses for any valid
Configuration (Requirements 9.1, 9.3). Later tasks add the validation property
(Property 2) to this same file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.config_loader import (
    Config,
    NormalEstimationParams,
    RegistrationParams,
    dump_config,
    load_config,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Floats restricted to a few decimals and a realistic mm-scale magnitude so
# they survive a YAML dump/parse round-trip exactly (no NaN/inf, no precision
# surprises from arbitrary-width doubles).
_POSITIVE = st.floats(
    min_value=1.0e-3, max_value=1.0e6, allow_nan=False, allow_infinity=False
).map(lambda x: round(x, 4)).filter(lambda x: x > 0)

# ransac_confidence must lie in (0, 1].
_CONFIDENCE = st.floats(
    min_value=1.0e-3, max_value=1.0, allow_nan=False, allow_infinity=False
).map(lambda x: round(x, 4)).filter(lambda x: 0 < x <= 1)

# centering_offset components may be zero or negative (Req 5.5).
_ANY_FINITE = st.floats(
    min_value=-1.0e5, max_value=1.0e5, allow_nan=False, allow_infinity=False
).map(lambda x: round(x, 4))

_POSITIVE_INT = st.integers(min_value=1, max_value=100_000)
_IMAGE_DIM = st.integers(min_value=1024, max_value=8192)


def _matrix():
    """Strategy for a 4x4 matrix of round-trippable floats."""
    return st.lists(
        st.lists(_ANY_FINITE, min_size=4, max_size=4),
        min_size=4,
        max_size=4,
    )


@st.composite
def config_specs(draw):
    """Build the field values for a valid Config (excluding on-disk paths).

    Path fields (``full_model_path``, ``fragment_paths``, ``dataset_dir``,
    ``image_output_path``) are filled in by the test because ``load_config``
    requires the input PLY paths to exist on disk.
    """
    normal_params = NormalEstimationParams(
        search_radius_mm=draw(_POSITIVE),
        max_neighbors=draw(_POSITIVE_INT),
        orientation_neighbors=draw(_POSITIVE_INT),
    )
    registration_params = RegistrationParams(
        feature_voxel_size_mm=draw(_POSITIVE),
        fpfh_radius_mm=draw(_POSITIVE),
        fpfh_max_neighbors=draw(_POSITIVE_INT),
        normal_radius_mm=draw(_POSITIVE),
        ransac_distance_mm=draw(_POSITIVE),
        ransac_max_iterations=draw(_POSITIVE_INT),
        ransac_confidence=draw(_CONFIDENCE),
        icp_max_distance_mm=draw(_POSITIVE),
        icp_max_iterations=draw(_POSITIVE_INT),
        seed=draw(_POSITIVE_INT),
    )

    # precomputed_transforms: possibly empty, else a few Fragment_ID -> 4x4.
    transform_ids = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    whitelist_characters="_-",
                ),
                min_size=1,
                max_size=12,
            ),
            min_size=0,
            max_size=3,
            unique=True,
        )
    )
    precomputed_transforms = {fid: draw(_matrix()) for fid in transform_ids}

    return {
        "num_fragments": draw(st.integers(min_value=0, max_value=5)),
        "voxel_size_mm": draw(_POSITIVE),
        "target_density_pts_per_mm3": draw(_POSITIVE),
        "normal_params": normal_params,
        "registration_params": registration_params,
        "alignment_error_threshold_mm": draw(_POSITIVE),
        "correspondence_distance_mm": draw(_POSITIVE),
        "reconstruction_error_threshold_mm": draw(_POSITIVE),
        "centering_enabled": draw(st.booleans()),
        "scaling_enabled": draw(st.booleans()),
        # scale_factor kept > 0 so the config is accepted regardless of the
        # scaling_enabled flag.
        "scale_factor": draw(_POSITIVE),
        "centering_offset": (draw(_ANY_FINITE), draw(_ANY_FINITE), draw(_ANY_FINITE)),
        "precomputed_transforms": precomputed_transforms,
        "image_width": draw(_IMAGE_DIM),
        "image_height": draw(_IMAGE_DIM),
    }


# ---------------------------------------------------------------------------
# Property 1: Configuration round-trip
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 1: Configuration round-trip
@settings(max_examples=100)
@given(spec=config_specs())
def test_config_round_trip(spec):
    """Serializing then parsing any valid Configuration reproduces an equal one.

    **Validates: Requirements 9.1, 9.3**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # load_config requires the input PLY paths to exist on disk, so create
        # real (empty) files for the full model and each fragment.
        full_model_path = base / "full_model.ply"
        full_model_path.write_bytes(b"")

        fragment_paths = []
        for index in range(spec["num_fragments"]):
            fragment = base / f"fragment_{index}.ply"
            fragment.write_bytes(b"")
            fragment_paths.append(str(fragment))

        original = Config(
            full_model_path=str(full_model_path),
            fragment_paths=fragment_paths,
            dataset_dir=str(base / "dataset"),
            voxel_size_mm=spec["voxel_size_mm"],
            target_density_pts_per_mm3=spec["target_density_pts_per_mm3"],
            normal_params=spec["normal_params"],
            registration_params=spec["registration_params"],
            alignment_error_threshold_mm=spec["alignment_error_threshold_mm"],
            correspondence_distance_mm=spec["correspondence_distance_mm"],
            reconstruction_error_threshold_mm=spec["reconstruction_error_threshold_mm"],
            centering_enabled=spec["centering_enabled"],
            scaling_enabled=spec["scaling_enabled"],
            scale_factor=spec["scale_factor"],
            centering_offset=spec["centering_offset"],
            precomputed_transforms=spec["precomputed_transforms"],
            image_output_path=str(base / "render.png"),
            image_width=spec["image_width"],
            image_height=spec["image_height"],
        )

        serialized = dump_config(original)
        config_file = base / "config.yaml"
        config_file.write_text(serialized, encoding="utf-8")

        reloaded = load_config(str(config_file))

        assert reloaded == original


# ---------------------------------------------------------------------------
# Property 2: Configuration validation rejects invalid parameters
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
import yaml  # noqa: E402

from dataset_foundation.errors import ConfigValidationError  # noqa: E402

# Numeric parameters that MUST be strictly > 0. Deliberately excludes:
#   * centering_offset components (may be zero/negative, Req 5.5)
#   * scale_factor (only constrained > 0 when scaling_enabled is True)
# so that setting any of these <= 0 is always a genuine invalidation.
# Each entry is (scope, key); scope=None means a top-level parameter.
_NUMERIC_TARGETS = [
    (None, "voxel_size_mm"),
    (None, "target_density_pts_per_mm3"),
    (None, "alignment_error_threshold_mm"),
    (None, "correspondence_distance_mm"),
    (None, "reconstruction_error_threshold_mm"),
    (None, "image_width"),
    (None, "image_height"),
    ("normal_params", "search_radius_mm"),
    ("normal_params", "max_neighbors"),
    ("normal_params", "orientation_neighbors"),
    ("registration_params", "feature_voxel_size_mm"),
    ("registration_params", "fpfh_radius_mm"),
    ("registration_params", "fpfh_max_neighbors"),
    ("registration_params", "normal_radius_mm"),
    ("registration_params", "ransac_distance_mm"),
    ("registration_params", "ransac_max_iterations"),
    ("registration_params", "ransac_confidence"),
    ("registration_params", "icp_max_distance_mm"),
    ("registration_params", "icp_max_iterations"),
    ("registration_params", "seed"),
]


def _valid_config_dict(spec, base):
    """Create the on-disk input files and return a valid config as a plain dict.

    Reuses the ``config_specs`` field values and the same file-creation approach
    as the round-trip test, then serializes through ``dump_config`` so the result
    is exactly the accepted YAML form (a mapping of only YAML-native types) that
    we can safely mutate before writing back out.
    """
    full_model_path = base / "full_model.ply"
    full_model_path.write_bytes(b"")

    fragment_paths = []
    for index in range(spec["num_fragments"]):
        fragment = base / f"fragment_{index}.ply"
        fragment.write_bytes(b"")
        fragment_paths.append(str(fragment))

    config = Config(
        full_model_path=str(full_model_path),
        fragment_paths=fragment_paths,
        dataset_dir=str(base / "dataset"),
        voxel_size_mm=spec["voxel_size_mm"],
        target_density_pts_per_mm3=spec["target_density_pts_per_mm3"],
        normal_params=spec["normal_params"],
        registration_params=spec["registration_params"],
        alignment_error_threshold_mm=spec["alignment_error_threshold_mm"],
        correspondence_distance_mm=spec["correspondence_distance_mm"],
        reconstruction_error_threshold_mm=spec["reconstruction_error_threshold_mm"],
        centering_enabled=spec["centering_enabled"],
        scaling_enabled=spec["scaling_enabled"],
        scale_factor=spec["scale_factor"],
        centering_offset=spec["centering_offset"],
        precomputed_transforms=spec["precomputed_transforms"],
        image_output_path=str(base / "render.png"),
        image_width=spec["image_width"],
        image_height=spec["image_height"],
    )
    return yaml.safe_load(dump_config(config))


# Feature: dataset-foundation, Property 2: Configuration validation rejects invalid parameters
@settings(max_examples=100)
@given(spec=config_specs(), data=st.data())
def test_config_validation_rejects_invalid_parameters(spec, data):
    """Any valid Configuration with EXACTLY ONE invalidation fails to load.

    Introduces exactly one of: (a) a deleted required parameter, (b) a numeric
    parameter set to <= 0, or (c) a required input path pointed at a
    non-existent file. ``load_config`` must halt with a ``ConfigValidationError``
    naming the offending parameter and must not produce a Config.

    **Validates: Requirements 9.2**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        cfg = _valid_config_dict(spec, base)

        # Which categories of invalidation are applicable to this config.
        kinds = ["delete", "numeric", "path"]
        kind = data.draw(st.sampled_from(kinds))

        if kind == "delete":
            # Remove exactly one required key (top-level or a nested params key).
            scope = data.draw(
                st.sampled_from([None, "normal_params", "registration_params"])
            )
            if scope is None:
                key = data.draw(st.sampled_from(sorted(cfg.keys())))
                del cfg[key]
            else:
                key = data.draw(st.sampled_from(sorted(cfg[scope].keys())))
                del cfg[scope][key]
            offending = key

        elif kind == "numeric":
            # Set exactly one must-be-positive numeric parameter to <= 0.
            scope, key = data.draw(st.sampled_from(_NUMERIC_TARGETS))
            bad_value = data.draw(st.sampled_from([0, -1, -7]))
            if scope is None:
                cfg[key] = bad_value
            else:
                cfg[scope][key] = bad_value
            offending = key

        else:  # kind == "path"
            # Point exactly one required input path at a non-existent file.
            targets = ["full_model_path"]
            if cfg["fragment_paths"]:
                targets.append("fragment_paths")
            target = data.draw(st.sampled_from(targets))
            if target == "full_model_path":
                cfg["full_model_path"] = str(base / "missing_full_model.ply")
                offending = "full_model_path"
            else:
                index = data.draw(
                    st.integers(min_value=0, max_value=len(cfg["fragment_paths"]) - 1)
                )
                cfg["fragment_paths"][index] = str(base / f"missing_fragment_{index}.ply")
                offending = "fragment_paths"

        config_file = base / "invalid_config.yaml"
        config_file.write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )

        # Loading must halt with an error naming the offending parameter and
        # must not produce a Config (no pipeline output).
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(str(config_file))

        named = str(getattr(exc_info.value, "parameter", ""))
        assert offending in named or offending in str(exc_info.value)
