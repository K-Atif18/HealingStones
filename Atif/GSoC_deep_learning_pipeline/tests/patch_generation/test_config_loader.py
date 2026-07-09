"""Tests for ``patch_generation.config_loader``.

Covers three spec tasks that share this file:

* Task 2.2 -- Property 1: Configuration round-trip (Requirements 10.1, 10.3).
* Task 2.3 -- Property 2: Configuration validation rejects invalid parameters
  (Requirement 10.2).
* Task 2.4 -- Unit test: an unreadable/malformed configuration raises
  ``ConfigParseError`` naming the path (Requirement 10.7).

The property tests build Configurations with a Hypothesis strategy. Because
``load_config`` requires ``dataset_dir`` to exist on disk, each example creates
its own ``tempfile.TemporaryDirectory`` inside the test body (rather than using
a function-scoped fixture, which Hypothesis disallows).
"""

from __future__ import annotations

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patch_generation.config_loader import Config, dump_config, load_config
from patch_generation.errors import ConfigParseError, ConfigValidationError


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Finite floats rounded to a few decimals so YAML serialize/parse round-trips
# exactly (no precision drift between dump_config and load_config).


def _rounded_float(min_value: float, max_value: float):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    ).map(lambda v: round(v, 4))


@st.composite
def _config_field_dicts(draw):
    """Draw the fields of a valid Configuration as a plain dict.

    ``dataset_dir`` is intentionally omitted here; the test body injects a real
    temporary directory path so ``load_config``'s existence check passes.
    Exactly one of ``target_center_count`` / ``target_center_density_per_mm2``
    is set (the other is ``None``), chosen by a drawn boolean.
    """
    use_count = draw(st.booleans())
    if use_count:
        target_center_count = draw(st.integers(min_value=1, max_value=100_000))
        target_center_density = None
    else:
        target_center_count = None
        target_center_density = draw(_rounded_float(1e-3, 10.0))
        if target_center_density <= 0:  # guard against rounding to 0.0
            target_center_density = 0.001

    patch_radius = draw(_rounded_float(1e-3, 100.0))
    if patch_radius <= 0:
        patch_radius = 0.001

    return {
        "patch_dir": draw(st.sampled_from(["patches/", "out/patches", "p"])),
        "patch_radius_mm": patch_radius,
        "max_patch_points": draw(st.integers(min_value=1, max_value=1024)),
        "target_center_count": target_center_count,
        "target_center_density_per_mm2": target_center_density,
        "fps_seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
        "coverage_threshold": draw(_rounded_float(0.0, 1.0)),
        "min_patch_size": draw(st.integers(min_value=1, max_value=64)),
        "image_output_path": draw(
            st.sampled_from(["patches/vis.png", "out/coverage.png"])
        ),
        "image_width": draw(st.integers(min_value=1024, max_value=4096)),
        "image_height": draw(st.integers(min_value=1024, max_value=4096)),
    }


def _build_config(fields: dict, dataset_dir: str) -> Config:
    return Config(dataset_dir=dataset_dir, **fields)


# ---------------------------------------------------------------------------
# Task 2.2 -- Property 1: Configuration round-trip
# ---------------------------------------------------------------------------


# Feature: patch-generation, Property 1: Configuration round-trip
# Validates: Requirements 10.1, 10.3
@settings(max_examples=100)
@given(fields=_config_field_dicts())
def test_config_round_trip(fields):
    """dump_config -> write -> load_config reproduces the original Config."""
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = os.path.join(tmp, "dataset")
        os.mkdir(dataset_dir)  # dataset_dir must exist for load_config to accept it
        original = _build_config(fields, dataset_dir)

        config_path = os.path.join(tmp, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(dump_config(original))

        reloaded = load_config(config_path)

        assert reloaded == original


# ---------------------------------------------------------------------------
# Task 2.3 -- Property 2: Configuration validation rejects invalid parameters
# ---------------------------------------------------------------------------


# Each invalidator maps a valid field dict -> (offending_parameter_names, bad_dict).
# ``offending_parameter_names`` lists acceptable substrings; the raised error's
# .parameter or its str() must name one of them.
def _missing_key(key):
    def mutate(cfg):
        bad = dict(cfg)
        bad.pop(key, None)
        return ([key], bad)

    return mutate


def _set(key, value, expected=None):
    def mutate(cfg):
        bad = dict(cfg)
        bad[key] = value
        return ([expected or key], bad)

    return mutate


def _both_center_specs_null(cfg):
    bad = dict(cfg)
    bad["target_center_count"] = None
    bad["target_center_density_per_mm2"] = None
    return (["target_center_count", "target_center_density_per_mm2"], bad)


def _both_center_specs_set(cfg):
    bad = dict(cfg)
    bad["target_center_count"] = 1000
    bad["target_center_density_per_mm2"] = 0.5
    return (["target_center_count", "target_center_density_per_mm2"], bad)


_INVALIDATORS = [
    _missing_key("patch_radius_mm"),
    _missing_key("max_patch_points"),
    _missing_key("fps_seed"),
    _missing_key("image_width"),
    _set("patch_radius_mm", 0.0),
    _set("patch_radius_mm", -1.0),
    _set("max_patch_points", 0),
    _set("coverage_threshold", 1.5),
    _set("coverage_threshold", -0.1),
    _set("min_patch_size", 0),
    _set("image_width", 1023),
    _set("image_height", 1023),
    _both_center_specs_null,
    _both_center_specs_set,
    "nonexistent_dataset_dir",  # special-cased in the test body
]


# Feature: patch-generation, Property 2: Configuration validation rejects invalid parameters
# Validates: Requirements 10.2
@settings(max_examples=100)
@given(fields=_config_field_dicts(), invalidator=st.sampled_from(_INVALIDATORS))
def test_config_validation_rejects_invalid_parameters(fields, invalidator):
    """A single invalidation makes load_config raise ConfigValidationError
    naming the offending parameter."""
    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = os.path.join(tmp, "dataset")
        os.mkdir(dataset_dir)

        base = dict(fields)
        base["dataset_dir"] = dataset_dir

        if invalidator == "nonexistent_dataset_dir":
            bad = dict(base)
            bad["dataset_dir"] = os.path.join(tmp, "does_not_exist")
            expected = ["dataset_dir"]
        else:
            expected, bad = invalidator(base)

        config_path = os.path.join(tmp, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(bad, handle, sort_keys=False)

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(config_path)

        exc = exc_info.value
        named = f"{exc.parameter} {exc}"
        assert any(name in named for name in expected), (
            f"expected one of {expected} to be named, got: {named}"
        )


# ---------------------------------------------------------------------------
# Task 2.4 -- Unit test: malformed / unreadable config raises ConfigParseError
# ---------------------------------------------------------------------------


def test_malformed_config_raises_parse_error_naming_path():
    """A syntactically broken YAML file raises ConfigParseError naming the path."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "broken.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write("key: : :\n  - broken\n:::not valid yaml:::\n")

        with pytest.raises(ConfigParseError) as exc_info:
            load_config(config_path)

        exc = exc_info.value
        assert config_path in str(exc)
        assert exc.path == config_path


def test_nonexistent_config_path_raises_parse_error_naming_path():
    """A path that does not exist raises ConfigParseError naming the path."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "missing.yaml")

        with pytest.raises(ConfigParseError) as exc_info:
            load_config(config_path)

        exc = exc_info.value
        assert config_path in str(exc)
        assert exc.path == config_path


def test_non_mapping_config_raises_parse_error_naming_path():
    """A YAML file that is valid but not a top-level mapping raises ConfigParseError."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "list.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write("- just\n- a\n- list\n")

        with pytest.raises(ConfigParseError) as exc_info:
            load_config(config_path)

        assert config_path in str(exc_info.value)
