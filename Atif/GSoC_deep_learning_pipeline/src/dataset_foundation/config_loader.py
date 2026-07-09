"""Configuration loading, validation, and serialization.

Parses, validates, and serializes the Configuration. The in-memory ``Config``
dataclass is the canonical representation; on disk it is YAML with the same
field structure. ``load_config`` and ``dump_config`` are exact inverses so a
Configuration round-trips through serialize/parse without loss (Req 9.3).

Validation (Req 9.2): every numeric parameter must be ``> 0`` (except
``centering_offset``, which may be zero or negative per Req 5.5, and
``scale_factor`` which is only required ``> 0`` when scaling is enabled);
``ransac_confidence`` must lie in ``(0, 1]``; ``image_width``/``image_height``
must be ``>= 1024``; required input paths must exist. Missing or invalid
parameters raise :class:`ConfigValidationError` naming the offending parameter.
A parse failure raises :class:`ConfigParseError` naming the file path (Req 9.6).

Requirements: 9.1, 9.2, 9.3, 9.6.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import yaml

from dataset_foundation.errors import ConfigParseError, ConfigValidationError

__all__ = [
    "NormalEstimationParams",
    "RegistrationParams",
    "Config",
    "load_config",
    "dump_config",
]


# ---------------------------------------------------------------------------
# Dataclasses (canonical in-memory representation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalEstimationParams:
    """Parameters controlling normal estimation and consistent orientation."""

    search_radius_mm: float          # > 0
    max_neighbors: int               # > 0
    orientation_neighbors: int       # k for consistent orientation, > 0


@dataclass(frozen=True)
class RegistrationParams:
    """Parameters controlling FPFH + RANSAC global registration and ICP."""

    feature_voxel_size_mm: float     # downsample size for FPFH, > 0
    fpfh_radius_mm: float            # > 0
    fpfh_max_neighbors: int          # > 0
    normal_radius_mm: float          # > 0
    ransac_distance_mm: float        # > 0
    ransac_max_iterations: int       # > 0
    ransac_confidence: float         # (0, 1]
    icp_max_distance_mm: float       # > 0
    icp_max_iterations: int          # > 0
    seed: int                        # determinism, > 0


@dataclass(frozen=True)
class Config:
    """Canonical Configuration model for the dataset-foundation pipeline."""

    full_model_path: str
    fragment_paths: list[str]
    dataset_dir: str
    voxel_size_mm: float                       # > 0
    target_density_pts_per_mm3: float          # > 0
    normal_params: NormalEstimationParams
    registration_params: RegistrationParams
    alignment_error_threshold_mm: float        # > 0
    correspondence_distance_mm: float          # > 0
    reconstruction_error_threshold_mm: float   # > 0
    centering_enabled: bool
    scaling_enabled: bool
    scale_factor: float                        # validated > 0 only when scaling_enabled
    centering_offset: tuple[float, float, float]
    precomputed_transforms: dict[str, list[list[float]]]  # fragment_id -> 4x4
    image_output_path: str
    image_width: int                           # >= 1024
    image_height: int                          # >= 1024


# ---------------------------------------------------------------------------
# Serialization: Config -> YAML string
# ---------------------------------------------------------------------------


def dump_config(config: Config) -> str:
    """Serialize a :class:`Config` to a YAML string (inverse of ``load_config``).

    The nested ``NormalEstimationParams`` and ``RegistrationParams`` are emitted
    as nested mappings, ``centering_offset`` as a 3-element list, and
    ``precomputed_transforms`` as a mapping of Fragment_ID to a 4x4 nested list.
    """
    data = _config_to_dict(config)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a Config into a plain, YAML-friendly dict (lists, not tuples)."""
    data = asdict(config)
    # asdict recurses into nested dataclasses already; normalize tuple -> list
    # so the serialized form contains only YAML-native container types.
    data["centering_offset"] = list(config.centering_offset)
    data["fragment_paths"] = list(config.fragment_paths)
    data["precomputed_transforms"] = {
        fid: [list(row) for row in matrix]
        for fid, matrix in config.precomputed_transforms.items()
    }
    return data


# ---------------------------------------------------------------------------
# Parsing: YAML -> Config (with validation)
# ---------------------------------------------------------------------------


def load_config(path: str) -> Config:
    """Parse and validate a YAML configuration file into a :class:`Config`.

    Raises :class:`ConfigParseError` naming ``path`` if the file cannot be read
    or parsed, and :class:`ConfigValidationError` naming the offending parameter
    if a required key is missing or a value violates its constraint.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigParseError(path, str(exc)) from exc

    if not isinstance(raw, dict):
        raise ConfigParseError(
            path, f"expected a mapping at the top level, got {type(raw).__name__}"
        )

    return _config_from_dict(raw)


def _config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a validated Config from a parsed mapping."""
    normal_raw = _require(raw, "normal_params", dict)
    normal_params = NormalEstimationParams(
        search_radius_mm=_positive_float(normal_raw, "search_radius_mm", "normal_params"),
        max_neighbors=_positive_int(normal_raw, "max_neighbors", "normal_params"),
        orientation_neighbors=_positive_int(
            normal_raw, "orientation_neighbors", "normal_params"
        ),
    )

    reg_raw = _require(raw, "registration_params", dict)
    registration_params = RegistrationParams(
        feature_voxel_size_mm=_positive_float(reg_raw, "feature_voxel_size_mm", "registration_params"),
        fpfh_radius_mm=_positive_float(reg_raw, "fpfh_radius_mm", "registration_params"),
        fpfh_max_neighbors=_positive_int(reg_raw, "fpfh_max_neighbors", "registration_params"),
        normal_radius_mm=_positive_float(reg_raw, "normal_radius_mm", "registration_params"),
        ransac_distance_mm=_positive_float(reg_raw, "ransac_distance_mm", "registration_params"),
        ransac_max_iterations=_positive_int(reg_raw, "ransac_max_iterations", "registration_params"),
        ransac_confidence=_confidence(reg_raw, "ransac_confidence", "registration_params"),
        icp_max_distance_mm=_positive_float(reg_raw, "icp_max_distance_mm", "registration_params"),
        icp_max_iterations=_positive_int(reg_raw, "icp_max_iterations", "registration_params"),
        seed=_positive_int(reg_raw, "seed", "registration_params"),
    )

    centering_enabled = _require_bool(raw, "centering_enabled")
    scaling_enabled = _require_bool(raw, "scaling_enabled")

    # scale_factor is only constrained > 0 when scaling is enabled (Req 5.7);
    # otherwise it is stored as given (still coerced to float).
    scale_factor_value = _require(raw, "scale_factor", Real)
    if scaling_enabled:
        scale_factor = _positive_float(raw, "scale_factor")
    else:
        scale_factor = float(scale_factor_value)

    config = Config(
        full_model_path=_require_str(raw, "full_model_path"),
        fragment_paths=_require_str_list(raw, "fragment_paths"),
        dataset_dir=_require_str(raw, "dataset_dir"),
        voxel_size_mm=_positive_float(raw, "voxel_size_mm"),
        target_density_pts_per_mm3=_positive_float(raw, "target_density_pts_per_mm3"),
        normal_params=normal_params,
        registration_params=registration_params,
        alignment_error_threshold_mm=_positive_float(raw, "alignment_error_threshold_mm"),
        correspondence_distance_mm=_positive_float(raw, "correspondence_distance_mm"),
        reconstruction_error_threshold_mm=_positive_float(raw, "reconstruction_error_threshold_mm"),
        centering_enabled=centering_enabled,
        scaling_enabled=scaling_enabled,
        scale_factor=scale_factor,
        centering_offset=_offset(raw, "centering_offset"),
        precomputed_transforms=_transforms(raw, "precomputed_transforms"),
        image_output_path=_require_str(raw, "image_output_path"),
        image_width=_min_int(raw, "image_width", 1024),
        image_height=_min_int(raw, "image_height", 1024),
    )

    _validate_input_paths(config)
    return config


# ---------------------------------------------------------------------------
# Field extraction / validation helpers
# ---------------------------------------------------------------------------


def _qualified(name: str, scope: str | None) -> str:
    return f"{scope}.{name}" if scope else name


def _require(raw: dict[str, Any], name: str, expected: type) -> Any:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    if not isinstance(value, expected):
        raise ConfigValidationError(
            name, f"expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def _require_str(raw: dict[str, Any], name: str) -> str:
    return _require(raw, name, str)


def _require_bool(raw: dict[str, Any], name: str) -> bool:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    if not isinstance(value, bool):
        raise ConfigValidationError(
            name, f"expected bool, got {type(value).__name__}"
        )
    return value


def _require_str_list(raw: dict[str, Any], name: str) -> list[str]:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigValidationError(name, "expected a list of strings")
    return list(value)


def _as_real(raw: dict[str, Any], name: str, scope: str | None) -> float:
    qualified = _qualified(name, scope)
    if name not in raw:
        raise ConfigValidationError(qualified, "missing required parameter")
    value = raw[name]
    # bool is a subclass of int; reject it as a numeric parameter.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigValidationError(
            qualified, f"expected a number, got {type(value).__name__}"
        )
    return float(value)


def _positive_float(raw: dict[str, Any], name: str, scope: str | None = None) -> float:
    value = _as_real(raw, name, scope)
    if not value > 0:
        raise ConfigValidationError(
            _qualified(name, scope), f"must be > 0, got {value}"
        )
    return value


def _positive_int(raw: dict[str, Any], name: str, scope: str | None = None) -> int:
    qualified = _qualified(name, scope)
    if name not in raw:
        raise ConfigValidationError(qualified, "missing required parameter")
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            qualified, f"expected an integer, got {type(value).__name__}"
        )
    if not value > 0:
        raise ConfigValidationError(qualified, f"must be > 0, got {value}")
    return value


def _min_int(raw: dict[str, Any], name: str, minimum: int, scope: str | None = None) -> int:
    value = _positive_int(raw, name, scope)
    if value < minimum:
        raise ConfigValidationError(
            _qualified(name, scope), f"must be >= {minimum}, got {value}"
        )
    return value


def _confidence(raw: dict[str, Any], name: str, scope: str | None = None) -> float:
    value = _positive_float(raw, name, scope)
    if value > 1:
        raise ConfigValidationError(
            _qualified(name, scope), f"must be in (0, 1], got {value}"
        )
    return value


def _offset(raw: dict[str, Any], name: str) -> tuple[float, float, float]:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigValidationError(name, "expected a 3-element list")
    coords = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ConfigValidationError(
                name, f"expected numeric components, got {type(component).__name__}"
            )
        coords.append(float(component))  # may be zero/negative (Req 5.5)
    return (coords[0], coords[1], coords[2])


def _transforms(raw: dict[str, Any], name: str) -> dict[str, list[list[float]]]:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    if not isinstance(value, dict):
        raise ConfigValidationError(name, "expected a mapping of Fragment_ID to 4x4 matrix")
    result: dict[str, list[list[float]]] = {}
    for fragment_id, matrix in value.items():
        if not isinstance(fragment_id, str):
            raise ConfigValidationError(name, "transform keys must be strings")
        if not isinstance(matrix, list) or len(matrix) != 4:
            raise ConfigValidationError(
                name, f"transform for {fragment_id} must be a 4x4 matrix"
            )
        rows: list[list[float]] = []
        for row in matrix:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ConfigValidationError(
                    name, f"transform for {fragment_id} must be a 4x4 matrix"
                )
            parsed_row = []
            for element in row:
                if isinstance(element, bool) or not isinstance(element, Real):
                    raise ConfigValidationError(
                        name, f"transform for {fragment_id} must contain numbers"
                    )
                parsed_row.append(float(element))
            rows.append(parsed_row)
        result[fragment_id] = rows
    return result


def _validate_input_paths(config: Config) -> None:
    """Ensure the required input PLY paths exist on disk (Req 9.2)."""
    if not os.path.exists(config.full_model_path):
        raise ConfigValidationError(
            "full_model_path", f"input path does not exist: {config.full_model_path}"
        )
    for index, fragment_path in enumerate(config.fragment_paths):
        if not os.path.exists(fragment_path):
            raise ConfigValidationError(
                f"fragment_paths[{index}]",
                f"input path does not exist: {fragment_path}",
            )
