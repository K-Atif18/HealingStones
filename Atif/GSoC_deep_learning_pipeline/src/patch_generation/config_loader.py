"""Phase 2 Configuration parsing, validation, and serialization.

Defines the frozen :class:`Config` dataclass and ``load_config``/``dump_config``
(exact inverses via YAML), plus parameter validation raising typed errors that
name the offending parameter or path.

The in-memory ``Config`` dataclass is the canonical representation; on disk it
is YAML with the same field names. ``load_config`` and ``dump_config`` are
exact inverses so a Configuration round-trips through serialize/parse without
loss (Req 10.1, 10.3).

Validation (Req 10.2): a missing required key raises
:class:`ConfigValidationError` naming it; ``patch_radius_mm <= 0``,
``max_patch_points < 1``, ``coverage_threshold`` outside ``[0, 1]``,
``min_patch_size < 1``, ``image_width``/``image_height < 1024``, a non-existent
``dataset_dir``, and a violation of "exactly one of ``target_center_count`` /
``target_center_density_per_mm2`` is set" all raise naming the offending
parameter. A parse failure raises :class:`ConfigParseError` naming the path
(Req 10.7).

Requirements: 10.1, 10.2, 10.3, 10.7.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import yaml

from patch_generation.errors import ConfigParseError, ConfigValidationError

__all__ = [
    "Config",
    "load_config",
    "dump_config",
]


# ---------------------------------------------------------------------------
# Dataclass (canonical in-memory representation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Canonical Configuration model for the patch-generation pipeline."""

    dataset_dir: str                                # Phase 1 output dir (must exist)
    patch_dir: str                                  # Phase 2 output dir
    patch_radius_mm: float                          # > 0
    max_patch_points: int                           # >= 1
    target_center_count: int | None                 # >= 1 when set
    target_center_density_per_mm2: float | None     # > 0 when set
    fps_seed: int                                   # determinism
    coverage_threshold: float                       # in [0.0, 1.0]
    min_patch_size: int                             # >= 1
    image_output_path: str
    image_width: int                                # >= 1024
    image_height: int                               # >= 1024


# ---------------------------------------------------------------------------
# Serialization: Config -> YAML string
# ---------------------------------------------------------------------------


def dump_config(config: Config) -> str:
    """Serialize a :class:`Config` to a YAML string (inverse of ``load_config``).

    The nullable ``target_center_count`` / ``target_center_density_per_mm2``
    fields are emitted as ``null`` when unset, so the serialized form always
    contains every field and round-trips exactly through ``load_config``.
    """
    data: dict[str, Any] = asdict(config)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


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
    target_center_count = _optional_center_count(raw, "target_center_count")
    target_center_density = _optional_density(raw, "target_center_density_per_mm2")
    _validate_exactly_one_center_spec(target_center_count, target_center_density)

    config = Config(
        dataset_dir=_require_str(raw, "dataset_dir"),
        patch_dir=_require_str(raw, "patch_dir"),
        patch_radius_mm=_positive_float(raw, "patch_radius_mm"),
        max_patch_points=_min_int(raw, "max_patch_points", 1),
        target_center_count=target_center_count,
        target_center_density_per_mm2=target_center_density,
        fps_seed=_require_int(raw, "fps_seed"),
        coverage_threshold=_unit_interval(raw, "coverage_threshold"),
        min_patch_size=_min_int(raw, "min_patch_size", 1),
        image_output_path=_require_str(raw, "image_output_path"),
        image_width=_min_int(raw, "image_width", 1024),
        image_height=_min_int(raw, "image_height", 1024),
    )

    _validate_dataset_dir(config)
    return config


# ---------------------------------------------------------------------------
# Field extraction / validation helpers
# ---------------------------------------------------------------------------


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


def _require_int(raw: dict[str, Any], name: str) -> int:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    # bool is a subclass of int; reject it as an integer parameter.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            name, f"expected an integer, got {type(value).__name__}"
        )
    return value


def _as_real(raw: dict[str, Any], name: str) -> float:
    if name not in raw:
        raise ConfigValidationError(name, "missing required parameter")
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigValidationError(
            name, f"expected a number, got {type(value).__name__}"
        )
    return float(value)


def _positive_float(raw: dict[str, Any], name: str) -> float:
    value = _as_real(raw, name)
    if not value > 0:
        raise ConfigValidationError(name, f"must be > 0, got {value}")
    return value


def _min_int(raw: dict[str, Any], name: str, minimum: int) -> int:
    value = _require_int(raw, name)
    if value < minimum:
        raise ConfigValidationError(name, f"must be >= {minimum}, got {value}")
    return value


def _unit_interval(raw: dict[str, Any], name: str) -> float:
    value = _as_real(raw, name)
    if value < 0 or value > 1:
        raise ConfigValidationError(name, f"must be in [0, 1], got {value}")
    return value


def _optional_center_count(raw: dict[str, Any], name: str) -> int | None:
    """Return an int center count (>= 1) or ``None`` when unset/null."""
    value = raw.get(name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            name, f"expected an integer or null, got {type(value).__name__}"
        )
    if value < 1:
        raise ConfigValidationError(name, f"must be >= 1 when set, got {value}")
    return value


def _optional_density(raw: dict[str, Any], name: str) -> float | None:
    """Return a float density (> 0) or ``None`` when unset/null."""
    value = raw.get(name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigValidationError(
            name, f"expected a number or null, got {type(value).__name__}"
        )
    density = float(value)
    if not density > 0:
        raise ConfigValidationError(name, f"must be > 0 when set, got {density}")
    return density


def _validate_exactly_one_center_spec(
    target_center_count: int | None,
    target_center_density: float | None,
) -> None:
    """Enforce that exactly one center specification is provided (Req 10.2)."""
    provided = [
        spec
        for spec in ("target_center_count", "target_center_density_per_mm2")
        if (target_center_count if spec == "target_center_count" else target_center_density)
        is not None
    ]
    if len(provided) != 1:
        raise ConfigValidationError(
            "target_center_count/target_center_density_per_mm2",
            "exactly one of target_center_count / target_center_density_per_mm2 "
            f"must be set, but {len(provided)} were provided",
        )


def _validate_dataset_dir(config: Config) -> None:
    """Ensure the Phase 1 dataset directory exists on disk (Req 10.2)."""
    if not os.path.isdir(config.dataset_dir):
        raise ConfigValidationError(
            "dataset_dir", f"dataset directory does not exist: {config.dataset_dir}"
        )
