"""Phase 4 configuration parsing and validation.

Defines the frozen :class:`Config` dataclass and ``load_config`` / ``dump_config``
(exact inverses via YAML), mirroring the Phase 2 discipline: a missing required
key or an out-of-range value raises :class:`ConfigValidationError` naming the
offending parameter, and a parse failure raises :class:`ConfigParseError` naming
the path.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any

import yaml

from baseline_geometry.errors import ConfigParseError, ConfigValidationError

__all__ = ["Config", "load_config", "dump_config"]


@dataclass(frozen=True)
class Config:
    """Canonical configuration model for the baseline-geometry pipeline."""

    # Inputs / outputs
    dataset_dir: str
    patches_dir: str
    pairs_npz: str
    pairs_dataset_json: str
    output_dir: str

    # Descriptors
    descriptors: tuple[str, ...] = ("fpfh", "shot")
    fpfh_radius_mm: float = 8.0
    fpfh_max_nn: int = 100
    normal_radius_mm: float = 4.0
    normal_max_nn: int = 30
    shot_radius_mm: float = 8.0
    shot_cos_bins: int = 11
    patch_pooling: str = "mean"

    # Retrieval
    retrieval_k_values: tuple[int, ...] = (1, 5, 10, 20)
    retrieval_query_sample: int = 800
    pair_auc_sample: int = 200000

    # Registration
    registration_descriptor: str = "fpfh"
    perturb_max_rotation_deg: float = 30.0
    perturb_max_translation_mm: float = 20.0
    perturb_seed: int = 42
    registration_voxel_mm: float = 1.5
    ransac_max_corr_dist_mm: float = 3.0
    ransac_n: int = 4
    ransac_max_iterations: int = 100000
    ransac_confidence: float = 0.999
    icp_max_corr_dist_mm: float = 3.0
    icp_max_iterations: int = 50
    success_rotation_deg: float = 15.0
    success_translation_mm: float = 10.0

    # Validation gate
    min_pair_auc: float = 0.5


_VALID_DESCRIPTORS = {"fpfh", "shot"}
_VALID_POOLING = {"mean", "center"}


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigValidationError(key, "missing required key")
    return mapping[key]


def _positive_number(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or value <= 0:
        raise ConfigValidationError(name, "must be a positive number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigValidationError(name, "must be an integer >= 1")
    return int(value)


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigValidationError(name, "must be an integer >= 0")
    return int(value)


def load_config(config_path: str) -> Config:
    """Parse and validate a YAML configuration file into a :class:`Config`."""
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigParseError(config_path, str(exc)) from exc

    if not isinstance(raw, dict):
        raise ConfigParseError(config_path, "top-level YAML must be a mapping")

    dataset_dir = str(_require(raw, "dataset_dir"))
    patches_dir = str(_require(raw, "patches_dir"))
    pairs_npz = str(_require(raw, "pairs_npz"))
    pairs_dataset_json = str(_require(raw, "pairs_dataset_json"))
    output_dir = str(_require(raw, "output_dir"))

    if not os.path.isdir(dataset_dir):
        raise ConfigValidationError("dataset_dir", f"directory does not exist: {dataset_dir}")
    if not os.path.isdir(patches_dir):
        raise ConfigValidationError("patches_dir", f"directory does not exist: {patches_dir}")

    descriptors = tuple(str(d).lower() for d in raw.get("descriptors", ["fpfh", "shot"]))
    if not descriptors:
        raise ConfigValidationError("descriptors", "must list at least one descriptor")
    for d in descriptors:
        if d not in _VALID_DESCRIPTORS:
            raise ConfigValidationError("descriptors", f"unknown descriptor: {d}")

    pooling = str(raw.get("patch_pooling", "mean")).lower()
    if pooling not in _VALID_POOLING:
        raise ConfigValidationError("patch_pooling", f"must be one of {sorted(_VALID_POOLING)}")

    reg_desc = str(raw.get("registration_descriptor", "fpfh")).lower()
    if reg_desc not in _VALID_DESCRIPTORS:
        raise ConfigValidationError("registration_descriptor", f"unknown descriptor: {reg_desc}")

    k_values = tuple(_positive_int(k, "retrieval_k_values") for k in raw.get("retrieval_k_values", [1, 5, 10, 20]))

    confidence = raw.get("ransac_confidence", 0.999)
    if not isinstance(confidence, Real) or not (0.0 < float(confidence) < 1.0):
        raise ConfigValidationError("ransac_confidence", "must be in (0, 1)")

    min_auc = raw.get("min_pair_auc", 0.5)
    if not isinstance(min_auc, Real) or not (0.0 <= float(min_auc) <= 1.0):
        raise ConfigValidationError("min_pair_auc", "must be in [0, 1]")

    return Config(
        dataset_dir=dataset_dir,
        patches_dir=patches_dir,
        pairs_npz=pairs_npz,
        pairs_dataset_json=pairs_dataset_json,
        output_dir=output_dir,
        descriptors=descriptors,
        fpfh_radius_mm=_positive_number(raw.get("fpfh_radius_mm", 8.0), "fpfh_radius_mm"),
        fpfh_max_nn=_positive_int(raw.get("fpfh_max_nn", 100), "fpfh_max_nn"),
        normal_radius_mm=_positive_number(raw.get("normal_radius_mm", 4.0), "normal_radius_mm"),
        normal_max_nn=_positive_int(raw.get("normal_max_nn", 30), "normal_max_nn"),
        shot_radius_mm=_positive_number(raw.get("shot_radius_mm", 8.0), "shot_radius_mm"),
        shot_cos_bins=_positive_int(raw.get("shot_cos_bins", 11), "shot_cos_bins"),
        patch_pooling=pooling,
        retrieval_k_values=k_values,
        retrieval_query_sample=_nonneg_int(raw.get("retrieval_query_sample", 800), "retrieval_query_sample"),
        pair_auc_sample=_nonneg_int(raw.get("pair_auc_sample", 200000), "pair_auc_sample"),
        registration_descriptor=reg_desc,
        perturb_max_rotation_deg=_positive_number(raw.get("perturb_max_rotation_deg", 30.0), "perturb_max_rotation_deg"),
        perturb_max_translation_mm=_positive_number(raw.get("perturb_max_translation_mm", 20.0), "perturb_max_translation_mm"),
        perturb_seed=_nonneg_int(raw.get("perturb_seed", 42), "perturb_seed"),
        registration_voxel_mm=_positive_number(raw.get("registration_voxel_mm", 1.5), "registration_voxel_mm"),
        ransac_max_corr_dist_mm=_positive_number(raw.get("ransac_max_corr_dist_mm", 3.0), "ransac_max_corr_dist_mm"),
        ransac_n=_positive_int(raw.get("ransac_n", 4), "ransac_n"),
        ransac_max_iterations=_positive_int(raw.get("ransac_max_iterations", 100000), "ransac_max_iterations"),
        ransac_confidence=float(confidence),
        icp_max_corr_dist_mm=_positive_number(raw.get("icp_max_corr_dist_mm", 3.0), "icp_max_corr_dist_mm"),
        icp_max_iterations=_positive_int(raw.get("icp_max_iterations", 50), "icp_max_iterations"),
        success_rotation_deg=_positive_number(raw.get("success_rotation_deg", 15.0), "success_rotation_deg"),
        success_translation_mm=_positive_number(raw.get("success_translation_mm", 10.0), "success_translation_mm"),
        min_pair_auc=float(min_auc),
    )


def dump_config(config: Config) -> str:
    """Serialize a :class:`Config` back to YAML (inverse of ``load_config``)."""
    data = asdict(config)
    # tuples -> lists for clean YAML round-trip
    for key in ("descriptors", "retrieval_k_values"):
        data[key] = list(data[key])
    return yaml.safe_dump(data, sort_keys=True)
