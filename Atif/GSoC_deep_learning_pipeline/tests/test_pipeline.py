"""Tests for :mod:`dataset_foundation.pipeline`.

Covers the pipeline orchestrator's end-to-end guarantees:

- Property 27 (provenance): every completed run records a run timestamp, the
  serialized Config, and a per-fragment alignment method (Req 9.5).
- Property 21 (output completeness): a completed run writes exactly one aligned
  cloud, transform, normals cloud, normalized cloud, and fragment-metadata
  record per Fragment_ID -- including empty fragments -- plus one dataset-level
  metadata record (Req 1.6, 7.1).
- Property 22 (dataset-record enumeration): the dataset record enumerates every
  Fragment_ID and maps each to its artifact paths (Req 7.4).
- Unit tests (16.5): success/failure log records carry stage, Fragment_ID, and
  status (Req 9.4, 9.7); any output write failure halts the run and it is not
  marked completed (Req 7.6).

To keep every run fast and deterministic these tests force the *import* path of
the aligner: each Fragment_ID is given a precomputed identity transform in the
Configuration, so the pipeline never runs the slow FPFH/RANSAC/ICP compute path.
The datasets are tiny synthetic meshes written to a temporary directory, and at
least one *empty* fragment is always included so the empty-geometry guarantees
(Req 1.6/7.1) are exercised.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import dataset_foundation.pipeline as pipeline_mod
from dataset_foundation.config_loader import (
    Config,
    NormalEstimationParams,
    RegistrationParams,
    dump_config,
)
from dataset_foundation.dataset_layout import ensure_layout
from dataset_foundation.errors import WriteError
from dataset_foundation.fragment_registry import sanitize_fragment_id
from dataset_foundation.geometry import Mesh
from dataset_foundation.logging_setup import ROOT_LOGGER_NAME
from dataset_foundation.pipeline import run
from dataset_foundation.ply_io import write_mesh

# A valid 4x4 rigid transform (identity) used as every fragment's precomputed
# transform so the aligner takes the fast IMPORT path rather than computing.
_IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


# ---------------------------------------------------------------------------
# Synthetic dataset construction helpers.
# ---------------------------------------------------------------------------


def _small_mesh(offset: float) -> Mesh:
    """Build a tiny non-empty tetrahedron mesh translated by ``offset`` mm."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=np.float64,
    ) + float(offset)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return Mesh(vertices=vertices, faces=faces)


def _build_config(base_dir: str, n_fragments: int, include_empty: bool) -> Config:
    """Write synthetic model + fragment PLYs and return a matching Config.

    Every non-empty fragment gets a small tetrahedron mesh; when
    ``include_empty`` is set, one additional zero-vertex fragment is written to
    exercise the empty-geometry guarantees (Req 1.6/7.1). Each Fragment_ID is
    seeded with a precomputed identity transform so alignment uses the fast
    import path.
    """
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    model_path = os.path.join(data_dir, "model.ply")
    write_mesh(model_path, _small_mesh(0.0))

    fragment_paths: list[str] = []
    for i in range(n_fragments):
        frag_path = os.path.join(data_dir, f"frag{i}.ply")
        write_mesh(frag_path, _small_mesh(float(i)))
        fragment_paths.append(frag_path)

    if include_empty:
        empty_path = os.path.join(data_dir, "empty.ply")
        write_mesh(empty_path, Mesh())  # zero-vertex, zero-face mesh
        fragment_paths.append(empty_path)

    precomputed = {
        sanitize_fragment_id(path): [row[:] for row in _IDENTITY_4X4]
        for path in fragment_paths
    }

    return Config(
        full_model_path=model_path,
        fragment_paths=fragment_paths,
        dataset_dir=os.path.join(base_dir, "dataset"),
        voxel_size_mm=1.0,
        target_density_pts_per_mm3=1.0,
        normal_params=NormalEstimationParams(
            search_radius_mm=3.0, max_neighbors=30, orientation_neighbors=15
        ),
        registration_params=RegistrationParams(
            feature_voxel_size_mm=4.0,
            fpfh_radius_mm=20.0,
            fpfh_max_neighbors=100,
            normal_radius_mm=8.0,
            ransac_distance_mm=6.0,
            ransac_max_iterations=1000,
            ransac_confidence=0.999,
            icp_max_distance_mm=1.5,
            icp_max_iterations=50,
            seed=42,
        ),
        # Kept large so imported identity transforms are not flagged; irrelevant
        # to the completeness/provenance guarantees under test.
        alignment_error_threshold_mm=1000.0,
        correspondence_distance_mm=3.0,
        reconstruction_error_threshold_mm=1000.0,
        centering_enabled=False,
        scaling_enabled=False,
        scale_factor=1.0,
        centering_offset=(0.0, 0.0, 0.0),
        precomputed_transforms=precomputed,
        image_output_path=os.path.join(base_dir, "dataset", "visualization.png"),
        image_width=1024,
        image_height=1024,
    )


def _write_config_file(base_dir: str, config: Config) -> str:
    """Serialize ``config`` to ``config.yaml`` under ``base_dir`` and return path."""
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(dump_config(config))
    return config_path


def _prepare(base_dir: str, n_fragments: int, include_empty: bool):
    """Build the synthetic dataset and return (config, config_path, fragment_ids)."""
    config = _build_config(base_dir, n_fragments, include_empty)
    config_path = _write_config_file(base_dir, config)
    fragment_ids = [sanitize_fragment_id(p) for p in config.fragment_paths]
    return config, config_path, fragment_ids


_ARTIFACT_METHODS = {"imported", "computed", "skipped"}


# ---------------------------------------------------------------------------
# Property 27: Provenance is recorded per run and per fragment
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 27: Provenance is recorded per run and per fragment
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(n_fragments=st.integers(min_value=1, max_value=3))
def test_provenance_recorded_per_run_and_per_fragment(n_fragments):
    """A completed run records timestamp, config, and per-fragment method.

    **Validates: Requirements 9.5**
    """
    with tempfile.TemporaryDirectory() as tmp:
        _config, config_path, fragment_ids = _prepare(
            tmp, n_fragments, include_empty=True
        )
        record = run(config_path)

        # Run-level provenance: a non-empty run timestamp string.
        assert isinstance(record["run_timestamp"], str)
        assert record["run_timestamp"].strip() != ""

        # Config provenance: the serialized Config is recorded as a dict with the
        # expected top-level parameter keys (Req 9.5).
        config_used = record["config_used"]
        assert isinstance(config_used, dict)
        for key in (
            "full_model_path",
            "fragment_paths",
            "dataset_dir",
            "voxel_size_mm",
            "normal_params",
            "registration_params",
            "precomputed_transforms",
        ):
            assert key in config_used

        # Per-fragment provenance: every fragment carries an alignment method.
        fragments = record["fragments"]
        assert set(fragments.keys()) == set(fragment_ids)
        for fragment_id in fragment_ids:
            assert fragments[fragment_id]["alignment_method"] in _ARTIFACT_METHODS


# ---------------------------------------------------------------------------
# Property 21: Output completeness across all fragments
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 21: Output completeness across all fragments
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(n_fragments=st.integers(min_value=1, max_value=3))
def test_output_completeness_across_all_fragments(n_fragments):
    """Every Fragment_ID gets all five artifacts; one dataset record exists.

    An empty fragment is always included, so this also asserts empty fragments
    receive the full artifact set (Req 1.6/7.1).

    **Validates: Requirements 1.6, 7.1**
    """
    with tempfile.TemporaryDirectory() as tmp:
        config, config_path, fragment_ids = _prepare(
            tmp, n_fragments, include_empty=True
        )
        run(config_path)

        paths = ensure_layout(config.dataset_dir)

        for fragment_id in fragment_ids:
            # Exactly one artifact of each stage per Fragment_ID.
            assert paths.fragment_pc(fragment_id).is_file()
            assert paths.transform(fragment_id).is_file()
            assert paths.normals_pc(fragment_id).is_file()
            assert paths.normalized_pc(fragment_id).is_file()
            assert paths.metadata_record(fragment_id).is_file()

        # Exactly one dataset-level metadata record.
        assert paths.dataset_metadata().is_file()


# Feature: dataset-foundation, Property 21: Output completeness across all fragments
def test_empty_fragment_receives_full_artifact_set(tmp_path):
    """An empty fragment still gets all five artifacts + a dataset record (Req 1.6/7.1)."""
    config, config_path, fragment_ids = _prepare(
        str(tmp_path), n_fragments=1, include_empty=True
    )
    record = run(config_path)

    paths = ensure_layout(config.dataset_dir)

    # The last id corresponds to the empty fragment written by _build_config.
    empty_id = fragment_ids[-1]
    assert paths.fragment_pc(empty_id).is_file()
    assert paths.transform(empty_id).is_file()
    assert paths.normals_pc(empty_id).is_file()
    assert paths.normalized_pc(empty_id).is_file()
    assert paths.metadata_record(empty_id).is_file()
    assert paths.dataset_metadata().is_file()

    # The empty fragment was skipped by the aligner but still enumerated.
    assert record["fragments"][empty_id]["alignment_method"] in _ARTIFACT_METHODS


# ---------------------------------------------------------------------------
# Property 22: Dataset record enumerates every fragment and its artifact paths
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 22: Dataset record enumerates every fragment and its artifact paths
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(n_fragments=st.integers(min_value=1, max_value=3))
def test_dataset_record_enumerates_every_fragment_and_paths(n_fragments):
    """The dataset record lists every Fragment_ID and maps each to artifacts.

    **Validates: Requirements 7.4**
    """
    with tempfile.TemporaryDirectory() as tmp:
        _config, config_path, fragment_ids = _prepare(
            tmp, n_fragments, include_empty=True
        )
        record = run(config_path)

        # fragment_ids enumerates exactly the assigned Fragment_IDs.
        assert set(record["fragment_ids"]) == set(fragment_ids)
        assert len(record["fragment_ids"]) == len(fragment_ids)

        fragments = record["fragments"]
        assert set(fragments.keys()) == set(fragment_ids)

        for fragment_id in fragment_ids:
            entry = fragments[fragment_id]
            for key in ("aligned", "transform", "normals", "normalized", "metadata"):
                assert key in entry, f"missing artifact path {key!r} for {fragment_id}"
            # Aligned/transform/normals/normalized/metadata paths are recorded as
            # non-empty relative path strings.
            for key in ("aligned", "transform", "normals", "normalized", "metadata"):
                assert isinstance(entry[key], str) and entry[key] != ""
            # Recorded paths reference the correct Fragment_ID.
            assert fragment_id in entry["aligned"]
            assert fragment_id in entry["metadata"]


# ---------------------------------------------------------------------------
# Unit tests (16.5): structured logging and fatal write-failure handling.
# ---------------------------------------------------------------------------


class _RecordCollector(logging.Handler):
    """Logging handler that collects emitted :class:`logging.LogRecord` objects."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)


def test_log_records_carry_stage_fragment_id_and_status(tmp_path):
    """Success/failure records carry stage, status, and Fragment_ID (Req 9.4, 9.7)."""
    config, config_path, fragment_ids = _prepare(
        str(tmp_path), n_fragments=2, include_empty=False
    )

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    collector = _RecordCollector()
    logger.addHandler(collector)
    try:
        run(config_path)
    finally:
        logger.removeHandler(collector)

    stage_records = [r for r in collector.records if hasattr(r, "stage")]
    assert stage_records, "expected structured stage log records"

    # Every stage record carries a stage name and a success/failure status.
    for record in stage_records:
        assert isinstance(record.stage, str) and record.stage != ""
        assert record.status in ("success", "failure")

    # At least one success record was emitted.
    success_records = [r for r in stage_records if r.status == "success"]
    assert success_records

    # Per-fragment stages carry the Fragment_ID; at least one references a known
    # Fragment_ID (Req 9.4).
    fragment_records = [r for r in stage_records if hasattr(r, "fragment_id")]
    assert fragment_records
    assert any(r.fragment_id in set(fragment_ids) for r in fragment_records)


def test_output_write_failure_halts_run_and_is_not_marked_completed(tmp_path, monkeypatch):
    """An output write failure halts the run without a dataset record (Req 7.6)."""
    config, config_path, _fragment_ids = _prepare(
        str(tmp_path), n_fragments=2, include_empty=False
    )

    def _raise_write_error(path, _pc):
        raise WriteError(path, "simulated disk failure")

    # Patch the point-cloud writer used by the pipeline so the first artifact
    # write fails fatally.
    monkeypatch.setattr(pipeline_mod, "write_point_cloud", _raise_write_error)

    with pytest.raises(WriteError):
        run(config_path)

    # The run was NOT marked completed: no dataset-level metadata record exists.
    paths = ensure_layout(config.dataset_dir)
    assert not paths.dataset_metadata().exists()
