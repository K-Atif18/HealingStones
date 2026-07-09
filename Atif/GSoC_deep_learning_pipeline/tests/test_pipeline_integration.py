"""End-to-end integration test for the dataset-foundation pipeline (task 16.6).

This is a genuine end-to-end exercise of ``pipeline.run(config_path)`` but bounded
in runtime. It:

1. Builds a feature-rich synthetic "full model" point cloud (a box surface with
   an asymmetric bump, a few thousand points) and derives a handful of
   "fragments" by taking subsets of the model and applying KNOWN rigid transforms
   so each fragment lives in its own coordinate frame.
2. Writes the model and fragments as binary little-endian MESH PLY files via
   ``ply_io.write_mesh`` (each mesh carries vertices plus a few trivial faces so
   ``load_mesh`` round-trips the geometry; only vertices are used downstream).
3. Supplies ``precomputed_transforms`` equal to the INVERSE of each known
   transform, so the aligner imports them and reconstruction is exact and fast.
4. Dumps the Config to a YAML file and runs the full pipeline.

It then asserts artifact completeness (Req 7.1), metadata consistency (Req 7.4),
and the reconstruction-validation outcome (Req 8.7).

Validates: Requirements 7.1, 7.4, 8.7.

This is a plain pytest test (not Hypothesis) with fixed seeds for determinism.
"""

from __future__ import annotations

import numpy as np

from dataset_foundation.config_loader import (
    Config,
    NormalEstimationParams,
    RegistrationParams,
    dump_config,
)
from dataset_foundation.fragment_registry import sanitize_fragment_id
from dataset_foundation.geometry import Mesh
from dataset_foundation.metadata_manager import read_metadata
from dataset_foundation.ply_io import load_mesh, write_mesh
from dataset_foundation.pipeline import run

# Fixed seed for full determinism.
SEED = 12345
MODEL_SIZE_MM = 100.0


# ---------------------------------------------------------------------------
# Synthetic geometry helpers
# ---------------------------------------------------------------------------


def _build_model_points() -> np.ndarray:
    """Build a feature-rich synthetic full-model point cloud (a few thousand pts).

    The shape is the surface of a ``MODEL_SIZE_MM`` cube (six faces) plus an
    asymmetric hemispherical bump on the top face, giving distinctive geometry.
    """
    rng = np.random.default_rng(SEED)
    size = MODEL_SIZE_MM
    n_per_face = 400

    faces: list[np.ndarray] = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for plane in (0.0, size):
            uv = rng.uniform(0.0, size, size=(n_per_face, 2))
            pts = np.zeros((n_per_face, 3), dtype=np.float64)
            pts[:, others[0]] = uv[:, 0]
            pts[:, others[1]] = uv[:, 1]
            pts[:, axis] = plane
            faces.append(pts)

    # Asymmetric bump: a hemisphere protruding from the top (+z) face, offset
    # toward one corner so the shape has no symmetry.
    n_bump = 250
    center = np.array([size * 0.7, size * 0.3, size], dtype=np.float64)
    radius = size * 0.15
    dirs = rng.normal(size=(n_bump, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 2] = np.abs(dirs[:, 2])  # upper hemisphere only
    bump = center + radius * dirs
    faces.append(bump)

    return np.concatenate(faces, axis=0)


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a rotation matrix from intrinsic X, Y, Z Euler angles (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return r_z @ r_y @ r_x


def _rigid_transform(angles: tuple[float, float, float],
                     translation: tuple[float, float, float]) -> np.ndarray:
    """Assemble a 4x4 homogeneous rigid transform from angles and a translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_matrix(*angles)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``(N, 3)`` points through a 4x4 homogeneous transform."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def _mesh_from_points(points: np.ndarray) -> Mesh:
    """Wrap points as a Mesh with a few trivial (valid) triangular faces.

    Only vertices are used downstream, but a handful of faces keeps the written
    PLY a well-formed triangle mesh that ``load_mesh`` round-trips reliably.
    """
    n = points.shape[0]
    triples = [[i, i + 1, i + 2] for i in range(0, n - 2, 3)]
    faces = (
        np.asarray(triples, dtype=np.int32)
        if triples
        else np.empty((0, 3), dtype=np.int32)
    )
    return Mesh(vertices=points, faces=faces)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _build_config(
    *,
    full_model_path: str,
    fragment_paths: list[str],
    dataset_dir: str,
    image_output_path: str,
    precomputed_transforms: dict[str, list[list[float]]],
) -> Config:
    """Build a Config with generous thresholds so imported inverses PASS."""
    return Config(
        full_model_path=full_model_path,
        fragment_paths=fragment_paths,
        dataset_dir=dataset_dir,
        voxel_size_mm=2.0,
        target_density_pts_per_mm3=1.0,
        normal_params=NormalEstimationParams(
            search_radius_mm=10.0,
            max_neighbors=30,
            orientation_neighbors=10,
        ),
        registration_params=RegistrationParams(
            feature_voxel_size_mm=4.0,
            fpfh_radius_mm=20.0,
            fpfh_max_neighbors=100,
            normal_radius_mm=8.0,
            ransac_distance_mm=6.0,
            ransac_max_iterations=100000,
            ransac_confidence=0.999,
            icp_max_distance_mm=2.0,
            icp_max_iterations=50,
            seed=42,
        ),
        # Generous thresholds: with exact inverse transforms the reconstruction
        # RMSE is ~0, so alignment and reconstruction comfortably pass.
        alignment_error_threshold_mm=25.0,
        correspondence_distance_mm=5.0,
        reconstruction_error_threshold_mm=10.0,
        centering_enabled=False,
        scaling_enabled=False,
        scale_factor=1.0,
        centering_offset=(0.0, 0.0, 0.0),
        precomputed_transforms=precomputed_transforms,
        image_output_path=image_output_path,
        image_width=1024,
        image_height=1024,
    )


# ---------------------------------------------------------------------------
# The end-to-end integration test
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end_synthetic(tmp_path):
    """Run synthetic fragments through the full pipeline and verify outputs."""
    model_points = _build_model_points()

    # Write the full model as a MESH PLY and confirm load_mesh round-trips it.
    model_path = tmp_path / "model.ply"
    write_mesh(str(model_path), _mesh_from_points(model_points))
    loaded_model = load_mesh(str(model_path))
    assert loaded_model.vertex_count == model_points.shape[0]

    # Derive fragments from disjoint subsets of the model, each moved into its
    # own frame by a distinct KNOWN rigid transform. The precomputed transform
    # supplied to the pipeline is the exact inverse, so alignment is exact.
    n = model_points.shape[0]
    subsets = [
        model_points[0 : n // 3],
        model_points[n // 3 : 2 * n // 3],
        model_points[2 * n // 3 :],
    ]
    known_transforms = [
        _rigid_transform((0.3, -0.2, 0.5), (40.0, -15.0, 25.0)),
        _rigid_transform((-0.4, 0.6, -0.1), (-30.0, 55.0, -10.0)),
        _rigid_transform((0.15, 0.25, -0.35), (10.0, 20.0, -45.0)),
    ]

    fragment_paths: list[str] = []
    fragment_ids: list[str] = []
    precomputed: dict[str, list[list[float]]] = {}

    for index, (subset, transform) in enumerate(zip(subsets, known_transforms)):
        fragment_points = _apply_transform(transform, subset)
        fragment_path = tmp_path / f"frag_{index}.ply"
        write_mesh(str(fragment_path), _mesh_from_points(fragment_points))

        fragment_id = sanitize_fragment_id(str(fragment_path))
        fragment_paths.append(str(fragment_path))
        fragment_ids.append(fragment_id)
        # Inverse of the known transform maps the fragment back into the model
        # frame -> excellent (near-zero RMSE) reconstruction.
        precomputed[fragment_id] = np.linalg.inv(transform).tolist()

    dataset_dir = tmp_path / "dataset"
    config = _build_config(
        full_model_path=str(model_path),
        fragment_paths=fragment_paths,
        dataset_dir=str(dataset_dir),
        image_output_path=str(tmp_path / "vis.png"),
        precomputed_transforms=precomputed,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(dump_config(config), encoding="utf-8")

    # --- Run the full pipeline end-to-end ----------------------------------
    result = run(str(config_path))

    # --- Artifact completeness (Req 7.1) -----------------------------------
    for fragment_id in fragment_ids:
        assert (dataset_dir / "fragments" / f"{fragment_id}.ply").exists()
        assert (dataset_dir / "transforms" / f"{fragment_id}.txt").exists()
        assert (dataset_dir / "normals" / f"{fragment_id}.ply").exists()
        assert (dataset_dir / "normalized" / f"{fragment_id}.ply").exists()
        assert (dataset_dir / "metadata" / f"{fragment_id}.json").exists()
    dataset_metadata_path = dataset_dir / "metadata" / "dataset.json"
    assert dataset_metadata_path.exists()

    # --- Metadata consistency (Req 7.4) ------------------------------------
    assert set(result["fragment_ids"]) == set(fragment_ids)
    for fragment_id in fragment_ids:
        entry = result["fragments"][fragment_id]
        assert entry["aligned"] == f"fragments/{fragment_id}.ply"
        assert entry["transform"] == f"transforms/{fragment_id}.txt"
        assert entry["normals"] == f"normals/{fragment_id}.ply"
        assert entry["normalized"] == f"normalized/{fragment_id}.ply"
        assert entry["metadata"] == f"metadata/{fragment_id}.json"
        assert entry["alignment_method"] == "imported"

        record = read_metadata(str(dataset_dir / "metadata" / f"{fragment_id}.json"))
        assert record["fragment_id"] == fragment_id
        assert record["alignment_method"] == "imported"

    # The dataset metadata written to disk matches the returned record.
    on_disk = read_metadata(str(dataset_metadata_path))
    assert set(on_disk["fragment_ids"]) == set(fragment_ids)
    assert set(on_disk["fragments"].keys()) == set(fragment_ids)

    # --- Reconstruction outcome (Req 8.7) ----------------------------------
    reconstruction = result["reconstruction"]
    for key in (
        "mean_distance_mm",
        "rmse_mm",
        "coverage_fraction",
        "threshold_mm",
        "passed",
        "empty",
        "offending_fragment_ids",
    ):
        assert key in reconstruction

    assert reconstruction["empty"] is False
    assert reconstruction["rmse_mm"] is not None
    # Exact inverse transforms -> near-zero RMSE, so validation passes.
    assert reconstruction["passed"] is True
    assert reconstruction["rmse_mm"] <= reconstruction["threshold_mm"]
    assert reconstruction["offending_fragment_ids"] == []
