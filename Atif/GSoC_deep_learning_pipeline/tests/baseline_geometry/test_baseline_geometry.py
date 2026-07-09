"""Unit tests for Phase 4 baseline-geometry components.

These tests are self-contained (synthetic geometry) except for a couple of
integration checks that read the real Phase 1-3 artifacts if present.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_geometry import descriptors as dm
from baseline_geometry import registration as reg
from baseline_geometry import retrieval as ret
from baseline_geometry.config_loader import load_config, dump_config, Config
from baseline_geometry.data_access import PatchTable, PairTable
from baseline_geometry.errors import ConfigValidationError


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_roc_auc_perfect_and_worst():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert ret.roc_auc(scores, labels) == pytest.approx(1.0)
    assert ret.roc_auc(-scores, labels) == pytest.approx(0.0)


def test_roc_auc_chance_on_ties():
    scores = np.ones(6)
    labels = np.array([0, 1, 0, 1, 0, 1])
    assert ret.roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_degenerate_returns_nan():
    assert np.isnan(ret.roc_auc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_average_precision_perfect():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert ret.average_precision(scores, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rotation / translation error + random transform
# ---------------------------------------------------------------------------
def test_identity_has_zero_error():
    T = np.eye(4)
    assert reg.rotation_error_deg(T) == pytest.approx(0.0)
    assert reg.translation_error_mm(T) == pytest.approx(0.0)


def test_random_transform_within_bounds_and_recovery():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = reg.random_rigid_transform(30.0, 20.0, rng)
        # Rotation part is a proper rotation.
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)
        assert reg.rotation_error_deg(T) <= 30.0 + 1e-6
        assert reg.translation_error_mm(T) <= np.sqrt(3) * 20.0 + 1e-6
        # T @ T^{-1} = identity -> zero residual error.
        residual = T @ np.linalg.inv(T)
        assert reg.rotation_error_deg(residual) == pytest.approx(0.0, abs=1e-6)
        assert reg.translation_error_mm(residual) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# SHOT descriptor properties
# ---------------------------------------------------------------------------
def _sphere_points(n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_shot_dimension_and_normalisation():
    pts = _sphere_points(300) * 10.0
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    q = np.arange(20)
    d = dm.compute_shot_points(pts, normals, q, radius_mm=8.0, cos_bins=11)
    assert d.shape == (20, 32 * 11)
    norms = np.linalg.norm(d, axis=1)
    # Each descriptor is L2-normalised (or exactly zero if no neighbours).
    for nrm in norms:
        assert nrm == pytest.approx(1.0) or nrm == pytest.approx(0.0)


def test_shot_rotation_invariance_approx():
    # SHOT with a repeatable LRF should be approximately rotation invariant.
    pts = _sphere_points(500, seed=1) * 10.0
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    q = np.array([0])
    d0 = dm.compute_shot_points(pts, normals, q, 8.0, 11)[0]

    # Rotate the whole cloud.
    theta = 0.7
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    pts_r = pts @ Rz.T
    normals_r = normals @ Rz.T
    d1 = dm.compute_shot_points(pts_r, normals_r, q, 8.0, 11)[0]
    # Not bit-identical (hard binning), but strongly correlated.
    if np.linalg.norm(d0) > 0 and np.linalg.norm(d1) > 0:
        cos = float(d0 @ d1)
        assert cos > 0.8


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def test_pool_mean_matches_manual():
    # 2 patches over 5 points, 4-dim point descriptors.
    point_desc = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.float64)
    table = PatchTable(
        fragment_id="f",
        patch_ids=np.array([0, 1]),
        center_indices=np.array([0, 3]),
        centers=np.zeros((2, 3)),
        offsets=np.array([[0, 3], [3, 5]]),  # not used by pooling directly
        global_coords=np.zeros((5, 3)),
        normals=np.zeros((5, 3)),
        source_indices=np.array([0, 1, 2, 3, 4]),
    )
    pooled = dm.pool_to_patches(point_desc, table, "mean")
    # patch 0 = mean of rows 0,1,2 -> normalised
    manual0 = point_desc[[0, 1, 2]].mean(axis=0)
    manual0 /= np.linalg.norm(manual0)
    assert np.allclose(pooled[0], manual0)


def test_pool_center_selects_center_index():
    point_desc = np.eye(5)[:, :4].astype(np.float64)
    point_desc = np.pad(point_desc, ((0, 0), (0, 0)))
    table = PatchTable(
        fragment_id="f",
        patch_ids=np.array([0]),
        center_indices=np.array([2]),
        centers=np.zeros((1, 3)),
        offsets=np.array([[0, 3]]),
        global_coords=np.zeros((3, 3)),
        normals=np.zeros((3, 3)),
        source_indices=np.array([0, 1, 2]),
    )
    pooled = dm.pool_to_patches(point_desc, table, "center")
    expected = point_desc[2] / np.linalg.norm(point_desc[2])
    assert np.allclose(pooled[0], expected)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_roundtrip(tmp_path):
    cfg_path = os.path.join(ROOT, "config", "baseline_geometry.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config not present")
    cfg = load_config(cfg_path)
    text = dump_config(cfg)
    out = tmp_path / "rt.yaml"
    out.write_text(text)
    cfg2 = load_config(str(out))
    assert cfg2.descriptors == cfg.descriptors
    assert cfg2.fpfh_radius_mm == cfg.fpfh_radius_mm


def test_config_rejects_bad_descriptor(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "dataset_dir: {d}\npatches_dir: {p}\npairs_npz: x\n"
        "pairs_dataset_json: y\noutput_dir: z\ndescriptors: [banana]\n".format(
            d=os.path.join(ROOT, "dataset"), p=os.path.join(ROOT, "patches")
        )
    )
    with pytest.raises(ConfigValidationError):
        load_config(str(bad))


# ---------------------------------------------------------------------------
# Pair-level evaluation on synthetic descriptors
# ---------------------------------------------------------------------------
def test_evaluate_pairs_separates_by_construction():
    # Two fragments, 3 patches each. Positives: identical vectors. Negatives:
    # orthogonal vectors. AUC should be 1.
    from baseline_geometry.descriptors import PatchDescriptors

    da_desc = PatchDescriptors("A", "fpfh", np.array([0, 1, 2]),
                               np.array([[1, 0], [0, 1], [1, 0]], float))
    db_desc = PatchDescriptors("B", "fpfh", np.array([0, 1, 2]),
                               np.array([[1, 0], [0, 1], [0, 1]], float))
    desc = {"A": da_desc, "B": db_desc}

    pairs = PairTable(
        fragment_vocab=np.array(["A", "B"]),
        fragment_A_idx=np.array([0, 0]),
        fragment_B_idx=np.array([1, 1]),
        patch_A_ids=np.array([0, 0]),   # A0=[1,0]
        patch_B_ids=np.array([0, 1]),   # B0=[1,0] (pos, identical), B1=[0,1] (neg, orthogonal)
        labels=np.array([1, -1]),
        contact_overlap_A=np.array([0.9, 0.0]),
        contact_overlap_B=np.array([0.9, 0.0]),
        center_dist_mm=np.array([0.5, 50.0]),
    )
    res = ret.evaluate_pairs(pairs, desc, sample=0)
    assert res["n_pairs_evaluated"] == 2
    assert res["roc_auc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration (only if real artifacts exist)
# ---------------------------------------------------------------------------
def _artifacts_present() -> bool:
    return os.path.isdir(os.path.join(ROOT, "dataset", "normalized")) and os.path.isdir(
        os.path.join(ROOT, "patches")
    )


@pytest.mark.skipif(not _artifacts_present(), reason="Phase 1-2 artifacts not present")
def test_real_fragment_descriptor_shapes():
    from baseline_geometry import data_access as da
    cfg = load_config(os.path.join(ROOT, "config", "baseline_geometry.yaml"))
    fid = da.load_fragment_ids(cfg.dataset_dir)[0]
    cloud = da.load_fragment_cloud(cfg.dataset_dir, fid)
    table = da.load_patch_table(cfg.patches_dir, fid)
    normals = dm.ensure_normals(cloud, cfg.normal_radius_mm, cfg.normal_max_nn)
    fpfh = dm.compute_patch_descriptors("fpfh", cloud, table, normals, cfg)
    assert fpfh.descriptors.shape == (table.patch_count, 33)
    assert np.all(np.isfinite(fpfh.descriptors))
