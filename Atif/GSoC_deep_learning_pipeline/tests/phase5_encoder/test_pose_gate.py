"""Phase 5A pose-invariance gate + input-feature invariance tests.

These are the HARD BLOCKER tests from PHASE5_PREREGISTRATION.md section 6: the
encoder's full input+forward pipeline must satisfy
``pose_invariance_check.passed == True`` at tol=1e-3. Training must not proceed
if this fails. The gate function itself is imported unmodified from the existing,
tested diagnostics infrastructure (``phase5_diagnostics.probes``).
"""

from __future__ import annotations

import numpy as np
import torch

from phase5_diagnostics.probes import pose_invariance_check
from phase5_encoder.features import (
    ppf_features, knn_ppf_features, FEATURE_DIM, PAIR_FEATURE_DIM,
)
from phase5_encoder.preprocess import fps_resample
from phase5_encoder.model import PointNetEncoder, build_encode_patch


# The pre-registered tolerance. Do NOT loosen this to make a test pass; any
# change must be justified to the human first (pre-registration discipline).
POSE_GATE_TOL = 1e-3


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ----------------------------------------------------------------------
# Feature-level invariance (the mechanism)
# ----------------------------------------------------------------------
def test_ppf_features_are_rigid_invariant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pts = rng.standard_normal((80, 3)) * 4.0
        nrm = rng.standard_normal((80, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        f0 = ppf_features(pts, nrm)

        Rt = _rand_rotation(rng)
        t = rng.standard_normal(3) * 50.0
        f1 = ppf_features(pts @ Rt.T + t, nrm @ Rt.T)

        assert f0.shape == (80, FEATURE_DIM)
        assert np.allclose(f0, f1, atol=1e-9), np.abs(f0 - f1).max()


def test_ppf_cosines_bounded():
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((50, 3)) * 3.0
    nrm = rng.standard_normal((50, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    f = ppf_features(pts, nrm)
    assert np.all(f[:, 1:] >= -1.0 - 1e-9) and np.all(f[:, 1:] <= 1.0 + 1e-9)
    assert np.all(f[:, 0] >= 0.0)  # radial distance non-negative


def test_knn_ppf_features_are_rigid_invariant():
    """The k-NN pairwise PPF features must be exactly rigid-invariant too."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        pts = rng.standard_normal((90, 3)) * 4.0
        nrm = rng.standard_normal((90, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        f0 = knn_ppf_features(pts, nrm, k=16)

        Rt = _rand_rotation(rng)
        t = rng.standard_normal(3) * 50.0
        f1 = knn_ppf_features(pts @ Rt.T + t, nrm @ Rt.T, k=16)

        assert f0.shape == (90, 16, PAIR_FEATURE_DIM)
        assert np.allclose(f0, f1, atol=1e-9), np.abs(f0 - f1).max()


def test_knn_ppf_handles_small_patch():
    """Fewer points than k must not crash; output shape stays (N,k,4)."""
    rng = np.random.default_rng(4)
    pts = rng.standard_normal((5, 3))
    nrm = rng.standard_normal((5, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    f = knn_ppf_features(pts, nrm, k=16)
    assert f.shape == (5, 16, PAIR_FEATURE_DIM)


# ----------------------------------------------------------------------
# FPS index selection is frame-independent
# ----------------------------------------------------------------------
def test_fps_selection_is_frame_independent():
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((120, 3)) * 5.0
    nrm = rng.standard_normal((120, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    p0, n0 = fps_resample(pts, nrm, 64)
    Rt = _rand_rotation(rng)
    t = rng.standard_normal(3) * 30.0
    p1, n1 = fps_resample(pts @ Rt.T + t, nrm @ Rt.T, 64)

    # Same indices selected => p1 == R p0 + t exactly (up to fp error).
    assert np.allclose(p1, p0 @ Rt.T + t, atol=1e-9)
    assert np.allclose(n1, n0 @ Rt.T, atol=1e-9)


# ----------------------------------------------------------------------
# THE HARD GATE: full untrained pipeline through the pre-registered check
# ----------------------------------------------------------------------
def test_pose_invariance_gate_untrained_encoder():
    torch.manual_seed(0)
    model = PointNetEncoder(out_dim=64, pair_input=True)   # the real Phase 5A path
    # Random (untrained) weights: invariance must hold regardless of weights,
    # since it comes from the PPF input features, not from training.
    encode_patch = build_encode_patch(model, n_points=64, k=16, device="cpu")

    result = pose_invariance_check(
        encode_patch, n_trials=50, n_points=100, tol=POSE_GATE_TOL, seed=0
    )
    assert result["passed"], (
        f"POSE GATE FAILED: max_dev={result['max_deviation']:.3e} "
        f"> tol={POSE_GATE_TOL:.0e}. Training must not proceed. "
        f"Do NOT loosen tol to pass -- escalate."
    )


def test_pose_invariance_gate_single_reference_ablation():
    """The single-reference (N,4) variant must also pass, for the ablation path."""
    torch.manual_seed(0)
    model = PointNetEncoder(out_dim=64, pair_input=False)
    encode_patch = build_encode_patch(model, n_points=64, device="cpu")
    result = pose_invariance_check(
        encode_patch, n_trials=30, n_points=100, tol=POSE_GATE_TOL, seed=0
    )
    assert result["passed"], result
