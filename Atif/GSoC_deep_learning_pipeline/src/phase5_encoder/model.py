"""A small PointNet patch encoder + the ``encode_patch`` factory.

Architecture (deliberately the simplest option first, per the phase plan):
    per-point PPF features (N,4)
      -> shared MLP (4 -> 64 -> 128 -> 256)   [1x1 conv == per-point Linear]
      -> symmetric max-pool over points        (permutation invariant)
      -> MLP head (256 -> 128 -> out_dim)
      -> L2 normalise                          (matches PatchDescriptors contract)

Pose invariance is inherited from the PPF input features (see ``features.py``);
there is intentionally **no T-Net** -- a learned spatial transformer would break
the by-construction invariance and reintroduce the exact failure mode the pose
gate exists to catch.

``build_encode_patch`` returns a ``encode_patch(points, normals) -> vector``
callable wrapping the *entire* input pipeline + forward pass. This is the object
the pre-registered ``pose_invariance_check`` gates on, and the same callable is
later used to export ``PatchDescriptors`` through the unmodified diagnostic
battery. One callable, one code path, no per-consumer forks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from phase5_encoder.features import ppf_features, knn_ppf_features, FEATURE_DIM, PAIR_FEATURE_DIM
from phase5_encoder.preprocess import prepare_patch


__all__ = ["PointNetEncoder", "build_encode_patch"]


class PointNetEncoder(nn.Module):
    """PointNet over k-NN pairwise PPF features: (B, N, k, 4) -> (B, out_dim).

    Two-stage symmetric pooling (PPFNet-style):
      per-pair MLP (4 -> 64 -> 128)  -> max-pool over k neighbours -> per-point (128)
      per-point MLP (128 -> 256)     -> max-pool over N points     -> patch (256)
      head (256 -> 128 -> out_dim)   -> L2 normalise

    Invariance is inherited from the PPF input (each pair feature is exactly
    rigid-invariant); there is deliberately **no T-Net**. Set ``pair_input=False``
    to consume the single-reference (N,4) features instead (kept for ablation).

    Uses ``nn.LayerNorm`` rather than ``nn.BatchNorm1d`` (see
    PHASE5A_TRAINING_DESIGN_REVISED.md item 6): BatchNorm's eval-mode running
    statistics would be fit only on the six LOFO training fragments, so a
    held-out fragment's distribution shift becomes indistinguishable from a
    genuine interface-association failure -- the same confound shape already
    used to justify PPF over PCA canonicalisation in Deviation 2. LayerNorm
    normalises each row over its own feature dimension, independent of any
    other row in the batch, which removes both the LOFO confound and the
    batch-size dependence in one change.
    """

    def __init__(self, out_dim: int = 64, pair_input: bool = True):
        super().__init__()
        self.pair_input = pair_input
        in_dim = PAIR_FEATURE_DIM if pair_input else FEATURE_DIM
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128),
        )
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, N, k, in_dim) if pair_input else (B, N, in_dim)."""
        if self.pair_input:
            b, n, k, d = feats.shape
            x = feats.reshape(b * n * k, d)
            x = self.pair_mlp(x)                 # (B*N*k, 128)
            x = x.reshape(b * n, k, -1).max(dim=1).values   # pool neighbours -> (B*N,128)
            x = self.point_mlp(x)                # (B*N, 256)
            x = x.reshape(b, n, -1).max(dim=1).values       # pool points -> (B,256)
        else:
            b, n, d = feats.shape
            x = feats.reshape(b * n, d)
            x = self.pair_mlp(x)
            x = self.point_mlp(x)
            x = x.reshape(b, n, -1).max(dim=1).values
        x = self.head(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def build_encode_patch(
    model: PointNetEncoder,
    *,
    n_points: int,
    k: int = 16,
    device: str = "cpu",
    training: bool = False,
    seed: int = 0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Wrap the full pipeline into ``encode_patch(points, normals) -> (out_dim,)``.

    Deterministic and eval-mode by default. With LayerNorm (see model docstring)
    eval-mode is not strictly required for single-patch inference the way it
    was under BatchNorm, but the model is still switched to eval() here for
    determinism (disables dropout, were any added later) and to keep this
    function's behaviour independent of the caller's train/eval state.
    Used by the pose gate and the exporter alike. Feature kind (pairwise vs
    single-reference) follows ``model.pair_input``.
    """
    def encode_patch(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        p, nrm = prepare_patch(
            points, normals, n_points, training=training, seed=seed
        )
        if model.pair_input:
            feats = knn_ppf_features(p, nrm, k=k)           # (N,k,4)
            t = torch.from_numpy(feats).float().unsqueeze(0).to(device)  # (1,N,k,4)
        else:
            feats = ppf_features(p, nrm)                    # (N,4)
            t = torch.from_numpy(feats).float().unsqueeze(0).to(device)  # (1,N,4)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(t).cpu().numpy()[0]
        if was_training:
            model.train()
        return out

    return encode_patch
