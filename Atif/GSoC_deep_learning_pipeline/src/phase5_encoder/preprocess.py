"""Input pipeline: density normalisation (fixed-count resample) + jitter.

Density normalisation is *mandatory* for this encoder independent of anything
else: the Phase 5 diagnostics measured a density/spacing fingerprint that
predicts fragment identity at 0.356 (chance 0.143), a back-door to the
fragment-ID shortcut. Resampling every patch to a fixed point count with FPS
removes per-fragment point-count/spacing signatures before the network sees them.

Frame-independence
------------------
Both operations are used *upstream* of the invariant PPF features, so they must
not reintroduce pose sensitivity:

* ``fps_resample`` selects points by Euclidean distances between points, which
  are preserved under any rigid transform, and seeds from the point farthest
  from the centroid (an invariant choice). Hence the *set of selected indices*
  is identical for a patch and its rotated/translated copy -> composes cleanly
  with the pose gate.
* ``jitter`` is training-only (Gaussian noise on coordinates); it is disabled on
  the eval/export path and in the pose-invariance check.
"""

from __future__ import annotations

import numpy as np


__all__ = ["fps_resample", "jitter", "prepare_patch"]


def fps_resample(
    points: np.ndarray,
    normals: np.ndarray,
    n_out: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Farthest-point-sample (or pad) a patch to exactly ``n_out`` points.

    The FPS seed point is the point farthest from the patch centroid -- a
    rigid-invariant choice -- so selection is deterministic and frame
    independent. If the patch has fewer than ``n_out`` points, indices are
    repeated (sampled with replacement) to pad; this preserves the geometry
    distribution without inventing new coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    m = points.shape[0]
    if m == 0:
        raise ValueError("cannot resample an empty patch")

    if m >= n_out:
        sel = _fps_indices(points, n_out)
    else:
        # Keep all points, then pad by repeating (deterministic RNG).
        rng = np.random.default_rng(seed)
        pad = rng.integers(0, m, size=n_out - m)
        sel = np.concatenate([np.arange(m), pad])

    return points[sel], normals[sel]


def _fps_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Farthest point sampling returning ``k`` indices into ``points``.

    Deterministic: starts from the point farthest from the centroid.
    """
    n = points.shape[0]
    c = points.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(points - c, axis=1)))

    selected = np.empty(k, dtype=np.int64)
    selected[0] = start
    dist = np.linalg.norm(points - points[start], axis=1)
    for i in range(1, k):
        nxt = int(np.argmax(dist))
        selected[i] = nxt
        d = np.linalg.norm(points - points[nxt], axis=1)
        dist = np.minimum(dist, d)
    return selected


def jitter(
    points: np.ndarray, sigma: float = 0.1, clip: float = 0.3, *, seed: int | None = None
) -> np.ndarray:
    """Add clipped Gaussian noise to coordinates (training augmentation only).

    ``sigma``/``clip`` are in millimetres (dataset unit). Disabled on eval.
    """
    rng = np.random.default_rng(seed)
    noise = np.clip(rng.normal(0.0, sigma, size=points.shape), -clip, clip)
    return points + noise


def prepare_patch(
    points: np.ndarray,
    normals: np.ndarray,
    n_out: int,
    *,
    training: bool = False,
    jitter_sigma: float = 0.1,
    seed: int = 0,
    jitter_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full input preprocessing: FPS resample (+ optional jitter if training).

    Returns ``(points_out, normals_out)`` both ``(n_out, 3)``. Jitter is applied
    *after* resampling and only when ``training`` is True, so the eval/export and
    pose-gate paths are deterministic and frame independent.

    ``seed`` drives the FPS resample (only its pad branch actually uses it -- the
    main farthest-point path is argmax-from-centroid, RNG-free -- and it must
    stay deterministic for the pose gate). ``jitter_seed`` drives ONLY the
    training-time jitter noise; if ``None`` it falls back to ``seed`` (preserving
    the original behaviour byte-for-byte for any caller that does not pass it,
    e.g. eval / mining / pose-gate, which all use ``training=False`` anyway).
    Passing a per-(epoch, patch) ``jitter_seed`` is what turns jitter from a
    frozen fixed perturbation into real stochastic augmentation (Deviation 4).
    """
    p, nrm = fps_resample(points, normals, n_out, seed=seed)
    if training:
        js = jitter_seed if jitter_seed is not None else seed
        p = jitter(p, sigma=jitter_sigma, seed=js)
    return p, nrm
