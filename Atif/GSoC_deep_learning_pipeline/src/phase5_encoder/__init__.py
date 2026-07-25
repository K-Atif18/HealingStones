"""Phase 5A: learned patch encoder.

Modules
-------
* ``features``   : rotation+translation-invariant per-point input features
                   (Point Pair Features relative to the patch centroid and mean
                   normal). Invariance is *by construction*, so any network on
                   top satisfies the pose-invariance gate to machine precision --
                   this is deliberately chosen over PCA-frame canonicalisation,
                   which is numerically ill-conditioned on near-isotropic patches
                   (exactly the data the gate stresses).
* ``preprocess`` : density normalisation (FPS resample to a fixed point count) +
                   optional training jitter. The eval path is deterministic and
                   coordinate-frame-independent in its *index selection* so it
                   composes with the invariant features to pass the gate.
* ``model``      : a small PointNet (shared per-point MLP + max-pool) and the
                   ``encode_patch`` factory used both by the pose gate and, later,
                   to export ``PatchDescriptors`` through the unmodified Phase 5
                   diagnostic battery.

Nothing here trains yet; this is the input pipeline + forward pass whose pose
invariance must be demonstrated (``pose_invariance_check.passed == True``,
tol=1e-3) before any training loop is written.
"""

from __future__ import annotations

__all__ = ["features", "preprocess", "model"]
