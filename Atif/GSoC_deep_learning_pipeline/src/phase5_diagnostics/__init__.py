"""Phase 5 pre-registration diagnostic infrastructure.

This package is *instrumentation*, not a model. It exists so that any future
Phase 5 retrieval result can be interpreted scientifically rather than merely
statistically. Every function here operates on a generic embedding source --
``dict[str, PatchDescriptors]`` mapping ``fragment_id -> PatchDescriptors`` --
which is exactly the object the learned encoder will emit. That means the same
diagnostics run unchanged on:

* an untrained *random* embedding (null floor),
* a trivial *centroid / point-count* feature (weak baseline),
* the Phase 4 *FPFH* descriptors (the handcrafted bar to beat),
* any future *learned* checkpoint.

Diagnostic items (see the module docstrings and ``PHASE5_PREREGISTRATION.md``):

1. Null / weak baseline calibration              -> ``ranking.calibrate_null``
2. Orthogonal shortcut-detection battery         -> ``probes`` + ``ranking``
3. Hard-negative stratification (FPFH-mined)     -> ``ranking.easy_vs_hard``
4. Leave-one-fragment-out per-fold retrieval     -> ``ranking.lofo_per_fold``

The orchestrator ``runner.run_diagnostics`` wires them together and returns a
single JSON-serialisable report.
"""

from __future__ import annotations

__all__ = [
    "context",
    "embeddings",
    "ranking",
    "probes",
    "runner",
]
