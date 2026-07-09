"""Phase 4: Baseline Geometry.

Non-learning geometric baselines for fragment assembly:

* :mod:`baseline_geometry.descriptors` -- FPFH and SHOT patch descriptors.
* :mod:`baseline_geometry.retrieval` -- descriptor-based patch retrieval and
  pair-level separability metrics against the Phase 3 ground truth.
* :mod:`baseline_geometry.registration` -- FPFH + RANSAC + ICP rigid
  registration of adjacent fragment pairs, with a non-adjacent control.
* :mod:`baseline_geometry.pipeline` -- orchestration and CLI entry point.

Everything is expressed in millimetres in the Phase 1 assembled coordinate
frame. The package deliberately contains no learned components: it establishes
the reference point that later (deep-learning) phases must justify themselves
against, and it quantifies the *similarity vs complementarity* gap that
motivates the rest of the project.
"""

__all__ = [
    "config_loader",
    "descriptors",
    "retrieval",
    "registration",
    "data_access",
    "pipeline",
]
