"""Patch Generation (Phase 2) package.

Converts each Phase 1 Fragment (aligned, density-standardized point cloud with
per-point normals under ``dataset/``) into thousands of overlapping radius-based
patches that fully cover the Fragment surface, quantifies overlap and coverage,
serializes a per-Fragment patch dataset plus metadata, renders coverage/overlap
visualizations, and enforces a validation checkpoint (full coverage + sane size
distribution).

Mirrors the Phase 1 ``dataset_foundation`` module layout. See the design
document for module-level responsibilities.
"""

__version__ = "0.1.0"
