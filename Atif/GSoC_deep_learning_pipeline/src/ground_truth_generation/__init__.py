"""Ground Truth Generation for Fragment Assembly (Phase 3).

This module generates positive and negative patch pair training data
based on geometric analysis of fragment relationships.

Key components:
- Fragment geometry analysis
- Fragment pair distance computation
- Patch distinctiveness metrics
- Positive pair generation (contact-region based)
- Negative pair generation (tiered: easy/medium/hard)
"""

__version__ = "0.1.0"
