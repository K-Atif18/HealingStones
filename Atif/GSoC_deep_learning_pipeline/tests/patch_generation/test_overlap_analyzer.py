"""Property-based tests for :mod:`patch_generation.overlap_analyzer`.

Covers Property 10 (overlap fraction bounds, self-overlap, disjoint overlap)
and the summary-statistics behavior of :func:`analyze_overlap`.
"""

from __future__ import annotations

import types

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from patch_generation.overlap_analyzer import (
    OverlapStats,
    analyze_overlap,
    overlap_fraction,
)

# Small integer source-index sets keep the input space dense enough that
# intersections/disjointness are both exercised across examples.
_index_sets = st.sets(st.integers(min_value=0, max_value=20), max_size=12)
_nonempty_index_sets = st.sets(st.integers(min_value=0, max_value=20), min_size=1, max_size=12)


def _patch(indices):
    """Lightweight stand-in exposing only ``.source_indices`` (an np.array)."""

    return types.SimpleNamespace(source_indices=np.array(sorted(indices), dtype=np.int64))


# Feature: patch-generation, Property 10: Overlap fraction is bounded, self is 1.0, disjoint is 0.0
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
@given(a=_nonempty_index_sets, b=_index_sets)
def test_overlap_fraction_bounds_self_and_disjoint(a, b):
    frac = overlap_fraction(a, b)

    # Req 5.2: bounded in [0, 1].
    assert 0.0 <= frac <= 1.0

    # Req 5.4: self-overlap is exactly 1.0.
    assert overlap_fraction(a, a) == 1.0

    # Req 5.3: disjoint sets overlap 0.0.
    if a.isdisjoint(b):
        assert frac == 0.0

    # Reference definition: |A ∩ B| / |A|.
    expected = len(a & b) / len(a)
    assert frac == expected


# Feature: patch-generation, Property 10: Overlap fraction is bounded, self is 1.0, disjoint is 0.0
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
@given(sets=st.lists(_nonempty_index_sets, min_size=0, max_size=6))
def test_analyze_overlap_summary_stats_bounded(sets):
    patches = [_patch(s) for s in sets]
    stats = analyze_overlap(patches)

    assert isinstance(stats, OverlapStats)
    # Req 5.1: pair count is a non-negative tally of analyzed fractions.
    assert stats.analyzed_pair_count >= 0
    # Req 5.5 / 5.2: summary stats stay within the fraction bounds.
    assert 0.0 <= stats.mean_overlap <= 1.0
    assert 0.0 <= stats.max_overlap <= 1.0


# Feature: patch-generation, Property 10: Overlap fraction is bounded, self is 1.0, disjoint is 0.0
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
@given(
    data=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=6, unique=True)
)
def test_analyze_overlap_pairwise_disjoint_yields_no_pairs(data):
    # Build patches with pairwise-disjoint index sets by giving each patch its
    # own isolated block of indices (spaced far apart, no shared points).
    patches = [_patch({base * 1000, base * 1000 + 1}) for base in data]
    stats = analyze_overlap(patches)

    # Req 5.1 / 5.5: no shared points => no analyzed pairs, stats default to 0.0.
    assert stats.analyzed_pair_count == 0
    assert stats.mean_overlap == 0.0
    assert stats.max_overlap == 0.0


# Feature: patch-generation, Property 10: Overlap fraction is bounded, self is 1.0, disjoint is 0.0
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
@given(indices=_nonempty_index_sets)
def test_analyze_overlap_identical_patches_max_is_one(indices):
    # Two identical patches fully overlap => max overlap fraction 1.0.
    patches = [_patch(indices), _patch(indices)]
    stats = analyze_overlap(patches)

    assert stats.analyzed_pair_count > 0
    assert stats.max_overlap == 1.0
    assert 0.0 <= stats.mean_overlap <= 1.0
