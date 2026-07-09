"""Property-based and unit tests for coverage_analyzer.

Feature: patch-generation
Module: coverage_analyzer
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from patch_generation.coverage_analyzer import analyze_coverage, CoverageResult


# ----------------------- Property 11: Coverage Analysis -----------------------
# Feature: patch-generation, Property 11: Coverage fraction equals the union
# fraction with consistent uncovered accounting.

@given(
    # Generate random patches with source_indices
    num_patches=st.integers(min_value=0, max_value=50),
    fragment_size=st.integers(min_value=0, max_value=200),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100)
def test_property_11_coverage_union_fraction_and_uncovered_accounting(
    num_patches, fragment_size, seed
):
    """Property 11: Coverage fraction equals union fraction, uncovered accounting consistent.
    
    Validates Requirements 6.1, 6.2, 6.3, 6.4, 6.5.
    """
    rng = np.random.default_rng(seed)
    
    # Build mock patches
    patches = []
    for _ in range(num_patches):
        if fragment_size == 0:
            indices = np.array([], dtype=np.int64)
        else:
            # Random subset of fragment points
            size = rng.integers(0, min(fragment_size + 1, 30))
            indices = rng.choice(fragment_size, size=size, replace=False)
        patches.append(type('Patch', (), {'source_indices': indices})())
    
    result = analyze_coverage(patches, fragment_size)
    
    # Property: coverage_fraction in [0, 1] (Req 6.2, 6.3)
    assert 0.0 <= result.coverage_fraction <= 1.0, \
        f"coverage_fraction {result.coverage_fraction} out of bounds [0, 1]"
    
    # Compute expected coverage via brute-force union
    union = set()
    for patch in patches:
        union.update(patch.source_indices.tolist())
    
    # Only valid indices count
    valid_union = {i for i in union if 0 <= i < fragment_size}
    expected_covered_count = len(valid_union)
    
    if fragment_size == 0:
        # Zero-point fragment: defined as fully covered (Req 6.3 edge case)
        assert result.coverage_fraction == 1.0
        assert result.covered_count == 0
        assert result.uncovered_count == 0
        assert result.uncovered_indices == []
    else:
        expected_fraction = expected_covered_count / fragment_size
        assert abs(result.coverage_fraction - expected_fraction) < 1e-9, \
            f"Expected fraction {expected_fraction}, got {result.coverage_fraction}"
        
        assert result.covered_count == expected_covered_count
        
        # Uncovered accounting (Req 6.4)
        expected_uncovered = sorted(set(range(fragment_size)) - valid_union)
        assert result.uncovered_indices == expected_uncovered
        assert result.uncovered_count == len(expected_uncovered)
    
    # Property: coverage 1.0 implies uncovered count 0 (Req 6.5)
    if abs(result.coverage_fraction - 1.0) < 1e-9:
        assert result.uncovered_count == 0
        assert result.uncovered_indices == []


# ----------------------- Unit Tests -----------------------

def test_coverage_empty_patches_zero_coverage():
    """No patches means zero coverage."""
    result = analyze_coverage([], fragment_point_count=100)
    assert result.coverage_fraction == 0.0
    assert result.covered_count == 0
    assert result.uncovered_count == 100
    assert result.uncovered_indices == list(range(100))


def test_coverage_full_coverage_one_patch():
    """Single patch covering all points."""
    patch = type('Patch', (), {'source_indices': np.arange(50)})()
    result = analyze_coverage([patch], fragment_point_count=50)
    assert result.coverage_fraction == 1.0
    assert result.covered_count == 50
    assert result.uncovered_count == 0
    assert result.uncovered_indices == []


def test_coverage_partial_coverage():
    """Patches cover only some points."""
    p1 = type('Patch', (), {'source_indices': np.array([0, 1, 2])})()
    p2 = type('Patch', (), {'source_indices': np.array([2, 3, 4])})()
    result = analyze_coverage([p1, p2], fragment_point_count=10)
    # Covers 0,1,2,3,4 = 5/10 = 0.5
    assert result.coverage_fraction == 0.5
    assert result.covered_count == 5
    assert result.uncovered_count == 5
    assert result.uncovered_indices == [5, 6, 7, 8, 9]


def test_coverage_out_of_bounds_indices_ignored():
    """Patch indices outside [0, N) are ignored."""
    patch = type('Patch', (), {'source_indices': np.array([-1, 0, 1, 100, 200])})()
    result = analyze_coverage([patch], fragment_point_count=10)
    # Only 0, 1 are valid
    assert result.coverage_fraction == 0.2  # 2/10
    assert result.covered_count == 2


def test_coverage_negative_fragment_count_raises():
    """Negative fragment_point_count raises ValueError."""
    with pytest.raises(ValueError, match="fragment_point_count must be >= 0"):
        analyze_coverage([], fragment_point_count=-1)


def test_coverage_zero_fragment_vacuously_covered():
    """Zero-point fragment is defined as fully covered."""
    patch = type('Patch', (), {'source_indices': np.array([])})()
    result = analyze_coverage([patch], fragment_point_count=0)
    assert result.coverage_fraction == 1.0
    assert result.covered_count == 0
    assert result.uncovered_count == 0
    assert result.uncovered_indices == []
