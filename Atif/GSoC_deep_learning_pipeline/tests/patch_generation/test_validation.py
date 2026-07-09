"""Property-based and unit tests for validation.

Feature: patch-generation
Module: validation
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np

from patch_generation.validation import (
    size_summary,
    validate_fragment,
    overall_pass,
    SizeSummary,
    FragmentValidation,
)
from patch_generation.config_loader import Config


# ----------------------- Property 14: Coverage Validation -----------------------
# Feature: patch-generation, Property 14: Coverage validation decision.

@given(
    coverage_fraction=st.floats(min_value=0.0, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    uncovered_count=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_14_coverage_validation_decision(
    coverage_fraction, threshold, uncovered_count
):
    """Property 14: Coverage validation decision based on threshold.
    
    Validates Requirements 9.1, 9.2.
    """
    # Mock coverage result
    coverage = type('CoverageResult', (), {
        'coverage_fraction': coverage_fraction,
        'uncovered_count': uncovered_count,
    })()
    
    # Mock config with just the needed fields
    config = type('Config', (), {
        'coverage_threshold': threshold,
        'min_patch_size': 1,
        'max_patch_points': 1000,
    })()
    
    result = validate_fragment(coverage, [], config, fragment_id="test_frag")
    
    # Decision: pass iff coverage_fraction >= threshold (Req 9.1)
    expected_pass = coverage_fraction >= threshold
    assert result.coverage_pass == expected_pass, \
        f"Expected coverage_pass={expected_pass} for fraction={coverage_fraction}, threshold={threshold}"
    
    # Reported values match input (Req 9.2)
    assert result.coverage_fraction == coverage_fraction
    assert result.uncovered_count == uncovered_count
    assert result.fragment_id == "test_frag"


# ----------------------- Property 15: Size Distribution -----------------------
# Feature: patch-generation, Property 15: Size distribution summary and bounds decision.

@given(
    sizes=st.lists(st.integers(min_value=0, max_value=500), min_size=1, max_size=100),
    min_threshold=st.integers(min_value=1, max_value=10),
    max_threshold=st.integers(min_value=50, max_value=200),
)
@settings(max_examples=100)
def test_property_15_size_distribution_summary_and_bounds(
    sizes, min_threshold, max_threshold
):
    """Property 15: Size distribution summary and bounds decision.
    
    Validates Requirements 9.3, 9.4.
    """
    # Build mock patches with given sizes
    patches = []
    for size in sizes:
        patch = type('Patch', (), {
            'source_indices': np.arange(size),
            'global_coords': np.zeros((size, 3)),
        })()
        patches.append(patch)
    
    # Test size_summary separately (Req 9.3)
    summary = size_summary(patches)
    
    assert summary.min_size == min(sizes)
    assert summary.max_size == max(sizes)
    assert summary.mean_size == np.mean(sizes)
    assert summary.median_size == np.median(sizes)
    
    # Mock coverage and config
    coverage = type('CoverageResult', (), {
        'coverage_fraction': 1.0,
        'uncovered_count': 0,
    })()
    
    config = type('Config', (), {
        'coverage_threshold': 1.0,
        'min_patch_size': min_threshold,
        'max_patch_points': max_threshold,
    })()
    
    result = validate_fragment(coverage, patches, config, fragment_id="test")
    
    # Size pass decision and offending bound (Req 9.4)
    expected_min_violation = summary.min_size < min_threshold
    expected_max_violation = summary.max_size > max_threshold
    
    if expected_min_violation:
        assert not result.size_pass
        assert result.offending_bound == "min_patch_size"
    elif expected_max_violation:
        assert not result.size_pass
        assert result.offending_bound == "max_patch_points"
    else:
        assert result.size_pass
        assert result.offending_bound is None
    
    # Summary is recorded
    assert result.size_summary.min_size == summary.min_size
    assert result.size_summary.max_size == summary.max_size


# ----------------------- Property 16: Overall Validation -----------------------
# Feature: patch-generation, Property 16: Overall validation aggregation.

@given(
    num_fragments=st.integers(min_value=0, max_value=20),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100)
def test_property_16_overall_validation_aggregation(num_fragments, seed):
    """Property 16: Overall validation passes iff every processed non-empty fragment passes.
    
    Validates Requirement 9.6.
    """
    rng = np.random.default_rng(seed)
    
    results = []
    skipped_ids = set()
    
    # Generate random validation results and skip decisions
    for i in range(num_fragments):
        frag_id = f"frag_{i}"
        
        # Random skip decision
        if rng.random() < 0.2:  # 20% skip rate
            skipped_ids.add(frag_id)
        
        # Random pass/fail
        coverage_pass = rng.random() > 0.3
        size_pass = rng.random() > 0.3
        
        result = FragmentValidation(
            fragment_id=frag_id,
            coverage_pass=coverage_pass,
            coverage_fraction=0.9 if coverage_pass else 0.5,
            uncovered_count=0 if coverage_pass else 100,
            size_pass=size_pass,
            size_summary=SizeSummary(1, 100, 50.0, 50.0),
            offending_bound=None if size_pass else "min_patch_size",
        )
        results.append(result)
    
    overall = overall_pass(results, skipped_ids)
    
    # Compute expected: every non-skipped fragment must pass both checks
    expected = all(
        r.coverage_pass and r.size_pass
        for r in results
        if r.fragment_id not in skipped_ids
    )
    
    assert overall == expected, \
        f"Expected overall_pass={expected}, got {overall}"
    
    # Edge case: no processed fragments (all skipped) is vacuously True
    if all(r.fragment_id in skipped_ids for r in results):
        assert overall is True


# ----------------------- Unit Tests -----------------------

def test_size_summary_empty_patches_edge_case():
    """Empty patch list should not crash."""
    summary = size_summary([])
    # When no patches, could return zeros or raise; current impl returns 0,0,nan,nan
    # Just check it doesn't crash
    assert isinstance(summary, SizeSummary)


def test_size_summary_single_patch():
    """Single patch size summary."""
    patch = type('Patch', (), {
        'source_indices': np.arange(42),
        'global_coords': np.zeros((42, 3)),
    })()
    summary = size_summary([patch])
    assert summary.min_size == 42
    assert summary.max_size == 42
    assert summary.mean_size == 42.0
    assert summary.median_size == 42.0


def test_validate_fragment_coverage_fail_reports_shortfall():
    """Coverage failure reports the fraction and uncovered count."""
    coverage = type('CoverageResult', (), {
        'coverage_fraction': 0.75,
        'uncovered_count': 250,
    })()
    
    config = type('Config', (), {
        'coverage_threshold': 1.0,
        'min_patch_size': 1,
        'max_patch_points': 1000,
    })()
    
    patch = type('Patch', (), {
        'source_indices': np.arange(10),
        'global_coords': np.zeros((10, 3)),
    })()
    
    result = validate_fragment(coverage, [patch], config, fragment_id="test")
    
    assert not result.coverage_pass
    assert result.coverage_fraction == 0.75
    assert result.uncovered_count == 250


def test_validate_fragment_size_fail_min_reports_offending_bound():
    """Size failure due to min_patch_size reports correct bound."""
    coverage = type('CoverageResult', (), {
        'coverage_fraction': 1.0,
        'uncovered_count': 0,
    })()
    
    config = type('Config', (), {
        'coverage_threshold': 1.0,
        'min_patch_size': 10,
        'max_patch_points': 1000,
    })()
    
    # Patch with only 5 points, below min
    patch = type('Patch', (), {
        'source_indices': np.arange(5),
        'global_coords': np.zeros((5, 3)),
    })()
    
    result = validate_fragment(coverage, [patch], config)
    
    assert not result.size_pass
    assert result.offending_bound == "min_patch_size"


def test_validate_fragment_size_fail_max_reports_offending_bound():
    """Size failure due to max_patch_points reports correct bound."""
    coverage = type('CoverageResult', (), {
        'coverage_fraction': 1.0,
        'uncovered_count': 0,
    })()
    
    config = type('Config', (), {
        'coverage_threshold': 1.0,
        'min_patch_size': 1,
        'max_patch_points': 50,
    })()
    
    # Patch with 100 points, above max
    patch = type('Patch', (), {
        'source_indices': np.arange(100),
        'global_coords': np.zeros((100, 3)),
    })()
    
    result = validate_fragment(coverage, [patch], config)
    
    assert not result.size_pass
    assert result.offending_bound == "max_patch_points"


def test_validate_fragment_both_pass():
    """Both coverage and size pass."""
    coverage = type('CoverageResult', (), {
        'coverage_fraction': 1.0,
        'uncovered_count': 0,
    })()
    
    config = type('Config', (), {
        'coverage_threshold': 1.0,
        'min_patch_size': 1,
        'max_patch_points': 100,
    })()
    
    patch = type('Patch', (), {
        'source_indices': np.arange(50),
        'global_coords': np.zeros((50, 3)),
    })()
    
    result = validate_fragment(coverage, [patch], config)
    
    assert result.coverage_pass
    assert result.size_pass
    assert result.offending_bound is None


def test_overall_pass_all_pass():
    """All fragments pass both checks."""
    results = [
        FragmentValidation(
            fragment_id="f1",
            coverage_pass=True,
            coverage_fraction=1.0,
            uncovered_count=0,
            size_pass=True,
            size_summary=SizeSummary(10, 100, 50.0, 50.0),
            offending_bound=None,
        ),
        FragmentValidation(
            fragment_id="f2",
            coverage_pass=True,
            coverage_fraction=1.0,
            uncovered_count=0,
            size_pass=True,
            size_summary=SizeSummary(10, 100, 50.0, 50.0),
            offending_bound=None,
        ),
    ]
    
    assert overall_pass(results, []) is True


def test_overall_pass_one_fails():
    """One fragment fails, overall fails."""
    results = [
        FragmentValidation(
            fragment_id="f1",
            coverage_pass=True,
            coverage_fraction=1.0,
            uncovered_count=0,
            size_pass=True,
            size_summary=SizeSummary(10, 100, 50.0, 50.0),
            offending_bound=None,
        ),
        FragmentValidation(
            fragment_id="f2",
            coverage_pass=False,  # FAIL
            coverage_fraction=0.5,
            uncovered_count=100,
            size_pass=True,
            size_summary=SizeSummary(10, 100, 50.0, 50.0),
            offending_bound=None,
        ),
    ]
    
    assert overall_pass(results, []) is False


def test_overall_pass_skipped_fragments_excluded():
    """Skipped fragments are excluded from aggregate."""
    results = [
        FragmentValidation(
            fragment_id="f1",
            coverage_pass=True,
            coverage_fraction=1.0,
            uncovered_count=0,
            size_pass=True,
            size_summary=SizeSummary(10, 100, 50.0, 50.0),
            offending_bound=None,
        ),
        FragmentValidation(
            fragment_id="f2_skipped",
            coverage_pass=False,  # Would fail, but skipped
            coverage_fraction=0.0,
            uncovered_count=1000,
            size_pass=False,
            size_summary=SizeSummary(0, 0, 0.0, 0.0),
            offending_bound="min_patch_size",
        ),
    ]
    
    # f2 is skipped, so only f1 counts, which passes
    assert overall_pass(results, ["f2_skipped"]) is True


def test_overall_pass_empty_is_vacuously_true():
    """No processed fragments is vacuously True."""
    assert overall_pass([], []) is True
