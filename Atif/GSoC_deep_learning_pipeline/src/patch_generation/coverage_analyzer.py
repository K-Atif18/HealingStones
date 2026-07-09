"""Surface coverage analysis.

Defines the frozen :class:`CoverageResult` dataclass and
:func:`analyze_coverage`, which determines whether every Fragment point belongs
to at least one Patch by unioning the per-Patch ``source_indices``.

For a Fragment with ``N = fragment_point_count`` points and a collection of
patches:

1. **Union of covered indices.** The set of covered source indices is the union
   of every Patch's ``source_indices`` (Req 6.1). Only indices in the valid
   range ``{0..N-1}`` are counted as covering the Fragment.
2. **Coverage_Fraction.** ``coverage_fraction = covered_count / N``, bounded in
   ``[0.0, 1.0]`` (Req 6.2, 6.3).
3. **Uncovered accounting.** ``uncovered_indices`` is the sorted list of source
   indices in ``{0..N-1}`` covered by no Patch (Req 6.4); ``uncovered_count``
   is its length. When ``coverage_fraction == 1.0`` the uncovered count is zero
   (Req 6.5).
4. **Zero-point Fragment.** When ``N == 0`` coverage is defined as ``1.0`` with
   zero covered and zero uncovered points (vacuously fully covered).

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CoverageResult",
    "analyze_coverage",
]


@dataclass(frozen=True)
class CoverageResult:
    """Coverage summary for a single Fragment.

    Attributes:
        coverage_fraction: Fraction of Fragment points belonging to at least
            one Patch, bounded in ``[0.0, 1.0]`` (Req 6.2, 6.3). Defined as
            ``1.0`` for a zero-point Fragment.
        covered_count: Number of Fragment points belonging to at least one
            Patch.
        uncovered_count: Number of Uncovered_Points (Fragment points belonging
            to no Patch); zero when ``coverage_fraction == 1.0`` (Req 6.5).
        uncovered_indices: Sorted list of source Fragment point indices of the
            Uncovered_Points (Req 6.4).
    """

    coverage_fraction: float
    covered_count: int
    uncovered_count: int
    uncovered_indices: list[int]


def analyze_coverage(patches, fragment_point_count: int) -> CoverageResult:
    """Analyze surface coverage for a Fragment's patches.

    Args:
        patches: Iterable of :class:`~patch_generation.patch_extractor.Patch`
            objects (anything exposing a ``source_indices`` attribute) for the
            Fragment.
        fragment_point_count: The Fragment's total point count ``N`` (``>= 0``).

    Returns:
        A :class:`CoverageResult` recording the Coverage_Fraction, covered and
        uncovered counts, and the sorted uncovered source indices.

    Raises:
        ValueError: if ``fragment_point_count`` is negative.
    """
    n = int(fragment_point_count)
    if n < 0:
        raise ValueError(
            f"fragment_point_count must be >= 0, got {fragment_point_count}"
        )

    # Union of all covered source indices across patches (Req 6.1).
    union: set[int] = set()
    for patch in patches:
        idx = np.asarray(patch.source_indices, dtype=np.int64).reshape(-1)
        union.update(int(i) for i in idx.tolist())

    # Zero-point Fragment: vacuously fully covered (coverage defined as 1.0).
    if n == 0:
        return CoverageResult(
            coverage_fraction=1.0,
            covered_count=0,
            uncovered_count=0,
            uncovered_indices=[],
        )

    all_indices = set(range(n))
    # Only indices within the valid range count toward coverage.
    covered = union & all_indices
    covered_count = len(covered)

    # coverage_fraction in [0, 1] (Req 6.2, 6.3).
    coverage_fraction = covered_count / n

    # Uncovered accounting (Req 6.4); sorted ascending list of int.
    uncovered_indices = sorted(all_indices - covered)
    uncovered_count = len(uncovered_indices)

    return CoverageResult(
        coverage_fraction=coverage_fraction,
        covered_count=covered_count,
        uncovered_count=uncovered_count,
        uncovered_indices=uncovered_indices,
    )
