"""Validation checkpoint logic.

Defines the frozen :class:`SizeSummary` and :class:`FragmentValidation`
dataclasses and :func:`size_summary`, :func:`validate_fragment`, and
:func:`overall_pass`, which implement the Phase 2 validation checkpoint:
per-Fragment coverage-threshold and size-distribution checks, and the overall
aggregate outcome.

For a single processed, non-empty Fragment:

1. **Coverage decision.** ``coverage_pass`` is true iff
   ``coverage.coverage_fraction >= config.coverage_threshold`` (Req 9.1). On
   failure the recorded ``coverage_fraction`` and ``uncovered_count`` describe
   the shortfall (Req 9.2).
2. **Size summary.** :func:`size_summary` computes the min/max/mean/median
   Patch point count over the Fragment's patches (Req 9.3).
3. **Size decision.** ``size_pass`` is true iff no Patch is smaller than
   ``config.min_patch_size`` and no Patch is larger than
   ``config.max_patch_points`` (Req 9.4). When it fails, ``offending_bound``
   names the violated bound (``"min_patch_size"`` when ``min_size`` is too
   small, else ``"max_patch_points"`` when ``max_size`` is too large).
4. **Overall aggregation.** :func:`overall_pass` is true iff every processed,
   non-empty Fragment (i.e. every :class:`FragmentValidation` whose
   ``fragment_id`` is not in ``skipped_ids``) passes both checks (Req 9.6).
   Skipped and empty Fragments are excluded from the aggregate.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.6.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "SizeSummary",
    "FragmentValidation",
    "size_summary",
    "validate_fragment",
    "overall_pass",
]


@dataclass(frozen=True)
class SizeSummary:
    """Summary of the Patch size (point count) distribution for a Fragment.

    Attributes:
        min_size: Smallest Patch point count.
        max_size: Largest Patch point count.
        mean_size: Arithmetic mean Patch point count.
        median_size: Median Patch point count.
    """

    min_size: int
    max_size: int
    mean_size: float
    median_size: float


@dataclass(frozen=True)
class FragmentValidation:
    """Validation outcome for a single Fragment.

    Attributes:
        fragment_id: Owning Fragment_ID.
        coverage_pass: True iff ``coverage_fraction >= coverage_threshold``
            (Req 9.1).
        coverage_fraction: The Fragment's Coverage_Fraction (reported on
            success and failure, Req 9.2).
        uncovered_count: Number of Uncovered_Points (Req 9.2).
        size_pass: True iff every Patch size is within
            ``[min_patch_size, max_patch_points]`` (Req 9.4).
        size_summary: The Patch size distribution summary (Req 9.3).
        offending_bound: Name of the violated size bound when ``size_pass`` is
            False (``"min_patch_size"`` or ``"max_patch_points"``); ``None``
            when the size check passes.
    """

    fragment_id: str
    coverage_pass: bool
    coverage_fraction: float
    uncovered_count: int
    size_pass: bool
    size_summary: SizeSummary
    offending_bound: str | None


def size_summary(patches) -> SizeSummary:
    """Summarize the Patch size (point count) distribution (Req 9.3).

    Each Patch's size is ``len(patch.source_indices)``.

    Args:
        patches: Iterable of :class:`~patch_generation.patch_extractor.Patch`
            objects (anything exposing a ``source_indices`` attribute).

    Returns:
        A :class:`SizeSummary` with the min/max/mean/median point count. For an
        empty patch collection every field is zero (``min = max = 0``,
        ``mean = median = 0.0``); note :func:`validate_fragment` is only called
        for non-empty Fragments that have patches.
    """
    sizes = [len(patch.source_indices) for patch in patches]
    if not sizes:
        return SizeSummary(min_size=0, max_size=0, mean_size=0.0, median_size=0.0)

    return SizeSummary(
        min_size=int(min(sizes)),
        max_size=int(max(sizes)),
        mean_size=float(statistics.fmean(sizes)),
        median_size=float(statistics.median(sizes)),
    )


def validate_fragment(coverage, patches, config, fragment_id: str = "") -> FragmentValidation:
    """Validate a single Fragment's coverage and Patch size distribution.

    Args:
        coverage: A
            :class:`~patch_generation.coverage_analyzer.CoverageResult` for the
            Fragment (exposes ``coverage_fraction`` and ``uncovered_count``).
        patches: The Fragment's list of
            :class:`~patch_generation.patch_extractor.Patch` objects.
        config: The :class:`~patch_generation.config_loader.Config` supplying
            ``coverage_threshold``, ``min_patch_size``, and ``max_patch_points``.
        fragment_id: The owning Fragment_ID recorded on the result.

    Returns:
        A :class:`FragmentValidation` describing the coverage and size outcomes.
    """
    # Coverage decision (Req 9.1); fraction + uncovered count reported for both
    # outcomes so a failure describes the shortfall (Req 9.2).
    coverage_fraction = float(coverage.coverage_fraction)
    uncovered_count = int(coverage.uncovered_count)
    coverage_pass = coverage_fraction >= config.coverage_threshold

    # Size distribution summary (Req 9.3).
    summary = size_summary(patches)

    # Size decision + offending bound (Req 9.4).
    if summary.min_size < config.min_patch_size:
        size_pass = False
        offending_bound: str | None = "min_patch_size"
    elif summary.max_size > config.max_patch_points:
        size_pass = False
        offending_bound = "max_patch_points"
    else:
        size_pass = True
        offending_bound = None

    return FragmentValidation(
        fragment_id=str(fragment_id),
        coverage_pass=coverage_pass,
        coverage_fraction=coverage_fraction,
        uncovered_count=uncovered_count,
        size_pass=size_pass,
        size_summary=summary,
        offending_bound=offending_bound,
    )


def overall_pass(results: Iterable[FragmentValidation], skipped_ids: Iterable[str]) -> bool:
    """Aggregate per-Fragment outcomes into the overall checkpoint result.

    Args:
        results: The :class:`FragmentValidation` records for processed
            Fragments.
        skipped_ids: Fragment_IDs that were skipped or empty and are therefore
            excluded from the aggregate.

    Returns:
        True iff every processed, non-empty Fragment (every result whose
        ``fragment_id`` is not in ``skipped_ids``) passes both the coverage and
        the size checks (Req 9.6). Vacuously True when there are no processed
        non-empty Fragments.
    """
    skipped = set(skipped_ids)
    return all(
        result.coverage_pass and result.size_pass
        for result in results
        if result.fragment_id not in skipped
    )
