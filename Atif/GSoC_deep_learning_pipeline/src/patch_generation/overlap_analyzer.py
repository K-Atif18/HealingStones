"""Pairwise Patch overlap analysis.

Defines the frozen :class:`OverlapStats` dataclass and the
:func:`overlap_fraction` / :func:`analyze_overlap` functions computing
Overlap_Fraction over shared source-index sets plus summary statistics.

Overlap is a pure set operation on source Fragment point indices, so it is
exact (no floating-point matching). For two patches with index sets ``A`` and
``B``::

    overlap_fraction(A, B) = |A ∩ B| / |A|

which is bounded in ``[0, 1]`` (numerator never exceeds denominator, Req 5.2),
equals ``1.0`` for self-overlap ``A -> A`` (Req 5.4), and equals ``0.0`` when
the sets are disjoint (Req 5.3). By convention an empty ``A`` yields ``0.0``
(avoids division by zero; a zero-size set shares nothing).

:func:`analyze_overlap` evaluates the Overlap_Fraction for every pair of
patches that share at least one source index (Req 5.1). Candidate pairs are
found by inverting the point -> patch membership map, so only pairs that
co-occur at some point are considered (avoiding a full O(M^2) scan). Both
directed fractions of each co-occurring unordered pair are recorded, and the
mean and maximum over all analyzed fractions are reported (0.0 when there are
no analyzed pairs, Req 5.5).

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

__all__ = [
    "OverlapStats",
    "overlap_fraction",
    "analyze_overlap",
]


@dataclass(frozen=True)
class OverlapStats:
    """Summary statistics of pairwise Patch overlap for a Fragment.

    Attributes:
        analyzed_pair_count: Number of directed overlap fractions analyzed
            (pairs sharing at least one source index, Req 5.1).
        mean_overlap: Mean Overlap_Fraction over analyzed pairs, or ``0.0``
            when there are none (Req 5.5).
        max_overlap: Maximum Overlap_Fraction over analyzed pairs, or ``0.0``
            when there are none (Req 5.5).
    """

    analyzed_pair_count: int
    mean_overlap: float
    max_overlap: float


def overlap_fraction(
    a_indices: Iterable[int],
    b_indices: Iterable[int],
) -> float:
    """Compute the directed Overlap_Fraction ``|A ∩ B| / |A|``.

    Accepts any iterables/sets/arrays of integer source indices. The result is
    the fraction of ``A``'s points shared with ``B``.

    Args:
        a_indices: Source indices of the first Patch (the denominator set).
        b_indices: Source indices of the second Patch.

    Returns:
        A float in ``[0.0, 1.0]``: ``1.0`` for self-overlap (``B == A``,
        Req 5.4), ``0.0`` for disjoint sets (Req 5.3), and ``0.0`` when ``A``
        is empty (defined to avoid division by zero).
    """
    a = set(int(i) for i in a_indices)
    if not a:
        return 0.0
    b = set(int(i) for i in b_indices)
    intersection = len(a & b)
    return intersection / len(a)


def analyze_overlap(patches) -> OverlapStats:
    """Analyze pairwise overlap across a Fragment's patches.

    Only pairs of patches that share at least one source index are analyzed
    (Req 5.1). Such co-occurring pairs are discovered by inverting the point
    -> patch membership map rather than scanning all O(M^2) pairs. For each
    co-occurring unordered pair ``{i, j}`` both directed Overlap_Fractions
    (``i -> j`` and ``j -> i``) are recorded, and the mean and maximum over all
    recorded fractions are returned (Req 5.5).

    Args:
        patches: Iterable of :class:`~patch_generation.patch_extractor.Patch`
            objects (each exposing a ``source_indices`` array).

    Returns:
        An :class:`OverlapStats` with ``analyzed_pair_count``, ``mean_overlap``,
        and ``max_overlap`` (all ``0.0``/``0`` when there are no co-occurring
        pairs).
    """
    patch_list = list(patches)

    # Precompute each patch's source-index set once.
    index_sets: list[set[int]] = [
        set(int(i) for i in np.asarray(p.source_indices).reshape(-1))
        for p in patch_list
    ]

    # Invert the point -> patch membership map: for each source point, collect
    # the positions of patches that contain it.
    point_to_patches: dict[int, list[int]] = {}
    for pos, idx_set in enumerate(index_sets):
        for point_index in idx_set:
            point_to_patches.setdefault(point_index, []).append(pos)

    # Candidate co-occurring unordered pairs: any two distinct patches that
    # share at least one point. Use a set to de-duplicate pairs seen at
    # multiple shared points.
    co_occurring: set[tuple[int, int]] = set()
    for positions in point_to_patches.values():
        if len(positions) < 2:
            continue
        for i, j in combinations(sorted(positions), 2):
            co_occurring.add((i, j))

    if not co_occurring:
        return OverlapStats(analyzed_pair_count=0, mean_overlap=0.0, max_overlap=0.0)

    fractions: list[float] = []
    for i, j in co_occurring:
        fractions.append(overlap_fraction(index_sets[i], index_sets[j]))
        fractions.append(overlap_fraction(index_sets[j], index_sets[i]))

    return OverlapStats(
        analyzed_pair_count=len(fractions),
        mean_overlap=float(sum(fractions) / len(fractions)),
        max_overlap=float(max(fractions)),
    )
