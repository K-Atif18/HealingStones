"""Shared data context for the Phase 5 diagnostics.

Loads (read-only) the Phase 1-3 artifacts once and derives the structures every
diagnostic needs, so the individual diagnostic functions stay pure and fast:

* ``fragment_ids``                : ordered fragment vocabulary (Phase 1).
* ``patch_tables``                : Phase 2 patches per fragment.
* ``pairs``                       : Phase 3 compact labelled-pair table.
* ``adjacent_pairs``              : the 11 true-neighbour fragment pairs.
* ``contact_patch_ids``           : per fragment, the set of patch ids that are
                                    "contact patches" (>= threshold of their
                                    points lie in *some* interface's contact
                                    region). This is the Level-1 label.
* ``interface_patch_ids``         : per (fid_a, fid_b) interface, the contact
                                    patch ids on each side (the Level-2 label).
* ``positive_partners``           : (fid, pid) -> set of global gallery rows that
                                    are its Phase-3 positive partners (Level-3
                                    ground truth, as the retrieval harness sees).
* ``distinctiveness``             : (fid, pid) -> (score, class).

Nothing here mutates upstream outputs. All indices/ids follow the exact
conventions verified from the on-disk archives:
  - patch ids are 0..M-1 per fragment (int).
  - pairs.npz labels are +1 (positive) / -1 (random negative); no hard negs.
  - contact-region npz stores per-fragment *point* indices into the fragment
    cloud; a patch is "in contact" when enough of its source_indices fall in
    that set (mirrors Phase 3 ``compute_patch_contact_overlap``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Reuse the Phase 4 read-only loaders verbatim so we share one contract.
from baseline_geometry import data_access as da
from baseline_geometry.data_access import PatchTable, PairTable


__all__ = ["DiagnosticContext", "load_context"]


# Mirror Phase 3's positive-pair construction threshold so "contact patch" here
# means the same thing it meant when the labels were generated.
DEFAULT_MIN_CONTACT_OVERLAP = 0.30


@dataclass
class DiagnosticContext:
    """Everything the diagnostics need, loaded once."""

    dataset_dir: str
    patches_dir: str
    pairs_npz: str
    pairs_dataset_json: str
    contact_dir: str

    fragment_ids: list[str]
    patch_tables: dict[str, PatchTable]
    pairs: PairTable
    adjacent_pairs: list[tuple[str, str]]

    # Derived label structures.
    contact_patch_ids: dict[str, set[int]] = field(default_factory=dict)
    interface_patch_ids: dict[tuple[str, str], dict[str, set[int]]] = field(default_factory=dict)
    distinctiveness: dict[tuple[str, int], tuple[float, str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def patch_count(self, fid: str) -> int:
        return self.patch_tables[fid].patch_count

    def total_patches(self) -> int:
        return int(sum(t.patch_count for t in self.patch_tables.values()))

    def neighbors_of(self, fid: str) -> set[str]:
        """Fragments adjacent to ``fid`` per Phase 3 adjacency."""
        out: set[str] = set()
        for a, b in self.adjacent_pairs:
            if a == fid:
                out.add(b)
            elif b == fid:
                out.add(a)
        return out

    def is_contact_patch(self, fid: str, pid: int) -> bool:
        return pid in self.contact_patch_ids.get(fid, set())


# ----------------------------------------------------------------------
# Loading / derivation
# ----------------------------------------------------------------------
def _contact_patches_for_side(
    table: PatchTable,
    contact_point_indices: np.ndarray,
    min_overlap: float,
) -> set[int]:
    """Patch ids on this fragment whose contact overlap >= ``min_overlap``.

    Replicates Phase 3 ``compute_patch_contact_overlap``: overlap =
    |patch.source_indices ∩ contact_indices| / |patch.source_indices|.
    """
    contact_set = set(int(i) for i in contact_point_indices.tolist())
    out: set[int] = set()
    for row in range(table.patch_count):
        start, end = int(table.offsets[row, 0]), int(table.offsets[row, 1])
        src = table.source_indices[start:end]
        if src.size == 0:
            continue
        inter = sum(1 for s in src.tolist() if int(s) in contact_set)
        if inter / src.size >= min_overlap:
            out.add(int(table.patch_ids[row]))
    return out


def _load_distinctiveness(root: str) -> dict[tuple[str, int], tuple[float, str]]:
    path = os.path.join(root, "phase3_final_review", "patch_distinctiveness.npz")
    lookup: dict[tuple[str, int], tuple[float, str]] = {}
    if not os.path.isfile(path):
        return lookup
    with np.load(path, allow_pickle=True) as d:
        for fid, pid, score, cls in zip(
            d["fragment_ids"], d["patch_ids"], d["distinctiveness"], d["classes"]
        ):
            lookup[(str(fid), int(pid))] = (float(score), str(cls))
    return lookup


def load_context(
    *,
    dataset_dir: str,
    patches_dir: str,
    pairs_npz: str,
    pairs_dataset_json: str,
    contact_dir: Optional[str] = None,
    repo_root: Optional[str] = None,
    min_contact_overlap: float = DEFAULT_MIN_CONTACT_OVERLAP,
) -> DiagnosticContext:
    """Load and derive the full diagnostic context."""
    if contact_dir is None:
        contact_dir = os.path.join(os.path.dirname(pairs_dataset_json), "contact_regions")
    if repo_root is None:
        repo_root = os.getcwd()

    fragment_ids = da.load_fragment_ids(dataset_dir)
    patch_tables = {fid: da.load_patch_table(patches_dir, fid) for fid in fragment_ids}
    pairs = da.load_pairs(pairs_npz)
    adjacent_pairs = da.load_adjacent_pairs(pairs_dataset_json)

    # Contact patches per fragment + per interface, from contact-region archives.
    contact_patch_ids: dict[str, set[int]] = {fid: set() for fid in fragment_ids}
    interface_patch_ids: dict[tuple[str, str], dict[str, set[int]]] = {}

    for fid_a, fid_b in adjacent_pairs:
        region = da.load_contact_region(contact_dir, fid_a, fid_b)
        if region is None:
            interface_patch_ids[(fid_a, fid_b)] = {fid_a: set(), fid_b: set()}
            continue
        # The archive labels its own A/B; align to (fid_a, fid_b).
        ra_id = region["fragment_A_id"]
        idx_a = region["contact_indices_A"] if ra_id == fid_a else region["contact_indices_B"]
        idx_b = region["contact_indices_B"] if ra_id == fid_a else region["contact_indices_A"]

        pa = _contact_patches_for_side(patch_tables[fid_a], idx_a, min_contact_overlap)
        pb = _contact_patches_for_side(patch_tables[fid_b], idx_b, min_contact_overlap)
        interface_patch_ids[(fid_a, fid_b)] = {fid_a: pa, fid_b: pb}
        contact_patch_ids[fid_a] |= pa
        contact_patch_ids[fid_b] |= pb

    distinctiveness = _load_distinctiveness(repo_root)

    return DiagnosticContext(
        dataset_dir=dataset_dir,
        patches_dir=patches_dir,
        pairs_npz=pairs_npz,
        pairs_dataset_json=pairs_dataset_json,
        contact_dir=contact_dir,
        fragment_ids=fragment_ids,
        patch_tables=patch_tables,
        pairs=pairs,
        adjacent_pairs=adjacent_pairs,
        contact_patch_ids=contact_patch_ids,
        interface_patch_ids=interface_patch_ids,
        distinctiveness=distinctiveness,
    )
