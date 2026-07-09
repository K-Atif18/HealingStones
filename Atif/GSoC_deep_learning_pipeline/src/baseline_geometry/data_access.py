"""Unified access to Phase 1-3 artifacts for the baseline experiments.

Loads the Phase 1 normalized clouds (points + normals), the Phase 2 patch
archives, and the Phase 3 compact ``pairs.npz`` labelled-pair table. All of it
is read-only; nothing here mutates upstream outputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from baseline_geometry.errors import InputLoadError

__all__ = [
    "FragmentCloud",
    "PatchTable",
    "PairTable",
    "load_fragment_ids",
    "load_fragment_cloud",
    "load_patch_table",
    "load_pairs",
    "load_adjacent_pairs",
    "load_contact_region",
]


@dataclass(frozen=True)
class FragmentCloud:
    """A Phase 1 fragment: points and per-point normals in the assembled frame."""

    fragment_id: str
    points: np.ndarray   # (N, 3) float64
    normals: np.ndarray  # (N, 3) float64

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class PatchTable:
    """All Phase 2 patches for one fragment, in stacked form.

    ``offsets[i] = [start, end)`` slices the stacked per-point arrays for patch
    ``patch_ids[i]``. ``source_indices`` indexes into the owning fragment's
    normalized cloud (so a patch's points/normals can be recovered from either
    the stacked arrays here or from the :class:`FragmentCloud`).
    """

    fragment_id: str
    patch_ids: np.ndarray       # (M,) int64
    center_indices: np.ndarray  # (M,) int64  -> index into fragment cloud
    centers: np.ndarray         # (M, 3) float64
    offsets: np.ndarray         # (M, 2) int64
    global_coords: np.ndarray   # (T, 3) float64
    normals: np.ndarray         # (T, 3) float64
    source_indices: np.ndarray  # (T,) int64

    @property
    def patch_count(self) -> int:
        return int(self.patch_ids.shape[0])

    def patch_points(self, row: int) -> np.ndarray:
        start, end = int(self.offsets[row, 0]), int(self.offsets[row, 1])
        return self.global_coords[start:end]

    def patch_normals(self, row: int) -> np.ndarray:
        start, end = int(self.offsets[row, 0]), int(self.offsets[row, 1])
        return self.normals[start:end]

    def row_for_patch_id(self, patch_id: int) -> int:
        """Return the stacked-array row for a given patch id (ids are 0..M-1)."""
        # Patch ids are assigned 0..M-1 in stored order, but be robust anyway.
        if 0 <= patch_id < self.patch_count and int(self.patch_ids[patch_id]) == patch_id:
            return patch_id
        matches = np.where(self.patch_ids == patch_id)[0]
        if matches.size == 0:
            raise InputLoadError(self.fragment_id, f"patch_id {patch_id} not found")
        return int(matches[0])


@dataclass(frozen=True)
class PairTable:
    """Phase 3 compact labelled pairs (from ``pairs.npz``)."""

    fragment_vocab: np.ndarray   # (V,) str
    fragment_A_idx: np.ndarray   # (P,) int
    fragment_B_idx: np.ndarray   # (P,) int
    patch_A_ids: np.ndarray      # (P,) int
    patch_B_ids: np.ndarray      # (P,) int
    labels: np.ndarray           # (P,) int8: 1=pos, -1=rand-neg, -2=hard-neg
    contact_overlap_A: np.ndarray
    contact_overlap_B: np.ndarray
    center_dist_mm: np.ndarray

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])


def load_fragment_ids(dataset_dir: str) -> list[str]:
    """Read the ordered fragment ids from Phase 1 ``metadata/dataset.json``."""
    path = os.path.join(dataset_dir, "metadata", "dataset.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputLoadError(path, str(exc)) from exc
    ids = record.get("fragment_ids")
    if not ids:
        raise InputLoadError(path, "no fragment_ids in dataset metadata")
    return [str(fid) for fid in ids]


def load_fragment_cloud(dataset_dir: str, fragment_id: str) -> FragmentCloud:
    """Load a Phase 1 normalized cloud (points + normals) via Open3D."""
    import open3d as o3d

    path = os.path.join(dataset_dir, "normalized", f"{fragment_id}.ply")
    if not os.path.isfile(path):
        raise InputLoadError(path, f"normalized cloud missing for {fragment_id}")
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if points.shape[0] == 0:
        raise InputLoadError(path, f"empty cloud for {fragment_id}")
    if normals.shape[0] != points.shape[0]:
        # Normals absent/mismatched; leave empty so the descriptor stage can
        # re-estimate them from the geometry.
        normals = np.empty((0, 3), dtype=np.float64)
    return FragmentCloud(fragment_id=fragment_id, points=points, normals=normals)


def load_patch_table(patches_dir: str, fragment_id: str) -> PatchTable:
    """Load a Phase 2 patch archive for one fragment."""
    path = os.path.join(patches_dir, fragment_id, "patches.npz")
    if not os.path.isfile(path):
        raise InputLoadError(path, f"patch archive missing for {fragment_id}")
    try:
        with np.load(path, allow_pickle=False) as data:
            table = PatchTable(
                fragment_id=str(data["fragment_id"].item()),
                patch_ids=np.asarray(data["patch_ids"], dtype=np.int64),
                center_indices=np.asarray(data["center_indices"], dtype=np.int64),
                centers=np.asarray(data["centers"], dtype=np.float64),
                offsets=np.asarray(data["offsets"], dtype=np.int64).reshape(-1, 2),
                global_coords=np.asarray(data["global_coords"], dtype=np.float64),
                normals=np.asarray(data["normals"], dtype=np.float64),
                source_indices=np.asarray(data["source_indices"], dtype=np.int64),
            )
    except (OSError, KeyError, ValueError) as exc:
        raise InputLoadError(path, str(exc)) from exc
    return table


def load_pairs(pairs_npz: str) -> PairTable:
    """Load the Phase 3 compact labelled-pair table."""
    if not os.path.isfile(pairs_npz):
        raise InputLoadError(pairs_npz, "pairs.npz not found")
    try:
        with np.load(pairs_npz, allow_pickle=True) as data:
            table = PairTable(
                fragment_vocab=np.asarray(data["fragment_vocab"]).astype(str),
                fragment_A_idx=np.asarray(data["fragment_A_idx"]).astype(np.int64),
                fragment_B_idx=np.asarray(data["fragment_B_idx"]).astype(np.int64),
                patch_A_ids=np.asarray(data["patch_A_ids"]).astype(np.int64),
                patch_B_ids=np.asarray(data["patch_B_ids"]).astype(np.int64),
                labels=np.asarray(data["labels"]).astype(np.int64),
                contact_overlap_A=np.asarray(data["contact_overlap_A"]).astype(np.float64),
                contact_overlap_B=np.asarray(data["contact_overlap_B"]).astype(np.float64),
                center_dist_mm=np.asarray(data["center_dist_mm"]).astype(np.float64),
            )
    except (OSError, KeyError, ValueError) as exc:
        raise InputLoadError(pairs_npz, str(exc)) from exc
    return table


def load_adjacent_pairs(pairs_dataset_json: str) -> list[tuple[str, str]]:
    """Read the list of adjacent fragment pairs from Phase 3 ``dataset.json``."""
    if not os.path.isfile(pairs_dataset_json):
        raise InputLoadError(pairs_dataset_json, "pairs dataset.json not found")
    try:
        with open(pairs_dataset_json, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputLoadError(pairs_dataset_json, str(exc)) from exc
    pairs = record.get("adjacent_pair_ids", [])
    return [(str(a), str(b)) for a, b in pairs]


def load_contact_region(contact_dir: str, fid_a: str, fid_b: str) -> Optional[dict]:
    """Load a Phase 3 contact-region archive for an adjacent pair.

    Returns a dict with ``contact_indices_A`` / ``contact_indices_B`` (indices
    into the respective fragment clouds) and the contact point coordinates, or
    ``None`` if no archive exists for this ordering.
    """
    path = os.path.join(contact_dir, f"{fid_a}_{fid_b}.npz")
    if not os.path.isfile(path):
        path = os.path.join(contact_dir, f"{fid_b}_{fid_a}.npz")
        if not os.path.isfile(path):
            return None
    with np.load(path, allow_pickle=True) as data:
        return {
            "fragment_A_id": str(data["fragment_A_id"].item()),
            "fragment_B_id": str(data["fragment_B_id"].item()),
            "contact_indices_A": np.asarray(data["contact_indices_A"], dtype=np.int64),
            "contact_indices_B": np.asarray(data["contact_indices_B"], dtype=np.int64),
            "contact_points_A": np.asarray(data["contact_points_A"], dtype=np.float64),
            "contact_points_B": np.asarray(data["contact_points_B"], dtype=np.float64),
        }
