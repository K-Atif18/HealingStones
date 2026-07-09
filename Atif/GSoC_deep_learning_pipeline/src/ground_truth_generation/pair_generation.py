"""Pair generation for ground truth labels.

Generates positive, hard negative, and random negative patch pairs
for training contrastive learning models.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, asdict
from scipy.spatial import cKDTree
import json


@dataclass
class PatchPair:
    """A labeled patch pair for training."""
    
    patch_A_id: str
    patch_B_id: str
    fragment_A_id: str
    fragment_B_id: str
    label: str  # "positive" | "negative_easy" | "negative_medium" | "negative_hard"
    
    # Spatial context
    patch_A_center: Tuple[float, float, float]
    patch_B_center: Tuple[float, float, float]
    center_distance_mm: float
    
    # Contact evidence (for positives)
    contact_overlap_A: float = 0.0  # Fraction of patch A points in contact region
    contact_overlap_B: float = 0.0
    
    def to_dict(self):
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d['patch_A_center'] = list(d['patch_A_center'])
        d['patch_B_center'] = list(d['patch_B_center'])
        return d


def compute_patch_contact_overlap(
    patch_source_indices: np.ndarray,
    contact_indices: np.ndarray
) -> float:
    """Compute fraction of patch points that are in contact region."""
    if len(patch_source_indices) == 0:
        return 0.0
    
    overlap_count = len(set(patch_source_indices) & set(contact_indices))
    return overlap_count / len(patch_source_indices)


def generate_positive_pairs(
    patches_A: List,
    patches_B: List,
    contact_indices_A: np.ndarray,
    contact_indices_B: np.ndarray,
    fragment_A_id: str,
    fragment_B_id: str,
    min_overlap: float = 0.3
) -> List[PatchPair]:
    """
    Generate positive pairs from adjacent fragments.
    
    A pair is positive if both patches have significant overlap with contact region.
    
    Args:
        patches_A: Patches from fragment A
        patches_B: Patches from fragment B
        contact_indices_A: Fragment A point indices in contact region
        contact_indices_B: Fragment B point indices in contact region
        fragment_A_id: Fragment A identifier
        fragment_B_id: Fragment B identifier
        min_overlap: Minimum fraction of patch points in contact region
        
    Returns:
        List of positive PatchPair objects
    """
    
    positive_pairs = []
    contact_set_A = set(contact_indices_A)
    contact_set_B = set(contact_indices_B)
    
    # Filter patches with sufficient contact overlap
    candidate_patches_A = []
    for patch in patches_A:
        overlap = compute_patch_contact_overlap(patch.source_indices, contact_indices_A)
        if overlap >= min_overlap:
            candidate_patches_A.append((patch, overlap))
    
    candidate_patches_B = []
    for patch in patches_B:
        overlap = compute_patch_contact_overlap(patch.source_indices, contact_indices_B)
        if overlap >= min_overlap:
            candidate_patches_B.append((patch, overlap))
    
    # Generate pairs
    for patch_A, overlap_A in candidate_patches_A:
        for patch_B, overlap_B in candidate_patches_B:
            center_dist = float(np.linalg.norm(patch_A.center - patch_B.center))
            
            pair = PatchPair(
                patch_A_id=patch_A.patch_id,
                patch_B_id=patch_B.patch_id,
                fragment_A_id=fragment_A_id,
                fragment_B_id=fragment_B_id,
                label="positive",
                patch_A_center=tuple(patch_A.center.tolist()),
                patch_B_center=tuple(patch_B.center.tolist()),
                center_distance_mm=center_dist,
                contact_overlap_A=overlap_A,
                contact_overlap_B=overlap_B
            )
            positive_pairs.append(pair)
    
    return positive_pairs


def generate_hard_negatives(
    patches: List,
    fragment_id: str,
    count: int
) -> List[PatchPair]:
    """
    Generate hard negative pairs from same fragment.
    
    Hard negatives: non-overlapping patches with similar geometry.
    Uses spatial proximity as proxy for geometric similarity.
    
    Args:
        patches: Patches from one fragment
        fragment_id: Fragment identifier
        count: Number of hard negative pairs to generate
        
    Returns:
        List of hard negative PatchPair objects
    """
    
    hard_negatives = []
    
    if len(patches) < 2:
        return hard_negatives
    
    # Build spatial tree
    centers = np.array([p.center for p in patches])
    tree = cKDTree(centers)
    
    # For each patch, find nearby non-overlapping patches
    attempts = 0
    max_attempts = count * 10
    
    while len(hard_negatives) < count and attempts < max_attempts:
        attempts += 1
        
        # Pick random patch
        idx_A = np.random.randint(len(patches))
        patch_A = patches[idx_A]
        
        # Find nearby patches
        distances, indices = tree.query(patch_A.center, k=min(20, len(patches)))
        
        # Filter: nearby but non-overlapping
        for dist, idx_B in zip(distances[1:], indices[1:]):  # Skip self
            if dist > 16.0:  # 2× patch radius (8mm)
                patch_B = patches[idx_B]
                
                # Check no overlap in source indices
                overlap = len(set(patch_A.source_indices) & set(patch_B.source_indices))
                if overlap == 0:
                    pair = PatchPair(
                        patch_A_id=patch_A.patch_id,
                        patch_B_id=patch_B.patch_id,
                        fragment_A_id=fragment_id,
                        fragment_B_id=fragment_id,
                        label="negative_hard",
                        patch_A_center=tuple(patch_A.center.tolist()),
                        patch_B_center=tuple(patch_B.center.tolist()),
                        center_distance_mm=float(dist)
                    )
                    hard_negatives.append(pair)
                    break
    
    return hard_negatives


def generate_random_negatives(
    patches_A: List,
    patches_B: List,
    fragment_A_id: str,
    fragment_B_id: str,
    count: int
) -> List[PatchPair]:
    """
    Generate random negative pairs from non-adjacent fragments.
    
    Args:
        patches_A: Patches from fragment A
        patches_B: Patches from fragment B
        fragment_A_id: Fragment A identifier
        fragment_B_id: Fragment B identifier
        count: Number of random negative pairs to generate
        
    Returns:
        List of random negative PatchPair objects
    """
    
    random_negatives = []
    
    for _ in range(count):
        patch_A = patches_A[np.random.randint(len(patches_A))]
        patch_B = patches_B[np.random.randint(len(patches_B))]
        
        center_dist = float(np.linalg.norm(patch_A.center - patch_B.center))
        
        pair = PatchPair(
            patch_A_id=patch_A.patch_id,
            patch_B_id=patch_B.patch_id,
            fragment_A_id=fragment_A_id,
            fragment_B_id=fragment_B_id,
            label="negative_random",
            patch_A_center=tuple(patch_A.center.tolist()),
            patch_B_center=tuple(patch_B.center.tolist()),
            center_distance_mm=center_dist
        )
        random_negatives.append(pair)
    
    return random_negatives
