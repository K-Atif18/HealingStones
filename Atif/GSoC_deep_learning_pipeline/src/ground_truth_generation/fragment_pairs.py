"""Fragment pair distance analysis and contact region detection.

Computes pairwise distances between all fragment pairs and identifies
contact regions (points that are close enough to be assembly-relevant).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
from scipy.spatial import cKDTree
import json
from pathlib import Path


@dataclass
class FragmentPairDistanceStats:
    """Distance statistics between a pair of fragments."""
    
    fragment_A_id: str
    fragment_B_id: str
    
    # Distance statistics
    min_distance_mm: float
    mean_distance_mm: float
    median_distance_mm: float
    
    # Contact point counts at various thresholds
    contact_counts_at_thresholds: Dict[float, Tuple[int, int]]
    
    # Classification based on distance
    adjacency_category: str  # "adjacent" | "near" | "separated"
    
    def to_dict(self):
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d['contact_counts_at_thresholds'] = {
            str(k): list(v) for k, v in d['contact_counts_at_thresholds'].items()
        }
        return d


@dataclass
class ContactRegion:
    """Contact region data for a fragment pair."""
    
    fragment_A_id: str
    fragment_B_id: str
    threshold_mm: float
    
    # Contact point indices
    contact_indices_A: np.ndarray
    contact_indices_B: np.ndarray
    
    # Contact point coordinates
    contact_points_A: np.ndarray
    contact_points_B: np.ndarray
    
    def save(self, output_path: Path):
        """Save contact region to .npz file."""
        np.savez_compressed(
            output_path,
            fragment_A_id=self.fragment_A_id,
            fragment_B_id=self.fragment_B_id,
            threshold_mm=self.threshold_mm,
            contact_indices_A=self.contact_indices_A,
            contact_indices_B=self.contact_indices_B,
            contact_points_A=self.contact_points_A,
            contact_points_B=self.contact_points_B
        )


def detect_contact_points(
    points_A: np.ndarray,
    points_B: np.ndarray,
    threshold_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Find points in each fragment within threshold of the other."""
    tree_B = cKDTree(points_B)
    tree_A = cKDTree(points_A)
    
    distances_A, _ = tree_B.query(points_A)
    contact_indices_A = np.where(distances_A < threshold_mm)[0]
    
    distances_B, _ = tree_A.query(points_B)
    contact_indices_B = np.where(distances_B < threshold_mm)[0]
    
    return contact_indices_A, contact_indices_B


def analyze_fragment_pair(
    points_A: np.ndarray,
    points_B: np.ndarray,
    fragment_A_id: str,
    fragment_B_id: str,
    thresholds: List[float] = [1.0, 2.0, 3.0, 5.0, 10.0]
) -> FragmentPairDistanceStats:
    """Compute distance statistics for a fragment pair."""
    
    # Compute pairwise distances (sample for efficiency)
    tree_B = cKDTree(points_B)
    sample_size = min(1000, len(points_A))
    sample_indices = np.random.choice(len(points_A), sample_size, replace=False)
    distances, _ = tree_B.query(points_A[sample_indices])
    
    min_dist = float(distances.min())
    mean_dist = float(distances.mean())
    median_dist = float(np.median(distances))
    
    # Contact counts at thresholds
    contact_counts = {}
    for threshold in thresholds:
        indices_A, indices_B = detect_contact_points(points_A, points_B, threshold)
        contact_counts[threshold] = (len(indices_A), len(indices_B))
    
    # Classify adjacency
    if min_dist < 3.0:
        category = "adjacent"
    elif min_dist < 10.0:
        category = "near"
    else:
        category = "separated"
    
    return FragmentPairDistanceStats(
        fragment_A_id=fragment_A_id,
        fragment_B_id=fragment_B_id,
        min_distance_mm=min_dist,
        mean_distance_mm=mean_dist,
        median_distance_mm=median_dist,
        contact_counts_at_thresholds=contact_counts,
        adjacency_category=category
    )


def generate_contact_region(
    points_A: np.ndarray,
    points_B: np.ndarray,
    fragment_A_id: str,
    fragment_B_id: str,
    threshold_mm: float
) -> ContactRegion:
    """Generate contact region data at specified threshold."""
    
    indices_A, indices_B = detect_contact_points(points_A, points_B, threshold_mm)
    
    contact_pts_A = points_A[indices_A]
    contact_pts_B = points_B[indices_B]
    
    return ContactRegion(
        fragment_A_id=fragment_A_id,
        fragment_B_id=fragment_B_id,
        threshold_mm=threshold_mm,
        contact_indices_A=indices_A,
        contact_indices_B=indices_B,
        contact_points_A=contact_pts_A,
        contact_points_B=contact_pts_B
    )
