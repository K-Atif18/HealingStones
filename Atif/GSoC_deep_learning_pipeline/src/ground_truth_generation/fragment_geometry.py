"""Fragment geometry statistics computation.

Measures:
- Point spacing (nearest neighbor distances)
- Curvature distribution (PCA-based local curvature)
- Normal variance (surface flatness)
- Fragment bounding boxes and extents

These measurements inform all threshold decisions in Phase 3.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial import cKDTree
import json
from pathlib import Path


@dataclass
class FragmentGeometryStats:
    """Geometric statistics for a single fragment."""
    
    fragment_id: str
    
    # Basic properties
    point_count: int
    bounding_box_min: Tuple[float, float, float]
    bounding_box_max: Tuple[float, float, float]
    extent_mm: Tuple[float, float, float]
    
    # Point spacing (local density)
    mean_nn_distance_mm: float
    median_nn_distance_mm: float
    std_nn_distance_mm: float
    percentile_5_mm: float
    percentile_25_mm: float
    percentile_50_mm: float
    percentile_75_mm: float
    percentile_95_mm: float
    
    # Curvature (local surface variation)
    mean_curvature: float
    median_curvature: float
    std_curvature: float
    curvature_percentiles: Dict[int, float]
    
    # Normal variation (flatness)
    mean_normal_variance: float
    median_normal_variance: float
    std_normal_variance: float
    normal_variance_percentiles: Dict[int, float]
    
    def to_dict(self):
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert tuples to lists for JSON
        d['bounding_box_min'] = list(d['bounding_box_min'])
        d['bounding_box_max'] = list(d['bounding_box_max'])
        d['extent_mm'] = list(d['extent_mm'])
        return d


def compute_nearest_neighbor_distances(points: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Compute nearest neighbor distance for each point.
    
    Args:
        points: (N, 3) point cloud
        k: Number of neighbors (2 = self + nearest)
        
    Returns:
        (N,) array of distances to nearest neighbor
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k)
    # distances[:, 0] is distance to self (0.0)
    # distances[:, 1] is distance to nearest neighbor
    return distances[:, 1]


def estimate_local_curvature(points: np.ndarray, normals: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Estimate local curvature using PCA on neighborhood normals.
    
    Higher curvature = normals vary more in local neighborhood.
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) unit normals
        k: Number of neighbors for local estimation
        
    Returns:
        (N,) array of curvature estimates
    """
    tree = cKDTree(points)
    curvatures = np.zeros(len(points))
    
    for i, point in enumerate(points):
        # Find k nearest neighbors
        distances, indices = tree.query(point, k=min(k, len(points)))
        neighbor_normals = normals[indices]
        
        # Curvature proxy: variance of normal directions
        # High curvature = normals point in different directions
        mean_normal = neighbor_normals.mean(axis=0)
        normal_deviations = np.linalg.norm(neighbor_normals - mean_normal, axis=1)
        curvatures[i] = normal_deviations.mean()
    
    return curvatures


def compute_local_normal_variance(points: np.ndarray, normals: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Compute variance of normals in local neighborhood.
    
    Low variance = flat region (all normals aligned).
    High variance = curved/complex region.
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) unit normals
        k: Number of neighbors
        
    Returns:
        (N,) array of normal variance values
    """
    tree = cKDTree(points)
    variances = np.zeros(len(points))
    
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=min(k, len(points)))
        neighbor_normals = normals[indices]
        
        # Variance of normal components
        variance = np.var(neighbor_normals, axis=0).sum()
        variances[i] = variance
    
    return variances


def analyze_fragment_geometry(
    points: np.ndarray,
    normals: np.ndarray,
    fragment_id: str
) -> FragmentGeometryStats:
    """
    Compute comprehensive geometric statistics for a fragment.
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) unit normals
        fragment_id: Fragment identifier
        
    Returns:
        FragmentGeometryStats object with all measurements
    """
    
    # Basic properties
    point_count = len(points)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    
    # Point spacing
    nn_distances = compute_nearest_neighbor_distances(points)
    
    # Curvature
    print(f"  Computing curvature for {point_count} points...")
    curvatures = estimate_local_curvature(points, normals)
    
    # Normal variance
    print(f"  Computing normal variance for {point_count} points...")
    normal_variances = compute_local_normal_variance(points, normals)
    
    # Compute percentiles
    curvature_percentiles = {
        p: float(np.percentile(curvatures, p))
        for p in [5, 25, 50, 75, 95]
    }
    
    normal_variance_percentiles = {
        p: float(np.percentile(normal_variances, p))
        for p in [5, 25, 50, 75, 95]
    }
    
    stats = FragmentGeometryStats(
        fragment_id=fragment_id,
        point_count=point_count,
        bounding_box_min=tuple(bbox_min.tolist()),
        bounding_box_max=tuple(bbox_max.tolist()),
        extent_mm=tuple(extent.tolist()),
        
        # Point spacing statistics
        mean_nn_distance_mm=float(nn_distances.mean()),
        median_nn_distance_mm=float(np.median(nn_distances)),
        std_nn_distance_mm=float(nn_distances.std()),
        percentile_5_mm=float(np.percentile(nn_distances, 5)),
        percentile_25_mm=float(np.percentile(nn_distances, 25)),
        percentile_50_mm=float(np.percentile(nn_distances, 50)),
        percentile_75_mm=float(np.percentile(nn_distances, 75)),
        percentile_95_mm=float(np.percentile(nn_distances, 95)),
        
        # Curvature statistics
        mean_curvature=float(curvatures.mean()),
        median_curvature=float(np.median(curvatures)),
        std_curvature=float(curvatures.std()),
        curvature_percentiles=curvature_percentiles,
        
        # Normal variance statistics
        mean_normal_variance=float(normal_variances.mean()),
        median_normal_variance=float(np.median(normal_variances)),
        std_normal_variance=float(normal_variances.std()),
        normal_variance_percentiles=normal_variance_percentiles,
    )
    
    return stats


def save_fragment_stats(stats: FragmentGeometryStats, output_dir: Path) -> None:
    """Save fragment statistics to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stats.fragment_id}.json"
    
    with open(output_path, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)
    
    print(f"  Saved statistics to {output_path}")


def compute_dataset_summary(all_stats: list) -> Dict:
    """
    Compute dataset-level summary statistics.
    
    Args:
        all_stats: List of FragmentGeometryStats
        
    Returns:
        Dictionary with aggregated statistics
    """
    
    # Aggregate point spacing
    all_nn_distances = [s.mean_nn_distance_mm for s in all_stats]
    
    # Aggregate curvature
    all_mean_curvatures = [s.mean_curvature for s in all_stats]
    
    # Aggregate normal variance
    all_mean_normal_variances = [s.mean_normal_variance for s in all_stats]
    
    summary = {
        "fragment_count": len(all_stats),
        "total_points": sum(s.point_count for s in all_stats),
        
        # Point spacing across all fragments
        "point_spacing": {
            "mean_across_fragments_mm": float(np.mean(all_nn_distances)),
            "std_across_fragments_mm": float(np.std(all_nn_distances)),
            "min_mm": float(np.min(all_nn_distances)),
            "max_mm": float(np.max(all_nn_distances)),
        },
        
        # Curvature across all fragments
        "curvature": {
            "mean_across_fragments": float(np.mean(all_mean_curvatures)),
            "std_across_fragments": float(np.std(all_mean_curvatures)),
            "min": float(np.min(all_mean_curvatures)),
            "max": float(np.max(all_mean_curvatures)),
        },
        
        # Normal variance across all fragments
        "normal_variance": {
            "mean_across_fragments": float(np.mean(all_mean_normal_variances)),
            "std_across_fragments": float(np.std(all_mean_normal_variances)),
            "min": float(np.min(all_mean_normal_variances)),
            "max": float(np.max(all_mean_normal_variances)),
        },
        
        # Recommendations based on measurements
        "recommendations": {
            "adjacency_threshold_mm": {
                "conservative": float(2 * np.mean(all_nn_distances)),
                "moderate": float(3 * np.mean(all_nn_distances)),
                "inclusive": float(5 * np.mean(all_nn_distances)),
                "reasoning": "Based on 2×, 3×, 5× mean point spacing"
            }
        }
    }
    
    return summary
