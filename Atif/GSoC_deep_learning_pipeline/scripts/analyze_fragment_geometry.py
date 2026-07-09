#!/usr/bin/env python3
"""Analyze fragment geometry statistics (Phase 3.0.1).

Computes:
- Point spacing (nearest neighbor distances)
- Curvature distribution (PCA-based)
- Normal variance (flatness indicator)
- Fragment bounding boxes

Outputs:
- geometry_stats/fragment_*.json  (per-fragment statistics)
- geometry_stats/summary.json     (dataset-level aggregation)
- geometry_stats/recommendations.json (data-driven thresholds)

Usage:
    PYTHONPATH=src python3 scripts/analyze_fragment_geometry.py
"""

import sys
import numpy as np
from pathlib import Path
import json
import argparse

sys.path.insert(0, 'src')

from dataset_foundation.ply_io import read_point_cloud
from ground_truth_generation.fragment_geometry import (
    analyze_fragment_geometry,
    save_fragment_stats,
    compute_dataset_summary
)


def load_fragment_with_normals(fragment_id: str, dataset_dir: Path):
    """Load fragment point cloud with normals from Phase 1 outputs."""
    normalized_path = dataset_dir / "normalized" / f"{fragment_id}.ply"
    
    # Load normalized point cloud (already has normals from Phase 1)
    pc = read_point_cloud(normalized_path)
    
    # Verify normals exist
    if pc.normals is None:
        raise ValueError(f"No normals found for {fragment_id}")
    
    if len(pc.points) != len(pc.normals):
        raise ValueError(
            f"Point count mismatch for {fragment_id}: "
            f"{len(pc.points)} points vs {len(pc.normals)} normals"
        )
    
    return pc.points, pc.normals, fragment_id


def main():
    parser = argparse.ArgumentParser(
        description='Analyze fragment geometry statistics'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Path to Phase 1 dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='geometry_stats',
        help='Output directory for statistics'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("PHASE 3.0.1: FRAGMENT GEOMETRY ANALYSIS")
    print("="*60)
    print()
    
    # Load dataset metadata to get fragment IDs
    metadata_path = dataset_dir / "metadata" / "dataset.json"
    with open(metadata_path) as f:
        dataset_metadata = json.load(f)
    
    fragment_ids = dataset_metadata['fragment_ids']
    print(f"Found {len(fragment_ids)} fragments to analyze")
    print()
    
    # Analyze each fragment
    all_stats = []
    
    for i, fragment_id in enumerate(fragment_ids, 1):
        print(f"[{i}/{len(fragment_ids)}] Analyzing {fragment_id}...")
        
        try:
            # Load fragment
            points, normals, fid = load_fragment_with_normals(fragment_id, dataset_dir)
            print(f"  Loaded {len(points)} points")
            
            # Compute statistics
            stats = analyze_fragment_geometry(points, normals, fragment_id)
            
            # Save per-fragment stats
            save_fragment_stats(stats, output_dir)
            
            all_stats.append(stats)
            
            # Print summary
            print(f"  Point spacing: {stats.mean_nn_distance_mm:.3f} ± {stats.std_nn_distance_mm:.3f} mm")
            print(f"  Curvature: {stats.mean_curvature:.4f} ± {stats.std_curvature:.4f}")
            print(f"  Normal variance: {stats.mean_normal_variance:.4f} ± {stats.std_normal_variance:.4f}")
            print(f"  Extent: [{stats.extent_mm[0]:.1f}, {stats.extent_mm[1]:.1f}, {stats.extent_mm[2]:.1f}] mm")
            print()
            
        except Exception as e:
            print(f"  ✗ Error analyzing {fragment_id}: {e}")
            print()
            continue
    
    # Compute dataset-level summary
    print("="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print()
    
    summary = compute_dataset_summary(all_stats)
    
    print(f"Total fragments: {summary['fragment_count']}")
    print(f"Total points: {summary['total_points']}")
    print()
    
    print("Point Spacing:")
    print(f"  Mean across fragments: {summary['point_spacing']['mean_across_fragments_mm']:.3f} mm")
    print(f"  Std: {summary['point_spacing']['std_across_fragments_mm']:.3f} mm")
    print(f"  Range: [{summary['point_spacing']['min_mm']:.3f}, {summary['point_spacing']['max_mm']:.3f}] mm")
    print()
    
    print("Curvature:")
    print(f"  Mean across fragments: {summary['curvature']['mean_across_fragments']:.4f}")
    print(f"  Std: {summary['curvature']['std_across_fragments']:.4f}")
    print(f"  Range: [{summary['curvature']['min']:.4f}, {summary['curvature']['max']:.4f}]")
    print()
    
    print("Normal Variance (flatness indicator):")
    print(f"  Mean across fragments: {summary['normal_variance']['mean_across_fragments']:.4f}")
    print(f"  Std: {summary['normal_variance']['std_across_fragments']:.4f}")
    print(f"  Range: [{summary['normal_variance']['min']:.4f}, {summary['normal_variance']['max']:.4f}]")
    print()
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved dataset summary to {summary_path}")
    print()
    
    # Print recommendations
    print("="*60)
    print("DATA-DRIVEN RECOMMENDATIONS")
    print("="*60)
    print()
    
    recommendations = summary['recommendations']
    
    print("Fragment Adjacency Threshold:")
    print(f"  Conservative (2× point spacing): {recommendations['adjacency_threshold_mm']['conservative']:.2f} mm")
    print(f"  Moderate (3× point spacing):     {recommendations['adjacency_threshold_mm']['moderate']:.2f} mm")
    print(f"  Inclusive (5× point spacing):    {recommendations['adjacency_threshold_mm']['inclusive']:.2f} mm")
    print(f"  Reasoning: {recommendations['adjacency_threshold_mm']['reasoning']}")
    print()
    
    # Add Phase 1 RMSE context
    reconstruction_rmse = dataset_metadata.get('reconstruction', {}).get('rmse_mm', None)
    if reconstruction_rmse:
        print("Phase 1 Context:")
        print(f"  Reconstruction RMSE: {reconstruction_rmse:.2f} mm")
        print(f"  Suggested threshold: max(3× RMSE, 2× point_spacing)")
        suggested = max(3 * reconstruction_rmse, recommendations['adjacency_threshold_mm']['conservative'])
        print(f"  = max({3*reconstruction_rmse:.2f}, {recommendations['adjacency_threshold_mm']['conservative']:.2f})")
        print(f"  = {suggested:.2f} mm")
        print()
        
        # Update recommendations
        recommendations['adjacency_threshold_mm']['suggested_with_alignment_error'] = float(suggested)
    
    # Save recommendations
    rec_path = output_dir / "recommendations.json"
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"✓ Saved recommendations to {rec_path}")
    print()
    
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Review geometry_stats/*.json for per-fragment details")
    print("  2. Review geometry_stats/summary.json for dataset overview")
    print("  3. Use recommendations for Phase 3.0.2 (fragment pair analysis)")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
