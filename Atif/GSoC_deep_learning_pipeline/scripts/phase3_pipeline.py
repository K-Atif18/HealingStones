#!/usr/bin/env python3
"""Complete Phase 3: Ground Truth Generation Pipeline.

This script runs the complete Phase 3 pipeline:
1. Analyze fragment pair distances
2. Detect contact regions
3. Generate positive pairs (from adjacent fragments)
4. Generate hard negatives (within fragments)
5. Generate random negatives (across non-adjacent fragments)

Usage:
    PYTHONPATH=src python3 scripts/phase3_pipeline.py --config config/ground_truth.yaml
"""

import sys
import numpy as np
from pathlib import Path
import json
import argparse
from itertools import combinations

sys.path.insert(0, 'src')

from dataset_foundation.ply_io import read_point_cloud
from patch_generation.patch_record import read_patch_records
from ground_truth_generation.fragment_pairs import (
    analyze_fragment_pair,
    generate_contact_region
)
from ground_truth_generation.pair_generation import (
    generate_positive_pairs,
    generate_hard_negatives,
    generate_random_negatives
)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Ground Truth Generation')
    parser.add_argument(
        '--config',
        type=str,
        default='config/ground_truth.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    dataset_dir = Path(config['dataset_dir'])
    patches_dir = Path(config['patches_dir'])
    output_dir = Path(config['output_dir'])
    
    adjacency_threshold = config['adjacency_threshold_mm']
    contact_threshold = config['contact_threshold_mm']
    min_contact_overlap = config['min_contact_overlap']
    
    positive_negative_ratio = config.get('positive_negative_ratio', [1, 1, 1])  # [hard, random]
    
    print("="*70)
    print("PHASE 3: GROUND TRUTH GENERATION")
    print("="*70)
    print()
    print(f"Adjacency threshold: {adjacency_threshold:.2f} mm")
    print(f"Contact threshold: {contact_threshold:.2f} mm")
    print(f"Min contact overlap: {min_contact_overlap:.1%}")
    print()
    
    # Load dataset metadata
    with open(dataset_dir / "metadata" / "dataset.json") as f:
        dataset_metadata = json.load(f)
    
    fragment_ids = dataset_metadata['fragment_ids']
    print(f"Found {len(fragment_ids)} fragments")
    print()
    
    # ========================================================================
    # PHASE 3.1: Fragment Pair Analysis
    # ========================================================================
    
    print("="*70)
    print("PHASE 3.1: FRAGMENT PAIR DISTANCE ANALYSIS")
    print("="*70)
    print()
    
    # Load all fragments
    fragments = {}
    for fid in fragment_ids:
        path = dataset_dir / "normalized" / f"{fid}.ply"
        pc = read_point_cloud(path)
        fragments[fid] = pc.points
        print(f"Loaded {fid}: {len(pc.points)} points")
    print()
    
    # Analyze all pairs
    all_pairs = list(combinations(fragment_ids, 2))
    print(f"Analyzing {len(all_pairs)} fragment pairs...")
    print()
    
    pair_stats_list = []
    adjacent_pairs = []
    
    for i, (fid_A, fid_B) in enumerate(all_pairs, 1):
        print(f"[{i}/{len(all_pairs)}] {fid_A} <-> {fid_B}")
        
        points_A = fragments[fid_A]
        points_B = fragments[fid_B]
        
        stats = analyze_fragment_pair(points_A, points_B, fid_A, fid_B)
        pair_stats_list.append(stats)
        
        print(f"  Min distance: {stats.min_distance_mm:.2f} mm")
        print(f"  Category: {stats.adjacency_category}")
        print(f"  Contact points @ {contact_threshold}mm: "
              f"{stats.contact_counts_at_thresholds[contact_threshold][0]} (A), "
              f"{stats.contact_counts_at_thresholds[contact_threshold][1]} (B)")
        
        if stats.min_distance_mm < adjacency_threshold:
            adjacent_pairs.append((fid_A, fid_B))
            print(f"  → Adjacent (will generate positive pairs)")
        print()
    
    # Save pair statistics
    pair_stats_dir = output_dir / "pair_statistics"
    pair_stats_dir.mkdir(parents=True, exist_ok=True)
    
    for stats in pair_stats_list:
        filename = f"{stats.fragment_A_id}_{stats.fragment_B_id}.json"
        with open(pair_stats_dir / filename, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
    
    print(f"✓ Saved pair statistics to {pair_stats_dir}/")
    print()
    print(f"Summary: {len(adjacent_pairs)} adjacent pairs out of {len(all_pairs)} total")
    print()
    
    # ========================================================================
    # PHASE 3.2: Contact Region Detection & Positive Pair Generation
    # ========================================================================
    
    print("="*70)
    print("PHASE 3.2: POSITIVE PAIR GENERATION")
    print("="*70)
    print()
    
    contact_regions_dir = output_dir / "contact_regions"
    contact_regions_dir.mkdir(parents=True, exist_ok=True)
    
    all_positive_pairs = []
    
    for fid_A, fid_B in adjacent_pairs:
        print(f"Generating positive pairs: {fid_A} <-> {fid_B}")
        
        points_A = fragments[fid_A]
        points_B = fragments[fid_B]
        
        # Generate contact region
        contact_region = generate_contact_region(
            points_A, points_B, fid_A, fid_B, contact_threshold
        )
        
        # Save contact region
        contact_filename = f"{fid_A}_{fid_B}.npz"
        contact_region.save(contact_regions_dir / contact_filename)
        
        print(f"  Contact points: {len(contact_region.contact_indices_A)} (A), "
              f"{len(contact_region.contact_indices_B)} (B)")
        
        # Load patches for both fragments
        patches_A = read_patch_records(patches_dir / fid_A / "patches.npz")
        patches_B = read_patch_records(patches_dir / fid_B / "patches.npz")
        
        print(f"  Loaded patches: {len(patches_A)} (A), {len(patches_B)} (B)")
        
        # Generate positive pairs
        positive_pairs = generate_positive_pairs(
            patches_A,
            patches_B,
            contact_region.contact_indices_A,
            contact_region.contact_indices_B,
            fid_A,
            fid_B,
            min_overlap=min_contact_overlap
        )
        
        print(f"  Generated {len(positive_pairs)} positive pairs")
        all_positive_pairs.extend(positive_pairs)
        print()
    
    print(f"Total positive pairs: {len(all_positive_pairs)}")
    print()
    
    # Save positive pairs
    positive_pairs_path = output_dir / "positive_pairs.json"
    with open(positive_pairs_path, 'w') as f:
        json.dump({
            'pairs': [p.to_dict() for p in all_positive_pairs],
            'count': len(all_positive_pairs)
        }, f, indent=2)
    
    print(f"✓ Saved positive pairs to {positive_pairs_path}")
    print()
    
    # ========================================================================
    # PHASE 3.3: Negative Pair Generation
    # ========================================================================
    
    print("="*70)
    print("PHASE 3.3: NEGATIVE PAIR GENERATION")
    print("="*70)
    print()
    
    # Calculate how many negatives to generate
    n_positives = len(all_positive_pairs)
    n_hard = int(n_positives * positive_negative_ratio[0])
    n_random = int(n_positives * positive_negative_ratio[1])
    
    print(f"Target counts:")
    print(f"  Positive: {n_positives}")
    print(f"  Hard negatives: {n_hard}")
    print(f"  Random negatives: {n_random}")
    print()
    
    # Generate hard negatives (within-fragment)
    print("Generating hard negatives...")
    all_hard_negatives = []
    
    n_hard_per_fragment = n_hard // len(fragment_ids)
    
    for fid in fragment_ids:
        patches = read_patch_records(patches_dir / fid / "patches.npz")
        hard_negs = generate_hard_negatives(patches, fid, n_hard_per_fragment)
        all_hard_negatives.extend(hard_negs)
        print(f"  {fid}: {len(hard_negs)} hard negatives")
    
    print(f"Total hard negatives: {len(all_hard_negatives)}")
    print()
    
    # Save hard negatives
    hard_negatives_path = output_dir / "hard_negative_pairs.json"
    with open(hard_negatives_path, 'w') as f:
        json.dump({
            'pairs': [p.to_dict() for p in all_hard_negatives],
            'count': len(all_hard_negatives)
        }, f, indent=2)
    
    print(f"✓ Saved hard negatives to {hard_negatives_path}")
    print()
    
    # Generate random negatives (cross-fragment, non-adjacent)
    print("Generating random negatives...")
    all_random_negatives = []
    
    # Get non-adjacent pairs
    adjacent_set = set(adjacent_pairs)
    non_adjacent_pairs = [
        (fid_A, fid_B) for fid_A, fid_B in all_pairs
        if (fid_A, fid_B) not in adjacent_set
    ]
    
    if len(non_adjacent_pairs) > 0:
        n_random_per_pair = n_random // len(non_adjacent_pairs)
        
        for fid_A, fid_B in non_adjacent_pairs:
            patches_A = read_patch_records(patches_dir / fid_A / "patches.npz")
            patches_B = read_patch_records(patches_dir / fid_B / "patches.npz")
            
            random_negs = generate_random_negatives(
                patches_A, patches_B, fid_A, fid_B, n_random_per_pair
            )
            all_random_negatives.extend(random_negs)
            print(f"  {fid_A} <-> {fid_B}: {len(random_negs)} random negatives")
    
    print(f"Total random negatives: {len(all_random_negatives)}")
    print()
    
    # Save random negatives
    random_negatives_path = output_dir / "random_negative_pairs.json"
    with open(random_negatives_path, 'w') as f:
        json.dump({
            'pairs': [p.to_dict() for p in all_random_negatives],
            'count': len(all_random_negatives)
        }, f, indent=2)
    
    print(f"✓ Saved random negatives to {random_negatives_path}")
    print()
    
    # ========================================================================
    # Summary & Dataset Statistics
    # ========================================================================
    
    print("="*70)
    print("PHASE 3 COMPLETE")
    print("="*70)
    print()
    
    total_pairs = n_positives + len(all_hard_negatives) + len(all_random_negatives)
    
    print("Dataset Statistics:")
    print(f"  Fragments: {len(fragment_ids)}")
    print(f"  Fragment pairs analyzed: {len(all_pairs)}")
    print(f"  Adjacent pairs: {len(adjacent_pairs)}")
    print()
    print(f"  Positive pairs: {n_positives} ({n_positives/total_pairs*100:.1f}%)")
    print(f"  Hard negative pairs: {len(all_hard_negatives)} ({len(all_hard_negatives)/total_pairs*100:.1f}%)")
    print(f"  Random negative pairs: {len(all_random_negatives)} ({len(all_random_negatives)/total_pairs*100:.1f}%)")
    print(f"  Total pairs: {total_pairs}")
    print()
    
    # Save dataset summary
    summary = {
        'config': config,
        'fragment_count': len(fragment_ids),
        'fragment_ids': fragment_ids,
        'fragment_pairs_analyzed': len(all_pairs),
        'adjacent_pairs': len(adjacent_pairs),
        'adjacent_pair_ids': adjacent_pairs,
        'pair_counts': {
            'positive': n_positives,
            'hard_negative': len(all_hard_negatives),
            'random_negative': len(all_random_negatives),
            'total': total_pairs
        },
        'ratios': {
            'positive': n_positives / total_pairs,
            'hard_negative': len(all_hard_negatives) / total_pairs,
            'random_negative': len(all_random_negatives) / total_pairs
        }
    }
    
    summary_path = output_dir / "dataset.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved dataset summary to {summary_path}")
    print()
    
    print("Output directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── dataset.json                # Dataset summary")
    print(f"    ├── positive_pairs.json         # {n_positives} positive pairs")
    print(f"    ├── hard_negative_pairs.json    # {len(all_hard_negatives)} hard negatives")
    print(f"    ├── random_negative_pairs.json  # {len(all_random_negatives)} random negatives")
    print(f"    ├── pair_statistics/            # {len(pair_stats_list)} fragment pair analyses")
    print(f"    └── contact_regions/            # {len(adjacent_pairs)} contact region files")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
