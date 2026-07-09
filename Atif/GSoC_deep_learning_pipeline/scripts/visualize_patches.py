#!/usr/bin/env python3
"""Visualize patches from Phase 2 outputs.

Usage:
    # Show random patches from a fragment
    python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode random --count 10
    
    # Show coverage heatmap (how many patches cover each point)
    python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode heatmap
    
    # Show a specific patch in detail
    python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode single --patch-id 42
    
    # Show all patch centers
    python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode centers
    
    # Show patch size distribution
    python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode sizes
"""

import sys
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import json

sys.path.insert(0, 'src')

from patch_generation.patch_record import read_patch_records
from dataset_foundation.ply_io import read_point_cloud


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2.
    
    Args:
        vec1: A 3d "source" vector
        vec2: A 3d "destination" vector
        
    Returns:
        A 3x3 rotation matrix
    """
    a = np.array(vec1) / np.linalg.norm(vec1)
    b = np.array(vec2) / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-10:  # Vectors are parallel
        return np.eye(3) if c > 0 else -np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix


def visualize_random_patches(fragment_id: str, num_patches: int = 10):
    """
    Visualize random patches from a fragment in interactive 3D viewer.
    
    Args:
        fragment_id: Fragment identifier
        num_patches: Number of random patches to show
    """
    
    print(f"\n{'='*60}")
    print(f"RANDOM PATCHES VISUALIZATION")
    print(f"{'='*60}")
    
    # Load fragment
    fragment_path = f'dataset/normalized/{fragment_id}.ply'
    fragment_pc = read_point_cloud(fragment_path)
    print(f"✓ Loaded fragment: {len(fragment_pc.points)} points")
    
    # Load patches
    patches_path = f'patches/{fragment_id}/patches.npz'
    patches = read_patch_records(patches_path)
    print(f"✓ Loaded patches: {len(patches)} total")
    
    # Load metadata
    metadata_path = f'patches/{fragment_id}/metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Coverage: {metadata['coverage_fraction']*100:.1f}%")
    print(f"  Mean overlap: {metadata['overlap']['mean_overlap']*100:.1f}%")
    print(f"  Patch size: {metadata['size_summary']['min_size']}-{metadata['size_summary']['max_size']} pts")
    print(f"  Mean size: {metadata['size_summary']['mean_size']:.1f} pts")
    
    # Select random patches
    np.random.seed(42)
    selected_indices = np.random.choice(len(patches), 
                                       size=min(num_patches, len(patches)), 
                                       replace=False)
    
    print(f"\nVisualizing {len(selected_indices)} random patches...")
    
    # Create Open3D geometries
    geometries = []
    
    # 1. Show full fragment (gray, semi-transparent)
    fragment_o3d = o3d.geometry.PointCloud()
    fragment_o3d.points = o3d.utility.Vector3dVector(fragment_pc.points)
    fragment_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    geometries.append(fragment_o3d)
    
    # 2. Show selected patches (each in different color)
    # Use distinct colors
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
        [0.0, 1.0, 0.5],  # Teal
        [1.0, 0.0, 0.5],  # Pink
    ]
    
    for i, patch_idx in enumerate(selected_indices):
        patch = patches[patch_idx]
        
        # Patch points
        patch_o3d = o3d.geometry.PointCloud()
        patch_o3d.points = o3d.utility.Vector3dVector(patch.global_coords)
        patch_o3d.normals = o3d.utility.Vector3dVector(patch.normals)
        
        color = colors[i % len(colors)]
        patch_o3d.paint_uniform_color(color)
        geometries.append(patch_o3d)
        
        # Patch center (sphere)
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        center_sphere.translate(patch.center)
        center_sphere.paint_uniform_color(color)
        geometries.append(center_sphere)
        
        print(f"  Patch {patch_idx}: {len(patch.global_coords)} points, center={patch.center}")
    
    print(f"\n{'='*60}")
    print("CONTROLS:")
    print("  - Mouse drag: Rotate")
    print("  - Mouse wheel: Zoom")
    print("  - Ctrl+Mouse drag: Pan")
    print("  - Press 'H' for help")
    print(f"{'='*60}\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Random Patches: {fragment_id}",
        width=1920,
        height=1080,
        left=50,
        top=50
    )


def visualize_coverage_heatmap(fragment_id: str):
    """
    Visualize how many patches each fragment point belongs to (heatmap).
    Blue = low coverage, Red = high coverage.
    """
    
    print(f"\n{'='*60}")
    print(f"COVERAGE HEATMAP")
    print(f"{'='*60}")
    
    # Load fragment
    fragment_path = f'dataset/normalized/{fragment_id}.ply'
    fragment_pc = read_point_cloud(fragment_path)
    print(f"✓ Loaded fragment: {len(fragment_pc.points)} points")
    
    # Load patches
    patches_path = f'patches/{fragment_id}/patches.npz'
    patches = read_patch_records(patches_path)
    print(f"✓ Loaded patches: {len(patches)} total")
    
    # Count how many patches each point belongs to
    point_coverage_counts = np.zeros(len(fragment_pc.points), dtype=int)
    
    for patch in patches:
        for idx in patch.source_indices:
            if idx < len(point_coverage_counts):  # Safety check
                point_coverage_counts[idx] += 1
    
    print(f"\nCoverage statistics:")
    print(f"  Uncovered points: {np.sum(point_coverage_counts == 0)}")
    print(f"  Min coverage: {point_coverage_counts.min()} patches")
    print(f"  Max coverage: {point_coverage_counts.max()} patches")
    print(f"  Mean coverage: {point_coverage_counts.mean():.2f} patches")
    print(f"  Median coverage: {np.median(point_coverage_counts):.0f} patches")
    
    # Create Open3D point cloud with heatmap coloring
    fragment_o3d = o3d.geometry.PointCloud()
    fragment_o3d.points = o3d.utility.Vector3dVector(fragment_pc.points)
    fragment_o3d.normals = o3d.utility.Vector3dVector(fragment_pc.normals)
    
    # Normalize counts to [0, 1] for colormap
    max_coverage = point_coverage_counts.max()
    if max_coverage > 0:
        normalized_counts = point_coverage_counts / max_coverage
    else:
        normalized_counts = point_coverage_counts
    
    # Apply colormap (blue = low coverage, red = high coverage)
    # Manual colormap: Blue -> Cyan -> Green -> Yellow -> Red
    colors = np.zeros((len(normalized_counts), 3))
    for i, val in enumerate(normalized_counts):
        if val < 0.25:
            # Blue to Cyan
            t = val / 0.25
            colors[i] = [0, t, 1]
        elif val < 0.5:
            # Cyan to Green
            t = (val - 0.25) / 0.25
            colors[i] = [0, 1, 1-t]
        elif val < 0.75:
            # Green to Yellow
            t = (val - 0.5) / 0.25
            colors[i] = [t, 1, 0]
        else:
            # Yellow to Red
            t = (val - 0.75) / 0.25
            colors[i] = [1, 1-t, 0]
    
    fragment_o3d.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\n{'='*60}")
    print("COLOR LEGEND:")
    print("  Blue:   Low coverage (few patches)")
    print("  Cyan:   Below average")
    print("  Green:  Average coverage")
    print("  Yellow: Above average")
    print("  Red:    High coverage (many patches)")
    print(f"{'='*60}\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [fragment_o3d],
        window_name=f"Coverage Heatmap: {fragment_id}",
        width=1920,
        height=1080
    )


def visualize_single_patch(fragment_id: str, patch_idx: int):
    """
    Visualize a single patch in detail with normals.
    """
    
    print(f"\n{'='*60}")
    print(f"SINGLE PATCH VISUALIZATION")
    print(f"{'='*60}")
    
    # Load fragment
    fragment_path = f'dataset/normalized/{fragment_id}.ply'
    fragment_pc = read_point_cloud(fragment_path)
    print(f"✓ Loaded fragment: {len(fragment_pc.points)} points")
    
    # Load patches
    patches_path = f'patches/{fragment_id}/patches.npz'
    patches = read_patch_records(patches_path)
    print(f"✓ Loaded patches: {len(patches)} total")
    
    if patch_idx >= len(patches):
        print(f"✗ Error: Patch index {patch_idx} out of range [0, {len(patches)-1}]")
        return
    
    patch = patches[patch_idx]
    
    print(f"\nPatch {patch_idx} details:")
    print(f"  Fragment: {patch.fragment_id}")
    print(f"  Points: {len(patch.global_coords)}")
    print(f"  Center: [{patch.center[0]:.2f}, {patch.center[1]:.2f}, {patch.center[2]:.2f}]")
    
    bbox_min = patch.global_coords.min(axis=0)
    bbox_max = patch.global_coords.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"  Bounding box: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to")
    print(f"                [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
    print(f"  Extent: [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}] mm")
    
    # Create geometries
    geometries = []
    
    # 1. Full fragment (light gray)
    fragment_o3d = o3d.geometry.PointCloud()
    fragment_o3d.points = o3d.utility.Vector3dVector(fragment_pc.points)
    fragment_o3d.paint_uniform_color([0.85, 0.85, 0.85])
    geometries.append(fragment_o3d)
    
    # 2. Patch points (red with normals)
    patch_o3d = o3d.geometry.PointCloud()
    patch_o3d.points = o3d.utility.Vector3dVector(patch.global_coords)
    patch_o3d.normals = o3d.utility.Vector3dVector(patch.normals)
    patch_o3d.paint_uniform_color([1.0, 0.0, 0.0])
    geometries.append(patch_o3d)
    
    # 3. Patch center (green sphere)
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    center_sphere.translate(patch.center)
    center_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    geometries.append(center_sphere)
    
    # 4. Patch radius (blue wireframe sphere - 8mm radius)
    radius_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=8.0)
    radius_sphere.translate(patch.center)
    radius_sphere.paint_uniform_color([0.0, 0.0, 1.0])
    radius_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(radius_sphere)
    geometries.append(radius_sphere)
    
    # 5. Normal arrows (sample 10 to avoid clutter)
    num_arrow_samples = min(10, len(patch.normals))
    sample_indices = np.linspace(0, len(patch.normals)-1, num_arrow_samples, dtype=int)
    
    print(f"\nShowing {num_arrow_samples} normal arrows (sampled from {len(patch.normals)} total)")
    
    for idx in sample_indices:
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.2,
            cone_radius=0.4,
            cylinder_height=2.0,
            cone_height=1.0
        )
        
        # Rotate arrow to align with normal
        normal = patch.normals[idx]
        rotation = rotation_matrix_from_vectors([0, 0, 1], normal)
        arrow.rotate(rotation, center=[0, 0, 0])
        arrow.translate(patch.global_coords[idx])
        arrow.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow arrows
        geometries.append(arrow)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION GUIDE:")
    print("  Gray cloud:   Full fragment")
    print("  Red cloud:    Selected patch points")
    print("  Green sphere: Patch center")
    print("  Blue sphere:  Patch radius (8mm)")
    print("  Yellow arrows: Surface normals (sampled)")
    print(f"{'='*60}\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Patch {patch_idx}: {fragment_id}",
        width=1920,
        height=1080
    )


def visualize_all_centers(fragment_id: str):
    """
    Visualize all patch centers as colored spheres.
    """
    
    print(f"\n{'='*60}")
    print(f"PATCH CENTERS VISUALIZATION")
    print(f"{'='*60}")
    
    # Load fragment
    fragment_path = f'dataset/normalized/{fragment_id}.ply'
    fragment_pc = read_point_cloud(fragment_path)
    print(f"✓ Loaded fragment: {len(fragment_pc.points)} points")
    
    # Load patches
    patches_path = f'patches/{fragment_id}/patches.npz'
    patches = read_patch_records(patches_path)
    print(f"✓ Loaded patches: {len(patches)} total")
    
    # Extract all centers
    centers = np.array([patch.center for patch in patches])
    
    print(f"\nPatch center statistics:")
    print(f"  Count: {len(centers)}")
    print(f"  Bounding box: {centers.min(axis=0)} to {centers.max(axis=0)}")
    
    # Compute inter-center distances
    from scipy.spatial.distance import pdist
    distances = pdist(centers)
    print(f"  Min inter-center distance: {distances.min():.2f} mm")
    print(f"  Mean inter-center distance: {distances.mean():.2f} mm")
    print(f"  Max inter-center distance: {distances.max():.2f} mm")
    
    # Create geometries
    geometries = []
    
    # 1. Full fragment (gray)
    fragment_o3d = o3d.geometry.PointCloud()
    fragment_o3d.points = o3d.utility.Vector3dVector(fragment_pc.points)
    fragment_o3d.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(fragment_o3d)
    
    # 2. All centers as spheres (red)
    centers_o3d = o3d.geometry.PointCloud()
    centers_o3d.points = o3d.utility.Vector3dVector(centers)
    centers_o3d.paint_uniform_color([1.0, 0.0, 0.0])
    geometries.append(centers_o3d)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION GUIDE:")
    print("  Gray cloud: Full fragment")
    print("  Red points: Patch centers (FPS-sampled)")
    print(f"  Total centers: {len(centers)}")
    print(f"{'='*60}\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Patch Centers: {fragment_id}",
        width=1920,
        height=1080
    )


def visualize_patch_sizes(fragment_id: str):
    """
    Visualize patches colored by size (number of points).
    """
    
    print(f"\n{'='*60}")
    print(f"PATCH SIZE VISUALIZATION")
    print(f"{'='*60}")
    
    # Load fragment
    fragment_path = f'dataset/normalized/{fragment_id}.ply'
    fragment_pc = read_point_cloud(fragment_path)
    print(f"✓ Loaded fragment: {len(fragment_pc.points)} points")
    
    # Load patches
    patches_path = f'patches/{fragment_id}/patches.npz'
    patches = read_patch_records(patches_path)
    print(f"✓ Loaded patches: {len(patches)} total")
    
    # Compute patch sizes
    patch_sizes = np.array([len(patch.global_coords) for patch in patches])
    
    print(f"\nPatch size statistics:")
    print(f"  Min: {patch_sizes.min()} points")
    print(f"  Max: {patch_sizes.max()} points")
    print(f"  Mean: {patch_sizes.mean():.2f} points")
    print(f"  Median: {np.median(patch_sizes):.0f} points")
    print(f"  Std: {patch_sizes.std():.2f} points")
    
    # Create a single point cloud with all patch points, colored by patch size
    all_points = []
    all_colors = []
    
    # Normalize sizes to [0, 1]
    size_min = patch_sizes.min()
    size_max = patch_sizes.max()
    
    for i, patch in enumerate(patches):
        patch_size = len(patch.global_coords)
        normalized_size = (patch_size - size_min) / (size_max - size_min) if size_max > size_min else 0.5
        
        # Color: Blue (small) -> Green -> Yellow -> Red (large)
        if normalized_size < 0.33:
            t = normalized_size / 0.33
            color = [0, t, 1-t]  # Blue to Cyan
        elif normalized_size < 0.67:
            t = (normalized_size - 0.33) / 0.34
            color = [t, 1, 0]  # Cyan to Yellow
        else:
            t = (normalized_size - 0.67) / 0.33
            color = [1, 1-t, 0]  # Yellow to Red
        
        all_points.append(patch.global_coords)
        all_colors.append(np.tile(color, (len(patch.global_coords), 1)))
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    # Create Open3D point cloud
    patches_o3d = o3d.geometry.PointCloud()
    patches_o3d.points = o3d.utility.Vector3dVector(all_points)
    patches_o3d.colors = o3d.utility.Vector3dVector(all_colors)
    
    print(f"\n{'='*60}")
    print("COLOR LEGEND:")
    print(f"  Blue:   Small patches ({size_min} pts)")
    print(f"  Cyan:   Below average")
    print(f"  Yellow: Above average")
    print(f"  Red:    Large patches ({size_max} pts)")
    print(f"{'='*60}\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [patches_o3d],
        window_name=f"Patch Sizes: {fragment_id}",
        width=1920,
        height=1080
    )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize patches from Phase 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show 10 random patches
  python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode random --count 10
  
  # Show coverage heatmap
  python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode heatmap
  
  # Show specific patch
  python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode single --patch-id 42
  
  # Show all patch centers
  python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode centers
  
  # Show patches colored by size
  python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode sizes
        """
    )
    
    parser.add_argument(
        '--fragment',
        type=str,
        default='fragment_caesar_fragment_1',
        help='Fragment ID (default: fragment_caesar_fragment_1)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'heatmap', 'single', 'centers', 'sizes'],
        default='random',
        help='Visualization mode (default: random)'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='Number of patches to show in random mode (default: 10)'
    )
    
    parser.add_argument(
        '--patch-id',
        type=int,
        default=0,
        help='Patch index to visualize in single mode (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    fragment_path = f'dataset/normalized/{args.fragment}.ply'
    patches_path = f'patches/{args.fragment}/patches.npz'
    
    if not Path(fragment_path).exists():
        print(f"✗ Error: Fragment file not found: {fragment_path}")
        print(f"\nAvailable fragments:")
        for f in Path('dataset/normalized').glob('*.ply'):
            print(f"  - {f.stem}")
        return 1
    
    if not Path(patches_path).exists():
        print(f"✗ Error: Patches file not found: {patches_path}")
        print(f"\nAvailable patch sets:")
        for d in Path('patches').iterdir():
            if d.is_dir() and (d / 'patches.npz').exists():
                print(f"  - {d.name}")
        return 1
    
    # Run visualization
    if args.mode == 'random':
        visualize_random_patches(args.fragment, args.count)
    elif args.mode == 'heatmap':
        visualize_coverage_heatmap(args.fragment)
    elif args.mode == 'single':
        visualize_single_patch(args.fragment, args.patch_id)
    elif args.mode == 'centers':
        visualize_all_centers(args.fragment)
    elif args.mode == 'sizes':
        visualize_patch_sizes(args.fragment)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
