import open3d as o3d
import os
import copy
import csv
import numpy as np

def load_and_preprocess_pcd(file_path, voxel_size=1.0):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    return pcd, pcd_down

def run_icp(source_down, target_down):
    threshold = 2.0
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

def get_filenames(folder_path, limit=None):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".PLY") or f.endswith(".ply")])
    return files if limit is None else files[:limit]

if __name__ == "__main__":
    data_path = "data/fragments"
    fragment_files = get_filenames(data_path, limit=None)
    results = []

    print("Running ICP pairwise comparisons...\n")

    pair_count = 0
    total_pairs = len(fragment_files) * (len(fragment_files) - 1) // 2

    for i in range(len(fragment_files)):
        for j in range(i + 1, len(fragment_files)):  # ensures j > i
            if i == j:
                continue  # skip self-pair
            pair_count += 1

            source_file = fragment_files[i]
            target_file = fragment_files[j]

            print(f"Current pair count is {pair_count} out of total count {total_pairs}")
            print(f"Aligning {source_file} → {target_file}")

            source_raw, source_down = load_and_preprocess_pcd(os.path.join(data_path, source_file))
            target_raw, target_down = load_and_preprocess_pcd(os.path.join(data_path, target_file))

            result = run_icp(source_down, target_down)

            results.append({
                "Source": source_file,
                "Target": target_file,
                "Fitness": round(result.fitness, 6),
                "RMSE": round(result.inlier_rmse, 6)
            })

    # Write to CSV
    csv_file = "icp_results.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Source", "Target", "Fitness", "RMSE"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to: {csv_file}")


"""Explanation:
By opting for the point to plane method we are now making pairwise comparisons for the first 5 objects
This would generally make 20 combinations(5 x 4) but we are optimising it by:

Only running A→B where A < B, this way, we reduced our combinations to half, it still gives us 1-way fitness which is enough
for comparisons.


"""
