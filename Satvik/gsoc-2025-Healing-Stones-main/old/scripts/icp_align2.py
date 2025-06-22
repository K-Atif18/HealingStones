import open3d as o3d
import os
import numpy as np
import copy

def load_and_preprocess_pcd(file_path, voxel_size=1.0):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    return pcd, pcd_down

def run_icp(source_down, target_down, source_raw, target_raw, method="point_to_point"):
    print(f"Running ICP using method: {method}")
    threshold = 2.0  # max distance for point matching

    if method == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError("Unknown method: " + method)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        np.identity(4),
        estimation
    )

    print("Transformation matrix:\n", result.transformation)
    print(f"Fitness: {result.fitness:.4f}")
    print(f"RMSE: {result.inlier_rmse:.4f}")

    return result



if __name__ == "__main__":
    data_path = "data/fragments"

    # Replace with the exact filenames
    target_file = os.path.join(data_path, "NAR_ST_43B_FR_01_F_01_R_02.PLY")
    source_file = os.path.join(data_path, "NAR_ST_43B_FR_02_F_01_R_01.PLY")

    source_raw, source_down = load_and_preprocess_pcd(source_file)
    target_raw, target_down = load_and_preprocess_pcd(target_file)

    # Point-to-Point ICP
    print("\n===== Point-to-Point ICP =====")
    aligned_ptp, _ = run_icp(source_down, target_down, source_raw, target_raw, method="point_to_point")

    # Point-to-Plane ICP
    print("\n===== Point-to-Plane ICP =====")
    aligned_ptpl, _ = run_icp(source_down, target_down, source_raw, target_raw, method="point_to_plane")

    # Visualize both
    aligned_ptp.paint_uniform_color([1, 0, 0])   # red
    aligned_ptpl.paint_uniform_color([0, 0, 1])  # blue
    target_raw.paint_uniform_color([0, 1, 0])    # green

    print("\nVisualizing both alignments:")
    o3d.visualization.draw_geometries([aligned_ptp, target_raw, aligned_ptpl])


"""
Observations:
This is an evolved script of the previous icp_align, here we are comparing the point-to-point method with point-to-plane:

point to point: 
Fitness: 0.0200
RMSE: 1.1749

point to plane: 
Fitness: 0.0203
RMSE: 1.1799

Both methods performed very similarly, with slight edge to point-to-plane (0.0203 > 0.0200).
The blue (point-to-plane) overlaps more accurately on surface details.

Preferred method going forward = point-to-plane

"""