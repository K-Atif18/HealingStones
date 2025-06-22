# refer to icp_align.py for understanding


import open3d as o3d
import os

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def downsample_point_cloud(pcd, voxel_size=1.0):
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down_pcd

def compare_original_vs_downsampled(original_pcd, downsampled_pcd):
    print(f"Original point count: {len(original_pcd.points)}")
    print(f"Downsampled point count: {len(downsampled_pcd.points)}")

    original_colored = original_pcd.paint_uniform_color([0, 0, 1])     # Blue
    downsampled_colored = downsampled_pcd.paint_uniform_color([1, 0, 0])  # Red

    o3d.visualization.draw_geometries([original_colored, downsampled_colored])

if __name__ == "__main__":
    data_path = "data/fragments"
    file_name = "NAR_ST_43B_FR_01_F_01_R_02.PLY"  
    file_path = os.path.join(data_path, file_name)

    # Load and downsample
    original = load_point_cloud(file_path)
    downsampled = downsample_point_cloud(original, voxel_size=1.0)

    # Compare visually
    compare_original_vs_downsampled(original, downsampled)
