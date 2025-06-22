import open3d as o3d
import numpy as np

# Load the global assembly
pcd = o3d.io.read_point_cloud("global_assembly.ply")
print(f"Original point cloud has {len(pcd.points)} points")

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_clean = pcd.select_by_index(ind)
print(f"After outlier removal: {len(pcd_clean.points)} points")

pcd_down = pcd_clean.voxel_down_sample(voxel_size=0.8)
print(f"After downsampling: {len(pcd_down.points)} points")

colors = np.random.rand(len(pcd_down.points), 3)
pcd_down.colors = o3d.utility.Vector3dVector(colors)

# Save the cleaned and colorized model
o3d.io.write_point_cloud("global_assembly_cleaned.ply", pcd_down)
print("✅ Cleaned and colorized global assembly saved as 'global_assembly_cleaned.ply'")


o3d.visualization.draw_geometries([pcd_down])


"""
Code explanation:
1. Load the global assembly.
2. Statistical outlier removal.
3. Downsample point cloud using voxel downsampling.
4. Assign random colors per point.
"""
