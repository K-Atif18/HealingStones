import open3d as o3d
import os
import copy
import numpy as np

def load_and_preprocess_pcd(file_path, voxel_size=1.0):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    return pcd, pcd_down

def run_icp(source_down, target_down, source_raw, target_raw):
    print("Running ICP...")
    threshold = 2.0  # max distance for point matching
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold,
        np.identity(4),  # initial transformation = identity matrix
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print("Transformation matrix:\n", result.transformation)
    
    # Apply transformation to original source
    transformed_source = copy.deepcopy(source_raw)
    transformed_source.transform(result.transformation)
    
    return transformed_source, target_raw

if __name__ == "__main__":
    data_path = "data/fragments"

    # Replace with the exact filenames
    target_file = os.path.join(data_path, "NAR_ST_43B_FR_01_F_01_R_02.PLY")
    source_file = os.path.join(data_path, "NAR_ST_43B_FR_02_F_01_R_01.PLY")

    source_raw, source_down = load_and_preprocess_pcd(source_file)
    target_raw, target_down = load_and_preprocess_pcd(target_file)

    aligned_source, target = run_icp(source_down, target_down, source_raw, target_raw)

    print("Visualizing alignment...")
    o3d.visualization.draw_geometries([aligned_source.paint_uniform_color([1, 0, 0]), target.paint_uniform_color([0, 1, 0])])







"""
Understanding the code:

1. load_and_preprocess_pcd function: 

When we are loading a raw .PLY file it contains thousands or even millions of 3D points, this can slow down the icp process.
So we downsample the pcd object using voxel method, this breaks the 3D cube into voxels of a given size, eg: 1.0 unit wide.
All points that fall within the same voxel cube are replaced by a single point, this reduces the number of points.

result: for 2.PLY: original point count- 11860704, downsampled point cloud- 880549

2. run_icp function:

Threshold = 2: this is the max distance between a source point and its closest match in target, smaller value = stricter matching
In the next step, 'result' we apply icp on the downsampled data, it provides a 4x4 transformation matrix
Then we create a copy of original size source to avoid transforming the original copy

3. finally we visualise the source(red) and the target(green).

"""