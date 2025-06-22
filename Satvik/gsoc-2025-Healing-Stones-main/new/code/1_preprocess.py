import os
import numpy as np
import open3d as o3d

# Set directory paths
RAW_DIR = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/data/fragments/"
NORM_DIR = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/normalisedData"
FEATURE_DIR = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/features"

os.makedirs(NORM_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

def normalize_point_cloud(pcd):
    pcd.remove_non_finite_points()
    pcd.estimate_normals()
    centroid = pcd.get_center()
    pcd.translate(-centroid)
    return pcd

def extract_fpfh_features(pcd):
    diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    radius_normal = diag * 0.05
    radius_feature = diag * 0.1

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return fpfh

def process_all_files():
    files = []
    for root, _, filenames in os.walk(RAW_DIR):
        for fname in filenames:
            if fname.lower().endswith(".ply"):
                files.append(os.path.join(root, fname))

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            pcd = o3d.io.read_point_cloud(fpath)
            norm_pcd = normalize_point_cloud(pcd)
            o3d.io.write_point_cloud(os.path.join(NORM_DIR, fname), norm_pcd)

            fpfh = extract_fpfh_features(norm_pcd)
            np.save(os.path.join(FEATURE_DIR, fname.replace(".ply", ".npy")), fpfh.data)
        except Exception:
            pass  # Optional: add logging to track failures

if __name__ == "__main__":
    process_all_files()



"""
1. Function normalise_point_cloud: removes nan/infinite points from the mesh. 
    Computes surface normals at each point, needed for descriptors like FPFH.
    Finds the centroid of the cloud.
    hifts the whole cloud so that the centroid is at the origin (0,0,0)

2. Function extract_fpfh_features: calculates the diagonal length of the point clouds bounding box.
    radius_normal and radius_feature are adaptive radii based on size — ensures FPFH works on both large and small fragments.
    again re-estimates normals using a neighborhood defined by radius_normal.
    calculates FPFH (Fast Point Feature Histogram) — a 33D descriptor vector per point.

"""


"""
Fast Point Feature Histogram(FPFH): It helps you describe the shape of the surface around each point.

- First, you need surface normals at each point. These are used to measure angles between points, which define the surface shape.
- Second, compute SPFH(simplified), find all neighboring points within a radius, for each neighbor find the angular features between the normals and the vector.
- Last, FPFH = SPFH + weighted sum of SPFH neighbors. For each point now you have a 33-length vector. It encodes local shape and curvature.


"""
