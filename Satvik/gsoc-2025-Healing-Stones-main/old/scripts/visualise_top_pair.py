from icp_align2 import load_and_preprocess_pcd, run_icp
import open3d as o3d
import os

if __name__ == "__main__":
    data_path = "data/fragments"

    # === 👇 Change source and target filenames here to test other pairs ===
    source_file = "NAR_ST_43B_FR_07_F_01_R_01.PLY"
    target_file = "NAR_ST_43B_FR_14_F_01_R_01.PLY"
    # Alternative test pair (your second best):
    # source_file = "NAR_ST_43B_FR_07_F_01_R_01.PLY"
    # target_file = "NAR_ST_43B_FR_14_F_01_R_01.PLY"

    print(f"Source: {source_file}")
    print(f"Target: {target_file}")

    # === Load and preprocess clouds ===
    source_raw, source_down = load_and_preprocess_pcd(os.path.join(data_path, source_file))
    target_raw, target_down = load_and_preprocess_pcd(os.path.join(data_path, target_file))

    # === Print point counts for sanity check ===
    print(f"Original source points: {len(source_raw.points)}")
    print(f"Target points: {len(target_raw.points)}")

    # === Run ICP ===
    transformed_source, _ = run_icp(source_down, target_down, source_raw, target_raw, method="point_to_plane")

    # === Print transformation matrix ===
    print("\nTransformation matrix:")
    print(transformed_source.get_rotation_matrix_from_xyz((0, 0, 0)))  # If you want, we can also print this
    # (Though this just returns identity, let's stick with result.transformation next time)

    # === Color the clouds ===
    target_raw.paint_uniform_color([0, 1, 0])  # Green (target)
    source_raw.paint_uniform_color([1, 0, 0])  # Red (original source)
    transformed_source.paint_uniform_color([0, 0, 1])  # Blue (aligned source)

    # === Visualize all together ===
    o3d.visualization.draw_geometries([target_raw, source_raw, transformed_source])
