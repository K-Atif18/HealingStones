import open3d as o3d
import pandas as pd
import numpy as np
import os
import copy
from icp_align2 import load_and_preprocess_pcd, run_icp

# Configuration
DATA_PATH = "data/fragments"
CSV_PATH = "sorted_icp_results.csv"
OUTPUT_PATH = "global_assembly.ply"


def load_raw_pcd(filename):
    file_path = os.path.join(DATA_PATH, filename)
    return o3d.io.read_point_cloud(file_path)

df = pd.read_csv(CSV_PATH)


# Strategy: Fragment that appears most in top 20 matches
top_matches = df.head(20)
base_fragment = top_matches['Source'].mode()[0]
print(f"Base fragment selected: {base_fragment}")

assembled_cloud = load_raw_pcd(base_fragment)

added_fragments = set([base_fragment])

for index, row in top_matches.iterrows():
    source = row['Source']
    target = row['Target']

    if source in added_fragments and target in added_fragments:
        continue

    if source in added_fragments:
        new_fragment_file = target
    elif target in added_fragments:
        new_fragment_file = source
    else:
        continue  # Skip pairs with no connection to current assembly

    print(f"\n🔗 Adding fragment: {new_fragment_file}")

    new_raw, new_down = load_and_preprocess_pcd(os.path.join(DATA_PATH, new_fragment_file))
    assembled_down = assembled_cloud.voxel_down_sample(voxel_size=1.0)
    assembled_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )

    # Perform ICP
    result = run_icp(new_down, assembled_down, new_raw, assembled_cloud, method="point_to_plane")

    transformed_new = copy.deepcopy(new_raw)
    transformed_new.transform(result.transformation)

    # Merge point clouds
    assembled_cloud += transformed_new

    added_fragments.add(new_fragment_file)

print("\nGlobal assembly completed.")

o3d.io.write_point_cloud(OUTPUT_PATH, assembled_cloud)
print(f"Global assembly saved to '{OUTPUT_PATH}'")



"""
Code explanation:
1. Define the data path, path to the sorted csv file and output path for global assembly.
2. Read sorted ICP results.
3. Pick base fragment. (Most frequent fragment in top 20)
4. Initialise the global assembly and keep track of the fragments that are already added using set.
5. Start chaining fragments and skip if both already added.
6. Load and preprocess fragments and then run icp, both the functions are imported from icp_align2.
7. Finally, merge the point clouds.
7. Save the global assembly.(NOTE: added to gitignore due to large size)

"""
