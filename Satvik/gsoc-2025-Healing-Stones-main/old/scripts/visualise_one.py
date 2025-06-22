import os
import open3d as o3d

def load_point_cloud(file_path):
    ext = os.path.splitext(file_path)[1].lower()  # Get extension in lowercase
    if ext == '.ply' or ext == '.obj':
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd
    else:
        raise ValueError("Unsupported file format: " + file_path)

def list_available_files(folder_path):
    files = sorted(os.listdir(folder_path))
    print("Available files in fragments folder:")
    for file in files:
        print(" -", file)
    return files

if __name__ == "__main__":
    #data_path = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/normalisedData"
    data_path = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/data/fragments"
    # List and pick one file
    available_files = list_available_files(data_path)

    if available_files:
        selected_file = available_files[16]  # Or manually pick a file name
        print(f"\nLoading and visualizing: {selected_file}")
        file_path = os.path.join(data_path, selected_file)
        pcd = load_point_cloud(file_path)
        o3d.visualization.draw_geometries([pcd])
    else:
        print("No 3D files found in data/fragments.")
