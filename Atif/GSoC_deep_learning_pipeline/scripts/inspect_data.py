"""Quick, memory-light inspection of the Caesar dataset scale."""
import os
import numpy as np
import open3d as o3d

files = ["data/caesar_full_model.ply"] + [f"data/caesar_fragment_{i}.ply" for i in range(1, 8)]
for f in files:
    m = o3d.io.read_triangle_mesh(f)
    v = np.asarray(m.vertices)
    n = len(v)
    if n:
        ext = v.max(0) - v.min(0)
        print(f"{os.path.basename(f):26s} verts={n:>9d}  bbox_mm=[{ext[0]:.1f},{ext[1]:.1f},{ext[2]:.1f}]")
    else:
        print(f"{os.path.basename(f):26s} verts=0")
