"""Quick, memory-light inspection of the raw_hand dataset scale."""
import os
import glob
import numpy as np
import open3d as o3d

files = sorted(glob.glob("raw_hand/*.ply")) + sorted(glob.glob("raw_hand/*.PLY"))
for f in files:
    m = o3d.io.read_triangle_mesh(f)
    v = np.asarray(m.vertices)
    n = len(v)
    faces = len(np.asarray(m.triangles))
    if n:
        mn = v.min(0)
        mx = v.max(0)
        ext = mx - mn
        print(f"{os.path.basename(f):48s} verts={n:>9d} faces={faces:>9d}  "
              f"bbox=[{ext[0]:.2f},{ext[1]:.2f},{ext[2]:.2f}]  "
              f"min=[{mn[0]:.1f},{mn[1]:.1f},{mn[2]:.1f}]")
    else:
        print(f"{os.path.basename(f):48s} verts=0")
