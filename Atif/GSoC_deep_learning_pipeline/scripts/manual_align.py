"""Assisted (manual) fragment alignment: point-picking + ICP refinement.

Fully automatic FPFH+RANSAC alignment of partial fragments to a symmetric whole
is unreliable: a fragment can snap to a geometrically similar but WRONG region
and still score high fitness / low RMSE. This tool lets you supply a correct
coarse pose by clicking a few corresponding points on the fragment and on the
model; multi-scale ICP then refines it precisely. The resulting 4x4 transforms
are written into the config's ``precomputed_transforms`` so the normal pipeline
run imports them (validated) instead of recomputing.

Workflow per fragment:
  1. A window shows the FRAGMENT. Shift+click at least 3 recognisable points
     (e.g. corner of an eye, tip of the nose). Then close the window (press Q).
  2. A window shows the MODEL. Shift+click the SAME points in the SAME ORDER.
     Close the window (press Q).
  3. A coarse transform is estimated from the correspondences and refined with
     multi-scale point-to-plane ICP.
  4. A preview window shows the model (gray) + aligned fragment (color). Close it.

Requires a display. Run on your laptop:

    PYTHONPATH=src python3 scripts/manual_align.py                 # all fragments
    PYTHONPATH=src python3 scripts/manual_align.py --fragments fragment_caesar_fragment_2,fragment_caesar_fragment_3
    PYTHONPATH=src python3 scripts/manual_align.py --min-points 4  # require >=4 picks

After it finishes it updates config/default.yaml (a timestamped backup is saved),
then re-run the pipeline:

    PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
import open3d as o3d
import yaml

from dataset_foundation.config_loader import load_config
from dataset_foundation.fragment_registry import assign_fragment_ids
from dataset_foundation.normalizer import voxel_downsample
from dataset_foundation.ply_io import load_mesh, mesh_to_point_cloud

# Multi-scale ICP schedule (multipliers of icp_max_distance_mm), coarse -> fine.
_ICP_SCALES = (8.0, 4.0, 2.0, 1.0)


def _to_o3d_with_normals(pc, radius):
    pcd = pc.to_open3d()
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pcd


def _pick_points(pcd, title):
    print(f"\n>>> {title}")
    print("    Shift+Left-Click to pick points. Press 'Q' when done.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)
    vis.run()  # user picks; blocks until window closed
    vis.destroy_window()
    return vis.get_picked_points()


def _coarse_from_correspondences(source, target, src_idx, tgt_idx):
    corres = np.array([[s, t] for s, t in zip(src_idx, tgt_idx)], dtype=np.int32)
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    return estimator.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corres)
    )


def _icp_refine(source, target, init, icp_max_distance_mm, max_iter):
    transform = np.asarray(init, dtype=np.float64)
    for mult in _ICP_SCALES:
        result = o3d.pipelines.registration.registration_icp(
            source, target, icp_max_distance_mm * mult, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        transform = np.asarray(result.transformation, dtype=np.float64)
    return transform


def _align_one(fragment_pc, model_pcd, config, min_points, label):
    frag_pcd = _to_o3d_with_normals(fragment_pc, config.registration_params.normal_radius_mm)

    src_idx = _pick_points(frag_pcd, f"{label}  |  FRAGMENT - pick landmark points")
    tgt_idx = _pick_points(
        model_pcd, f"{label}  |  MODEL - pick the SAME points in the SAME order"
    )

    if len(src_idx) < min_points or len(tgt_idx) < min_points:
        raise RuntimeError(
            f"need >= {min_points} picks on each; got {len(src_idx)} and {len(tgt_idx)}"
        )
    k = min(len(src_idx), len(tgt_idx))
    coarse = _coarse_from_correspondences(frag_pcd, model_pcd, src_idx[:k], tgt_idx[:k])
    transform = _icp_refine(
        frag_pcd, model_pcd, coarse,
        config.registration_params.icp_max_distance_mm,
        config.registration_params.icp_max_iterations,
    )

    # Preview.
    frag_preview = frag_pcd.transform(np.array(transform, copy=True))
    frag_preview.paint_uniform_color([0.9, 0.2, 0.2])
    model_preview = o3d.geometry.PointCloud(model_pcd)
    model_preview.paint_uniform_color([0.7, 0.7, 0.7])
    print("    Preview: close the window to accept.")
    o3d.visualization.draw_geometries(
        [model_preview, frag_preview],
        window_name=f"{label}  |  PREVIEW - aligned fragment (close to accept)",
    )
    return transform


def _update_config_file(config_path, transforms):
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    pre = data.get("precomputed_transforms") or {}
    for fid, matrix in transforms.items():
        pre[fid] = [[float(x) for x in row] for row in matrix]
    data["precomputed_transforms"] = pre

    backup = f"{config_path}.bak.{int(time.time())}"
    shutil.copy2(config_path, backup)
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    print(f"\nUpdated {config_path} (backup: {backup})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assisted fragment alignment (point-picking + ICP).")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fragments", default="", help="Comma-separated fragment IDs or substrings (default: all).")
    parser.add_argument("--parts", default="", help="Comma-separated part numbers, e.g. 2,6,7 (matches 'part_NN' in fragment IDs). Short alternative to --fragments.")
    parser.add_argument("--min-points", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)

    model_pc = mesh_to_point_cloud(load_mesh(config.full_model_path))
    model_pc = voxel_downsample(model_pc, config.voxel_size_mm)
    model_pcd = _to_o3d_with_normals(model_pc, config.registration_params.normal_radius_mm)

    registry = assign_fragment_ids(config.fragment_paths)
    all_ids = list(registry.ids())

    # Resolve the selection. Tokens from --fragments are matched as exact IDs
    # first, then as substrings; --parts expands numbers like "6" to the
    # fragment whose ID contains "part_06". Order preserved, de-duplicated.
    selected: list[str] = []

    def _add(fid: str) -> None:
        if fid not in selected:
            selected.append(fid)

    for token in (t.strip() for t in args.fragments.split(",")):
        if not token:
            continue
        if token in all_ids:
            _add(token)
            continue
        matches = [fid for fid in all_ids if token in fid]
        if matches:
            for fid in matches:
                _add(fid)
        else:
            print(f"    WARNING: no fragment matches '{token}'")

    for num in (n.strip() for n in args.parts.split(",")):
        if not num:
            continue
        try:
            needle = f"part_{int(num):02d}"
        except ValueError:
            print(f"    WARNING: --parts value '{num}' is not a number")
            continue
        matches = [fid for fid in all_ids if needle in fid]
        if matches:
            for fid in matches:
                _add(fid)
        else:
            print(f"    WARNING: no fragment matches '{needle}'")

    if not selected:
        selected = all_ids

    transforms = {}
    for fid in selected:
        path = registry.filename_for(fid)
        frag_pc = mesh_to_point_cloud(load_mesh(path))
        frag_pc = voxel_downsample(frag_pc, config.voxel_size_mm)
        label = f"{fid}  ({os.path.basename(path)})"
        print(f"\n=== Aligning {label} ===")
        try:
            transforms[fid] = _align_one(frag_pc, model_pcd, config, args.min_points, label)
            print(f"    OK: transform captured for {fid}")
        except Exception as exc:  # keep going; skip this fragment
            print(f"    SKIPPED {fid}: {exc}")

    if transforms:
        _update_config_file(args.config, transforms)
        print("\nDone. Re-run the pipeline to apply:")
        print("  PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml")
    else:
        print("\nNo transforms captured; config unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
