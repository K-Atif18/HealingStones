"""Visualize the aligned Caesar dataset produced by the pipeline, with labels.

Loads the full model plus every aligned fragment (already written in the model
coordinate frame) and renders them together, each fragment in a distinct color
and labelled with its name.

Interactive mode uses Open3D's O3DVisualizer, which shows:
  * a floating 3D text label at each fragment's centroid, and
  * a named entry per geometry in the side panel (toggle visibility).
A color legend (fragment name -> RGB) is also printed to the console so names
map to colors in the saved image too.

Usage:
    # Auto: interactive window if a display is available, else save a PNG.
    PYTHONPATH=src python3 scripts/visualize.py

    # Force an interactive labelled window (run on your laptop):
    PYTHONPATH=src python3 scripts/visualize.py --interactive

    # Force saving an off-screen image to the configured path:
    PYTHONPATH=src python3 scripts/visualize.py --save
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from dataset_foundation.config_loader import load_config
from dataset_foundation.normalizer import voxel_downsample
from dataset_foundation.ply_io import load_mesh, mesh_to_point_cloud, read_point_cloud
from dataset_foundation.visualizer import MODEL_COLOR, assign_fragment_colors, visualize_all


def _short_label(fragment_id: str) -> str:
    """Human-friendly short label.

    'fragment_caesar_fragment_2'                 -> 'Frag 2'
    'fragment_healing_stones_hand_part_07_P01H_01' -> 'Part 07'
    """
    import re

    part = re.search(r"part_(\d+)", fragment_id)
    if part:
        return f"Part {part.group(1)}"
    tail = fragment_id.split("_")[-1]
    return f"Frag {tail}" if tail.isdigit() else fragment_id


def _load_scene(config):
    """Return (model_point_cloud, {fragment_id: PointCloud}) at display resolution."""
    model_pc = mesh_to_point_cloud(load_mesh(config.full_model_path))
    model_pc = voxel_downsample(model_pc, config.voxel_size_mm)

    fragments = {}
    frag_dir = os.path.join(config.dataset_dir, "fragments")
    for name in sorted(os.listdir(frag_dir)):
        if name.endswith(".ply"):
            fragment_id = name[: -len(".ply")]
            fragments[fragment_id] = read_point_cloud(os.path.join(frag_dir, name))
    return model_pc, fragments


def _print_legend(colors):
    print("\nColor legend (fragment -> RGB):")
    print(f"  {'MODEL':28s} {tuple(round(c, 2) for c in MODEL_COLOR)}  (gray)")
    for fid, rgb in colors.items():
        print(f"  {fid:28s} {tuple(round(c, 2) for c in rgb)}   [{_short_label(fid)}]")
    print()


def _render_interactive_with_labels(model_pc, fragments, colors):
    """Interactive O3DVisualizer with named geometries and 3D text labels."""
    import open3d as o3d
    import open3d.visualization.gui as gui

    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Aligned fragments (labelled)", 1400, 1000)
    vis.show_settings = True

    model_pcd = model_pc.to_open3d()
    model_pcd.paint_uniform_color(list(MODEL_COLOR))
    vis.add_geometry("MODEL", model_pcd)

    for fid, pc in fragments.items():
        if pc.point_count == 0:
            continue
        pcd = pc.to_open3d()
        pcd.paint_uniform_color(list(colors[fid]))
        vis.add_geometry(_short_label(fid), pcd)
        centroid = np.asarray(pc.points, dtype=np.float64).mean(axis=0)
        vis.add_3d_label(centroid, _short_label(fid))

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the aligned dataset with labels.")
    parser.add_argument("--config", default="config/default.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true", help="Force labelled interactive window.")
    mode.add_argument("--save", action="store_true", help="Force off-screen image save.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_pc, fragments = _load_scene(config)
    colors = assign_fragment_colors(list(fragments.keys()))
    _print_legend(colors)

    # Decide mode: interactive unless --save, auto-detect when neither flag set.
    want_interactive = args.interactive or (not args.save)
    if want_interactive:
        try:
            _render_interactive_with_labels(model_pc, fragments, colors)
            print("Visualization mode=interactive (labelled)")
            return 0
        except Exception as exc:
            if args.interactive:
                print(f"Interactive labelled view unavailable ({exc}); falling back to image.")
            # Fall through to off-screen image.

    # Off-screen image (labels are not drawn off-screen; use the legend above).
    result = visualize_all(model_pc, fragments, transforms=None, config=config,
                           interactive=False)
    print(f"Visualization mode={result.mode}"
          + (f" image={result.image_path}" if getattr(result, "image_path", None) else ""))
    print("Match colors to fragment names using the legend printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
