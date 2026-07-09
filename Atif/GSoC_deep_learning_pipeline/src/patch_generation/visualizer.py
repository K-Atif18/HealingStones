"""Coverage and overlap visualization for Phase 2 (Patch Generation).

Defines the frozen :class:`VisualizationResult` dataclass and the two public
renderers :func:`visualize_coverage` and :func:`visualize_overlap`.

:func:`visualize_coverage` paints each Fragment point by whether it belongs to
at least one Patch: covered points in one color and the ``Uncovered_Points``
(``coverage.uncovered_indices``) in a distinct color (Req 8.1).

:func:`visualize_overlap` paints each Fragment point by the number of Patches
whose ``source_indices`` include that point index, mapping the per-point
membership count through a blue->red colormap normalized by the maximum count;
points belonging to no Patch are painted in a distinct color (Req 8.2).

Rendering mode (Req 8.3, 8.4): when an interactive display is available the
scene is shown in an interactive Open3D window and no image is written
(Req 8.3). Otherwise the scene is rendered off-screen and saved as a PNG of at
least 1024x1024 (the configured ``image_width`` / ``image_height``, both
validated ``>= 1024``) to ``config.image_output_path`` (Req 8.4). On any save
failure a :class:`VisualizationError` naming the target path is raised and no
partial file is left behind (Req 8.5): the image is written to a sibling
temporary file and only atomically moved into place once fully written.

This module mirrors the interactive-vs-offscreen render pattern established in
``dataset_foundation.visualizer``.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from patch_generation.errors import VisualizationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from patch_generation.config_loader import Config
    from patch_generation.coverage_analyzer import CoverageResult
    from patch_generation.patch_extractor import Patch

__all__ = [
    "VisualizationResult",
    "visualize_coverage",
    "visualize_overlap",
]

RGB = Tuple[float, float, float]

# Coverage-view colors (Req 8.1): covered points are neutral gray, uncovered
# points are a saturated red that is clearly distinct from the covered color.
COVERED_COLOR: RGB = (0.7, 0.7, 0.7)
UNCOVERED_COLOR: RGB = (0.9, 0.05, 0.05)

# Overlap-view color for points belonging to no Patch (Req 8.2): a distinct
# gray that is not part of the blue->red membership-count colormap.
ZERO_MEMBERSHIP_COLOR: RGB = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualizationResult:
    """Outcome of a visualization request.

    Attributes:
        mode: ``"interactive"`` when the scene was shown in an interactive
            window (no image saved, Req 8.3), or ``"image"`` when the scene was
            rendered off-screen and written to disk (Req 8.4).
        image_path: Consumer-supplied path of the saved PNG when
            ``mode == "image"``, otherwise ``None``.
    """

    mode: str
    image_path: Optional[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_coverage(
    points: np.ndarray,
    coverage: "CoverageResult",
    config: "Config" = None,
    interactive: Optional[bool] = None,
) -> VisualizationResult:
    """Render Fragment points colored by coverage (covered vs uncovered).

    Every point in ``coverage.uncovered_indices`` is painted
    :data:`UNCOVERED_COLOR`; all other points are painted :data:`COVERED_COLOR`.
    The two colors are clearly distinct (Req 8.1).

    Args:
        points: ``(N, 3)`` array of Fragment point coordinates.
        coverage: The Fragment's :class:`CoverageResult`; its
            ``uncovered_indices`` determine which points render as uncovered.
        config: Configuration providing the image output path and dimensions
            (required only when rendering off-screen).
        interactive: Explicit rendering mode. ``True`` forces an interactive
            window, ``False`` forces an off-screen saved image, and ``None``
            auto-detects based on display availability.

    Returns:
        A :class:`VisualizationResult` describing the chosen mode and, for image
        mode, the saved image path.

    Requirements: 8.1, 8.3, 8.4, 8.5.
    """
    pts = _as_points(points)
    n = pts.shape[0]

    colors = np.tile(np.asarray(COVERED_COLOR, dtype=np.float64), (n, 1))
    if n > 0:
        uncovered = np.asarray(
            coverage.uncovered_indices, dtype=np.int64
        ).reshape(-1)
        if uncovered.size > 0:
            valid = uncovered[(uncovered >= 0) & (uncovered < n)]
            colors[valid] = np.asarray(UNCOVERED_COLOR, dtype=np.float64)

    geometry = _colored_point_cloud(pts, colors)
    return _render(geometry, config, interactive)


def visualize_overlap(
    points: np.ndarray,
    patches: Sequence["Patch"],
    config: "Config" = None,
    interactive: Optional[bool] = None,
) -> VisualizationResult:
    """Render Fragment points colored by how many Patches include each point.

    For each point index the membership count is the number of Patches whose
    ``source_indices`` contain that index. Counts ``>= 1`` are mapped through a
    blue (low) -> red (high) colormap normalized by the maximum count; points
    with a zero count are painted the distinct :data:`ZERO_MEMBERSHIP_COLOR`
    (Req 8.2).

    Args:
        points: ``(N, 3)`` array of Fragment point coordinates.
        patches: The Fragment's patches (anything exposing ``source_indices``).
        config: Configuration providing the image output path and dimensions
            (required only when rendering off-screen).
        interactive: Explicit rendering mode (see :func:`visualize_coverage`).

    Returns:
        A :class:`VisualizationResult` describing the chosen mode and, for image
        mode, the saved image path.

    Requirements: 8.2, 8.3, 8.4, 8.5.
    """
    pts = _as_points(points)
    n = pts.shape[0]

    counts = _membership_counts(patches, n)
    colors = _overlap_colors(counts)

    geometry = _colored_point_cloud(pts, colors)
    return _render(geometry, config, interactive)


# ---------------------------------------------------------------------------
# Color / geometry assembly
# ---------------------------------------------------------------------------


def _as_points(points: np.ndarray) -> np.ndarray:
    """Validate and coerce ``points`` to a contiguous ``(N, 3)`` float64 array."""
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    return pts


def _membership_counts(patches: Sequence["Patch"], n: int) -> np.ndarray:
    """Return an ``(N,)`` int array of per-point Patch membership counts."""
    counts = np.zeros(n, dtype=np.int64)
    if n == 0:
        return counts
    for patch in patches:
        idx = np.asarray(patch.source_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            continue
        valid = idx[(idx >= 0) & (idx < n)]
        # A point appears at most once per patch; guard against duplicates so a
        # single patch never contributes more than one to a point's count.
        np.add.at(counts, np.unique(valid), 1)
    return counts


def _overlap_colors(counts: np.ndarray) -> np.ndarray:
    """Map per-point membership counts to ``(N, 3)`` RGB colors (Req 8.2).

    Zero-count points get the distinct :data:`ZERO_MEMBERSHIP_COLOR`. Counts
    ``>= 1`` are normalized by the maximum count and mapped blue (low) -> red
    (high). When the maximum count is 1 all covered points map to blue.
    """
    n = counts.shape[0]
    colors = np.tile(np.asarray(ZERO_MEMBERSHIP_COLOR, dtype=np.float64), (n, 1))
    if n == 0:
        return colors

    covered_mask = counts >= 1
    if not np.any(covered_mask):
        return colors

    max_count = int(counts.max())
    if max_count <= 1:
        # Every covered point shares the same (lowest) membership level.
        normalized = np.zeros(n, dtype=np.float64)
    else:
        # Normalize covered counts into [0, 1]; count 1 -> 0.0, max -> 1.0.
        normalized = (counts.astype(np.float64) - 1.0) / (max_count - 1)

    # Blue (0,0,1) at low -> Red (1,0,0) at high, linearly interpolated.
    covered_norm = normalized[covered_mask]
    covered_rgb = np.empty((covered_norm.shape[0], 3), dtype=np.float64)
    covered_rgb[:, 0] = covered_norm          # red rises with count
    covered_rgb[:, 1] = 0.0
    covered_rgb[:, 2] = 1.0 - covered_norm    # blue falls with count
    colors[covered_mask] = covered_rgb
    return colors


def _colored_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Build an Open3D point cloud with the given per-point colors."""
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(points)
    if points.shape[0] > 0:
        geometry.colors = o3d.utility.Vector3dVector(
            np.clip(colors, 0.0, 1.0)
        )
    return geometry


# ---------------------------------------------------------------------------
# Rendering / mode selection
# ---------------------------------------------------------------------------


def _render(
    geometry: o3d.geometry.PointCloud,
    config: "Config",
    interactive: Optional[bool],
) -> VisualizationResult:
    """Render ``geometry`` interactively or off-screen based on ``interactive``.

    When interactive rendering is selected, an Open3D window is shown and no
    image is written (Req 8.3). Otherwise the scene is rendered off-screen and
    saved to the configured path at the configured (``>= 1024``) resolution
    (Req 8.4).
    """
    if interactive is None:
        interactive = _interactive_available()

    if interactive:
        _render_interactive(geometry)
        return VisualizationResult(mode="interactive", image_path=None)

    if config is None:
        raise ValueError("config is required to render and save an off-screen image")

    width = int(config.image_width)
    height = int(config.image_height)
    image = _render_offscreen(geometry, width, height)
    _save_image_atomic(image, config.image_output_path)
    return VisualizationResult(mode="image", image_path=config.image_output_path)


def _interactive_available() -> bool:
    """Best-effort, non-blocking check for an available interactive display.

    On macOS and Windows a window server is assumed present. On other platforms
    an interactive window requires an X11 (``DISPLAY``) or Wayland
    (``WAYLAND_DISPLAY``) session. This check never opens a window, so it cannot
    hang in a headless environment.
    """
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _render_interactive(geometry: o3d.geometry.PointCloud) -> None:
    """Show the scene in an interactive Open3D window (no image is written)."""
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    try:
        viewer.add_geometry(geometry)
        viewer.run()
    finally:
        viewer.destroy_window()


def _render_offscreen(
    geometry: o3d.geometry.PointCloud,
    width: int,
    height: int,
) -> np.ndarray:
    """Render ``geometry`` to an ``(height, width, 3)`` uint8 RGB array.

    Prefers Open3D's real off-screen renderers and degrades gracefully: first
    :class:`~open3d.visualization.rendering.OffscreenRenderer` (GPU/EGL), then a
    hidden :class:`~open3d.visualization.Visualizer`, and finally a best-effort
    CPU rasterization. Every path yields an array of exactly the requested size
    so the ``>= 1024`` dimension guarantee (Req 8.4) always holds.
    """
    for renderer in (_offscreen_via_renderer, _offscreen_via_hidden_window):
        try:
            image = renderer(geometry, width, height)
        except Exception:
            image = None
        if image is not None:
            return _ensure_size(image, width, height)
    # Last-resort CPU rasterization so an image of the requested size is always
    # produced and the save-failure semantics remain testable.
    return _rasterize(geometry, width, height)


def _offscreen_via_renderer(
    geometry: o3d.geometry.PointCloud,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Render via ``OffscreenRenderer`` (requires a working GPU/EGL context)."""
    rendering = getattr(o3d.visualization, "rendering", None)
    if rendering is None or not hasattr(rendering, "OffscreenRenderer"):
        return None
    renderer = rendering.OffscreenRenderer(width, height)
    try:
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0
        if len(geometry.points) > 0:
            renderer.scene.add_geometry("geometry", geometry, material)
        _setup_camera_offscreen(renderer, geometry)
        image = renderer.render_to_image()
        return np.asarray(image)
    finally:
        del renderer


def _setup_camera_offscreen(
    renderer,
    geometry: o3d.geometry.PointCloud,
) -> None:
    """Aim the off-screen camera at the scene's bounding sphere."""
    center, radius = _scene_bounds(geometry)
    eye = center + np.array([0.0, 0.0, radius * 3.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    renderer.setup_camera(60.0, center, eye, up)


def _offscreen_via_hidden_window(
    geometry: o3d.geometry.PointCloud,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Render via a hidden (``visible=False``) Open3D window and capture pixels."""
    viewer = o3d.visualization.Visualizer()
    created = viewer.create_window(visible=False, width=width, height=height)
    if created is False:
        viewer.destroy_window()
        return None
    try:
        if len(geometry.points) > 0:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        if opt is not None:
            opt.background_color = np.array([1.0, 1.0, 1.0])
        viewer.poll_events()
        viewer.update_renderer()
        buffer = viewer.capture_screen_float_buffer(do_render=True)
        array = np.asarray(buffer)
        if array.size == 0:
            return None
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)
    finally:
        viewer.destroy_window()


def _rasterize(
    geometry: o3d.geometry.PointCloud,
    width: int,
    height: int,
) -> np.ndarray:
    """CPU fallback: orthographically project points onto a white canvas.

    Produces a valid ``(height, width, 3)`` uint8 image even when no GPU/display
    is available, preserving the requested dimensions (Req 8.4).
    """
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    points = np.asarray(geometry.points)
    if points.shape[0] == 0:
        return canvas

    if geometry.has_colors():
        colors = np.asarray(geometry.colors)
    else:
        colors = np.tile([0.4, 0.4, 0.4], (points.shape[0], 1))

    # Orthographic projection onto the XY plane, scaled to fit the canvas with a
    # small margin. Y is flipped so +Y renders upward.
    mins = points[:, :2].min(axis=0)
    maxs = points[:, :2].max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    margin = 0.05
    usable_w = width * (1.0 - 2.0 * margin)
    usable_h = height * (1.0 - 2.0 * margin)
    scale = min(usable_w / span[0], usable_h / span[1])

    xs = (points[:, 0] - mins[0]) * scale + width * margin
    ys = (points[:, 1] - mins[1]) * scale + height * margin
    px = np.clip(xs.astype(np.int64), 0, width - 1)
    py = np.clip((height - 1 - ys).astype(np.int64), 0, height - 1)
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    canvas[py, px] = rgb
    return canvas


def _scene_bounds(
    geometry: o3d.geometry.PointCloud,
) -> Tuple[np.ndarray, float]:
    """Return the scene center and a bounding radius (min 1.0) over all points."""
    points = np.asarray(geometry.points)
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.linalg.norm(maxs - mins) / 2.0)
    return center.astype(np.float64), max(radius, 1.0)


def _ensure_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Coerce a rendered image to ``(height, width, 3)`` uint8.

    Guards the ``>= 1024`` dimension guarantee against renderers that return a
    differently sized or padded buffer by cropping/padding onto a white canvas.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[0] == height and array.shape[1] == width:
        return np.ascontiguousarray(array)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    copy_h = min(height, array.shape[0])
    copy_w = min(width, array.shape[1])
    canvas[:copy_h, :copy_w] = array[:copy_h, :copy_w]
    return canvas


# ---------------------------------------------------------------------------
# Atomic image save (Req 8.5)
# ---------------------------------------------------------------------------


def _save_image_atomic(image: np.ndarray, path: str) -> None:
    """Write ``image`` (uint8 RGB) to ``path`` atomically.

    The PNG is first written to a temporary file in the same directory and then
    atomically moved into place, so a failure never leaves a partially written
    file at ``path`` (Req 8.5). Any failure raises :class:`VisualizationError`
    naming the target path.
    """
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".viz_", suffix=".png", dir=directory
        )
        os.close(fd)
        o3d_image = o3d.geometry.Image(np.ascontiguousarray(image))
        if not o3d.io.write_image(tmp_path, o3d_image):
            raise OSError("Open3D failed to encode the image")
        # Confirm a non-empty file was produced before publishing it.
        if os.path.getsize(tmp_path) == 0:
            raise OSError("wrote an empty image file")
        os.replace(tmp_path, path)
        tmp_path = None
    except Exception as exc:
        raise VisualizationError(path, str(exc)) from exc
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
