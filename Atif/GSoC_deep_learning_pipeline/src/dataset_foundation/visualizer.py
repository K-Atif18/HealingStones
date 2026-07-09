"""Scene visualization for alignment inspection.

Render Full_Model + aligned fragments (interactive or off-screen image).
Implemented in tasks 13.1/13.3. Requirements: 6.

Scene assembly (task 13.3): :func:`visualize_all` renders the Full_Model
together with every supplied Fragment, each transformed into the model frame
(Req 6.7) and painted with a pairwise-unique, model-distinct color keyed by
Fragment_ID (Req 6.1, 6.2). :func:`visualize_single` renders exactly one
Fragment over the model with no other Fragment present (Req 6.3).

Rendering mode (Req 6.4, 6.5): when an interactive window is available (or is
explicitly requested) the scene is shown in an interactive Open3D window and no
image is written. Otherwise the scene is rendered off-screen and saved as a PNG
of at least 1024x1024 (the configured ``image_width`` / ``image_height``, both
validated ``>= 1024``) to ``config.image_output_path``. On any save failure a
:class:`WriteError` naming the target path is raised and no partial file is left
behind (Req 6.6): the image is written to a sibling temporary file and only
atomically moved into place once fully written.
"""

from __future__ import annotations

import colorsys
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from dataset_foundation.errors import WriteError
from dataset_foundation.geometry import PointCloud

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataset_foundation.config_loader import Config

RGB = Tuple[float, float, float]

# Neutral gray used for the Full_Model render. Fragment colors are guaranteed
# to differ from this by a clear margin (Req 6.2).
MODEL_COLOR: RGB = (0.7, 0.7, 0.7)

# Minimum per-channel-sum distance a fragment color must keep from MODEL_COLOR.
_MODEL_COLOR_MARGIN = 0.15


def _color_distance(a: RGB, b: RGB) -> float:
    """Euclidean distance between two RGB colors."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def assign_fragment_colors(fragment_ids: Sequence[str]) -> Dict[str, RGB]:
    """Assign a deterministic, pairwise-unique render color to each Fragment_ID.

    Colors are produced by evenly spacing hues around the HSV wheel by index and
    converting to RGB. This is deterministic (the same input order yields the same
    colors) and yields pairwise-unique RGB triples. Any color that lands too close
    to ``MODEL_COLOR`` is nudged (via saturation/value adjustment) so every fragment
    color differs from the Full_Model color by a clear margin.

    Args:
        fragment_ids: Ordered sequence of Fragment_IDs. Duplicate IDs collapse to a
            single entry (mapping semantics), preserving first-seen order.

    Returns:
        Mapping from Fragment_ID to an ``(r, g, b)`` tuple with each channel in [0, 1].
        The returned colors are pairwise-unique and each differs from ``MODEL_COLOR``.

    Requirements: 6.2
    """
    # Preserve first-seen order while dropping duplicates so the hue spacing is
    # computed against the count of distinct fragments.
    unique_ids = list(dict.fromkeys(fragment_ids))
    n = len(unique_ids)

    colors: Dict[str, RGB] = {}
    used: list[RGB] = []

    for index, fragment_id in enumerate(unique_ids):
        hue = index / n if n > 0 else 0.0
        color = _make_distinct_color(hue, used)
        colors[fragment_id] = color
        used.append(color)

    return colors


def _make_distinct_color(hue: float, used: Sequence[RGB]) -> RGB:
    """Build an RGB color for ``hue`` that differs from MODEL_COLOR and existing colors.

    Starts from a fully saturated, bright color at the given hue and, if it collides
    with ``MODEL_COLOR`` or an already-assigned color, deterministically nudges
    saturation and value until it is clearly distinct.
    """
    # (saturation, value) candidates tried in a fixed, deterministic order.
    candidates = [
        (1.0, 1.0),
        (1.0, 0.6),
        (0.6, 1.0),
        (1.0, 0.8),
        (0.8, 0.5),
        (0.5, 0.9),
    ]
    for saturation, value in candidates:
        color = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb: RGB = (color[0], color[1], color[2])
        if _color_distance(rgb, MODEL_COLOR) < _MODEL_COLOR_MARGIN:
            continue
        if any(_color_distance(rgb, other) == 0.0 for other in used):
            continue
        return rgb

    # Fallback: fully saturated bright color, which for any hue on the wheel is far
    # from the neutral gray MODEL_COLOR.
    color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (color[0], color[1], color[2])


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class VisualizationResult(NamedTuple):
    """Outcome of a visualization request.

    Attributes:
        mode: ``"interactive"`` when the scene was shown in an interactive window
            (no image saved), or ``"image"`` when the scene was rendered
            off-screen and written to disk.
        image_path: Absolute/consumer-supplied path of the saved PNG when
            ``mode == "image"``, otherwise ``None``.
    """

    mode: str
    image_path: Optional[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_all(
    full_model_pc: PointCloud,
    fragments: Dict[str, PointCloud],
    transforms: Optional[Dict[str, np.ndarray]] = None,
    config: "Config" = None,
    interactive: Optional[bool] = None,
) -> VisualizationResult:
    """Render the Full_Model together with all supplied Fragments.

    Each Fragment is transformed into the Full_Model coordinate frame using its
    Rigid_Transform (Req 6.7) when one is provided in ``transforms`` (keyed by
    Fragment_ID); fragments without an entry are rendered as-is (the pipeline may
    pass clouds that are already in the model frame). Every Fragment is painted
    with a pairwise-unique color keyed by Fragment_ID that is distinct from the
    Full_Model color (Req 6.1, 6.2).

    Args:
        full_model_pc: The Full_Model point cloud (rendered in ``MODEL_COLOR``).
        fragments: Mapping of Fragment_ID to its aligned point cloud.
        transforms: Optional mapping of Fragment_ID to a 4x4 Rigid_Transform to
            apply before rendering. When ``None`` or missing an entry, the
            corresponding fragment is rendered without an additional transform.
        config: Configuration providing the image output path and dimensions.
        interactive: Explicit rendering mode. ``True`` forces an interactive
            window, ``False`` forces an off-screen saved image, and ``None``
            auto-detects based on display availability.

    Returns:
        A :class:`VisualizationResult` describing the chosen mode and, for image
        mode, the saved image path.

    Requirements: 6.1, 6.2, 6.4, 6.5, 6.6, 6.7.
    """
    colors = assign_fragment_colors(list(fragments.keys()))
    prepared: List[o3d.geometry.PointCloud] = [_model_geometry(full_model_pc)]
    for fragment_id, fragment_pc in fragments.items():
        transform = None if transforms is None else transforms.get(fragment_id)
        prepared.append(
            _fragment_geometry(fragment_pc, transform, colors[fragment_id])
        )
    return _render(prepared, config, interactive)


def visualize_single(
    full_model_pc: PointCloud,
    fragment_id: str,
    fragment_pc: PointCloud,
    transform: Optional[np.ndarray] = None,
    config: "Config" = None,
    interactive: Optional[bool] = None,
) -> VisualizationResult:
    """Render exactly one Fragment overlaid on the Full_Model (Req 6.3).

    No other Fragment is present in the scene. The single Fragment is transformed
    into the model frame via ``transform`` when provided (Req 6.7) and colored by
    its Fragment_ID (Req 6.2).

    Args:
        full_model_pc: The Full_Model point cloud (rendered in ``MODEL_COLOR``).
        fragment_id: Fragment_ID used to key the Fragment's render color.
        fragment_pc: The Fragment's aligned point cloud.
        transform: Optional 4x4 Rigid_Transform to apply before rendering.
        config: Configuration providing the image output path and dimensions.
        interactive: Explicit rendering mode (see :func:`visualize_all`).

    Returns:
        A :class:`VisualizationResult` describing the chosen mode and, for image
        mode, the saved image path.

    Requirements: 6.2, 6.3, 6.4, 6.5, 6.6, 6.7.
    """
    color = assign_fragment_colors([fragment_id])[fragment_id]
    prepared = [
        _model_geometry(full_model_pc),
        _fragment_geometry(fragment_pc, transform, color),
    ]
    return _render(prepared, config, interactive)


# ---------------------------------------------------------------------------
# Geometry assembly
# ---------------------------------------------------------------------------


def _model_geometry(full_model_pc: PointCloud) -> o3d.geometry.PointCloud:
    """Convert the Full_Model to Open3D geometry painted in ``MODEL_COLOR``."""
    geometry = full_model_pc.to_open3d()
    if len(geometry.points) > 0:
        geometry.paint_uniform_color(list(MODEL_COLOR))
    return geometry


def _fragment_geometry(
    fragment_pc: PointCloud,
    transform: Optional[np.ndarray],
    color: RGB,
) -> o3d.geometry.PointCloud:
    """Convert a Fragment to Open3D geometry in the model frame, painted ``color``.

    Applies the Rigid_Transform (Req 6.7) when supplied and paints the cloud with
    its Fragment_ID color (Req 6.2).
    """
    geometry = fragment_pc.to_open3d()
    if transform is not None and len(geometry.points) > 0:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"transform must be a 4x4 matrix, got shape {matrix.shape}"
            )
        geometry.transform(matrix)
    if len(geometry.points) > 0:
        geometry.paint_uniform_color(list(color))
    return geometry


# ---------------------------------------------------------------------------
# Rendering / mode selection
# ---------------------------------------------------------------------------


def _render(
    geometries: Sequence[o3d.geometry.PointCloud],
    config: "Config",
    interactive: Optional[bool],
) -> VisualizationResult:
    """Render ``geometries`` interactively or off-screen based on ``interactive``.

    When interactive rendering is selected, an Open3D window is shown and no image
    is written (Req 6.4). Otherwise the scene is rendered off-screen and saved to
    the configured path at the configured (``>= 1024``) resolution (Req 6.5).
    """
    if interactive is None:
        interactive = _interactive_available()

    if interactive:
        _render_interactive(geometries)
        return VisualizationResult(mode="interactive", image_path=None)

    if config is None:
        raise ValueError("config is required to render and save an off-screen image")

    width = int(config.image_width)
    height = int(config.image_height)
    image = _render_offscreen(geometries, width, height)
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


def _render_interactive(geometries: Sequence[o3d.geometry.PointCloud]) -> None:
    """Show the scene in an interactive Open3D window (no image is written)."""
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    try:
        for geometry in geometries:
            viewer.add_geometry(geometry)
        viewer.run()
    finally:
        viewer.destroy_window()


def _render_offscreen(
    geometries: Sequence[o3d.geometry.PointCloud],
    width: int,
    height: int,
) -> np.ndarray:
    """Render ``geometries`` to an ``(height, width, 3)`` uint8 RGB array.

    Prefers Open3D's real off-screen renderers and degrades gracefully: first
    :class:`~open3d.visualization.rendering.OffscreenRenderer` (GPU/EGL), then a
    hidden :class:`~open3d.visualization.Visualizer`, and finally a best-effort
    CPU rasterization. Every path yields an array of exactly the requested size
    so the ``>= 1024`` dimension guarantee (Req 6.5) always holds.
    """
    for renderer in (_offscreen_via_renderer, _offscreen_via_hidden_window):
        try:
            image = renderer(geometries, width, height)
        except Exception:
            image = None
        if image is not None:
            return _ensure_size(image, width, height)
    # Last-resort CPU rasterization so an image of the requested size is always
    # produced and the save-failure semantics remain testable.
    return _rasterize(geometries, width, height)


def _offscreen_via_renderer(
    geometries: Sequence[o3d.geometry.PointCloud],
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
        for index, geometry in enumerate(geometries):
            if len(geometry.points) > 0:
                renderer.scene.add_geometry(f"geometry_{index}", geometry, material)
        _setup_camera_offscreen(renderer, geometries, width, height)
        image = renderer.render_to_image()
        return np.asarray(image)
    finally:
        del renderer


def _setup_camera_offscreen(
    renderer,
    geometries: Sequence[o3d.geometry.PointCloud],
    width: int,
    height: int,
) -> None:
    """Aim the off-screen camera at the scene's bounding sphere."""
    center, radius = _scene_bounds(geometries)
    eye = center + np.array([0.0, 0.0, radius * 3.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    renderer.setup_camera(60.0, center, eye, up)


def _offscreen_via_hidden_window(
    geometries: Sequence[o3d.geometry.PointCloud],
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
        for geometry in geometries:
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
    geometries: Sequence[o3d.geometry.PointCloud],
    width: int,
    height: int,
) -> np.ndarray:
    """CPU fallback: orthographically project points onto a white canvas.

    Produces a valid ``(height, width, 3)`` uint8 image even when no GPU/display
    is available, preserving the requested dimensions (Req 6.5).
    """
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    points_list = []
    colors_list = []
    for geometry in geometries:
        pts = np.asarray(geometry.points)
        if pts.shape[0] == 0:
            continue
        if geometry.has_colors():
            cols = np.asarray(geometry.colors)
        else:
            cols = np.tile([0.4, 0.4, 0.4], (pts.shape[0], 1))
        points_list.append(pts)
        colors_list.append(cols)

    if not points_list:
        return canvas

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)

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
    geometries: Sequence[o3d.geometry.PointCloud],
) -> Tuple[np.ndarray, float]:
    """Return the scene center and a bounding radius (min 1.0) over all points."""
    all_points = [
        np.asarray(g.points) for g in geometries if len(g.points) > 0
    ]
    if not all_points:
        return np.zeros(3, dtype=np.float64), 1.0
    points = np.vstack(all_points)
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
# Atomic image save (Req 6.6)
# ---------------------------------------------------------------------------


def _save_image_atomic(image: np.ndarray, path: str) -> None:
    """Write ``image`` (uint8 RGB) to ``path`` atomically.

    The PNG is first written to a temporary file in the same directory and then
    atomically moved into place, so a failure never leaves a partially written
    file at ``path`` (Req 6.6). Any failure raises :class:`WriteError` naming the
    target path.
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
        raise WriteError(path, str(exc)) from exc
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
