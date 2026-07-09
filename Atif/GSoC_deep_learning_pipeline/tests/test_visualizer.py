"""Tests for :mod:`dataset_foundation.visualizer`.

Covers:
- Property 18 (task 13.2): fragment colors are pairwise-unique and distinct from
  the Full_Model color (Req 6.2).
- Visualization behavior (task 13.4): all-fragments and single-fragment scene
  assembly (Req 6.1, 6.3, 6.7); image-mode selection and image dimensions
  ``>= 1024`` (Req 6.4, 6.5); image-save-failure leaves no partial file
  (Req 6.6).

These tests run in a headless environment (no DISPLAY). The rendering tests pass
``interactive=False`` explicitly to force the off-screen/save path, which has
robust fallbacks and always writes a PNG of the requested size. The interactive
branch is never exercised because opening a window would block/hang here.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import open3d as o3d
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.config_loader import Config, load_config
from dataset_foundation.errors import WriteError
from dataset_foundation.geometry import PointCloud
from dataset_foundation.visualizer import (
    MODEL_COLOR,
    VisualizationResult,
    assign_fragment_colors,
    visualize_all,
    visualize_single,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _color_distance(a, b) -> float:
    """Euclidean distance between two RGB triples."""
    return (
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    ) ** 0.5


def _make_cloud(n: int = 50, offset: float = 0.0, seed: int = 0) -> PointCloud:
    """Build a small deterministic point cloud of ``n`` points."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-10.0, 10.0, size=(n, 3)) + offset
    return PointCloud(points=points.astype(np.float64))


def _image_config(tmp_path, width: int = 1024, height: int = 1024) -> Config:
    """Load the default config and point it at ``tmp_path`` with 1024x1024 output."""
    base = load_config("config/default.yaml")
    return dataclasses.replace(
        base,
        image_output_path=str(tmp_path / "out.png"),
        image_width=width,
        image_height=height,
    )


def _read_image_dims(path: str):
    """Return the (height, width) of a saved PNG via Open3D."""
    image = o3d.io.read_image(path)
    array = np.asarray(image)
    assert array.ndim >= 2, f"unexpected image array shape: {array.shape}"
    return array.shape[0], array.shape[1]


# ---------------------------------------------------------------------------
# Task 13.2 — Property 18: fragment colors are pairwise-unique and model-distinct
# ---------------------------------------------------------------------------


# Feature: dataset-foundation, Property 18: Fragment colors are pairwise-unique and distinct from the model
@settings(max_examples=100)
@given(
    fragment_ids=st.lists(
        st.text(min_size=1, max_size=12),
        min_size=0,
        max_size=12,
        unique=True,
    )
)
def test_property_fragment_colors_unique_and_model_distinct(fragment_ids):
    """Validates: Requirements 6.2

    ``assign_fragment_colors`` returns exactly one in-gamut color per distinct
    Fragment_ID, all colors are pairwise-unique, and each differs from
    ``MODEL_COLOR`` by a clear (non-zero) margin. The empty input yields ``{}``.
    """
    colors = assign_fragment_colors(fragment_ids)

    # Empty input -> empty mapping.
    if not fragment_ids:
        assert colors == {}
        return

    # One color per distinct Fragment_ID, keyed exactly by the inputs.
    assert set(colors.keys()) == set(fragment_ids)
    assert len(colors) == len(fragment_ids)

    color_list = list(colors.values())

    # Every channel is a valid RGB component in [0, 1].
    for rgb in color_list:
        assert len(rgb) == 3
        for channel in rgb:
            assert 0.0 <= channel <= 1.0

    # Pairwise-unique: no two fragment colors are identical.
    rounded = [tuple(round(c, 9) for c in rgb) for rgb in color_list]
    assert len(set(rounded)) == len(rounded)

    # Each fragment color is clearly distinct from the model color.
    for rgb in color_list:
        assert _color_distance(rgb, MODEL_COLOR) > 0.0


# ---------------------------------------------------------------------------
# Task 13.4 — visualization behavior
# ---------------------------------------------------------------------------


def test_visualize_all_writes_image_with_transforms(tmp_path):
    """Validates: Requirements 6.1, 6.5, 6.7

    ``visualize_all`` assembles the Full_Model plus several transformed fragments
    and, in headless (interactive=False) mode, writes a PNG of at least
    1024x1024 to the configured path.
    """
    config = _image_config(tmp_path)
    full_model = _make_cloud(n=60, seed=1)
    fragments = {
        "frag_a": _make_cloud(n=50, seed=2),
        "frag_b": _make_cloud(n=50, seed=3),
        "frag_c": _make_cloud(n=50, seed=4),
    }
    # Identity transform for one fragment, a translation for another (Req 6.7).
    identity = np.eye(4, dtype=np.float64)
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = [5.0, -3.0, 2.0]
    transforms = {"frag_a": identity, "frag_b": translation}

    result = visualize_all(
        full_model, fragments, transforms=transforms, config=config, interactive=False
    )

    assert isinstance(result, VisualizationResult)
    assert result.mode == "image"
    assert result.image_path == config.image_output_path
    assert os.path.exists(result.image_path)

    height, width = _read_image_dims(result.image_path)
    assert width >= 1024
    assert height >= 1024


def test_visualize_single_writes_image(tmp_path):
    """Validates: Requirements 6.3, 6.5

    ``visualize_single`` renders exactly one fragment over the model and writes a
    PNG of at least 1024x1024 in headless mode.
    """
    config = _image_config(tmp_path)
    full_model = _make_cloud(n=60, seed=5)
    fragment = _make_cloud(n=50, seed=6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    result = visualize_single(
        full_model, "frag_solo", fragment, transform=transform, config=config,
        interactive=False,
    )

    assert result.mode == "image"
    assert result.image_path == config.image_output_path
    assert os.path.exists(result.image_path)

    height, width = _read_image_dims(result.image_path)
    assert width >= 1024
    assert height >= 1024


def test_image_save_failure_leaves_no_partial_file(tmp_path):
    """Validates: Requirements 6.6

    When the target image path cannot be written (its parent is a regular file,
    so no temp file can be created and atomically moved into place), a
    ``WriteError`` naming the path is raised and no partial file is left behind.
    """
    # Create a regular file and use it as the *parent directory* of the target,
    # which makes both temp-file creation and the atomic replace impossible.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file, not a directory")
    target = str(blocker / "out.png")

    config = _image_config(tmp_path)
    config = dataclasses.replace(config, image_output_path=target)

    full_model = _make_cloud(n=40, seed=7)
    fragments = {"frag_a": _make_cloud(n=40, seed=8)}

    with pytest.raises(WriteError) as excinfo:
        visualize_all(full_model, fragments, config=config, interactive=False)

    # The error names the offending target path.
    assert target in str(excinfo.value)

    # No partial artifact exists at the target path.
    assert not os.path.exists(target)
