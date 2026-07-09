"""Real-dataset smoke test (Task 17.1).

Loads the eight real Caesar PLY files from ``data/`` and asserts non-zero
vertex/point counts for the full model and each of the seven fragments. This is
a plain single-execution pytest (not Hypothesis): it exercises genuine on-disk
loading of the real dataset via ``dataset_foundation.ply_io``.

Validates: Requirements 2.4.
"""

from __future__ import annotations

import os

import pytest

from dataset_foundation.ply_io import load_mesh, mesh_to_point_cloud

# Resolve ``data/`` relative to the repository root (parent of ``tests/``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

_FULL_MODEL = "caesar_full_model.ply"
_FRAGMENTS = [f"caesar_fragment_{i}.ply" for i in range(1, 8)]
_ALL_FILES = [_FULL_MODEL, *_FRAGMENTS]


def _data_files_present() -> bool:
    """True when the data directory and every expected PLY file exist."""
    if not os.path.isdir(_DATA_DIR):
        return False
    return all(os.path.isfile(os.path.join(_DATA_DIR, name)) for name in _ALL_FILES)


# Skip gracefully where the real dataset is unavailable so the suite still runs
# in other environments; the files ARE present here, so this test executes.
pytestmark = pytest.mark.skipif(
    not _data_files_present(),
    reason=f"real Caesar dataset not found under {_DATA_DIR}",
)


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_real_ply_loads_with_nonzero_geometry(filename: str) -> None:
    """Each real PLY loads as a non-empty mesh whose derived cloud matches."""
    path = os.path.join(_DATA_DIR, filename)

    mesh = load_mesh(path)
    assert mesh.vertex_count > 0, f"{filename} loaded zero vertices"
    assert mesh.face_count > 0, f"{filename} loaded zero faces"

    pc = mesh_to_point_cloud(mesh)
    assert pc.point_count == mesh.vertex_count > 0, (
        f"{filename} point count {pc.point_count} != vertex count {mesh.vertex_count}"
    )
