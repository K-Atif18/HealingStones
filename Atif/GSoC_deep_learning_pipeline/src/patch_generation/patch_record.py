"""Patch_Record serialization and deserialization.

Provides :func:`write_patch_records` and :func:`read_patch_records` using a
single compressed ``.npz`` archive per Fragment. All patches for a Fragment are
stored as concatenated arrays (``global_coords``, ``local_coords``, ``normals``,
``source_indices``) delimited by an ``(M, 2)`` ``offsets`` table of
``[start, end)`` row ranges, plus per-patch ``patch_ids``, ``center_indices``
and ``centers`` and a scalar ``fragment_id``.

Layout of the archive (M patches, T = total points across patches):

* ``fragment_id``    : scalar ``str`` (the owning Fragment_ID)
* ``patch_ids``      : ``(M,)`` int64
* ``center_indices`` : ``(M,)`` int64
* ``centers``        : ``(M, 3)`` float64
* ``offsets``        : ``(M, 2)`` int64  -- ``[start, end)`` into the stacked rows
* ``global_coords``  : ``(T, 3)`` float64
* ``local_coords``   : ``(T, 3)`` float64
* ``normals``        : ``(T, 3)`` float64
* ``source_indices`` : ``(T,)`` int64

Writes are atomic: the archive is first written to a sibling temporary file
which is then :func:`os.replace`'d onto the target path (mirroring
:mod:`dataset_foundation.metadata_manager`). A failure removes the temporary
file and raises :class:`WriteError` naming the path, leaving any previously
written archive untouched. Coordinates, local coordinates, normals and centers
are stored as ``float64`` so the round-trip is exact. An empty Fragment (no
patches) writes a valid zero-patch archive (``M = 0``, ``T = 0``).

Requirements: 4.1, 7.1, 7.3.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from patch_generation.errors import WriteError
from patch_generation.patch_extractor import Patch

__all__ = [
    "write_patch_records",
    "read_patch_records",
]


def write_patch_records(
    path: str,
    patches: list[Patch],
    fragment_id: str,
) -> str:
    """Serialize ``patches`` to a compressed ``.npz`` archive at ``path``.

    Builds stacked arrays from the list of :class:`Patch` objects: the per-point
    arrays are concatenated in patch order and delimited by an ``(M, 2)``
    ``offsets`` table, while ``patch_ids``, ``center_indices`` and ``centers``
    carry one row per patch. Coordinates, local coordinates, normals and centers
    are stored as ``float64`` so :func:`read_patch_records` reproduces them
    exactly.

    The archive is written to a temporary file in the same directory and then
    atomically moved onto ``path`` via :func:`os.replace`. The parent directory
    is created if needed. On any failure the temporary file is removed and a
    :class:`WriteError` naming ``path`` is raised; an existing archive at
    ``path`` is left untouched.

    An empty ``patches`` list produces a valid zero-patch archive.

    Args:
        path: Destination ``.npz`` path.
        patches: Patches to serialize, in the order they should be stored.
        fragment_id: Owning Fragment_ID, stored as a scalar in the archive.

    Returns:
        The path written to.

    Raises:
        WriteError: if the archive cannot be written.
    """
    m = len(patches)

    patch_ids = np.empty((m,), dtype=np.int64)
    center_indices = np.empty((m,), dtype=np.int64)
    centers = np.empty((m, 3), dtype=np.float64)
    offsets = np.empty((m, 2), dtype=np.int64)

    global_chunks: list[np.ndarray] = []
    local_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    source_chunks: list[np.ndarray] = []

    cursor = 0
    for row, patch in enumerate(patches):
        patch_ids[row] = int(patch.patch_id)
        center_indices[row] = int(patch.center_index)
        centers[row] = np.asarray(patch.center, dtype=np.float64).reshape(3)

        g = np.asarray(patch.global_coords, dtype=np.float64).reshape(-1, 3)
        lo = np.asarray(patch.local_coords, dtype=np.float64).reshape(-1, 3)
        nrm = np.asarray(patch.normals, dtype=np.float64).reshape(-1, 3)
        src = np.asarray(patch.source_indices, dtype=np.int64).reshape(-1)

        count = src.shape[0]
        offsets[row, 0] = cursor
        offsets[row, 1] = cursor + count
        cursor += count

        global_chunks.append(g)
        local_chunks.append(lo)
        normal_chunks.append(nrm)
        source_chunks.append(src)

    if global_chunks:
        global_coords = np.concatenate(global_chunks, axis=0).astype(
            np.float64, copy=False
        )
        local_coords = np.concatenate(local_chunks, axis=0).astype(
            np.float64, copy=False
        )
        normals = np.concatenate(normal_chunks, axis=0).astype(
            np.float64, copy=False
        )
        source_indices = np.concatenate(source_chunks, axis=0).astype(
            np.int64, copy=False
        )
    else:
        # Empty Fragment: valid zero-patch, zero-point archive.
        global_coords = np.empty((0, 3), dtype=np.float64)
        local_coords = np.empty((0, 3), dtype=np.float64)
        normals = np.empty((0, 3), dtype=np.float64)
        source_indices = np.empty((0,), dtype=np.int64)

    directory = os.path.dirname(os.path.abspath(path))

    tmp_path: str | None = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                fragment_id=np.asarray(str(fragment_id)),
                patch_ids=patch_ids,
                center_indices=center_indices,
                centers=centers,
                offsets=offsets,
                global_coords=global_coords,
                local_coords=local_coords,
                normals=normals,
                source_indices=source_indices,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)
        tmp_path = None  # ownership transferred to the target path
    except (OSError, ValueError, TypeError) as exc:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise WriteError(path, str(exc)) from exc

    return path


def read_patch_records(path: str) -> list[Patch]:
    """Deserialize the ``.npz`` archive at ``path`` into a list of Patches.

    Reconstructs each :class:`Patch` by slicing the stacked per-point arrays
    with the ``offsets`` table. ``patch_id``, ``fragment_id`` and
    ``source_indices`` are reproduced exactly; coordinates, local coordinates,
    normals and centers are reproduced exactly (``float64`` storage).

    Patches are returned in stored order (which is Patch_ID ascending as written
    by :func:`extract_patches`). An empty archive yields an empty list.

    Args:
        path: Path to a ``.npz`` archive written by :func:`write_patch_records`.

    Returns:
        The reconstructed list of :class:`Patch` objects.
    """
    with np.load(path, allow_pickle=False) as data:
        fragment_id = str(data["fragment_id"].item())
        patch_ids = np.asarray(data["patch_ids"], dtype=np.int64)
        center_indices = np.asarray(data["center_indices"], dtype=np.int64)
        centers = np.asarray(data["centers"], dtype=np.float64)
        offsets = np.asarray(data["offsets"], dtype=np.int64).reshape(-1, 2)
        global_coords = np.asarray(data["global_coords"], dtype=np.float64)
        local_coords = np.asarray(data["local_coords"], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64)
        source_indices = np.asarray(data["source_indices"], dtype=np.int64)

    patches: list[Patch] = []
    for row in range(offsets.shape[0]):
        start = int(offsets[row, 0])
        end = int(offsets[row, 1])
        patches.append(
            Patch(
                patch_id=int(patch_ids[row]),
                fragment_id=fragment_id,
                center=np.ascontiguousarray(centers[row], dtype=np.float64),
                center_index=int(center_indices[row]),
                source_indices=np.ascontiguousarray(
                    source_indices[start:end], dtype=np.int64
                ),
                global_coords=np.ascontiguousarray(
                    global_coords[start:end], dtype=np.float64
                ),
                local_coords=np.ascontiguousarray(
                    local_coords[start:end], dtype=np.float64
                ),
                normals=np.ascontiguousarray(
                    normals[start:end], dtype=np.float64
                ),
            )
        )
    return patches
