"""Patch_Directory layout creation and path resolution.

Create/reuse the Patch_Directory and resolve per-Fragment artifact paths.
Requirements: 7.1, 7.5.

``ensure_layout`` creates (or reuses) the root Patch_Directory and returns a
:class:`PatchPaths` object exposing:

* ``fragment_dir(fragment_id)``  -> ``<patch_dir>/<id>/``
* ``patch_records(fragment_id)`` -> ``<patch_dir>/<id>/patches.npz``
* ``fragment_metadata(fragment_id)`` -> ``<patch_dir>/<id>/metadata.json``
* ``dataset_metadata()``         -> ``<patch_dir>/dataset.json``

Per-Fragment subdirectories are created on demand via
:meth:`PatchPaths.ensure_fragment_dir`, which mirrors the resilient
per-Fragment handling of the pipeline. Any directory creation failure is fatal
and raises :class:`LayoutError` naming the offending path (Req 7.5). A
pre-existing non-directory at a required path is likewise a :class:`LayoutError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import LayoutError

__all__ = ["PatchPaths", "ensure_layout"]

# File names per artifact within a Fragment subdirectory / at the dataset root.
_PATCH_RECORDS_NAME = "patches.npz"
_FRAGMENT_METADATA_NAME = "metadata.json"
_DATASET_METADATA_NAME = "dataset.json"


@dataclass(frozen=True)
class PatchPaths:
    """Resolved paths for a Patch_Directory and its per-Fragment artifacts.

    Exposes the resolved per-Fragment subdirectory and artifact paths keyed by
    ``fragment_id`` plus the dataset-level metadata path. Per-Fragment
    subdirectories are created on demand by :meth:`ensure_fragment_dir`.
    """

    root: Path

    # --- per-fragment artifact path helpers (referenced by Fragment_ID) -------

    def fragment_dir(self, fragment_id: str) -> Path:
        """Per-Fragment subdirectory ``<patch_dir>/<id>/`` (Req 7.1)."""
        return self.root / str(fragment_id)

    def patch_records(self, fragment_id: str) -> Path:
        """Patch_Records archive ``<patch_dir>/<id>/patches.npz`` (Req 7.1)."""
        return self.fragment_dir(fragment_id) / _PATCH_RECORDS_NAME

    def fragment_metadata(self, fragment_id: str) -> Path:
        """Per-Fragment Patch_Metadata ``<patch_dir>/<id>/metadata.json`` (Req 7.2)."""
        return self.fragment_dir(fragment_id) / _FRAGMENT_METADATA_NAME

    def dataset_metadata(self) -> Path:
        """Dataset-level Patch_Metadata ``<patch_dir>/dataset.json`` (Req 7.2)."""
        return self.root / _DATASET_METADATA_NAME

    def ensure_fragment_dir(self, fragment_id: str) -> Path:
        """Create (or reuse) the per-Fragment subdirectory and return its path.

        Creates ``<patch_dir>/<id>/`` if missing and reuses an existing
        subdirectory without modifying its contents. Any creation failure halts
        and raises :class:`LayoutError` naming the offending path; a
        pre-existing non-directory at the path is likewise a :class:`LayoutError`
        (Req 7.5).
        """
        subdir = self.fragment_dir(fragment_id)
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LayoutError(subdir, str(exc)) from exc
        if not subdir.is_dir():
            raise LayoutError(subdir, "path exists but is not a directory")
        return subdir


def ensure_layout(patch_dir: str | Path) -> PatchPaths:
    """Create or reuse the Patch_Directory and return a :class:`PatchPaths`.

    Creates the root Patch_Directory (parents allowed) if missing and reuses an
    existing directory without deleting, overwriting, or modifying its contents
    (Req 7.1). Any creation failure halts and raises :class:`LayoutError` naming
    the offending path (Req 7.5). A pre-existing non-directory at ``patch_dir``
    is likewise a :class:`LayoutError`.

    Per-Fragment subdirectories are created on demand via
    :meth:`PatchPaths.ensure_fragment_dir`.
    """
    root = Path(patch_dir)

    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LayoutError(root, str(exc)) from exc
    if not root.is_dir():
        raise LayoutError(root, "path exists but is not a directory")

    return PatchPaths(root=root)
