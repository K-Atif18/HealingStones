"""Dataset directory layout creation and reuse.

Create/reuse the Dataset_Directory subdirectory tree and resolve per-fragment
artifact paths. Requirements: 1.1-1.3.

``ensure_layout`` creates each required subdirectory if missing, reuses
existing subdirectories without modifying their contents, and raises
``LayoutError`` naming the offending subdirectory if creation fails. It returns
a :class:`DatasetPaths` object exposing the resolved path for every subdirectory
plus helpers that resolve the per-fragment artifact path for each stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import LayoutError

__all__ = ["REQUIRED_SUBDIRS", "DatasetPaths", "ensure_layout"]

#: The required subdirectories of a Dataset_Directory, in creation order (Req 1.1).
REQUIRED_SUBDIRS: tuple[str, ...] = (
    "original",
    "fragments",
    "transforms",
    "normals",
    "normalized",
    "metadata",
)

# File extensions per artifact stage. Point clouds are binary LE PLY, transforms
# are plain-text matrices, and metadata records are JSON (see design.md).
_FRAGMENT_EXT = ".ply"
_TRANSFORM_EXT = ".txt"
_NORMALS_EXT = ".ply"
_NORMALIZED_EXT = ".ply"
_METADATA_EXT = ".json"


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved paths for a Dataset_Directory and its required subdirectories.

    Exposes the resolved path for each of the six required subdirectories and
    helper methods that resolve per-fragment artifact paths by ``fragment_id``.
    """

    root: Path
    original: Path
    fragments: Path
    transforms: Path
    normals: Path
    normalized: Path
    metadata: Path

    def subdir(self, name: str) -> Path:
        """Return the resolved path for the named required subdirectory.

        Raises ``KeyError`` if ``name`` is not one of ``REQUIRED_SUBDIRS``.
        """
        if name not in REQUIRED_SUBDIRS:
            raise KeyError(f"unknown subdirectory: {name!r}")
        return getattr(self, name)

    # --- per-fragment artifact path helpers (referenced by Fragment_ID) -------

    def fragment_pc(self, fragment_id: str) -> Path:
        """Aligned Point_Cloud PLY path under ``fragments/`` (Req 7.1)."""
        return self.fragments / f"{fragment_id}{_FRAGMENT_EXT}"

    def transform(self, fragment_id: str) -> Path:
        """4x4 Rigid_Transform text file path under ``transforms/`` (Req 3.6)."""
        return self.transforms / f"{fragment_id}{_TRANSFORM_EXT}"

    def normals_pc(self, fragment_id: str) -> Path:
        """Point_Cloud-with-Normals PLY path under ``normals/`` (Req 4.7).

        Named ``normals_pc`` to avoid shadowing the ``normals`` subdir field.
        """
        return self.normals / f"{fragment_id}{_NORMALS_EXT}"

    def normalized_pc(self, fragment_id: str) -> Path:
        """Normalized Point_Cloud PLY path under ``normalized/`` (Req 5.6).

        Named ``normalized_pc`` to avoid shadowing the ``normalized`` subdir field.
        """
        return self.normalized / f"{fragment_id}{_NORMALIZED_EXT}"

    def metadata_record(self, fragment_id: str) -> Path:
        """Per-fragment Metadata JSON path under ``metadata/`` (Req 1.6)."""
        return self.metadata / f"{fragment_id}{_METADATA_EXT}"

    def dataset_metadata(self) -> Path:
        """Dataset-level Metadata JSON path under ``metadata/`` (Req 7.4)."""
        return self.metadata / f"dataset{_METADATA_EXT}"


def ensure_layout(dataset_dir: str | Path) -> DatasetPaths:
    """Create or reuse the Dataset_Directory subdirectory tree.

    Creates the root directory and each required subdirectory if missing, and
    reuses existing subdirectories without deleting, overwriting, or modifying
    their contents (Req 1.1, 1.2). Any creation failure halts and raises
    :class:`LayoutError` naming the offending subdirectory path (Req 1.3).

    Returns a :class:`DatasetPaths` exposing every resolved subdirectory path
    and per-fragment artifact path helpers.
    """
    root = Path(dataset_dir)

    # Create the root itself first so a failure there is reported clearly.
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LayoutError(root, str(exc)) from exc

    resolved: dict[str, Path] = {}
    for name in REQUIRED_SUBDIRS:
        subdir = root / name
        # mkdir with exist_ok=True reuses an existing directory without touching
        # its contents. A pre-existing non-directory at the path is a failure.
        try:
            subdir.mkdir(exist_ok=True)
        except OSError as exc:
            raise LayoutError(subdir, str(exc)) from exc
        if not subdir.is_dir():
            raise LayoutError(subdir, "path exists but is not a directory")
        resolved[name] = subdir

    return DatasetPaths(root=root, **resolved)
