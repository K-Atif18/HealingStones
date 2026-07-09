"""Deterministic Fragment_ID assignment.

Assign deterministic, unique Fragment_IDs and maintain a bidirectional
ID<->filename mapping. A Fragment_ID is derived purely from the source
filename (``fragment_<sanitized stem>``) so the same filename resolves to the
same ID on every run against the same inputs (Req 1.4). The mapping between
Fragment_IDs and source filenames is a bijection recorded for the metadata
(Req 1.5); a collision (two distinct filenames deriving the same ID) is fatal.

Implemented in task 6.1. Requirements: 1.4, 1.5. Standard library only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

__all__ = ["FragmentRegistry", "sanitize_fragment_id", "assign_fragment_ids"]

#: Prefix applied to every derived Fragment_ID.
_ID_PREFIX = "fragment_"

#: Any character outside this class is replaced with an underscore so the
#: resulting identifier is filesystem- and reference-safe.
_UNSAFE_CHARS = re.compile(r"[^0-9A-Za-z_]")


def sanitize_fragment_id(filename: str) -> str:
    """Derive the deterministic Fragment_ID for a single source filename.

    The identifier is ``fragment_<stem>`` where ``<stem>`` is the basename with
    its extension removed and every unsafe character replaced by an underscore.
    The derivation depends only on the basename, so the same filename always
    yields the same ID (Req 1.4).

    Args:
        filename: Source filename or path of a fragment mesh.

    Returns:
        The derived Fragment_ID.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    sanitized = _UNSAFE_CHARS.sub("_", stem)
    return f"{_ID_PREFIX}{sanitized}"


@dataclass(frozen=True)
class FragmentRegistry:
    """A bijective mapping between Fragment_IDs and their source filenames.

    Both directions are exposed as read-only mappings so callers can resolve an
    ID to its filename and a filename back to its ID (Req 1.5). Instances are
    immutable; the underlying mappings cannot be mutated through the exposed
    views.
    """

    id_to_filename: Mapping[str, str]
    filename_to_id: Mapping[str, str]

    def ids(self) -> tuple[str, ...]:
        """Return the assigned Fragment_IDs in assignment order."""
        return tuple(self.id_to_filename.keys())

    def filenames(self) -> tuple[str, ...]:
        """Return the source filenames in assignment order."""
        return tuple(self.filename_to_id.keys())

    def id_for(self, filename: str) -> str:
        """Return the Fragment_ID assigned to ``filename``."""
        return self.filename_to_id[filename]

    def filename_for(self, fragment_id: str) -> str:
        """Return the source filename assigned to ``fragment_id``."""
        return self.id_to_filename[fragment_id]


def assign_fragment_ids(filenames: Iterable[str]) -> FragmentRegistry:
    """Assign deterministic, pairwise-unique Fragment_IDs to source filenames.

    Each filename is mapped to ``fragment_<sanitized stem>``. The derivation is
    deterministic, so repeated calls with the same filenames produce identical
    IDs (Req 1.4). Pairwise uniqueness is verified: if two distinct filenames
    derive the same Fragment_ID the collision is fatal and a ``ValueError`` is
    raised naming both offending filenames and the colliding ID. A filename that
    appears more than once maps to its single ID idempotently rather than being
    treated as a collision. The resulting ID<->filename mapping is a bijection
    recorded for the metadata (Req 1.5).

    Args:
        filenames: Source filenames (or paths) of the fragment meshes.

    Returns:
        A :class:`FragmentRegistry` exposing ``id_to_filename`` and
        ``filename_to_id`` mappings.

    Raises:
        ValueError: If two distinct filenames derive the same Fragment_ID.
    """
    id_to_filename: dict[str, str] = {}
    filename_to_id: dict[str, str] = {}

    for filename in filenames:
        fragment_id = sanitize_fragment_id(filename)

        existing_filename = id_to_filename.get(fragment_id)
        if existing_filename is not None and existing_filename != filename:
            raise ValueError(
                "Fragment_ID collision: filenames "
                f"{existing_filename!r} and {filename!r} both derive "
                f"Fragment_ID {fragment_id!r}"
            )

        # Idempotent for a repeated identical filename; records the mapping once.
        id_to_filename[fragment_id] = filename
        filename_to_id[filename] = fragment_id

    return FragmentRegistry(
        id_to_filename=MappingProxyType(id_to_filename),
        filename_to_id=MappingProxyType(filename_to_id),
    )
