"""Property-based tests for the dataset directory layout.

Covers task 5.2 / Property 4: existing subdirectory contents are preserved when
``ensure_layout`` is re-run against an existing Dataset_Directory.
Requirements: 1.2 (reuse existing subdirectories without deleting, overwriting,
or modifying their contents).
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dataset_foundation.dataset_layout import REQUIRED_SUBDIRS, ensure_layout

# Filesystem-safe base filenames (no path separators, no dots that would create
# nested paths). Restricting to [A-Za-z0-9_-] keeps names portable and lets us
# treat (subdir, filename) as a stable identity key for de-duplication.
_safe_filename = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=24,
)

# A single planted file: which required subdir it lives in, its name, and its
# raw byte content (arbitrary bytes, including empty).
_planted_file = st.tuples(
    st.sampled_from(REQUIRED_SUBDIRS),
    _safe_filename,
    st.binary(min_size=0, max_size=256),
)

# A set of planted files plus a unique dataset directory name for this example.
# The per-example dataset name keeps re-runs (function-scoped tmp_path is shared
# across examples) from clashing with one another.
_example = st.tuples(
    _safe_filename,
    st.lists(_planted_file, min_size=0, max_size=20),
)


# Feature: dataset-foundation, Property 4: Existing subdirectory contents are preserved
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(example=_example)
def test_existing_subdirectory_contents_are_preserved(tmp_path, example):
    """Validates: Requirements 1.2"""
    dataset_name, planted = example

    # Distinct dataset dir per example so repeated draws on the shared,
    # function-scoped tmp_path never collide.
    dataset_dir = tmp_path / f"dataset_{dataset_name}"

    # 1. Create the layout tree.
    paths = ensure_layout(dataset_dir)

    # 2. Write the Hypothesis-generated files into the created subdirs. Collapse
    #    duplicate (subdir, filename) keys so each planted file has one expected
    #    content (last write wins), mirroring real filesystem semantics.
    expected: dict[tuple[str, str], bytes] = {}
    for subdir_name, filename, content in planted:
        expected[(subdir_name, filename)] = content

    for (subdir_name, filename), content in expected.items():
        (paths.subdir(subdir_name) / filename).write_bytes(content)

    # 3. Re-run layout initialization on the same dataset dir (Req 1.2 reuse).
    ensure_layout(dataset_dir)

    # 4. Every previously written file still exists and is byte-identical.
    for (subdir_name, filename), content in expected.items():
        planted_path = paths.subdir(subdir_name) / filename
        assert planted_path.exists(), f"missing after reuse: {planted_path}"
        assert planted_path.read_bytes() == content, (
            f"content changed after reuse: {planted_path}"
        )


# --- Task 5.3: unit tests for layout creation (success + failure) -----------
#
# Example-based tests (Requirements 1.1, 1.3):
#   * A fresh dataset dir yields all six REQUIRED_SUBDIRS as real directories.
#   * A non-directory occupying a required subdir path halts initialization and
#     raises LayoutError naming the offending path.
#   * A read-only parent directory (creation permission denied) also raises
#     LayoutError naming the offending path -- skipped when effective uid is 0,
#     since root bypasses the permission bits and chmod is ineffective.

import os

import pytest

from dataset_foundation.errors import LayoutError


def test_ensure_layout_creates_all_required_subdirs(tmp_path):
    """A fresh dataset dir gains all six required subdirs as directories (Req 1.1)."""
    dataset_dir = tmp_path / "fresh_dataset"

    paths = ensure_layout(dataset_dir)

    assert dataset_dir.is_dir()
    for name in REQUIRED_SUBDIRS:
        subdir = paths.subdir(name)
        assert subdir.is_dir(), f"expected created subdir: {subdir}"
        assert subdir == dataset_dir / name
    # Exactly the six required subdirs were created, nothing more, nothing less.
    assert len(REQUIRED_SUBDIRS) == 6


def test_ensure_layout_raises_when_nondirectory_occupies_subdir(tmp_path):
    """A regular file at a required subdir path halts and names it (Req 1.3)."""
    dataset_dir = tmp_path / "occupied_dataset"
    dataset_dir.mkdir()

    # Pre-create a *file* where the 'fragments' subdir must go. ensure_layout
    # must detect the non-directory and refuse to continue.
    clashing = dataset_dir / "fragments"
    clashing.write_bytes(b"not a directory")

    with pytest.raises(LayoutError) as excinfo:
        ensure_layout(dataset_dir)

    err = excinfo.value
    # The offending path is named both structurally (.path) and in the message.
    assert "fragments" in str(err.path)
    assert "fragments" in str(err)


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root bypasses directory permission bits, so chmod cannot force failure",
)
def test_ensure_layout_raises_on_readonly_parent(tmp_path):
    """A read-only parent makes subdir creation fail, named in LayoutError (Req 1.3)."""
    dataset_dir = tmp_path / "readonly_dataset"
    dataset_dir.mkdir()
    # Strip all write/execute permission so child subdir creation is denied.
    os.chmod(dataset_dir, 0o500)
    try:
        with pytest.raises(LayoutError) as excinfo:
            ensure_layout(dataset_dir)
        # The failure names one of the required subdirs under the read-only root.
        assert str(dataset_dir) in str(excinfo.value.path)
    finally:
        # Restore permissions so tmp_path teardown can remove the tree.
        os.chmod(dataset_dir, 0o700)
