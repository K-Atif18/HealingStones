"""Property-based tests for deterministic Fragment_ID assignment.

Covers task 6.2 / Property 3: Fragment_IDs are deterministic, unique, and
bijective with source filenames. Requirements: 1.4, 1.5.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from dataset_foundation.fragment_registry import assign_fragment_ids

# Fragment_IDs are derived from the sanitized filename *stem*, so two different
# filenames can sanitize to the same stem and legitimately collide. To keep the
# "distinct source filenames" precondition meaningful (and avoid false
# collisions), constrain stems to the [A-Za-z0-9] alphabet -- sanitization is a
# no-op on this alphabet -- and require the stems themselves to be unique.
_distinct_ply_filenames = st.lists(
    st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        min_size=1,
        max_size=16,
    ),
    min_size=1,
    max_size=8,
    unique=True,
).map(lambda stems: [f"{stem}.ply" for stem in stems])


# Feature: dataset-foundation, Property 3: Fragment_IDs are deterministic, unique, and bijective with filenames
@settings(max_examples=100)
@given(names=_distinct_ply_filenames)
def test_fragment_ids_deterministic_unique_bijective(names):
    """Validates: Requirements 1.4, 1.5"""
    registry = assign_fragment_ids(names)

    ids = registry.ids()

    # Pairwise-unique Fragment_IDs: one distinct ID per distinct filename.
    assert len(set(ids)) == len(names)
    assert len(ids) == len(names)

    # Determinism: assigning IDs twice on the same input yields an identical
    # id_to_filename mapping (Req 1.4).
    registry_again = assign_fragment_ids(names)
    assert dict(registry.id_to_filename) == dict(registry_again.id_to_filename)
    assert dict(registry.filename_to_id) == dict(registry_again.filename_to_id)

    # Bijection: id_to_filename and filename_to_id are consistent inverses
    # covering exactly the input filenames (Req 1.5).
    assert set(registry.filename_to_id.keys()) == set(names)
    assert set(registry.id_to_filename.keys()) == set(ids)

    for fragment_id, filename in registry.id_to_filename.items():
        assert registry.filename_to_id[filename] == fragment_id
        assert registry.filename_for(fragment_id) == filename

    for filename, fragment_id in registry.filename_to_id.items():
        assert registry.id_to_filename[fragment_id] == filename
        assert registry.id_for(filename) == fragment_id
