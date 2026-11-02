"""Public behavior of canonical, immutable semantic fact snapshots."""

from __future__ import annotations

import pytest

from scgsim.semantics.ownership import unique_ids
from scgsim.semantics.snapshot import canonical_sha256, create_snapshot


def test_canonical_hash_is_stable_across_mapping_order() -> None:
    """Keep the canonical JSON hash stable for equivalent fact mappings."""
    first = {"entities": {"q1": {"material_kind": "vacuum"}}, "revision": 2}
    reordered = {"revision": 2, "entities": {"q1": {"material_kind": "vacuum"}}}

    assert canonical_sha256(first) == (
        "abcec92880f8e2f7b38bcc5b581cfd64c1ba10e49db7d8d2b1e9c5b605d1413c"
    )
    assert canonical_sha256(reordered) == canonical_sha256(first)


def test_snapshot_freezes_nested_facts_and_detaches_both_directions() -> None:
    """Separate source, frozen snapshot, and detached facts recursively."""
    facts = {"nested": {"items": [{"labels": {"b", "a"}}]}}
    snapshot = create_snapshot(
        facts,
        source_revision="layout-r4",
        geometry_revision="geometry-r7",
        projection_version="facts-v1",
    )
    facts["nested"]["items"][0]["labels"].add("source-only")
    facts["nested"]["items"].append({"labels": set()})

    nested = snapshot.facts["nested"]
    items = nested["items"]
    labels = items[0]["labels"]
    assert items == ({"labels": ("a", "b")},)
    with pytest.raises(TypeError):
        nested["new"] = "value"
    with pytest.raises(AttributeError):
        items.append({})

    detached = snapshot.detached()
    detached["facts"]["nested"]["items"][0]["labels"].append("copy-only")
    assert snapshot.facts["nested"]["items"][0]["labels"] == labels
    assert snapshot.detached()["facts"]["nested"]["items"][0]["labels"] == [
        "a",
        "b",
    ]


def test_missing_revisions_fall_back_to_the_canonical_content_hash() -> None:
    """Use the exact fact digest for both absent revision identifiers."""
    facts = {"entities": {"q1": {"material_kind": "vacuum"}}}

    snapshot = create_snapshot(
        facts,
        source_revision=None,
        geometry_revision=None,
        projection_version="facts-v1",
    )

    fallback = f"sha256:{canonical_sha256(facts)}"
    assert snapshot.source_revision == fallback
    assert snapshot.geometry_revision == fallback
    assert snapshot.source_revision_kind == "content_hash_fallback"
    assert snapshot.geometry_revision_kind == "content_hash_fallback"


@pytest.mark.parametrize(
    ("facts", "error", "message"),
    [
        ({1: "value"}, TypeError, "mapping keys must be strings"),
        ({"value": object()}, TypeError, "require plain values"),
        ({"value": float("nan")}, ValueError, "Out of range float values"),
    ],
)
def test_snapshot_rejects_noncanonical_fact_values(
    facts: object, error: type[Exception], message: str
) -> None:
    """Keep invalid key and scalar failures explicit at snapshot creation."""
    with pytest.raises(error, match=message):
        create_snapshot(
            facts,  # type: ignore[arg-type] -- exercise invalid runtime fact shapes
            source_revision=None,
            geometry_revision=None,
            projection_version="facts-v1",
        )


def test_revision_kind_cannot_claim_a_missing_declared_revision() -> None:
    """Reject a fallback revision kind that conflicts with a missing value."""
    with pytest.raises(ValueError, match="must be content_hash_fallback"):
        create_snapshot(
            {},
            source_revision=None,
            geometry_revision="geometry-r1",
            projection_version="facts-v1",
            source_revision_kind="declared_revision",
        )


def test_unique_ids_accepts_unhashable_string_subclasses() -> None:
    """Deduplicate legacy identifiers without requiring hashable text values."""

    class UnhashableIdentifier(str):
        """Model string identifiers whose custom value disables hashing."""

        __hash__ = None

        def __str__(self) -> str:
            return self

    owner = UnhashableIdentifier("owner-a")

    assert unique_ids([owner, owner]) == ("owner-a",)
