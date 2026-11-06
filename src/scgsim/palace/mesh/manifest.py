"""Build the mesh manifest (schema 2): what produced ``palace.msh`` and how its groups map to attributes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .physical_groups import Groups
from .provenance import SGB_RUNTIME_AUTHORITY

MANIFEST_SCHEMA_VERSION = 2

_ATTRIBUTE_FIELDS = (
    "interface_type",
    "face_kind",
    "entities",
    "net",
    "adjacent_volumes",
    "port_index",
    "host_entity_ids",
)


def build_mesh_manifest(  # noqa: PLR0913 -- the manifest records each independent identity of the run
    *,
    groups: Groups,
    sidecar: Mapping[str, Any],
    sidecar_file: Mapping[str, Any],
    mesh_name: str,
    route: str,
    route_a_thin_film: Mapping[str, Any] | None,
    meshing: Mapping[str, Any],
) -> dict[str, Any]:
    """The manifest of one mesh.

    ``sidecar_file`` is the path (relative to the run) and SHA-256 of the sidecar, ``meshing`` the
    settings and quality report of the meshing module plus the observations of the written mesh.
    ``groups`` lists one entry per solver attribute with the primary SGB group it came from.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "route": route,
        "route_policy": dict(sidecar["route_policy"]),
        **(
            {"route_a_thin_film": dict(route_a_thin_film)}
            if route_a_thin_film is not None
            else {}
        ),
        "mesh_file": mesh_name,
        "source": {
            "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
            "sidecar_schema": sidecar["schema"],
            "model_identity_digest": sidecar["model"]["identity_digest"],
            "sidecar": dict(sidecar_file),
        },
        "arrangement": dict(sidecar["arrangement"]),
        "curved_wall_faces": len(sidecar["curved_wall_faces"]),
        "insets": [_inset_record(inset) for inset in sidecar["insets"]],
        "meshing": dict(meshing),
        "groups": attribute_entries(groups),
    }


def _inset_record(inset: Mapping[str, Any]) -> dict[str, Any]:
    """One inset request as given; a type-level request also lists the families it banded and the faces it skipped."""
    record = {
        "request_id": inset["request_id"],
        "support": inset["support"],
        "margins_um": inset["margins_um"],
    }
    if "expansion" in inset:
        record["banded"] = [
            family["request_id"] for family in inset["expansion"]["banded"]
        ]
        record["skipped"] = inset["expansion"]["skipped"]
    return record


def attribute_entries(groups: Groups) -> list[dict[str, Any]]:
    """One entry per solver attribute: its dimension, number, name, entity tags and primary SGB group."""
    entries: list[dict[str, Any]] = []
    for name, record in groups["volumes"].items():
        entries.append(
            _entry(
                "volumes",
                record,
                name,
                record["attributes"][0],
                record["tags"],
                kind=record["kind"],
                material=record["material"],
            )
        )
    for section in ("boundary_surfaces", "port_surfaces"):
        for record in groups[section].values():
            entries.extend(
                _entry(
                    section,
                    record,
                    part["name"],
                    part["attribute"],
                    part["tags"],
                    rows=part["rows"],
                )
                for part in record["parts"]
            )
    return sorted(entries, key=lambda e: (e["dimension"], e["attributes"][0]))


def _entry(  # noqa: PLR0913 -- the fixed identity of an attribute plus its section's extras
    section: str,
    record: Mapping[str, Any],
    name: str,
    attribute: int,
    tags: list[int],
    **extra: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "group": record["group"],
        "section": section,
        "dimension": int(record["dim"]),
        "attributes": [int(attribute)],
        "tags": list(tags),
    }
    entry.update({key: record[key] for key in _ATTRIBUTE_FIELDS if key in record})
    entry.update(extra)
    return entry
