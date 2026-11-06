"""Build Palace's group dict from the SGB sidecar alone, with every group lowered to solver attributes.

The rules are those of ``zh-TW/docs/design/sgb/consumer-mapping.qmd``.  The dict has four keys, each a
mapping from a name to a JSON-serialisable record:

``volumes``
    the ``dim == 3`` groups (vacuum and dielectric), by group name;
``boundary_surfaces``
    the ``dim == 2`` groups except ports: conductor surfaces and ``SA`` / ``MM`` / ``SS`` / ``AA``
    interfaces (``source == "volume_interface"``) and the ``EXT`` outer boundary
    (``source == "domain_boundary"``).  ``surface_epr`` is true when the group carries a Surface-EPR row, and
    ``rows`` holds those rows.  A row on a band names its band family and carries the family's ``margins_um``
    and ``family_bands`` (band index to groups), read from the sidecar's ``insets`` -- from ``bands`` for a
    concrete request, from ``expansion.banded[*].bands`` for a type-level one;
``pec_surfaces``
    the groups of interface type ``MA``, ``MS`` or ``SHEET`` that have a Net: the conductor surfaces;
``port_surfaces``
    ``P<port_index>`` for each lumped port, joined to its ``PORT::`` group.

Every record carries ``attributes`` (the solver attributes of the group, see :mod:`.attributes`) and ``tags``
(the native entity tags the group covers).  A record keeps nothing of the old compiler records (role,
representation, provenance, equipotential id): the route policy is recorded once in the mesh manifest, and
electrical identity is the Net.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .attributes import SurfacePart, lower_surface_group

SIDECAR_SCHEMA = "scgsim.sgb.sidecar.v2"
CONDUCTOR_INTERFACE_TYPES = ("MA", "MS", "SHEET")
SOLUTION_KINDS = ("vacuum", "dielectric")

Groups = Mapping[str, Mapping[str, Mapping[str, Any]]]


def groups_from_sidecar(sidecar: Mapping[str, Any]) -> Groups:
    """The group dict of one decoded ``scgsim.sgb.sidecar.v2`` sidecar, with attributes assigned.

    Attributes are numbered per dimension from 1 in sorted group-name order (the parts of a split group in
    signature order), the numbering SGB itself uses for physical tags.
    """
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(
            f"Palace reads sidecar schema {SIDECAR_SCHEMA!r}, got {sidecar.get('schema')!r}."
        )
    physical = sidecar["physical_groups"]
    volumes = _volume_groups(physical)
    surfaces, ports = _surface_groups(sidecar)
    boundary_surfaces = {
        name: record for name, record in surfaces.items() if name not in ports
    }
    pec_surfaces = {
        name: _pec_record(record)
        for name, record in boundary_surfaces.items()
        if record["interface_type"] in CONDUCTOR_INTERFACE_TYPES
        and record["net"] is not None
    }
    port_surfaces = _port_records(sidecar, surfaces)
    return {
        "volumes": volumes,
        "boundary_surfaces": boundary_surfaces,
        "pec_surfaces": pec_surfaces,
        "port_surfaces": port_surfaces,
    }


def _volume_groups(physical: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    volumes: dict[str, dict[str, Any]] = {}
    for attribute, name in enumerate(
        sorted(n for n, g in physical.items() if g["dim"] == 3), start=1
    ):
        record = physical[name]
        if record["kind"] not in SOLUTION_KINDS:
            raise ValueError(
                f"volume group {name!r} has kind {record['kind']!r}; Palace meshes vacuum and "
                "dielectric volumes (retained conductor volumes are Route C)."
            )
        volumes[name] = {
            "dim": 3,
            "group": name,
            "attributes": [attribute],
            "tags": list(record["native_tags"]),
            "kind": record["kind"],
            "material": record["material"],
            "entities": list(record["entities"]),
            "net": record["net"],
        }
    return volumes


def _surface_groups(
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Every dim-2 group lowered to attributes, and the names of the port groups among them."""
    physical = sidecar["physical_groups"]
    rows_by_group = _rows_by_group(sidecar)
    port_groups = {port["group"] for port in sidecar["ports"]}
    records: dict[str, dict[str, Any]] = {}
    next_attribute = 1
    for name in sorted(n for n, g in physical.items() if g["dim"] == 2):
        record = physical[name]
        parts = lower_surface_group(name, record, rows_by_group[name], next_attribute)
        next_attribute += len(parts)
        records[name] = _surface_record(
            name, record, parts, rows_by_group[name], sidecar["lineage"]["membership"]
        )
    return records, port_groups


def _rows_by_group(sidecar: Mapping[str, Any]) -> defaultdict[str, dict[str, Any]]:
    """The Surface-EPR rows of each group, each with the facts a consumer needs to select its attributes.

    A banded row names its band family and carries the family's margins and bands (band index to the groups
    that carry it), all read from the sidecar's ``insets``.
    """
    families = band_families(sidecar)
    physical = sidecar["physical_groups"]
    by_group: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for row_id, row in sidecar["surface_epr_rows"].items():
        group = physical[row["group"]]
        family = group["band"]["family"] if group["band"] else None
        record = {**row, "net": group["net"], "face_kind": group["face_kind"]}
        if family is None:
            record.update(family=None, margins_um=None, family_bands=None)
        else:
            if family not in families:
                raise ValueError(
                    f"group {row['group']!r} is a band of {family!r}, which no sidecar inset lists."
                )
            record.update(
                family=family,
                margins_um=list(families[family]["margins_um"]),
                family_bands={
                    index: list(names)
                    for index, names in families[family]["bands"].items()
                },
            )
        by_group[row["group"]][row_id] = record
    return by_group


def band_families(sidecar: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Every band family of the sidecar by family id, each with its ``margins_um`` and ``bands``.

    A concrete inset request is its own family.  A type-level request (``expansion``) lists the families it
    was expanded into under ``expansion.banded``; a family there with ``bands == {}`` has no band left to
    integrate (a lumped port covers the whole face), and the faces under ``expansion.skipped`` have no
    family at all.
    """
    families: dict[str, Mapping[str, Any]] = {}
    for inset in sidecar["insets"]:
        if "expansion" in inset:
            families.update(
                (family["request_id"], family)
                for family in inset["expansion"]["banded"]
            )
        else:
            families[inset["request_id"]] = inset
    return families


def _surface_record(
    name: str,
    record: Mapping[str, Any],
    parts: list[SurfacePart],
    rows: Mapping[str, Mapping[str, Any]],
    membership: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    return {
        "dim": 2,
        "group": name,
        "source": "domain_boundary"
        if record["interface_type"] == "EXT"
        else "volume_interface",
        "interface_type": record["interface_type"],
        "surface_epr": bool(rows),
        "rows": dict(rows),
        "face_kind": record["face_kind"],
        "entities": list(record["entities"]),
        "net": record["net"],
        "band": record["band"],
        "adjacent_domains": list(record["adjacent_domains"]),
        "adjacent_volumes": list(record["adjacent_volumes"]),
        "area_um2": record["area_um2"],
        "attributes": [part.attribute for part in parts],
        "tags": list(record["native_tags"]),
        "parts": [
            {
                **part.to_payload(),
                # Lineage membership is attribution only: it weights nothing.
                "membership": {face: membership.get(face, {}) for face in part.faces},
            }
            for part in parts
        ],
    }


def _pec_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """A conductor surface: the Entity it belongs to (Palace's conductor component), its Net and attributes."""
    (entity,) = record["entities"]
    return {
        "dim": 2,
        "group": record["group"],
        "interface_type": record["interface_type"],
        "entity": entity,
        "entities": list(record["entities"]),
        "net": record["net"],
        "attributes": list(record["attributes"]),
        "tags": list(record["tags"]),
    }


def _port_records(
    sidecar: Mapping[str, Any], surfaces: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    ports: dict[str, dict[str, Any]] = {}
    for port in sidecar["ports"]:
        key = f"P{port['port_index']}"
        if key in ports:
            raise ValueError(f"duplicate lumped-port index {port['port_index']}.")
        surface = surfaces[port["group"]]
        ports[key] = {
            "dim": 2,
            "group": port["group"],
            "port_sheet_id": port["port_sheet_id"],
            "port_index": port["port_index"],
            "source_name": port["source_name"],
            "source_layer": port["source_layer"],
            "target_layer": port["target_layer"],
            "direction": list(port["direction"]),
            "host_entity_ids": list(port["host_entity_ids"]),
            "adjacent_volumes": list(port["adjacent_volumes"]),
            "host_volumes": list(port["host_volumes"]),
            "z_um": port["z_um"],
            "faces": list(port["faces"]),
            "attributes": list(surface["attributes"]),
            "tags": list(surface["tags"]),
            "parts": list(surface["parts"]),
        }
    return ports


def rows_of(groups: Groups) -> dict[str, dict[str, Any]]:
    """Every Surface-EPR row by row id, with the side it integrates on each attribute that carries it."""
    rows: dict[str, dict[str, Any]] = {}
    for name, record in groups.get("boundary_surfaces", {}).items():
        for row_id, row in record["rows"].items():
            sides = {
                part["attribute"]: part_row["side"]
                for part in record["parts"]
                for part_row in part["rows"]
                if part_row["row"] == row_id
            }
            membership: dict[str, Any] = {}
            for part in record["parts"]:
                if any(part_row["row"] == row_id for part_row in part["rows"]):
                    membership.update(part["membership"])
            rows[row_id] = {
                **row,
                "group": name,
                "attribute_sides": sides,
                "membership": membership,
            }
    return rows


def same_length(first: float, second: float) -> bool:
    """Whether two margins in micrometres are the same length (the sidecar rounds to twelve digits)."""
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-12)
