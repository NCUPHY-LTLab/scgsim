"""Project each neutral lumped support from source to effective (native) coordinates.

A lumped support is the sheet a lumped port or lumped RLC sits on. The same
physical layer map that moves every body in the model (the Thin Film approximation
removes stack-film thickness) must also move the support, or its contacts would
no longer touch their conductors. This module only computes that projection and
checks the support is consistent with the model; it never touches AEDT.
Preparation uses it to write the supports, and request decoding uses it to
re-derive and compare them.
"""

from typing import Any

from scgsim.aedt.specs.modeling import PhysicalLayerSpec, _ZMap
from scgsim.geometry.models.physical import PLAN_SCHEMA, PhysicalRepresentationPlan


def _prepared_lumped_supports(supports, source):
    """Derive explicit support XY/Z from the same physical map as all bodies."""
    trace = source["modeling"]
    # `current` marks a neutral physical plan; the older `_ZMap` form is read-only.
    current = trace.get("schema_version") == PLAN_SCHEMA
    mapping = (
        PhysicalRepresentationPlan.from_payload(trace)
        if current
        else _ZMap(
            trace["modeling"],
            tuple(PhysicalLayerSpec(**item) for item in trace["source_layers"]),
        )
    )
    layers = mapping.declaration_index if current else mapping.layers
    conductors = source["conductors"]
    available = {item["semantic_id"] for item in conductors}
    available.update(item["source_semantic_id"] for item in conductors)
    records = [
        _support_record(support, mapping, layers, available, current=current)
        for support in supports
    ]
    if len({item["support_id"] for item in records}) != len(records):
        raise ValueError("neutral support IDs must be unique")
    return records


def _support_record(support, mapping, layers, available, *, current: bool):
    """Validate one support, then return its source payload and effective geometry."""
    contacts = _checked_contacts(support, layers, available, current=current)
    a, b, c, d, _, _ = support.source_transform
    nx, ny, nz = support.normal
    normal = (a * nx + b * ny, c * nx + d * ny, nz)
    layer_id = support.contact_physical_layer_id

    def project(point):
        return _project_point(mapping, point, layer_id, current=current)

    def effective_xy(point):
        # Rings stay XY; the common XYZ map receives the authored locator plane.
        world = support.world_point((*point, support.source_z_um))
        return list(project(world)[:2] if current else world[:2])

    return {
        "support_id": support.support_id,
        "source": support.to_payload(),
        "effective": {
            "exterior": [effective_xy(point) for point in support.exterior],
            "holes": [
                [effective_xy(point) for point in ring] for ring in support.holes
            ],
            "normal": list(normal),
            "z_um": project((0.0, 0.0, support.source_z_um))[2],
            "contact_points_um": [list(project(point)) for point in contacts],
            "contact_physical_layer_id": layer_id,
        },
    }


def _checked_contacts(
    support, layers, available: set[str], *, current: bool
) -> tuple[Any, ...]:
    """Check ownership and the contact plane, and return the world-space contacts."""
    if not support.exterior:
        raise ValueError(
            f"support {support.support_id!r} has no resolved original source polygon"
        )
    if support.contact_physical_layer_id not in layers:
        raise ValueError(
            f"support {support.support_id!r} references an undeclared contact physical layer"
        )
    if not set(
        (*support.terminal_a_entity_ids, *support.terminal_b_entity_ids)
    ).issubset(available):
        raise ValueError(
            f"support {support.support_id!r} has missing conductor ownership"
        )
    contacts = tuple(support.world_point(point) for point in support.contact_points_um)
    if support.source_z_um is None:
        raise ValueError("neutral support requires its original locator plane")
    if any(point[2] != support.source_z_um for point in contacts):
        raise ValueError(
            "horizontal lumped support contacts differ from its original locator plane"
        )
    layer = layers[support.contact_physical_layer_id]
    low, high = (
        layer.source_interval_um if current else (layer.z_min_um, layer.z_max_um)
    )
    if not low <= contacts[0][2] <= high:
        raise ValueError(
            "lumped support source contact plane is outside its declared physical layer"
        )
    return contacts


def _project_point(mapping, point, contact_layer_id: str, *, current: bool):
    """Map one source-space point to effective space with the model's own map."""
    if current:
        return mapping.project_support_point(point, contact_layer_id)
    return (point[0], point[1], mapping.map_z(point[2]))
