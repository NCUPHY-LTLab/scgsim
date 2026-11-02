"""Pure canonical source-to-effective support projection for preparation and decoding."""

from typing import Any, Mapping, Sequence, cast

from scgsim.aedt.specs.modeling import PhysicalLayerSpec, ZMap
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.physical import PLAN_SCHEMA, PhysicalRepresentationPlan


def prepared_lumped_supports(
    supports: Sequence[LumpedSupport], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive explicit support XY/Z from the same physical map as all bodies."""

    trace = source["modeling"]
    current = trace.get("schema_version") == PLAN_SCHEMA
    mapping, layers = _source_mapping(trace, current=current)
    available = _available_conductors(source["conductors"])
    records = [
        _project_support(support, mapping, layers, available, current=current)
        for support in supports
    ]
    if len({item["support_id"] for item in records}) != len(records):
        raise ValueError("neutral support IDs must be unique")
    return records


def _source_mapping(
    trace: Mapping[str, Any], *, current: bool
) -> tuple[PhysicalRepresentationPlan | ZMap, Mapping[str, Any]]:
    if current:
        mapping = PhysicalRepresentationPlan.from_payload(trace)
        return mapping, mapping.declaration_index
    mapping = ZMap(
        trace["modeling"],
        tuple(PhysicalLayerSpec(**item) for item in trace["source_layers"]),
    )
    return mapping, mapping.layers


def _available_conductors(conductors: Sequence[Mapping[str, Any]]) -> set[str]:
    available = {item["semantic_id"] for item in conductors}
    available.update(item["source_semantic_id"] for item in conductors)
    return available


def _project_support(
    support: LumpedSupport,
    mapping: PhysicalRepresentationPlan | ZMap,
    layers: Mapping[str, Any],
    available: set[str],
    *,
    current: bool,
) -> dict[str, Any]:
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
    source_z_um = support.source_z_um
    if any(point[2] != source_z_um for point in contacts):
        raise ValueError(
            "horizontal lumped support contacts differ from their original locator plane"
        )
    if current:
        plan = cast(PhysicalRepresentationPlan, mapping)
        layer = plan.declaration_index[support.contact_physical_layer_id]
        low, high = layer.source_interval_um
    else:
        z_map = cast(ZMap, mapping)
        layer = z_map.layers[support.contact_physical_layer_id]
        low, high = layer.z_min_um, layer.z_max_um
    if not low <= contacts[0][2] <= high:
        raise ValueError(
            "lumped support source contact plane is outside its declared physical layer"
        )
    a, b, c, d, _, _ = support.source_transform
    nx, ny, nz = support.normal
    normal = (a * nx + b * ny, c * nx + d * ny, nz)

    def project(point):
        if current:
            plan = cast(PhysicalRepresentationPlan, mapping)
            return plan.project_support_point(point, support.contact_physical_layer_id)
        z_map = cast(ZMap, mapping)
        return (point[0], point[1], z_map.map_z(point[2]))

    def effective_xy(point):
        # Rings stay XY; the common XYZ map receives the authored locator plane.
        world = support.world_point((*point, source_z_um))
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
            "z_um": project((0.0, 0.0, source_z_um))[2],
            "contact_points_um": [list(project(point)) for point in contacts],
            "contact_physical_layer_id": support.contact_physical_layer_id,
        },
    }
