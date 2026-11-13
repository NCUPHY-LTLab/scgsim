"""Lumped-port sheet regions as sheet prisms (stage S2).

Owns turning each adapter-authored ``PortSheetRegionRecord`` into a sheet prism that
carries the port identity:

* **Plane.**  The port sheet lies on the *physical support plane* of its host conductors:
  the plane the source authored for them, projected through the common
  ``PhysicalRepresentationPlan`` into the effective model.  A film host declares it as the
  ``layer_attachment_face`` (lower or upper) of its ``PhysicalLayerDeclaration``; a host
  that is a genuine zero-thickness source sheet is its own plane.  The authored planes of
  all hosts must agree, every finite body interval must still meet the plane, and every
  host must project to one effective plane.  A host without a declaration or an
  attachment face fails closed: nothing is guessed from the body's interval, so a 0.2 um
  metal on a substrate puts its port on the substrate surface, never in the vacuum above
  it.  A port
  therefore lies in the plane of the surface under it and occupies part of that surface
  instead of floating over it.  One rule for every route, because the prism catalogue
  does not know the route.
* **Footprint.**  The authored region minus its authored holes, whole.  The part of it that
  lands on a host conductor is not removed here: the adapter's ``overlap_loop`` follows the
  host's raw GDS polyline, while SGB may have reconstructed the host as an exact arc, and the
  two would meet as one edge with two carriers.  Labelling instead reads the landing from the
  host footprint SGB built: a port-sheet cell over a host is the host
  (``arrangement.labelling.prism_set.without_landings``), so the port keeps only its *active remainder*
  and shares the host's own edges.  A region bound to a rectangular junction partition is
  already the junction's central rectangle and keeps its full footprint, so it has no landing.
* **Identity.**  Port sheet id, source name, 1-based port index, normalised XY
  direction, host entity ids and source/target layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import hypot, isfinite

from scgsim.geometry.models.grid import format_box
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.physical import (
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.exact.footprints import lattice_ring
from scgsim.geometry.sgb.exact.grid import Grid, exact_z
from scgsim.geometry.sgb.exact.prisms import Footprint, PortIdentity, PrismRecord

PORT_MATERIAL = "lumped_port"
JUNCTION_PARTITION_KEY = "rectangular_junction_partition"
FILM_KINDS = ("stack_film", "local_film")


def port_prisms(
    build_input: GeometryBuildInput,
    grid: Grid,
    representation: RepresentationContext,
) -> list[PrismRecord]:
    """One sheet prism per port sheet region, in source order.

    ``representation`` carries each conductor's source and effective Z facts; the common
    physical plan in the input's metadata projects the authored support planes.
    """
    regions = build_input.port_sheet_regions
    if not regions:
        return []
    plan = _physical_plan(build_input)
    return [_port_prism(region, grid, representation, plan) for region in regions]


def _physical_plan(build_input: GeometryBuildInput) -> PhysicalRepresentationPlan:
    raw = build_input.metadata.get("physical_representation")
    if raw is None:
        raise InputError(
            "a lumped port needs the common physical representation plan "
            "(metadata 'physical_representation') to project its support plane; "
            "interpret the physical geometry first"
        )
    try:
        return PhysicalRepresentationPlan.from_payload(raw)
    except (TypeError, ValueError, KeyError) as error:
        raise InputError(
            f"the physical representation plan is invalid: {error}"
        ) from error


def _port_prism(
    region: PortSheetRegionRecord,
    grid: Grid,
    representation: RepresentationContext,
    plan: PhysicalRepresentationPlan,
) -> PrismRecord:
    what = f"port sheet {region.port_sheet_id!r}"
    hosts = tuple(dict.fromkeys(o.host_semantic_id for o in region.overlaps))
    z = _support_plane(what, _place(region, grid), hosts, representation, plan)
    exterior = lattice_ring(region.exterior, grid, f"{what} exterior")
    holes = [
        lattice_ring(hole, grid, f"{what} hole {k}")
        for k, hole in enumerate(region.holes)
    ]
    footprint = Footprint.from_rings(exterior, holes)
    if footprint.twice_area() <= 0:
        raise InputError(f"{what} has an empty footprint")
    return PrismRecord(
        entity_id=region.port_sheet_id,
        footprint=footprint,
        z0=z,
        z1=z,
        material=PORT_MATERIAL,
        material_kind="port",
        port=PortIdentity(
            port_sheet_id=region.port_sheet_id,
            source_name=str(region.metadata.get("source_name", region.port_sheet_id)),
            port_index=int(region.metadata.get("port_index", 0)),
            direction=port_direction(region.metadata.get("direction"), what),
            host_entity_ids=hosts,
            target_layer=str(region.metadata.get("target_layer", "")),
            source_layer=region.source_layer,
            lands_on_hosts=JUNCTION_PARTITION_KEY not in region.metadata,
        ),
    )


def _place(region: PortSheetRegionRecord, grid: Grid) -> str:
    """Where a port sheet is drawn: its source layer and its bounding box in micrometres."""
    xs = [p[0] for p in region.exterior]
    ys = [p[1] for p in region.exterior]
    box = format_box((min(xs), min(ys), max(xs), max(ys)), grid.size_um)
    return f"on layer {region.source_layer} at {box}"


def _support_plane(
    what: str,
    place: str,
    hosts: Sequence[str],
    representation: RepresentationContext,
    plan: PhysicalRepresentationPlan,
) -> Fraction:
    """The effective plane every host of the port agrees on (fail closed otherwise).

    ``place`` says where the port sheet is drawn, for the refusals a layout author acts on.
    """
    if not hosts:
        raise InputError(
            f"{what} has no host overlap, so its plane is undefined: the sheet {place} "
            "overlaps no conductor of its target layer. Fix the layout or the port "
            "declaration: draw the sheet onto the conductors it excites, or name their "
            "layer as its target layer."
        )
    unknown = [h for h in hosts if h not in representation.entities]
    if unknown:
        raise InputError(
            f"{what} references host(s) that are not conductor entities: {unknown}"
        )
    sources = {
        host: _authored_plane(what, host, representation.entities[host], plan)
        for host in hosts
    }
    if len({z for z, _ in sources.values()}) != 1:
        raise InputError(
            f"{what} hosts do not share one authored support plane (source Z, um): "
            f"{ {host: z for host, (z, _) in sources.items()} }; the sheet is drawn "
            f"{place}. Fix the declarations or the layout: give the host layers one "
            "layer_attachment_face plane, or land the sheet only on hosts of one plane."
        )
    source_z = next(iter(sources.values()))[0]
    effective = {
        host: exact_z(_project(plan, source_z, layer_id))
        for host, (_, layer_id) in sources.items()
    }
    if len(set(effective.values())) != 1:
        raise InputError(
            f"{what} hosts project to different effective support planes: "
            f"{ {host: str(z) for host, z in effective.items()} }; the sheet is drawn "
            f"{place}. Fix the declarations: the physical representation must project "
            "the hosts' shared authored plane to one effective plane."
        )
    z = next(iter(effective.values()))
    for host in hosts:
        low, high = (
            exact_z(v) for v in representation.entities[host].effective_z_interval_um
        )
        if not low <= z <= high:
            raise InputError(
                f"{what} host {host!r} does not meet the effective support plane "
                f"z = {z} um (its effective interval is [{low}, {high}] um)"
            )
    return z


def _authored_plane(
    what: str,
    host: str,
    record: EntityRepresentation,
    plan: PhysicalRepresentationPlan,
) -> tuple[float, str | None]:
    """The source Z a host authored for its support plane, and the layer that owns it.

    A film's plane is its declared attachment face; a zero-thickness source interval is its
    own plane.  A finite body interval may confirm the plane but never invents one.
    """
    layer_id = record.physical_layer_id
    declaration = plan.declaration_index.get(layer_id) if layer_id is not None else None
    low, high = record.source_z_interval_um
    if declaration is not None and declaration.feature_kind in FILM_KINDS:
        z = _attachment_z(what, host, declaration)
        if not low <= z <= high:
            raise InputError(
                f"{what} host {host!r}: the authored attachment plane z = {z} um of "
                f"{layer_id!r} lies outside its source interval [{low}, {high}] um"
            )
        return z, layer_id
    if low == high:
        return low, layer_id
    reason = (
        "has no physical layer declaration"
        if declaration is None
        else f"is a {declaration.feature_kind} body, not a film"
    )
    raise InputError(
        f"{what} host {host!r} {reason}, so it has no authored support plane; the PDK "
        "must declare layer_attachment_face (lower or upper) in the host layer's "
        "physical_layer source_contact_facts"
    )


def _attachment_z(what: str, host: str, declaration: PhysicalLayerDeclaration) -> float:
    try:
        return declaration.attachment_z_um
    except ValueError as error:
        raise InputError(
            f"{what} host {host!r}: physical layer {declaration.physical_layer_id!r} "
            "has no attachment face; the PDK must declare layer_attachment_face "
            "(lower or upper) in its source_contact_facts"
        ) from error


def _project(
    plan: PhysicalRepresentationPlan, source_z: float, layer_id: str | None
) -> float:
    """The effective Z of an authored plane, through the common plan."""
    point = (0.0, 0.0, source_z)
    try:
        if layer_id is None:
            return plan.map_point(point)[2]
        return plan.project_support_point(point, layer_id)[2]
    except (KeyError, ValueError) as error:
        raise InputError(f"cannot project the support plane: {error}") from error


def port_direction(raw: object, what: str) -> tuple[float, float, float]:
    """The authored XY direction, normalised; a defect fails closed."""
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise InputError(f"{what} requires a 3D direction")
    try:
        values = tuple(float(v) for v in raw)
    except (TypeError, ValueError) as error:
        raise InputError(f"{what} direction must be numeric") from error
    if (
        not all(isfinite(v) for v in values)
        or values[2] != 0.0
        or values[0] == values[1] == 0.0
    ):
        raise InputError(f"{what} direction must be finite, XY, and nonzero")
    length = hypot(values[0], values[1])
    return (values[0] / length, values[1] / length, 0.0)
