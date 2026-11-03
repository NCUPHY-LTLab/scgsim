"""Stage S2: the effective input as a prism catalogue, from real ``GeometryBuildInput`` objects."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import gdstk
import pytest
from sgb_inputs import (
    EffectiveInput,
    gds_pad_in_ground,
    geometry_plan_electrodes,
    thin_film_pair,
)

from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    SourceCurveSpec,
)
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.derivation.labelled import (
    derive_from_catalogue,
    derive_labelled_arrangements,
)
from scgsim.geometry.sgb.errors import (
    ContactError,
    InputError,
    OverlapError,
    UnsupportedCurveError,
)
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue

UM = 1000  # lattice units per micrometre on the 1 nm grid


def by_id(catalogue, entity_id):
    """The prisms of one entity."""
    return [p for p in catalogue.prisms if p.entity_id == entity_id]


def prism_of(catalogue, entity_id):
    """The only prism of one entity."""
    (prism,) = by_id(catalogue, entity_id)
    return prism


@pytest.fixture
def pad(tmp_path):
    """The GDS pad-in-ground input with its finite 0.2 um metal."""
    return gds_pad_in_ground(tmp_path)


@pytest.fixture
def thin_film():
    """Two Route-A sheets joined by a port."""
    return thin_film_pair(0.5)


# --------------------------------------------------------------------------- the grid


def test_the_grid_is_read_from_the_declared_context(pad):
    """The grid is read from the declared context."""
    catalogue = build_prism_catalogue(pad.build_input, pad.representation)
    assert catalogue.grid.size_um == 0.001


def test_an_input_without_a_grid_context_is_rejected_not_defaulted(pad):
    """An input without a grid context is rejected not defaulted."""
    metadata = {
        k: v for k, v in pad.build_input.metadata.items() if k != "geometry_grid"
    }
    bare = replace(pad.build_input, metadata=metadata)
    with pytest.raises(InputError, match="declares no geometry grid"):
        build_prism_catalogue(bare, pad.representation)


def test_a_vertex_off_the_declared_grid_fails_naming_the_entity(pad):
    """A vertex off the declared grid fails naming the entity."""
    entities = tuple(
        replace(
            e,
            geometry={
                **e.geometry,
                "outer_loop": ((40.0004, 40.0), (60.0, 40.0), (60.0, 60.0)),
            },
            polygon_ids=(),
        )
        if e.semantic_id == "pad"
        else e
        for e in pad.build_input.entities
    )
    with pytest.raises(
        InputError, match=r"pad polygon 0 outer ring.*off the 0.001 um model grid"
    ):
        build_prism_catalogue(
            replace(pad.build_input, entities=entities), pad.representation
        )


# --------------------------------------------------------------------------- prisms from entities


def test_every_entity_becomes_prisms_with_exact_z_and_declared_nets(pad):
    """Every entity becomes prisms with exact z and declared nets."""
    catalogue = build_prism_catalogue(pad.build_input, pad.representation)
    ground = prism_of(catalogue, "ground_0000")
    assert (ground.z0, ground.z1) == (Fraction(0), Fraction(1, 5))
    assert ground.material_kind == "conductor" and ground.net_id == "G"
    assert len(ground.footprint.holes) == 1  # the cutout
    assert ground.footprint.twice_area() == 2 * 8400 * UM * UM
    pad_prism = prism_of(catalogue, "pad")
    assert (
        pad_prism.net_id == "P"
        and pad_prism.footprint.twice_area() == 2 * 400 * UM * UM
    )
    substrate = prism_of(catalogue, "substrate")
    assert (substrate.z0, substrate.z1) == (Fraction(-10), Fraction(0))
    assert substrate.material_kind == "dielectric" and substrate.net_id is None


def test_the_automatic_vacuum_envelope_comes_from_the_prepared_enclosure(pad):
    """The automatic vacuum envelope comes from the prepared enclosure."""
    catalogue = build_prism_catalogue(pad.build_input, pad.representation)
    vacuum = prism_of(catalogue, "VACUUM_REGION")
    assert vacuum.material_kind == "vacuum"
    assert (vacuum.z0, vacuum.z1) == (Fraction(-30), Fraction(101, 5))
    assert vacuum.footprint.outer == (
        (-20 * UM, -20 * UM),
        (120 * UM, -20 * UM),
        (120 * UM, 120 * UM),
        (-20 * UM, 120 * UM),
    )


def test_a_port_sheet_is_a_sheet_prism_at_the_host_mid_plane_with_its_landings_removed(
    pad,
):
    """A port sheet is a sheet prism at the host mid plane with its landings removed."""
    catalogue = build_prism_catalogue(pad.build_input, pad.representation)
    port = prism_of(catalogue, "PORT_SHEET__P1__0000")
    assert port.is_sheet and port.z0 == Fraction(
        1, 10
    )  # middle of the finite 0..0.2 um hosts
    assert port.material_kind == "port"
    # 12 x 4 um region minus one 1 x 4 um landing on each host
    assert port.footprint.twice_area() == 2 * 40 * UM * UM
    assert len(port.footprint.holes) == 2
    identity = port.port
    assert identity is not None
    assert identity.source_name == "P1" and identity.port_index == 1
    assert identity.direction == (1.0, 0.0, 0.0)
    assert identity.host_entity_ids == ("ground_0000", "pad")
    assert identity.target_layer == "metal" and identity.source_layer == "9/0"


def test_thin_film_conductors_are_sheets_and_the_port_lies_on_their_plane(thin_film):
    """Thin film conductors are sheets and the port lies on their plane."""
    catalogue = build_prism_catalogue(thin_film.build_input, thin_film.representation)
    for name in ("chip/left", "chip/right"):
        sheet = prism_of(catalogue, name)
        assert sheet.is_sheet and sheet.z0 == 0
    port = prism_of(catalogue, "port_sheet")
    assert port.z0 == port.z1 == 0
    assert (
        port.footprint.twice_area() == 2 * 4 * UM * UM
    )  # 6 um^2 region minus two 1 um^2 landings
    upper = prism_of(catalogue, "upper_air")
    assert (upper.z0, upper.z1) == (
        Fraction(0),
        Fraction(4),
    )  # 4.25 um minus the collapsed film


def test_a_conductor_without_a_net_is_its_own_net_so_touching_ones_fail_closed(
    tmp_path,
):
    """A conductor without a net is its own net so touching ones fail closed."""
    floating = gds_pad_in_ground(tmp_path, nets=False, with_port=False)
    catalogue = build_prism_catalogue(floating.build_input, floating.representation)
    assert prism_of(catalogue, "pad").net_id == "pad"
    assert prism_of(catalogue, "ground_0000").net_id == "ground_0000"
    # a pad that reaches into the ground plane overlaps a different "Net"
    overlapping = gds_pad_in_ground(
        tmp_path,
        nets=False,
        with_port=False,
        pad_polygon=gdstk.rectangle((25, 40), (60, 60)),
    )
    with pytest.raises(OverlapError, match=r"pad.*ground_0000|ground_0000.*pad"):
        derive_labelled_arrangements(
            overlapping.build_input, overlapping.representation
        )


def test_a_conductor_missing_from_the_representation_context_fails_closed(pad):
    """A conductor missing from the representation context fails closed."""
    context = RepresentationContext(
        {k: v for k, v in pad.representation.entities.items() if k != "pad"},
        dict(pad.representation.enclosures),
    )
    with pytest.raises(
        InputError, match="pad is a conductor missing from the representation context"
    ):
        build_prism_catalogue(pad.build_input, context)


def test_a_solid_conductor_with_zero_thickness_fails_closed(pad):
    """A solid conductor with zero thickness fails closed."""
    records = dict(pad.representation.entities)
    records["pad"] = replace_interval(records["pad"], (0.0, 0.0))
    context = RepresentationContext(records, dict(pad.representation.enclosures))
    with pytest.raises(InputError, match=r"pad has a non-positive Z extent \[0, 0\]"):
        build_prism_catalogue(pad.build_input, context)


def test_a_sheet_whose_interval_is_not_flat_fails_closed(thin_film):
    """A sheet whose interval is not flat fails closed."""
    records = dict(thin_film.representation.entities)
    sheet = records["chip/left"]
    records["chip/left"] = type(sheet)(
        sheet.physical_layer_id, "sheet", sheet.source_z_interval_um, (0.0, 0.25)
    )
    context = RepresentationContext(records, dict(thin_film.representation.enclosures))
    with pytest.raises(
        InputError, match="chip/left is a sheet but its effective Z interval"
    ):
        build_prism_catalogue(thin_film.build_input, context)


def test_a_slab_thinner_than_the_grid_survives_from_the_source_to_the_arrangements(
    tmp_path,
):
    """A slab thinner than the grid survives from the source to the arrangements."""
    thin = gds_pad_in_ground(tmp_path, thickness_um=0.0004, with_port=False)
    result = derive_labelled_arrangements(thin.build_input, thin.representation)
    assert (
        Fraction(1, 2500) in result.levels.levels
    )  # 0.4 nm < the 1 nm grid, kept as declared


def test_port_hosts_with_different_intervals_fail_closed(thin_film):
    """Port hosts with different intervals fail closed."""
    entities = tuple(
        replace(e, semantic_id=e.semantic_id) for e in thin_film.build_input.entities
    )
    entity_records = dict(thin_film.representation.entities)
    right = entity_records["chip/right"]
    entity_records["chip/right"] = replace_interval(right, (0.0, 0.25))
    context = RepresentationContext(
        entity_records, dict(thin_film.representation.enclosures)
    )
    with pytest.raises(InputError, match="do not share one effective Z interval"):
        build_prism_catalogue(
            replace(thin_film.build_input, entities=entities), context
        )


def replace_interval(record, interval):
    """An ``EntityRepresentation`` with another effective interval (and as a solid)."""
    return type(record)(
        record.physical_layer_id, "solid", record.source_z_interval_um, interval
    )


def test_a_port_without_any_host_overlap_fails_closed(thin_film):
    """A port without any host overlap fails closed."""
    port = replace(thin_film.build_input.port_sheet_regions[0], overlaps=())
    bare = replace(thin_film.build_input, port_sheet_regions=(port,))
    with pytest.raises(InputError, match="no host overlap"):
        build_prism_catalogue(bare, thin_film.representation)


def test_a_port_covered_entirely_by_its_hosts_has_no_active_remainder(thin_film):
    """A port covered entirely by its hosts has no active remainder."""
    region = thin_film.build_input.port_sheet_regions[0]
    covering = replace(
        region.overlaps[0], overlap_loop=region.exterior, overlap_id="everything"
    )
    port = replace(region, overlaps=(covering, region.overlaps[1]))
    bare = replace(thin_film.build_input, port_sheet_regions=(port,))
    with pytest.raises(InputError, match="no active remainder"):
        build_prism_catalogue(bare, thin_film.representation)


def test_the_catalogue_is_deterministic_and_prism_order_follows_the_input(pad):
    """The catalogue is deterministic and prism order follows the input."""
    first = build_prism_catalogue(pad.build_input, pad.representation)
    second = build_prism_catalogue(pad.build_input, pad.representation)
    assert first == second
    assert [p.entity_id for p in first.prisms] == [
        e.semantic_id for e in pad.build_input.entities
    ] + ["PORT_SHEET__P1__0000"]


def test_lineage_names_the_entity_polygon_ring_and_edge_of_every_carrier(pad):
    """Lineage names the entity polygon ring and edge of every carrier."""
    catalogue = build_prism_catalogue(pad.build_input, pad.representation)
    ground = prism_of(catalogue, "ground_0000")
    carriers = ground.carriers()
    assert {c.lineage.ring for c in carriers} == {0, 1}
    assert all(
        c.lineage.entity_id == "ground_0000" and c.lineage.polygon == 0
        for c in carriers
    )
    assert carriers[0].id == carriers[0].lineage.describe()


# --------------------------------------------------------------------------- derivation end to end


def test_the_vacuum_envelope_yields_to_the_substrate_and_ports_label_their_level(pad):
    """The vacuum envelope yields to the substrate and ports label their level."""
    result = derive_labelled_arrangements(pad.build_input, pad.representation)
    substrate_slab = next(
        s for s in result.slabs if s.z == (Fraction(-10), Fraction(0))
    )
    domains = {(d.kind, d.owner) for d in substrate_slab.labels if d is not None}
    assert domains == {("dielectric", "substrate"), ("vacuum", "VACUUM_REGION")}
    port_level = next(a for a in result.level_layers if a.z == Fraction(1, 10))
    owners = {f.sheet.name for f in port_level.labels[1:] if f.sheet is not None}
    assert owners == {"PORT_SHEET__P1__0000"}


def test_thin_film_sheets_label_their_level_by_net_and_port(thin_film):
    """Thin film sheets label their level by net and port."""
    result = derive_labelled_arrangements(
        thin_film.build_input, thin_film.representation
    )
    level = next(a for a in result.level_layers if a.z == 0)
    owners = {
        (f.sheet.kind, f.sheet.name) for f in level.labels[1:] if f.sheet is not None
    }
    assert owners == {
        ("conductor", "left"),
        ("conductor", "right"),
        ("port", "port_sheet"),
    }
    # sheets of different owners share edges at the landings, which is legal for a port
    assert len(result.slabs) == 2


def test_real_inputs_have_the_same_identity_for_every_tile_and_worker_count(
    pad, thin_film
):
    """Real inputs have the same identity for every tile and worker count."""
    for effective in (pad, thin_film):
        reference = derive_labelled_arrangements(
            effective.build_input, effective.representation
        )
        tiled = derive_labelled_arrangements(
            effective.build_input, effective.representation, tiles=3, workers=2
        )
        assert tiled.identity_sets() == reference.identity_sets()
        assert tiled.identity_digest() == reference.identity_digest()


# --------------------------------------------------------------------------- boundary curves


def circle_chain(centre, radius, entity="pad", boundary="pad_circle"):
    """An authored closed chain of four quarter arcs (three points each)."""
    cx, cy = centre
    east, north, west, south = (
        (cx + radius, cy),
        (cx, cy + radius),
        (cx - radius, cy),
        (cx, cy - radius),
    )
    s = math.sqrt(0.5) * radius
    curves = tuple(
        SourceCurveSpec("circular_arc", (a, mid, b))
        for a, mid, b in (
            (east, (cx + s, cy + s), north),
            (north, (cx - s, cy + s), west),
            (west, (cx - s, cy - s), south),
            (south, (cx + s, cy - s), east),
        )
    )
    return BoundaryCurveChainSpec(boundary, entity, centre, curves)


def with_boundaries(effective: EffectiveInput, **records) -> EffectiveInput:
    """The same effective input with boundary records attached."""
    return EffectiveInput(
        replace(effective.build_input, **records), effective.representation
    )


def test_an_authored_circle_replaces_the_pad_ring_with_exact_arcs(tmp_path):
    """An authored circle replaces the pad ring with exact arcs."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    curved = with_boundaries(pad, boundary_curves=(circle_chain((50.0, 50.0), 10.0),))
    catalogue = build_prism_catalogue(curved.build_input, curved.representation)
    prism = prism_of(catalogue, "pad")
    assert len(prism.arcs) == 4 and len(prism.footprint.arc_runs) == 4
    assert {arc.lineage.edge for arc in prism.arcs} == {
        0,
        1,
        2,
        3,
    }  # the curve index in the chain
    assert all(
        arc.lineage.entity_id == "pad" and arc.lineage.ring == 0 for arc in prism.arcs
    )
    result = derive_from_catalogue(catalogue)
    slab = next(s for s in result.slabs if s.z == (Fraction(0), Fraction(1, 5)))
    (face,) = [i for i, d in enumerate(slab.labels) if d is not None and d.net == "P"]
    area_um2 = face_exact_area(slab.arr, slab.edge_curve, face) * 1e-6
    assert area_um2 == pytest.approx(math.pi * 100, rel=1e-9)


def test_a_closed_circular_reconstruction_turns_the_polygon_into_a_circle(tmp_path):
    """A closed circular reconstruction turns the polygon into a circle."""
    ellipse = gdstk.ellipse((50, 50), 10, tolerance=0.01)
    pad = gds_pad_in_ground(tmp_path, with_port=False, pad_polygon=ellipse)
    (polygon,) = [p for p in pad.build_input.polygons if p.object_name == "pad"]
    count = len(polygon.exterior)
    record = GdsBoundaryReconstructionSpec(
        "pad_circle",
        "pad",
        (50.0, 50.0),
        "circular_arc",
        closed=True,
        segment_indices=tuple(range(count)),
        anchor_indices=(0, count // 3, 2 * count // 3),
    )
    curved = with_boundaries(pad, boundary_reconstruction=(record,))
    catalogue = build_prism_catalogue(curved.build_input, curved.representation)
    prism = prism_of(catalogue, "pad")
    assert len(prism.arcs) == 4  # a full circle is four quarter arcs
    result = derive_from_catalogue(catalogue)
    slab = next(s for s in result.slabs if s.z == (Fraction(0), Fraction(1, 5)))
    (face,) = [i for i, d in enumerate(slab.labels) if d is not None and d.net == "P"]
    area_um2 = face_exact_area(slab.arr, slab.edge_curve, face) * 1e-6
    assert area_um2 == pytest.approx(math.pi * 100, rel=2e-3)


def test_an_open_reconstruction_keeps_the_unselected_straight_segments(tmp_path):
    """An open reconstruction keeps the unselected straight segments."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    (polygon,) = [p for p in pad.build_input.polygons if p.object_name == "pad"]
    ring = polygon.exterior
    # replace the first two sides of the square by an arc through the middle corner
    record = GdsBoundaryReconstructionSpec(
        "pad_corner",
        "pad",
        (50.0, 50.0),
        "circular_arc",
        segment_indices=(0, 1),
        anchor_indices=(0, 1, 2),
    )
    curved = with_boundaries(pad, boundary_reconstruction=(record,))
    catalogue = build_prism_catalogue(curved.build_input, curved.representation)
    prism = prism_of(catalogue, "pad")
    assert {arc.lineage.edge for arc in prism.arcs} == {
        0
    }  # numbered by the first claimed segment
    kept = [c for c in prism.carriers() if "#" not in c.id]
    assert sorted(c.lineage.edge for c in kept) == [2, 3]  # the other two source sides
    assert len(ring) == 4


def test_a_spline_boundary_fails_closed_naming_the_boundary_and_the_entity(tmp_path):
    """A spline boundary fails closed naming the boundary and the entity."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    spline = BoundaryCurveChainSpec(
        "pad_spline",
        "pad",
        (50.0, 50.0),
        (
            SourceCurveSpec(
                "bspline", ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0)), degree=2
            ),
        ),
    )
    curved = with_boundaries(pad, boundary_curves=(spline,))
    with pytest.raises(UnsupportedCurveError, match=r"pad_spline.*'pad'.*'bspline'"):
        build_prism_catalogue(curved.build_input, curved.representation)


def test_a_spline_reconstruction_fails_closed(tmp_path):
    """A spline reconstruction fails closed."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    record = GdsBoundaryReconstructionSpec(
        "pad_fit",
        "pad",
        (50.0, 50.0),
        "bspline",
        closed=True,
        segment_indices=(0, 1, 2, 3),
        degree=3,
    )
    curved = with_boundaries(pad, boundary_reconstruction=(record,))
    with pytest.raises(UnsupportedCurveError, match=r"pad_fit.*'bspline'"):
        build_prism_catalogue(curved.build_input, curved.representation)


def test_a_boundary_selector_that_binds_no_polygon_fails_closed(tmp_path):
    """A boundary selector that binds no polygon fails closed."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    chain = replace(circle_chain((50.0, 50.0), 10.0), selector_point_um=(5.0, 5.0))
    curved = with_boundaries(pad, boundary_curves=(chain,))
    with pytest.raises(
        InputError, match="selector must bind one Entity-owned logical polygon"
    ):
        build_prism_catalogue(curved.build_input, curved.representation)


def test_an_authored_chain_that_is_not_closed_fails_closed(tmp_path):
    """An authored chain that is not closed fails closed."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    chain = circle_chain((50.0, 50.0), 10.0)
    open_chain = replace(chain, curves=chain.curves[:3])
    curved = with_boundaries(pad, boundary_curves=(open_chain,))
    with pytest.raises(InputError, match="do not form one closed chain"):
        build_prism_catalogue(curved.build_input, curved.representation)


def test_two_records_claiming_one_ring_fail_closed(tmp_path):
    """Two records claiming one ring fail closed."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    curved = with_boundaries(
        pad,
        boundary_curves=(
            circle_chain((50.0, 50.0), 10.0),
            circle_chain((50.0, 50.0), 10.0, boundary="again"),
        ),
    )
    with pytest.raises(InputError, match="multiple complete authored boundaries claim"):
        build_prism_catalogue(curved.build_input, curved.representation)


def test_a_collinear_arc_defines_no_circle(tmp_path):
    """A collinear arc defines no circle."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    line = BoundaryCurveChainSpec(
        "flat",
        "pad",
        (50.0, 50.0),
        (SourceCurveSpec("circular_arc", ((40.0, 40.0), (50.0, 40.0), (60.0, 40.0))),),
    )
    curved = with_boundaries(pad, boundary_curves=(line,))
    with pytest.raises(InputError, match="collinear"):
        build_prism_catalogue(curved.build_input, curved.representation)


# --------------------------------------------------------------------------- a real GeometryPlan


def test_a_real_geometry_plan_snapshot_becomes_prisms_with_its_nets_and_z(monkeypatch):
    """GeometryPlan.prepare() -> interpret_physical_geometry -> prisms, end to end."""
    pytest.importorskip("gdsfactory", reason="GeometryPlan needs the layout stack")
    effective = geometry_plan_electrodes(monkeypatch)
    catalogue = build_prism_catalogue(effective.build_input, effective.representation)
    pad, electrode = (
        prism_of(catalogue, f"chip/{name}_0000") for name in ("pad", "electrode")
    )
    assert (pad.net_id, electrode.net_id) == ("A", "B")
    assert (pad.z0, pad.z1) == (Fraction(0), Fraction(1, 5))
    assert pad.footprint.twice_area() == 2 * 400 * UM * UM
    assert electrode.footprint.twice_area() == 2 * 600 * UM * UM
    result = derive_from_catalogue(catalogue)
    substrate_slab = next(s for s in result.slabs if s.z == (Fraction(-5), Fraction(0)))
    owners = {d.owner for d in substrate_slab.labels if d is not None}
    # the vacuum envelope covers the substrate's footprint, and the dielectric wins there
    assert owners == {"substrate"}


def test_two_planned_electrodes_that_touch_fail_closed_naming_both_entities(
    monkeypatch,
):
    """A zero gap between different Nets is a contact error with both qualified Entity ids."""
    pytest.importorskip("gdsfactory", reason="GeometryPlan needs the layout stack")
    touching = geometry_plan_electrodes(monkeypatch, gap_um=0.0)
    with pytest.raises(ContactError, match=r"chip/pad_0000.*chip/electrode_0000"):
        derive_labelled_arrangements(touching.build_input, touching.representation)
