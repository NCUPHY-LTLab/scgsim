"""Equal-priority overlaps say which source must supply the priority (lower gdsfactory ``mesh_order`` wins)."""

from __future__ import annotations

import pytest
from sgb_scenes import (
    GRID,
    chip_prisms,
    footprint,  # noqa: E402,I001 -- grouped with the scene helpers above
    rect_um,
)

from scgsim.geometry.sgb.errors import OverlapError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import Footprint, PrismCatalogue, PrismRecord


def sourced(  # noqa: PLR0913 - a test builder
    name, box, *, layer, occurrence, priority=0, kind="conductor", material="Al"
):
    """A prism that knows its source layer and occurrence (what S2 records for a real entity)."""
    return PrismRecord(
        name,
        footprint(rect_um(*box)),
        exact_z(0),
        exact_z("0.2"),
        material,
        kind,
        net_id="N" if kind == "conductor" else None,
        priority=priority,
        layer=layer,
        occurrence=occurrence,
    )


def overlap_error(first, second, route="B"):
    """The OverlapError message of two overlapping prisms."""
    catalogue = PrismCatalogue(GRID, (*chip_prisms(GRID, 40.0), first, second))
    with pytest.raises(OverlapError) as error:
        build_model_from_catalogue(catalogue, route)
    return str(error.value)


def test_entities_on_different_layers_are_ordered_by_the_layer_stack_mesh_order():
    """Cross-layer overlap: the LayerStack decides."""
    message = overlap_error(
        sourced("a", (0, 0, 10, 10), layer="M1", occurrence="chip"),
        sourced("b", (5, 5, 15, 15), layer="M2", occurrence="chip"),
    )
    assert "a and b" in message and "(M1, M2)" in message
    assert "LayerStack `mesh_order` must differ (lower mesh_order wins)" in message


def test_entities_of_one_layer_in_one_component_are_ordered_by_the_component_annotation():
    """Same layer, same component: the PDK-filled annotation decides."""
    message = overlap_error(
        sourced("a", (0, 0, 10, 10), layer="M1", occurrence="chip/q0"),
        sourced("b", (5, 5, 15, 15), layer="M1", occurrence="chip/q0"),
    )
    assert "share layer M1 in one component" in message
    assert "component annotation `priority` must differ" in message


def test_entities_of_one_layer_from_different_occurrences_need_an_assembly_priority():
    """Same layer, different occurrences: only the assembly can order them."""
    message = overlap_error(
        sourced("a", (0, 0, 10, 10), layer="M1", occurrence="chip/q0"),
        sourced("b", (5, 5, 15, 15), layer="M1", occurrence="chip/q1"),
    )
    assert "come from different occurrences (chip/q0, chip/q1)" in message
    assert (
        "GeometryPlan assembly must give the overlap an assembly-level priority"
        in message
    )


def test_a_layer_difference_wins_over_an_occurrence_difference():
    """Different layers are resolved by mesh_order whatever their occurrences."""
    message = overlap_error(
        sourced("a", (0, 0, 10, 10), layer="M1", occurrence="chip/q0"),
        sourced("b", (5, 5, 15, 15), layer="M2", occurrence="chip/q1"),
    )
    assert "LayerStack `mesh_order`" in message and "assembly-level" not in message


def test_sheets_get_the_same_advice():
    """The Route A sheet overlap error names the source too."""
    first = sourced("a", (0, 0, 10, 10), layer="M1", occurrence="chip/q0")
    second = sourced("b", (5, 5, 15, 15), layer="M1", occurrence="chip/q1")
    sheets = [
        PrismRecord(
            p.entity_id,
            p.footprint,
            exact_z(0),
            exact_z(0),
            "Al",
            "conductor",
            net_id="N",
            layer=p.layer,
            occurrence=p.occurrence,
        )
        for p in (first, second)
    ]
    catalogue = PrismCatalogue(GRID, (*chip_prisms(GRID, 40.0), *sheets))
    with pytest.raises(OverlapError, match="assembly-level priority"):
        build_model_from_catalogue(catalogue, "A")


def test_solution_regions_of_equal_rank_name_their_source_too():
    """Two dielectrics of different materials with equal priority."""
    first = sourced(
        "oxide",
        (-30, -30, 30, 30),
        layer="D1",
        occurrence="chip",
        kind="dielectric",
        material="SiO2",
    )
    second = sourced(
        "resist",
        (-20, -20, 20, 20),
        layer="D2",
        occurrence="chip",
        kind="dielectric",
        material="resist",
    )
    catalogue = PrismCatalogue(GRID, (*chip_prisms(GRID, 40.0), first, second))
    with pytest.raises(OverlapError, match=r"LayerStack `mesh_order` must differ"):
        build_model_from_catalogue(catalogue, "B")


def test_hand_made_prisms_without_a_source_get_the_generic_advice():
    """Unknown source: the message still tells the user to supply distinct priorities."""
    a = PrismRecord(
        "a",
        Footprint.from_rings([(0, 0), (10000, 0), (10000, 10000), (0, 10000)]),
        exact_z(0),
        exact_z(1),
        "Al",
        "conductor",
        net_id="N",
    )
    b = PrismRecord(
        "b",
        Footprint.from_rings(
            [(5000, 5000), (15000, 5000), (15000, 15000), (5000, 15000)]
        ),
        exact_z(0),
        exact_z(1),
        "Al",
        "conductor",
        net_id="N",
    )
    catalogue = PrismCatalogue(GRID, (*chip_prisms(GRID, 40.0), a, b))
    with pytest.raises(
        OverlapError,
        match="the source that declares them must supply distinct priorities",
    ):
        build_model_from_catalogue(catalogue, "B")


def test_s2_records_the_source_layer_and_occurrence_of_real_entities():
    """The thin-film fixture's entities carry their logical layer and (when authored) their occurrence."""
    from sgb_inputs import rectangular_junction_pair  # noqa: PLC0415 -- a fixture used by this test only

    from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue  # noqa: PLC0415 -- used by this test only

    effective = rectangular_junction_pair()
    catalogue = build_prism_catalogue(effective.build_input, effective.representation)
    left = next(p for p in catalogue.prisms if p.entity_id == "chip/left")
    assert (left.layer, left.occurrence) == ("metal", "chip")
