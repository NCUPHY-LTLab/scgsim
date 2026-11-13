"""Validation is structural, fail closed and cannot be switched off; each invariant catches a tampered model."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
from sgb_scenes import pad_in_ground

from scgsim.geometry.sgb import model as model_module
from scgsim.geometry.sgb.checks.validation.arrangements import check_band_areas
from scgsim.geometry.sgb.checks.validation.report import ValidationReport
from scgsim.geometry.sgb.checks.validation.validate import validate_model
from scgsim.geometry.sgb.errors import SgbError, ValidationError
from scgsim.geometry.sgb.exact.labels import LevelLabel
from scgsim.geometry.sgb.exact.stopwatch import Stopwatch
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.topology.assembly.builder import register
from scgsim.geometry.sgb.topology.routes.groups import PhysicalGroup


def fresh(route: str = "B"):
    """A model nobody else has tampered with."""
    return build_model_from_catalogue(pad_in_ground(route), route)


def failed(model) -> set[str]:
    """Names of the invariants that now fail."""
    report = validate_model(model.labelled, model.topology, model.routed, Stopwatch())
    return {c.name for c in report.failures}


def test_the_untouched_model_passes_every_invariant():
    """The baseline of the tampering tests below."""
    assert failed(fresh()) == set()


def test_no_production_entry_point_can_skip_validation():
    """There is no switch: no parameter of the build functions turns validation off."""
    for function in (
        model_module.build_conformal_model,
        model_module.build_model_from_catalogue,
    ):
        names = set(inspect.signature(function).parameters)
        assert not names & {
            "validate",
            "run_validation",
            "check_planarity",
            "skip_validation",
        }


def test_a_failed_invariant_stops_the_build(monkeypatch):
    """The build raises ValidationError; no model is returned."""
    original = model_module.validate_model

    def sabotaged(labelled, topology, routed, rt):
        """Sabotaged."""
        next(iter(topology.volumes.values())).shells.clear()
        report = original(labelled, topology, routed, rt)
        report.add("sabotage", False, "injected failure")
        return report

    monkeypatch.setattr(model_module, "validate_model", sabotaged)
    with pytest.raises(ValidationError, match="invariant"):
        build_model_from_catalogue(pad_in_ground("A"), "A")


def test_a_face_in_two_groups_breaks_the_partition():
    """Primary groups partition each dimension."""
    model = fresh()
    groups = model.routed.groups
    first, second = sorted(g for g in groups if groups[g].dim == 2)[:2]
    groups[second].ids.append(groups[first].ids[0])
    assert "groups_partition_each_dimension" in failed(model)


def test_an_empty_group_and_a_missing_face_are_caught():
    """Every group is non-empty and every emitted face is in one."""
    model = fresh()
    model.routed.groups["MA::ghost"] = PhysicalGroup(2)
    assert "physical_groups_non_empty" in failed(model)
    model = fresh()
    group = next(g for g in model.routed.groups.values() if g.dim == 2)
    group.ids.pop()
    assert "every_emitted_face_in_one_group" in failed(model)


def test_a_union_naming_an_unknown_group_is_caught():
    """Unions reference primary groups only."""
    model = fresh()
    model.routed.unions["NET::pad"].append("MA::unknown")
    assert "unions_reference_primary_groups" in failed(model)


def test_a_cylinder_whose_top_arc_is_on_another_circle_is_caught():
    """Curved walls are single cylinders."""
    model = fresh()
    wall = next(f for f in model.topology.faces.values() if f.curved)
    top = model.topology.edges[wall.cylinder[1]]
    assert top.curve is not None
    top.curve = replace(top.curve, radius=top.curve.radius + 1.0)
    assert "curved_walls_are_single_cylinders" in failed(model)


def test_a_cylinder_whose_top_arc_runs_the_other_way_is_accepted():
    """An edge's direction is arbitrary: the same arc travelled from its other end is the same cylinder."""
    model = fresh()
    wall = next(f for f in model.topology.faces.values() if f.curved)
    top = model.topology.edges[wall.cylinder[1]]
    assert top.curve is not None
    top.v0, top.v1 = top.v1, top.v0
    top.curve = replace(
        top.curve, ccw=not top.curve.ccw, bulge_left=not top.curve.bulge_left
    )
    assert "curved_walls_are_single_cylinders" not in failed(model)


def test_a_cylinder_whose_top_arc_is_the_complementary_arc_is_caught():
    """Same circle and end points but the other way round the circle: not the same arc."""
    model = fresh()
    wall = next(f for f in model.topology.faces.values() if f.curved)
    top = model.topology.edges[wall.cylinder[1]]
    assert top.curve is not None
    top.v0, top.v1 = top.v1, top.v0
    assert "curved_walls_are_single_cylinders" in failed(model)


def test_a_cylinder_whose_top_arc_ends_elsewhere_is_caught():
    """Same circle, different end point: the top arc spans another part of the circle."""
    model = fresh()
    wall = next(f for f in model.topology.faces.values() if f.curved)
    top = model.topology.edges[wall.cylinder[1]]
    other = next(
        v
        for v in model.topology.vertices.values()
        if v.id not in (top.v0, top.v1) and v.z == model.topology.vertices[top.v1].z
    )
    top.v1 = other.id
    assert "curved_walls_are_single_cylinders" in failed(model)


def test_a_lost_face_leaves_dangling_edges_and_an_open_shell():
    """Every edge is shared by two faces; shells are closed."""
    model = fresh()
    face_id = next(f.id for f in model.topology.faces.values() if f.kind == "wall")
    del model.topology.faces[face_id]
    assert failed(model) >= {
        "every_edge_shared_by_two_faces"
    } or "shells_closed_and_oriented" in failed(model)


def test_an_interior_face_missing_a_side_is_caught():
    """An interior face has a volume on each side."""
    model = fresh()
    face = next(
        f
        for f in model.topology.faces.values()
        if f.neg_domain is not None and f.pos_domain is not None
    )
    face.neg_volume = None
    assert "interior_faces_have_two_sides" in failed(model)


def test_a_flipped_loop_orientation_is_caught_by_the_shell_check():
    """Orientation: each edge used twice with opposite sense."""
    model = fresh()
    face = next(f for f in model.topology.faces.values() if f.kind == "horizontal")
    face.loops = tuple(tuple((eid, -sign) for eid, sign in loop) for loop in face.loops)
    assert "shells_closed_and_oriented" in failed(model)


def test_area_conservation_is_compared_against_the_raw_arrangement():
    """Raw, dissolved and enclosed areas must agree."""
    model = fresh()
    model.labelled.level_layers[1].raw.total_twice_area += 2
    assert "area_conservation_per_arrangement" in failed(model)


def test_a_prism_whose_footprint_area_is_lost_over_its_cells_is_caught():
    """Per-prism footprint area is conserved within the snap bound."""
    model = fresh()
    slab = model.labelled.slabs[1]
    assert slab.raw.prism_twice_area is not None
    key = next(iter(slab.raw.prism_twice_area))
    slab.raw.prism_twice_area[key] //= 2
    assert "prism_area_conserved_over_cells" in failed(model)


@pytest.mark.parametrize(
    ("dimension", "ids_of", "check"),
    [
        ("vertex", lambda layer: layer.vertex_ids, "vertex_ids_unique"),
        ("edge", lambda layer: layer.edge_ids, "edge_ids_unique"),
        ("face", lambda layer: layer.face_ids, "face_ids_unique"),
    ],
)
def test_two_entities_sharing_an_atomic_id_are_caught(dimension, ids_of, check):
    """A repeated id would merge two entities in the dict-keyed topology; validation names the dimension."""
    model = fresh()
    ids = ids_of(model.labelled.level_layers[1])
    first, second = (1, 2) if dimension == "face" else (0, 1)
    ids[second] = ids[first]
    assert check in failed(model)


def test_a_record_stored_under_another_entitys_id_is_caught():
    """The topology tables are keyed by the id each record carries."""
    model = fresh()
    first, second = list(model.topology.faces.values())[:2]
    first.id = second.id
    assert "topology_keyed_by_entity_id" in failed(model)


def test_a_volume_that_swallowed_another_volumes_cells_is_caught():
    """Two volumes with one id leave the lost one's cells in no volume."""
    model = fresh()
    del model.topology.volumes[next(iter(model.topology.volumes))]
    assert "every_cell_in_exactly_one_volume" in failed(model)


def test_assembly_refuses_to_register_a_different_entity_under_a_known_id():
    """The assembly tables never let a later entity replace an earlier one."""
    model = fresh()
    vertices = model.topology.vertices
    known = next(iter(vertices.values()))
    register(vertices, replace(known))  # the same record again is fine
    with pytest.raises(SgbError, match="share the atomic id"):
        register(vertices, replace(known, x=known.x + 1))


def test_the_band_area_hook_checks_monotonicity_and_the_support_area():
    """Sub-stage 1c supplies support areas; the hook already compares bands against them."""
    model = fresh("A")
    layer = model.labelled.level_layers[1]
    faces = [
        f for f in range(1, len(layer.arr.faces)) if layer.labels[f].sheet is not None
    ][:2]
    for band, face in enumerate(faces):
        label = layer.labels[face]
        layer.dissolved.labels[face] = LevelLabel(
            label.below, label.above, label.sheet, (("fam", band),)
        )
    consistent = ValidationReport()
    check_band_areas(consistent, [layer])
    assert not consistent.failures
    wrong = ValidationReport()
    check_band_areas(wrong, [layer], {"fam": (1.0, 0.0)})
    assert [c.name for c in wrong.failures] == ["band_area_monotone"]
