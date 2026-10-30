"""Shared sidewalls for neighboring solution regions with different Z spans."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from scgsim.geometry import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.compiler.validation import validate_volume_surface_closure
from scgsim.geometry.planning.domain import (
    _merge_solution_sidewall_interfaces,
    _plan_substrate_air_surfaces,
    _reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.evidence import build_semantic_evidence_facade
from scgsim.geometry.planning.topology import plan_canonical_topology
from scgsim.geometry.planning.volumes import plan_route_volumes


def _solution(name, x0, x1, lo, hi, *, material="air", kind="vacuum"):
    geometry = {
        "outer_loop": ((x0, 0), (x1, 0), (x1, 1), (x0, 1)),
        "hole_loops": (),
        "z_min_um": lo,
        "z_max_um": hi,
        "thickness_um": hi - lo,
        "domain_bounds_um": {
            "x_min_um": x0,
            "x_max_um": x1,
            "y_min_um": 0,
            "y_max_um": 1,
            "z_min_um": lo,
            "z_max_um": hi,
        },
    }
    return SemanticEntitySpec(
        name,
        "solution_region",
        material,
        kind,
        0,
        "polygon",
        geometry=geometry,
        metadata={"custom_region_label": "source:" + name},
    )


def adjacent_solution_source():
    """A tall dielectric block shares its x=1 wall with three authored air strips."""
    entities = (
        _solution("tall", 0, 1, 0, 3, material="Si", kind="dielectric"),
        *(_solution(f"strip_{i}", 1, 2, i, i + 1) for i in range(3)),
    )
    regions = {
        entity.semantic_id: {
            "semantic_id": entity.semantic_id,
            "geometry": dict(entity.geometry),
            "metadata": dict(entity.metadata),
        }
        for entity in entities
    }
    return GeometryBuildInput((), entities, regions, {"source_dbu_um": 0.001})


def _merge(source, surfaces):
    reconciled = _reconcile_solution_domain_boundaries(source, surfaces=surfaces)
    return _merge_solution_sidewall_interfaces(
        source,
        surfaces=reconciled,
        semantic_facts=build_semantic_evidence_facade(source, route="A"),
    )


def _interface_wall(surface):
    quad = surface.geometry_ref.get("quad_points", ())
    return bool(quad) and all(point[0] == 1 for point in quad)


@pytest.mark.parametrize("reverse_tall", [False, True])
def test_shared_children_preserve_source_regions_lineage_and_orientations(reverse_tall):
    source = adjacent_solution_source()
    original_entities = deepcopy(source.entities)
    original_regions = deepcopy(source.solution_regions)
    surfaces = _plan_substrate_air_surfaces(
        source,
        route="A",
        semantic_facts=build_semantic_evidence_facade(source, route="A"),
    )
    if reverse_tall:
        surfaces = tuple(
            replace(
                surface,
                geometry_ref={
                    **surface.geometry_ref,
                    "quad_points": tuple(reversed(surface.geometry_ref["quad_points"])),
                },
            )
            if surface.owner_semantic_id == "tall" and _interface_wall(surface)
            else surface
            for surface in surfaces
        )
    source_wall = next(
        s for s in surfaces if s.owner_semantic_id == "tall" and _interface_wall(s)
    )
    merged = _merge(source, surfaces)
    shared = [s for s in merged if _interface_wall(s)]
    assert len(shared) == 3
    assert {
        tuple(sorted(p[2] for p in s.geometry_ref["quad_points"])) for s in shared
    } == {
        (0, 0, 1, 1),
        (1, 1, 2, 2),
        (2, 2, 3, 3),
    }
    assert len({surface.interface_id for surface in shared}) == 3
    assert (
        sum(
            max(p[2] for p in s.geometry_ref["quad_points"])
            - min(p[2] for p in s.geometry_ref["quad_points"])
            for s in shared
        )
        == 3
    )
    for surface in shared:
        owners = surface.metadata["boundary_volume_ids"]
        assert len(owners) == 2 and "tall" in owners
        assert surface.metadata["interface_kinds"] == ("SA",)
        assert surface.surface_role == "solution_interface"
        sources = surface.metadata["solution_sidewall_sources"]
        assert {record["owner_semantic_id"] for record in sources} == set(owners)
        tall = next(r for r in sources if r["owner_semantic_id"] == "tall")
        assert tall["parent_surface_id"] == source_wall.surface_id
        assert tall["source_surface_id"] == source_wall.surface_id
        assert tall["source_quad_points"] == source_wall.geometry_ref["quad_points"]
        assert tall["child_surface_id"].startswith(source_wall.surface_id + "_Z")
        quad = surface.geometry_ref["quad_points"]
        # Subdivision retains the exact XY ordering, including reversed authored rings.
        assert tuple(p[:2] for p in quad) == tuple(
            p[:2] for p in source_wall.geometry_ref["quad_points"]
        )
    _, _, loops, final_surfaces = plan_canonical_topology(surfaces=merged)
    volumes = plan_route_volumes(source, route="A", surfaces=final_surfaces)
    validate_volume_surface_closure(
        volumes=volumes, surfaces=final_surfaces, surface_loops=loops
    )
    for surface in shared:
        uses = [
            ref
            for volume in volumes
            for ref in volume.surface_refs
            if ref.surface_id == surface.surface_id
        ]
        assert {ref.orientation for ref in uses} == {"forward", "reversed"}
    assert {volume.owner_semantic_id: volume.material_id for volume in volumes} == {
        entity.semantic_id: entity.material_id for entity in source.entities
    }
    assert source.entities == original_entities
    assert source.solution_regions == original_regions


def test_unmatched_disjoint_z_and_other_xy_support_stay_unchanged():
    entities = (
        _solution("first", 0, 1, 0, 1),
        _solution("above_gap", 1, 2, 2, 3),
        _solution("elsewhere", 4, 5, 0, 3),
    )
    source = GeometryBuildInput((), entities)
    surfaces = _plan_substrate_air_surfaces(
        source,
        route="A",
        semantic_facts=build_semantic_evidence_facade(source, route="A"),
    )
    assert {s.surface_id: s for s in _merge(source, surfaces)} == {
        s.surface_id: s for s in surfaces
    }


def test_shared_wall_tags_reach_palace_as_sa_interfaces_and_exteriors_stay_exterior():
    from scgsim.geometry.construction.export import _structured_surface_group
    from scgsim.geometry.models.construction import ConstructionPlanRecord
    from scgsim.geometry.planning.surfaces import _with_surface_contract_metadata
    from scgsim.geometry.planning.tags import plan_route_tags
    from scgsim.palace.mesh.build import (
        _add_surface_group_records,
        _build_groups_from_sgb_records,
    )

    source = adjacent_solution_source()
    initial = _plan_substrate_air_surfaces(
        source, route="A",
        semantic_facts=build_semantic_evidence_facade(source, route="A"),
    )
    surfaces = _with_surface_contract_metadata(
        source, route="A", interfaces=(), mm_contacts=(),
        surfaces=_merge(source, initial),
    )
    tags = plan_route_tags(route="A", surfaces=surfaces, volumes=())
    plan = ConstructionPlanRecord("A", surfaces=surfaces, tags=tags)
    tags_by_source = {tag.source_record_id: tag for tag in tags}
    groups = _build_groups_from_sgb_records()
    kinds = {entity.semantic_id: entity.material_kind for entity in source.entities}
    shared_count = exterior_count = 0
    for surface in surfaces:
        shared = _interface_wall(surface)
        exterior = surface.surface_role == "domain_boundary"
        if not (shared or exterior):
            continue
        tag = tags_by_source[surface.surface_id]
        assert tag.role == ("solution_interface" if shared else "domain_boundary")
        # Opaque native IDs exercise only detached record projection, without Gmsh.
        record = _structured_surface_group(plan, (tag,), (surface,), (1,))
        _add_surface_group_records(
            groups=groups, name=record.physical_name, phys_group=1,
            entity_tags=record.entity_tags, record=asdict(record), route="A",
            solution_material_kinds=kinds,
        )
        info = groups["boundary_surfaces"][record.surface_id]
        if shared:
            shared_count += 1
            assert info["source"] == "volume_interface"
            assert info["interface_type"] == "SA"
            assert info["surface_epr"] is True
            assert set(info["adjacent_solution_volume_ids"]) == set(
                surface.metadata["boundary_volume_ids"]
            )
            assert surface.metadata["solution_sidewall_sources"]
        else:
            exterior_count += 1
            assert info["source"] == "domain_boundary"
            assert info["surface_epr"] is False
    assert shared_count == 3
    assert exterior_count > 0
    assert not groups["pec_surfaces"]
