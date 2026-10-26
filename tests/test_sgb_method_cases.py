"""The method narrative and figures use these same real compiler inputs."""

from scgsim.geometry import interpret_physical_geometry
from scgsim.geometry.compiler import build_route_construction_plan
from sgb_method_cases import junction_hole_case, two_volume_case


def area(ring):
    return abs(sum(a[0] * b[1] - a[1] * b[0]
                   for a, b in zip(ring, (*ring[1:], ring[0])))) / 2


def test_complex_case_source_hole_and_terminal_completion_agree():
    source, stack, profile = junction_hole_case()
    left = next(p for p in source.polygons if p.polygon_id == "left")
    assert area(left.exterior) - sum(map(area, left.holes)) == 8
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    junction = effective.metadata["junction_partitions"]["junction"]
    central = next(p for p in effective.polygons
                   if p.polygon_id == junction["central_polygon_id"])
    assert {(3, 0), (7, 0), (7, 2), (3, 2)} <= set(central.exterior)
    assert area(central.exterior) == 8
    assert all(x in (3, 7) and 0 <= y <= 2 for x, y in central.exterior)
    plan = build_route_construction_plan(effective, route="A", representation_context=physical.context)
    assert not any("surface_distance_band" in s.metadata for s in plan.surfaces)
    left_faces = [s for s in plan.surfaces
                  if "chip/left" in s.metadata.get("owner_semantic_ids", ())
                  and "MS" in s.metadata.get("interface_kinds", ())]
    assert left_faces


def test_two_volumes_reference_one_shared_face_oppositely():
    source, stack, profile = two_volume_case()
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    plan = build_route_construction_plan(effective, route="A", representation_context=physical.context)
    assert len(plan.volumes) == 2
    uses = {}
    for volume in plan.volumes:
        for ref in volume.surface_refs:
            uses.setdefault(ref.surface_id, []).append(ref.orientation)
    shared = [value for value in uses.values() if len(value) == 2]
    assert shared == [["forward", "reversed"]] or shared == [["reversed", "forward"]]
