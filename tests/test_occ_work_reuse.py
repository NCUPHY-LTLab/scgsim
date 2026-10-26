"""Check indexed repair against complete source incidence on real OCC shells."""

from dataclasses import replace

import pytest
from sgb_method_cases import circular_hole_case, junction_hole_case, two_volume_case

from scgsim.geometry import interpret_physical_geometry
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.construction import gmsh_occ, native


def _complete_loop_rebind(surfaces, loops, curve_tags, surface_tags, repaired):
    """Exhaustive reference: examine every loop and every incident live face."""
    uses = {}
    for surface in surfaces:
        if not surface.construction_only:
            for loop_id in (surface.outer_loop_ref, *surface.hole_loop_refs):
                uses.setdefault(loop_id, []).append(surface.surface_id)
    result = []
    for loop in loops:
        signature = native._wire_signature(
            tuple(curve_tags[ref.curve_id] for ref in loop.curve_refs)
        )
        requirements = [
            repaired.wire_parities.get(surface_tags[sid], {}).get(signature, 1)
            for sid in uses.get(loop.loop_id, ())
        ]
        assert len(set(requirements)) <= 1
        if requirements and requirements[0] < 0:
            loop = replace(
                loop,
                curve_refs=tuple(
                    replace(ref, orientation=-ref.orientation)
                    for ref in reversed(loop.curve_refs)
                ),
            )
        result.append(loop)
    return tuple(result)


@pytest.mark.parametrize(
    ("case", "route"),
    [
        (two_volume_case, "A"),
        (two_volume_case, "B"),
        (junction_hole_case, "A"),
        (circular_hole_case, "A"),
    ],
)
def test_indexed_repair_matches_complete_incidence(case, route, tmp_path, monkeypatch):
    pytest.importorskip("gmsh")
    source, stack, profile = case()
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    plan = build_route_construction_plan(
        effective, route=route, representation_context=physical.context
    )
    observations = []

    class CheckedIndex(native._SurfaceRepairIndex):
        def __init__(self, surfaces, loops, curve_tags, surface_tags):
            super().__init__(surfaces, loops, curve_tags, surface_tags)
            self.surfaces = surfaces
            self.curve_tags = curve_tags

        def rebind(self, repaired):
            before_tags = dict(self.surface_tags)
            expected = _complete_loop_rebind(
                self.surfaces,
                self.surface_loops,
                self.curve_tags,
                before_tags,
                repaired,
            )
            expected_tags = {
                sid: repaired.face_bindings.get(tag, (tag, 1))[0]
                for sid, tag in before_tags.items()
            }
            result = super().rebind(repaired)
            assert self.surface_loops == expected
            assert self.surface_tags == expected_tags
            loops_by_id = {loop.loop_id: loop for loop in self.surface_loops}
            expected_outer = {
                self.surface_tags[surface.surface_id]: tuple(
                    self.curve_tags[ref.curve_id]
                    for ref in loops_by_id[surface.outer_loop_ref].curve_refs
                )
                for surface in self.surfaces
                if surface.surface_id in self.surface_tags
            }
            # Directed order can reverse; complete cyclic edge correspondence stays fixed.
            assert {
                tag: native._wire_signature(edges)
                for tag, edges in self.outer_wire_edges.items()
            } == {
                tag: native._wire_signature(edges)
                for tag, edges in expected_outer.items()
            }
            observations.append(len(result))
            return result

    monkeypatch.setattr(gmsh_occ, "_SurfaceRepairIndex", CheckedIndex)
    final = gmsh_occ.write_occ_geometry_from_plan(
        plan, xao_path=tmp_path / "geometry.xao"
    )
    assert len(observations) == len(
        [v for v in plan.volumes if not v.construction_only]
    )
    assert final.backend_entity_tags
    assert (tmp_path / "geometry.xao").stat().st_size
