"""Own supported surface-first native shell repair and exact source-edge/loop correspondence. Preserve shared final face ownership."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.diagnostics import _stage


@dataclass(frozen=True)
class _SurfaceFirstResult:
    volume_tag: int
    # Original face -> (result face, typed outer-wire basis parity).
    face_bindings: dict[int, tuple[int, int]]
    wire_parities: dict[int, dict[tuple[int, ...], int]]


def _wire_signature(loop: Sequence[int]) -> tuple[int, ...]:
    edges = tuple(abs(int(edge)) for edge in loop)
    # Native WireExplorer supplies cyclic incidence, not an outer-wire label.
    return min(
        sequence[i:] + sequence[:i]
        for sequence in (edges, edges[::-1])
        for i in range(len(edges))
    )


def _face_boundary(gmsh: Any, tag: int) -> dict[tuple[int, ...], Counter]:
    """Complete per-wire directed incidence, without geometry matching."""
    directed = Counter(
        int(edge)
        for dim, edge in gmsh.model.getBoundary(
            [(2, tag)], combined=False, oriented=True
        )
        if dim == 1
    )
    wires = {}
    used = Counter()
    for loop in gmsh.model.occ.getCurveLoops(tag)[1]:
        signature = _wire_signature(loop)
        edges = Counter(abs(int(edge)) for edge in loop)
        if signature in wires or used.keys() & edges.keys():
            raise ValueError(f"native face wire incidence is ambiguous: face={tag}")
        wire = Counter(
            {edge: count for edge, count in directed.items() if abs(edge) in edges}
        )
        if Counter(abs(edge) for edge in wire.elements()) != edges:
            raise ValueError(f"native face wire incidence is incomplete: face={tag}")
        wires[signature] = wire
        used.update(edges)
    if used != Counter(abs(edge) for edge in directed.elements()):
        raise ValueError(f"native face wire incidence is incomplete: face={tag}")
    return wires


class _SurfaceRepairIndex:
    """Reuse source incidence while updating only faces touched by shell repair.

    Native curves remain fixed during shell repair. Their cyclic signatures
    survive loop reversal, so shared-loop users and outer-wire edges can be
    indexed once. Each affected loop still checks every incident live surface.
    """

    def __init__(self, surfaces, loops, curve_tags, surface_tags):
        with _stage(
            "geometry.occ.index_shared_maps",
            counts={"surfaces": len(surfaces), "loops": len(loops)},
        ):
            self.surface_tags = surface_tags
            self.basis_parity = {surface.surface_id: 1 for surface in surfaces}
            self._loops = list(loops)
            self._loop_indices = {loop.loop_id: i for i, loop in enumerate(loops)}
            self._loop_signatures = {
                loop.loop_id: _wire_signature(
                    tuple(curve_tags[ref.curve_id] for ref in loop.curve_refs)
                )
                for loop in loops
            }
            self._loop_uses = {}
            self._surface_loops = {}
            self._outer_edges = {}
            for surface in surfaces:
                sid = surface.surface_id
                if not surface.construction_only:
                    loop_ids = (surface.outer_loop_ref, *surface.hole_loop_refs)
                    self._surface_loops[sid] = loop_ids
                    for loop_id in loop_ids:
                        self._loop_uses.setdefault(loop_id, []).append(sid)
                if sid in surface_tags:
                    outer = self._loops[self._loop_indices[surface.outer_loop_ref]]
                    self._outer_edges[sid] = tuple(
                        curve_tags[ref.curve_id] for ref in outer.curve_refs
                    )
            self._surface_ids_by_tag = {}
            self.outer_wire_edges = {}
            for sid, tag in surface_tags.items():
                self._surface_ids_by_tag.setdefault(tag, []).append(sid)
                self.outer_wire_edges[tag] = self._outer_edges[sid]

    @property
    def surface_loops(self):
        return tuple(self._loops)

    def rebind(self, repaired):
        affected = {
            sid: (old_tag, *repaired.face_bindings[old_tag])
            for old_tag in repaired.face_bindings
            for sid in self._surface_ids_by_tag.get(old_tag, ())
        }
        loop_ids = {
            loop_id for sid in affected for loop_id in self._surface_loops.get(sid, ())
        }
        with _stage(
            "geometry.occ.rebind_affected_faces",
            counts={
                "affected_surfaces": len(affected),
                "affected_loops": len(loop_ids),
                "incident_surface_uses": sum(
                    len(self._loop_uses[loop_id]) for loop_id in loop_ids
                ),
            },
        ):
            for loop_id in sorted(loop_ids, key=self._loop_indices.__getitem__):
                signature = self._loop_signatures[loop_id]
                requirements = {}
                for sid in self._loop_uses[loop_id]:
                    tag = self.surface_tags[sid]
                    if tag in repaired.wire_parities:
                        parities = repaired.wire_parities[tag]
                        if signature not in parities:
                            raise ValueError(
                                f"native repair lost source loop correspondence: {loop_id}, surface={sid}"
                            )
                        requirements[sid] = parities[signature]
                    else:
                        requirements[sid] = 1
                if len(set(requirements.values())) > 1:
                    raise ValueError(
                        f"native repair has conflicting shared loop directions: "
                        f"loop={loop_id}, surfaces={requirements}"
                    )
                if requirements and next(iter(requirements.values())) < 0:
                    index = self._loop_indices[loop_id]
                    loop = self._loops[index]
                    self._loops[index] = replace(
                        loop,
                        curve_refs=tuple(
                            replace(ref, orientation=-ref.orientation)
                            for ref in reversed(loop.curve_refs)
                        ),
                    )
            # Remove old keys first so a repaired face-tag permutation remains valid.
            for old_tag in repaired.face_bindings:
                self._surface_ids_by_tag.pop(old_tag, None)
                self.outer_wire_edges.pop(old_tag, None)
            for sid, (_, new_tag, parity) in affected.items():
                self.surface_tags[sid] = new_tag
                self.basis_parity[sid] *= parity
                self._surface_ids_by_tag.setdefault(new_tag, []).append(sid)
                self.outer_wire_edges[new_tag] = self._outer_edges[sid]
        return {sid: new_tag for sid, (_, new_tag, _) in affected.items()}


def _rebind_volume_surface_refs(volumes, parities):
    """Express retained shell directions in the final typed outer-wire basis."""

    def rebind(ref):
        return (
            ref
            if parities[ref.surface_id] > 0
            else replace(
                ref,
                orientation="reversed" if ref.orientation == "forward" else "forward",
            )
        )

    return tuple(
        replace(
            volume,
            surface_refs=tuple(rebind(ref) for ref in volume.surface_refs),
            exterior_surface_refs=tuple(
                rebind(ref) for ref in volume.exterior_surface_refs
            ),
            inner_pec_void_shells=tuple(
                replace(
                    void, surface_refs=tuple(rebind(ref) for ref in void.surface_refs)
                )
                for void in volume.inner_pec_void_shells
            ),
        )
        for volume in volumes
    )


def add_surface_first_volume(
    gmsh: Any,
    shell_surfaces: Sequence[Sequence[int]],
    *,
    outer_wire_edges: Mapping[int, Sequence[int]],
) -> _SurfaceFirstResult:
    """Construct with stock correction and return unique canonical face bindings.

    Native errors propagate and the caller's AutoFix option is restored on
    failure. Complete wire/edge incidence binds repaired faces to their source.
    Wires may reverse independently; the typed outer loop defines face basis;
    ambiguous or changed boundaries cannot establish that correspondence.
    Final coowner bodies must still consume the same source-bound native faces.
    """
    with _stage("geometry.occ.synchronize_before_shell"):
        gmsh.model.occ.synchronize()
    with _stage(
        "geometry.occ.capture_input_wires",
        counts={
            "shells": len(shell_surfaces),
            "surface_uses": sum(map(len, shell_surfaces)),
        },
    ):
        inputs = {
            tag: _face_boundary(gmsh, tag) for shell in shell_surfaces for tag in shell
        }
    with _stage(
        "geometry.occ.build_shells_and_volume", counts={"shells": len(shell_surfaces)}
    ):
        previous = gmsh.option.getNumber("Geometry.OCCAutoFix")
        try:
            gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
            shell_tags = tuple(
                gmsh.model.occ.addSurfaceLoop(tags, sewing=False)
                for tags in shell_surfaces
            )
            volume = gmsh.model.occ.addVolume(shell_tags)
        finally:
            gmsh.option.setNumber("Geometry.OCCAutoFix", previous)
    with _stage("geometry.occ.synchronize_after_shell"):
        gmsh.model.occ.synchronize()
    with _stage("geometry.occ.capture_repaired_wires"):
        boundary = tuple(
            (dim, int(tag))
            for dim, tag in gmsh.model.getBoundary(
                [(3, volume)], combined=False, oriented=True
            )
            if dim == 2
        )
        actual = {abs(tag): _face_boundary(gmsh, abs(tag)) for _, tag in boundary}
    with _stage(
        "geometry.occ.match_repaired_faces",
        counts={"input_faces": len(inputs), "actual_faces": len(actual)},
    ):
        bindings = {}
        wire_parities = {}
        for original, wires in inputs.items():
            candidates = [
                tag
                for tag, candidate_wires in actual.items()
                if candidate_wires.keys() == wires.keys()
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"native repair face correspondence is not unique: "
                    f"input={original}, candidates={candidates}, volume={volume}"
                )
            repaired = candidates[0]
            parities = {}
            for signature, directed in wires.items():
                repaired_directed = actual[repaired][signature]
                if directed == repaired_directed:
                    parities[signature] = 1
                elif (
                    Counter({-edge: count for edge, count in directed.items()})
                    == repaired_directed
                ):
                    parities[signature] = -1
                else:
                    raise ValueError(
                        f"native repair changed directed wire incidence: "
                        f"input={original}, result={repaired}, volume={volume}"
                    )
            outer = _wire_signature(outer_wire_edges[original])
            if outer not in parities:
                raise ValueError(
                    f"native repair lost typed outer wire: input={original}, volume={volume}"
                )
            bindings[original] = (repaired, parities[outer])
            wire_parities[original] = parities
        if len({tag for tag, _ in bindings.values()}) != len(inputs) or {
            tag for tag, _ in bindings.values()
        } != set(actual):
            raise ValueError(
                f"native repair boundary correspondence is incomplete: volume={volume}"
            )
    return _SurfaceFirstResult(volume, bindings, wire_parities)
