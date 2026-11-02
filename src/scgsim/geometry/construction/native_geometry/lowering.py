"""Lower planned native boundaries, source-bound volume shells and physical groups."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from dataclasses import replace
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy
    from numpy.typing import NDArray
from pathlib import Path

from scgsim.geometry.construction.conformality import engine_gate_gmsh_brep_conformality
from scgsim.geometry.construction.export import group_tag_plans
from scgsim.geometry.construction.native import (
    add_surface_first_volume,
    rebind_surface_loops,
    rebind_volume_surface_refs,
    surface_outer_wire_edges,
)
from scgsim.geometry.construction.native_geometry.timing import debug_stage
from scgsim.geometry.construction.native_geometry.topology import (
    add_curve_loop_from_plan,
    backend_entity_tags,
    chosen_surface_orientations,
    physical_entity_tags,
    surface_edge_count,
    surface_ref_tags,
)
from scgsim.geometry.construction.native_types import GmshApi, NativeTimingRecord
from scgsim.geometry.models.common import GmshDimTag
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.topology import VolumePlanRecord


class OccPlanLowerer:
    """Share one native model, evolving face bindings and plan through lowering."""

    def __init__(
        self,
        plan: ConstructionPlanRecord,
        gmsh: GmshApi,
        timings: list[NativeTimingRecord],
        debug_logging: bool,
    ):
        self.plan = plan
        self.gmsh = gmsh
        self.timings = timings
        self.debug_logging = debug_logging

    def write(self, xao_path: Path) -> ConstructionPlanRecord:
        """Lower boundaries and volumes before checking and exporting shared topology."""
        if "curved_boundary_xao" in self.plan.metadata:
            self.import_curved_boundaries()
        else:
            self.build_planar_boundaries()
        self.build_volumes()
        self.check_native_topology()
        self.attach_physical_groups()
        self.attach_transport_groups()
        return self.write_xao(xao_path)

    def import_curved_boundaries(self):
        """Restore exact detached faces and bind their canonical source identities."""
        payload = base64.b64decode(self.plan.metadata["curved_boundary_xao"])
        if (
            hashlib.sha256(payload).hexdigest()
            != self.plan.metadata["curved_boundary_xao_sha256"]
        ):
            raise ValueError("detached curved boundary XAO identity differs")
        with TemporaryDirectory(prefix="scgsim-curved-lowering-") as temporary:
            boundary_path = Path(temporary) / "boundaries.xao"
            boundary_path.write_bytes(payload)
            self.gmsh.merge(str(boundary_path))
        self.gmsh.model.occ.synchronize()
        self.curve_tags: dict[str, int] = {}
        self.source_tags: dict[tuple[str, str], list[GmshDimTag]] = {}
        for dimension, group in self.gmsh.model.getPhysicalGroups():
            name = self.gmsh.model.getPhysicalName(dimension, group)
            # Gmsh returns NumPy integer vectors in the required NumPy runtime.
            tags = cast(
                "NDArray[numpy.int32]",
                self.gmsh.model.getEntitiesForPhysicalGroup(dimension, group),
            ).tolist()
            if name.startswith("SGB_CURVE::"):
                if dimension != 1 or len(tags) != 1:
                    raise ValueError(
                        "detached curved topology has an invalid curve binding"
                    )
                self.curve_tags[name.removeprefix("SGB_CURVE::")] = tags[0]
            elif name.startswith("SGB_SURFACE::"):
                if dimension != 2 or len(tags) != 1:
                    raise ValueError(
                        "detached curved topology has an invalid surface binding"
                    )
                self.source_tags[("surface", name.removeprefix("SGB_SURFACE::"))] = [
                    (2, tags[0])
                ]
        self.gmsh.model.removePhysicalGroups()
        if set(self.curve_tags) != {
            curve.curve_id for curve in self.plan.curves
        } or set(self.source_tags) != {
            ("surface", surface.surface_id)
            for surface in self.plan.surfaces
            if not surface.construction_only
        }:
            raise ValueError(
                "detached curved topology does not cover its planned boundaries"
            )

    def build_planar_boundaries(self):
        """Create points, curves, loops and planar faces in their source order."""
        point_tags: dict[str, int] = {}
        with debug_stage(
            self.debug_logging,
            f"add {len(self.plan.points)} OCC points",
            self.timings,
        ):
            for point in self.plan.points:
                point_tags[point.point_id] = self.gmsh.model.occ.addPoint(
                    *point.coordinate
                )
        self.curve_tags: dict[str, int] = {}
        with debug_stage(
            self.debug_logging,
            f"add {len(self.plan.curves)} OCC curves",
            self.timings,
        ):
            for curve in self.plan.curves:
                self.curve_tags[curve.curve_id] = self.gmsh.model.occ.addLine(
                    point_tags[curve.start_point_id],
                    point_tags[curve.end_point_id],
                )
        loop_tags: dict[str, int] = {}
        surface_orientations = (
            chosen_surface_orientations(self.plan)
            if self.plan.route in {"A", "B"}
            else {}
        )
        with debug_stage(
            self.debug_logging,
            f"add {len(self.plan.surface_loops)} OCC curve loops",
            self.timings,
        ):
            for loop in self.plan.surface_loops:
                if self.debug_logging and len(loop.curve_refs) > 512:
                    print(
                        "[sgb:gmsh] "
                        f"large loop {loop.loop_id}: {len(loop.curve_refs)} curves",
                        flush=True,
                    )
                if self.plan.route in {"A", "B"}:
                    reverse = (
                        surface_orientations.get(cast(str, loop.surface_id), "forward")
                        == "reversed"
                    )
                    loop_tags[loop.loop_id] = add_curve_loop_from_plan(
                        self.gmsh,
                        loop,
                        self.curve_tags,
                        reverse=reverse,
                    )
                else:
                    loop_tags[loop.loop_id] = add_curve_loop_from_plan(
                        self.gmsh,
                        loop,
                        self.curve_tags,
                    )
        self.source_tags: dict[tuple[str, str], list[GmshDimTag]] = {}
        live_surfaces = tuple(
            surface for surface in self.plan.surfaces if not surface.construction_only
        )
        with debug_stage(
            self.debug_logging,
            f"add {len(live_surfaces)} OCC plane surfaces",
            self.timings,
        ):
            for surface in live_surfaces:
                if self.debug_logging:
                    edge_count = surface_edge_count(surface, self.plan.surface_loops)
                    if edge_count > 512:
                        print(
                            "[sgb:gmsh] "
                            f"large surface {surface.surface_id}: {edge_count} edges",
                            flush=True,
                        )
                surface_tag = self.gmsh.model.occ.addPlaneSurface(
                    [
                        loop_tags[cast(str, surface.outer_loop_ref)],
                        *(loop_tags[loop_id] for loop_id in surface.hole_loop_refs),
                    ]
                )
                self.source_tags.setdefault(("surface", surface.surface_id), []).append(
                    (2, surface_tag)
                )

    def build_volumes(self):
        """Create volumes while retaining the global source-face replacement map."""
        live_volumes = tuple(
            volume for volume in self.plan.volumes if not volume.construction_only
        )
        with debug_stage(
            self.debug_logging, f"add {len(live_volumes)} OCC volumes", self.timings
        ):
            self.largest_volume_boundary: tuple[int, str] = (0, "")
            self.surface_basis_parity = {
                surface.surface_id: 1 for surface in self.plan.surfaces
            }
            for volume in live_volumes:
                volume_tag = self.add_volume(volume)
                self.source_tags.setdefault(("volume", volume.volume_id), []).append(
                    (3, volume_tag)
                )
            if self.debug_logging and self.largest_volume_boundary[1]:
                print(
                    "[sgb:gmsh] "
                    f"largest volume boundary {self.largest_volume_boundary[1]}: "
                    f"{self.largest_volume_boundary[0]} surfaces",
                    flush=True,
                )
        if self.plan.route in {"A", "B"}:
            self.plan = replace(
                self.plan,
                volumes=rebind_volume_surface_refs(
                    self.plan.volumes, self.surface_basis_parity
                ),
            )

    def add_volume(self, volume: VolumePlanRecord) -> int:
        """Select the existing A/B shell repair or retained Route C construction."""
        if self.plan.route in {"A", "B"}:
            exterior_refs = (
                getattr(volume, "exterior_surface_refs", ()) or volume.surface_refs
            )
            exterior_tags = surface_ref_tags(
                volume.volume_id, exterior_refs, self.source_tags
            )
            shell_surfaces = [exterior_tags]
            shell_surfaces.extend(
                surface_ref_tags(void.shell_id, void.surface_refs, self.source_tags)
                for void in getattr(volume, "inner_pec_void_shells", ())
            )
            self.observe_boundary(volume.volume_id, exterior_tags)
            return self.add_repaired_volume(shell_surfaces)
        exterior_tags = surface_ref_tags(
            volume.volume_id, volume.surface_refs, self.source_tags
        )
        shell_tags = [self.gmsh.model.occ.addSurfaceLoop(exterior_tags)]
        self.observe_boundary(volume.volume_id, exterior_tags)
        return self.gmsh.model.occ.addVolume(shell_tags)

    def observe_boundary(self, volume_id: str, exterior_tags: Sequence[int]) -> None:
        """Retain the largest boundary for the existing native diagnostic message."""
        if len(exterior_tags) > self.largest_volume_boundary[0]:
            self.largest_volume_boundary = (len(exterior_tags), volume_id)

    def add_repaired_volume(self, shell_surfaces: Sequence[Sequence[int]]) -> int:
        """Rebind loops, face parity and native tags from complete wire correspondence."""
        surface_tags = {
            sid: values[0][1]
            for (kind, sid), values in self.source_tags.items()
            if kind == "surface"
        }
        repaired = add_surface_first_volume(
            self.gmsh,
            shell_surfaces,
            outer_wire_edges=surface_outer_wire_edges(
                self.plan.surfaces,
                self.plan.surface_loops,
                self.curve_tags,
                surface_tags,
            ),
        )
        self.plan = replace(
            self.plan,
            surface_loops=rebind_surface_loops(
                self.plan.surfaces,
                self.plan.surface_loops,
                self.curve_tags,
                surface_tags,
                repaired,
            ),
        )
        for sid, tag in surface_tags.items():
            if tag in repaired.face_bindings:
                self.surface_basis_parity[sid] *= repaired.face_bindings[tag][1]
        volume_tag = repaired.volume_tag
        for key, values in tuple(self.source_tags.items()):
            if key[0] != "surface":
                continue
            self.source_tags[key] = [
                (
                    dim,
                    repaired.face_bindings[tag][0]
                    if dim == 2 and tag in repaired.face_bindings
                    else tag,
                )
                for dim, tag in values
            ]

        return volume_tag

    def check_native_topology(self):
        """Synchronize before recording native topology conformity and backend tags."""
        with debug_stage(self.debug_logging, "synchronize OCC model", self.timings):
            self.gmsh.model.occ.synchronize()
        with debug_stage(
            self.debug_logging, "engine gate gmsh_brep_conformality", self.timings
        ):
            self.gmsh_brep_gate = engine_gate_gmsh_brep_conformality(
                self.plan,
                gmsh=self.gmsh,
                source_tags=self.source_tags,
                curve_tags=self.curve_tags,
            )
        self.backend_tags = backend_entity_tags(self.source_tags)
        self.grouped_tag_plans = group_tag_plans(self.plan.tags)

    def attach_physical_groups(self):
        """Attach each planned physical group after the native boundary check."""
        with debug_stage(
            self.debug_logging,
            f"add {len(self.grouped_tag_plans)} physical groups",
            self.timings,
        ):
            for tag_plans in self.grouped_tag_plans:
                first_tag = tag_plans[0]
                entity_tags = physical_entity_tags(tag_plans, self.source_tags)
                if not entity_tags:
                    raise ValueError(
                        f"{first_tag.physical_name} has no backend entity tags"
                    )
                group_tag = self.gmsh.model.addPhysicalGroup(
                    first_tag.dimension,
                    entity_tags,
                )
                self.gmsh.model.setPhysicalName(
                    first_tag.dimension,
                    group_tag,
                    first_tag.physical_name,
                )

    def attach_transport_groups(self):
        """Retain internal port and volume names for XAO tag renumbering."""
        port_host_ids = dict.fromkeys(
            host
            for surface in self.plan.surfaces
            for host in surface.metadata.get("embedded_volume_plan_ids", ())
        )
        for surface in self.plan.surfaces:
            if "embedded_volume_plan_ids" not in surface.metadata:
                continue
            members = [
                tag
                for dim, tag in self.source_tags[("surface", surface.surface_id)]
                if dim == 2
            ]
            group_tag = self.gmsh.model.addPhysicalGroup(2, members)
            self.gmsh.model.setPhysicalName(
                2, group_tag, "SGB_PORT_SURFACE::" + surface.surface_id
            )
        for volume_id in port_host_ids:
            members = [
                tag for dim, tag in self.source_tags[("volume", volume_id)] if dim == 3
            ]
            group_tag = self.gmsh.model.addPhysicalGroup(3, members)
            self.gmsh.model.setPhysicalName(3, group_tag, "SGB_VOLUME::" + volume_id)

    def write_xao(self, xao_path: Path):
        """Write the final native model and attach its measured lowering evidence."""
        with debug_stage(self.debug_logging, f"write XAO {xao_path}", self.timings):
            self.gmsh.write(str(xao_path))

        return replace(
            self.plan,
            backend_entity_tags=self.backend_tags,
            metadata={
                **dict(self.plan.metadata),
                "xao_path": str(xao_path),
                "backend_timings": self.timings,
                "engine_gate_gmsh_brep_conformality": self.gmsh_brep_gate,
            },
        )
