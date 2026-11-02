"""Q3D v3 source-backed lowering into native preparation records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from scgsim.aedt.data import canonical_sha256, detached
from scgsim.aedt.io import file_sha256, write_json
from scgsim.aedt.preparation.modeling import interpret_geometry
from scgsim.aedt.specs.common import MatrixRunControl, PdkMaterial
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec, Q3dTerminalPair
from scgsim.aedt.versions import normalize_aedt_version
from scgsim.geometry import GeometryPlanSnapshot
from scgsim.semantics.route_a import geometry_z_range


def _entity_polygons(entity, polygons):
    if entity.polygon_ids:
        return [
            (pid, polygons[pid].exterior, polygons[pid].holes)
            for pid in sorted(entity.polygon_ids)
        ]
    if entity.material_kind == "conductor":
        raise ValueError(f"Q3D conductor {entity.semantic_id!r} has no source polygons")
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return [(None, geometry["outer_loop"], geometry.get("hole_loops", ()))]
    bounds = geometry["domain_bounds_um"]
    xmin, ymin = bounds["x_min_um"], bounds["y_min_um"]
    xmax, ymax = bounds["x_max_um"], bounds["y_max_um"]
    return [(None, ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), ())]


def lower_q3d_geometry(  # noqa: PLR0913 -- preserve the source-backed preparation keyword API.
    snapshot: GeometryPlanSnapshot,
    *,
    aedt_version: str,
    modeling: Modeling,
    directory: Path,
    project_name: str,
    design_name: str,
    materials: Mapping[str, PdkMaterial],
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    terminal_pairs: Mapping[str, Q3dTerminalPair] | None = None,
    run_control: MatrixRunControl | None = None,
    region_padding_um: Sequence[float],
) -> Q3dSpec:
    """Write canonical source provenance and body declarations; never open AEDT."""
    return _Q3dLowering(
        snapshot,
        aedt_version,
        modeling,
        directory,
        project_name,
        design_name,
        materials,
        net_types,
        physical_ground_nets,
        terminal_pairs,
        run_control,
        region_padding_um,
    ).build()


@dataclass
class _Q3dLowering:
    """Carry one source snapshot through native body, Net, and trace lowering."""

    snapshot: GeometryPlanSnapshot
    aedt_version: str
    modeling: Modeling
    directory: Path
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    net_types: Mapping[str, Literal["Signal", "Ground"]]
    physical_ground_nets: Sequence[str]
    terminal_pairs: Mapping[str, Q3dTerminalPair] | None
    run_control: MatrixRunControl | None
    region_padding_um: Sequence[float]

    def build(self) -> Q3dSpec:
        self._validate_source()
        self._bind_materials()
        self._lower_bodies()
        self._lower_nets()
        self._write_source_trace()
        return self._assemble_spec()

    def _validate_source(self) -> None:
        self.aedt_version = normalize_aedt_version(self.aedt_version)
        if not isinstance(self.snapshot, GeometryPlanSnapshot):
            raise TypeError("snapshot must be GeometryPlanSnapshot")
        if self.terminal_pairs is not None and not isinstance(
            self.terminal_pairs, Mapping
        ):
            raise TypeError("terminal_pairs must map final Net IDs to Q3dTerminalPair")
        self.pairs = {} if self.terminal_pairs is None else dict(self.terminal_pairs)
        if any(not isinstance(pair, Q3dTerminalPair) for pair in self.pairs.values()):
            raise TypeError("terminal_pairs values must be Q3dTerminalPair")
        self.source = self.snapshot.geometry_input
        if self.source.boundary_curves or self.source.boundary_reconstruction:
            raise NotImplementedError(
                "AEDT finite Q3D lowering does not support active source curve intent"
            )
        self.stack = self.snapshot.stack
        _, _, self.physical_plan = interpret_geometry(
            self.source, self.stack, self.modeling
        )
        self.physical_layers = self.physical_plan.declarations
        self.layer_bindings = {
            key: value["physical_layer_id"]
            for key, value in self.physical_plan.entities.items()
        }
        self.conductors = [
            entity
            for entity in self.source.entities
            if entity.material_kind == "conductor"
        ]
        if any(entity.net_id is None for entity in self.conductors):
            raise ValueError(
                "Q3D conductors require explicit final GeometryPlan Net membership"
            )
        self.net_ids = {cast(str, entity.net_id) for entity in self.conductors}
        if set(self.net_types) != self.net_ids:
            raise ValueError("net_types must cover exactly the final conductor Nets")
        if not set(self.pairs) <= self.net_ids:
            raise ValueError("terminal_pairs must select final conductor Nets")
        self.ground_nets = tuple(self.physical_ground_nets)
        if (
            isinstance(self.physical_ground_nets, str | bytes)
            or not set(self.ground_nets) <= self.net_ids
        ):
            raise ValueError("physical_ground_nets must select final conductor Nets")
        self.imported = [
            entity
            for entity in self.source.entities
            if entity.material_kind in {"conductor", "dielectric"}
        ]

    def _bind_materials(self) -> None:
        self.source_materials = self.stack["materials"]
        for entity in self.source.entities:
            material = self.materials[entity.material_id]
            record = self.source_materials[entity.material_id]
            expected_kind = (
                "superconductor"
                if entity.material_kind == "conductor"
                else entity.material_kind
            )
            if (
                material.material_id != entity.material_id
                or material.kind != expected_kind
                or record["kind"] != entity.material_kind
            ):
                raise ValueError(
                    f"Q3D explicit material binding differs from source {entity.material_id!r}"
                )
        vacuum_ids = {
            entity.material_id
            for entity in self.source.entities
            if entity.material_kind == "vacuum"
        }
        if not vacuum_ids:
            vacuum_ids = {
                key
                for key, value in self.source_materials.items()
                if value["kind"] == "vacuum"
            }
        if len(vacuum_ids) != 1:
            raise ValueError("Q3D source requires exactly one vacuum material identity")
        self.vacuum_id = next(iter(vacuum_ids))
        if (
            self.materials[self.vacuum_id].kind != "vacuum"
            or self.materials[self.vacuum_id].material_id != self.vacuum_id
        ):
            raise ValueError("Q3D vacuum binding differs from source")

    def _lower_bodies(self) -> None:
        self.dbu_um = self.source.metadata["source_dbu_um"]
        self.polygons = {
            polygon.polygon_id: polygon for polygon in self.source.polygons
        }
        self.bodies = []
        self.members = {net_id: [] for net_id in sorted(self.net_ids)}
        for entity in sorted(self.imported, key=lambda value: value.semantic_id):
            zmin, zmax = geometry_z_range(entity.geometry, entity.semantic_id)
            if zmax <= zmin:
                raise ValueError(
                    f"Q3D entity {entity.semantic_id!r} requires positive finite thickness"
                )
            role = (
                "substrate"
                if entity.material_kind == "dielectric"
                else "ground"
                if entity.net_id in self.ground_nets
                else "signal"
            )
            for polygon_id, exterior, holes in _entity_polygons(entity, self.polygons):
                body_id = f"Q3D_BODY_{len(self.bodies) + 1:06d}"
                source_pair = (
                    (entity.geometry["gds_layer"], entity.geometry["gds_datatype"])
                    if "gds_layer" in entity.geometry
                    else None
                )
                self.bodies.append(
                    Q3dBodySpec(
                        body_id=body_id,
                        physical_layer_id=self.layer_bindings.get(entity.semantic_id),
                        exterior_um=exterior,
                        holes_um=holes,
                        z_min_um=zmin,
                        z_max_um=zmax,
                        material_id=entity.material_id,
                        physical_role=role,
                        net_id=entity.net_id
                        if entity.material_kind == "conductor"
                        else None,
                        source_entity_id=entity.semantic_id,
                        source_polygon_id=polygon_id,
                        source_occurrence_path=entity.metadata.get(
                            "source_occurrence_path"
                        ),
                        source_local_entity_id=entity.metadata.get(
                            "source_local_entity_id"
                        ),
                        source_level=entity.metadata.get(
                            "logical_layer_id", entity.metadata.get("pdk_level_id")
                        ),
                        source_layer_datatype=source_pair,
                    )
                )
                if entity.material_kind == "conductor":
                    self.members[cast(str, entity.net_id)].append(body_id)

    def _lower_nets(self) -> None:
        self.bodies_by_entity = {}
        for body in self.bodies:
            self.bodies_by_entity.setdefault(body.source_entity_id, []).append(
                body.body_id
            )

        def terminal_body(entity_id: str) -> str:
            matches = self.bodies_by_entity.get(entity_id, ())
            if not matches:
                raise ValueError(
                    f"Q3D terminal source Entity {entity_id!r} has no emitted body"
                )
            if len(matches) != 1:
                raise ValueError(
                    f"Q3D terminal source Entity {entity_id!r} maps to multiple bodies"
                )
            return matches[0]

        self.nets = []
        for net_id in sorted(self.net_ids):
            pair = self.pairs.get(net_id)
            self.nets.append(
                Q3dNetSpec(
                    name=net_id,
                    net_type=self.net_types[net_id],
                    object_names=tuple(self.members[net_id]),
                    source_object=None
                    if pair is None
                    else terminal_body(pair.source_entity_id),
                    source_side=None if pair is None else pair.source_side,
                    sink_object=None
                    if pair is None
                    else terminal_body(pair.sink_entity_id),
                    sink_side=None if pair is None else pair.sink_side,
                )
            )

    def _write_source_trace(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        original_path = self.directory / "source.gds"
        original_path.write_bytes(self.snapshot.gds_bytes)
        stack_path = self.directory / "geometry_stack.json"
        write_json(stack_path, self.stack)
        source_payload = detached(asdict(self.source))
        # These adapter paths identify temporary snapshot transport, not source
        # semantics. Keep the detached trace portable without changing the snapshot.
        source_payload["metadata"]["gds_file"] = "geometry/source.gds"
        source_payload["metadata"]["stack_file"] = "metadata/geometry_stack.json"
        self.geometry_source = {
            "schema_version": "scgsim.aedt.q3d-geometry-source.v3",
            "source_gds_sha256": hashlib.sha256(self.snapshot.gds_bytes).hexdigest(),
            "source_geometry_sha256": canonical_sha256(source_payload),
            "source_stack_sha256": file_sha256(stack_path),
            "source_dbu_um": self.dbu_um,
            "physical_ground_nets": list(self.ground_nets),
            "bodies": [body.to_payload() for body in self.bodies],
        }
        trace_path = self.directory / "geometry_trace.json"
        write_json(
            trace_path,
            {
                "geometry_source": self.geometry_source,
                "source_geometry": source_payload,
                "source_occurrences": detached(self.snapshot.source_occurrences),
                "body_geometry": "normalized source exterior/holes without fragmentation; no locator sheets",
            },
        )
        self.geometry_source["files"] = {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in (
                ("canonical_gds", original_path),
                ("stack", stack_path),
                ("trace", trace_path),
            )
        }

    def _assemble_spec(self) -> Q3dSpec:
        return Q3dSpec(
            aedt_version=self.aedt_version,
            modeling=self.modeling,
            physical_layers=self.physical_layers,
            physical_representation=self.physical_plan,
            project_name=self.project_name,
            design_name=self.design_name,
            materials=self.materials,
            vacuum_material_id=self.vacuum_id,
            bodies=tuple(self.bodies),
            nets=tuple(self.nets),
            run_control=self.run_control,
            region_padding_um=cast(
                tuple[float, float, float, float, float, float], self.region_padding_um
            ),
            solve_ac_rl=False,
            geometry_source=self.geometry_source,
        )
