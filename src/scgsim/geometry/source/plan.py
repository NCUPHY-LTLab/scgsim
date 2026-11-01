"""Bind named source occurrences and final Nets into a detached geometry snapshot.

Owns: the `GeometryPlan` builder (instances, nets, vacuum region, lumped supports,
boundary curves) and the `GeometryPlanSnapshot` it freezes. Component-local facts
remain the source authority; the plan only places them in the root frame.

Does not own: reading component semantics into a stack (`plan_stack.py`) or
lowering the stack and GDS into geometry (`adapter.py`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    VacuumRegionSpec,
)
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.source._normalization import finite_number
from scgsim.geometry.source._occurrence import (
    entity_from_layer_record,
    occurrence_ground_polygons_for_entity,
    occurrence_include_polygon,
    occurrence_lumped_polygon,
    occurrence_polygons_for_entity,
    occurrence_port_polygon,
)
from scgsim.geometry.source.adapter import build_gds_stack_geometry_input
from scgsim.geometry.source.curves import boundary_binding, transform_boundary
from scgsim.geometry.source.plan_stack import (
    InstanceRecord,
    affine,
    build_plan_stack,
    checked_entity_path,
    compose,
    local_semantics,
    transformed_point,
)
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack

_DERIVED_GROUND = "die_face_minus_ground_mask"


@dataclass(frozen=True)
class GeometryPlanSnapshot:
    """One detached GDS, stack, normalized input, and source-identity pair."""

    _geometry_input: GeometryBuildInput
    _stack_json: str
    gds_bytes: bytes
    gds_sha256: str
    _source_occurrences: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self._geometry_input, GeometryBuildInput):
            raise TypeError("snapshot requires GeometryBuildInput")
        if hashlib.sha256(self.gds_bytes).hexdigest() != self.gds_sha256:
            raise ValueError("snapshot GDS digest differs")
        object.__setattr__(self, "_geometry_input", copy.deepcopy(self._geometry_input))
        object.__setattr__(
            self, "_source_occurrences", copy.deepcopy(self._source_occurrences)
        )
        stack = json.loads(self._stack_json)
        metadata = stack.get("metadata", {})
        if metadata.get("gds_sha256") != self.gds_sha256:
            raise ValueError("snapshot stack GDS identity differs")
        if metadata.get("source_occurrences") != list(self._source_occurrences):
            raise ValueError("snapshot occurrence identity differs")
        if self._geometry_input.metadata.get("gds_sha256") != self.gds_sha256:
            raise ValueError("snapshot normalized-input GDS identity differs")

    @property
    def geometry_input(self) -> GeometryBuildInput:
        """Deep copy of the normalized geometry input, so callers cannot mutate the snapshot."""
        return copy.deepcopy(self._geometry_input)

    @property
    def stack(self) -> dict[str, Any]:
        """Decoded copy of the stored level stack."""
        return json.loads(self._stack_json)

    @property
    def source_occurrences(self) -> tuple[Mapping[str, Any], ...]:
        """Deep copy of the recorded GDS source occurrences."""
        return copy.deepcopy(self._source_occurrences)

    @property
    def lumped_supports(self) -> tuple[LumpedSupport, ...]:
        """Deep copy of the lumped supports in the geometry input."""
        return copy.deepcopy(self._geometry_input.lumped_supports)


class GeometryPlan:
    """Bind named source occurrences and final Nets before backend preparation."""

    def __init__(
        self,
        component: Any,
        *,
        root_id: str,
        layer_stack: Any,
        material_records: Mapping[str, Mapping[str, Any]],
        coupon_padding_um: float,
    ) -> None:
        if not callable(getattr(component, "write_gds", None)):
            raise TypeError("component must be an assembled GDSFactory Component")
        self.root_id = checked_entity_path(root_id)
        if "/" in self.root_id:
            raise ValueError("root_id must be one path segment")
        local_semantics(component, self.root_id)
        self._component = component
        self._layer_stack = layer_stack
        self._material_records = material_records
        self._coupon_padding_um = finite_number(coupon_padding_um, "coupon_padding_um")
        if self._coupon_padding_um < 0:
            raise ValueError("coupon_padding_um must be nonnegative")
        self._instances: dict[str, InstanceRecord] = {
            self.root_id: (component, None, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        }
        self._nets: dict[str, tuple[str, ...]] = {}
        self._vacuum_region: VacuumRegionSpec | None = None
        self._boundary_curves = ()
        self._boundary_reconstruction = ()
        self._lumped_supports = ()

    def set_lumped_supports(self, supports: Sequence[LumpedSupport]) -> None:
        """Bind caller-declared local nonmetal supports; no excitation is inferred."""
        records = tuple(supports)
        if any(not isinstance(record, LumpedSupport) for record in records):
            raise TypeError("lumped supports must be LumpedSupport records")
        if len({record.support_id for record in records}) != len(records):
            raise ValueError("lumped support IDs must be unique")
        self._lumped_supports = records

    def set_boundary_curves(self, boundaries) -> None:
        """Bind explicitly authored curves in assembled source coordinates."""
        prepared = tuple(boundary_binding(value) for value in boundaries)
        self._boundary_curves = prepared

    def set_boundary_reconstruction(self, boundaries) -> None:
        """Select GDS boundary reconstruction explicitly; omitted inputs stay polygonal."""
        prepared = tuple(
            boundary_binding(value, reconstruction=True) for value in boundaries
        )
        self._boundary_reconstruction = prepared

    def set_vacuum_region(
        self, padding: float | Sequence[float] | Mapping[str, Any] = 0.0
    ) -> None:
        """Set one explicit six-face outer envelope for the next snapshot."""
        prepared = VacuumRegionSpec.from_padding(padding)
        self._vacuum_region = prepared

    def register_instance(self, path: str, reference: Any) -> None:
        """Register a placed instance (child cell reference) under its parent's path."""
        path = checked_entity_path(path)
        parent_path, _, _ = path.rpartition("/")
        if (
            not parent_path
            or parent_path not in self._instances
            or path in self._instances
        ):
            raise ValueError(
                "instance path needs a registered parent and must be unique"
            )
        parent_component, _, parent_transform = self._instances[parent_path]
        if getattr(reference, "parent_cell", None) != parent_component:
            raise ValueError("reference is not an occurrence in its declared parent")
        matches = [
            item
            for item in parent_component.insts
            if item.instance == reference.instance
        ]
        if len(matches) != 1:
            raise ValueError("reference must resolve to exactly one parent occurrence")
        if any(
            other_parent == parent_path
            and other_ref is not None
            and other_ref.instance == reference.instance
            for key, (_, other_ref, _) in self._instances.items()
            if (other_parent := key.rpartition("/")[0])
        ):
            raise ValueError("the same source occurrence is already registered")
        child = reference.cell
        local_semantics(child, path)
        transform = compose(parent_transform, affine(reference))
        self._instances[path] = (child, reference, transform)

    @property
    def entities(self) -> tuple[SemanticEntitySpec, ...]:
        """Semantic entities of the stack, in stack layer order."""
        stack = self._stack()
        return tuple(
            entity_from_layer_record(record, materials=stack["materials"])
            for record in stack["layers"]
        )

    def set_nets(self, nets: Mapping[str, Sequence[str]]) -> None:
        """Declare final nets as groups of qualified entity ids; one owner per entity."""
        if not isinstance(nets, Mapping):
            raise TypeError("nets must map final Net IDs to qualified Entity IDs")
        available = {entity.semantic_id for entity in self.entities}
        owners: dict[str, str] = {}
        prepared: dict[str, tuple[str, ...]] = {}
        for net, members in nets.items():
            if not isinstance(net, str) or not net:
                raise ValueError("Net ID must be nonempty text")
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise TypeError(f"Net {net!r} members must be Entity IDs")
            unique = tuple(dict.fromkeys(members))
            if not unique:
                raise ValueError(f"Net {net!r} needs at least one Entity ID")
            for entity_id in unique:
                if not isinstance(entity_id, str) or entity_id not in available:
                    raise ValueError(f"Net {net!r} has unknown Entity ID {entity_id!r}")
                prior = owners.setdefault(entity_id, net)
                if prior != net:
                    raise ValueError(f"Entity {entity_id!r} belongs to two Nets")
            prepared[net] = unique
        self._nets = prepared

    def _validate_registered_sources(self) -> None:
        """Reject authored descendants that would disappear during GDS flattening."""

        for path, (cell, reference, transform) in self._instances.items():
            if reference is None:
                continue
            parent_path = path.rpartition("/")[0]
            parent, _, parent_transform = self._instances[parent_path]
            if (
                reference.parent_cell != parent
                or reference.cell != cell
                or sum(item.instance == reference.instance for item in parent.insts)
                != 1
                or compose(parent_transform, affine(reference)) != transform
            ):
                raise ValueError(
                    f"registered occurrence {path!r} changed before prepare"
                )

        def visit(cell: Any, path: str) -> None:
            registered = (
                (reference.instance, child_path)
                for child_path, (_, reference, _) in self._instances.items()
                if reference is not None and child_path.rpartition("/")[0] == path
            )
            expected = tuple(registered)
            for reference in cell.insts:
                child = reference.cell
                child_path = next(
                    (
                        name
                        for instance, name in expected
                        if instance == reference.instance
                    ),
                    None,
                )
                if child_path is None and "component_semantics" in child.info:
                    raise ValueError(
                        f"authored source occurrence in {path!r} is not registered: "
                        f"{child.name!r}"
                    )
                visit(child, child_path or f"{path}/<unregistered>")

        visit(self._component, self.root_id)

    def _stack(self) -> dict[str, Any]:
        return build_plan_stack(
            component=self._component,
            layer_stack=self._layer_stack,
            material_records=self._material_records,
            coupon_padding_um=self._coupon_padding_um,
            instances=self._instances,
            nets=self._nets,
            root_id=self.root_id,
        )

    def prepare(self) -> GeometryPlanSnapshot:
        """Validate every registered source and freeze the plan into a snapshot."""
        self._validate_registered_sources()
        stack = self._stack()
        self._check_nets_still_valid(stack)
        if self._vacuum_region is not None:
            stack = dict(apply_vacuum_region_to_stack(stack, self._vacuum_region))
        occurrences = self._source_occurrences()
        with TemporaryDirectory(prefix="scgsim-geometry-plan-") as temporary:
            build_input, gds_bytes, gds_sha256, lumped_supports = (
                self._lower_through_files(stack, occurrences, Path(temporary))
            )
        # The source is self-contained after temporary input files disappear.
        metadata = dict(build_input.metadata)
        metadata.pop("gds_file", None)
        metadata.pop("stack_file", None)
        metadata["gds_sha256"] = gds_sha256
        metadata["geometry_plan_root_id"] = self.root_id
        polygons = _polygons_with_owner_metadata(build_input)
        boundary_curves, reconstruction = self._all_boundaries()
        build_input = replace(
            build_input,
            polygons=polygons,
            metadata=metadata,
            boundary_curves=boundary_curves,
            boundary_reconstruction=reconstruction,
            lumped_supports=tuple(lumped_supports),
        )
        return GeometryPlanSnapshot(
            build_input,
            json.dumps(stack, sort_keys=True),
            gds_bytes,
            gds_sha256,
            occurrences,
        )

    def _check_nets_still_valid(self, stack: Mapping[str, Any]) -> None:
        """Nets are declared against a stack; later registrations may invalidate them."""
        available = {record["semantic_id"] for record in stack["layers"]}
        for net, members in self._nets.items():
            if not members or any(member not in available for member in members):
                raise ValueError(
                    f"Net {net!r} changed after set_nets; call set_nets again"
                )

    def _source_occurrences(self) -> tuple[dict[str, Any], ...]:
        """One row per registered instance, sorted by path, with its GDS reference index."""
        return tuple(
            {
                "path": path,
                "parent_path": path.rpartition("/")[0] or None,
                "cell_name": cell.name,
                "transform": list(transform),
                "reference_index": (
                    None
                    if reference is None
                    else next(
                        index
                        for index, item in enumerate(
                            self._instances[path.rpartition("/")[0]][0].insts
                        )
                        if item.instance == reference.instance
                    )
                ),
            }
            for path, (cell, reference, transform) in sorted(self._instances.items())
        )

    def _lower_through_files(
        self,
        stack: dict[str, Any],
        occurrences: tuple[dict[str, Any], ...],
        root: Path,
    ) -> tuple[GeometryBuildInput, bytes, str, list[LumpedSupport]]:
        """Write the GDS, stamp source polygons into `stack`, and lower both.

        The stack is lowered through real files because the adapter re-reads the
        GDS and re-verifies every stamped polygon against it.
        """
        gds_file, stack_file = root / "source.gds", root / "source.stack.json"
        self._component.write_gds(gds_file)
        gds_bytes = gds_file.read_bytes()
        gds_sha256 = hashlib.sha256(gds_bytes).hexdigest()
        stack["metadata"]["gds_sha256"] = gds_sha256
        stack["metadata"]["source_occurrences"] = occurrences
        import gdstk

        source_cells = {cell.name: cell for cell in gdstk.read_gds(str(gds_file)).cells}
        # A cell's own polygons exclude the references that are separate occurrences.
        child_reference_indexes = {
            path: tuple(
                row["reference_index"]
                for row in occurrences
                if row["parent_path"] == path
            )
            for path in self._instances
        }
        self._stamp_layer_polygons(stack, source_cells, child_reference_indexes)
        self._stamp_ground_contribution_polygons(
            stack, source_cells, child_reference_indexes
        )
        self._stamp_port_polygons(stack, source_cells, child_reference_indexes)
        stack_file.write_text(json.dumps(stack, sort_keys=True), encoding="utf-8")
        build_input = build_gds_stack_geometry_input(
            gds_file=gds_file,
            stack_file=stack_file,
            top_cell_name=self._component.name,
        )
        lumped_supports = self._bind_lumped_supports(
            build_input, source_cells, child_reference_indexes
        )
        return build_input, gds_bytes, gds_sha256, lumped_supports

    def _stamp_layer_polygons(
        self,
        stack: dict[str, Any],
        source_cells: Mapping[str, Any],
        child_reference_indexes: Mapping[str, tuple[int, ...]],
    ) -> None:
        """Store each drawn (or non-root derived ground) layer's polygons from the GDS."""
        for record in stack["layers"]:
            geometry_source = record["geometry"].get("geometry_source", "gds_polygon")
            path = record["metadata"]["source_occurrence_path"]
            local_ground = geometry_source == _DERIVED_GROUND and path != self.root_id
            if geometry_source != "gds_polygon" and not local_ground:
                continue
            cell, _, transform = self._instances[path]
            if cell.name not in source_cells:
                raise ValueError(
                    f"source occurrence {path!r} is absent from written GDS"
                )
            selector = (
                occurrence_ground_polygons_for_entity
                if local_ground
                else occurrence_polygons_for_entity
            )
            record["geometry"]["source_occurrence_polygons_um"] = selector(
                entity_from_layer_record(record, materials=stack["materials"]),
                cell=source_cells[cell.name],
                transform=transform,
                excluded_reference_indexes=child_reference_indexes[path],
                **(
                    {
                        "plane_layer": stack["solution_regions"][
                            record["geometry"]["plane_bounds_ref"]
                        ]["metadata"]["source_layer_datatype"]
                    }
                    if local_ground
                    else {}
                ),
            )

    def _stamp_ground_contribution_polygons(
        self,
        stack: dict[str, Any],
        source_cells: Mapping[str, Any],
        child_reference_indexes: Mapping[str, tuple[int, ...]],
    ) -> None:
        for record in stack["layers"]:
            for contribution in record["geometry"].get(
                "source_occurrence_includes", ()
            ):
                path = contribution["source_occurrence_path"]
                cell, _, transform = self._instances[path]
                if cell.name not in source_cells:
                    raise ValueError(
                        f"ground contribution source {path!r} is absent from GDS"
                    )
                contribution["polygon"] = occurrence_include_polygon(
                    contribution,
                    cell=source_cells[cell.name],
                    transform=transform,
                    excluded_reference_indexes=child_reference_indexes[path],
                )

    def _stamp_port_polygons(
        self,
        stack: dict[str, Any],
        source_cells: Mapping[str, Any],
        child_reference_indexes: Mapping[str, tuple[int, ...]],
    ) -> None:
        for record in stack["metadata"]["port_sheet_source_layers"]:
            path = record["source_occurrence_path"]
            cell, _, transform = self._instances[path]
            if cell.name not in source_cells:
                raise ValueError(
                    f"source port occurrence {path!r} is absent from written GDS"
                )
            record["source_occurrence_polygon_um"] = occurrence_port_polygon(
                record,
                cell=source_cells[cell.name],
                transform=transform,
                excluded_reference_indexes=child_reference_indexes[path],
            )

    def _bind_lumped_supports(
        self,
        build_input: GeometryBuildInput,
        source_cells: Mapping[str, Any],
        child_reference_indexes: Mapping[str, tuple[int, ...]],
    ) -> list[LumpedSupport]:
        """Check each declared support against the lowered input and fill its polygon."""
        lumped_supports = []
        entities = {entity.semantic_id: entity for entity in build_input.entities}
        for support in self._lumped_supports:
            path = support.source_occurrence_path
            source_z_um = self._check_lumped_support(support, entities)
            cell, _, transform = self._instances[path]
            polygon = occurrence_lumped_polygon(
                support,
                cell=source_cells[cell.name],
                excluded_reference_indexes=child_reference_indexes[path],
            )
            lumped_supports.append(
                replace(
                    support,
                    exterior=polygon.exterior,
                    holes=polygon.holes,
                    source_transform=transform,
                    source_z_um=source_z_um,
                )
            )
        return lumped_supports

    def _check_lumped_support(
        self, support: LumpedSupport, entities: Mapping[str, SemanticEntitySpec]
    ) -> float:
        """Validate a support against the layer stack; returns its source plane z."""
        path = support.source_occurrence_path
        if path not in self._instances:
            raise ValueError(f"lumped support occurrence {path!r} is not registered")
        if support.source_level not in self._layer_stack.layers:
            raise ValueError(
                f"lumped support level {support.source_level!r} is not declared"
            )
        level = self._layer_stack.layers[support.source_level]
        from gdsfactory.pdk import get_layer_tuple

        if get_layer_tuple(level.layer.layer) != support.source_layer:
            raise ValueError("lumped support source level/layer binding differs")
        source_z_um = float(level.zmin)
        if support.source_z_um is not None and support.source_z_um != source_z_um:
            raise ValueError("lumped support source plane differs from locator Level")
        if any(point[2] != source_z_um for point in support.contact_points_um):
            raise ValueError("lumped support contacts differ from locator plane")
        for entity_id in (
            *support.terminal_a_entity_ids,
            *support.terminal_b_entity_ids,
        ):
            if (
                entity_id not in entities
                or entities[entity_id].material_kind != "conductor"
            ):
                raise ValueError(
                    f"lumped support terminal Entity {entity_id!r} is not a conductor"
                )
            if entities[entity_id].net_id is None:
                raise ValueError(
                    f"lumped support terminal Entity {entity_id!r} lacks final Net ownership"
                )
        return source_z_um

    def _all_boundaries(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Caller-bound curves first, then each cell's own, moved to the root frame."""
        boundary_curves = list(self._boundary_curves)
        reconstruction = list(self._boundary_reconstruction)
        for path, (cell, _, transform) in self._instances.items():
            semantics = local_semantics(cell, path)
            for key, target, reconstruct in (
                ("boundary_curves", boundary_curves, False),
                ("boundary_reconstruction", reconstruction, True),
            ):
                target.extend(
                    transform_boundary(
                        value,
                        path=path,
                        transform=transform,
                        point=transformed_point,
                        reconstruction=reconstruct,
                    )
                    for value in semantics.get(key, ())
                )
        return tuple(boundary_curves), tuple(reconstruction)


def _polygons_with_owner_metadata(
    build_input: GeometryBuildInput,
) -> tuple[LayoutPolygonSpec, ...]:
    """Tag each polygon with the entity (and cell occurrence) that owns it."""
    polygon_owners = {
        polygon_id: entity
        for entity in build_input.entities
        for polygon_id in entity.polygon_ids
    }
    return tuple(
        replace(
            polygon,
            metadata={
                **polygon.metadata,
                **(
                    {
                        "source_entity_id": polygon_owners[
                            polygon.polygon_id
                        ].semantic_id,
                        "source_occurrence_path": polygon_owners[
                            polygon.polygon_id
                        ].metadata["source_occurrence_path"],
                    }
                    if polygon.polygon_id in polygon_owners
                    else {}
                ),
            },
        )
        for polygon in build_input.polygons
    )
