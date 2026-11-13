"""Own detached source composition, occurrence transforms and caller final Net membership. Component local facts remain source authority."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, cast

from scgsim.diagnostics import diagnostic_stage, progress
from scgsim.geometry.grid.grid import geometry_grid_scope, source_recipe_map
from scgsim.geometry.grid.grid_cache import with_grid_work_cache
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    SemanticEntitySpec,
    VacuumRegionSpec,
)
from scgsim.geometry.models.junctions import RectangularJunctionSpec
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.source.adapter import layout_grid, lower_layout, stack_records
from scgsim.geometry.source.adapter_records import LoadedLibrary
from scgsim.geometry.source.curves import (
    boundary_binding,
    occurrence_drawn_walks,
    placed_boundary_curves,
    refuse_reconstructions_without_circle,
    transform_boundary,
    with_drawn_rings,
)
from scgsim.geometry.source.normalization import (
    finite_number,
)
from scgsim.geometry.source.occurrence import (
    entity_from_layer_record,
    failed_source_recipe,
    occurrence_ground_polygons_for_entity,
    occurrence_include_polygon,
    occurrence_lumped_polygon,
    occurrence_polygons_for_entity,
    occurrence_port_polygon,
    run_source_recipe,
    source_recipe_job,
    with_occurrence_extraction_cache,
)
from scgsim.geometry.source.occurrence_validation import (
    validate_ground_contribution_ownership,
)
from scgsim.geometry.source.plan_assembly import (
    GeometryPlanStackAssembler,
    component_semantics,
    source_path,
    transform_source_point,
)
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack


def _affine(reference: Any) -> tuple[float, float, float, float, float, float]:
    from klayout import db as kdb  # noqa: PLC0415 - transforms are needed only during occurrence registration

    transform = reference.dcplx_trans
    origin = transform * kdb.DPoint(0.0, 0.0)
    x_axis = transform * kdb.DPoint(1.0, 0.0)
    y_axis = transform * kdb.DPoint(0.0, 1.0)
    values = (
        float(x_axis.x - origin.x),
        float(y_axis.x - origin.x),
        float(x_axis.y - origin.y),
        float(y_axis.y - origin.y),
        float(origin.x),
        float(origin.y),
    )
    if not all(math.isfinite(item) for item in values):
        raise ValueError("instance transform must be finite")
    a, b, c, d, _, _ = values
    if not math.isclose(abs(a * d - b * c), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("GeometryPlan supports rigid rotation and mirror only")
    return values


def _compose(
    parent: tuple[float, ...], child: tuple[float, ...]
) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, x, y = parent
    e, f, g, h, u, v = child
    return (
        a * e + b * g,
        a * f + b * h,
        c * e + d * g,
        c * f + d * h,
        a * u + b * v + x,
        c * u + d * v + y,
    )


@dataclass(frozen=True)
class GeometryPlanSnapshot:
    """One detached GDS, stack, normalized input, and source-identity pair.

    ``GeometryPlan.prepare()`` builds it: the stack and the normalized input name
    ``gds_sha256``, the digest of ``gds_bytes``, because prepare() wrote all three
    from the one GDS it hashed.
    """

    _geometry_input: GeometryBuildInput
    _stack_json: str
    gds_bytes: bytes
    gds_sha256: str
    _source_occurrences: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self._geometry_input, GeometryBuildInput):
            raise TypeError("snapshot requires GeometryBuildInput")
        object.__setattr__(self, "_geometry_input", copy.deepcopy(self._geometry_input))
        object.__setattr__(
            self, "_source_occurrences", copy.deepcopy(self._source_occurrences)
        )

    @property
    def geometry_input(self) -> GeometryBuildInput:
        """Return a detached normalized input record."""
        return copy.deepcopy(self._geometry_input)

    @property
    def stack(self) -> dict[str, Any]:
        """Return a detached copy of the stack JSON."""
        return json.loads(self._stack_json)

    @property
    def source_occurrences(self) -> tuple[Mapping[str, Any], ...]:
        """Return detached occurrence identity and transform records."""
        return copy.deepcopy(self._source_occurrences)

    @property
    def lumped_supports(self) -> tuple[LumpedSupport, ...]:
        """Return the snapshot's detached lumped-support records."""
        return copy.deepcopy(self._geometry_input.lumped_supports)


@dataclass(frozen=True)
class _WrittenLayout:
    """The GDS prepare() wrote once, read once, and every phase after it works from."""

    gds_bytes: bytes
    gds_sha256: str
    library: LoadedLibrary
    grid: GeometryGridContext
    source_cells: Mapping[str, Any]
    child_reference_indexes: Mapping[str, tuple[int, ...]]


class GeometryPlan:
    """Bind named source occurrences and final Nets before backend preparation."""

    # Immutable default, so a plan assembled without ``__init__`` (tests build them that way) has no priorities.
    _entity_priorities: Mapping[str, int] = MappingProxyType({})

    def __init__(  # noqa: PLR0913 - constructor preserves the established plan contract
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
        self.root_id = source_path(root_id)
        if "/" in self.root_id:
            raise ValueError("root_id must be one path segment")
        component_semantics(component, self.root_id)
        self._component = component
        self._layer_stack = layer_stack
        self._material_records = material_records
        self._coupon_padding_um = finite_number(coupon_padding_um, "coupon_padding_um")
        if self._coupon_padding_um < 0:
            raise ValueError("coupon_padding_um must be nonnegative")
        self._instances: dict[str, tuple[Any, Any, tuple[float, ...]]] = {
            self.root_id: (component, None, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        }
        self._nets: dict[str, tuple[str, ...]] = {}
        self._entity_priorities = MappingProxyType({})
        self._vacuum_region: VacuumRegionSpec | None = None
        self._boundary_curves = ()
        self._boundary_reconstruction = ()
        self._lumped_supports = ()
        self._rectangular_junctions = ()

    def set_rectangular_junctions(self, specs) -> None:
        """Replace explicit source-sheet/terminal bindings for subsequent snapshots."""
        requests = tuple(specs)
        if any(not isinstance(item, RectangularJunctionSpec) for item in requests):
            raise TypeError("rectangular junctions require RectangularJunctionSpec")
        for field in ("junction_id", "port_sheet_id"):
            identities = [getattr(item, field) for item in requests]
            if len(set(identities)) != len(identities):
                raise ValueError(f"duplicate rectangular junction {field}")
        self._rectangular_junctions = requests

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
        """Register one authored child occurrence and its composed transform."""
        path = source_path(path)
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
        component_semantics(child, path)
        transform = _compose(parent_transform, _affine(reference))
        self._instances[path] = (child, reference, transform)

    @property
    def entities(self) -> tuple[SemanticEntitySpec, ...]:
        """Return current source Entities with their final Net ownership."""
        stack = self._stack()
        return tuple(
            entity_from_layer_record(record, materials=stack["materials"])
            for record in stack["layers"]
        )

    def set_nets(self, nets: Mapping[str, Sequence[str]]) -> None:
        """Replace final Net membership after validating all Entity references."""
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

    def set_entity_priorities(self, priorities: Mapping[str, int]) -> None:
        """Replace the assembly-level overlap priority of Entities after validating every reference.

        Where two Entities of one Net overlap, the one with the **higher** priority owns the
        overlap region (equal or missing priorities make the overlap an error that names both
        Entities).  This is the third source of priority, for overlaps between *occurrences*
        (for example two components placed over each other); the other two are the layer stack's
        ``mesh_order`` between layers (there a lower ``mesh_order`` wins, the gdsfactory
        convention) and the component annotation ``priority`` within one component and layer.
        A priority given here overrides what the source declared and is carried into the
        snapshot Entities' ``priority``.  The mapping is replaced as a whole and unchanged
        when any entry is invalid.
        """
        if not isinstance(priorities, Mapping):
            raise TypeError("priorities must map qualified Entity IDs to integers")
        available = {entity.semantic_id for entity in self.entities}
        prepared: dict[str, int] = {}
        for entity_id, value in priorities.items():
            if not isinstance(entity_id, str) or entity_id not in available:
                raise ValueError(f"priority has unknown Entity ID {entity_id!r}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"priority of Entity {entity_id!r} must be an integer")
            prepared[entity_id] = value
        self._entity_priorities = MappingProxyType(prepared)

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
                or _compose(parent_transform, _affine(reference)) != transform
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
        stack = GeometryPlanStackAssembler(
            component=self._component,
            root_id=self.root_id,
            instances=self._instances,
            nets=self._nets,
            layer_stack=self._layer_stack,
            material_records=self._material_records,
            coupon_padding_um=self._coupon_padding_um,
        ).assemble()
        for record in stack["layers"]:
            if record["semantic_id"] in self._entity_priorities:
                record["priority"] = self._entity_priorities[record["semantic_id"]]
        return stack

    @with_grid_work_cache
    @diagnostic_stage("geometry.plan.prepare")
    @with_occurrence_extraction_cache
    def prepare(self) -> GeometryPlanSnapshot:
        """Prepare a detached source snapshot in its required phase order."""
        self._validate_registered_sources()
        stack = self._validated_stack()
        occurrences = self._occurrence_records()
        source = self._write_layout(stack, occurrences)
        with geometry_grid_scope(source.grid):
            with diagnostic_stage("geometry.source.extract_occurrences"):
                self._extract_occurrence_polygons(stack, source)
            with diagnostic_stage("geometry.source.extract_includes"):
                self._extract_source_includes(stack, source)
            with diagnostic_stage("geometry.source.extract_ports"):
                self._extract_source_ports(stack, source)
            validate_ground_contribution_ownership(stack["layers"])
            build_input = self._lower_stack(stack, source)
            lumped_supports = self._resolve_lumped_supports(build_input, source)
            return self._assemble_snapshot(
                build_input, stack, source, occurrences, lumped_supports
            )

    def _validated_stack(self) -> dict[str, Any]:
        stack = self._stack()
        available = {record["semantic_id"] for record in stack["layers"]}
        for net, members in self._nets.items():
            if not members or any(member not in available for member in members):
                raise ValueError(
                    f"Net {net!r} changed after set_nets; call set_nets again"
                )
        for entity_id in self._entity_priorities:
            if entity_id not in available:
                raise ValueError(
                    f"priority of Entity {entity_id!r} changed after set_entity_priorities; "
                    "call set_entity_priorities again"
                )
        if self._vacuum_region is not None:
            stack = dict(apply_vacuum_region_to_stack(stack, self._vacuum_region))
        return stack

    def _occurrence_records(self) -> tuple[Mapping[str, Any], ...]:
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

    def _write_layout(
        self,
        stack: dict[str, Any],
        occurrences: tuple[Mapping[str, Any], ...],
    ) -> _WrittenLayout:
        import gdstk  # noqa: PLC0415 - GDS parsing is needed only during preparation

        with TemporaryDirectory(prefix="scgsim-geometry-plan-") as temporary:
            gds_file = Path(temporary) / "source.gds"
            with diagnostic_stage("geometry.source.write_gds"):
                self._component.write_gds(gds_file)
            gds_bytes = gds_file.read_bytes()
            with diagnostic_stage("geometry.source.read_gds"):
                # read_gds parses Cell objects only; no RawCell is inserted here.
                library = cast("LoadedLibrary", gdstk.read_gds(str(gds_file)))
        gds_sha256 = hashlib.sha256(gds_bytes).hexdigest()
        stack["metadata"]["gds_sha256"] = gds_sha256
        stack["metadata"]["source_occurrences"] = occurrences
        source_cells = {cell.name: cell for cell in library.cells}
        child_reference_indexes = {
            path: tuple(
                row["reference_index"]
                for row in occurrences
                if row["parent_path"] == path
            )
            for path in self._instances
        }
        return _WrittenLayout(
            gds_bytes,
            gds_sha256,
            library,
            layout_grid(library),
            source_cells,
            child_reference_indexes,
        )

    def _extract_occurrence_polygons(
        self, stack: dict[str, Any], source: _WrittenLayout
    ) -> None:
        occurrence_targets: list[dict[str, Any]] = []

        def occurrence_jobs():
            for diagnostic_index, record in enumerate(stack["layers"]):
                progress(diagnostic_index, len(stack["layers"]))
                try:
                    geometry_source = record["geometry"].get(
                        "geometry_source", "gds_polygon"
                    )
                    path = record["metadata"]["source_occurrence_path"]
                    local_ground = (
                        geometry_source == "die_face_minus_ground_mask"
                        and path != self.root_id
                    )
                    if geometry_source != "gds_polygon" and not local_ground:
                        continue
                    cell, _, transform = self._instances[path]
                    if cell.name not in source.source_cells:
                        raise ValueError(
                            f"source occurrence {path!r} is absent from written GDS"
                        )
                    selector = (
                        occurrence_ground_polygons_for_entity
                        if local_ground
                        else occurrence_polygons_for_entity
                    )
                    job = source_recipe_job(
                        selector,
                        entity_from_layer_record(record, materials=stack["materials"]),
                        source_cell=source.source_cells[cell.name],
                        transform=transform,
                        excluded_reference_indexes=source.child_reference_indexes[path],
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
                except Exception as error:  # noqa: BLE001 - retain ordered recipe failure results
                    yield failed_source_recipe(error)
                    return
                occurrence_targets.append(record)
                yield job

        values = source_recipe_map(run_source_recipe, occurrence_jobs())
        for record, value in zip(occurrence_targets, values, strict=True):
            record["geometry"]["source_occurrence_polygons_um"] = value

    def _extract_source_includes(
        self, stack: dict[str, Any], source: _WrittenLayout
    ) -> None:
        include_targets: list[dict[str, Any]] = []

        def include_jobs():
            for record in stack["layers"]:
                for contribution in record["geometry"].get(
                    "source_occurrence_includes", ()
                ):
                    try:
                        path = contribution["source_occurrence_path"]
                        cell, _, transform = self._instances[path]
                        if cell.name not in source.source_cells:
                            raise ValueError(
                                f"ground contribution source {path!r} is absent from GDS"
                            )
                        job = source_recipe_job(
                            occurrence_include_polygon,
                            contribution,
                            source_cell=source.source_cells[cell.name],
                            transform=transform,
                            excluded_reference_indexes=source.child_reference_indexes[
                                path
                            ],
                        )
                    except Exception as error:  # noqa: BLE001 - retain ordered recipe failure results
                        yield failed_source_recipe(error)
                        return
                    include_targets.append(contribution)
                    yield job

        values = source_recipe_map(run_source_recipe, include_jobs())
        for contribution, value in zip(include_targets, values, strict=True):
            contribution["polygon"] = value

    def _extract_source_ports(
        self, stack: dict[str, Any], source: _WrittenLayout
    ) -> None:
        port_targets: list[dict[str, Any]] = []

        def port_jobs():
            for record in stack["metadata"]["port_sheet_source_layers"]:
                try:
                    path = record["source_occurrence_path"]
                    cell, _, transform = self._instances[path]
                    if cell.name not in source.source_cells:
                        raise ValueError(
                            f"source port occurrence {path!r} is absent from written GDS"
                        )
                    job = source_recipe_job(
                        occurrence_port_polygon,
                        record,
                        source_cell=source.source_cells[cell.name],
                        transform=transform,
                        excluded_reference_indexes=source.child_reference_indexes[path],
                    )
                except Exception as error:  # noqa: BLE001 - retain ordered recipe failure results
                    yield failed_source_recipe(error)
                    return
                port_targets.append(record)
                yield job

        values = source_recipe_map(run_source_recipe, port_jobs())
        for record, value in zip(port_targets, values, strict=True):
            record["source_occurrence_polygon_um"] = value

    def _lower_stack(
        self, stack: dict[str, Any], source: _WrittenLayout
    ) -> GeometryBuildInput:
        # The stack is lowered in its JSON form, the form a saved snapshot pair is
        # read back in, so prepare() and build_gds_stack_geometry_input on that
        # pair give the same input.
        saved = json.loads(json.dumps(stack, sort_keys=True))
        with diagnostic_stage("geometry.source.lower_stack"):
            return lower_layout(
                stack_records(saved, None), source.library, self._component.name
            )

    def _resolve_lumped_supports(
        self, build_input: GeometryBuildInput, source: _WrittenLayout
    ) -> list[LumpedSupport]:
        lumped_supports = []
        entities = {entity.semantic_id: entity for entity in build_input.entities}
        for support in self._lumped_supports:
            path = support.source_occurrence_path
            if path not in self._instances:
                raise ValueError(
                    f"lumped support occurrence {path!r} is not registered"
                )
            cell, _, transform = self._instances[path]
            if support.source_level not in self._layer_stack.layers:
                raise ValueError(
                    f"lumped support level {support.source_level!r} is not declared"
                )
            level = self._layer_stack.layers[support.source_level]
            from gdsfactory.pdk import get_layer_tuple  # noqa: PLC0415 - needed only for authored lumped supports

            if get_layer_tuple(level.layer.layer) != support.source_layer:
                raise ValueError("lumped support source level/layer binding differs")
            source_z_um = float(level.zmin)
            if support.source_z_um is not None and support.source_z_um != source_z_um:
                raise ValueError(
                    "lumped support source plane differs from locator Level"
                )
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
            polygon = occurrence_lumped_polygon(
                support,
                cell=source.source_cells[cell.name],
                excluded_reference_indexes=source.child_reference_indexes[path],
                transform=transform,
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

    def _assemble_snapshot(
        self,
        build_input: GeometryBuildInput,
        stack: dict[str, Any],
        source: _WrittenLayout,
        occurrences: tuple[Mapping[str, Any], ...],
        lumped_supports: list[LumpedSupport],
    ) -> GeometryPlanSnapshot:
        # The source remains self-contained after its temporary files disappear.
        metadata = dict(build_input.metadata)
        metadata["gds_sha256"] = source.gds_sha256
        metadata["geometry_plan_root_id"] = self.root_id
        polygon_owners = {
            polygon_id: entity
            for entity in build_input.entities
            for polygon_id in entity.polygon_ids
        }
        polygons = tuple(
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
        boundary_curves = list(self._boundary_curves)
        reconstruction = list(self._boundary_reconstruction)
        for path, (cell, _, transform) in self._instances.items():
            semantics = component_semantics(cell, path)
            boundary_curves.extend(
                placed_boundary_curves(
                    semantics.get("boundary_curves", ()), path, transform, build_input
                )
            )
            reconstruction.extend(
                transform_boundary(
                    value,
                    path=path,
                    transform=transform,
                    point=transform_source_point,
                    reconstruction=True,
                )
                for value in semantics.get("boundary_reconstruction", ())
            )
        build_input = replace(
            build_input,
            polygons=polygons,
            metadata=metadata,
            boundary_curves=tuple(boundary_curves),
            boundary_reconstruction=tuple(reconstruction),
            lumped_supports=tuple(lumped_supports),
        )
        # Reconstruction indices name rings as drawn in the GDS; the source stage has
        # since fused, rounded and noded them, so each polygon keeps its drawn rings.
        drawn_walks = occurrence_drawn_walks(
            self._instances,
            source.source_cells,
            source.child_reference_indexes,
            build_input,
        )
        build_input = with_drawn_rings(build_input, drawn_walks)
        refuse_reconstructions_without_circle(build_input, drawn_walks)
        if self._rectangular_junctions:
            from scgsim.geometry.source.junctions import (  # noqa: PLC0415 - only used when junction requests are present
                with_rectangular_junctions,
            )

            build_input = with_rectangular_junctions(
                build_input, self._rectangular_junctions
            )
        stack = {
            **stack,
            "metadata": {
                **stack["metadata"],
                "geometry_grid": build_input.metadata["geometry_grid"],
                "source_dbu_um": build_input.metadata["source_dbu_um"],
                "unquantized_source_sha256": build_input.metadata[
                    "unquantized_source_sha256"
                ],
            },
        }
        return GeometryPlanSnapshot(
            build_input,
            json.dumps(stack, sort_keys=True),
            source.gds_bytes,
            source.gds_sha256,
            occurrences,
        )
