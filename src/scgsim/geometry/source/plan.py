"""Compose components into a detached source: place each occurrence with its transform and take the Net membership the caller sets last.

A component's local facts stay as the component declares them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, cast

from scgsim.diagnostics import diagnostic_stage, progress
from scgsim.geometry.grid.grid_cache import with_grid_work_cache
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    SemanticEntitySpec,
    VacuumRegionSpec,
)
from scgsim.geometry.models.junctions import RectangularJunctionSpec
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.refusals import Declared, internal
from scgsim.geometry.source.adapter import (
    LayoutOrigin,
    layout_grid,
    lower_layout,
    stack_records,
)
from scgsim.geometry.source.adapter_records import LoadedLibrary
from scgsim.geometry.source.curves import (
    boundary_binding,
    occurrence_drawn_walks,
    placed_boundary_curves,
    refuse_reconstructions_without_circle,
    transform_boundary,
    with_drawn_rings,
)
from scgsim.geometry.source.junctions import with_rectangular_junctions
from scgsim.geometry.source.normalization import (
    finite_number,
)
from scgsim.geometry.source.occurrence import (
    entity_from_layer_record,
    occurrence_ground_polygons_for_entity,
    occurrence_include_polygon,
    occurrence_lumped_polygon,
    occurrence_polygons_for_entity,
    occurrence_port_polygon,
    read_polygons_by_layer,
    with_occurrence_extraction_cache,
)
from scgsim.geometry.source.occurrence_validation import (
    validate_ground_contribution_ownership,
)
from scgsim.geometry.source.plan_assembly import (
    GeometryPlanStackAssembler,
    component_semantics,
    composed_transform,
    occurrence_transform,
    source_path,
    transform_source_point,
)
from scgsim.geometry.source.recipes import (
    SourceGridAudit,
    failed_source_recipe,
    merged_audits,
    run_source_recipe,
    source_recipe_job,
    source_recipe_map,
)
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack


def _call(method: str, name: str) -> Declared:
    """An argument of a GeometryPlan method, which its caller changes."""
    return Declared(name, f"the call to GeometryPlan.{method}")


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
            raise TypeError(
                internal(
                    "a GeometryPlanSnapshot was built without a GeometryBuildInput"
                )
            )
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

    def occurrence_polygons(self, cell: Any, path: str) -> Mapping[Any, Any]:
        """The polygons an occurrence draws itself, by layer: its cell without its registered children."""
        return read_polygons_by_layer(
            self.source_cells[cell.name],
            excluded_reference_indexes=self.child_reference_indexes[path],
        )


class GeometryPlan:
    """Bind named source occurrences and final Nets before backend preparation."""

    # Immutable default, so a plan assembled without ``__init__`` (tests build them that way) has no priorities.
    _entity_priorities: Mapping[str, int] = MappingProxyType({})

    def __init__(  # noqa: PLR0913 - each argument is a public setting of the plan
        self,
        component: Any,
        *,
        root_id: str,
        layer_stack: Any,
        material_records: Mapping[str, Mapping[str, Any]],
        coupon_padding_um: float,
    ) -> None:
        if not callable(getattr(component, "write_gds", None)):
            raise TypeError(
                _call("__init__", "component").expected(
                    "component", "an assembled GDSFactory Component", component
                )
            )
        self.root_id = source_path(root_id, _call("__init__", "root_id"), "root_id")
        if "/" in self.root_id:
            raise ValueError(
                _call("__init__", "root_id").expected(
                    "root_id", "one path segment", self.root_id
                )
            )
        component_semantics(component, self.root_id)
        self._component = component
        self._layer_stack = layer_stack
        self._material_records = material_records
        self._coupon_padding_um = finite_number(
            coupon_padding_um,
            _call("__init__", "coupon_padding_um"),
            "coupon_padding_um",
        )
        if self._coupon_padding_um < 0:
            raise ValueError(
                _call("__init__", "coupon_padding_um").expected(
                    "coupon_padding_um", "nonnegative", self._coupon_padding_um
                )
            )
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
            raise TypeError(
                _call("set_rectangular_junctions", "rectangular junctions").expected(
                    "every request", "a RectangularJunctionSpec", requests
                )
            )
        for field in ("junction_id", "port_sheet_id"):
            identities = [getattr(item, field) for item in requests]
            if len(set(identities)) != len(identities):
                raise ValueError(
                    _call("set_rectangular_junctions", "rectangular junctions").refusal(
                        f"duplicate rectangular junction {field}"
                    )
                )
        self._rectangular_junctions = requests

    def set_lumped_supports(self, supports: Sequence[LumpedSupport]) -> None:
        """Bind caller-declared local nonmetal supports; no excitation is inferred."""
        records = tuple(supports)
        if any(not isinstance(record, LumpedSupport) for record in records):
            raise TypeError(
                _call("set_lumped_supports", "lumped supports").expected(
                    "every support", "a LumpedSupport record", records
                )
            )
        if len({record.support_id for record in records}) != len(records):
            raise ValueError(
                _call("set_lumped_supports", "lumped supports").refusal(
                    "lumped support IDs must be unique"
                )
            )
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
        path = source_path(path, _call("register_instance", "path"), "path")
        parent_path, _, _ = path.rpartition("/")
        if (
            not parent_path
            or parent_path not in self._instances
            or path in self._instances
        ):
            raise ValueError(
                _call("register_instance", f"instance path {path!r}").refusal(
                    "it needs a registered parent and must be unique"
                )
            )
        parent_component, _, parent_transform = self._instances[parent_path]
        if getattr(reference, "parent_cell", None) != parent_component:
            raise ValueError(
                _call("register_instance", f"reference for {path!r}").refusal(
                    "reference is not an occurrence in its declared parent"
                )
            )
        matches = [
            item
            for item in parent_component.insts
            if item.instance == reference.instance
        ]
        if len(matches) != 1:
            raise ValueError(
                _call("register_instance", f"reference for {path!r}").refusal(
                    "reference must resolve to exactly one parent occurrence"
                )
            )
        if any(
            other_parent == parent_path
            and other_ref is not None
            and other_ref.instance == reference.instance
            for key, (_, other_ref, _) in self._instances.items()
            if (other_parent := key.rpartition("/")[0])
        ):
            raise ValueError(
                _call("register_instance", f"reference for {path!r}").refusal(
                    "the same source occurrence is already registered"
                )
            )
        child = reference.cell
        component_semantics(child, path)
        transform = composed_transform(
            parent_transform, occurrence_transform(reference)
        )
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
            raise TypeError(
                _call("set_nets", "nets").expected(
                    "nets", "a mapping of final Net IDs to qualified Entity IDs", nets
                )
            )
        available = {entity.semantic_id for entity in self.entities}
        owners: dict[str, str] = {}
        prepared: dict[str, tuple[str, ...]] = {}
        for net, members in nets.items():
            if not isinstance(net, str) or not net:
                raise ValueError(
                    _call("set_nets", "nets").expected("Net ID", "nonempty text", net)
                )
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise TypeError(
                    _call("set_nets", f"Net {net!r}").expected(
                        "members", "a sequence of Entity IDs", members
                    )
                )
            unique = tuple(dict.fromkeys(members))
            if not unique:
                raise ValueError(
                    _call("set_nets", f"Net {net!r}").refusal(
                        "it needs at least one Entity ID"
                    )
                )
            for entity_id in unique:
                if not isinstance(entity_id, str) or entity_id not in available:
                    raise ValueError(
                        _call("set_nets", f"Net {net!r}").refusal(
                            f"it has unknown Entity ID {entity_id!r}"
                        )
                    )
                prior = owners.setdefault(entity_id, net)
                if prior != net:
                    raise ValueError(
                        _call("set_nets", f"Entity {entity_id!r}").refusal(
                            "it belongs to two Nets"
                        )
                    )
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
            raise TypeError(
                _call("set_entity_priorities", "priorities").expected(
                    "priorities",
                    "a mapping of qualified Entity IDs to integers",
                    priorities,
                )
            )
        available = {entity.semantic_id for entity in self.entities}
        prepared: dict[str, int] = {}
        for entity_id, value in priorities.items():
            if not isinstance(entity_id, str) or entity_id not in available:
                raise ValueError(
                    _call("set_entity_priorities", "priorities").refusal(
                        f"priority has unknown Entity ID {entity_id!r}"
                    )
                )
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    _call(
                        "set_entity_priorities", f"priority of Entity {entity_id!r}"
                    ).expected("priority", "an integer", value)
                )
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
                or composed_transform(parent_transform, occurrence_transform(reference))
                != transform
            ):
                raise ValueError(
                    Declared(
                        f"registered occurrence {path!r}",
                        "the component, or register the occurrence again with GeometryPlan.register_instance",
                    ).refusal("it changed before prepare")
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
                        Declared(
                            f"authored source occurrence {child.name!r} in {path!r}",
                            "the plan: register it with GeometryPlan.register_instance",
                        ).refusal("it is not registered")
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
        scope = GridScope(source.grid)
        with diagnostic_stage("geometry.source.extract_occurrences"):
            occurrence_audits = self._extract_occurrence_polygons(scope, stack, source)
        with diagnostic_stage("geometry.source.extract_includes"):
            include_audits = self._extract_source_includes(scope, stack, source)
        with diagnostic_stage("geometry.source.extract_ports"):
            port_audits = self._extract_source_ports(scope, stack, source)
        validate_ground_contribution_ownership(scope, stack["layers"])
        build_input = self._lower_stack(
            stack,
            source,
            merged_audits(occurrence_audits, include_audits, port_audits),
        )
        lumped_supports = self._resolve_lumped_supports(scope, build_input, source)
        return self._assemble_snapshot(
            build_input, stack, source, occurrences, lumped_supports
        )

    def _validated_stack(self) -> dict[str, Any]:
        stack = self._stack()
        available = {record["semantic_id"] for record in stack["layers"]}
        for net, members in self._nets.items():
            if not members or any(member not in available for member in members):
                raise ValueError(
                    _call("set_nets", f"Net {net!r}").refusal(
                        "it changed after set_nets; call set_nets again"
                    )
                )
        for entity_id in self._entity_priorities:
            if entity_id not in available:
                raise ValueError(
                    _call(
                        "set_entity_priorities", f"priority of Entity {entity_id!r}"
                    ).refusal(
                        "it changed after set_entity_priorities; call set_entity_priorities again"
                    )
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
        self, scope: GridScope, stack: dict[str, Any], source: _WrittenLayout
    ) -> tuple[SourceGridAudit, ...]:
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
                            internal(
                                f"source occurrence {path!r} is absent from the GDS the plan wrote"
                            )
                        )
                    selector = (
                        occurrence_ground_polygons_for_entity
                        if local_ground
                        else occurrence_polygons_for_entity
                    )
                    job = source_recipe_job(
                        selector,
                        scope,
                        entity_from_layer_record(record, materials=stack["materials"]),
                        source_polygons=source.occurrence_polygons(cell, path),
                        transform=transform,
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

        extracted = source_recipe_map(run_source_recipe, occurrence_jobs())
        for record, value in zip(occurrence_targets, extracted.value, strict=True):
            record["geometry"]["source_occurrence_polygons_um"] = value
        return extracted.audits

    def _extract_source_includes(
        self, scope: GridScope, stack: dict[str, Any], source: _WrittenLayout
    ) -> tuple[SourceGridAudit, ...]:
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
                                internal(
                                    f"ground contribution source {path!r} is absent from the GDS the plan wrote"
                                )
                            )
                        job = source_recipe_job(
                            occurrence_include_polygon,
                            scope,
                            contribution,
                            source_polygons=source.occurrence_polygons(cell, path),
                            transform=transform,
                        )
                    except Exception as error:  # noqa: BLE001 - retain ordered recipe failure results
                        yield failed_source_recipe(error)
                        return
                    include_targets.append(contribution)
                    yield job

        extracted = source_recipe_map(run_source_recipe, include_jobs())
        for contribution, value in zip(include_targets, extracted.value, strict=True):
            contribution["polygon"] = value
        return extracted.audits

    def _extract_source_ports(
        self, scope: GridScope, stack: dict[str, Any], source: _WrittenLayout
    ) -> tuple[SourceGridAudit, ...]:
        port_targets: list[dict[str, Any]] = []

        def port_jobs():
            for record in stack["metadata"]["port_sheet_source_layers"]:
                try:
                    path = record["source_occurrence_path"]
                    cell, _, transform = self._instances[path]
                    if cell.name not in source.source_cells:
                        raise ValueError(
                            internal(
                                f"source port occurrence {path!r} is absent from the GDS the plan wrote"
                            )
                        )
                    job = source_recipe_job(
                        occurrence_port_polygon,
                        scope,
                        record,
                        source_polygons=source.occurrence_polygons(cell, path),
                        transform=transform,
                    )
                except Exception as error:  # noqa: BLE001 - retain ordered recipe failure results
                    yield failed_source_recipe(error)
                    return
                port_targets.append(record)
                yield job

        extracted = source_recipe_map(run_source_recipe, port_jobs())
        for record, value in zip(port_targets, extracted.value, strict=True):
            record["source_occurrence_polygon_um"] = value
        return extracted.audits

    def _lower_stack(
        self,
        stack: dict[str, Any],
        source: _WrittenLayout,
        source_audits: tuple[SourceGridAudit, ...],
    ) -> GeometryBuildInput:
        # The stack is lowered in its JSON form, the form a saved snapshot pair is
        # read back in, so prepare() and build_gds_stack_geometry_input on that
        # pair give the same input.
        saved = json.loads(json.dumps(stack, sort_keys=True))
        with diagnostic_stage("geometry.source.lower_stack"):
            return lower_layout(
                stack_records(saved, None),
                source.library,
                self._component.name,
                LayoutOrigin(source_audits=source_audits),
            )

    def _resolve_lumped_supports(
        self, scope: GridScope, build_input: GeometryBuildInput, source: _WrittenLayout
    ) -> list[LumpedSupport]:
        lumped_supports = []
        entities = {entity.semantic_id: entity for entity in build_input.entities}
        for support in self._lumped_supports:
            path = support.source_occurrence_path
            if path not in self._instances:
                raise ValueError(
                    _call(
                        "set_lumped_supports", f"lumped support {support.support_id!r}"
                    ).refusal(f"its occurrence {path!r} is not registered")
                )
            cell, _, transform = self._instances[path]
            if support.source_level not in self._layer_stack.layers:
                raise ValueError(
                    _call(
                        "set_lumped_supports", f"lumped support {support.support_id!r}"
                    ).refusal(
                        f"its level {support.source_level!r} is not declared in the layer stack"
                    )
                )
            level = self._layer_stack.layers[support.source_level]
            from gdsfactory.pdk import get_layer_tuple  # noqa: PLC0415 - needed only for authored lumped supports

            if get_layer_tuple(level.layer.layer) != support.source_layer:
                raise ValueError(
                    _call(
                        "set_lumped_supports", f"lumped support {support.support_id!r}"
                    ).refusal(
                        "its source level/layer binding differs from the layer stack"
                    )
                )
            source_z_um = float(level.zmin)
            if support.source_z_um is not None and support.source_z_um != source_z_um:
                raise ValueError(
                    _call(
                        "set_lumped_supports", f"lumped support {support.support_id!r}"
                    ).expected(
                        "source_z_um",
                        f"its level's plane {source_z_um}",
                        support.source_z_um,
                    )
                )
            if any(point[2] != source_z_um for point in support.contact_points_um):
                raise ValueError(
                    _call(
                        "set_lumped_supports", f"lumped support {support.support_id!r}"
                    ).refusal(
                        f"its contacts differ from its level's plane z = {source_z_um} um"
                    )
                )
            for entity_id in (
                *support.terminal_a_entity_ids,
                *support.terminal_b_entity_ids,
            ):
                if (
                    entity_id not in entities
                    or entities[entity_id].material_kind != "conductor"
                ):
                    raise ValueError(
                        _call(
                            "set_lumped_supports",
                            f"lumped support {support.support_id!r}",
                        ).refusal(
                            f"its terminal Entity {entity_id!r} is not a conductor"
                        )
                    )
                if entities[entity_id].net_id is None:
                    raise ValueError(
                        _call(
                            "set_lumped_supports",
                            f"lumped support {support.support_id!r}",
                        ).refusal(
                            f"its terminal Entity {entity_id!r} lacks final Net ownership; give it a Net with set_nets"
                        )
                    )
            polygon = occurrence_lumped_polygon(
                scope,
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
        polygons = _polygons_with_owners(build_input)
        boundary_curves, reconstruction = self._placed_boundaries(build_input)
        build_input = replace(
            build_input,
            polygons=polygons,
            metadata=metadata,
            boundary_curves=boundary_curves,
            boundary_reconstruction=reconstruction,
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

    def _placed_boundaries(
        self, build_input: GeometryBuildInput
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """The authored boundary curves and reconstructions, the plan's own then each occurrence's, in layout coordinates."""
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
        return tuple(boundary_curves), tuple(reconstruction)


def _polygons_with_owners(build_input: GeometryBuildInput) -> tuple[Any, ...]:
    """Each polygon an Entity owns, with that Entity and its occurrence path in its metadata."""
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
