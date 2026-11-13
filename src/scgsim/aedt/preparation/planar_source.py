"""Detach source entities, materials, and vacuum-region identity for AEDT."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.data import canonical_sha256
from scgsim.aedt.specs.modeling import native_physical_record
from scgsim.geometry import GeometryBuildInput
from scgsim.semantics.route_a import geometry_z_range


def source_payload(
    build_input: GeometryBuildInput,
    prepared_stack: Mapping[str, Any],
    *,
    source_dbu_um: float,
) -> dict[str, Any]:
    """Detach source geometry and its prepared stack in their canonical payload."""
    return _PlanarSource(build_input, prepared_stack, source_dbu_um).build()


@dataclass
class _PlanarSource:
    """Carry validated source catalogues through detached AEDT serialization."""

    build_input: GeometryBuildInput
    prepared_stack: Mapping[str, Any]
    source_dbu_um: float

    def build(self) -> dict[str, Any]:
        self._validate_stack()
        self._collect_conductors()
        self._collect_solution_regions()
        self._validate_vacuum_identity()
        self._validate_vacuum_envelope()
        self._build_native_region()
        return self._assemble_payload()

    def _validate_stack(self) -> None:
        self.materials = self.prepared_stack.get("materials")
        self.solution_regions = cast(
            Mapping[str, Any], self.prepared_stack.get("solution_regions")
        )
        self.layers = self.prepared_stack.get("layers")
        self.metadata = self.prepared_stack.get("metadata", {})
        if (
            not isinstance(self.materials, Mapping)
            or not isinstance(self.solution_regions, Mapping)
            or isinstance(self.layers, (str, bytes))
            or not isinstance(self.layers, Sequence)
            or not isinstance(self.metadata, Mapping)
        ):
            raise TypeError(
                "prepared_stack must contain materials, solution_regions, layers, and metadata"
            )
        self.material_catalog = plain(self.materials)
        for entity in self.build_input.entities:
            material = self.materials.get(entity.material_id)
            if not isinstance(material, Mapping):
                raise ValueError(
                    f"geometry entity {entity.semantic_id!r} material is absent from prepared_stack"
                )
            if material.get("kind") != entity.material_kind:
                raise ValueError(
                    f"geometry entity {entity.semantic_id!r} material kind contradicts prepared_stack"
                )
        self.source_layers = {
            str(record.get("semantic_id")): record
            for record in self.layers
            if isinstance(record, Mapping)
            and isinstance(record.get("semantic_id"), str)
        }

    def _collect_conductors(self) -> None:
        self.conductors = []
        for item in self.build_input.entities:
            if item.material_kind != "conductor":
                continue
            source_id = str(
                item.metadata.get("source_semantic_id", item.semantic_id)
                if "split_polygon_index" in item.metadata
                else item.semantic_id
            )
            if (
                "split_polygon_index" in item.metadata
                and "source_semantic_id" not in item.metadata
            ):
                source_id = str(
                    item.metadata.get("semantic_group_id", item.semantic_id)
                )
            if source_id not in self.source_layers:
                raise ValueError(
                    f"geometry conductor {item.semantic_id!r} has no prepared-stack source record"
                )
            source_record = self.source_layers[source_id]
            if source_record.get("material_id") != item.material_id:
                raise ValueError(
                    f"geometry conductor {item.semantic_id!r} material contradicts prepared_stack"
                )
            self.conductors.append(
                {
                    "semantic_id": item.semantic_id,
                    "source_semantic_id": source_id,
                    "role": item.role,
                    "material_id": item.material_id,
                    "material_kind": item.material_kind,
                    "part_role": item.part_role,
                    "representation": item.metadata["physical_representation"][
                        "representation"
                    ],
                    "physical_layer": native_physical_record(
                        item.metadata["physical_representation"]
                    ),
                    "net_id": item.net_id,
                    "polygon_ids": list(item.polygon_ids),
                    "geometry": plain(item.geometry),
                    "metadata": {
                        key: plain(item.metadata[key])
                        for key in (
                            "semantic_group_id",
                            "source_semantic_id",
                            "source_occurrence_path",
                            "source_local_entity_id",
                            "split_polygon_index",
                        )
                        if key in item.metadata
                    },
                }
            )

    def _collect_solution_regions(self) -> None:
        self.normalized_regions = []
        for entity in self.build_input.entities:
            if entity.material_kind not in {"vacuum", "dielectric"}:
                continue
            semantic_id = entity.semantic_id
            record = self.solution_regions.get(semantic_id)
            if record is not None and not isinstance(record, Mapping):
                raise TypeError("prepared_stack solution regions must be mappings")
            if (
                record is not None
                and record.get("material_id", semantic_id) != entity.material_id
            ):
                raise ValueError(
                    f"solution region {semantic_id!r} material contradicts normalized geometry"
                )
            generated_auto = bool(entity.metadata.get("is_auto_vacuum_region"))
            geometry = (
                entity.geometry
                if record is None or generated_auto
                else record.get("geometry", record)
            )
            geometry_z_range(geometry, semantic_id)
            self.normalized_regions.append(
                {
                    "semantic_id": semantic_id,
                    "material_id": entity.material_id,
                    "material_kind": entity.material_kind,
                    "representation": "solution_domain",
                    "geometry": plain(geometry),
                    "metadata": plain(entity.metadata),
                }
            )
        missing_regions = set(self.solution_regions) - {
            item["semantic_id"] for item in self.normalized_regions
        }
        if missing_regions:
            raise ValueError(
                f"prepared-stack solution regions lack normalized geometry: {sorted(missing_regions)!r}"
            )
        if any(item["semantic_id"] == "Region" for item in self.normalized_regions):
            raise ValueError(
                "source solution-domain id 'Region' is reserved for native EPR CAD"
            )

    def _validate_vacuum_identity(self) -> None:
        vacuum_regions = [
            item
            for item in self.normalized_regions
            if item["material_kind"] == "vacuum"
        ]
        self.auto_vacuum_regions = [
            item
            for item in vacuum_regions
            if item["metadata"].get("is_auto_vacuum_region")
        ]
        if not self.auto_vacuum_regions:
            raise ValueError("EPR Region requires planner-owned auto vacuum components")
        self.vacuum_material_ids = {
            item["material_id"] for item in self.auto_vacuum_regions
        }
        vacuum_groups = {
            item["metadata"].get("auto_vacuum_group_id")
            for item in self.auto_vacuum_regions
        }
        padding_records = {
            canonical_sha256(item["metadata"].get("vacuum_region_padding_um"))
            for item in self.auto_vacuum_regions
        }
        if (
            len(self.vacuum_material_ids) != 1
            or len(vacuum_groups) != 1
            or None in vacuum_groups
            or len(padding_records) != 1
        ):
            raise ValueError("EPR auto vacuum components disagree on Region identity")

    def _validate_vacuum_envelope(self) -> None:
        self.envelope_trace = self.metadata["physical_representation"]["enclosures"][
            self.auto_vacuum_regions[0]["semantic_id"]
        ]
        padding = self.envelope_trace["effective_padding_um"]
        if not isinstance(padding, Mapping) or set(padding) != {
            "x_plus_um",
            "x_minus_um",
            "y_plus_um",
            "y_minus_um",
            "z_plus_um",
            "z_minus_um",
        }:
            raise ValueError("EPR Region requires exact six-face source padding")
        self.padding_um = [
            float(padding[key])
            for key in (
                "x_plus_um",
                "x_minus_um",
                "y_plus_um",
                "y_minus_um",
                "z_plus_um",
                "z_minus_um",
            )
        ]
        if any(not math.isfinite(value) or value < 0.0 for value in self.padding_um):
            raise ValueError("EPR Region padding must be finite and nonnegative")
        loops = {
            canonical_sha256(item["metadata"].get("auto_vacuum_envelope_outer_loop"))
            for item in self.auto_vacuum_regions
        }
        if len(loops) != 1:
            raise ValueError("EPR auto vacuum components disagree on envelope")
        self.envelope_loop = self.auto_vacuum_regions[0]["metadata"].get(
            "auto_vacuum_envelope_outer_loop"
        )
        if not isinstance(self.envelope_loop, Sequence) or len(self.envelope_loop) < 3:
            raise ValueError("EPR Region requires an auto vacuum envelope loop")
        # Complement slabs omit dielectric and metal occupancy. Native Region
        # encloses those bodies too, so its bounds belong to the complete envelope.
        self.envelope_bounds = self.auto_vacuum_regions[0]["metadata"][
            "auto_vacuum_envelope_bounds_um"
        ]

    def _build_native_region(self) -> None:
        self.native_region = {
            "method": "single_region_absolute_offset.v1",
            "outer_boundary_policy": "hfss_default.v1",
            "name": "Region",
            "material_id": next(iter(self.vacuum_material_ids)),
            "logical_vacuum_ids": sorted(
                item["semantic_id"] for item in self.auto_vacuum_regions
            ),
            "padding_um": self.padding_um,
            **(
                {"source_effective_envelope": plain(self.envelope_trace)}
                if isinstance(self.envelope_trace, Mapping)
                else {}
            ),
            "envelope_outer_loop_um": plain(self.envelope_loop),
            "z_range_um": [
                self.envelope_bounds["z_min_um"],
                self.envelope_bounds["z_max_um"],
            ],
        }

    def _assemble_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "scgsim.aedt.epr-planar-source.v1",
            "modeling": plain(self.metadata.get("physical_representation")),
            "source_dbu_um": self.source_dbu_um,
            "prepared_stack_sha256": canonical_sha256(plain(self.prepared_stack)),
            "source_stack_sha256": self.metadata["physical_source_stack_sha256"],
            "materials": self.material_catalog,
            "solution_regions": self.normalized_regions,
            "native_region": self.native_region,
            "native_entity_name_policy": (
                "readable.v1"
                if self.metadata.get("component_semantics_schema_version") == 2
                else "hash.v1"
            ),
            "surface_evaluation_policy": "unmasked_plus_requested.v1",
            **(
                {
                    "junction_partitions": plain(
                        self.build_input.metadata["aedt_junction_partitions"]
                    ),
                    "authored_input_sha256": self.build_input.metadata[
                        "aedt_authored_input_sha256"
                    ],
                }
                if "aedt_junction_partitions" in self.build_input.metadata
                else {}
            ),
            "conductors": self.conductors,
            "junction_regions": [
                {
                    "port_sheet_id": item.port_sheet_id,
                    "source_polygon_id": item.source_polygon_id,
                    "source_layer": item.source_layer,
                    "exterior": [list(point) for point in item.exterior],
                    "holes": [[list(point) for point in ring] for ring in item.holes],
                    "host_semantic_ids": list(
                        dict.fromkeys(
                            overlap.host_semantic_id for overlap in item.overlaps
                        )
                    ),
                    "metadata": plain(item.metadata),
                }
                for item in self.build_input.port_sheet_regions
            ],
            "polygons": [
                {
                    "polygon_id": item.polygon_id,
                    "layer": item.layer,
                    "exterior": [list(point) for point in item.exterior],
                    "holes": [[list(point) for point in ring] for ring in item.holes],
                    "object_name": item.object_name,
                    "net_name": item.net_name,
                    "port_name": item.port_name,
                }
                for item in self.build_input.polygons
            ],
        }


def plain(value: Any) -> Any:
    """Detach source records into JSON-compatible dictionaries and sequences."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"geometry source contains unsupported {type(value).__name__}")


def native_name(kind: str, *identity: Any) -> str:
    """Return a stable AEDT-safe name while retaining raw ids only as metadata."""

    if not isinstance(kind, str) or not kind.isidentifier():
        raise ValueError("native name kind must be an identifier")
    digest = canonical_sha256({"kind": kind, "identity": list(identity)})[:24]
    return f"scgsim_{kind}_{digest}"


def native_entity_name(source: Mapping[str, Any], kind: str, *identity: Any) -> str:
    """Name a native entity by the source's ``native_entity_name_policy``: ``hash.v1`` hashes the identity, ``readable.v1`` spells it."""

    policy = source.get("native_entity_name_policy", "hash.v1")
    if policy == "hash.v1":
        return native_name(kind, *identity)
    if policy != "readable.v1":
        raise ValueError("unknown native entity name policy")
    raw = "__".join(str(value) for value in identity)
    label = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in raw
    ).strip("_")
    if not label:
        raise ValueError("native entity name has no readable identity")
    digest = canonical_sha256({"kind": kind, "identity": list(identity)})[:10]
    prefix = f"SCGSim_{kind}_"
    label = label[: 60 - len(prefix) - len(digest) - 1].rstrip("_")
    return f"{prefix}{label}_{digest}"


def entity_z_range(entity: Mapping[str, Any]) -> tuple[float, float]:
    """Read the source entity geometry height range."""
    return geometry_z_range(entity.get("geometry"), str(entity.get("semantic_id")))
