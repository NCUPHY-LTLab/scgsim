"""Bind source-owned solid cap references and verify their saved native identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.data import canonical_sha256, detached
from scgsim.aedt.epr.fields import create_verified_face_list
from scgsim.aedt.preparation.planar_source import (
    entity_z_range,
    native_name,
)
from scgsim.aedt.runtime.native.planar_bodies import native_conductor_geometry
from scgsim.aedt.runtime.native.saved_geometry import read_saved_hfss_design
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.geometry.planning.spatial import geometry_ref_surface_z_um


def bind_native_surface_references(
    app: Any,
    prepared: PreparedPlanarGeometry,
    native_geometry: Mapping[str, Any],
    request: Any,
) -> list[dict[str, Any]]:
    """Bind source-owned whole solid caps without changing solved topology."""
    if prepared.modeling != "solid":
        return []
    return _SurfaceReferences(app, prepared, native_geometry, request).bind()


@dataclass
class _SurfaceReferences:
    """Keep one request's source bindings, cap observations, and references together."""

    app: Any
    prepared: PreparedPlanarGeometry
    native_geometry: Mapping[str, Any]
    request: Any

    def bind(self) -> list[dict[str, Any]]:
        self.selected = (
            {item.contribution_id for item in self.prepared.contributions}
            if self.request.surface_contribution_ids is None
            else set(self.request.surface_contribution_ids)
        )
        self.source = self.prepared.source
        self.specs = {
            item.contribution_id: item for item in self.prepared.contributions
        }
        self.objects = self.native_geometry["objects"]
        self.references: dict[str, dict[str, Any]] = {}
        for binding in self.prepared.surface_bindings:
            self.binding = binding
            self._bind_surface()
        return detached([self.references[key] for key in sorted(self.references)])

    def _bind_surface(self) -> None:
        self.contribution = self.binding["contribution"]
        self.cid = self.contribution["contribution_id"]
        self.kind = self.contribution["classification"]
        if self.cid not in self.selected or self.kind not in {"MA", "MS", "SA"}:
            return
        self.z_um = geometry_ref_surface_z_um(self.binding["geometry_ref"])
        if self.kind == "SA":
            self.owner = self.binding["substrate_domain_id"]
            entities = [
                item
                for item in self.source["solution_regions"]
                if item["semantic_id"] == self.owner
            ]
            self.candidates = [
                item
                for item in self.objects
                if item["kind"] == "solution_domain"
                and item["semantic_id"] == self.owner
            ]
        else:
            self.owner = self.binding["owner_semantic_id"]
            entities = [
                item
                for item in self.source["conductors"]
                if item["semantic_id"] == self.owner
            ]
            source_ids = {
                self.specs[self.cid].source_polygon_id,
                *(self.contribution.get("outer_source_ids") or ()),
                *(self.contribution.get("hole_source_ids") or ()),
            }
            self.candidates = [
                item
                for item in self.objects
                if item["kind"] == "conductor"
                and item["semantic_id"] == self.owner
                and item["source_polygon_id"] in source_ids
            ]
        if len(entities) != 1 or not self.candidates:
            raise RuntimeError(
                f"native reference {self.cid!r} lacks source-owned Solid {self.owner!r}"
            )
        entity = entities[0]
        if self.kind == "SA":
            z_min, z_max = entity_z_range(entity)
        else:
            representation, z_min, z_max = native_conductor_geometry(
                self.source, entity
            )
            if representation != "solid":
                raise RuntimeError(
                    f"native reference {self.cid!r} owner is not a Solid"
                )
        if self.z_um == z_max and self.z_um != z_min:
            self.cap = "top"
        elif self.z_um == z_min and self.z_um != z_max:
            self.cap = "bottom"
        else:
            raise RuntimeError(
                f"native reference {self.cid!r} is not a source cap of {self.owner!r}"
            )
        for native in self.candidates:
            self.native = native
            self._bind_cap()

    def _bind_cap(self) -> None:
        self.name = self.native["object_name"]
        obj = self.app.modeler.get_object_from_name(self.name)
        if obj is None or self.native["native_object_type"] != "Solid":
            raise RuntimeError(f"native reference Solid {self.name!r} is unavailable")
        centers = []
        for face in obj.faces:
            self.face = face
            self.center = self.face.center_from_aedt
            if self.center is False or self.center is None or len(self.center) != 3:
                raise RuntimeError(
                    f"native reference {self.name!r} face {self.face.id} center unavailable"
                )
            centers.append((float(self.center[2]), self.face, list(self.center)))
        if not centers:
            raise RuntimeError(f"native reference {self.name!r} has no faces")
        extreme = (max if self.cap == "top" else min)(item[0] for item in centers)
        matches = [item for item in centers if item[0] == extreme]
        if len(matches) != 1:
            raise RuntimeError(
                f"native reference {self.name!r} {self.cap} cap is ambiguous"
            )
        _, self.face, self.center = matches[0]
        self.normal = self.face.normal
        if self.normal is None or len(self.normal) != 3:
            raise RuntimeError(
                f"native reference {self.name!r} cap projection unavailable"
            )
        self.normal = [float(value) for value in self.normal]
        identity = {
            "object_name": self.name,
            "cap": self.cap,
            "projection_normal": self.normal,
        }
        self.reference_id = "native_cap_" + canonical_sha256(identity)[:24]
        self.record = self.references.get(self.reference_id)
        if self.record is None:
            self.record = self._create_reference()
        for key, value in (
            ("contribution_ids", self.cid),
            ("binding_ids", self.binding["binding_id"]),
            ("interface_kinds", self.kind),
        ):
            if value not in self.record[key]:
                self.record[key].append(value)
        self.record["source_supports"].append(
            {
                "binding_id": self.binding["binding_id"],
                "geometry_ref": self.binding["geometry_ref"],
            }
        )
        incident = {
            "effective_domain_id": self.binding["effective_domain_id"],
            "effective_material_id": self.binding["effective_material_id"],
            "substrate_domain_id": self.binding["substrate_domain_id"],
        }
        if incident not in self.record["incident_domains"]:
            self.record["incident_domains"].append(incident)

    def _create_reference(self) -> dict[str, Any]:
        self.edges = self._read_edges()
        area = self.face.area
        if area is None or area is False:
            raise RuntimeError(f"native reference {self.name!r} cap area unavailable")
        face_ids = [int(self.face.id)]
        face_list = create_verified_face_list(
            self.app,
            name=native_name("reference", self.reference_id),
            face_ids=face_ids,
            reuse_existing=True,
        )
        self.record = {
            "reference_id": self.reference_id,
            "contribution_ids": [],
            "binding_ids": [],
            "interface_kinds": [],
            "selection_name": face_list["name"],
            "face_ids": face_ids,
            "face_list": face_list,
            "owner_semantic_id": self.owner,
            "source_polygon_id": self.native.get("source_polygon_id"),
            "object_name": self.name,
            "native_object_id": self.native["native_object_id"],
            "source_cap": self.cap,
            "source_z_um": self.z_um,
            "source_supports": [],
            "incident_domains": [],
            "native_normal": self.normal,
            "projection_normal": self.normal,
            "native_normal_basis": "PyAEDT FacePrimitive.normal geometric projection",
            "native_area_m2": float(area) * 1e-12,
            "native_faces": [
                {
                    "face_id": int(self.face.id),
                    "center_um": self.center,
                    "edges": self.edges,
                    "area_um2": float(area),
                }
            ],
            "support_kind": "whole_native_cap",
            "sampling_basis": "native_solid_owner_face; incident_trace_unproven",
        }
        self.references[self.reference_id] = self.record
        return self.record

    def _read_edges(self) -> list[dict[str, Any]]:
        self.edges = []
        for edge in self.face.edges:
            vertices = []
            samples = []
            for vertex in edge.vertices:
                position = vertex.position
                if int(vertex.id) <= 0:
                    # PyAEDT can synthesize an edge-parameter sample when
                    # the native edge has no vertex IDs (e.g. a closed edge).
                    samples.append(
                        {
                            "basis": "PyAEDT synthesized edge-parameter sample",
                            "position_um": list(position)
                            if position is not None
                            else None,
                            "native_vertex_id": None,
                        }
                    )
                    continue
                if position is None or len(position) != 3:
                    raise RuntimeError(
                        f"native reference {self.name!r} vertex {vertex.id} unavailable"
                    )
                vertices.append(
                    {"vertex_id": int(vertex.id), "position_um": list(position)}
                )
            self.edges.append(
                {
                    "edge_id": int(edge.id),
                    "vertices": vertices,
                    "native_vertex_identity_available": bool(vertices) and not samples,
                    "parametric_samples": samples,
                }
            )
        return self.edges


def verify_saved_native_surface_references(
    project_path, design_name, references
) -> list[dict[str, Any]]:
    """Bind reference membership to the native save made before Analyze.

    Created PyAEDT List properties hold the authored assignment. Only the saved
    GeometryEntityListOperation supplies native serialization membership here;
    this function neither saves nor changes the model or the reference support.
    """
    verified = detached(references)
    if not verified:
        return verified
    digest, design = read_saved_hfss_design(
        project_path, design_name, purpose="native surface reference"
    )
    setup = design.get("ModelSetup")
    core = setup.get("GeometryCore") if isinstance(setup, Mapping) else None
    operations = core.get("GeometryOperations") if isinstance(core, Mapping) else None
    lists = (
        operations.get("GeometryEntityLists")
        if isinstance(operations, Mapping)
        else None
    )
    records = (
        lists.get("GeometryEntityListOperation") if isinstance(lists, Mapping) else None
    )
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, list):
        raise RuntimeError(
            "saved native surface reference list records are unavailable"
        )
    for reference in verified:
        requested = reference["face_list"]
        name = requested["name"]
        matching = [
            record
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("Attributes"), Mapping)
            and record["Attributes"].get("Name") == name
        ]
        expected = {
            "name": name,
            "native_id": requested["native_id"],
            "entity_type": "Face",
            "face_ids": requested["face_ids"],
        }
        if len(matching) != 1:
            raise RuntimeError(
                f"saved native surface reference list missing or ambiguous: "
                f"requested={expected!r}; matching_records={matching!r}"
            )
        record = matching[0]
        parameters = record.get("GeometryEntityListParameters")
        actual = {
            "name": record["Attributes"]["Name"],
            "native_id": record.get("ID"),
            "entity_type": parameters.get("EntityType")
            if isinstance(parameters, Mapping)
            else None,
            "face_ids": parameters.get("EntityList")
            if isinstance(parameters, Mapping)
            else None,
        }
        if (
            actual != expected
            or type(actual["native_id"]) is not int
            or not isinstance(actual["face_ids"], list)
            or any(type(value) is not int for value in actual["face_ids"])
        ):
            raise RuntimeError(
                f"saved native surface reference membership differs: "
                f"requested={expected!r}; actual={actual!r}"
            )
        requested["saved_membership"] = {
            "basis": "saved_native_project_geometry_entity_list",
            "project_sha256": digest,
            "design_name": design_name,
            **actual,
        }
    return verified
