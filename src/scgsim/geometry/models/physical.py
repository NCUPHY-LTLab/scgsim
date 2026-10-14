"""Neutral source declarations and immutable selected-profile interpretation.

Physical dimensions, selectors and attachments belong to the source owner.
Profiles own approximations; neither native tags nor material/role names select
one. A plan's context is a projection of this authority, not another source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from scgsim.geometry.models.representation import (
    EntityRepresentation,
    EffectiveEnclosure,
    RepresentationContext,
)
from scgsim.semantics.snapshot import _plain, canonical_sha256, freeze_plain

FeatureKind = Literal["stack_film", "local_film", "finite_volume"]
ProfileName = Literal[
    "solid", "thin_film", "substrate_face", "metal_gap_equivalent", "route_b"
]
PHYSICAL_LAYER_FIELD = "physical_layer"
PLAN_SCHEMA = "scgsim.geometry.physical-representation.v1"


@dataclass(frozen=True)
class PhysicalLayerDeclaration:
    physical_layer_id: str
    source_level_id: str
    source_level_and_gds_pair: Mapping[str, Any]
    source_interval_um: tuple[float, float]
    material_id: str
    feature_kind: FeatureKind
    physical_dimensions: Mapping[str, Any]
    source_contact_facts: Mapping[str, Any]
    actual_entity_contact_graph: Any = None

    def __post_init__(self) -> None:
        for name in ("physical_layer_id", "source_level_id", "material_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be authored nonempty text")
        if self.feature_kind not in {"stack_film", "local_film", "finite_volume"}:
            raise ValueError("unsupported physical feature kind")
        low, high = self.source_interval_um
        if high <= low:
            raise ValueError("physical source interval must have positive span")
        object.__setattr__(self, "source_interval_um", (float(low), float(high)))
        for name in (
            "source_level_and_gds_pair",
            "physical_dimensions",
            "source_contact_facts",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a source mapping")
            object.__setattr__(self, name, freeze_plain(getattr(self, name)))
        object.__setattr__(
            self,
            "actual_entity_contact_graph",
            freeze_plain(self.actual_entity_contact_graph),
        )
        if self.feature_kind != "finite_volume":
            if self.source_contact_facts.get("layer_attachment_face") not in {
                "lower",
                "upper",
            }:
                raise ValueError("a physical film needs its authored attachment face")
            if "normal_thickness_um" not in self.physical_dimensions:
                raise ValueError("a physical film needs its original normal thickness")

    @property
    def attachment_z_um(self) -> float:
        side = self.source_contact_facts.get("layer_attachment_face")
        if side not in {"lower", "upper"}:
            raise ValueError(
                f"physical layer {self.physical_layer_id!r} has no attachment face"
            )
        return self.source_interval_um[0 if side == "lower" else 1]

    def to_payload(self) -> dict[str, Any]:
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> PhysicalLayerDeclaration:
        return cls(**value)


@dataclass(frozen=True)
class ApproximationProfile:
    name: ProfileName
    revision: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in {
            "solid",
            "thin_film",
            "substrate_face",
            "metal_gap_equivalent",
            "route_b",
        }:
            raise ValueError("unsupported physical approximation profile")
        if self.revision != "v1":
            raise ValueError("unsupported physical profile revision")
        object.__setattr__(self, "parameters", freeze_plain(self.parameters))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "parameters": _plain(self.parameters),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ApproximationProfile:
        return cls(**value)


@dataclass(frozen=True)
class PhysicalRepresentationPlan:
    source_sha256: str
    profile: ApproximationProfile
    declarations: tuple[PhysicalLayerDeclaration, ...]
    collapse_intervals: tuple[tuple[str, float, float], ...]
    sheet_anchors_um: Mapping[str, float]
    entities: Mapping[str, Any] = field(default_factory=dict)
    enclosures: Mapping[str, Any] = field(default_factory=dict)
    supporting_entities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "declarations", tuple(self.declarations))
        object.__setattr__(
            self,
            "collapse_intervals",
            tuple(tuple(item) for item in self.collapse_intervals),
        )
        for name in (
            "sheet_anchors_um",
            "entities",
            "enclosures",
            "supporting_entities",
        ):
            object.__setattr__(self, name, freeze_plain(getattr(self, name)))

    @property
    def declaration_index(self) -> Mapping[str, PhysicalLayerDeclaration]:
        from types import MappingProxyType

        return MappingProxyType(
            {item.physical_layer_id: item for item in self.declarations}
        )

    @property
    def context(self) -> RepresentationContext:
        return RepresentationContext(
            entities={
                key: EntityRepresentation(
                    physical_layer_id=value["physical_layer_id"],
                    representation=value["representation"],
                    source_z_interval_um=tuple(value["source_interval_um"]),
                    effective_z_interval_um=tuple(value["effective_interval_um"]),
                )
                for key, value in self.entities.items()
            },
            enclosures={
                key: EffectiveEnclosure(**_plain(value))
                for key, value in self.enclosures.items()
            },
        )

    def map_body(
        self,
        physical_layer_id: str | None,
        source_interval_um: tuple[float, float],
        source_geometry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Map actual source body endpoints; finite levels never replace them."""
        low, high = source_interval_um
        declaration = (
            self.declaration_index[physical_layer_id]
            if physical_layer_id is not None
            else None
        )
        if (
            declaration is not None
            and "material_id" in source_geometry
            and source_geometry["material_id"] != declaration.material_id
        ):
            raise ValueError(
                "source body material differs from its declared physical layer"
            )
        if physical_layer_id in self.sheet_anchors_um:
            if tuple(source_interval_um) != declaration.source_interval_um:
                raise ValueError(
                    f"film body interval differs from {physical_layer_id!r}"
                )
            effective_low = effective_high = self.sheet_anchors_um[physical_layer_id]
            representation = "sheet"
        else:
            effective_low, effective_high = self.map_z(low), self.map_z(high)
            representation = "solid"
        film = declaration is not None and declaration.feature_kind in {
            "stack_film",
            "local_film",
        }
        physical_thickness = (
            declaration.physical_dimensions["normal_thickness_um"]
            if film
            else source_geometry.get("thickness_um", high - low)
        )
        geometry = _plain(source_geometry)
        geometry.update(z_min_um=effective_low, z_max_um=effective_high)
        if "z_um" in geometry:
            geometry["z_um"] = effective_low
        if "thickness_um" in geometry:
            geometry["thickness_um"] = effective_high - effective_low
        return {
            "physical_layer_id": physical_layer_id,
            "representation": representation,
            "source_interval_um": [low, high],
            "effective_interval_um": [effective_low, effective_high],
            "source_geometry": _plain(source_geometry),
            "effective_geometry": geometry,
            "physical_thickness_um": physical_thickness,
            "physical_thickness_basis": "source_film_normal"
            if film
            else "source_body_axial",
            "physical_dimensions": _plain(declaration.physical_dimensions)
            if declaration
            else {},
            "source_contact_facts": _plain(declaration.source_contact_facts)
            if declaration
            else {},
            "material_id": declaration.material_id if declaration else None,
        }

    def project_enclosure(
        self,
        *,
        effective_body_bounds_um: Mapping[str, float],
        source_padding_um: Mapping[str, float],
        source_body_bounds_um: Mapping[str, float] | None = None,
        source_envelope_bounds_um: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Map the original full envelope once, then derive native offsets."""
        if (source_body_bounds_um is None) == (source_envelope_bounds_um is None):
            raise ValueError(
                "provide exactly one original body or full-envelope bounds record"
            )
        if source_body_bounds_um is not None:
            bounds = {
                f"{axis}_{edge}_um": source_body_bounds_um[f"{axis}_{edge}_um"]
                + (
                    -source_padding_um[f"{axis}_minus_um"]
                    if edge == "min"
                    else source_padding_um[f"{axis}_plus_um"]
                )
                for axis in ("x", "y", "z")
                for edge in ("min", "max")
            }
        else:
            bounds = dict(source_envelope_bounds_um)
        effective = {
            **bounds,
            "z_min_um": self.map_z(bounds["z_min_um"]),
            "z_max_um": self.map_z(bounds["z_max_um"]),
        }
        padding = {
            f"{axis}_{side}_um": (
                effective_body_bounds_um[f"{axis}_min_um"] - effective[f"{axis}_min_um"]
                if side == "minus"
                else effective[f"{axis}_max_um"]
                - effective_body_bounds_um[f"{axis}_max_um"]
            )
            for axis in ("x", "y", "z")
            for side in ("minus", "plus")
        }
        return {
            "source_bounds_um": bounds,
            "effective_bounds_um": effective,
            "effective_body_bounds_um": _plain(effective_body_bounds_um),
            "source_padding_um": _plain(source_padding_um),
            "effective_padding_um": padding,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> PhysicalRepresentationPlan:
        # Frozen prepared provenance and JSON share one exact plain-container basis.
        value = _plain(value)
        if value.get("schema_version") != PLAN_SCHEMA:
            raise ValueError("unsupported physical representation plan schema")
        expected = {
            "schema_version",
            "source_sha256",
            "profile",
            "declarations",
            "collapse_intervals",
            "sheet_anchors_um",
            "entities",
            "enclosures",
            "supporting_entities",
        }
        if set(value) != expected:
            raise ValueError("physical representation plan members are not canonical")
        from scgsim.geometry.planning.physical import interpret_physical_layers

        profile = ApproximationProfile.from_payload(value["profile"])
        declarations = tuple(
            PhysicalLayerDeclaration.from_payload(item)
            for item in value["declarations"]
        )
        result = interpret_physical_layers(
            declarations,
            profile,
            supporting_entities=value["supporting_entities"],
            source_sha256=value["source_sha256"],
        )
        if (
            _plain(result.collapse_intervals) != value["collapse_intervals"]
            or _plain(result.sheet_anchors_um) != value["sheet_anchors_um"]
            or _plain(result.supporting_entities) != value["supporting_entities"]
        ):
            raise ValueError(
                "serialized profile coordinates differ from their source declarations"
            )
        for item in value["entities"].values():
            expected = result.map_body(
                item["physical_layer_id"],
                tuple(item["source_interval_um"]),
                item["source_geometry"],
            )
            if expected != item:
                raise ValueError(
                    "serialized body projection differs from its actual source interval"
                )
        for item in value["enclosures"].values():
            expected = result.project_enclosure(
                effective_body_bounds_um=item["effective_body_bounds_um"],
                source_padding_um=item["source_padding_um"],
                source_envelope_bounds_um=item["source_bounds_um"],
            )
            if expected != item:
                raise ValueError(
                    "serialized enclosure differs from its original mapped bounds"
                )
        from dataclasses import replace

        return replace(
            result, entities=value["entities"], enclosures=value["enclosures"]
        )

    def map_z(self, source_z_um: float) -> float:
        value = float(source_z_um)
        return value - sum(
            min(max(value - low, 0.0), high - low)
            for _, low, high in self.collapse_intervals
        )

    def map_point(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        x, y, z = point
        return x, y, self.map_z(z)

    def project_support_point(
        self, point: tuple[float, float, float], contact_physical_layer_id: str
    ) -> tuple[float, float, float]:
        """Project an explicitly layer-bound support, not an unrelated point."""
        declaration = self.declaration_index[contact_physical_layer_id]
        x, y, z = point
        if contact_physical_layer_id in self.sheet_anchors_um:
            low, high = declaration.source_interval_um
            if not low <= z <= high:
                raise ValueError(
                    "support source plane is outside its explicitly bound physical film"
                )
            return x, y, self.sheet_anchors_um[contact_physical_layer_id]
        return self.map_point(point)

    def layer(self, physical_layer_id: str) -> dict[str, Any]:
        declaration = next(
            item
            for item in self.declarations
            if item.physical_layer_id == physical_layer_id
        )
        if physical_layer_id in self.sheet_anchors_um:
            low = high = self.sheet_anchors_um[physical_layer_id]
            representation = "sheet"
        else:
            low, high = (self.map_z(z) for z in declaration.source_interval_um)
            representation = "solid"
        return {
            "physical_layer_id": physical_layer_id,
            "representation": representation,
            "source_interval_um": list(declaration.source_interval_um),
            "effective_interval_um": [low, high],
            "physical_dimensions": _plain(declaration.physical_dimensions),
            "source_contact_facts": _plain(declaration.source_contact_facts),
            "material_id": declaration.material_id,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "source_sha256": self.source_sha256,
            "profile": self.profile.to_payload(),
            "declarations": [item.to_payload() for item in self.declarations],
            "collapse_intervals": _plain(self.collapse_intervals),
            "sheet_anchors_um": _plain(self.sheet_anchors_um),
            "entities": _plain(self.entities),
            "enclosures": _plain(self.enclosures),
            "supporting_entities": _plain(self.supporting_entities),
        }

    @property
    def plan_id(self) -> str:
        return canonical_sha256(self.to_payload())
