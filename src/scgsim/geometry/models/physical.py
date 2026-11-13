"""Neutral source declarations and immutable selected-profile interpretation.

Physical dimensions, selectors and attachments belong to the source owner.
Profiles own approximations; neither native tags nor material/role names select
one. A plan's context is a projection of this authority, not another source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

from scgsim.geometry.models.refusals import Declared, internal
from scgsim.geometry.models.representation import (
    EffectiveEnclosure,
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.semantics.route_a import geometry_z_range, same_z
from scgsim.semantics.snapshot import canonical_sha256, freeze_plain, to_plain

FeatureKind = Literal["stack_film", "local_film", "finite_volume"]
ProfileName = Literal[
    "solid", "thin_film", "substrate_face", "metal_gap_equivalent", "route_b"
]
PHYSICAL_LAYER_FIELD = "physical_layer"
PLAN_SCHEMA = "scgsim.geometry.physical-representation.v1"
# A saved plan is checked against the declarations it records; it is changed by interpreting again.
_SAVED_PLAN = Declared(
    "the physical representation plan in the input's metadata",
    "no file: interpret the prepared input again instead of editing its plan",
)


class _MappedInterval(NamedTuple):
    """A body's Z interval under the plan, and how the body is represented there."""

    low: float
    high: float
    representation: str


def _layer_declared(physical_layer_id: object) -> Declared:
    """A physical layer declaration, which the PDK layer stack declares."""
    return Declared(
        f"physical layer {physical_layer_id!r}",
        "the physical_layer declaration of its PDK layer stack level",
    )


def _profile_declared(profile: ApproximationProfile) -> Declared:
    """An approximation profile, which the caller selects."""
    return Declared(
        f"approximation profile {profile.name!r}",
        "the ApproximationProfile passed to the simulation",
    )


@dataclass(frozen=True)
class PhysicalLayerDeclaration:
    """Carry source-owned film or finite-volume identity and physical facts."""

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
                raise ValueError(
                    _layer_declared(self.physical_layer_id).expected(
                        name, "nonempty text", getattr(self, name)
                    )
                )
        if self.feature_kind not in {"stack_film", "local_film", "finite_volume"}:
            raise ValueError(
                _layer_declared(self.physical_layer_id).expected(
                    "feature_kind",
                    "stack_film, local_film or finite_volume",
                    self.feature_kind,
                )
            )
        low, high = self.source_interval_um
        if high <= low:
            raise ValueError(
                _layer_declared(self.physical_layer_id).expected(
                    "source_interval_um", "a positive span", self.source_interval_um
                )
            )
        object.__setattr__(self, "source_interval_um", (float(low), float(high)))
        for name in (
            "source_level_and_gds_pair",
            "physical_dimensions",
            "source_contact_facts",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(
                    _layer_declared(self.physical_layer_id).expected(
                        name, "a mapping", getattr(self, name)
                    )
                )
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
                raise ValueError(
                    _layer_declared(self.physical_layer_id).expected(
                        "layer_attachment_face of a film",
                        "lower or upper",
                        self.source_contact_facts.get("layer_attachment_face"),
                    )
                )
            if "normal_thickness_um" not in self.physical_dimensions:
                raise ValueError(
                    _layer_declared(self.physical_layer_id).refusal(
                        "a film needs its normal_thickness_um in physical_dimensions"
                    )
                )

    @property
    def attachment_z_um(self) -> float:
        """Return the authored lower or upper film attachment coordinate."""
        side = self.source_contact_facts.get("layer_attachment_face")
        if side not in {"lower", "upper"}:
            raise ValueError(
                internal(
                    f"the attachment plane of physical layer {self.physical_layer_id!r} "
                    "was asked for, but it is a finite volume"
                )
            )
        return self.source_interval_um[0 if side == "lower" else 1]

    def to_payload(self) -> dict[str, Any]:
        """Serialize this record using plain canonical containers."""
        return {
            name: to_plain(getattr(self, name)) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> PhysicalLayerDeclaration:
        """Reconstruct a declaration using its source-field validation."""
        return cls(**value)


@dataclass(frozen=True)
class ApproximationProfile:
    """Select one explicit versioned physical approximation and its parameters."""

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
            raise ValueError(
                _profile_declared(self).expected(
                    "name",
                    "solid, thin_film, substrate_face, metal_gap_equivalent or route_b",
                    self.name,
                )
            )
        if self.revision != "v1":
            raise ValueError(
                _profile_declared(self).expected("revision", "v1", self.revision)
            )
        object.__setattr__(self, "parameters", freeze_plain(self.parameters))

    def to_payload(self) -> dict[str, Any]:
        """Serialize this record using plain canonical containers."""
        return {
            "name": self.name,
            "revision": self.revision,
            "parameters": to_plain(self.parameters),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ApproximationProfile:
        """Reconstruct an explicit profile from its serialized fields."""
        return cls(**value)


@dataclass(frozen=True)
class PhysicalRepresentationPlan:
    """Own selected-profile coordinates and source-bound projection records."""

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
        """Index source declarations by physical identity without mutable access."""
        return MappingProxyType(
            {item.physical_layer_id: item for item in self.declarations}
        )

    @property
    def context(self) -> RepresentationContext:
        """Project body and enclosure records for shared geometry planning."""
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
                key: EffectiveEnclosure(**to_plain(value))
                for key, value in self.enclosures.items()
            },
        )

    def map_body(
        self,
        physical_layer_id: str | None,
        source_interval_um: tuple[float, float],
        source_geometry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Map actual source body endpoints; finite levels never replace them.

        The steps: find the body's declaration (its material must be the declared one),
        map its Z interval (a film anchored as a sheet collapses to its anchor), then
        record the body as declared and as mapped.
        """
        low, high = source_interval_um
        declaration = self._body_declaration(physical_layer_id, source_geometry)
        effective_low, effective_high, representation = self._mapped_interval(
            declaration, source_interval_um
        )
        film_declaration = (
            declaration
            if declaration is not None
            and declaration.feature_kind in {"stack_film", "local_film"}
            else None
        )
        physical_thickness = (
            film_declaration.physical_dimensions["normal_thickness_um"]
            if film_declaration is not None
            else source_geometry.get("thickness_um", high - low)
        )
        geometry = to_plain(source_geometry)
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
            "source_geometry": to_plain(source_geometry),
            "effective_geometry": geometry,
            "physical_thickness_um": physical_thickness,
            "physical_thickness_basis": "source_film_normal"
            if film_declaration is not None
            else "source_body_axial",
            "physical_dimensions": to_plain(declaration.physical_dimensions)
            if declaration
            else {},
            "source_contact_facts": to_plain(declaration.source_contact_facts)
            if declaration
            else {},
            "material_id": declaration.material_id if declaration else None,
        }

    def _body_declaration(
        self, physical_layer_id: str | None, source_geometry: Mapping[str, Any]
    ) -> PhysicalLayerDeclaration | None:
        """The declaration a body binds to, whose material the body must have."""
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
                _layer_declared(physical_layer_id).expected(
                    "material of its source body",
                    f"its declared material {declaration.material_id!r}",
                    source_geometry["material_id"],
                )
            )
        return declaration

    def _mapped_interval(
        self,
        declaration: PhysicalLayerDeclaration | None,
        source_interval_um: tuple[float, float],
    ) -> _MappedInterval:
        """A body's Z interval under the plan: a sheet at its film's anchor, or the mapped solid."""
        if (
            declaration is not None
            and declaration.physical_layer_id in self.sheet_anchors_um
        ):
            if tuple(source_interval_um) != declaration.source_interval_um:
                raise ValueError(
                    _layer_declared(declaration.physical_layer_id).expected(
                        "Z interval of its film body",
                        f"its declared interval {declaration.source_interval_um}",
                        tuple(source_interval_um),
                    )
                )
            anchor = self.sheet_anchors_um[declaration.physical_layer_id]
            return _MappedInterval(anchor, anchor, "sheet")
        low, high = source_interval_um
        return _MappedInterval(self.map_z(low), self.map_z(high), "solid")

    def project_enclosure(
        self,
        *,
        effective_body_bounds_um: Mapping[str, float],
        source_padding_um: Mapping[str, float],
        source_body_bounds_um: Mapping[str, float] | None = None,
        source_envelope_bounds_um: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Map the original full envelope once, then derive native offsets."""
        if source_body_bounds_um is not None:
            if source_envelope_bounds_um is not None:
                raise ValueError(
                    internal(
                        "an enclosure was projected from both a body and an envelope"
                    )
                )
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
        elif source_envelope_bounds_um is not None:
            bounds = dict(source_envelope_bounds_um)
        else:
            raise ValueError(
                internal(
                    "an enclosure was projected from neither a body nor an envelope"
                )
            )
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
            "effective_body_bounds_um": to_plain(effective_body_bounds_um),
            "source_padding_um": to_plain(source_padding_um),
            "effective_padding_um": padding,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> PhysicalRepresentationPlan:
        """Rebuild the source interpretation and verify every recorded projection."""
        # Frozen prepared provenance and JSON share one exact plain-container basis.
        value = to_plain(value)
        _check_saved_plan_members(value)
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
        _check_recorded_projections(result, value)
        return replace(
            result, entities=value["entities"], enclosures=value["enclosures"]
        )

    def map_z(self, source_z_um: float) -> float:
        """Remove each declared collapse interval from one source coordinate."""
        value = float(source_z_um)
        return value - sum(
            min(max(value - low, 0.0), high - low)
            for _, low, high in self.collapse_intervals
        )

    def map_point(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Preserve the lateral coordinates and map the source height."""
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
                    Declared(
                        "lumped support contact",
                        "the LumpedSupport passed to GeometryPlan.set_lumped_supports",
                    ).expected(
                        "source plane",
                        f"inside its bound film {contact_physical_layer_id!r}, "
                        f"between {low} and {high} um",
                        z,
                    )
                )
            return x, y, self.sheet_anchors_um[contact_physical_layer_id]
        return self.map_point(point)

    def layer(self, physical_layer_id: str) -> dict[str, Any]:
        """Describe the selected representation of one declared physical layer."""
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
            "physical_dimensions": to_plain(declaration.physical_dimensions),
            "source_contact_facts": to_plain(declaration.source_contact_facts),
            "material_id": declaration.material_id,
        }

    def to_payload(self) -> dict[str, Any]:
        """Serialize this record using plain canonical containers."""
        return {
            "schema_version": PLAN_SCHEMA,
            "source_sha256": self.source_sha256,
            "profile": self.profile.to_payload(),
            "declarations": [item.to_payload() for item in self.declarations],
            "collapse_intervals": to_plain(self.collapse_intervals),
            "sheet_anchors_um": to_plain(self.sheet_anchors_um),
            "entities": to_plain(self.entities),
            "enclosures": to_plain(self.enclosures),
            "supporting_entities": to_plain(self.supporting_entities),
        }

    @property
    def plan_id(self) -> str:
        """Hash the complete canonical serialized interpretation."""
        return canonical_sha256(self.to_payload())


def source_plain(value: Any) -> Any:
    """Convert authored dataclasses and containers to canonical plain source data."""
    if is_dataclass(value):
        return {
            item.name: source_plain(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: source_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [source_plain(item) for item in value]
    return to_plain(value)


def interpret_physical_layers(
    declarations: Sequence[PhysicalLayerDeclaration],
    profile: ApproximationProfile,
    *,
    supporting_entities: Mapping[str, Any] | None = None,
    source_sha256: str | None = None,
) -> PhysicalRepresentationPlan:
    """Interpret declared levels without requiring a qualified contact graph."""
    index = _index_declarations(declarations)
    films = (
        _volume_profile_films(profile, index)
        if profile.name in {"solid", "thin_film", "route_b"}
        else _face_profile_films(profile, index, supporting_entities)
    )
    removed, anchors = films.removed, films.anchors
    supporting_geometry, support_planes = (
        films.supporting_geometry,
        films.support_planes,
    )
    removed.sort(key=lambda item: item[1])
    for lower, upper in zip(removed, removed[1:], strict=False):
        if lower[2] > upper[1]:
            raise ValueError(
                _profile_declared(profile).refusal(
                    f"the films {lower[0]!r} and {upper[0]!r} it collapses overlap in Z"
                )
            )
    _validate_opposing_films(index, profile, removed, anchors, support_planes)
    plan = PhysicalRepresentationPlan(
        source_sha256=source_sha256
        or canonical_sha256([item.to_payload() for item in index.values()]),
        profile=profile,
        declarations=tuple(index.values()),
        collapse_intervals=tuple(removed),
        sheet_anchors_um={},
        supporting_entities=supporting_geometry,
    )
    return replace(
        plan, sheet_anchors_um={key: plan.map_z(z) for key, z in anchors.items()}
    )


def _check_saved_plan_members(value: Mapping[str, Any]) -> None:
    """Refuse a saved plan of another schema or with other members."""
    if value.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(
            _SAVED_PLAN.expected(
                "schema_version", PLAN_SCHEMA, value.get("schema_version")
            )
        )
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
        raise ValueError(
            _SAVED_PLAN.expected(
                "members", f"exactly {sorted(expected)}", sorted(value)
            )
        )


def _check_recorded_projections(
    result: PhysicalRepresentationPlan, value: Mapping[str, Any]
) -> None:
    """Refuse a saved plan whose recorded projections differ from the ones its declarations give."""
    if (
        to_plain(result.collapse_intervals) != value["collapse_intervals"]
        or to_plain(result.sheet_anchors_um) != value["sheet_anchors_um"]
        or to_plain(result.supporting_entities) != value["supporting_entities"]
    ):
        raise ValueError(
            _SAVED_PLAN.refusal(
                "its collapse intervals, sheet anchors or supporting Entities differ "
                "from those its declarations give"
            )
        )
    for item in value["entities"].values():
        expected = result.map_body(
            item["physical_layer_id"],
            tuple(item["source_interval_um"]),
            item["source_geometry"],
        )
        if expected != item:
            raise ValueError(
                _SAVED_PLAN.refusal(
                    f"its mapping of the body on {item['physical_layer_id']!r} differs "
                    "from the one its source interval gives"
                )
            )
    for item in value["enclosures"].values():
        expected = result.project_enclosure(
            effective_body_bounds_um=item["effective_body_bounds_um"],
            source_padding_um=item["source_padding_um"],
            source_envelope_bounds_um=item["source_bounds_um"],
        )
        if expected != item:
            raise ValueError(
                _SAVED_PLAN.refusal(
                    "a vacuum enclosure differs from the one its source bounds give"
                )
            )


class _ProfileFilms(NamedTuple):
    """The films a profile collapses (id, low, high), each film's anchor Z, and the supporting faces it binds."""

    removed: list[tuple[str, float, float]]
    anchors: dict[str, float]
    supporting_geometry: dict[str, Any]
    support_planes: dict[str, tuple[float, str]]


def _volume_profile_films(
    profile: ApproximationProfile, index: Mapping[str, PhysicalLayerDeclaration]
) -> _ProfileFilms:
    """A volume profile's films: none for solid and route_b; every stack film, anchored, for thin_film."""
    parameters = profile.parameters
    removed: list[tuple[str, float, float]] = []
    anchors: dict[str, float] = {}
    if parameters:
        raise ValueError(
            _profile_declared(profile).expected("parameters", "empty", dict(parameters))
        )
    if profile.name == "thin_film":
        removed = [
            (item.physical_layer_id, *item.source_interval_um)
            for item in index.values()
            if item.feature_kind == "stack_film"
        ]
        anchors = {
            item.physical_layer_id: item.attachment_z_um
            for item in index.values()
            if item.feature_kind in {"stack_film", "local_film"}
        }
    return _ProfileFilms(removed, anchors, {}, {})


def _face_profile_films(
    profile: ApproximationProfile,
    index: Mapping[str, PhysicalLayerDeclaration],
    supporting_entities: Mapping[str, Any] | None,
) -> _ProfileFilms:
    """A face profile's films: each bound film anchored at its attachment, on its supporting face."""
    removed: list[tuple[str, float, float]] = []
    anchors: dict[str, float] = {}
    supporting_geometry: dict[str, Any] = {}
    support_planes: dict[str, tuple[float, str]] = {}
    bindings = _face_profile_bindings(profile)
    for physical_id, binding in bindings.items():
        item = index[physical_id]
        if item.feature_kind != "stack_film":
            raise ValueError(
                _profile_declared(profile).expected(
                    f"film binding {physical_id!r}",
                    "a stack film",
                    item.feature_kind,
                )
            )
        if set(binding) != {"supporting_entity_id", "supporting_side"}:
            raise ValueError(
                _profile_declared(profile).expected(
                    f"film binding {physical_id!r}",
                    "exactly supporting_entity_id and supporting_side",
                    sorted(binding),
                )
            )
        host = _supporting_entity(binding, supporting_entities)
        geometry = host["geometry"]
        supporting_geometry[binding["supporting_entity_id"]] = source_plain(host)
        bounds = geometry_z_range(geometry, binding["supporting_entity_id"])
        side = binding["supporting_side"]
        if side not in {"lower", "upper"}:
            raise ValueError(
                _profile_declared(profile).expected(
                    f"film binding {physical_id!r} supporting_side",
                    "lower or upper",
                    side,
                )
            )
        plane = bounds[0 if side == "lower" else 1]
        if not same_z(plane, item.attachment_z_um):
            raise ValueError(
                _profile_declared(profile).expected(
                    f"film binding {physical_id!r} supporting face",
                    f"the film's attachment plane z = {item.attachment_z_um} um",
                    plane,
                )
            )
        support_planes[physical_id] = (plane, side)
        anchors[physical_id] = item.attachment_z_um
        if profile.name == "metal_gap_equivalent":
            removed.append((physical_id, *item.source_interval_um))
    return _ProfileFilms(removed, anchors, supporting_geometry, support_planes)


def _face_profile_bindings(profile: ApproximationProfile) -> Mapping[str, Any]:
    parameters = profile.parameters
    if set(parameters) != {"film_bindings"} or not isinstance(
        parameters["film_bindings"], Mapping
    ):
        raise ValueError(
            _profile_declared(profile).expected(
                "parameters", "exactly film_bindings, a mapping", dict(parameters)
            )
        )
    bindings = parameters["film_bindings"]
    if not bindings:
        raise ValueError(
            _profile_declared(profile).expected(
                "film_bindings", "at least one binding", bindings
            )
        )
    if profile.name == "metal_gap_equivalent" and len(bindings) != 2:
        raise ValueError(
            _profile_declared(profile).expected(
                "film_bindings", "two bindings", sorted(bindings)
            )
        )
    if profile.name == "substrate_face" and len(bindings) not in {1, 2}:
        raise NotImplementedError(
            _profile_declared(profile).expected(
                "film_bindings", "one or two bindings", sorted(bindings)
            )
        )
    if (
        profile.name == "metal_gap_equivalent"
        and len({binding["supporting_entity_id"] for binding in bindings.values()}) != 2
    ):
        raise ValueError(
            _profile_declared(profile).refusal(
                "its two film bindings must name two different supporting Entities"
            )
        )
    return bindings


def _validate_opposing_films(
    index: Mapping[str, PhysicalLayerDeclaration],
    profile: ApproximationProfile,
    removed: Sequence[tuple[str, float, float]],
    anchors: Mapping[str, float],
    support_planes: Mapping[str, tuple[float, str]],
) -> None:
    parameters = profile.parameters
    if profile.name == "metal_gap_equivalent":
        if (
            index[removed[0][0]].source_contact_facts["layer_attachment_face"]
            != "lower"
            or index[removed[1][0]].source_contact_facts["layer_attachment_face"]
            != "upper"
        ):
            raise ValueError(
                _profile_declared(profile).refusal(
                    "its two films must attach one at its lower and one at its upper face"
                )
            )
    if len(anchors) == 2 and profile.name in {"substrate_face", "metal_gap_equivalent"}:
        lower, upper = sorted(anchors, key=lambda physical_id: anchors[physical_id])
        lower_host = parameters["film_bindings"][lower]["supporting_entity_id"]
        upper_host = parameters["film_bindings"][upper]["supporting_entity_id"]
        if (
            lower_host == upper_host
            or support_planes[lower][1] != "upper"
            or support_planes[upper][1] != "lower"
        ):
            raise ValueError(
                _profile_declared(profile).refusal(
                    "its two films must be bound to the upper face and the lower face of "
                    "two different dielectrics"
                )
            )
        if index[upper].source_interval_um[0] <= index[lower].source_interval_um[1]:
            raise ValueError(
                _profile_declared(profile).refusal(
                    f"its films {lower!r} and {upper!r} must leave a gap between them in Z"
                )
            )


def _supporting_entity(
    binding: Mapping[str, Any],
    supporting_entities: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if (
        supporting_entities is None
        or binding["supporting_entity_id"] not in supporting_entities
    ):
        raise ValueError(
            Declared(
                "film binding", "the ApproximationProfile passed to the simulation"
            ).expected(
                "supporting_entity_id",
                "an Entity of the input",
                binding["supporting_entity_id"],
            )
        )
    host = supporting_entities[binding["supporting_entity_id"]]
    required = {
        "geometry",
        "material_id",
        "material_kind",
        "material_properties",
    }
    if not isinstance(host, Mapping) or not required <= set(host):
        raise ValueError(
            internal(
                f"supporting Entity {binding['supporting_entity_id']!r} was handed over "
                "without its geometry, material and material properties"
            )
        )
    if host["material_kind"] != "dielectric":
        raise ValueError(
            Declared(
                "film binding", "the ApproximationProfile passed to the simulation"
            ).expected(
                f"supporting Entity {binding['supporting_entity_id']!r}",
                "a dielectric",
                host["material_kind"],
            )
        )
    if not host["material_id"] or not isinstance(host["material_properties"], Mapping):
        raise ValueError(
            Declared(
                f"material of supporting Entity {binding['supporting_entity_id']!r}",
                "the stack's materials",
            ).refusal("it needs a material id and a mapping of its properties")
        )
    return host


def _index_declarations(
    declarations: Sequence[PhysicalLayerDeclaration],
) -> dict[str, PhysicalLayerDeclaration]:
    index = {}
    for item in declarations:
        if not isinstance(item, PhysicalLayerDeclaration):
            raise TypeError(
                internal("a physical layer declaration was handed over untyped")
            )
        previous = index.setdefault(item.physical_layer_id, item)
        if previous != item:
            raise ValueError(
                _layer_declared(item.physical_layer_id).refusal(
                    "it is declared twice, differently"
                )
            )
    return index
