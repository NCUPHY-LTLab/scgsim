"""The checks of a physical representation: who declares each part, a saved plan's members and projections, and a face profile's film bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from scgsim.geometry.models.refusals import Declared, internal
from scgsim.semantics.snapshot import to_plain

if TYPE_CHECKING:
    from scgsim.geometry.models.physical import (
        ApproximationProfile,
        PhysicalLayerDeclaration,
        PhysicalRepresentationPlan,
    )

PLAN_SCHEMA = "scgsim.geometry.physical-representation.v1"


# A saved plan is checked against the declarations it records; it is changed by interpreting again.
SAVED_PLAN = Declared(
    "the physical representation plan in the input's metadata",
    "no file: interpret the prepared input again instead of editing its plan",
)


def layer_declared(physical_layer_id: object) -> Declared:
    """A physical layer declaration, which the PDK layer stack declares."""
    return Declared(
        f"physical layer {physical_layer_id!r}",
        "the physical_layer declaration of its PDK layer stack level",
    )


def profile_declared(profile: ApproximationProfile) -> Declared:
    """An approximation profile, which the caller selects."""
    return Declared(
        f"approximation profile {profile.name!r}",
        "the ApproximationProfile passed to the simulation",
    )


def check_saved_plan_members(value: Mapping[str, Any]) -> None:
    """Refuse a saved plan of another schema or with other members."""
    if value.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(
            SAVED_PLAN.expected(
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
            SAVED_PLAN.expected("members", f"exactly {sorted(expected)}", sorted(value))
        )


def check_recorded_projections(
    result: PhysicalRepresentationPlan, value: Mapping[str, Any]
) -> None:
    """Refuse a saved plan whose recorded projections differ from the ones its declarations give."""
    if (
        to_plain(result.collapse_intervals) != value["collapse_intervals"]
        or to_plain(result.sheet_anchors_um) != value["sheet_anchors_um"]
        or to_plain(result.supporting_entities) != value["supporting_entities"]
    ):
        raise ValueError(
            SAVED_PLAN.refusal(
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
                SAVED_PLAN.refusal(
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
                SAVED_PLAN.refusal(
                    "a vacuum enclosure differs from the one its source bounds give"
                )
            )


def face_profile_bindings(profile: ApproximationProfile) -> Mapping[str, Any]:
    """The film bindings of a face profile, which it must declare and which must suit the profile."""
    parameters = profile.parameters
    if set(parameters) != {"film_bindings"} or not isinstance(
        parameters["film_bindings"], Mapping
    ):
        raise ValueError(
            profile_declared(profile).expected(
                "parameters", "exactly film_bindings, a mapping", dict(parameters)
            )
        )
    bindings = parameters["film_bindings"]
    if not bindings:
        raise ValueError(
            profile_declared(profile).expected(
                "film_bindings", "at least one binding", bindings
            )
        )
    if profile.name == "metal_gap_equivalent" and len(bindings) != 2:
        raise ValueError(
            profile_declared(profile).expected(
                "film_bindings", "two bindings", sorted(bindings)
            )
        )
    if profile.name == "substrate_face" and len(bindings) not in {1, 2}:
        raise NotImplementedError(
            profile_declared(profile).expected(
                "film_bindings", "one or two bindings", sorted(bindings)
            )
        )
    if (
        profile.name == "metal_gap_equivalent"
        and len({binding["supporting_entity_id"] for binding in bindings.values()}) != 2
    ):
        raise ValueError(
            profile_declared(profile).refusal(
                "its two film bindings must name two different supporting Entities"
            )
        )
    return bindings


def validate_opposing_films(
    index: Mapping[str, PhysicalLayerDeclaration],
    profile: ApproximationProfile,
    removed: Sequence[tuple[str, float, float]],
    anchors: Mapping[str, float],
    support_planes: Mapping[str, tuple[float, str]],
) -> None:
    """Refuse two films a profile binds that do not face each other across a gap."""
    parameters = profile.parameters
    if profile.name == "metal_gap_equivalent":
        if (
            index[removed[0][0]].source_contact_facts["layer_attachment_face"]
            != "lower"
            or index[removed[1][0]].source_contact_facts["layer_attachment_face"]
            != "upper"
        ):
            raise ValueError(
                profile_declared(profile).refusal(
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
                profile_declared(profile).refusal(
                    "its two films must be bound to the upper face and the lower face of "
                    "two different dielectrics"
                )
            )
        if index[upper].source_interval_um[0] <= index[lower].source_interval_um[1]:
            raise ValueError(
                profile_declared(profile).refusal(
                    f"its films {lower!r} and {upper!r} must leave a gap between them in Z"
                )
            )


def supporting_entity(
    binding: Mapping[str, Any],
    supporting_entities: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """The supporting Entity a film binding names, which must be a declared dielectric."""
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
