"""Physical-layer declarations and pure source-to-effective AEDT Z-map rules.

Two approximations exist. "solid" keeps every layer at its real thickness.
"thin_film" collapses film layers to zero-thickness sheets, which removes the
film thickness from the Z axis. The Z map in this module moves every height above
a collapsed film down by that film's thickness.
TODO(human): state why Thin Film is used.

Layer identities and film/contact roles belong to the source author. This
module never classifies a layer from a Net, material name or geometric size.
Source acquisition and Entity/stack conversion belong to preparation.modeling;
these pure rules also support canonical request decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from scgsim.geometry.models.physical import (
    ApproximationProfile,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
)
from scgsim.geometry.planning.physical import interpret_physical_layers

Modeling = Literal["solid", "thin_film"]
LayerModelKind = Literal["stack_film", "local_film", "retained_solid"]
ContactSide = Literal["lower", "upper"]


def checked_modeling(value: Any) -> Modeling:
    if value not in {"solid", "thin_film"}:
        raise ValueError("modeling must be explicitly 'solid' or 'thin_film'")
    return value


@dataclass(frozen=True)
class PhysicalLayerSpec:
    """One declared physical level in the source's common Z coordinate system.

    Multiple Entities, occurrences and polygons may bind this same identity.
    Its physical interval is removed once, not once per object. Local films
    declare their actual contact side and never compress the stack.
    source_thickness_um retains an independently authored physical fact;
    endpoint spans remain the coordinate-map authority.

    `kind` is `stack_film` (a film spanning the stack, removed from Z in Thin
    Film), `local_film` (a film only where drawn: it becomes a sheet at its contact
    side without compressing the stack) or `retained_solid` (kept solid).
    `contact_side` says which face of the film the sheet sits on.
    """

    physical_layer_id: str
    kind: LayerModelKind
    z_min_um: float
    z_max_um: float
    contact_side: ContactSide | None = None
    source_thickness_um: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.physical_layer_id, str) or not self.physical_layer_id:
            raise ValueError("physical_layer_id must be explicit nonempty text")
        if self.kind not in {"stack_film", "local_film", "retained_solid"}:
            raise ValueError("physical layer kind is invalid")
        low, high = float(self.z_min_um), float(self.z_max_um)
        if high <= low:
            raise ValueError("a physical layer requires a positive source interval")
        object.__setattr__(self, "z_min_um", low)
        object.__setattr__(self, "z_max_um", high)
        if self.source_thickness_um is not None:
            object.__setattr__(
                self, "source_thickness_um", float(self.source_thickness_um)
            )
        if self.kind in {"stack_film", "local_film"}:
            if self.contact_side not in {"lower", "upper"}:
                raise ValueError("a film requires an explicit sheet contact side")
        elif self.contact_side is not None:
            raise ValueError("a retained solid has no sheet contact side")

    @property
    def physical_thickness_um(self) -> float:
        """Retain an authored thickness; endpoint-only inputs define their own span."""
        return (
            self.source_thickness_um
            if self.source_thickness_um is not None
            else self.z_max_um - self.z_min_um
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize every field, leaving out `source_thickness_um` when it was never
        authored.
        """
        payload = asdict(self)
        if self.source_thickness_um is None:
            payload.pop("source_thickness_um")
        return payload


def native_physical_record(value: Any) -> dict[str, Any]:
    """Project common body facts into native assignment evidence without remapping."""
    from scgsim.aedt._data import detached

    return {
        **detached(value),
        "source_z_min_um": value["source_interval_um"][0],
        "source_z_max_um": value["source_interval_um"][1],
        "effective_z_min_um": value["effective_interval_um"][0],
        "effective_z_max_um": value["effective_interval_um"][1],
    }


def physical_layers_from_payload(payload: Any):
    """Decode the neutral physical-layer declarations of a request payload."""
    if "physical_representation" not in payload:
        raise ValueError(
            "fresh AEDT physical requests require a tagged neutral physical_representation; reprepare the source with PhysicalLayerDeclaration"
        )
    return tuple(
        PhysicalLayerDeclaration(**item) for item in payload.get("physical_layers", ())
    )


def physical_fields_from_payload(payload: Any) -> dict[str, Any]:
    """Constructor fields that decode a payload's physical layers and neutral plan."""
    return {
        "physical_layers": physical_layers_from_payload(payload),
        "physical_representation": PhysicalRepresentationPlan.from_payload(
            payload["physical_representation"]
        ),
    }


def source_plan_layers(source: Any):
    return PhysicalRepresentationPlan.from_payload(source["modeling"]).declarations


def request_modeling(spec: Any):
    """Bind one neutral physical plan to the request."""
    layers = tuple(spec.physical_layers)
    if any(not isinstance(item, PhysicalLayerDeclaration) for item in layers):
        raise TypeError(
            "fresh physical_layers require neutral PhysicalLayerDeclaration; PhysicalLayerSpec is not supported"
        )
    plan = spec.physical_representation
    source = getattr(getattr(spec, "geometry", None), "source", None)
    if plan is None and source is not None:
        plan = PhysicalRepresentationPlan.from_payload(source["modeling"])
    if plan is None:
        plan = interpret_physical_layers(
            layers, ApproximationProfile(spec.modeling, "v1", {})
        )
    if not isinstance(plan, PhysicalRepresentationPlan):
        raise TypeError("physical_representation must be PhysicalRepresentationPlan")
    if plan.profile.name != spec.modeling:
        raise ValueError("request modeling differs from its neutral physical profile")
    if layers and {item.physical_layer_id: item for item in layers} != dict(
        plan.declaration_index
    ):
        raise ValueError(
            "request physical declarations differ from the bound neutral plan"
        )
    object.__setattr__(spec, "physical_layers", plan.declarations)
    object.__setattr__(spec, "physical_representation", plan)
    return plan


def modeling_payload(spec: Any) -> dict[str, Any]:
    return {
        "modeling": spec.modeling,
        "physical_layers": [layer.to_payload() for layer in spec.physical_layers],
        "physical_representation": request_modeling(spec).to_payload(),
    }


def effective_record(
    record: Any, mapping: PhysicalRepresentationPlan, *, retained_domain: bool = False
) -> dict[str, Any]:
    """Source coordinates and physical thickness accompany native geometry."""
    if (
        record.physical_layer_id is None
        and mapping.profile.name == "thin_film"
        and not retained_domain
        and getattr(record, "physical_role", None) != "substrate"
    ):
        raise ValueError("Thin Film source objects require a neutral physical_layer_id")
    body = mapping.map_body(
        record.physical_layer_id,
        (record.z_min_um, record.z_max_um),
        record.to_payload(),
    )
    return {
        **record.to_payload(),
        "physical_layer_id": record.physical_layer_id,
        "physical_thickness_um": body["physical_thickness_um"],
        "representation": body["representation"],
        "effective_z_min_um": body["effective_interval_um"][0],
        "effective_z_max_um": body["effective_interval_um"][1],
    }


def source_record_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Decode source fields; native lowering always comes from the authored map."""
    return {
        key: item
        for key, item in value.items()
        if not (key == "physical_layer_id" and item is None)
        and key
        not in {
            "representation",
            "effective_z_min_um",
            "effective_z_max_um",
            "physical_thickness_um",
            "native_import_layer",
        }
    }


def verify_effective_records(actual, expected) -> None:
    if list(actual) != list(expected):
        raise ValueError(
            "canonical native lowering records differ from authored modeling map"
        )
