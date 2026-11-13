"""The Surface-EPR specs a simulation holds: one normalized film per interface type, with its band margins.

``MA`` is the metal-air interface, ``MS`` metal-substrate and ``SA`` substrate-air.  ``set_surface_epr`` builds
``SurfaceEprSpecs.normalized`` from what the user passes; the rules are those of
``scgsim.geometry.surface_epr.specs``, which Palace and AEDT share.  ``mesh`` cuts the margins into the model
as bands and ``config`` writes one Palace ``Dielectric`` row per spec and margin.  ``to_payload()`` gives the
normalized mapping back key for key, as the geometry build reads it and the index map records it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.refusals import internal
from scgsim.geometry.surface_epr.specs import margins_of, normalize_surface_epr_specs


@dataclass(frozen=True)
class SurfaceEprSpec:
    """The film of one interface type and the margins (µm) of its bands, as ``set_surface_epr`` normalized them.

    Exactly one of ``thickness`` (in units of Palace ``Model.L0``) and ``film_thickness_m`` is set.  An unknown
    ``loss_tangent`` stays None; Palace gets 0.  ``material_id`` names the dielectric the permittivity was read
    from, if any.  ``face_kinds`` restricts the spec to some faces of its type.  ``inset_margins_um`` applies to
    every face of the type, ``inset_margins_by_entity_um`` to chosen conductors; None means the key was not given.
    """

    interface_type: str
    thickness: float | None
    film_thickness_m: float | None
    permittivity: float
    loss_tangent: float | None
    source: Any
    preset: Any
    required: bool
    material_id: str | None
    face_kinds: tuple[str, ...] | None
    inset_margins_um: tuple[float, ...] | None
    inset_margins_by_entity_um: Mapping[str, tuple[float, ...]] | None

    def to_payload(self) -> dict[str, Any]:
        """The normalized spec as a mapping, in the key order the shared normalization writes."""
        payload: dict[str, Any] = (
            {"thickness": self.thickness}
            if self.thickness is not None
            else {"film_thickness_m": self.film_thickness_m}
        )
        payload.update(
            permittivity=self.permittivity,
            loss_tangent=self.loss_tangent,
            source=self.source,
            preset=self.preset,
            required=self.required,
        )
        if self.material_id is not None:
            payload["material_id"] = self.material_id
        if self.face_kinds is not None:
            payload["face_kinds"] = list(self.face_kinds)
        if self.inset_margins_um is not None:
            payload["inset_margins_um"] = list(self.inset_margins_um)
        if self.inset_margins_by_entity_um is not None:
            payload["inset_margins_by_entity_um"] = {
                entity: list(margins)
                for entity, margins in self.inset_margins_by_entity_um.items()
            }
        return payload

    def native_thickness(self, model_l0_m: float) -> float:
        """The film thickness in units of Palace ``Model.L0`` (``model_l0_m`` metres)."""
        if self.film_thickness_m is not None:
            return self.film_thickness_m / float(model_l0_m)
        if self.thickness is None:
            raise ValueError(
                internal(
                    f"Surface EPR {self.interface_type} has no thickness: build it with SurfaceEprSpecs.normalized(), which requires one"
                )
            )
        return self.thickness

    def margins_for(self, entities: Iterable[str]) -> list[float]:
        """The inset margins (µm) of a face of ``entities``: a chosen conductor's own, else the type's."""
        return margins_of(self.to_payload(), entities)


@dataclass(frozen=True)
class SurfaceEprSpecs:
    """A simulation's Surface-EPR specs by interface type, in type order; empty asks for no film and no band."""

    by_type: Mapping[str, SurfaceEprSpec]

    @classmethod
    def normalized(
        cls,
        specs: Mapping[str, Mapping[str, Any]],
        *,
        materials: Mapping[str, Mapping[str, Any]],
    ) -> SurfaceEprSpecs:
        """Check ``specs`` against the stack's ``materials`` and hold them normalized.

        Raises:
            TypeError: ``specs`` or a spec is not a mapping, or a field has the wrong type.
            ValueError: an interface type other than MA, MS, SA, an unknown field, or an invalid value.
        """
        normalized = normalize_surface_epr_specs(specs, materials=materials)
        return cls(
            {
                interface_type: _spec(interface_type, payload)
                for interface_type, payload in normalized.items()
            }
        )

    def to_payload(self) -> dict[str, dict[str, Any]]:
        """The normalized specs as ``{interface_type: spec mapping}``, as the geometry build reads them."""
        return {
            interface_type: spec.to_payload()
            for interface_type, spec in self.by_type.items()
        }


def _spec(interface_type: str, payload: Mapping[str, Any]) -> SurfaceEprSpec:
    """One normalized spec mapping as its record."""
    by_entity = payload.get("inset_margins_by_entity_um")
    margins = payload.get("inset_margins_um")
    face_kinds = payload.get("face_kinds")
    return SurfaceEprSpec(
        interface_type=interface_type,
        thickness=payload.get("thickness"),
        film_thickness_m=payload.get("film_thickness_m"),
        permittivity=payload["permittivity"],
        loss_tangent=payload["loss_tangent"],
        source=payload["source"],
        preset=payload["preset"],
        required=payload["required"],
        material_id=payload.get("material_id"),
        face_kinds=None if face_kinds is None else tuple(face_kinds),
        inset_margins_um=None if margins is None else tuple(margins),
        inset_margins_by_entity_um=None
        if by_entity is None
        else {entity: tuple(values) for entity, values in by_entity.items()},
    )


__all__ = ["SurfaceEprSpec", "SurfaceEprSpecs"]
