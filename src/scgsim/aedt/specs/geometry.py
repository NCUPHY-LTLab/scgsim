"""Unique detached prepared geometry carrier; source/model identity is preserved."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from scgsim.aedt.data import canonical_sha256, detached, freeze
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.aedt.specs.epr import PlanarJunction, SurfaceEprSpec
from scgsim.aedt.specs.modeling import native_physical_record
from scgsim.geometry.models.physical import PLAN_SCHEMA, PhysicalRepresentationPlan

Route = Literal["A", "B"]

NATIVE_LUMPED_REFUSAL = (
    "LumpedRlc and LumpedTerminalPort name a port sheet, which only the SGB model "
    "(prepare_sgb_model) realizes; the native planar builder has none (issue #67 decision 8)"
)


@dataclass(frozen=True)
class PreparedPlanarGeometry:
    """Detached source request before any native AEDT object exists."""

    modeling: str | None
    source: Mapping[str, Any]
    junctions: tuple[PlanarJunction, ...]
    contribution_catalog: tuple[Mapping[str, Any], ...]
    contributions: tuple[SurfaceEprSpec, ...]
    surface_bindings: tuple[Mapping[str, Any], ...]
    model_sha256: str
    source_sha256: str
    route: Route | None = None
    _historical_modeling: bool = field(default=False, repr=False, compare=False)
    _legacy_payload: bool = field(default=False, repr=False, compare=False)
    lumped_rlcs: tuple[LumpedRlc, ...] = ()

    @property
    def legacy_payload(self) -> bool:
        """Report whether this geometry came from a legacy payload schema."""
        return self._legacy_payload

    def __post_init__(self) -> None:
        self._validate_modeling_identity()
        rlcs = self._validated_lumped_rlcs()
        junctions, catalog, contributions, bindings = self._validated_records()
        source = self._validated_source()
        self._validate_support_links(source, rlcs)
        self._store_normalized_records(
            source, junctions, catalog, contributions, bindings
        )
        self._validate_digests(source)

    def _validate_modeling_identity(self) -> None:
        if self._historical_modeling:
            if self.route not in {"A", "B"} or self.modeling is not None:
                raise ValueError(
                    "historical planar geometry requires its original route basis"
                )
        elif self.modeling not in {"solid", "thin_film"} or self.route is not None:
            raise ValueError(
                "new planar geometry requires explicit modeling without route alias"
            )

    def _validated_lumped_rlcs(self) -> tuple[LumpedRlc, ...]:
        rlcs = tuple(self.lumped_rlcs)
        if any(not isinstance(item, LumpedRlc) for item in rlcs):
            raise TypeError("lumped_rlcs must contain LumpedRlc records")
        object.__setattr__(self, "lumped_rlcs", rlcs)

        return rlcs

    def _validated_records(
        self,
    ) -> tuple[
        tuple[PlanarJunction, ...],
        tuple[Mapping[str, Any], ...],
        tuple[SurfaceEprSpec, ...],
        tuple[Mapping[str, Any], ...],
    ]:
        junctions = tuple(self.junctions)
        catalog = tuple(self.contribution_catalog)
        contributions = tuple(self.contributions)
        bindings = tuple(self.surface_bindings)
        if any(not isinstance(item, PlanarJunction) for item in junctions):
            raise TypeError("junctions must contain PlanarJunction records")
        if any(not isinstance(item, SurfaceEprSpec) for item in contributions):
            raise TypeError("contributions must contain SurfaceEprSpec records")
        if any(not isinstance(item, Mapping) for item in catalog):
            raise TypeError("contribution_catalog must contain mappings")
        if any(not isinstance(item, Mapping) for item in bindings):
            raise TypeError("surface_bindings must contain mappings")
        object.__setattr__(self, "junctions", junctions)
        catalog = tuple(freeze(item) for item in catalog)
        object.__setattr__(self, "contribution_catalog", catalog)
        object.__setattr__(self, "contributions", contributions)

        return junctions, catalog, contributions, bindings

    def _validated_source(self) -> Mapping[str, Any]:
        source = freeze(self.source)
        trace = source.get("modeling")
        if isinstance(trace, Mapping) and trace.get("schema_version") == PLAN_SCHEMA:
            plan = PhysicalRepresentationPlan.from_payload(trace)
            if plan.profile.name != self.modeling:
                raise ValueError(
                    "prepared geometry modeling differs from its neutral profile"
                )
            for entity in source["conductors"]:
                expected = native_physical_record(plan.entities[entity["semantic_id"]])
                if detached(entity["physical_layer"]) != expected:
                    raise ValueError(
                        "prepared conductor physical evidence differs from the neutral plan"
                    )
                if entity["representation"] != expected["representation"]:
                    raise ValueError(
                        "prepared conductor representation differs from the neutral plan"
                    )
        if "lumped_supports" in source:
            raise ValueError(
                "lumped supports were removed by issue #67 decision 8; prepare the model again"
            )
        return source

    def _validate_support_links(
        self, source: Mapping[str, Any], rlcs: tuple[LumpedRlc, ...]
    ) -> None:
        if rlcs:
            raise ValueError(NATIVE_LUMPED_REFUSAL)

    def _store_normalized_records(
        self,
        source: Mapping[str, Any],
        junctions: tuple[PlanarJunction, ...],
        catalog: tuple[Mapping[str, Any], ...],
        contributions: tuple[SurfaceEprSpec, ...],
        bindings: tuple[Mapping[str, Any], ...],
    ) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "junctions", junctions)
        object.__setattr__(self, "contribution_catalog", catalog)
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(
            self,
            "surface_bindings",
            tuple(freeze(item) for item in bindings),
        )

    def _validate_digests(self, source: Mapping[str, Any]) -> None:
        expected_model = canonical_sha256(
            {
                **self._modeling_identity,
                "source": source,
                "junctions": [item.to_payload() for item in self.junctions],
                **self._rlc_payload,
            }
        )
        if self.model_sha256 != expected_model:
            raise ValueError("prepared planar model digest is inconsistent")
        expected = canonical_sha256(
            {
                **self._modeling_identity,
                "source": source,
                "junctions": [item.to_payload() for item in self.junctions],
                **self._rlc_payload,
                "contribution_catalog": self.contribution_catalog,
                "contributions": [item.to_payload() for item in self.contributions],
                "surface_bindings": self.surface_bindings,
            }
        )
        if self.source_sha256 != expected:
            raise ValueError("prepared planar source digest is inconsistent")

    @property
    def _rlc_payload(self) -> dict[str, Any]:
        return (
            {"lumped_rlcs": [item.to_payload() for item in self.lumped_rlcs]}
            if self.lumped_rlcs
            else {}
        )

    def with_lumped_rlcs(
        self, treatments: Sequence[LumpedRlc]
    ) -> PreparedPlanarGeometry:
        """Bind treatment before execution; saved analysis retains this model identity."""
        records = tuple(treatments)
        if any(not isinstance(item, LumpedRlc) for item in records):
            raise TypeError("treatments must contain LumpedRlc records")
        model = {
            **self._modeling_identity,
            "source": self.source,
            "junctions": [item.to_payload() for item in self.junctions],
            **(
                {"lumped_rlcs": [item.to_payload() for item in records]}
                if records
                else {}
            ),
        }
        identity = {
            **model,
            "contribution_catalog": self.contribution_catalog,
            "contributions": [item.to_payload() for item in self.contributions],
            "surface_bindings": self.surface_bindings,
        }
        return replace(
            self,
            lumped_rlcs=records,
            model_sha256=canonical_sha256(model),
            source_sha256=canonical_sha256(identity),
        )

    @property
    def _modeling_identity(self) -> dict[str, Any]:
        return (
            {"route": self.route}
            if self._historical_modeling
            else {"modeling": self.modeling}
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical PreparedPlanarGeometry payload."""
        return {
            "schema_version": (
                "scgsim.aedt.epr-planar.v1"
                if self._legacy_payload
                else "scgsim.aedt.epr-planar.v2"
            ),
            **self._modeling_identity,
            "source": detached(self.source),
            "junctions": [item.to_payload() for item in self.junctions],
            **self._rlc_payload,
            "contribution_catalog": [
                detached(item) for item in self.contribution_catalog
            ],
            "contributions": [item.to_payload() for item in self.contributions],
            "surface_bindings": [detached(item) for item in self.surface_bindings],
            "model_sha256": self.model_sha256,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_payload(cls, value: Any) -> PreparedPlanarGeometry:
        """Decode a PreparedPlanarGeometry from its canonical payload."""
        if not isinstance(value, Mapping):
            raise TypeError("prepared planar payload must be a mapping")
        expected = {
            "schema_version",
            "route",
            "source",
            "junctions",
            "contribution_catalog",
            "contributions",
            "surface_bindings",
            "model_sha256",
            "source_sha256",
        }
        if "modeling" in value:
            expected.remove("route")
            expected.add("modeling")
        if "lumped_rlcs" in value:
            expected.add("lumped_rlcs")
        schema = value.get("schema_version")
        if set(value) != expected or schema not in {
            "scgsim.aedt.epr-planar.v1",
            "scgsim.aedt.epr-planar.v2",
        }:
            raise ValueError("prepared planar payload is not canonical")
        return cls(
            lumped_rlcs=tuple(
                LumpedRlc.from_payload(item) for item in value.get("lumped_rlcs", ())
            ),
            modeling=value.get("modeling"),
            route=value.get("route"),
            _historical_modeling="modeling" not in value,
            source=value["source"],
            junctions=tuple(
                PlanarJunction.from_payload(item) for item in value["junctions"]
            ),
            contribution_catalog=tuple(value["contribution_catalog"]),
            contributions=tuple(
                SurfaceEprSpec.from_payload(item) for item in value["contributions"]
            ),
            surface_bindings=tuple(value["surface_bindings"]),
            model_sha256=value["model_sha256"],
            source_sha256=value["source_sha256"],
            _legacy_payload=schema.endswith(".v1"),
        )
