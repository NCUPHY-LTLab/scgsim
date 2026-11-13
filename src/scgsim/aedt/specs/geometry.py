"""``PreparedPlanarGeometry``: the detached prepared geometry, carrying its source and model identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.aedt.data import canonical_sha256, detached, freeze
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.aedt.specs.epr import PlanarJunction, SurfaceEprSpec
from scgsim.aedt.specs.modeling import native_physical_record
from scgsim.aedt.specs.support import prepared_lumped_supports
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.physical import PhysicalRepresentationPlan
from scgsim.geometry.models.physical_checks import PLAN_SCHEMA


@dataclass(frozen=True)
class PreparedPlanarGeometry:
    """Detached source request before any native AEDT object exists."""

    modeling: str
    source: Mapping[str, Any]
    junctions: tuple[PlanarJunction, ...]
    contribution_catalog: tuple[Mapping[str, Any], ...]
    contributions: tuple[SurfaceEprSpec, ...]
    surface_bindings: tuple[Mapping[str, Any], ...]
    model_sha256: str
    source_sha256: str
    lumped_rlcs: tuple[LumpedRlc, ...] = ()

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
        if self.modeling not in {"solid", "thin_film"}:
            raise ValueError("planar geometry requires explicit modeling")

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
        if "modeling" in source:
            self._validate_modeling_trace(source)
        if "lumped_supports" in source:
            supports = tuple(
                LumpedSupport.from_payload(item["source"])
                for item in source["lumped_supports"]
            )
            derived = prepared_lumped_supports(supports, source)
            if detached(source["lumped_supports"]) != derived:
                raise ValueError(
                    "serialized effective support geometry differs from its source map"
                )
        return source

    def _validate_modeling_trace(self, source: Mapping[str, Any]) -> None:
        """Check the source's physical representation plan against its conductors.

        Only a tagged plan is read. An untagged trace (a ``PhysicalLayerSpec``
        layer map) was written by an earlier SCGSim and is refused here.
        """
        trace = source["modeling"]
        if not isinstance(trace, Mapping) or trace.get("schema_version") != PLAN_SCHEMA:
            raise ValueError(
                "prepared planar source carries no tagged physical representation "
                f"plan ({PLAN_SCHEMA}); prepare the geometry again with "
                "prepare_planar_geometry_input"
            )
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

    def _validate_support_links(
        self, source: Mapping[str, Any], rlcs: tuple[LumpedRlc, ...]
    ) -> None:
        support_ids = {item["support_id"] for item in source.get("lumped_supports", ())}
        if any(item.support_id not in support_ids for item in rlcs):
            raise ValueError("RLC treatment references an undeclared neutral support")
        if len({item.support_id for item in rlcs}) != len(rlcs):
            raise ValueError("one support cannot own multiple RLC treatments")

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
                "modeling": self.modeling,
                "source": source,
                "junctions": [item.to_payload() for item in self.junctions],
                **self._rlc_payload,
            }
        )
        if self.model_sha256 != expected_model:
            raise ValueError("prepared planar model digest is inconsistent")
        expected = canonical_sha256(
            {
                "modeling": self.modeling,
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
            "modeling": self.modeling,
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

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical PreparedPlanarGeometry payload."""
        return {
            "schema_version": "scgsim.aedt.epr-planar.v2",
            "modeling": self.modeling,
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
        if "modeling" not in value:
            raise ValueError(
                "prepared planar payload has no 'modeling'; an earlier SCGSim wrote "
                "it on a route basis, which this version does not read; prepare the "
                "geometry again with prepare_planar_geometry_input"
            )
        expected = {
            "schema_version",
            "modeling",
            "source",
            "junctions",
            "contribution_catalog",
            "contributions",
            "surface_bindings",
            "model_sha256",
            "source_sha256",
        }
        if "lumped_rlcs" in value:
            expected.add("lumped_rlcs")
        schema = value.get("schema_version")
        if set(value) != expected or schema != "scgsim.aedt.epr-planar.v2":
            raise ValueError("prepared planar payload is not canonical")
        return cls(
            lumped_rlcs=tuple(
                LumpedRlc.from_payload(item) for item in value.get("lumped_rlcs", ())
            ),
            modeling=value["modeling"],
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
        )
