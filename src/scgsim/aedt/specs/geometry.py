"""The prepared planar geometry that body-first HFSS requests carry.

`PreparedPlanarGeometry` freezes the normalized layout source together with the
junctions, lumped RLC treatments and surface-EPR requests chosen for it, and
stores two SHA-256 digests. The model digest covers what changes the physics, and
the source digest also covers the EPR requests. Construction recomputes both, so a
payload that was edited after preparation fails to load. Producing it from a
layout is the job of `scgsim.aedt.preparation`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.aedt._data import _freeze_str_keys_only, canonical_sha256, detached
from scgsim.aedt.specs._support import _prepared_lumped_supports
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.aedt.specs.epr import PlanarJunction, SurfaceEprSpec
from scgsim.aedt.specs.modeling import _native_physical_record
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.physical import PLAN_SCHEMA, PhysicalRepresentationPlan


@dataclass(frozen=True)
class PreparedPlanarGeometry:
    """Normalized layout source plus EPR inputs, before any AEDT object exists.

    `modeling` is "solid" or "thin_film". `source` is the frozen layout description (conductors,
    solution regions, native Region, lumped supports). `junctions`,
    `contribution_catalog`, `contributions` and `surface_bindings` are the EPR
    inputs; `lumped_rlcs` are treatments bound to lumped supports.
    """

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
        self._check_modeling_basis()
        rlcs, junctions, contributions = self._freeze_records()
        source = _freeze_str_keys_only(self.source)
        _check_source_matches_plan(source, self.modeling)
        _check_lumped_supports(source)
        support_ids = {item["support_id"] for item in source.get("lumped_supports", ())}
        if any(item.support_id not in support_ids for item in rlcs):
            raise ValueError("RLC treatment references an undeclared neutral support")
        if len({item.support_id for item in rlcs}) != len(rlcs):
            raise ValueError("one support cannot own multiple RLC treatments")
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "surface_bindings",
            tuple(_freeze_str_keys_only(item) for item in self.surface_bindings),
        )
        self._check_digests(source)

    def _check_modeling_basis(self) -> None:
        """Planar geometry must name its modeling explicitly."""
        if self.modeling not in {"solid", "thin_film"}:
            raise ValueError(
                "new planar geometry requires explicit modeling without route alias"
            )

    def _freeze_records(
        self,
    ) -> tuple[
        tuple[LumpedRlc, ...], tuple[PlanarJunction, ...], tuple[SurfaceEprSpec, ...]
    ]:
        """Type-check every record sequence and store each as a tuple."""
        rlcs = tuple(self.lumped_rlcs)
        if any(not isinstance(item, LumpedRlc) for item in rlcs):
            raise TypeError("lumped_rlcs must contain LumpedRlc records")
        object.__setattr__(self, "lumped_rlcs", rlcs)
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
        object.__setattr__(
            self,
            "contribution_catalog",
            tuple(_freeze_str_keys_only(item) for item in catalog),
        )
        object.__setattr__(self, "contributions", contributions)
        return rlcs, junctions, contributions

    def _check_digests(self, source: Mapping[str, Any]) -> None:
        """The model digest covers the physics; the source digest adds EPR requests."""
        model = {
            **self._modeling_identity,
            "source": source,
            "junctions": [item.to_payload() for item in self.junctions],
            **self._rlc_payload,
        }
        if self.model_sha256 != canonical_sha256(model):
            raise ValueError("prepared planar model digest is inconsistent")
        expected = canonical_sha256(
            {
                **model,
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
        return {"modeling": self.modeling}

    def to_payload(self) -> dict[str, Any]:
        """Serialize as `epr-planar.v2`; `lumped_rlcs` appears only when present."""
        return {
            "schema_version": "scgsim.aedt.epr-planar.v2",
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
        """Rebuild from a payload whose member set is exactly one of the canonical shapes;
        digests are re-verified by the constructor.
        """
        if not isinstance(value, Mapping):
            raise TypeError("prepared planar payload must be a mapping")
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
        if schema != "scgsim.aedt.epr-planar.v2":
            raise ValueError(
                f"prepared planar schema {schema!r} is unsupported; "
                "only 'scgsim.aedt.epr-planar.v2' is supported"
            )
        if set(value) != expected:
            raise ValueError("prepared planar payload is not canonical")
        return cls(
            lumped_rlcs=tuple(
                LumpedRlc.from_payload(item) for item in value.get("lumped_rlcs", ())
            ),
            modeling=value.get("modeling"),
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


def _check_source_matches_plan(source: Mapping[str, Any], modeling: str) -> None:
    """When the source carries a neutral physical plan, its conductors must match it."""
    trace = source.get("modeling")
    found = trace.get("schema_version") if isinstance(trace, Mapping) else None
    if found != PLAN_SCHEMA:
        raise ValueError(
            f"prepared source physical plan schema {found!r} is unsupported; "
            f"only {PLAN_SCHEMA!r} is supported"
        )
    plan = PhysicalRepresentationPlan.from_payload(trace)
    if plan.profile.name != modeling:
        raise ValueError("prepared geometry modeling differs from its neutral profile")
    for entity in source["conductors"]:
        expected = _native_physical_record(plan.entities[entity["semantic_id"]])
        if detached(entity["physical_layer"]) != expected:
            raise ValueError(
                "prepared conductor physical evidence differs from the neutral plan"
            )
        if entity["representation"] != expected["representation"]:
            raise ValueError(
                "prepared conductor representation differs from the neutral plan"
            )


def _check_lumped_supports(source: Mapping[str, Any]) -> None:
    """The serialized effective support geometry must re-derive from its source."""
    if "lumped_supports" not in source:
        return
    supports = tuple(
        LumpedSupport.from_payload(item["source"]) for item in source["lumped_supports"]
    )
    if detached(source["lumped_supports"]) != _prepared_lumped_supports(
        supports, source
    ):
        raise ValueError(
            "serialized effective support geometry differs from its source map"
        )
