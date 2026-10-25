"""Detached EPR results; declaration and saved-solution names directly re-export specs."""

from __future__ import annotations


import math

from collections.abc import Mapping, Sequence

from dataclasses import dataclass, field



from typing import Any, Literal

from scgsim.semantics.epr import film_assumptions

from scgsim.aedt._data import (
    _text as _text, _number as _number, _freeze as _freeze,
    detached as detached, canonical_sha256 as canonical_sha256,
)
from scgsim.aedt.specs.common import LumpedRlc as LumpedRlc
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry as PreparedPlanarGeometry
from scgsim.aedt.specs.epr import (
    InterfaceKind as InterfaceKind, FieldSide as FieldSide,
    surface_evaluations as surface_evaluations,
    PlanarJunction as PlanarJunction, SurfaceEprSpec as SurfaceEprSpec,
    EprAnalysisRequest as EprAnalysisRequest,
    NativeExpressionDefinition as NativeExpressionDefinition,
    NormalizedSurfaceEprTotal as NormalizedSurfaceEprTotal,
    ExpressionCacheConvergence as ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet as ExpressionCacheConvergenceSet,
    SavedSolution as SavedSolution,
    _contained_relative_path as _contained_relative_path,
)


Route = Literal["A", "B"]


@dataclass(frozen=True)
class EprResult:
    """Detached partial-history or final saved-field EPR result."""

    result_kind: Literal["adaptive_history", "saved_field"]
    setup_name: str
    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    _legacy_payload: bool = field(default=False, repr=False, compare=False)
    _verified_source: EprResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.result_kind not in {"adaptive_history", "saved_field"}:
            raise ValueError("unsupported EPR result kind")
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        if isinstance(self.rows, (str, bytes)):
            raise TypeError("EPR result rows must be a sequence of mappings")
        rows = tuple(self.rows)
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("EPR result rows must contain mappings")
        if self._verified_source is not None and not isinstance(
            self._verified_source, EprResult
        ):
            raise TypeError("verified source must be an EprResult")
        prior_rows = (
            self._verified_source.rows if self._verified_source is not None else ()
        )
        frozen_rows = tuple(
            _freeze(
                row, verified=prior_rows[index] if index < len(prior_rows) else None
            )
            for index, row in enumerate(rows)
        )
        identities: set[tuple[int, int | None]] = set()
        complete_members = {
            "schema_version",
            "frequency_hz",
            "electric_energy_j",
            "magnetic_energy_j",
            "junction_inductive_energy_j",
            "magnetic_energy_balance_j",
            "junction_capacitive_energy_j",
            "normalization_energy_j",
            "electric_domains",
            "surface_contributions",
            "junctions",
        }
        for row in frozen_rows:
            mode = row.get("mode")
            if type(mode) is not int or mode <= 0:
                raise ValueError("EPR result row mode must be a positive integer")
            status = row.get("status")
            if status not in {"complete", "partial"}:
                raise ValueError("EPR result row status must be complete or partial")
            if "raw_integrals" in row and not isinstance(row["raw_integrals"], Mapping):
                raise TypeError("EPR result row raw_integrals must be a mapping")
            if (
                isinstance(row.get("raw_integrals"), Mapping)
                and "native_surface_references" in row["raw_integrals"]
                and not isinstance(
                    row["raw_integrals"]["native_surface_references"], Mapping
                )
            ):
                raise TypeError("raw native_surface_references must be a mapping")
            if "raw_integral_evidence" in row:
                raw_evidence = row["raw_integral_evidence"]
                if (
                    isinstance(raw_evidence, (str, bytes))
                    or not isinstance(raw_evidence, Sequence)
                    or any(not isinstance(item, Mapping) for item in raw_evidence)
                ):
                    raise TypeError(
                        "EPR result row raw_integral_evidence must contain mappings"
                    )
            native_pass: int | None = None
            if self.result_kind == "adaptive_history":
                native_pass = row.get("native_pass")
                if type(native_pass) is not int or native_pass <= 0:
                    raise ValueError(
                        "adaptive EPR result row native_pass must be a positive integer"
                    )
            elif "native_pass" in row:
                raise ValueError("saved-field EPR rows cannot claim an adaptive pass")
            identity = (mode, native_pass)
            if identity in identities:
                raise ValueError("EPR result rows repeat a mode/pass identity")
            identities.add(identity)
            if status == "complete":
                if not complete_members.issubset(row) or row.get(
                    "schema_version"
                ) not in {
                    "scgsim.aedt.epr-mode-energy.v1",
                    "scgsim.aedt.epr-mode-energy.v2",
                }:
                    raise ValueError(
                        "complete EPR result row lacks canonical energy members"
                    )
                for name in (
                    "frequency_hz",
                    "electric_energy_j",
                    "magnetic_energy_j",
                    "junction_inductive_energy_j",
                    "magnetic_energy_balance_j",
                    "junction_capacitive_energy_j",
                    "normalization_energy_j",
                ):
                    number = _number(row[name], f"EPR result row {name}")
                    if (
                        name in {"frequency_hz", "normalization_energy_j"}
                        and number <= 0
                    ):
                        raise ValueError(f"complete EPR result row {name} must be > 0")
                for name in (
                    "electric_domains",
                    "surface_contributions",
                    "junctions",
                ):
                    value = row[name]
                    if isinstance(value, (str, bytes)) or not isinstance(
                        value, Sequence
                    ):
                        raise TypeError(
                            f"complete EPR result row {name} must be a sequence"
                        )
                    if any(not isinstance(item, Mapping) for item in value):
                        raise TypeError(
                            f"complete EPR result row {name} must contain mappings"
                        )
                for surface in row["surface_contributions"]:
                    _number(
                        surface.get("participation"),
                        "surface contribution participation",
                    )
                    if row["schema_version"].endswith(".v2"):
                        film_assumptions(surface.get("assumptions"), allow_unknown=True)
                for junction in row["junctions"]:
                    _number(
                        junction.get("inductive_participation"),
                        "junction inductive participation",
                    )
                    _number(
                        junction.get("capacitive_participation"),
                        "junction capacitive participation",
                    )
            if "native_surface_references" in row:
                references = row["native_surface_references"]
                if isinstance(references, (str, bytes)) or not isinstance(
                    references, Sequence
                ):
                    raise TypeError("native_surface_references must be a sequence")
                if any(not isinstance(item, Mapping) for item in references):
                    raise TypeError("native_surface_references must contain mappings")
                reference_ids: set[str] = set()
                required_reference_members = {
                    "reference_id",
                    "contribution_ids",
                    "binding_ids",
                    "selection_name",
                    "face_ids",
                    "owner_semantic_id",
                    "source_polygon_id",
                    "object_name",
                    "native_object_id",
                    "interface_kinds",
                    "source_cap",
                    "source_z_um",
                    "source_supports",
                    "incident_domains",
                    "native_faces",
                    "native_normal",
                    "projection_normal",
                    "native_area_m2",
                    "face_list",
                    "sampling_basis",
                    "support_kind",
                    "normal_integral_v2",
                    "tangential_integral_v2",
                    "area_m2",
                    "expression_identities",
                }
                for reference in references:
                    if not required_reference_members <= set(reference):
                        raise ValueError(
                            "native surface reference lacks canonical evidence"
                        )
                    reference_id = _text(
                        reference["reference_id"], "native reference ID"
                    )
                    if reference_id in reference_ids:
                        raise ValueError("native surface reference IDs repeat")
                    reference_ids.add(reference_id)
                    for name in (
                        "normal_integral_v2",
                        "tangential_integral_v2",
                        "area_m2",
                        "native_area_m2",
                    ):
                        _number(
                            reference[name],
                            f"native surface reference {name}",
                            minimum=-math.inf,
                        )
                    expression_identities = reference["expression_identities"]
                    if (
                        not isinstance(expression_identities, Mapping)
                        or set(expression_identities)
                        != {"normal", "tangential", "area"}
                        or any(
                            not isinstance(value, Mapping)
                            for value in expression_identities.values()
                        )
                    ):
                        raise ValueError(
                            "native surface reference expression evidence is invalid"
                        )
        provenance = _freeze(
            self.provenance,
            verified=(
                self._verified_source.provenance
                if self._verified_source is not None
                else None
            ),
        )
        if not isinstance(provenance, Mapping):
            raise TypeError("EPR result provenance must be a mapping")
        for name in ("model_source_sha256", "analysis_source_sha256"):
            value = provenance.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"EPR result provenance {name} is invalid")
        requested_modes = provenance.get("requested_modes")
        if (
            not isinstance(requested_modes, tuple)
            or not requested_modes
            or any(type(item) is not int or item <= 0 for item in requested_modes)
            or len(set(requested_modes)) != len(requested_modes)
        ):
            raise ValueError("EPR result provenance requested_modes is invalid")
        object.__setattr__(self, "rows", frozen_rows)
        object.__setattr__(self, "provenance", provenance)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "scgsim.aedt.epr-result.v1"
                if self._legacy_payload
                else "scgsim.aedt.epr-result.v2"
            ),
            "result_kind": self.result_kind,
            "setup_name": self.setup_name,
            "rows": [detached(row) for row in self.rows],
            "provenance": detached(self.provenance),
        }

    @classmethod
    def from_payload(cls, value: Any) -> EprResult:
        if not isinstance(value, Mapping):
            raise TypeError("EPR result payload must be a mapping")
        expected = {
            "schema_version",
            "result_kind",
            "setup_name",
            "rows",
            "provenance",
        }
        schema = value.get("schema_version")
        if set(value) != expected or schema not in {
            "scgsim.aedt.epr-result.v1",
            "scgsim.aedt.epr-result.v2",
        }:
            raise ValueError("EPR result payload is not canonical")
        return cls(
            result_kind=value["result_kind"],
            setup_name=value["setup_name"],
            rows=tuple(value["rows"]),
            provenance=value["provenance"],
            _legacy_payload=schema.endswith(".v1"),
        )
