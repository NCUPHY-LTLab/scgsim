"""Detached EPR and junction declarations, independent of native execution."""
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from pathlib import Path
from scgsim.semantics.epr import film_assumptions
from scgsim.aedt._data import _text, _number, _freeze, detached, canonical_sha256
from scgsim.aedt.versions import normalize_aedt_version

InterfaceKind = Literal["MA", "MS", "SA", "MM"]
FieldSide = Literal["top", "bottom", "sidewall"]

def surface_evaluations(
    margins_um: Sequence[float], *, policy: str | None
) -> tuple[tuple[str, float], ...]:
    """Keep the automatic baseline distinct from each requested mask."""

    requested = tuple(("requested_margin", float(value)) for value in margins_um)
    if policy is None:
        return requested  # Historical records had no automatic baseline.
    if policy != "unmasked_plus_requested.v1":
        raise ValueError("unsupported surface evaluation policy")
    return (("unmasked_baseline", 0.0), *requested)

@dataclass(frozen=True)
class PlanarJunction:
    """One source-linked linearized Lumped RLC junction sheet."""

    junction_id: str
    source_polygon_id: str
    terminal_a_net: str
    terminal_b_net: str
    direction_xy: tuple[float, float]
    width_um: float
    inductance_h: float
    capacitance_f: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "junction_id",
            "source_polygon_id",
            "terminal_a_net",
            "terminal_b_net",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.terminal_a_net == self.terminal_b_net:
            raise ValueError("junction terminal nets must be distinct")
        if len(self.direction_xy) != 2:
            raise ValueError("junction direction_xy must contain two values")
        dx = _number(self.direction_xy[0], "direction_xy[0]", minimum=-math.inf)
        dy = _number(self.direction_xy[1], "direction_xy[1]", minimum=-math.inf)
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            raise ValueError("junction direction_xy must be nonzero")
        object.__setattr__(self, "direction_xy", (dx / norm, dy / norm))
        object.__setattr__(self, "width_um", _number(self.width_um, "width_um"))
        if self.width_um <= 0.0:
            raise ValueError("junction width_um must be > 0")
        object.__setattr__(
            self, "inductance_h", _number(self.inductance_h, "inductance_h")
        )
        object.__setattr__(
            self, "capacitance_f", _number(self.capacitance_f, "capacitance_f")
        )
        if self.inductance_h <= 0.0 or self.capacitance_f < 0.0:
            raise ValueError("junction L must be > 0 and C must be >= 0")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "junction_id": self.junction_id,
            "source_polygon_id": self.source_polygon_id,
            "terminal_a_net": self.terminal_a_net,
            "terminal_b_net": self.terminal_b_net,
            "direction_xy": list(self.direction_xy),
            "width_um": self.width_um,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "metadata": detached(self.metadata),
        }

    @classmethod
    def from_payload(cls, value: Any) -> PlanarJunction:
        if not isinstance(value, Mapping):
            raise TypeError("junction payload must be a mapping")
        expected = {
            "junction_id",
            "source_polygon_id",
            "terminal_a_net",
            "terminal_b_net",
            "direction_xy",
            "width_um",
            "inductance_h",
            "capacitance_f",
            "metadata",
        }
        if set(value) != expected:
            raise ValueError("junction payload members are not canonical")
        return cls(**dict(value))

@dataclass(frozen=True)
class SurfaceEprSpec:
    """One complete physical-side contribution group and its alternative margins.

    ``source_polygon_id`` anchors provenance in the group's catalogue union; it
    does not filter patches. Each resolved binding retains its own contribution
    evidence and geometry reference for per-patch provenance.
    """

    contribution_id: str
    source_polygon_id: str
    interface_kind: InterfaceKind
    field_side: FieldSide
    margins_um: tuple[float, ...] = (0.0,)
    film_thickness_m: float | None = None
    film_relative_permittivity: float | None = None
    loss_tangent: float | None = None
    source: str | None = None
    preset: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _legacy_payload: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("contribution_id", "source_polygon_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.interface_kind not in {"MA", "MS", "SA", "MM"}:
            raise ValueError(f"unsupported EPR interface kind {self.interface_kind!r}")
        if self.field_side not in {"top", "bottom", "sidewall"}:
            raise ValueError(f"unsupported EPR field side {self.field_side!r}")
        margins = tuple(_number(value, "margin_um") for value in self.margins_um)
        if not margins or len(margins) != len(set(margins)):
            raise ValueError("margins_um must be non-empty and unique")
        object.__setattr__(self, "margins_um", margins)
        optional = (
            "film_thickness_m",
            "film_relative_permittivity",
        )
        for name in optional:
            value = getattr(self, name)
            if value is not None:
                number = _number(value, name)
                if number <= 0.0:
                    raise ValueError(f"{name} must be > 0 when provided")
                object.__setattr__(self, name, number)
        if self.interface_kind in {"MA", "MS", "SA"} and (
            self.film_thickness_m is None or self.film_relative_permittivity is None
        ):
            raise ValueError(
                f"{self.interface_kind} requires film thickness and permittivity"
            )
        film_assumptions(
            {
                "film_thickness_m": self.film_thickness_m,
                "film_relative_permittivity": self.film_relative_permittivity,
                "loss_tangent": self.loss_tangent,
                "source": self.source,
                "preset": self.preset,
            }
            if self.interface_kind in {"MA", "MS", "SA"}
            else {
                "loss_tangent": self.loss_tangent,
                "source": self.source,
                "preset": self.preset,
            },
            partial=self.interface_kind == "MM",
        )
        if self._legacy_payload and any(
            value is not None for value in (self.loss_tangent, self.source, self.preset)
        ):
            raise ValueError("legacy surface payload cannot carry new film assumptions")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "contribution_id": self.contribution_id,
            "source_polygon_id": self.source_polygon_id,
            "interface_kind": self.interface_kind,
            "field_side": self.field_side,
            "margins_um": list(self.margins_um),
            "film_thickness_m": self.film_thickness_m,
            "film_relative_permittivity": self.film_relative_permittivity,
            "metadata": detached(self.metadata),
        }
        if not self._legacy_payload:
            payload.update(
                loss_tangent=self.loss_tangent, source=self.source, preset=self.preset
            )
        return payload

    @classmethod
    def from_payload(cls, value: Any) -> SurfaceEprSpec:
        if not isinstance(value, Mapping):
            raise TypeError("surface EPR payload must be a mapping")
        expected = {
            "contribution_id",
            "source_polygon_id",
            "interface_kind",
            "field_side",
            "margins_um",
            "film_thickness_m",
            "film_relative_permittivity",
            "metadata",
        }
        new = expected | {"loss_tangent", "source", "preset"}
        if set(value) not in (expected, new):
            raise ValueError("surface EPR payload members are not canonical")
        return cls(**dict(value), _legacy_payload=set(value) == expected)

@dataclass(frozen=True)
class EprAnalysisRequest:
    """Optional analysis/cache selection independent of model construction."""

    mode_indices: tuple[int, ...] | None = None
    surface_contribution_ids: tuple[str, ...] | None = None
    junction_ids: tuple[str, ...] | None = None
    bulk_domain_ids: tuple[str, ...] | None = None
    _legacy_payload: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "mode_indices",
            "surface_contribution_ids",
            "junction_ids",
            "bulk_domain_ids",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{name} must be a sequence")
            values = tuple(value)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            if name == "mode_indices":
                if any(type(item) is not int or item <= 0 for item in values):
                    raise ValueError("mode_indices must contain positive integers")
            else:
                values = tuple(_text(item, name) for item in values)
            object.__setattr__(self, name, values)
        if self._legacy_payload and self.bulk_domain_ids is not None:
            raise ValueError("legacy EPR request cannot select bulk domains")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "mode_indices": (
                None if self.mode_indices is None else list(self.mode_indices)
            ),
            "surface_contribution_ids": (
                None
                if self.surface_contribution_ids is None
                else list(self.surface_contribution_ids)
            ),
            "junction_ids": (
                None if self.junction_ids is None else list(self.junction_ids)
            ),
        }
        if not self._legacy_payload:
            payload["bulk_domain_ids"] = (
                None if self.bulk_domain_ids is None else list(self.bulk_domain_ids)
            )
        return payload

    @classmethod
    def from_payload(cls, value: Any) -> EprAnalysisRequest:
        if not isinstance(value, Mapping):
            raise TypeError("EPR analysis request must be a mapping")
        expected = {
            "mode_indices",
            "surface_contribution_ids",
            "junction_ids",
        }
        if set(value) not in (expected, expected | {"bulk_domain_ids"}):
            raise ValueError("EPR analysis request members are not canonical")
        return cls(**dict(value), _legacy_payload=set(value) == expected)

@dataclass(frozen=True)
class NativeExpressionDefinition:
    """One caller-authored Field Calculator operation sequence."""

    name: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "expression name"))
        if isinstance(self.operations, (str, bytes)) or not isinstance(
            self.operations, Sequence
        ):
            raise TypeError("expression operations must be a sequence of strings")
        operations = tuple(self.operations)
        if not operations or any(
            not isinstance(item, str) or not item or "\n" in item or "\0" in item
            for item in operations
        ):
            raise ValueError("expression operations must be non-empty native commands")
        object.__setattr__(self, "operations", operations)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "native_expression",
            "name": self.name,
            "operations": list(self.operations),
        }

    @classmethod
    def from_payload(cls, value: Any) -> NativeExpressionDefinition:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"kind", "name", "operations"}
            or value.get("kind") != "native_expression"
        ):
            raise ValueError("native expression definition members are not canonical")
        return cls(name=value["name"], operations=value["operations"])

@dataclass(frozen=True)
class NormalizedSurfaceEprTotal:
    """Select one interface/evaluation/margin total for a single mode."""

    interface_kind: Literal["MA", "MS", "SA"]
    evaluation_kind: Literal["unmasked_baseline", "requested_margin"]
    margin_um: float

    def __post_init__(self) -> None:
        if self.interface_kind not in {"MA", "MS", "SA"}:
            raise ValueError("surface EPR total interface must be MA, MS, or SA")
        if self.evaluation_kind not in {"unmasked_baseline", "requested_margin"}:
            raise ValueError("surface EPR total evaluation kind is invalid")
        margin = _number(self.margin_um, "surface EPR total margin")
        if self.evaluation_kind == "unmasked_baseline" and margin != 0.0:
            raise ValueError("unmasked baseline margin must be exactly 0 um")
        object.__setattr__(self, "margin_um", margin)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "normalized_surface_epr_total",
            "interface_kind": self.interface_kind,
            "evaluation_kind": self.evaluation_kind,
            "margin_um": self.margin_um,
        }

    @classmethod
    def from_payload(cls, value: Any) -> NormalizedSurfaceEprTotal:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"kind", "interface_kind", "evaluation_kind", "margin_um"}
            or value.get("kind") != "normalized_surface_epr_total"
        ):
            raise ValueError("normalized surface total members are not canonical")
        return cls(
            interface_kind=value["interface_kind"],
            evaluation_kind=value["evaluation_kind"],
            margin_um=value["margin_um"],
        )

@dataclass(frozen=True)
class ExpressionCacheConvergence:
    """Select one native expression and its explicit convergence criterion."""

    mode: int
    target: NativeExpressionDefinition | NormalizedSurfaceEprTotal
    criterion: float
    use_relative_convergence: bool = True

    def __post_init__(self) -> None:
        if type(self.mode) is not int or self.mode <= 0:
            raise ValueError("convergence mode must be a positive integer")
        if not isinstance(
            self.target, (NativeExpressionDefinition, NormalizedSurfaceEprTotal)
        ):
            raise TypeError(
                "convergence target must be a native expression or normalized surface total"
            )
        criterion = _number(self.criterion, "convergence criterion", minimum=-math.inf)
        if type(self.use_relative_convergence) is not bool:
            raise TypeError("use_relative_convergence must be a bool")
        object.__setattr__(self, "criterion", criterion)

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target": self.target.to_payload(),
            "criterion": self.criterion,
            "use_relative_convergence": self.use_relative_convergence,
        }

    @classmethod
    def from_payload(cls, value: Any) -> ExpressionCacheConvergence:
        if not isinstance(value, Mapping) or set(value) != {
            "mode",
            "target",
            "criterion",
            "use_relative_convergence",
        }:
            raise ValueError("expression convergence members are not canonical")
        target = value["target"]
        if not isinstance(target, Mapping):
            raise TypeError("expression convergence target must be a mapping")
        kind = target.get("kind")
        if kind == "native_expression":
            parsed_target = NativeExpressionDefinition.from_payload(target)
        elif kind == "normalized_surface_epr_total":
            parsed_target = NormalizedSurfaceEprTotal.from_payload(target)
        else:
            raise ValueError("unsupported expression convergence target")
        return cls(
            mode=value["mode"],
            target=parsed_target,
            criterion=value["criterion"],
            use_relative_convergence=value["use_relative_convergence"],
        )


@dataclass(frozen=True)
class ExpressionCacheConvergenceSet:
    """Independent native expression criteria; frequency controls remain separate."""

    controls: tuple[ExpressionCacheConvergence, ...]

    def __post_init__(self) -> None:
        controls = tuple(self.controls)
        if not controls or any(
            not isinstance(item, ExpressionCacheConvergence) for item in controls
        ):
            raise TypeError("controls must be a nonempty collection of ExpressionCacheConvergence")
        identities = [
            (item.mode, canonical_sha256(item.target.to_payload())) for item in controls
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("expression convergence target/mode pairs must be unique")
        object.__setattr__(self, "controls", controls)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "expression_cache_convergence_set",
            "controls": [item.to_payload() for item in self.controls],
        }

    @classmethod
    def from_payload(cls, value: Any) -> ExpressionCacheConvergenceSet:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"kind", "controls"}
            or value["kind"] != "expression_cache_convergence_set"
        ):
            raise ValueError("expression convergence collection members are not canonical")
        return cls(controls=tuple(
            ExpressionCacheConvergence.from_payload(item) for item in value["controls"]
        ))


def iter_expression_convergence(
    control: ExpressionCacheConvergence | ExpressionCacheConvergenceSet | None,
) -> tuple[ExpressionCacheConvergence, ...]:
    """One normalization path without rewriting historical singleton requests."""
    if control is None:
        return ()
    if isinstance(control, ExpressionCacheConvergence):
        return (control,)
    if isinstance(control, ExpressionCacheConvergenceSet):
        return control.controls
    raise TypeError("expression convergence must be a unit control, collection, or None")


@dataclass(frozen=True)
class SavedSolution:
    """Verified immutable saved project and complete result-directory inventory."""

    root: Path
    project_path: Path
    result_path: Path
    members: tuple[Mapping[str, Any], ...]
    content_sha256: str
    identity: Mapping[str, Any]
    aedt_version: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        project = Path(self.project_path).resolve()
        result = Path(self.result_path).resolve()
        if not project.is_relative_to(root) or not result.is_relative_to(root):
            raise ValueError("saved solution members must be contained by root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "project_path", project)
        object.__setattr__(self, "result_path", result)
        if isinstance(self.members, (str, bytes)):
            raise TypeError("saved solution members must be a sequence of mappings")
        members = tuple(self.members)
        if any(not isinstance(item, Mapping) for item in members):
            raise TypeError("saved solution members must contain mappings")
        object.__setattr__(self, "members", tuple(_freeze(item) for item in members))
        if not isinstance(self.content_sha256, str) or len(self.content_sha256) != 64:
            raise ValueError("saved solution content_sha256 is invalid")
        identity = _freeze(self.identity)
        expected_identity = {
            "model_source_sha256",
            "project_name",
            "design_name",
            "setup_name",
            "physical_variation",
            "solver_last_completed_pass",
            "saved_fields_pass",
            "saved_fields",
        }
        if not isinstance(identity, Mapping) or set(identity) != expected_identity:
            raise ValueError("saved solution identity members are not canonical")
        for name in expected_identity - {
            "solver_last_completed_pass",
            "saved_fields_pass",
            "saved_fields",
        }:
            _text(identity[name], f"saved solution identity.{name}")
        for name in ("solver_last_completed_pass", "saved_fields_pass"):
            if type(identity[name]) is not int or identity[name] <= 0:
                raise ValueError(f"saved solution {name} must be positive")
        if identity["saved_fields"] is not True:
            raise ValueError("saved solution must explicitly attest saved fields")
        object.__setattr__(self, "identity", identity)
        if self.aedt_version is not None:
            object.__setattr__(
                self, "aedt_version", normalize_aedt_version(self.aedt_version)
            )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": (
                "scgsim.aedt.saved-solution.v2"
                if self.aedt_version is not None
                else "scgsim.aedt.saved-solution.v1"
            ),
            "project": self.project_path.relative_to(self.root).as_posix(),
            "results": self.result_path.relative_to(self.root).as_posix(),
            "members": [detached(item) for item in self.members],
            "content_sha256": self.content_sha256,
            "identity": detached(self.identity),
        }
        if self.aedt_version is not None:
            payload["aedt_version"] = self.aedt_version
        return payload

    @classmethod
    def from_payload(cls, root: str | Path, value: Any) -> SavedSolution:
        if not isinstance(value, Mapping):
            raise TypeError("saved solution payload must be a mapping")
        expected = {
            "schema_version",
            "project",
            "results",
            "members",
            "content_sha256",
            "identity",
        }
        schema = value.get("schema_version")
        if schema == "scgsim.aedt.saved-solution.v1":
            if set(value) != expected:
                raise ValueError("saved solution payload is not canonical")
            aedt_version = None
        elif schema == "scgsim.aedt.saved-solution.v2":
            if set(value) != expected | {"aedt_version"}:
                raise ValueError("saved solution payload is not canonical")
            aedt_version = normalize_aedt_version(value["aedt_version"])
        else:
            raise ValueError("saved solution payload is not canonical")
        base = Path(root)
        project = _contained_relative_path(value["project"], "saved solution project")
        results = _contained_relative_path(value["results"], "saved solution results")
        return cls(
            root=base,
            project_path=base / project,
            result_path=base / results,
            members=tuple(value["members"]),
            content_sha256=value["content_sha256"],
            identity=value["identity"],
            aedt_version=aedt_version,
        )

def _contained_relative_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path
