"""Own one geometry build transaction: sidecars, existing conformity checks, native lowering and final physical export. No mesh or solver state."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from scgsim.diagnostics import diagnostic_stage
from scgsim.geometry.compiler.dispatch import build_route_construction_plan
from scgsim.geometry.compiler.stages import BuildTimingPayload, StageTimings
from scgsim.geometry.compiler.validation import (
    validate_backend_tag_ledger,
)
from scgsim.geometry.construction.conformality import (
    assert_engine_gate_pass,
    engine_gate_volume_adjacency_conformality,
)
from scgsim.geometry.construction.export import export_physical_group_records
from scgsim.geometry.construction.gmsh_occ import write_occ_geometry_from_plan
from scgsim.geometry.models.common import (
    RUN_METADATA_DIR,
    SEMANTIC_GEOMETRY_METADATA_DIR,
    PathInput,
    RouteLiteral,
)
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.tags import FinalPhysicalGroupRecord
from scgsim.geometry.primitives.grid_cache import with_grid_work_cache
from scgsim.geometry.source.validation import validate_geometry_input


class SemanticGeometryBuilder:
    """Facade for one route-aware semantic geometry construction run.

    The builder must not construct arbitrary volumes and then ask the backend
    to fragment them to discover topology. Route selection is an early input:
    Route A plans sheets/shells, Route B plans cutout shells, and Route C plans
    retained material volumes with explicit shared surfaces where required.

    `build()` is the reviewable top-level pipeline:

    1. validate the adapter-owned `GeometryBuildInput`;
    2. build a route-specific `ConstructionPlanRecord`;
    3. write the pre-lowering volume-adjacency Engine Gate artifact;
    4. call the bottom-up Gmsh/OCC backend with that plan;
    5. write the post-lowering Gmsh BRep Engine Gate artifact;
    6. export `FinalPhysicalGroupRecord`s from backend dim-tags.

    One call writes one route XAO file. Mesh generation is downstream.
    """

    @diagnostic_stage("geometry.compiler.build")
    @with_grid_work_cache
    def build(
        self,
        build_input: GeometryBuildInput,
        *,
        route: RouteLiteral,
        run_folder: PathInput,
        representation_context=None,
    ) -> tuple[FinalPhysicalGroupRecord, ...]:
        """Validate input, plan geometry, build OCC, then export groups."""
        if representation_context is None:
            raise ValueError(
                "interpret_physical_geometry() and pass plan.context before native compilation"
            )
        run_path = Path(run_folder)
        metadata_dir = run_path / RUN_METADATA_DIR / SEMANTIC_GEOMETRY_METADATA_DIR
        geometry_dir = run_path / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        transaction = _BuildTransaction(route, metadata_dir, geometry_dir)
        try:
            validated_input = transaction.validate_input(build_input)
            plan = transaction.plan_geometry(validated_input, representation_context)
            built_plan = transaction.construct_geometry(plan)
            return transaction.export_groups(built_plan)
        finally:
            _write_stage_sidecar(
                metadata_dir, "00_timing", transaction.timing_payload()
            )


@dataclass
class _BuildTransaction:
    route: RouteLiteral
    metadata_dir: Path
    geometry_dir: Path
    timings: StageTimings = field(default_factory=StageTimings)
    sidecar_sizes: dict[str, int] = field(default_factory=dict)
    plan: ConstructionPlanRecord | None = None
    built_plan: ConstructionPlanRecord | None = None
    started: float = field(init=False)

    def __post_init__(self) -> None:
        """Start the transaction clock after initializing its progress records."""
        self.started = time.perf_counter()

    def write_sidecar(self, stage_name: str, payload: Any) -> None:
        """Track bytes only after the complete sidecar write succeeds."""
        self.sidecar_sizes[stage_name] = _write_stage_sidecar(
            self.metadata_dir, stage_name, payload
        )

    def validate_input(self, build_input: GeometryBuildInput) -> GeometryBuildInput:
        """Validate the source input and retain its pre-planning sidecar."""
        with self.timings.measure("validate_geometry_input"):
            validated_input = validate_geometry_input(build_input)
        with self.timings.measure("write_01_validate_geometry_input"):
            self.write_sidecar("01_validate_geometry_input", validated_input)
        return validated_input

    def plan_geometry(
        self,
        validated_input: GeometryBuildInput,
        representation_context: RepresentationContext,
    ) -> ConstructionPlanRecord:
        """Plan and record geometry before enforcing planned conformality."""
        with self.timings.measure("build_route_construction_plan"):
            plan = build_route_construction_plan(
                validated_input,
                route=self.route,
                representation_context=representation_context,
            )
        self.plan = plan
        with self.timings.measure("write_02_build_route_construction_plan"):
            self.write_sidecar("02_build_route_construction_plan", plan)
        with self.timings.measure("engine_gate_volume_adjacency_conformality"):
            volume_adjacency_gate = engine_gate_volume_adjacency_conformality(plan)
        with self.timings.measure("write_engine_gate_volume_adjacency_conformality"):
            self.write_sidecar(
                "engine_gate_volume_adjacency_conformality", volume_adjacency_gate
            )
        with self.timings.measure("assert_engine_gate_volume_adjacency_conformality"):
            assert_engine_gate_pass(volume_adjacency_gate)
        return plan

    def construct_geometry(
        self, plan: ConstructionPlanRecord
    ) -> ConstructionPlanRecord:
        """Lower native geometry, retain its evidence and enforce its result."""
        with self.timings.measure("write_occ_geometry_from_plan"):
            built_plan = write_occ_geometry_from_plan(
                plan,
                xao_path=self.geometry_dir
                / f"semantic_geometry_route_{self.route.lower()}.xao",
            )
        self.built_plan = built_plan
        with self.timings.measure("write_03_build_occ_geometry"):
            self.write_sidecar("03_build_occ_geometry", built_plan)
        # The existing gate validator owns missing/malformed native-report errors.
        gmsh_brep_gate: Any = built_plan.metadata.get(
            "engine_gate_gmsh_brep_conformality"
        )
        with self.timings.measure("write_engine_gate_gmsh_brep_conformality"):
            self.write_sidecar("engine_gate_gmsh_brep_conformality", gmsh_brep_gate)
        with self.timings.measure("assert_engine_gate_gmsh_brep_conformality"):
            assert_engine_gate_pass(gmsh_brep_gate)
        return built_plan

    def export_groups(
        self, built_plan: ConstructionPlanRecord
    ) -> tuple[FinalPhysicalGroupRecord, ...]:
        """Validate native tags and retain the exported physical-group records."""
        with self.timings.measure("validate_backend_tag_ledger"):
            validate_backend_tag_ledger(backend_tags=built_plan.backend_entity_tags)
        with self.timings.measure("export_physical_group_records"):
            physical_groups = export_physical_group_records(built_plan)
        with self.timings.measure("write_04_export_physical_groups"):
            self.write_sidecar("04_export_physical_groups", physical_groups)
        return physical_groups

    def timing_payload(self) -> BuildTimingPayload:
        """Snapshot completed progress for the final sidecar on every exit."""
        return {
            "route": self.route,
            "total_seconds": round(time.perf_counter() - self.started, 6),
            "counts": _timing_counts(self.plan, self.built_plan),
            "stages": self.timings.records,
            "sidecar_bytes": dict(self.sidecar_sizes or {}),
            "planning_stages": list(
                getattr(self.plan, "metadata", {}).get("timings", ())
            ),
            "backend_stages": list(
                getattr(self.built_plan, "metadata", {}).get("backend_timings", ())
            ),
        }


def _timing_counts(
    plan: ConstructionPlanRecord | None, built_plan: ConstructionPlanRecord | None
) -> dict[str, int]:
    if plan is None:
        return {}
    counts = {
        "interfaces": len(getattr(plan, "interfaces", ())),
        "surface_partitions": len(getattr(plan, "surface_partitions", ())),
        "points": len(getattr(plan, "points", ())),
        "curves": len(getattr(plan, "curves", ())),
        "surface_loops": len(getattr(plan, "surface_loops", ())),
        "surfaces": len(getattr(plan, "surfaces", ())),
        "volumes": len(getattr(plan, "volumes", ())),
        "construction_bodies": len(getattr(plan, "construction_bodies", ())),
        "cut_operations": len(getattr(plan, "cut_operations", ())),
        "tags": len(getattr(plan, "tags", ())),
    }
    xao_path = (
        getattr(built_plan, "metadata", {}).get("xao_path") if built_plan else None
    )
    if xao_path and Path(xao_path).is_file():
        counts["xao_bytes"] = Path(xao_path).stat().st_size
    return counts


def _write_stage_sidecar(path: Path, stage_name: str, payload: Any) -> int:
    path.mkdir(parents=True, exist_ok=True)
    # Sidecars retain the full evidence, including repeated ledger references.
    # Convert and redact shared records once without dataclasses.asdict's deep
    # copy; JSON still emits every occurrence using the existing file format.
    with diagnostic_stage("geometry.compiler.sidecar.redaction"):
        public_payload = _redact_local_paths(payload, run_root=path.parents[1])
    with diagnostic_stage("geometry.compiler.sidecar.json"):
        text = json.dumps(public_payload, indent=2, sort_keys=True)
    with diagnostic_stage(
        "geometry.compiler.sidecar.write", counts={"bytes": len(text.encode("utf-8"))}
    ):
        destination = path / f"{stage_name}.json"
        destination.write_text(text, encoding="utf-8")
    return destination.stat().st_size


def _redact_local_paths(payload: Any, *, run_root: Path) -> Any:
    """Return sidecar-safe data, sharing pure conversion within this call."""
    memo: dict[int, Any] = {}

    def convert(value):
        structured = (
            is_dataclass(value) and not isinstance(value, type)
        ) or isinstance(value, (Mapping, tuple, list))
        key = id(value)
        if structured and key in memo:
            return memo[key]
        if is_dataclass(value) and not isinstance(value, type):
            result = {
                field.name: convert(getattr(value, field.name))
                for field in fields(value)
            }
        elif isinstance(value, str):
            candidate = Path(value)
            if candidate.is_absolute():
                return _redact_one_path(candidate, run_root=run_root)
            return re.sub(
                r"(?<!\S)(/[^\s]+)",
                lambda match: _redact_one_path(Path(match.group(1)), run_root=run_root),
                value,
            )
        elif isinstance(value, Mapping):
            result = {key: convert(item) for key, item in value.items()}
        elif isinstance(value, tuple):
            result = tuple(convert(item) for item in value)
        elif isinstance(value, list):
            result = [convert(item) for item in value]
        else:
            return value
        if structured:
            memo[key] = result
        return result

    return convert(payload)


def _redact_one_path(candidate: Path, *, run_root: Path) -> str:
    try:
        return str(candidate.relative_to(run_root))
    except ValueError:
        return f"<external>/{candidate.name}"
