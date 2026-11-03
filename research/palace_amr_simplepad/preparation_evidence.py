"""Persist truthful route-preparation failures in the study directory."""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .contract import CaseDefinition

if TYPE_CHECKING:
    from scgsim.geometry.source.plan import GeometryPlanSnapshot


class RoutePreparer(Protocol):
    """Preparation operation that builds one route and seals its AMR cases."""

    def __call__(self, request: RoutePreparationRequest) -> None:
        """Prepare one route from its explicit source and evidence context."""


@dataclass(frozen=True)
class RoutePreparationRequest:
    """Inputs and state callback for preparing one route's two cases."""

    study_dir: Path
    snapshot: GeometryPlanSnapshot
    route_cases: tuple[CaseDefinition, ...]
    wrapper_path: Path
    preparation_timings: dict[str, Any]
    on_case_sealed: Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RoutePreparationContext:
    """Dependencies and current manifest for the ordered route preparation stage."""

    study_dir: Path
    manifest: dict[str, Any]
    route_groups: tuple[tuple[str, tuple[CaseDefinition, ...]], ...]
    snapshot: GeometryPlanSnapshot
    wrapper_path: Path
    prepare_route: RoutePreparer
    on_case_sealed: Callable[[Path, dict[str, Any], dict[str, Any]], None]
    write_manifest: Callable[[Path, dict[str, Any]], None]
    require_prepared: Callable[[dict[str, Any]], None]


def write_route_failure_evidence(
    study_dir: Path, route_id: str, error: Exception
) -> dict[str, str]:
    """Write the original route exception traceback and return its relative identity."""
    relative_path = Path("preparation") / "errors" / f"{route_id}.traceback.txt"
    traceback_path = study_dir / relative_path
    traceback_path.parent.mkdir(parents=True, exist_ok=True)
    traceback_bytes = "".join(traceback.format_exception(error)).encode("utf-8")
    traceback_path.write_bytes(traceback_bytes)
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback_path": relative_path.as_posix(),
        "traceback_sha256": hashlib.sha256(traceback_bytes).hexdigest(),
    }


def prepare_routes(context: RoutePreparationContext) -> int:
    """Attempt every route, retain sealed cases and failures, and set final status."""
    manifest = context.manifest
    study_dir = context.study_dir
    all_cases = tuple(case for _, cases in context.route_groups for case in cases)
    manifest["cases"] = [
        {
            "case_id": case.case_id,
            "route_id": case.route_id,
            "representation": case.representation,
            "approximation": case.approximation,
            "amr_nonconformal": case.nonconformal,
            "preparation_status": "planned",
        }
        for case in all_cases
    ]
    manifest["routes"] = [
        {
            "route_id": route_id,
            "case_ids": [case.case_id for case in cases],
            "preparation_status": "planned",
        }
        for route_id, cases in context.route_groups
    ]
    context.write_manifest(study_dir, manifest)

    failed_routes = 0
    for route_id, route_cases in context.route_groups:
        route = next(
            item for item in manifest["routes"] if item["route_id"] == route_id
        )
        route["preparation_status"] = "started"
        context.write_manifest(study_dir, manifest)
        started = perf_counter()
        try:
            context.prepare_route(
                RoutePreparationRequest(
                    study_dir=study_dir,
                    snapshot=context.snapshot,
                    route_cases=route_cases,
                    wrapper_path=context.wrapper_path,
                    preparation_timings=manifest["preparation_timings"],
                    on_case_sealed=lambda entry: context.on_case_sealed(
                        study_dir, manifest, entry
                    ),
                )
            )
        except Exception as error:  # noqa: BLE001 -- retain any route failure before trying its independent peer.
            route["preparation_status"] = "failed"
            failure = write_route_failure_evidence(study_dir, route_id, error)
            route["failure"] = failure
            for case in manifest["cases"]:
                if (
                    case["route_id"] == route_id
                    and case["preparation_status"] != "sealed"
                ):
                    case["preparation_status"] = "failed"
                    case["preparation_failure_traceback_path"] = failure[
                        "traceback_path"
                    ]
            failed_routes += 1
        else:
            route["preparation_status"] = "completed"
        finally:
            elapsed = perf_counter() - started
            manifest["preparation_timings"]["route_preparation_seconds"][route_id] = (
                elapsed
            )
            route["preparation_seconds"] = elapsed
        context.write_manifest(study_dir, manifest)

    if failed_routes:
        manifest["status"] = "preparation_failed"
    else:
        manifest["status"] = "prepared"
        context.require_prepared(manifest)
    context.write_manifest(study_dir, manifest)
    return int(failed_routes > 0)
