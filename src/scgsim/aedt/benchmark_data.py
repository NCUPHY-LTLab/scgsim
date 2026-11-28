"""Attribute captured AEDT profile memory to its native task context.

Native acquisition and unit conversion belong to runtime; this detached projection
owns the meaning and aggregation of memory in both capture and offline reports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict

BENCHMARK_RELATIVE = "results/simulation-benchmark.v1.json"
BENCHMARK_SCHEMA = "scgsim.aedt.simulation-benchmark.v1"
NATIVE_PROFILE_SOURCE = (
    "AEDT results/profile native OO tree GetChildNames/GetPropNames/GetPropValue"
)


def attribute_profile_memory(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Return a detached interpretation without changing raw native evidence.

    The v242 task table establishes executable peak RAM only for a task row with
    matching native Name and the real-time/CPU-time/Memory column schema. HPC
    resource information does not inherit that definition from a Memory label.
    """
    result = deepcopy(benchmark)
    source_bound = result.get("native_source") == NATIVE_PROFILE_SOURCE

    for profile in result.get("profiles", []):
        for group in profile["process_groups"]:
            _attribute_node_memory(
                group, (profile["native_setup_profile"],), False, source_bound
            )
    result["memory_interpretation"] = {
        "basis": "contextual v242 task-table attribution",
        "raw_native_values": "preserved",
        "unqualified_memory": "unknown scope; excluded from task-peak aggregates",
        "missing_qualified_peak": "unavailable; no resource or OS fallback",
    }
    return result


def _attribute_node_memory(
    node: dict[str, Any],
    ancestry: tuple[str, ...],
    in_resources: bool,
    source_bound: bool,
) -> list[_TaskPeakCandidate]:
    """Attribute qualified task memory while preserving every native raw value."""
    path = (*ancestry, node["name"])
    in_resources = in_resources or node["name"] == "HPC Group"
    properties = node["native_properties"]
    metrics = node["metrics"]
    qualified = (
        source_bound
        and not in_resources
        and properties.get("Name") == node["name"]
        and all(key in properties for key in ("Real time", "Cpu time", "Memory"))
        and all(
            key in metrics for key in ("real_seconds", "cpu_seconds", "memory_gb_sdk")
        )
    )
    candidates: list[_TaskPeakCandidate] = []
    if "memory_gb_sdk" in metrics:
        metrics["memory_scope"] = (
            "individual task executable peak, not whole host"
            if qualified
            else "unknown: unqualified native Memory"
        )
        metrics["memory_scope_basis"] = (
            "v242 task-table columns; matching native Name; parsed task timings; "
            "results/profile source; outside HPC resource subtree"
            if qualified
            else "native task-table context not established"
        )
        metrics["memory_task_peak_qualified"] = qualified
        if qualified:
            candidates.append({"node": node, "path": path})
    descendants: list[_TaskPeakCandidate] = []
    for child in node.get("children", []):
        descendants.extend(
            _attribute_node_memory(child, path, in_resources, source_bound)
        )
    # Drop any stage_memory_peak the node already carries and recompute it from
    # qualified descendants. Only the detached attribution is rebuilt, never
    # the receipt-bound artifact.
    node.pop("stage_memory_peak", None)
    if descendants:
        peak = max(
            descendants, key=lambda item: item["node"]["metrics"]["memory_gb_sdk"]
        )
        peak_metrics = peak["node"]["metrics"]
        node["stage_memory_peak"] = {
            "stage": peak["node"]["name"],
            "stage_path": list(peak["path"]),
            "memory_native": peak_metrics["memory_native"],
            "memory_gb_sdk": peak_metrics["memory_gb_sdk"],
            "scope": "maximum observed qualified descendant task peak, not host peak",
            "basis": peak_metrics["memory_scope_basis"],
        }
    return candidates + descendants


class _TaskPeakCandidate(TypedDict):
    """A qualified task node and its native setup/process ancestry."""

    node: dict[str, Any]
    path: tuple[str, ...]
