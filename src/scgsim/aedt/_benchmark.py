"""Decide what each memory number in an AEDT profile means, and find the peaks.

AEDT's profile tree reports a "Memory" value on many nodes, but only some of them
are the peak RAM of one solver task. This module labels each node as a qualified
task peak or as unknown scope, then records the largest qualified peak under each
stage. It works on a copy of the captured benchmark and never changes the raw
native values, so it can run both at capture time and offline on a saved report.

Reading the profile from AEDT and converting its units belong to the runtime;
this module owns only the meaning and the aggregation of memory.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

BENCHMARK_RELATIVE = "results/simulation-benchmark.v1.json"
BENCHMARK_SCHEMA = "scgsim.aedt.simulation-benchmark.v1"
NATIVE_PROFILE_SOURCE = (
    "AEDT results/profile native OO tree GetChildNames/GetPropNames/GetPropValue"
)

# The native columns a task row must carry before its Memory counts as a task peak.
_TASK_NATIVE_COLUMNS = ("Real time", "Cpu time", "Memory")
_TASK_METRICS = ("real_seconds", "cpu_seconds", "memory_gb_sdk")

_UNQUALIFIED_SCOPE = "unknown: unqualified native Memory"
_UNQUALIFIED_BASIS = "native task-table context not established"
_QUALIFIED_SCOPE = "individual task executable peak, not whole host"
_QUALIFIED_BASIS = (
    "v242 task-table columns; matching native Name; parsed task timings; "
    "results/profile source; outside HPC resource subtree"
)


def attribute_profile_memory(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Return a detached interpretation without changing raw native evidence.

    The v242 (AEDT 2024.2) task table establishes executable peak RAM only for a
    task row with matching native Name and the real-time/CPU-time/Memory column
    schema. HPC (high-performance computing) resource information does not inherit
    that definition from a Memory label.
    """
    result = deepcopy(benchmark)
    source_bound = result.get("native_source") == NATIVE_PROFILE_SOURCE
    for profile in result.get("profiles", []):
        for group in profile["process_groups"]:
            _attribute_node(
                group, (profile["native_setup_profile"],), False, source_bound
            )
    result["memory_interpretation"] = {
        "basis": "contextual v242 task-table attribution",
        "raw_native_values": "preserved",
        "unqualified_memory": "unknown scope; excluded from task-peak aggregates",
        "missing_qualified_peak": "unavailable; no resource or OS fallback",
    }
    return result


def _attribute_node(
    node: dict[str, Any],
    ancestry: tuple[str, ...],
    in_resources: bool,
    source_bound: bool,
) -> list[dict[str, Any]]:
    """Label this node's memory, then its subtree's; return qualified task peaks.

    Children are labelled before this node's own stage peak is built, because the
    stage peak is the maximum over all qualified descendants.
    """
    path = (*ancestry, node["name"])
    in_resources = in_resources or node["name"] == "HPC Group"
    metrics = node["metrics"]
    qualified = _is_task_row(node, source_bound, in_resources)
    own = []
    if "memory_gb_sdk" in metrics:
        _label_memory_scope(metrics, qualified)
        if qualified:
            own.append({"node": node, "path": path})
    descendants = []
    for child in node.get("children", []):
        descendants.extend(_attribute_node(child, path, in_resources, source_bound))
    # Stored historical aggregates may include unqualified Machine information.
    # Rebuild only the detached attribution, never the receipt-bound artifact.
    node.pop("stage_memory_peak", None)
    if descendants:
        node["stage_memory_peak"] = _stage_memory_peak(descendants)
    return own + descendants


def _is_task_row(node: dict[str, Any], source_bound: bool, in_resources: bool) -> bool:
    """True when Memory is the peak of one task: right source, columns and Name."""
    properties = node["native_properties"]
    metrics = node["metrics"]
    return (
        source_bound
        and not in_resources
        and properties.get("Name") == node["name"]
        and all(key in properties for key in _TASK_NATIVE_COLUMNS)
        and all(key in metrics for key in _TASK_METRICS)
    )


def _label_memory_scope(metrics: dict[str, Any], qualified: bool) -> None:
    """Record, beside the raw value, whether this Memory is a task peak or unknown."""
    metrics["memory_scope"] = _QUALIFIED_SCOPE if qualified else _UNQUALIFIED_SCOPE
    metrics["memory_scope_basis"] = (
        _QUALIFIED_BASIS if qualified else _UNQUALIFIED_BASIS
    )
    metrics["memory_task_peak_qualified"] = qualified


def _stage_memory_peak(descendants: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the largest qualified task peak among a stage's descendants."""
    peak = max(descendants, key=lambda item: item["node"]["metrics"]["memory_gb_sdk"])
    peak_metrics = peak["node"]["metrics"]
    return {
        "stage": peak["node"]["name"],
        "stage_path": list(peak["path"]),
        "memory_native": peak_metrics["memory_native"],
        "memory_gb_sdk": peak_metrics["memory_gb_sdk"],
        "scope": "maximum observed qualified descendant task peak, not host peak",
        "basis": peak_metrics["memory_scope_basis"],
    }
