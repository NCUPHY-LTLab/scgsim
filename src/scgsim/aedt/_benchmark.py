"""Pure contextual attribution of captured AEDT profile memory.

Native acquisition and unit conversion belong to runtime; this detached projection
owns the meaning and aggregation of memory in both capture and offline reports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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

    def visit(
        node: dict[str, Any], ancestry: tuple[str, ...], in_resources: bool
    ) -> list[dict[str, Any]]:
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
                key in metrics
                for key in ("real_seconds", "cpu_seconds", "memory_gb_sdk")
            )
        )
        candidates = []
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
        descendants = []
        for child in node.get("children", []):
            descendants.extend(visit(child, path, in_resources))
        # Stored historical aggregates may include unqualified Machine information.
        # Rebuild only the detached attribution, never the receipt-bound artifact.
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

    for profile in result.get("profiles", []):
        for group in profile["process_groups"]:
            visit(group, (profile["native_setup_profile"],), False)
    result["memory_interpretation"] = {
        "basis": "contextual v242 task-table attribution",
        "raw_native_values": "preserved",
        "unqualified_memory": "unknown scope; excluded from task-peak aggregates",
        "missing_qualified_peak": "unavailable; no resource or OS fallback",
    }
    return result
