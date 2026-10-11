"""EPR analysis-sheet naming and inset planning over prepared source geometry.

General source preparation is owned by preparation.planar; public preparation
helpers are direct reexports of those same function objects.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
from multiprocessing import get_context
from typing import Any
from scgsim.aedt.specs.epr import (
    surface_evaluations,
)
from scgsim.aedt.preparation.planar import (
    _klayout_region as _klayout_region,
    _plain as _plain,
    _regions_from_klayout as _regions_from_klayout,
    planar_junction_from_port as planar_junction_from_port,
    prepare_planar_geometry_input as prepare_planar_geometry_input,
    validate_geometry_workers as validate_geometry_workers,
)
from scgsim.aedt._data import canonical_sha256
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.preparation.planar import (
    _INSET_METHOD,
    _INSET_CIRCLE_POINTS,
)


def _sheet_member(binding: Mapping[str, Any], margin_label: str) -> tuple[str, ...]:
    contribution = binding["contribution"]
    return (
        str(binding["owner_semantic_id"]),
        str(binding["surface_role"]),
        str(contribution["classification"]),
        str(contribution["side"]),
        margin_label,
        str(contribution["contribution_id"]),
        str(binding["binding_id"]),
    )


def _sheet_name(
    members: set[tuple[str, ...]], geometry_sha256: str, *, parts: set[int]
) -> str:
    ordered = sorted(members)
    if not ordered:
        raise ValueError("analysis sheet has no physical membership")

    def label(values: set[str]) -> str:
        raw = "_".join(sorted(values))
        safe = "".join(
            character if character.isascii() and character.isalnum() else "_"
            for character in raw
        ).strip("_")
        return safe or "Unknown"

    owners = {member[0] for member in ordered}
    owner = (
        label(owners)
        if len(ordered) == 1
        else (f"Shared_{label(owners)}" if len(owners) == 1 else "Shared")
    )
    classification, side, margin = (
        label({member[index] for member in ordered}) for index in (2, 3, 4)
    )
    digest = canonical_sha256(
        {"geometry_sha256": geometry_sha256, "members": ordered, "parts": sorted(parts)}
    )[:12]
    part_label = (
        ""
        if not parts
        else "_" + "_".join(f"Part{index:02d}" for index in sorted(parts))
    )
    tail = f"_{classification}_{side}_{margin}{part_label}_{digest}"
    owner_budget = 60 - len("EPR_") - len(tail)
    if owner_budget < 1:
        raise ValueError(
            "EPR analysis sheet semantic labels exceed AEDT's 60-character limit"
        )
    return f"EPR_{owner[:owner_budget].rstrip('_')}{tail}"


def _base_sheet_plan(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_geometry: dict[str, set[tuple[str, ...]]] = {}
    keys: dict[str, str] = {}
    for binding in bindings:
        binding_id = str(binding["binding_id"])
        key = canonical_sha256(binding["geometry_ref"])
        keys[binding_id] = key
        by_geometry.setdefault(key, set()).add(_sheet_member(binding, "Unmasked"))
    names = {
        key: _sheet_name(members, key, parts=set())
        for key, members in by_geometry.items()
    }
    if len(set(names.values())) != len(names):
        raise RuntimeError("distinct EPR base sheet geometries share a native name")
    return {
        binding_id: {
            "name": names[key],
            "members": sorted(by_geometry[key]),
            "geometry_sha256": key,
        }
        for binding_id, key in keys.items()
    }


def plan_inset_sheet_names(
    bindings: Sequence[Mapping[str, Any]],
    inset_plan: Mapping[tuple[str, float], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Name each actual inset contour from all of its physical memberships."""

    grouped: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        binding_id = str(binding["binding_id"])
        dbu_um = float(binding["mask_support"]["source_dbu_um"])
        for _, margin in surface_evaluations(
            binding["margins_um"], policy="unmasked_plus_requested.v1"
        ):
            key = (binding_id, float(margin))
            if key not in inset_plan:
                continue
            planned = inset_plan[key]
            effective_nm = float(planned["effective_margin_um"]) * 1000.0
            margin_label = (
                "Unmasked" if effective_nm == 0.0 else f"Margin_{effective_nm:.12g}nm"
            )
            regions = planned["regions"]
            for index, region in enumerate(regions):
                key = canonical_sha256(
                    {
                        "plane": binding["mask_plane"],
                        "region": region,
                        "method": _INSET_METHOD,
                        "source_dbu_um": dbu_um,
                        "klayout_version": version("klayout"),
                    }
                )
                record = grouped.setdefault(key, {"members": set(), "parts": set()})
                record["members"].add(_sheet_member(binding, margin_label))
                if len(regions) > 1:
                    record["parts"].add(index + 1)
    names = {
        key: _sheet_name(record["members"], key, parts=record["parts"])
        for key, record in grouped.items()
    }
    if len(set(names.values())) != len(names):
        raise RuntimeError("distinct EPR inset sheet geometries share a native name")
    return {
        key: {
            "name": names[key],
            "members": sorted(record["members"]),
            "geometry_sha256": key,
        }
        for key, record in grouped.items()
    }


_KLAYOUT_COORD_MIN = -(2**31)

_KLAYOUT_COORD_MAX = 2**31 - 1


def _klayout_inset(support: Any, radius: int, k: Any) -> Any:
    if radius == 0:
        return support
    bounds = support.bbox()
    padding = 2 * radius + 1
    if any(
        coordinate < _KLAYOUT_COORD_MIN + radius
        or coordinate > _KLAYOUT_COORD_MAX - radius
        for coordinate in (
            bounds.left - padding,
            bounds.bottom - padding,
            bounds.right + padding,
            bounds.top + padding,
        )
    ):
        raise OverflowError("EPR inset complement exceeds KLayout integer range")
    complement = (
        k.Region(
            k.Box(
                bounds.left - padding,
                bounds.bottom - padding,
                bounds.right + padding,
                bounds.top + padding,
            )
        )
        - support
    )
    circle = k.Polygon.ellipse(
        k.Box(-radius, -radius, radius, radius), _INSET_CIRCLE_POINTS
    )
    return (support - complement.minkowski_sum(circle).merged()).merged()


def inset_surface_regions(
    binding: Mapping[str, Any],
    margin_um: float,
    support_insets: dict[tuple[str, int], Any],
) -> list[dict[str, Any]]:
    """Inset complete physical support, then attribute its surviving pieces."""

    import klayout.db as k

    if not math.isfinite(margin_um) or margin_um < 0.0:
        raise ValueError("surface EPR margin must be finite and nonnegative")
    support = binding["mask_support"]
    if support["klayout_version"] != version("klayout"):
        raise RuntimeError("prepared EPR KLayout version differs from runtime")
    dbu_um = float(support["source_dbu_um"])
    if margin_um / dbu_um >= _KLAYOUT_COORD_MAX:
        raise OverflowError("EPR margin exceeds KLayout integer range")
    radius = k.DPoint(margin_um, 0).to_itype(dbu_um).x
    key = (str(support["group_identity_sha256"]), radius)
    if key not in support_insets:
        physical = _klayout_region(support["support_regions"], dbu_um, k)
        support_insets[key] = _klayout_inset(physical, radius, k)
    attribution = _klayout_region(support["attribution_regions"], dbu_um, k)
    return _regions_from_klayout(support_insets[key] & attribution, dbu_um)


def _inset_group_task(
    task: tuple[dict[str, Any], list[dict[str, Any]]],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Pure integer geometry for one physical support and all its owners/margins."""

    import klayout.db as k

    support, members = task
    if support["klayout_version"] != version("klayout"):
        raise RuntimeError("prepared EPR KLayout version differs from worker runtime")
    dbu_um = float(support["source_dbu_um"])
    physical = _klayout_region(support["support_regions"], dbu_um, k)
    insets: dict[int, Any] = {0: physical}
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for member in members:
        attribution = _klayout_region(member["attribution_regions"], dbu_um, k)
        for requested in member["margins_um"]:
            margin_um = float(requested)
            if not math.isfinite(margin_um) or margin_um < 0:
                raise ValueError("surface EPR margin must be finite and nonnegative")
            if margin_um / dbu_um >= _KLAYOUT_COORD_MAX:
                raise OverflowError("EPR margin exceeds KLayout integer range")
            radius = k.DPoint(margin_um, 0).to_itype(dbu_um).x
            if radius not in insets:
                insets[radius] = _klayout_inset(physical, radius, k)
            regions = _regions_from_klayout(insets[radius] & attribution, dbu_um)
            result[(member["binding_id"], margin_um)] = {
                "regions": regions,
                "requested_margin_um": margin_um,
                "integer_margin_dbu": radius,
                "effective_margin_um": radius * dbu_um,
                "reuse_source_sheet": radius == 0
                and len(regions) == 1
                and (insets[radius] & attribution ^ attribution).is_empty(),
            }
    return result


def precompute_inset_surfaces(
    geometry: PreparedPlanarGeometry,
    request: Any,
    geometry_workers: int | None,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Finish spawned pure mask geometry before an owned Desktop is constructed."""

    validate_geometry_workers(geometry_workers)
    selected = (
        None
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for binding in geometry.surface_bindings:
        if binding["contribution"]["classification"] == "MM" or (
            selected is not None
            and binding["contribution"]["contribution_id"] not in selected
        ):
            continue
        support = _plain(binding["mask_support"])
        key = support["group_identity_sha256"]
        if key not in grouped:
            grouped[key] = (support, [])
        grouped[key][1].append(
            {
                "binding_id": binding["binding_id"],
                "attribution_regions": support["attribution_regions"],
                "margins_um": list(
                    dict.fromkeys(
                        margin
                        for _, margin in surface_evaluations(
                            binding["margins_um"],
                            policy=geometry.source.get("surface_evaluation_policy"),
                        )
                    )
                ),
            }
        )
    tasks = [grouped[key] for key in sorted(grouped)]
    if not tasks:
        return {}
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    workers = min(geometry_workers or available, len(tasks))
    if workers == 1:
        parts = [_inset_group_task(task) for task in tasks]
    else:
        parts = []
        task_iter = iter(tasks)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            pending = {
                pool.submit(_inset_group_task, task)
                for _, task in zip(range(2 * workers), task_iter)
            }
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    parts.append(future.result())
                    try:
                        pending.add(pool.submit(_inset_group_task, next(task_iter)))
                    except StopIteration:
                        pass
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for part in parts:
        if result.keys() & part.keys():
            raise RuntimeError("EPR inset plan has duplicate binding/margin identity")
        result.update(part)
    return result
