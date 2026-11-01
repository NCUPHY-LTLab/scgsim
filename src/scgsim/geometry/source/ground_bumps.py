"""Prepare source-authored ground-bump fill without taking backend ownership."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.geometry.source.ground_bump_geometry.audit import (
    AuthoredSiteAuditContext,
    audit_authored_indium_sites,
)
from scgsim.geometry.source.ground_bump_geometry.lattice import (
    LatticeSelectionContext,
    accepted_site_records,
    select_lattice_sites,
)
from scgsim.geometry.source.ground_bump_geometry.layout import (
    assert_generated_center_selectors,
    component_with_canonical_bumps,
    gdstk_cell,
    layer_pair,
    point2,
    relative_polygon,
)
from scgsim.geometry.source.ground_bump_geometry.records import (
    FillRecordContext,
    fill_semantic_records,
)


def prepare_indium_ground_bump_fill(
    *,
    component: Any,
    stack: Mapping[str, Any],
    fill: bool,
    fill_pitch_um: float,
    fill_clearance_um: float,
) -> dict[str, Any]:
    """Materialize PDK-defined ground-bump fill without mutating its input cell.

    The PDK owns bump polygons, layers, physical origin, and keepout meaning.
    SCGSim only enumerates the explicit lattice and carries the resulting
    per-site records through the existing Route-A/B planner.
    """
    if not isinstance(fill, bool):
        raise TypeError("fill must be a bool.")
    pitch = _positive_number(fill_pitch_um, "fill_pitch_um")
    clearance = _non_negative_number(fill_clearance_um, "fill_clearance_um")
    result_stack = deepcopy(dict(stack))
    layers = result_stack.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, str | bytes):
        raise TypeError("indium ground bumps require a semantic layer sequence.")
    has_authored_bumps = any(
        isinstance(record, Mapping) and record.get("part_role") == "bump_body"
        for record in layers
    )
    if not fill and not has_authored_bumps:
        receipt: dict[str, Any] = {
            "schema_version": 2,
            "rule": {
                "identity": "scgsim.indium_ground_bump_fill.v1",
                "version": 1,
            },
            "source": "scgsim.sgb.ground_bumps._prepare_indium_ground_bump_fill",
            "pdk_spec": None,
            "pdk_spec_sha256": None,
            "controls": {
                "fill": False,
                "fill_pitch_um": pitch,
                "fill_clearance_um": clearance,
            },
            "authored_sites": [],
            "accepted": [],
            "rejected": [],
            "semantic_contract": None,
            "counts": {"authored": 0, "generated": 0, "rejected": 0},
        }
        _set_indium_fill_metadata(result_stack, receipt)
        return {
            "component": component,
            "stack": result_stack,
            "injected_entities": (),
            "injected_polygons": (),
            "receipt": receipt,
        }

    return _prepare_existing_bump_fill(
        component=component,
        result_stack=result_stack,
        fill=fill,
        pitch=pitch,
        clearance=clearance,
    )


def _prepare_existing_bump_fill(
    *,
    component: Any,
    result_stack: dict[str, Any],
    fill: bool,
    pitch: float,
    clearance: float,
) -> dict[str, Any]:
    """Audit authored geometry, then either report or generate PDK fill."""
    import gdstk  # noqa: PLC0415  # Optional dependency; the empty path needs no GDS tools.

    from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec  # noqa: PLC0415  # No-op requests return before source preparation.
    from scgsim.geometry.source.adapter import build_gds_stack_geometry_input  # noqa: PLC0415  # No-op requests return before source preparation.

    template, spec = _ground_bump_fill_spec(result_stack)
    origin = point2(spec["lattice_origin_um"], "lattice_origin_um")
    bump_layer = layer_pair(spec["body_layer"], "body_layer")
    under_bump_layer = layer_pair(spec["contact_layer"], "contact_layer")
    body_polygon = relative_polygon(spec["body_polygon_um"], "body_polygon_um")
    contact_polygon = relative_polygon(spec["contact_polygon_um"], "contact_polygon_um")
    template_semantic_id = str(template["semantic_id"])
    fill_prefix = f"{template_semantic_id}_GROUND_FILL"
    geometry_digest = hashlib.sha256(
        json.dumps(
            {
                "body_layer": bump_layer,
                "body_polygon_um": body_polygon,
                "contact_layer": under_bump_layer,
                "contact_polygon_um": contact_polygon,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with TemporaryDirectory(prefix="scgsim-indium-fill-") as temporary:
        temporary_path = Path(temporary)
        source_gds = temporary_path / "source.gds"
        source_stack = temporary_path / "stack.json"
        component.write_gds(source_gds)
        source_stack.write_text(_json_text(result_stack), encoding="utf-8")
        build_input = build_gds_stack_geometry_input(
            gds_file=source_gds,
            stack_file=source_stack,
            top_cell_name=str(getattr(component, "name", "") or "") or None,
        )
        library = gdstk.read_gds(str(source_gds))
        cell = gdstk_cell(library, str(getattr(component, "name", "") or ""))
        audit = audit_authored_indium_sites(
            AuthoredSiteAuditContext(
                build_input=build_input,
                cell=cell,
                bump_layer=bump_layer,
                under_bump_layer=under_bump_layer,
                spec=spec,
                semantic_contract=_bump_semantic_contract(template),
            )
        )
        receipt: dict[str, Any] = {
            "schema_version": 2,
            "rule": {
                "identity": "scgsim.indium_ground_bump_fill.v1",
                "version": 1,
            },
            "source": "scgsim.sgb.ground_bumps._prepare_indium_ground_bump_fill",
            "pdk_spec": spec,
            "pdk_spec_sha256": geometry_digest,
            "input": {
                "component_gds_sha256": _sha256_file(source_gds),
                "canonical_stack_sha256": _sha256_file(source_stack),
            },
            "controls": {
                "fill": fill,
                "fill_pitch_um": pitch,
                "fill_clearance_um": clearance,
            },
            "authored_sites": audit["authored_sites"],
            "accepted": [],
            "rejected": [],
            "semantic_contract": audit["semantic_contract"],
        }
        if not fill:
            receipt["counts"] = {
                "authored": len(audit["authored_sites"]),
                "generated": 0,
                "rejected": 0,
            }
            _set_indium_fill_metadata(result_stack, receipt)
            return {
                "component": component,
                "stack": result_stack,
                "injected_entities": (),
                "injected_polygons": (),
                "receipt": receipt,
            }

        selection_context = LatticeSelectionContext(
            gdstk=gdstk,
            origin=origin,
            pitch=pitch,
            clearance=clearance,
            body_polygon=body_polygon,
            contact_polygon=contact_polygon,
            fill_prefix=fill_prefix,
            geometry_digest=geometry_digest,
            semantic_contract=audit["semantic_contract"],
            ground=audit["ground_intersection"],
            blockers=list(audit["blockers"]),
        )
        accepted, rejected = select_lattice_sites(
            selection_context, _coupon_bounds(result_stack)
        )
    return _complete_generated_fill(
        _GeneratedFillContext(
            component=component,
            result_stack=result_stack,
            pitch=pitch,
            receipt=receipt,
            template_semantic_id=template_semantic_id,
            spec=spec,
            accepted=accepted,
            rejected=rejected,
            body_polygon=body_polygon,
            bump_layer=bump_layer,
            fill_prefix=fill_prefix,
            geometry_digest=geometry_digest,
            layout_polygon_spec=LayoutPolygonSpec,
            semantic_entity_spec=SemanticEntitySpec,
            selection_context=selection_context,
        )
    )


@dataclass(frozen=True)
class _GeneratedFillContext:
    """Carry selected sites and source facts through final materialization."""

    component: Any
    result_stack: dict[str, Any]
    pitch: float
    receipt: dict[str, Any]
    template_semantic_id: str
    spec: Mapping[str, Any]
    accepted: Sequence[tuple[int, int, float, float]]
    rejected: list[dict[str, Any]]
    body_polygon: Sequence[tuple[float, float]]
    bump_layer: tuple[int, int]
    fill_prefix: str
    geometry_digest: str
    layout_polygon_spec: Any
    semantic_entity_spec: Any
    selection_context: LatticeSelectionContext


def _complete_generated_fill(context: _GeneratedFillContext) -> dict[str, Any]:
    """Build the derived component, semantic records and completed receipt."""
    derived = component_with_canonical_bumps(
        context.component, context.spec, context.accepted, context.pitch
    )
    assert_generated_center_selectors(derived, context.bump_layer, context.accepted)
    _exclude_generated_bumps_from_authored_layer(
        context.result_stack, context.template_semantic_id, context.accepted
    )
    fill_entities, fill_polygons = fill_semantic_records(
        FillRecordContext(
            accepted=context.accepted,
            fill_prefix=context.fill_prefix,
            template_semantic_id=context.template_semantic_id,
            bump_layer=context.bump_layer,
            body_polygon_um=context.body_polygon,
            result_stack=context.result_stack,
            semantic_contract=context.receipt["semantic_contract"],
            layout_polygon_spec=context.layout_polygon_spec,
            semantic_entity_spec=context.semantic_entity_spec,
        )
    )
    context.receipt["accepted"] = accepted_site_records(
        context.selection_context, context.accepted
    )
    context.receipt["rejected"] = context.rejected
    context.receipt["counts"] = {
        "authored": len(context.receipt["authored_sites"]),
        "generated": len(context.accepted),
        "rejected": len(context.rejected),
    }
    _set_indium_fill_metadata(context.result_stack, context.receipt)
    return {
        "component": derived,
        "stack": context.result_stack,
        "injected_entities": tuple(fill_entities),
        "injected_polygons": tuple(fill_polygons),
        "receipt": context.receipt,
    }


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ground_bump_fill_spec(
    stack: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    layers = stack.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, str | bytes):
        raise TypeError("ground-bump fill requires a semantic layer sequence.")
    matches = [
        record
        for record in layers
        if isinstance(record, Mapping)
        and isinstance(record.get("metadata"), Mapping)
        and "ground_bump_fill_spec" in record["metadata"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "ground-bump fill requires exactly one PDK ground_bump_fill_spec."
        )
    template = matches[0]
    raw = template["metadata"]["ground_bump_fill_spec"]
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("unsupported PDK ground_bump_fill_spec schema.")
    required = {
        "schema_version",
        "body_layer",
        "body_polygon_um",
        "contact_layer",
        "contact_polygon_um",
        "lattice_origin_um",
        "keepout_records",
    }
    if set(raw) != required:
        raise ValueError("PDK ground_bump_fill_spec fields are not canonical.")
    return template, deepcopy(dict(raw))


def _bump_semantic_contract(template: Mapping[str, Any]) -> dict[str, Any]:
    metadata = template.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("ground-bump template metadata must be a mapping.")
    owners = metadata.get("owner_semantic_ids")
    if (
        isinstance(owners, str | bytes)
        or not isinstance(owners, Sequence)
        or len(owners) != 2
        or any(not isinstance(owner, str) or not owner for owner in owners)
    ):
        raise ValueError("ground-bump template requires exactly two structured owners.")
    net_id = template.get("net_id")
    equipotential_id = metadata.get("equipotential_id")
    host = template.get("host_void_semantic_id")
    if any(
        not isinstance(value, str) or not value
        for value in (net_id, equipotential_id, host)
    ):
        raise ValueError(
            "ground-bump template lacks exact net/equipotential/host facts."
        )
    return {
        "owner_semantic_ids": tuple(owners),
        "net_id": net_id,
        "equipotential_id": equipotential_id,
        "host_solution_volume_id": host,
    }


def _coupon_bounds(stack: Mapping[str, Any]) -> Mapping[str, float]:
    metadata = stack.get("metadata")
    raw = (
        metadata.get("coupon_domain_bounds_um")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(raw, Mapping):
        raise TypeError("ground-bump fill requires explicit coupon_domain_bounds_um.")
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    if any(key not in raw for key in required):
        raise ValueError("coupon bounds must define x/y min/max.")
    return {key: float(raw[key]) for key in required}


def _exclude_generated_bumps_from_authored_layer(
    stack: dict[str, Any],
    template_semantic_id: str,
    accepted: Sequence[tuple[int, int, float, float]],
) -> None:
    layers = stack.get("layers")
    if not isinstance(layers, list):
        raise TypeError("fill stack needs a mutable layer list.")
    for record in layers:
        if (
            isinstance(record, dict)
            and record.get("semantic_id") == template_semantic_id
        ):
            geometry = dict(record.get("geometry", {}))
            geometry.update(
                {
                    "route_ab_fused_selector_mode": True,
                    "exclude_selector_points_um": [(x, y) for _, _, x, y in accepted],
                }
            )
            record["geometry"] = geometry
            record["metadata"] = {
                **dict(record.get("metadata", {})),
                "source_provenance": "authored_indium_bump",
            }
            return
    raise ValueError("fill stack lacks its authored ground-bump layer.")


def _set_indium_fill_metadata(
    stack: dict[str, Any], receipt: Mapping[str, Any]
) -> None:
    metadata = dict(stack.get("metadata", {}))
    metadata["indium_ground_bumps"] = {
        "controls": dict(receipt["controls"]),
        "counts": dict(receipt["counts"]),
        "source": receipt["source"],
    }
    stack["metadata"] = metadata


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result
