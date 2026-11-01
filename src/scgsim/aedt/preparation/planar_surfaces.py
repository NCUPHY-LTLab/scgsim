"""Surface-EPR contributions of a planar HFSS model: ledger, catalog and bindings.

Owns: reading the planner's surface contribution ledger, the contribution catalog,
and binding each requested `SurfaceEprSpec` to the surfaces, adjacent solution
domains and materials that establish it. Does not own: the KLayout mask-support
unions (`planar.py`), the source record (`planar_source.py`) or the numerical EPR
evaluation (`scgsim.aedt.epr`).

Interface kinds, by the two media on either side of a surface: MA is metal-air,
MS metal-substrate and SA substrate-air (docs/eigenmode-epr.qmd). MM is a
metal-metal contact (scgsim/geometry/planning/interfaces.py,
scgsim/palace/mesh/record_checks.py).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.aedt._data import canonical_sha256
from scgsim.aedt.preparation.planar_source import _plain_geometry_source
from scgsim.geometry._primitives.spatial import _geometry_ref_surface_z_um

# Interface kinds a catalog entry can describe, and the field sides it can face.
_CATALOG_INTERFACE_KINDS = {"MA", "MS", "SA", "MM"}
_FIELD_SIDES = {"top", "bottom"}
# A loop-bounded surface that is the top or bottom face of a shell carries its own Z.
_HORIZONTAL_SHELL_PARTS = {"top", "bottom"}


def _surface_ledger(planned_surfaces) -> list[dict[str, Any]]:
    """List one horizontal surface per contribution ledger entry of the planner.

    Sidewall entries are skipped: sidewall Surface-EPR is excluded.
    """
    surfaces: list[dict[str, Any]] = []
    for surface in planned_surfaces:
        provenance = surface.metadata.get("source_provenance")
        if not isinstance(provenance, Mapping):
            continue
        ledger = provenance.get("surface_contribution_ledger")
        if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
            continue
        for record in ledger:
            if not isinstance(record, Mapping):
                raise TypeError("surface contribution ledger entries must be mappings")
            if record.get("side") == "sidewall":
                continue
            surfaces.append(
                {
                    "surface_id": surface.surface_id,
                    "owner_semantic_id": surface.owner_semantic_id,
                    "geometry_ref": _plain_geometry_source(surface.geometry_ref),
                    "normal_hint": _plain_geometry_source(surface.normal_hint),
                    "surface_role": surface.surface_role,
                    "mask_plane": _surface_mask_plane(surface.geometry_ref),
                    "contribution": _plain_geometry_source(record),
                }
            )
    return surfaces


def _surface_mask_plane(geometry_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Z plane and loops that mask a horizontal surface."""
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if (
            not isinstance(plane, Mapping) or plane.get("axis") != "z"
        ) and geometry_ref.get("shell_part") not in _HORIZONTAL_SHELL_PARTS:
            raise ValueError("planar EPR surface loop requires an explicit Z plane")
        if geometry_ref.get("shell_part") not in _HORIZONTAL_SHELL_PARTS and (
            isinstance(plane.get("value_um"), bool)
            or not isinstance(plane.get("value_um"), (int, float))
        ):
            raise ValueError("planar EPR surface requires finite plane Z")
        z_um = _geometry_ref_surface_z_um(geometry_ref)
        if not math.isfinite(z_um):
            raise ValueError("planar EPR surface requires finite plane Z")
        return {
            "origin_um": [0.0, 0.0, z_um],
            "u": [1.0, 0.0, 0.0],
            "v": [0.0, 1.0, 0.0],
            "exterior": _plain_geometry_source(outer),
            "holes": _plain_geometry_source(geometry_ref.get("hole_loops", ())),
        }
    raise ValueError("horizontal EPR surface requires a planar loop")


def _plane_group_key(binding: Mapping[str, Any]) -> tuple[Any, ...]:
    """Key that groups bindings sharing a plane, side, classification and material.

    The normal is made sign-independent (first significant component positive) and
    rounded, so that coplanar surfaces whose axes differ only in handedness match.
    """
    plane = binding["mask_plane"]
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    length = math.sqrt(sum(value * value for value in normal))
    normal = tuple(value / length for value in normal)
    first = next((value for value in normal if abs(value) > 1e-12), 1.0)
    if first < 0.0:
        normal = tuple(-value for value in normal)
    origin = tuple(float(value) for value in plane["origin_um"])
    offset = sum(normal[index] * origin[index] for index in range(3))
    evidence = binding["contribution"]
    return (
        *(round(value, 12) for value in normal),
        round(offset, 12),
        evidence["classification"],
        evidence["side"],
        binding["effective_material_id"],
    )


def _project_plane_region(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    """Express a region given in `source` plane coordinates in `target` coordinates."""
    source_origin = tuple(float(value) for value in source["origin_um"])
    source_u = tuple(float(value) for value in source["u"])
    source_v = tuple(float(value) for value in source["v"])
    target_origin = tuple(float(value) for value in target["origin_um"])
    target_u = tuple(float(value) for value in target["u"])
    target_v = tuple(float(value) for value in target["v"])

    def project(point: Sequence[float]) -> list[float]:
        global_point = tuple(
            source_origin[index]
            + float(point[0]) * source_u[index]
            + float(point[1]) * source_v[index]
            for index in range(3)
        )
        delta = tuple(global_point[index] - target_origin[index] for index in range(3))
        return [
            sum(delta[index] * target_u[index] for index in range(3)),
            sum(delta[index] * target_v[index] for index in range(3)),
        ]

    return {
        "exterior": [project(point) for point in source["exterior"]],
        "holes": [
            [project(point) for point in ring] for ring in source.get("holes", ())
        ],
    }


def _source_polygon_ids(
    evidence: Mapping[str, Any], interface_kind: str, surface_id: str
) -> tuple:
    """Source polygons a ledger entry retains.

    An SA surface lies on no authored polygon, so it is identified by itself.
    """
    source_ids = tuple(evidence.get("outer_source_ids") or ()) + tuple(
        evidence.get("hole_source_ids") or ()
    )
    if not source_ids and interface_kind == "SA":
        source_ids = (surface_id,)
    return source_ids


def _binding_id(contribution_id: str, surface: Mapping[str, Any]) -> str:
    """Name one surface's binding to a contribution, stable across runs."""
    digest = canonical_sha256(
        {"surface_id": surface["surface_id"], "geometry_ref": surface["geometry_ref"]}
    )[:16]
    return f"{contribution_id}__{digest}"


def _contribution_catalog(surfaces: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Merge ledger entries into one catalog record per contribution id, sorted by id."""
    catalog_by_id: dict[str, dict[str, Any]] = {}
    for item in surfaces:
        evidence = item["contribution"]
        contribution_id = evidence.get("contribution_id")
        classification = evidence.get("classification")
        side = evidence.get("side")
        if (
            not isinstance(contribution_id, str)
            or classification not in _CATALOG_INTERFACE_KINDS
            or side not in _FIELD_SIDES
        ):
            continue
        record = catalog_by_id.setdefault(
            contribution_id,
            {
                "contribution_id": contribution_id,
                "interface_kind": classification,
                "field_side": side,
                "source_polygon_ids": [],
                "effective_domain_ids": [],
                "surface_roles": [],
            },
        )
        if record["interface_kind"] != classification or record["field_side"] != side:
            raise RuntimeError(
                f"surface contribution {contribution_id!r} has contradictory topology"
            )
        for source_id in _source_polygon_ids(
            evidence, classification, item["surface_id"]
        ):
            if source_id not in record["source_polygon_ids"]:
                record["source_polygon_ids"].append(source_id)
        for domain_id in evidence.get("effective_domain_ids", ()):
            if domain_id not in record["effective_domain_ids"]:
                record["effective_domain_ids"].append(domain_id)
        if item["surface_role"] not in record["surface_roles"]:
            record["surface_roles"].append(item["surface_role"])
    return tuple(catalog_by_id[key] for key in sorted(catalog_by_id))


def _support_bindings(
    surfaces: list[dict[str, Any]],
    contributions: Sequence[Any],
    regions_by_id: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind every surface of a requested side and kind as potential mask support.

    The union of these supports bounds each requested surface (see
    `_bind_physical_support_masks`), so supports include surfaces nobody asked
    for by id, as long as their classification and side were requested.
    """
    support_bindings: list[dict[str, Any]] = []
    requested_support_sides = {
        (item.interface_kind, item.field_side) for item in contributions
    }
    for surface in surfaces:
        evidence = surface["contribution"]
        if (
            evidence.get("classification"),
            evidence.get("side"),
        ) not in requested_support_sides:
            continue
        domain_ids = tuple(evidence.get("effective_domain_ids", ()))
        expected_count = 2 if evidence["classification"] == "SA" else 1
        if len(domain_ids) != expected_count:
            raise ValueError("physical EPR support has inconsistent adjacent domains")
        domains = [regions_by_id.get(domain_id) for domain_id in domain_ids]
        if any(domain is None for domain in domains):
            raise ValueError(
                "physical EPR support domain is absent from prepared_stack"
            )
        if evidence["classification"] == "SA":
            field_domains = [
                domain
                for domain in domains
                if materials.get(domain["material_id"], {}).get("kind") == "vacuum"
            ]
            if len(field_domains) != 1:
                raise ValueError("physical SA support needs one vacuum field domain")
            field_domain = field_domains[0]
        else:
            field_domain = domains[0]
        support_bindings.append(
            {
                **surface,
                "binding_id": _binding_id(evidence["contribution_id"], surface),
                "effective_material_id": field_domain["material_id"],
            }
        )
    return support_bindings


def _resolve_contributions(
    contributions: Sequence[Any],
    surfaces: list[dict[str, Any]],
    regions_by_id: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind each requested contribution to every surface that establishes it."""
    resolved_surfaces: list[dict[str, Any]] = []
    for requested in contributions:
        matches = [
            item
            for item in surfaces
            if item["contribution"].get("contribution_id") == requested.contribution_id
        ]
        if not matches:
            raise ValueError(
                f"contribution {requested.contribution_id!r} is not established by "
                "the shared source topology"
            )
        for match in matches:
            resolved_surfaces.append(
                _resolve_match(requested, match, regions_by_id, materials)
            )
    return resolved_surfaces


def _resolve_match(requested, match, regions_by_id, materials) -> dict[str, Any]:
    """Check one surface against the request and return its resolved binding."""
    evidence = match["contribution"]
    _check_request_matches_evidence(requested, evidence, match["surface_id"])
    adjacent_domains = _adjacent_domains(requested, evidence, regions_by_id)
    domain, substrate_domain = _field_and_substrate_domains(
        requested, adjacent_domains, materials
    )
    if domain is None:
        raise ValueError(
            f"contribution {requested.contribution_id!r} field-side domain "
            "is absent from prepared_stack"
        )
    material = materials.get(domain["material_id"])
    if not isinstance(material, Mapping):
        raise ValueError(
            f"contribution {requested.contribution_id!r} field-side material "
            "is absent from prepared_stack"
        )
    expected_kind = {
        "MA": "vacuum",
        "MS": "dielectric",
        "SA": "vacuum",
    }.get(requested.interface_kind)
    if expected_kind is not None and material.get("kind") != expected_kind:
        raise ValueError(
            f"contribution {requested.contribution_id!r} field-side material "
            f"must be {expected_kind}"
        )
    substrate_material = (
        materials.get(substrate_domain["material_id"])
        if substrate_domain is not None
        else None
    )
    if requested.interface_kind in {"MS", "SA"}:
        _check_substrate_permittivity(requested, substrate_material)
    return {
        **match,
        "binding_id": _binding_id(requested.contribution_id, match),
        "effective_domain_id": domain["semantic_id"],
        "substrate_domain_id": (
            substrate_domain["semantic_id"] if substrate_domain is not None else None
        ),
        "effective_material_id": domain["material_id"],
        "effective_material": _plain_geometry_source(material),
        "substrate_material": (
            _plain_geometry_source(substrate_material)
            if substrate_material is not None
            else None
        ),
        "margins_um": list(requested.margins_um),
        "film_thickness_m": requested.film_thickness_m,
        "film_relative_permittivity": requested.film_relative_permittivity,
    }


def _check_request_matches_evidence(requested, evidence, surface_id: str) -> None:
    """Require the request to restate the planner's classification, side and polygon."""
    if evidence.get("classification") != requested.interface_kind:
        raise ValueError(
            f"contribution {requested.contribution_id!r} classification "
            "contradicts shared source evidence"
        )
    if evidence.get("side") != requested.field_side:
        raise ValueError(
            f"contribution {requested.contribution_id!r} side contradicts "
            "shared source evidence"
        )
    source_polygon_ids = _source_polygon_ids(
        evidence, requested.interface_kind, surface_id
    )
    if requested.source_polygon_id not in source_polygon_ids:
        raise ValueError(
            f"contribution {requested.contribution_id!r} does not retain "
            f"source polygon {requested.source_polygon_id!r}"
        )


def _adjacent_domains(requested, evidence, regions_by_id) -> list:
    """Return the solution domains on the surface's two sides (one for MA / MS).

    An SA surface sits between a vacuum and a dielectric domain; MA and MS
    surfaces are described by their single field-side domain.
    """
    effective_domains = tuple(evidence.get("effective_domain_ids", ()))
    expected_domain_count = 2 if requested.interface_kind == "SA" else 1
    if len(effective_domains) != expected_domain_count:
        raise ValueError(
            f"contribution {requested.contribution_id!r} does not have "
            f"{expected_domain_count} physical adjacent domain(s)"
        )
    adjacent_domains = [regions_by_id.get(item) for item in effective_domains]
    if any(item is None for item in adjacent_domains):
        raise ValueError(
            f"contribution {requested.contribution_id!r} adjacent domain "
            "is absent from prepared_stack"
        )
    return adjacent_domains


def _field_and_substrate_domains(requested, adjacent_domains, materials):
    """Pick the field-side domain and the substrate domain, if the kind has one.

    MS surfaces use their one domain as both; MA has no substrate domain.
    """
    if requested.interface_kind == "SA":
        vacuum_domains = [
            item
            for item in adjacent_domains
            if materials.get(item["material_id"], {}).get("kind") == "vacuum"
        ]
        dielectric_domains = [
            item
            for item in adjacent_domains
            if materials.get(item["material_id"], {}).get("kind") == "dielectric"
        ]
        if len(vacuum_domains) != 1 or len(dielectric_domains) != 1:
            raise ValueError(
                f"contribution {requested.contribution_id!r} must bind one "
                "vacuum and one dielectric domain"
            )
        return vacuum_domains[0], dielectric_domains[0]
    domain = adjacent_domains[0]
    return domain, (domain if requested.interface_kind == "MS" else None)


def _check_substrate_permittivity(requested, substrate_material) -> None:
    permittivity = (
        substrate_material.get("permittivity")
        if isinstance(substrate_material, Mapping)
        else None
    )
    if (
        isinstance(permittivity, bool)
        or not isinstance(permittivity, (int, float))
        or not math.isfinite(float(permittivity))
        or float(permittivity) <= 0.0
    ):
        raise ValueError(
            f"contribution {requested.contribution_id!r} requires positive "
            "substrate permittivity from prepared_stack"
        )
