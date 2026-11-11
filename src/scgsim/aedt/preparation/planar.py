"""Shared body-first HFSS preparation. Source supports stay distinct from solver treatments and EPR junctions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.data import canonical_sha256
from scgsim.aedt.preparation.junction_partition import (
    junction_geometry,
    partition_junctions,
)
from scgsim.aedt.preparation.modeling import effective_geometry_input
from scgsim.aedt.preparation.planar_masks import (
    bind_physical_support_masks,
    surface_mask_plane,
)
from scgsim.aedt.preparation.planar_source import plain, source_payload
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.aedt.specs.epr import (
    PlanarJunction,
    SurfaceEprSpec,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.specs.support import prepared_lumped_supports
from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    PortSheetRegionRecord,
)
from scgsim.geometry.planning.junction_partition import (
    drawn_exterior,
    junction_terminal_line,
    lattice_rectangle,
)
from scgsim.geometry.planning.surfaces import plan_representation_contribution_geometry
from scgsim.geometry.source.validation import validate_geometry_input


def prepare_planar_geometry_input(  # noqa: PLR0913 -- preserve the supported geometry preparation keyword API.
    build_input: GeometryBuildInput,
    *,
    prepared_stack: Mapping[str, Any],
    modeling: Modeling,
    lumped_rlcs: Sequence[Any] = (),
    junctions: Sequence[PlanarJunction] = (),
    contributions: Sequence[SurfaceEprSpec] = (),
    source_dbu_um: float | None = None,
) -> PreparedPlanarGeometry:
    """Detach explicit source/effective Solid or Thin Film facts for HFSS EPR."""
    return _PlanarPreparation(
        build_input,
        prepared_stack,
        modeling,
        lumped_rlcs,
        junctions,
        contributions,
        source_dbu_um,
    ).run()


@dataclass
class _PlanarPreparation:
    """Carry one source through topology, support binding, and identity stages."""

    build_input: GeometryBuildInput
    prepared_stack: Mapping[str, Any]
    modeling: Modeling
    lumped_rlcs: Sequence[Any]
    junctions: Sequence[PlanarJunction]
    contributions: Sequence[SurfaceEprSpec]
    source_dbu_um: float | None

    def run(self) -> PreparedPlanarGeometry:
        self._validate_source()
        self._validate_treatments()
        self._bind_junctions()
        self._plan_surfaces()
        self._catalog_contributions()
        self._bind_supports()
        self._resolve_contributions()
        return self._assemble_geometry()

    def _validate_source(self) -> None:
        if not isinstance(self.build_input, GeometryBuildInput):
            raise TypeError("build_input must be GeometryBuildInput")
        if self.build_input.boundary_curves or self.build_input.boundary_reconstruction:
            raise NotImplementedError(
                "AEDT planar lowering does not support active source curve intent"
            )
        if not isinstance(self.prepared_stack, Mapping):
            raise TypeError("prepared_stack must be a mapping")
        recorded_dbu = self.build_input.metadata.get("source_dbu_um")
        if self.source_dbu_um is None:
            self.source_dbu_um = recorded_dbu
        elif recorded_dbu is not None and self.source_dbu_um != recorded_dbu:
            raise ValueError("explicit source_dbu_um differs from geometry source DBU")
        if (
            isinstance(self.source_dbu_um, bool)
            or not isinstance(self.source_dbu_um, (int, float))
            or not math.isfinite(float(self.source_dbu_um))
            or self.source_dbu_um <= 0
        ):
            raise ValueError("planar EPR requires a finite positive source_dbu_um")
        self.dbu_um = float(self.source_dbu_um)
        validate_geometry_input(self.build_input)
        self.build_input, self.prepared_stack, self.representation_context = (
            effective_geometry_input(
                self.build_input, self.prepared_stack, self.modeling
            )
        )

    def _validate_treatments(self) -> None:
        self.rlc_tuple = tuple(self.lumped_rlcs)
        if any(not isinstance(item, LumpedRlc) for item in self.rlc_tuple):
            raise TypeError("lumped_rlcs must contain LumpedRlc records")
        if len({item.support_id for item in self.rlc_tuple}) != len(self.rlc_tuple):
            raise ValueError("one support cannot own multiple RLC treatments")
        self.junction_tuple = tuple(self.junctions)
        self.contribution_tuple = tuple(self.contributions)
        if any(not isinstance(item, PlanarJunction) for item in self.junction_tuple):
            raise TypeError("junctions must contain PlanarJunction records")
        if any(
            not isinstance(item, SurfaceEprSpec) for item in self.contribution_tuple
        ):
            raise TypeError("contributions must contain SurfaceEprSpec records")
        if any(item.field_side == "sidewall" for item in self.contribution_tuple):
            raise ValueError(
                "sidewall Surface-EPR is excluded; reprepare with horizontal "
                "top/bottom contributions"
            )

    def _bind_junctions(self) -> None:
        polygons = {item.polygon_id for item in self.build_input.polygons}
        junction_region_ids = {
            item.source_polygon_id for item in self.build_input.port_sheet_regions
        }
        ids = [item.junction_id for item in self.junction_tuple]
        if len(ids) != len(set(ids)):
            raise ValueError("junction ids must be unique")
        contribution_ids = [item.contribution_id for item in self.contribution_tuple]
        if len(contribution_ids) != len(set(contribution_ids)):
            raise ValueError("surface contribution ids must be unique")
        for item in self.junction_tuple:
            if item.source_polygon_id not in polygons | junction_region_ids:
                raise ValueError(
                    f"junction {item.junction_id!r} references unknown polygon "
                    f"{item.source_polygon_id!r}"
                )
            authored, layer = self._authored_sheet(item.source_polygon_id)
            # The authored sheet as its four corners on the grid, as the partition reads it.
            corners = lattice_rectangle(
                authored.exterior,
                self.dbu_um,
                item.direction_xy,
                drawn=drawn_exterior(authored),
                port=f"junction {item.junction_id!r}",
                layer=layer,
            ).corners_um
            junction_terminal_line(
                {"exterior": corners, "holes": authored.holes},
                junction_geometry(item),
                0.0,
            )
        self.build_input = partition_junctions(
            self.build_input, self.junction_tuple, self.dbu_um
        )
        if self.junction_tuple:
            final_polygons = {p.polygon_id: p for p in self.build_input.polygons}
            for junction in self.junction_tuple:
                record = self.build_input.metadata["aedt_junction_partitions"][
                    junction.junction_id
                ]
                polygon = final_polygons[record["central_polygon_id"]]
                junction_terminal_line(
                    {"exterior": polygon.exterior, "holes": polygon.holes},
                    junction_geometry(junction),
                    0.0,
                )

    def _authored_sheet(
        self, polygon_id: str
    ) -> tuple[LayoutPolygonSpec | PortSheetRegionRecord, str]:
        """The polygon or port sheet region a junction is drawn on, and its layer."""
        for polygon in self.build_input.polygons:
            if polygon.polygon_id == polygon_id:
                return polygon, polygon.layer
        region = next(
            r
            for r in self.build_input.port_sheet_regions
            if r.source_polygon_id == polygon_id
        )
        return region, region.source_layer

    def _plan_surfaces(self) -> None:
        self.build_input, self.planned_surfaces = (
            plan_representation_contribution_geometry(
                self.build_input, self.representation_context
            )
        )
        self.surfaces: list[dict[str, Any]] = []
        for surface in self.planned_surfaces:
            provenance = surface.metadata.get("source_provenance")
            if not isinstance(provenance, Mapping):
                continue
            ledger = provenance.get("surface_contribution_ledger")
            if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
                continue
            for record in ledger:
                if not isinstance(record, Mapping):
                    raise TypeError(
                        "surface contribution ledger entries must be mappings"
                    )
                if record.get("side") == "sidewall":
                    continue
                self.surfaces.append(
                    {
                        "surface_id": surface.surface_id,
                        "owner_semantic_id": surface.owner_semantic_id,
                        "geometry_ref": plain(surface.geometry_ref),
                        "normal_hint": plain(surface.normal_hint),
                        "surface_role": surface.surface_role,
                        "mask_plane": surface_mask_plane(surface.geometry_ref),
                        "contribution": plain(record),
                    }
                )

    def _catalog_contributions(self) -> None:
        self.catalog_by_id: dict[str, dict[str, Any]] = {}
        for item in self.surfaces:
            evidence = item["contribution"]
            contribution_id = evidence.get("contribution_id")
            classification = evidence.get("classification")
            side = evidence.get("side")
            if (
                not isinstance(contribution_id, str)
                or classification not in {"MA", "MS", "SA", "MM"}
                or side not in {"top", "bottom"}
            ):
                continue
            record = self.catalog_by_id.setdefault(
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
            if (
                record["interface_kind"] != classification
                or record["field_side"] != side
            ):
                raise RuntimeError(
                    f"surface contribution {contribution_id!r} has contradictory topology"
                )
            source_ids = tuple(evidence.get("outer_source_ids") or ()) + tuple(
                evidence.get("hole_source_ids") or ()
            )
            if not source_ids and classification == "SA":
                source_ids = (item["surface_id"],)
            for source_id in source_ids:
                if source_id not in record["source_polygon_ids"]:
                    record["source_polygon_ids"].append(source_id)
            for domain_id in evidence.get("effective_domain_ids", ()):
                if domain_id not in record["effective_domain_ids"]:
                    record["effective_domain_ids"].append(domain_id)
            if item["surface_role"] not in record["surface_roles"]:
                record["surface_roles"].append(item["surface_role"])
        self.contribution_catalog = tuple(
            self.catalog_by_id[key] for key in sorted(self.catalog_by_id)
        )

    def _bind_supports(self) -> None:
        self.source = source_payload(
            self.build_input, self.prepared_stack, source_dbu_um=self.dbu_um
        )
        if self.build_input.lumped_supports:
            self.source["lumped_supports"] = prepared_lumped_supports(
                self.build_input.lumped_supports, self.source
            )
        support_ids = {
            item["support_id"] for item in self.source.get("lumped_supports", ())
        }
        if any(item.support_id not in support_ids for item in self.rlc_tuple):
            raise ValueError("RLC treatment references an undeclared neutral support")
        self.regions_by_id = {
            item["semantic_id"]: item for item in self.source["solution_regions"]
        }
        self.materials = self.source["materials"]
        self.support_bindings: list[dict[str, Any]] = []
        requested_support_sides = {
            (item.interface_kind, item.field_side) for item in self.contribution_tuple
        }
        for surface in self.surfaces:
            evidence = surface["contribution"]
            if (
                evidence.get("classification"),
                evidence.get("side"),
            ) not in requested_support_sides:
                continue
            domain_ids = tuple(evidence.get("effective_domain_ids", ()))
            expected_count = 2 if evidence["classification"] == "SA" else 1
            if len(domain_ids) != expected_count:
                raise ValueError(
                    "physical EPR support has inconsistent adjacent domains"
                )
            domains = [self.regions_by_id.get(domain_id) for domain_id in domain_ids]
            if any(domain is None for domain in domains):
                raise ValueError(
                    "physical EPR support domain is absent from prepared_stack"
                )
            domains = cast(list[dict[str, Any]], domains)
            if evidence["classification"] == "SA":
                field_domains = [
                    domain
                    for domain in domains
                    if self.materials.get(domain["material_id"], {}).get("kind")
                    == "vacuum"
                ]
                if len(field_domains) != 1:
                    raise ValueError(
                        "physical SA support needs one vacuum field domain"
                    )
                field_domain = field_domains[0]
            else:
                field_domain = domains[0]
            self.support_bindings.append(
                {
                    **surface,
                    "binding_id": (
                        f"{evidence['contribution_id']}__"
                        f"{canonical_sha256({'surface_id': surface['surface_id'], 'geometry_ref': surface['geometry_ref']})[:16]}"
                    ),
                    "effective_material_id": field_domain["material_id"],
                }
            )

    def _resolve_contributions(self) -> None:
        self.resolved_surfaces: list[dict[str, Any]] = []
        for requested in self.contribution_tuple:
            matches = [
                item
                for item in self.surfaces
                if item["contribution"].get("contribution_id")
                == requested.contribution_id
            ]
            if not matches:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} is not established by "
                    "the shared source topology"
                )
            # The catalogue union includes every group member and SA fallback surface ID.
            source_polygon_ids = self.catalog_by_id.get(
                requested.contribution_id, {}
            ).get("source_polygon_ids", ())
            if requested.source_polygon_id not in source_polygon_ids:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} does not retain "
                    f"source polygon {requested.source_polygon_id!r}"
                )
            for match in matches:
                self._resolve_surface(requested, match)

    def _assemble_geometry(self) -> PreparedPlanarGeometry:
        self.resolved_surfaces = bind_physical_support_masks(
            self.resolved_surfaces, self.support_bindings, source_dbu_um=self.dbu_um
        )

        digest_input = {
            "modeling": self.modeling,
            "source": self.source,
            "junctions": [item.to_payload() for item in self.junction_tuple],
            **(
                {"lumped_rlcs": [item.to_payload() for item in self.rlc_tuple]}
                if self.rlc_tuple
                else {}
            ),
            "contribution_catalog": self.contribution_catalog,
            "contributions": [item.to_payload() for item in self.contribution_tuple],
            "surface_bindings": self.resolved_surfaces,
        }
        return PreparedPlanarGeometry(
            modeling=self.modeling,
            source=self.source,
            junctions=self.junction_tuple,
            lumped_rlcs=self.rlc_tuple,
            contribution_catalog=self.contribution_catalog,
            contributions=self.contribution_tuple,
            surface_bindings=tuple(self.resolved_surfaces),
            model_sha256=canonical_sha256(
                {
                    "modeling": self.modeling,
                    "source": self.source,
                    "junctions": [item.to_payload() for item in self.junction_tuple],
                    **(
                        {"lumped_rlcs": [item.to_payload() for item in self.rlc_tuple]}
                        if self.rlc_tuple
                        else {}
                    ),
                }
            ),
            source_sha256=canonical_sha256(digest_input),
        )

    def _resolve_surface(
        self, requested: SurfaceEprSpec, match: dict[str, Any]
    ) -> None:
        evidence = match["contribution"]
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
        effective_domains = tuple(evidence.get("effective_domain_ids", ()))
        expected_domain_count = 2 if requested.interface_kind == "SA" else 1
        if len(effective_domains) != expected_domain_count:
            raise ValueError(
                f"contribution {requested.contribution_id!r} does not have "
                f"{expected_domain_count} physical adjacent domain(s)"
            )
        adjacent_domains = [self.regions_by_id.get(item) for item in effective_domains]
        if any(item is None for item in adjacent_domains):
            raise ValueError(
                f"contribution {requested.contribution_id!r} adjacent domain "
                "is absent from prepared_stack"
            )
        adjacent_domains = cast(list[dict[str, Any]], adjacent_domains)
        if requested.interface_kind == "SA":
            vacuum_domains = [
                item
                for item in adjacent_domains
                if self.materials.get(item["material_id"], {}).get("kind") == "vacuum"
            ]
            dielectric_domains = [
                item
                for item in adjacent_domains
                if self.materials.get(item["material_id"], {}).get("kind")
                == "dielectric"
            ]
            if len(vacuum_domains) != 1 or len(dielectric_domains) != 1:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} must bind one "
                    "vacuum and one dielectric domain"
                )
            domain = vacuum_domains[0]
            substrate_domain = dielectric_domains[0]
        else:
            domain = adjacent_domains[0]
            substrate_domain = domain if requested.interface_kind == "MS" else None
        if domain is None:
            raise ValueError(
                f"contribution {requested.contribution_id!r} field-side domain "
                "is absent from prepared_stack"
            )
        material = self.materials.get(domain["material_id"])
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
            self.materials.get(substrate_domain["material_id"])
            if substrate_domain is not None
            else None
        )
        if requested.interface_kind in {"MS", "SA"}:
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
        self.resolved_surfaces.append(
            {
                **match,
                "binding_id": (
                    f"{requested.contribution_id}__"
                    f"{canonical_sha256({'surface_id': match['surface_id'], 'geometry_ref': match['geometry_ref']})[:16]}"
                ),
                "effective_domain_id": domain["semantic_id"],
                "substrate_domain_id": (
                    substrate_domain["semantic_id"]
                    if substrate_domain is not None
                    else None
                ),
                "effective_material_id": domain["material_id"],
                "effective_material": plain(material),
                "substrate_material": (
                    plain(substrate_material)
                    if substrate_material is not None
                    else None
                ),
                "margins_um": list(requested.margins_um),
                "film_thickness_m": requested.film_thickness_m,
                "film_relative_permittivity": requested.film_relative_permittivity,
            }
        )


def validate_geometry_workers(geometry_workers: int | None) -> None:
    """Require a positive integer worker count or omitted override."""
    if geometry_workers is not None and (
        isinstance(geometry_workers, bool)
        or not isinstance(geometry_workers, int)
        or geometry_workers <= 0
    ):
        raise ValueError("geometry_workers must be a positive integer or None")
