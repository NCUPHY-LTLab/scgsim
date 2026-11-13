"""Bind EPR analysis sheets and their native field-side selections.

Common model construction and saved geometry readback belong to runtime.native.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Literal

from scgsim.aedt.data import canonical_sha256
from scgsim.aedt.epr.geometry import (
    base_sheet_plan,
)
from scgsim.aedt.preparation.planar_masks import INSET_CIRCLE_POINTS, INSET_METHOD
from scgsim.aedt.preparation.planar_source import (
    native_name,
)
from scgsim.aedt.runtime.native.common import (
    native_object_boolean_property,
    native_object_evidence,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.geometry.grid.spatial import geometry_ref_surface_z_um


def _analysis_surface_sheet(
    app: Any, geometry_ref: Mapping[str, Any], *, name: str
) -> tuple[Any, list[list[float]]]:
    local_region = geometry_ref.get("local_region")
    plane_basis = geometry_ref.get("plane_basis")
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(local_region, Mapping) and isinstance(plane_basis, Mapping):
        points = _lift_inset_ring(plane_basis, local_region["exterior"])
        holes = [
            _lift_inset_ring(plane_basis, ring)
            for ring in local_region.get("holes", ())
        ]
    elif isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if (
            not isinstance(plane, Mapping) or plane.get("axis") != "z"
        ) and geometry_ref.get("shell_part") not in {"top", "bottom"}:
            raise ValueError("analysis surface loop requires an explicit Z plane")
        z_um = geometry_ref_surface_z_um(geometry_ref)
        points = [[float(x), float(y), z_um] for x, y in outer]
        holes = [
            [[float(x), float(y), z_um] for x, y in ring]
            for ring in geometry_ref.get("hole_loops", ())
        ]
    else:
        raise ValueError("horizontal analysis surface requires a planar loop")
    if len(points) < 3 or any(
        len(point) != 3 or not all(math.isfinite(value) for value in point)
        for point in points
    ):
        raise ValueError("analysis surface points are invalid")
    sheet = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=True,
    )
    if sheet is False or sheet is None:
        raise RuntimeError(f"failed to create analysis surface {name!r}")
    for index, ring in enumerate(holes):
        hole_name = native_name("hole", name, index)
        hole = app.modeler.create_polyline(
            ring,
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=True,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create analysis hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract analysis hole {hole_name!r}")
    return app.modeler[name], points


def _native_sheet_side_binding(
    native_normal: Sequence[float],
    field_side: str,
    modeling: str | None,
    *,
    unknown_side_error: str,
    ambiguous_side_error: str,
) -> tuple[tuple[float, float, float], bool]:
    """Map a source side while limiting the empirical rule to ThinFilm."""

    if field_side == "top":
        desired = (0.0, 0.0, 1.0)
    elif field_side == "bottom":
        desired = (0.0, 0.0, -1.0)
    else:
        raise RuntimeError(unknown_side_error)
    orientation = sum(
        float(normal) * target
        for normal, target in zip(native_normal, desired, strict=False)
    )
    if abs(orientation) <= 1e-12:
        raise RuntimeError(ambiguous_side_error)
    if modeling == "thin_film":
        # Human-selected for pinned PyAEDT 1.3.0 / AEDT 2024.2: FacePrimitive.normal
        # points toward Adjacent. This is not a direct native arrow-vector accessor.
        adjacent_side = orientation > 0.0
    else:
        # Preserve the pre-existing Solid/historical convention; its physical
        # face-side correspondence has not been established by this change.
        adjacent_side = orientation < 0.0
    return desired, adjacent_side


def _resolve_surface_field_selector(
    default_adjacent_side: bool,
    requested_selector: Literal["surface", "adjacent"] | None,
    *,
    source_field_side: str,
) -> dict[str, Any]:
    """Resolve a native field selector without reinterpreting the physical side."""
    default_selector = "adjacent" if default_adjacent_side else "surface"
    applied = default_selector if requested_selector is None else requested_selector
    return {
        "source_field_side": source_field_side,
        "default_native_selector": default_selector,
        "requested_native_selector": requested_selector,
        "applied_native_selector": applied,
        "applied_native_token": {
            "surface": "EnterSurface",
            "adjacent": "EnterAdjacentSurface",
        }[applied],
        "physical_material_side": "unverified",
        "scope": "surface_field_quantities_only; masked area is side-independent",
    }


def _lift_inset_ring(
    plane: Mapping[str, Any], ring: Sequence[Sequence[float]]
) -> list[list[float]]:
    origin = tuple(float(value) for value in plane["origin_um"])
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    return [
        [
            origin[axis] + float(point[0]) * u[axis] + float(point[1]) * v[axis]
            for axis in range(3)
        ]
        for point in ring
    ]


@dataclass
class InsetSurfaceBinding:
    """Carry one source contribution and the session's shared inset-sheet facts."""

    app: Any
    binding: Mapping[str, Any]
    base_selection: Mapping[str, Any]
    margin_um: float
    inset_plan: Mapping[tuple[str, float], Mapping[str, Any]]
    sheet_facts: dict[str, dict[str, Any]]
    sheet_names: Mapping[str, Mapping[str, Any]]
    modeling: str | None
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None = None

    def bind(self) -> list[dict[str, Any]]:
        """Bind each inset component and its requested native field selector."""
        self.plane = self.binding["mask_plane"]
        # Field direction belongs to the source contribution independently of
        # the base sheet's orientation.
        self.field_side = str(self.binding["contribution"]["side"])
        self.planned = self.inset_plan[
            (str(self.binding["binding_id"]), self.margin_um)
        ]
        regions = self.planned["regions"]
        self.dbu_um = float(self.binding["mask_support"]["source_dbu_um"])
        self.result: list[dict[str, Any]] = []
        for index, region in enumerate(regions):
            self.index = index
            self.region = region
            self._bind_region()
        return self.result

    def _bind_region(self) -> None:
        self.contour_sha256 = canonical_sha256(
            {
                "plane": self.plane,
                "region": self.region,
                "method": INSET_METHOD,
                "source_dbu_um": self.dbu_um,
                "klayout_version": version("klayout"),
            }
        )
        self.planned_name = self.sheet_names[self.contour_sha256]
        self.name = str(self.planned_name["name"])
        facts = self.sheet_facts.get(self.name)
        if facts is None:
            facts = self._read_sheet_facts()
        if facts["contour_sha256"] != self.contour_sha256:
            raise RuntimeError(f"inset EPR selection name collision: {self.name!r}")
        native = facts["native"]
        native_normal = facts["native_normal"]
        desired, adjacent_side = _native_sheet_side_binding(
            native_normal,
            self.field_side,
            self.modeling,
            unknown_side_error=f"inset EPR selection {self.name!r} has unknown field side",
            ambiguous_side_error=(
                f"inset EPR selection {self.name!r} has ambiguous field side"
            ),
        )
        selector_evidence = {}
        if self.surface_field_selectors:
            selector_evidence = _resolve_surface_field_selector(
                adjacent_side,
                self.surface_field_selectors.get(
                    str(self.binding["contribution"]["contribution_id"])
                ),
                source_field_side=self.field_side,
            )
            adjacent_side = selector_evidence["applied_native_selector"] == "adjacent"
        self.result.append(
            {
                "selection_name": self.name,
                "component_index": self.index,
                "shared_sheet_members": self.planned_name["members"],
                "naming_identity_sha256": self.planned_name["naming_identity_sha256"],
                "aggregated_parts": self.planned_name["aggregated_parts"],
                "contour_sha256": self.contour_sha256,
                "contour": self.region,
                "adjacent_side": adjacent_side,
                **(
                    {"field_selector_binding": selector_evidence}
                    if selector_evidence
                    else {}
                ),
                "native_normal": native_normal,
                "polygon_approximation": {
                    "method": INSET_METHOD,
                    "source_dbu_um": self.dbu_um,
                    "points_per_circle": INSET_CIRCLE_POINTS,
                    "requested_margin_um": self.margin_um,
                    "integer_margin_dbu": self.planned["integer_margin_dbu"],
                    "effective_margin_um": self.planned["effective_margin_um"],
                    "klayout_version": version("klayout"),
                },
                "reused_source_sheet": False,
                **native,
            }
        )

    def _read_sheet_facts(self) -> dict[str, Any]:
        sheet = self.app.modeler.get_object_from_name(self.name)
        if sheet is None:
            sheet, _ = _analysis_surface_sheet(
                self.app,
                {"plane_basis": self.plane, "local_region": self.region},
                name=self.name,
            )
        native = native_object_evidence(self.app, self.name)
        faces = sheet.faces
        if native["native_object_type"] != "Sheet" or len(faces) != 1:
            raise RuntimeError(f"inset EPR selection {self.name!r} is not one sheet")
        if native_object_boolean_property(sheet, "Model"):
            raise RuntimeError(
                f"inset EPR selection {self.name!r} became model geometry"
            )
        normal = faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(
                f"inset EPR selection {self.name!r} has no native normal"
            )
        facts = {
            "contour_sha256": self.contour_sha256,
            "shared_sheet_members": self.planned_name["members"],
            "naming_identity_sha256": self.planned_name["naming_identity_sha256"],
            "aggregated_parts": self.planned_name["aggregated_parts"],
            "native": native,
            "native_normal": [float(value) for value in normal],
        }
        self.sheet_facts[self.name] = facts
        return facts


def add_analysis_surfaces(
    app: Any, prepared: PreparedPlanarGeometry, geometry: dict[str, Any]
) -> None:
    """Add EPR selection sheets to the already constructed common native model."""
    phase_seconds: dict[str, float] = {}
    surface_selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    base_names = base_sheet_plan(prepared.surface_bindings)
    base_points: dict[str, list[list[float]]] = {}
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        if name not in base_points:
            sheet, points = _analysis_surface_sheet(
                app, binding["geometry_ref"], name=name
            )
            base_points[name] = points
        else:
            sheet = app.modeler.get_object_from_name(name)
            if sheet is None:
                raise RuntimeError(f"shared analysis surface {name!r} vanished")
            points = base_points[name]
        native = native_object_evidence(app, sheet.name)
        if native["native_object_type"] != "Sheet" or len(sheet.faces) != 1:
            raise RuntimeError(f"analysis surface {name!r} is not one native sheet")
        normal = sheet.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"analysis surface {name!r} normal is unavailable")
        unit_normal = tuple(float(value) for value in normal)
        centroid = tuple(
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )
        field_side = str(binding["contribution"]["side"])
        desired_normal, adjacent_side = _native_sheet_side_binding(
            unit_normal,
            field_side,
            prepared.modeling,
            unknown_side_error=f"analysis surface {name!r} has unknown field side",
            ambiguous_side_error=(
                f"analysis surface {name!r} has ambiguous effective-domain side"
            ),
        )
        surface_selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": sheet.name,
                "shared_sheet_members": planned_name["members"],
                "naming_identity_sha256": planned_name["naming_identity_sha256"],
                "aggregated_parts": planned_name["aggregated_parts"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": adjacent_side,
                "field_side": field_side,
                "native_normal": list(unit_normal),
                "selection_centroid_um": list(centroid),
                "desired_field_side_normal": list(desired_normal),
                **native,
            }
        )
    phase_seconds["base_analysis_sheets_seconds"] = time.perf_counter() - started

    geometry["surface_selections"] = surface_selections
    geometry["geometry_phases"].update(
        {key: round(value, 6) for key, value in phase_seconds.items()}
    )


def bind_saved_analysis_surfaces(
    app: Any,
    prepared: PreparedPlanarGeometry,
    geometry: dict[str, Any],
    *,
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None = None,
) -> None:
    """Read EPR selection sheets from the saved common native model."""
    selections: list[dict[str, Any]] = []
    base_names = base_sheet_plan(prepared.surface_bindings)
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        obj = app.modeler.get_object_from_name(name)
        if obj is None or len(obj.faces) != 1:
            raise RuntimeError(f"saved analysis surface {name!r} is unavailable")
        evidence = native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved analysis surface {name!r} is not a sheet")
        normal = obj.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"saved analysis surface {name!r} normal is unavailable")
        native_normal = tuple(float(value) for value in normal)
        side = str(binding["contribution"]["side"])
        desired, adjacent_side = _native_sheet_side_binding(
            native_normal,
            side,
            prepared.modeling,
            unknown_side_error=f"saved analysis surface {name!r} has unknown side",
            ambiguous_side_error=(f"saved analysis surface {name!r} side is ambiguous"),
        )
        selector_evidence = {}
        if surface_field_selectors:
            selector_evidence = _resolve_surface_field_selector(
                adjacent_side,
                surface_field_selectors.get(
                    str(binding["contribution"]["contribution_id"])
                ),
                source_field_side=side,
            )
            adjacent_side = selector_evidence["applied_native_selector"] == "adjacent"
        selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": name,
                "shared_sheet_members": planned_name["members"],
                "naming_identity_sha256": planned_name["naming_identity_sha256"],
                "aggregated_parts": planned_name["aggregated_parts"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": adjacent_side,
                **(
                    {"field_selector_binding": selector_evidence}
                    if selector_evidence
                    else {}
                ),
                "field_side": side,
                "native_normal": list(native_normal),
                "desired_field_side_normal": list(desired),
                **evidence,
            }
        )
    geometry["surface_selections"] = selections
