"""Read live junction geometry and electrical properties before native save."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from typing import Any

from scgsim.aedt.preparation.planar_source import native_entity_name, native_name
from scgsim.aedt.runtime.native.common import (
    native_boundary_type,
    native_object_boolean_property,
    native_object_property,
    resolve_native_assignment,
)
from scgsim.aedt.runtime.native.junction_extrusion import native_extrusion_sides
from scgsim.aedt.runtime.native.junction_topology import (
    NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
    native_boundary_contact,
    native_closed_contact,
    native_coordinates_close,
    native_cross_sign,
    native_edge_covered,
    native_edge_section,
    native_junction_corners,
    native_planar_loops,
    native_polygon_correspondence,
    native_strict_inside,
    native_z_contact,
    native_z_may_contact,
)
from scgsim.aedt.runtime.native.planar_bodies import native_conductor_geometry
from scgsim.aedt.specs.epr import (
    PlanarJunction,
)


def native_junction_live_readback(
    app: Any, source: Mapping[str, Any], junction: PlanarJunction
) -> dict[str, Any]:
    """Verify live CAD/electrical state before authoring; line awaits native save.

    Face edges supply complete planar footprints, rather than bounding-box
    contact guesses. Actual returned geometry is preserved. Coordinate identity,
    straightness, Z and contact comparisons use absolute 1e-5 um, rel_tol=0;
    correspondence never substitutes source vertices into contact geometry.
    Below-resolution gaps/curvature remain numerically unresolved. Missing live
    RLC properties or ambiguous topology are explicit readback failures. The
    transaction-local result is not a completed partition_readback; the owned
    runtime must verify the saved assigned line before publishing that record.
    """
    return JunctionReadback(app, source, junction).run()


class JunctionReadback:
    """Keep one junction readback state across ordered verification stages."""

    AEDT_UNITS: Any
    BoundaryObject: Any
    app: Any
    boundary_name: Any
    central_matches: Any
    central_name: Any
    central_z: Any
    decompose_variable_value: Any
    deviations: Any
    edge_evidence: Any
    entity_nets: Any
    expected: Any
    junction: Any
    native_z_by_polygon: Any
    observations: Any
    points: Any
    polygons: Any
    record: Any
    rlc_readback: Any
    source: Any

    def __init__(self, app: Any, source: Any, junction: Any) -> None:
        self.app = app
        self.source = source
        self.junction = junction

    def run(self) -> dict[str, Any]:
        """Read native state and verify each dependency in its original order."""
        self._read_central_sheet()
        self._read_conductors()
        self._verify_central_interior()
        self._verify_end_contacts()
        self._verify_edge_contacts()
        self._read_electrical()
        return self._finish()

    def _read_central_sheet(self) -> None:
        from ansys.aedt.core.generic.constants import AEDT_UNITS  # noqa: PLC0415 -- optional PyAEDT is needed only for live junction readback; offline preparation does not require the SDK.

        self.AEDT_UNITS = AEDT_UNITS
        from ansys.aedt.core.generic.numbers_utils import decompose_variable_value  # noqa: PLC0415 -- optional PyAEDT is needed only for live junction readback; offline preparation does not require the SDK.

        self.decompose_variable_value = decompose_variable_value
        from ansys.aedt.core.modules.boundary.common import BoundaryObject  # noqa: PLC0415 -- optional PyAEDT is needed only for live junction readback; offline preparation does not require the SDK.

        self.BoundaryObject = BoundaryObject
        self.record = self.source["junction_partitions"][self.junction.junction_id]
        if self.app.modeler.model_units != "um":
            raise RuntimeError(
                "junction native readback requires source micrometre model units"
            )
        self.polygons = {p["polygon_id"]: p for p in self.source["polygons"]}
        z_um = float(self.record["z_um"])
        self.central_name = native_name("junction", self.junction.junction_id)
        central = self.app.modeler.get_object_from_name(self.central_name)
        if (
            central is None
            or len(central.faces) != 1
            or len(central.faces[0].edges) != 4
        ):
            raise RuntimeError("native central junction is not one four-edge face")
        self.observations, self.deviations = [], []
        central_loops = native_planar_loops(central, z_um, self.observations)
        self.expected = self.polygons[self.record["central_polygon_id"]]
        self.central_matches = native_polygon_correspondence(
            central_loops,
            self.expected,
            self.central_name,
            self.deviations,
            subdivisions=False,
        )
        self.central_z = tuple(
            operation(p[2] for p in self.observations[0]["face_vertices_um"])
            for operation in (min, max)
        )

    def _read_conductors(self) -> None:
        self.native_by_net: dict[str, list] = {}
        self.potential_by_net: dict[str, list] = {}
        self.native_by_polygon: dict[tuple[str, str], list] = {}
        self.native_z_by_polygon = {}
        self.entity_nets = {
            entity["semantic_id"]: entity["net_id"]
            for entity in self.source["conductors"]
        }
        authored_ids = {
            partition["authored_polygon_id"]
            for partition in self.source["junction_partitions"].values()
        }
        for entity in self.source["conductors"]:
            representation, effective_z_min_um, effective_z_max_um = (
                native_conductor_geometry(self.source, entity)
            )
            plane = effective_z_min_um
            for pid in entity["polygon_ids"]:
                if pid in authored_ids:
                    continue
                name = native_entity_name(
                    self.source, "conductor", entity["semantic_id"], pid
                )
                obj = self.app.modeler.get_object_from_name(name)
                if obj is None:
                    raise RuntimeError(
                        f"native junction source conductor unavailable: {name!r}"
                    )
                loops = native_planar_loops(obj, plane, self.observations)
                wanted = self.polygons[pid]
                bottom_matches = native_polygon_correspondence(
                    loops, wanted, name, self.deviations
                )
                bottom_observation = self.observations[-1]
                bottom_xyz = {
                    tuple(Fraction(str(v)) for v in p[:2]): p
                    for p in bottom_observation["face_vertices_um"]
                }
                if representation == "sheet":
                    native_z = tuple(
                        operation(p[2] for p in bottom_observation["face_vertices_um"])
                        for operation in (min, max)
                    )
                    self._verify_assignment(
                        native_name("pec_boundary", entity["semantic_id"], pid),
                        name,
                        "Perfect E",
                    )
                else:
                    if (
                        native_object_property(obj, "Material").strip('"').casefold()
                        != "pec"
                        or native_object_boolean_property(obj, "Solve Inside")
                        is not False
                    ):
                        raise RuntimeError(
                            f"native junction conductor lacks finite PEC lowering: {name!r}"
                        )
                    zs = [float(vertex.position[2]) for vertex in obj.vertices]
                    if not zs or not native_coordinates_close(
                        (min(zs), max(zs)),
                        (effective_z_min_um, effective_z_max_um),
                    ):
                        raise RuntimeError(
                            f"native junction source material/Z lineage differs: {name!r}"
                        )
                    top = native_planar_loops(obj, max(zs), self.observations)
                    top_matches = native_polygon_correspondence(
                        top, wanted, name, self.deviations
                    )
                    try:
                        native_polygon_correspondence(
                            top,
                            {"exterior": loops[0][0], "holes": loops[0][1:]},
                            name,
                            [],
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "native junction actual top/bottom extrusion differs"
                        ) from exc
                    top_xyz = {
                        tuple(Fraction(str(v)) for v in p[:2]): p
                        for p in self.observations[-1]["face_vertices_um"]
                    }
                    thickness = effective_z_max_um - effective_z_min_um
                    if any(
                        not native_coordinates_close(
                            bottom_matches[point], top_matches[point]
                        )
                        or not native_coordinates_close(
                            (
                                top_xyz[top_matches[point]][2]
                                - bottom_xyz[bottom_matches[point]][2],
                            ),
                            (thickness,),
                        )
                        for point in bottom_matches
                    ):
                        raise RuntimeError(
                            "native junction actual top/bottom extrusion differs"
                        )
                    native_extrusion_sides(
                        obj,
                        (effective_z_min_um, effective_z_max_um),
                        thickness,
                        self.observations,
                    )
                    native_z = (min(zs), max(zs))
                solid = representation != "sheet"
                key = entity["semantic_id"], pid
                self.native_by_polygon[key] = loops
                self.native_z_by_polygon[key] = native_z, solid
                if native_z_contact(self.central_z, False, native_z, solid):
                    self.native_by_net.setdefault(entity["net_id"], []).extend(loops)
                if native_z_may_contact(self.central_z, native_z):
                    self.potential_by_net.setdefault(entity["net_id"], []).extend(loops)

    def _verify_central_interior(self) -> None:
        self.points = native_junction_corners(
            self.expected, self.junction.direction_xy, self.central_matches
        )
        central_midpoint = tuple(sum(p[i] for p in self.points) / 4 for i in (0, 1))
        for bodies in self.potential_by_net.values():
            for rings in bodies:
                if any(
                    native_strict_inside(p, (self.points,)) for p in rings[0]
                ) or any(
                    native_strict_inside(p, rings)
                    for p in (*self.points, central_midpoint)
                ):
                    raise RuntimeError("native PEC occupies central RLC interior")
                for ring in rings:
                    for a, b in zip(ring, (*ring[1:], ring[0]), strict=False):
                        for c, d in zip(
                            self.points,
                            (*self.points[1:], self.points[0]),
                            strict=False,
                        ):
                            if (
                                native_cross_sign(a, b, c) * native_cross_sign(a, b, d)
                                < 0
                                and native_cross_sign(c, d, a)
                                * native_cross_sign(c, d, b)
                                < 0
                            ):
                                raise RuntimeError(
                                    "native PEC crosses central RLC interior"
                                )

    def _verify_end_contacts(self) -> None:
        derived_ids = {
            end["polygon_id"]
            for partition in self.source["junction_partitions"].values()
            for end in partition["ends"]
            if end["polygon_id"] is not None
        }

        for end in self.record["ends"]:
            if end["polygon_id"] is None:
                continue
            end_loops = self.native_by_polygon[
                end["source_entity_id"], end["polygon_id"]
            ]
            end_z, end_solid = self.native_z_by_polygon[
                end["source_entity_id"], end["polygon_id"]
            ]
            arms = [
                rings
                for (eid, pid), loops in self.native_by_polygon.items()
                if eid == end["source_entity_id"]
                and pid not in derived_ids
                and native_z_contact(
                    end_z, end_solid, *self.native_z_by_polygon[eid, pid]
                )
                for rings in loops
            ]
            if not native_boundary_contact(end_loops, arms):
                raise RuntimeError(
                    "native derived end lacks positive-length source-arm contact"
                )
            foreign = [
                rings
                for (eid, pid), loops in self.native_by_polygon.items()
                if self.entity_nets[eid] != end["net_id"]
                and native_z_may_contact(end_z, self.native_z_by_polygon[eid, pid][0])
                for rings in loops
            ]
            if native_closed_contact(end_loops, foreign):
                raise RuntimeError("native derived end contacts another net")

    def _verify_edge_contacts(self) -> None:
        edge_specs = (
            (self.points[3], self.points[0], self.junction.terminal_a_net),
            (self.points[1], self.points[2], self.junction.terminal_b_net),
            (self.points[0], self.points[1], None),
            (self.points[2], self.points[3], None),
        )
        self.edge_evidence = []
        for start, end, correct_net in edge_specs:
            if correct_net is not None and not native_edge_covered(
                self.native_by_net.get(correct_net, []), start, end
            ):
                raise RuntimeError(
                    "native designated junction end edge is not wholly covered"
                )
            for net, bodies in self.potential_by_net.items():
                for rings in bodies:
                    intervals, contacts = native_edge_section([rings], start, end)
                    if correct_net is not None:
                        forbidden = net != correct_net and bool(intervals or contacts)
                    else:
                        forbidden = bool(intervals) or any(0 < v < 1 for v in contacts)
                        # Open side corners belong to their respective end edge.
                        for v in contacts:
                            point = start if v == 0 else end
                            own_corner = (
                                self.junction.terminal_a_net
                                if point in (self.points[0], self.points[3])
                                else self.junction.terminal_b_net
                            )
                            forbidden |= net != own_corner
                    if forbidden:
                        raise RuntimeError(
                            "native junction has forbidden side/wrong-net corner contact"
                        )
            self.edge_evidence.append(
                {
                    "start_um": list(map(float, start)),
                    "end_um": list(map(float, end)),
                    "terminal_net": correct_net,
                }
            )

    def _read_electrical(self) -> None:
        self.boundary_name = native_name("junction_rlc", self.junction.junction_id)
        self._verify_assignment(self.boundary_name, self.central_name, "Lumped RLC")
        required = (
            "RLC Type",
            "Use Induct",
            "Inductance",
            "Use Cap",
            "Use Resist",
        )
        # Record only reads and comparisons reached by the existing strict checks.
        # Diagnostic collection never performs an additional native query.
        self.rlc_readback = {
            "requested": {
                "RLC Type": "Parallel",
                "Use Induct": True,
                "Inductance": {"value": self.junction.inductance_h, "unit": "H"},
                "Use Resist": False,
                "Use Cap": bool(self.junction.capacitance_f),
                "Capacitance": {"value": self.junction.capacitance_f, "unit": "F"},
            },
            "observed": {
                name: {"status": "not_read"} for name in (*required, "Capacitance")
            },
            "available_property_names": None,
            "normalized": {},
            "scalar_comparisons": {},
            "checks": {
                name: {"status": "not_evaluated"}
                for name in (
                    "parallel_type",
                    "inductance_enabled",
                    "resistance_disabled",
                    "inductance_matches_si",
                    "capacitance_enable_equal",
                    "capacitance_available",
                    "capacitance_matches_si",
                )
            },
            "failed_checks": [],
            "stage": "live_property_object",
        }

        try:
            self._verify_electrical_values(required)
        except Exception as exc:
            try:
                exc.add_note(
                    "junction_rlc_readback="
                    + json.dumps(self.rlc_readback, sort_keys=True)
                )
            except Exception:  # noqa: BLE001 -- diagnostic attachment must preserve any primary native readback failure.
                try:
                    exc.add_note(
                        "junction_rlc_readback=unavailable: diagnostic serialization failed"
                    )
                except Exception:  # noqa: BLE001 -- diagnostic attachment must preserve any primary native readback failure.
                    pass
            raise

    def _finish(self) -> dict[str, Any]:
        self.deviations.extend(
            observation["maximum_plane_coordinate_deviation_um"]
            for observation in self.observations
            if "maximum_plane_coordinate_deviation_um" in observation
        )
        return {
            "coordinate_comparison_policy": {
                "absolute_tolerance_um": NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
                "relative_tolerance": 0.0,
                "scope": "SCGSim native junction readback; not an AEDT precision guarantee",
                "geometry_moved": False,
            },
            "maximum_source_coordinate_deviation_um": max(self.deviations, default=0.0),
            "native_geometry_observations": self.observations,
            "edge_contacts": self.edge_evidence,
            "boundary_name": self.boundary_name,
            "electrical_readback": self.rlc_readback,
        }

    def _failed(self, name, condition):
        self.rlc_readback["checks"][name] = {
            "status": "evaluated",
            "passed": not condition,
        }
        if condition:
            self.rlc_readback["failed_checks"].append(name)
        return condition

    def _enabled(self, value):
        if value in (True, "true", "True", 1):
            return True
        if value in (False, "false", "False", 0):
            return False
        raise RuntimeError("native RLC enabled property is invalid")

    def _scalar(self, value, unit, property_name):
        # Preserve the raw normalized SI value; representation equivalence
        # is applied only to comparison operands below, never authoring.
        self.rlc_readback["stage"] = f"parse_scalar:{property_name}"
        number, observed_unit = self.decompose_variable_value(value)
        parsed = {
            "number": number,
            "number_type": type(number).__name__,
            "observed_unit": observed_unit,
            "target_unit": unit,
        }
        self.rlc_readback["normalized"][property_name] = parsed
        system = {"H": "Inductance", "F": "Capacitance"}[unit]
        if (
            observed_unit not in self.AEDT_UNITS[system]
            or not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
        ):
            raise RuntimeError("native RLC scalar unit/value unavailable")
        result = Decimal(str(number)) * Decimal(
            str(self.AEDT_UNITS[system][observed_unit])
        )
        parsed["unit_scale"] = str(self.AEDT_UNITS[system][observed_unit])
        parsed["value_si_decimal"] = str(result)
        self.rlc_readback["stage"] = f"compare_scalar:{property_name}"
        return result

    def _matches_si(self, property_name, observed, requested):
        """Apply SCG's comparison resolution, not a native precision claim."""
        comparison = {
            "observed_si_decimal": str(observed),
            "requested_si_decimal": str(requested),
            "exact_equal_si": observed == requested,
            "policy": "binary64_decimal_roundtrip",
            "significant_digits": sys.float_info.dig,
            "rounding": "ROUND_HALF_EVEN",
            "canonical_observed_si": None,
            "canonical_requested_si": None,
            "representation_equal_si": None,
            "matches_si": False,
            "method": "exact_si",
        }
        self.rlc_readback["scalar_comparisons"][property_name] = comparison
        if comparison["exact_equal_si"]:
            comparison["matches_si"] = True
            return True
        comparison["method"] = "comparison_operand_canonicalization"
        if not observed.is_finite() or not requested.is_finite():
            comparison["method"] = "nonfinite_operands"
            return False
        # Only the comparison operands are rounded into explicit decimal
        # bins. Agreement is neither source bit identity nor AEDT precision.
        with localcontext() as context:
            context.prec = sys.float_info.dig
            context.rounding = ROUND_HALF_EVEN
            canonical_observed = +observed
            canonical_requested = +requested
        comparison["canonical_observed_si"] = str(canonical_observed)
        comparison["canonical_requested_si"] = str(canonical_requested)
        comparison["representation_equal_si"] = (
            canonical_observed == canonical_requested
        )
        comparison["matches_si"] = comparison["representation_equal_si"]
        return comparison["matches_si"]

    def _verify_electrical_values(self, required: tuple[str, ...]) -> None:
        live = self.BoundaryObject(  # noqa: SLF001 -- pinned boundary property object supplies uncached native electrical readback.
            self.app, self.boundary_name, auto_update=False
        )._child_object
        if live is None:
            raise RuntimeError("native live junction RLC property object unavailable")
        self.rlc_readback["stage"] = "property_names"
        available = [str(v) for v in live.GetPropNames()]
        self.rlc_readback["available_property_names"] = available
        for name in self.rlc_readback["observed"]:
            self.rlc_readback["observed"][name]["available"] = name in available
        if any(name not in available for name in required):
            raise RuntimeError(
                f"native live RLC readback lacks required properties; available={available!r}"
            )
        values = {}
        for name in required:
            self.rlc_readback["stage"] = f"read_property:{name}"
            value = live.GetPropValue(name)
            values[name] = value
            self.rlc_readback["observed"][name] = {
                "status": "read",
                "available": True,
                "value": value,
                "type": type(value).__name__,
            }

        self.rlc_readback["stage"] = "compare_parallel_L_R"
        if (
            self._failed("parallel_type", values["RLC Type"] != "Parallel")
            or self._failed(
                "inductance_enabled", not self._enabled(values["Use Induct"])
            )
            or self._failed("resistance_disabled", self._enabled(values["Use Resist"]))
            or self._failed(
                "inductance_matches_si",
                not self._matches_si(
                    "Inductance",
                    self._scalar(values["Inductance"], "H", "Inductance"),
                    Decimal(str(self.junction.inductance_h)),
                ),
            )
        ):
            raise RuntimeError("native junction parallel L/R parameters differ")
        self.rlc_readback["stage"] = "compare_capacitance_enable"
        if self._failed(
            "capacitance_enable_equal",
            self._enabled(values["Use Cap"]) != bool(self.junction.capacitance_f),
        ):
            raise RuntimeError("native junction capacitance enable state differs")
        if self.junction.capacitance_f:
            self.rlc_readback["stage"] = "compare_capacitance"
            if self._failed("capacitance_available", "Capacitance" not in available):
                raise RuntimeError("native junction capacitance differs")
            self.rlc_readback["stage"] = "read_property:Capacitance"
            capacitance = live.GetPropValue("Capacitance")
            self.rlc_readback["observed"]["Capacitance"] = {
                "status": "read",
                "available": True,
                "value": capacitance,
                "type": type(capacitance).__name__,
            }
            if self._failed(
                "capacitance_matches_si",
                not self._matches_si(
                    "Capacitance",
                    self._scalar(capacitance, "F", "Capacitance"),
                    Decimal(str(self.junction.capacitance_f)),
                ),
            ):
                raise RuntimeError("native junction capacitance differs")

    def _verify_assignment(self, boundary_name, target, kind):
        if native_boundary_type(self.app, boundary_name) != kind:
            raise RuntimeError(
                f"native junction boundary type differs: {boundary_name!r}"
            )
        raw = self.app.oboundary.GetBoundaryAssignment(boundary_name)
        if raw is None:
            raise RuntimeError(
                f"native junction boundary target unavailable: {boundary_name!r}"
            )
        ids = [int(v) for v in raw]
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError("native junction boundary assignment IDs are invalid")
        _, _, names = resolve_native_assignment(self.app, ids)
        if names != {target}:
            raise RuntimeError(
                f"native junction boundary has wrong targets: {boundary_name!r}"
            )
