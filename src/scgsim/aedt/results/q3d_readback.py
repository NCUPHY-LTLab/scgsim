"""Check the Q3D evidence in a completed receipt against the spec.

Q3D is the ANSYS 3D quasi-static extractor for capacitance (C) and, optionally,
AC resistance/inductance (RL) matrices of named nets. Owns: net and body bindings,
setup, convergence and the matrix readback. It does not own the Region and its
grounded enclosure (``q3d_region``) or the output manifest.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, NamedTuple

from scgsim.aedt._terminal_chain import q3d_terminal_segments
from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.results.q3d_region import validate_q3d_region_ground
from scgsim.aedt.results.readback_common import (
    contained_in_handoff_root,
    positive_unique_ids,
)
from scgsim.aedt.specs.q3d import Q3dSpec

# Metres per native length unit, replaying the Length conversion of pinned PyAEDT 1.3
# without starting AEDT. Used to read a ThinConductor thickness such as ``1um``.
_NATIVE_LENGTH_METERS = {
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "dm": 1e-1,
    "meter": 1.0,
    "km": 1e3,
    "uin": 0.0254 * 1e-6,
    "mil": 0.0254 * 1e-3,
    "in": 0.0254,
    "ft": 0.0254 * 12,
    "yd": 0.0254 * 36,
    "mile": 0.0254 * 63360,
}
_NATIVE_LENGTH_PATTERN = re.compile(
    r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)([A-Za-z]+)"
)

_CAPACITANCE_AND_CONDUCTANCE = {"Capacitance Matrix": "C", "Conductance Matrix": "G"}


# ---------------------------------------------------------------------------
# Whole-run readback
# ---------------------------------------------------------------------------


def validate_q3d_readback(root: Path, receipt: dict[str, Any], spec: Q3dSpec) -> None:
    """Model, setup, convergence, then the solved matrices, in that order."""
    validate_q3d_model_readback(root, receipt, spec)
    validate_q3d_setup(receipt, spec)
    if receipt.get("convergence") != read_q3d_convergence(root, spec):
        raise RuntimeError("Q3D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    if spec.solve_ac_rl:
        _check_ac_rl_matrices(root, spec, matrices)
        return
    rows, summary = read_q3d_original_cg(root, spec)
    expected_matrices = {
        "path": "results/q3d/c_matrix.csv",
        "frequency_ghz": spec.run_control.frequency_ghz,
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q3D capacitance-only matrix readback is invalid")


def _check_ac_rl_matrices(root: Path, spec: Q3dSpec, matrices: Any) -> None:
    if (
        not isinstance(matrices, dict)
        or matrices.get("frequency_ghz") != spec.run_control.frequency_ghz
        or set(matrices.get("native", {})) != {"c", "ac_rl"}
        or not isinstance(matrices.get("normalized_rows"), int)
        or matrices["normalized_rows"] <= 0
    ):
        raise RuntimeError("Q3D matrix readback is invalid")
    _validate_normalized_matrices(
        contained_in_handoff_root(root, "results/q3d/matrices.csv"),
        matrices["normalized_rows"],
        {"C", "AC RL"},
    )
    read_q3d_original_cg(root, spec)


def read_q3d_original_cg(
    root: Path, spec: Q3dSpec
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse the solver's own capacitance and conductance export for the signal nets."""
    options = {
        "expected_label_set": {
            net.name for net in spec.nets if net.net_type == "Signal"
        }
    }
    return parse_matrix_export(
        contained_in_handoff_root(root, "results/q3d/c_matrix.csv"),
        "Q3D",
        "C",
        spec.run_control.frequency_ghz,
        _CAPACITANCE_AND_CONDUCTANCE,
        **options,
    )


def _validate_normalized_matrices(
    path: Path, expected_rows: int, problem_types: set[str]
) -> None:
    """The combined long-format matrix file must hold all four quantities."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or not rows:
        raise RuntimeError("normalized matrix row count is invalid")
    expected_quantities = {"C", "G", "L", "R"}
    if {row.get("quantity") for row in rows} != expected_quantities:
        raise RuntimeError("normalized matrix quantities are incomplete")
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("normalized matrix value is invalid") from exc
        if (
            not math.isfinite(value)
            or row.get("problem_type") not in problem_types
            or not row.get("row")
            or not row.get("column")
            or not row.get("unit")
        ):
            raise RuntimeError("normalized matrix row is invalid")


def validate_q3d_setup(receipt: dict[str, Any], spec: Q3dSpec) -> None:
    """The setup AEDT read back must equal the spec's run control."""
    expected_setup = {
        "name": spec.run_control.setup_name,
        "native": {
            "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
            "capacitance_maximum_passes": spec.run_control.maximum_passes,
            "capacitance_minimum_passes": spec.run_control.minimum_passes,
            "capacitance_minimum_converged_passes": (
                spec.run_control.minimum_converged_passes
            ),
            "capacitance_convergence_percent": spec.run_control.convergence_percent,
            "capacitance_percent_refinement": spec.run_control.percent_refinement,
            "dc_enabled": False,
        },
    }
    if spec.solve_ac_rl:
        expected_setup["native"].update(
            {
                "ac_rl_maximum_passes": spec.run_control.maximum_passes,
                "ac_rl_minimum_passes": spec.run_control.minimum_passes,
                "ac_rl_minimum_converged_passes": (
                    spec.run_control.minimum_converged_passes
                ),
                "ac_rl_convergence_percent": spec.run_control.convergence_percent,
                "ac_rl_percent_refinement": spec.run_control.percent_refinement,
            }
        )
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q3D native setup evidence is invalid")


# ---------------------------------------------------------------------------
# Model: nets, bodies, Region
# ---------------------------------------------------------------------------


def validate_q3d_model_readback(
    root: Path, receipt: dict[str, Any], spec: Q3dSpec, *, prepared: bool = False
) -> None:
    """Bind model assignments separately from setup and matrix evidence."""
    if ("ports" in receipt or not prepared) and receipt.get("ports") != []:
        raise RuntimeError("Q3D receipt must not contain HFSS ports")
    nets = receipt.get("nets")
    if not isinstance(nets, list) or len(nets) != len(spec.nets):
        raise RuntimeError("Q3D receipt has invalid net evidence")
    for record, expected in zip(nets, spec.nets, strict=True):
        _check_net_record(record, expected, spec)
    net_object_ids = [
        object_id for record in nets for object_id in record["native_object_ids"]
    ]
    if len(set(net_object_ids)) != len(net_object_ids):
        raise RuntimeError("Q3D native object IDs must be unique across nets")
    _validate_body_bindings(receipt, spec, nets, root)
    validate_q3d_region_ground(receipt, spec, nets)


def _check_net_record(record: Any, expected: Any, spec: Q3dSpec) -> None:
    if (
        not isinstance(record, dict)
        or record.get("name") != expected.name
        or record.get("net_type") != expected.net_type
        or record.get("object_names") != list(expected.object_names)
        or not isinstance(record.get("native_object_ids"), list)
        or not positive_unique_ids(
            record["native_object_ids"], len(expected.object_names)
        )
    ):
        raise RuntimeError("Q3D native net evidence does not match the spec")
    if expected.net_type == "Signal" and expected.source_object is not None:
        for end in (
            _TerminalEnd("source", expected.source_object, expected.source_side),
            _TerminalEnd("sink", expected.sink_object, expected.sink_side),
        ):
            _check_net_terminal(record, expected, spec, end)
    elif "source" in record or "sink" in record:
        raise RuntimeError("Q3D net receipt contains undeclared terminals")


class _TerminalEnd(NamedTuple):
    """One end of a signal net: ``source`` or ``sink``, its body and its side."""

    kind: str
    object_name: str
    side: str


def _check_net_terminal(
    record: dict[str, Any], expected: Any, spec: Q3dSpec, end: _TerminalEnd
) -> None:
    """A signal net's source or sink terminal sits on a Sheet's edges or a Solid's face."""
    kind, object_name, side = end
    terminal = record.get(kind)
    if (
        not isinstance(terminal, dict)
        or terminal.get("name") != f"{expected.name}{kind.title()}"
        or terminal.get("object_name") != object_name
        or terminal.get("side") != side
    ):
        raise RuntimeError("Q3D native terminal evidence is invalid")
    effective = {body["body_id"]: body for body in spec.effective_bodies}
    if effective and effective[object_name]["representation"] == "sheet":
        body = next(body for body in spec.bodies if body.body_id == object_name)
        indices = q3d_terminal_segments(body, side)
        edges = terminal.get("native_edge_ids")
        if (
            not isinstance(edges, list)
            or not positive_unique_ids(edges, len(indices))
            or terminal.get("source_edges")
            != [
                {"source_segment_index": index, "native_edge_id": edge}
                for index, edge in zip(indices, edges, strict=True)
            ]
            or terminal.get("terminal_type") != "ConstantVoltage"
            or "native_face_ids" in terminal
        ):
            raise RuntimeError(
                "Q3D Sheet terminal evidence differs from source exterior"
            )
    else:
        faces = terminal.get("native_face_ids")
        if not isinstance(faces, list) or not positive_unique_ids(faces, 1):
            raise RuntimeError("Q3D Solid terminal face evidence is invalid")


def _validate_body_bindings(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[dict[str, Any]], root: Path
) -> None:
    """Bind every created body to its declared geometry and final native identity.

    Native incidence may be unavailable; source declarations never substitute
    for native topology. The receipt must state which readback actually exists.
    """
    materials = receipt.get("materials")
    if not isinstance(materials, list) or len(materials) != len(spec.bodies):
        raise RuntimeError("Q3D body material readback is invalid")
    net_object_ids = {
        name: native_id
        for net_record, net_spec in zip(nets, spec.nets, strict=True)
        for name, native_id in zip(
            net_spec.object_names, net_record["native_object_ids"], strict=True
        )
    }
    records = {
        record.get("object_name"): record
        for record in materials
        if isinstance(record, dict)
    }
    if len(records) != len(materials) or set(records) != {
        body.body_id for body in spec.bodies
    }:
        raise RuntimeError("Q3D native body inventory differs from the spec")
    effective_bodies = {body["body_id"]: body for body in spec.effective_bodies}
    seen_native_ids: set[int] = set()
    for body in spec.bodies:
        record = records[body.body_id]
        effective = effective_bodies[body.body_id]
        observed_name = _check_body_material(record, body, spec, effective)
        evidence = record.get("body_binding")
        if not isinstance(evidence, dict):
            raise RuntimeError("Q3D native body binding is absent")
        native_id = evidence.get("native_object_id")
        faces = evidence.get("native_face_ids")
        _check_body_identity(evidence, body, effective, seen_native_ids, net_object_ids)
        if evidence.get("effective_geometry") != effective:
            raise RuntimeError("Q3D effective geometry binding differs from the spec")
        _check_thin_conductor(
            evidence.get("thin_conductor"), body, effective, native_id, observed_name
        )
        seen_native_ids.add(native_id)
        _check_incidence(evidence.get("incidence"), faces)


def _check_body_material(
    record: dict[str, Any], body: Any, spec: Q3dSpec, effective: dict[str, Any]
) -> str:
    """Check the body's material evidence; return the observed native material name."""
    material = spec.materials[body.material_id]
    observed = record.get("observed")
    # A superconductor is solved as PEC (perfect electric conductor).
    expected_native_material = (
        "pec" if material.is_superconducting else material.library_name
    )
    sheet = effective["representation"] == "sheet"
    # A Sheet has no volume material; AEDT reports only its boundary material.
    observed_key = "native_boundary_material_name" if sheet else "native_material_name"
    if (
        record.get("material_id") != body.material_id
        or record.get("role") != body.physical_role
        or record.get("kind") != material.kind
        or record.get("is_superconducting") is not material.is_superconducting
        or record.get("requested_library_name") != material.library_name
        or not isinstance(observed, dict)
        or not isinstance(observed.get(observed_key), str)
        or observed[observed_key].casefold() != expected_native_material.casefold()
    ):
        raise RuntimeError("Q3D body material binding differs from the spec")
    if sheet and (
        observed.get("native_material_property") != "not_applicable_sheet"
        or "native_material_name" in observed
    ):
        raise RuntimeError("Q3D Sheet evidence must identify boundary-only material")
    return observed[observed_key]


def _check_body_identity(
    evidence: dict[str, Any],
    body: Any,
    effective: dict[str, Any],
    seen_native_ids: set[int],
    net_object_ids: dict[str, int],
) -> None:
    native_id = evidence.get("native_object_id")
    faces = evidence.get("native_face_ids")
    if (
        evidence.get("body") != body.to_payload()
        or evidence.get("object_name") != body.body_id
        or evidence.get("native_object_type")
        != ("Sheet" if effective["representation"] == "sheet" else "Solid")
        or not positive_unique_ids([native_id], 1)
        or native_id in seen_native_ids
        or not isinstance(faces, list)
        or not faces
        or not positive_unique_ids(faces, len(faces))
        or (body.net_id is not None and net_object_ids.get(body.body_id) != native_id)
    ):
        raise RuntimeError(
            "Q3D body binding differs from the native identity/source declaration"
        )


def _check_thin_conductor(
    thin: Any,
    body: Any,
    effective: dict[str, Any],
    native_id: int,
    observed_name: str,
) -> None:
    """A Sheet body carries a ThinConductor boundary; a Solid body must not."""
    if effective["representation"] != "sheet":
        if thin is not None:
            raise RuntimeError("Q3D Solid body contains Sheet boundary evidence")
        return
    if not isinstance(thin, dict):
        raise RuntimeError("Q3D ThinConductor evidence is not recorded")
    raw_material = thin.get("material_raw")
    if (
        isinstance(raw_material, list)
        and len(raw_material) == 2
        and raw_material[0] == "Material:="
    ):
        raw_material = raw_material[1]
    if not isinstance(raw_material, str):
        raise RuntimeError("Q3D ThinConductor Material has unknown native shape")
    material_name = raw_material.strip('"')
    raw_thickness = thin.get("thickness")
    thickness_um = _native_length_um(raw_thickness)
    raw_ids = thin.get("native_assignment_ids_raw")
    if not isinstance(raw_ids, list) or any(
        not (
            type(value) is int
            or (isinstance(value, str) and re.fullmatch(r"[0-9]+", value))
        )
        for value in raw_ids
    ):
        raise RuntimeError("Q3D ThinConductor assignment readback is malformed")
    assigned_ids = [int(value) for value in raw_ids]
    expected_thin = {
        "name": f"{body.body_id}ThinConductor",
        "bound_type": "ThinConductor",
        "material": material_name,
        "material_raw": thin.get("material_raw"),
        "thickness": raw_thickness,
        "thickness_um": thickness_um,
        "native_assignment_ids_raw": raw_ids,
        "native_object_ids": assigned_ids,
    }
    if (
        thin != expected_thin
        or material_name.casefold() != "pec"
        or material_name != observed_name
        or thickness_um != effective["physical_thickness_um"]
        or assigned_ids != [native_id]
    ):
        raise RuntimeError("Q3D ThinConductor binding differs from physical source")


def _native_length_um(raw_thickness: Any) -> float:
    """Parse a native length string such as ``1um`` into micrometres."""
    length = _NATIVE_LENGTH_PATTERN.fullmatch(
        raw_thickness if isinstance(raw_thickness, str) else ""
    )
    if length is None or length[2].lower() not in _NATIVE_LENGTH_METERS:
        raise RuntimeError("Q3D ThinConductor Thickness is not a native length")
    return float(length[1]) * _NATIVE_LENGTH_METERS[length[2].lower()] / 1e-6


def _check_incidence(incidence: Any, faces: list[int]) -> None:
    """Face/edge/vertex incidence is either observed, or unavailable with a reason."""
    if not isinstance(incidence, dict):
        raise RuntimeError("Q3D body incidence availability is not recorded")
    if incidence.get("status") == "unavailable":
        if (
            not isinstance(incidence.get("unavailable_reason"), str)
            or not incidence["unavailable_reason"]
        ):
            raise RuntimeError("Q3D unavailable incidence lacks a reason")
    elif incidence.get("status") == "observed":
        if (
            incidence.get("face_ids") != faces
            or not isinstance(incidence.get("face_edge_ids"), dict)
            or not isinstance(incidence.get("edge_vertex_ids"), dict)
        ):
            raise RuntimeError("Q3D observed native incidence is malformed")
    else:
        raise RuntimeError("Q3D native incidence status is invalid")
