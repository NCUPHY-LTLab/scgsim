"""Check the Q2D evidence in a completed receipt against the spec.

Q2D is the ANSYS 2D extractor for per-unit-length RLGC (resistance, inductance,
conductance, capacitance) matrices of a cross-section. Owns: conductor and
rectangle model evidence, setup, convergence and the RLGC matrix readback. It does
not own the output manifest or file hashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt.results.convergence.q2d import read_q2d_convergence
from scgsim.aedt.results.matrices import read_q2d_rlgc_matrix
from scgsim.aedt.results.readback_common import (
    contained_in_handoff_root,
    positive_unique_ids,
)
from scgsim.aedt.specs.q2d import Q2dSpec


def validate_q2d_readback(root: Path, receipt: dict[str, Any], spec: Q2dSpec) -> None:
    """Model, setup, convergence, then the solved RLGC matrix, in that order."""
    validate_q2d_model_readback(receipt, spec)
    validate_q2d_setup(receipt, spec)
    if receipt.get("convergence") != read_q2d_convergence(root, spec):
        raise RuntimeError("Q2D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    rows, summary = read_q2d_rlgc_matrix(
        contained_in_handoff_root(root, "results/q2d/rlgc_matrix.csv"), spec
    )
    expected_matrices = {
        "path": "results/q2d/rlgc_matrix.csv",
        "frequency_ghz": spec.run_control.frequency_ghz,
        "length_setting": "Distributed",
        "length": "1meter",
        "matrix_type": "Maxwell, Spice, Couple",
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q2D matrix readback is invalid")


def validate_q2d_model_readback(
    receipt: dict[str, Any], spec: Q2dSpec, *, prepared: bool = False
) -> None:
    """Bind section and assignments separately from setup and solved matrices.

    ``prepared`` additionally checks the model-stage evidence (rectangles, saved
    boundaries and Region) that only a prepared model records.
    """
    if receipt.get("ports") != [] or receipt.get("nets") != []:
        raise RuntimeError("Q2D receipt must not contain HFSS ports or Q3D nets")
    conductors = receipt.get("conductors")
    if not isinstance(conductors, list) or len(conductors) != len(spec.conductors):
        raise RuntimeError("Q2D receipt has invalid conductor evidence")
    for record, expected in zip(conductors, spec.conductors, strict=True):
        _check_conductor_record(record, expected)
    if not prepared:
        return  # Historical solve records retain their original evidence scope.
    by_name = _check_prepared_rectangles(receipt, spec)
    _check_prepared_saved_conductors(conductors, spec, by_name)
    _check_prepared_region(receipt, spec)


def _check_conductor_record(record: Any, expected: Any) -> None:
    if (
        not isinstance(record, dict)
        or record.get("name") != expected.name
        or record.get("conductor_type") != expected.conductor_type
        or record.get("object_names") != list(expected.object_names)
        or record.get("thickness_um") != expected.thickness_um
        or record.get("solve_option") != "SolveOnBoundary"
        or not isinstance(record.get("native_object_ids"), list)
        or len(record["native_object_ids"]) != len(expected.object_names)
        or not all(isinstance(value, int) for value in record["native_object_ids"])
    ):
        raise RuntimeError("Q2D native conductor evidence does not match the spec")


def _check_prepared_rectangles(
    receipt: dict[str, Any], spec: Q2dSpec
) -> dict[str, int]:
    """Check every native rectangle sheet; return native object ID by rectangle name."""
    materials = receipt.get("materials")
    if not isinstance(materials, list) or len(materials) != len(spec.rectangles):
        raise RuntimeError("Q2D prepared model lacks its native rectangle inventory")
    by_name = {}
    for record, rectangle in zip(materials, spec.rectangles, strict=True):
        material = spec.materials[rectangle.material_id]
        if (
            not isinstance(record, dict)
            or record.get("object_name") != rectangle.name
            or record.get("origin_um") != list(rectangle.origin_um)
            or record.get("size_um") != list(rectangle.size_um)
            or record.get("material_id") != material.material_id
            or record.get("kind") != material.kind
            or record.get("is_superconducting") != material.is_superconducting
            or record.get("requested_library_name") != material.library_name
            # A superconductor is solved as PEC (perfect electric conductor).
            or str(
                record.get("observed", {}).get("native_material_name", "")
            ).casefold()
            != (
                "pec" if material.is_superconducting else material.library_name
            ).casefold()
            or not positive_unique_ids([record.get("native_object_id")], 1)
            or record.get("native_object_type") != "Sheet"
            or not isinstance(record.get("native_bounding_box_um"), list)
            or len(record["native_bounding_box_um"]) != 6
        ):
            raise RuntimeError("Q2D prepared rectangle source/native evidence differs")
        by_name[rectangle.name] = record["native_object_id"]
    if len(set(by_name.values())) != len(by_name):
        raise RuntimeError("Q2D prepared rectangle native IDs are not unique")
    return by_name


def _check_prepared_saved_conductors(
    conductors: list[Any], spec: Q2dSpec, by_name: dict[str, int]
) -> None:
    """Each conductor's native IDs and saved boundary must follow its rectangles."""
    fields = ("BoundType", "Objects", "SolveOption", "Thickness")
    for record, conductor in zip(conductors, spec.conductors, strict=True):
        saved = record.get("saved_boundary")
        properties = record.get("saved_properties")
        if (
            record.get("native_conductor_type") != conductor.conductor_type
            or record["native_object_ids"]
            != [by_name[name] for name in conductor.object_names]
            or not isinstance(saved, dict)
            or properties
            != {
                "values": {name: saved[name] for name in fields if name in saved},
                "unavailable": [name for name in fields if name not in saved],
            }
        ):
            raise RuntimeError("Q2D prepared saved conductor evidence differs")


def _check_prepared_region(receipt: dict[str, Any], spec: Q2dSpec) -> None:
    region = receipt.get("region")
    vacuum = spec.materials[spec.vacuum_material_id]
    if (
        not isinstance(region, dict)
        or region.get("material_id") != vacuum.material_id
        or region.get("requested_library_name") != vacuum.library_name
        or str(region.get("observed_material_name", "")).casefold() != "vacuum"
        or region.get("padding_um") != list(spec.region_padding_um)
        or not isinstance(region.get("native_bounding_box_um"), list)
        or len(region["native_bounding_box_um"]) != 6
    ):
        raise RuntimeError("Q2D prepared Region evidence differs")


def validate_q2d_setup(receipt: dict[str, Any], spec: Q2dSpec) -> None:
    """The setup AEDT read back must equal the spec's run control for both solves.

    CG is the capacitance/conductance solve and RL the resistance/inductance solve;
    both are given the same pass and convergence limits.
    """
    if spec.run_control is None:
        raise ValueError("Q2D setup evidence requires complete run controls")
    expected_setup = {
        "name": spec.run_control.setup_name,
        "native": {
            "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
            "cg_maximum_passes": spec.run_control.maximum_passes,
            "cg_minimum_passes": spec.run_control.minimum_passes,
            "cg_minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "cg_convergence_percent": spec.run_control.convergence_percent,
            "cg_percent_refinement": spec.run_control.percent_refinement,
            "rl_maximum_passes": spec.run_control.maximum_passes,
            "rl_minimum_passes": spec.run_control.minimum_passes,
            "rl_minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "rl_convergence_percent": spec.run_control.convergence_percent,
            "rl_percent_refinement": spec.run_control.percent_refinement,
        },
    }
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q2D native setup evidence is invalid")
