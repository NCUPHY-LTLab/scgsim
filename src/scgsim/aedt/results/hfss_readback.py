"""Check the HFSS evidence in a completed receipt against the spec.

Owns: setup and convergence evidence, Eigenmode results, and the driven
(Modal and plain Terminal) frequency, Touchstone and port evidence. It does not
own body-first Terminal ports (``body_first_readback``), PEC and material
evidence (``hfss_pec``), or the file hashes in the output manifest.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scgsim.aedt.results.body_first_readback import validate_body_terminal_readback
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.results.readback_common import uses_current_terminal_rules
from scgsim.aedt.specs.common import ModalPort
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.hfss_prepared import HfssDrivenGeometrySpec, HfssEprSpec


def validate_hfss_setup_and_convergence(
    root: Path, receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    """Check the native setup, then re-read the convergence files and compare."""
    validate_hfss_setup(receipt, spec)
    if isinstance(spec, HfssEigenmodeSpec):
        require_export_convergence_source(receipt)
    if receipt.get("convergence") != read_hfss_convergence(root, spec):
        raise RuntimeError("HFSS native convergence evidence is invalid")


def validate_hfss_setup(receipt: dict[str, Any], spec: Any) -> None:
    """The setup the receipt read back from AEDT must equal the spec's run control."""
    if isinstance(spec, (HfssEigenmodeSpec, HfssEprSpec)):
        native = {
            "minimum_frequency": f"{spec.run_control.minimum_frequency_ghz:g}GHz",
            "num_modes": spec.run_control.num_modes,
            "maximum_delta_frequency_percent": (
                spec.run_control.maximum_delta_frequency_percent
            ),
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    else:
        native = {
            "solve_type": "Broadband",
            "low_frequency": f"{spec.run_control.sweep.start_ghz:g}GHz",
            "high_frequency": f"{spec.run_control.sweep.stop_ghz:g}GHz",
            "maximum_delta_s": spec.run_control.maximum_delta_s,
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    setup = receipt.get("setup")
    expected = {"name": spec.run_control.setup_name, "native": native}
    # An EPR receipt may carry extra setup keys; only the expected ones are compared.
    if (
        not isinstance(setup, dict)
        or (
            {key: setup.get(key) for key in expected}
            if isinstance(spec, HfssEprSpec)
            else setup
        )
        != expected
    ):
        raise RuntimeError("HFSS native setup evidence is invalid")


def require_export_convergence_source(receipt: dict[str, Any]) -> None:
    """Eigenmode convergence must come from the exported convergence file only."""
    convergence = receipt.get("convergence")
    sources = convergence.get("sources") if isinstance(convergence, dict) else None
    if not isinstance(sources, dict):
        raise RuntimeError("Eigenmode convergence sources are invalid")
    if set(sources) != {"export_convergence"}:
        raise RuntimeError(
            f"Eigenmode convergence sources {sorted(sources)!r} are unsupported; "
            "only ['export_convergence'] is supported"
        )


def validate_eigenmode_readback(
    receipt: dict[str, Any], spec: HfssEigenmodeSpec | HfssEprSpec
) -> None:
    """Check the final mode frequencies and Q factors the receipt read back."""
    readback = receipt.get("result_readback")
    if isinstance(spec, HfssEprSpec):
        # Body-first loads belong to geometry; this producer has no ports field.
        if "ports" in receipt:
            raise RuntimeError("body-first Eigenmode receipt must not contain ports")
        readback = readback.get("final_modes") if isinstance(readback, dict) else None
    elif receipt.get("ports") != []:
        raise RuntimeError("HFSS Eigenmode receipt must not contain ports")
    values = readback.get("eigenmodes") if isinstance(readback, dict) else None
    if not isinstance(values, dict):
        raise TypeError("completed receipt has no Eigenmode result readback")
    modes = spec.run_control.num_modes
    frequencies = values.get("frequencies_ghz")
    q_factors = values.get("q_factors")
    if (
        values.get("modes") != modes
        or values.get("frequency_unit") != "GHz"
        or values.get("mode_indices") != list(range(1, modes + 1))
        or not isinstance(frequencies, list)
        or len(frequencies) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in frequencies
        )
        or not isinstance(q_factors, list)
        or len(q_factors) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in q_factors
        )
        or values.get("native_export") != "eigenmodes.eig"
        or not isinstance(values.get("native_export_bytes"), int)
        or values["native_export_bytes"] <= 0
    ):
        raise RuntimeError("completed receipt Eigenmode result readback is invalid")


def validate_driven_readback(
    root: Path, receipt: dict[str, Any], spec: HfssDrivenSpec | HfssDrivenGeometrySpec
) -> None:
    """Check a two-port Terminal or Modal run: sweep, Touchstone, ports, setup."""
    terminal = spec.mode == "terminal"
    readback = receipt.get("result_readback")
    _check_sweep_readback(readback, spec)
    ports = receipt.get("ports")
    _check_touchstone_readback(readback.get("touchstone"), ports, spec)
    if terminal:
        if isinstance(spec, HfssDrivenGeometrySpec):
            validate_body_terminal_readback(receipt, spec)
        else:
            validate_terminal_native_evidence(
                ports, spec, current=uses_current_terminal_rules(receipt)
            )
    else:
        validate_modal_native_evidence(ports, spec)
    validate_hfss_setup_and_convergence(root, receipt, spec)


def _check_sweep_readback(readback: Any, spec: Any) -> None:
    key = "terminal_st" if spec.mode == "terminal" else "modal_s"
    if not isinstance(readback, dict) or not isinstance(readback.get(key), dict):
        raise TypeError("completed receipt has no canonical result readback")
    if readback[key] != {
        "records": int(spec.run_control.sweep.points),
        "frequency_unit": "GHz",
        "first_frequency_ghz": spec.run_control.sweep.start_ghz,
        "last_frequency_ghz": spec.run_control.sweep.stop_ghz,
        "strictly_increasing": True,
    }:
        raise RuntimeError("completed receipt result frequency readback is invalid")


def _check_touchstone_readback(touchstone: Any, ports: Any, spec: Any) -> None:
    """The Touchstone file must be a two-port sweep with named, distinct ports."""
    port_key = "terminal_excitation" if spec.mode == "terminal" else "modal_excitation"
    port_order = (
        [record.get(port_key) for record in ports]
        if isinstance(ports, list)
        and len(ports) == 2
        and all(isinstance(record, dict) for record in ports)
        and [record.get("index") for record in ports] == [1, 2]
        else None
    )
    if (
        not isinstance(touchstone, dict)
        or touchstone.get("path") != f"{spec.mode}.s2p"
        or touchstone.get("ports") != 2
        or touchstone.get("records") != int(spec.run_control.sweep.points)
        or touchstone.get("frequency_unit") != "GHz"
        or touchstone.get("first_frequency_ghz") != spec.run_control.sweep.start_ghz
        or touchstone.get("last_frequency_ghz") != spec.run_control.sweep.stop_ghz
        or touchstone.get("strictly_increasing") is not True
        or touchstone.get("port_order") != port_order
        or not all(isinstance(name, str) and name for name in port_order or [])
        or len(set(port_order or [])) != 2
        or not isinstance(touchstone.get("bytes"), int)
        or touchstone["bytes"] <= 0
    ):
        raise RuntimeError(
            "completed receipt is missing a two-port Touchstone readback"
        )


def validate_modal_native_evidence(ports: Any, spec: HfssDrivenSpec) -> None:
    """Check each Modal Wave Port: request, effective endpoints and saved boundary."""
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid modal ports")
    native_ports = spec.effective_ports
    for record, port, native_port in zip(ports, spec.ports, native_ports, strict=True):
        if not isinstance(port, ModalPort) or not isinstance(record, dict):
            raise TypeError("completed receipt has invalid modal port identity")
        _check_modal_request(record, port, native_port)
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native modal evidence")
        _check_modal_native_boundary(record, native, port, native_port, spec)


def _check_modal_request(
    record: dict[str, Any], port: ModalPort, native_port: Any
) -> None:
    if (
        record.get("index") != port.index
        or record.get("boundary") != port.name
        or record.get("modal_excitation") != port.name
        or not isinstance(record.get("face_id"), int)
        or not isinstance(record.get("face_center_um"), list)
        or len(record["face_center_um"]) != 3
        or record.get("requested")
        != {
            "integration_line_um": [list(point) for point in port.integration_line_um],
            "modes": 1,
            "renormalize": False,
            "deembed_um": 0.0,
            "characteristic_impedance": "Zpi",
        }
    ):
        raise RuntimeError("completed receipt modal request does not match spec")
    if record.get("effective") != {
        "integration_line_um": [
            list(point) for point in native_port.integration_line_um
        ]
    }:
        raise RuntimeError(
            "completed receipt modal effective endpoints differ from source map"
        )


def _check_modal_native_boundary(
    record: dict[str, Any],
    native: dict[str, Any],
    port: ModalPort,
    native_port: Any,
    spec: HfssDrivenSpec,
) -> None:
    """The saved Wave Port boundary must match, and its integration line to 1e-9 um."""
    expected_names = [item.name for item in spec.ports[: port.index]]
    expected_boundary = {
        "bound_type": "Wave Port",
        "wave_port_type": "Modal",
        "faces": [record["face_id"]],
        "num_modes": 1,
        "deembed": False,
        "mode_number": 1,
        "use_integration_line": True,
        "characteristic_impedance": "Zpi",
    }
    saved_boundary = native.get("saved_boundary")
    integration_line = (
        saved_boundary.get("integration_line_um")
        if isinstance(saved_boundary, dict)
        else None
    )
    if (
        native.get("excitation_names") != expected_names
        or native.get("boundary_properties")
        != {
            "Deembed": False,
            "Name": port.name,
            "Num Modes": "1",
            "Renorm All Modes": False,
            "Type": "Wave Port",
        }
        or not isinstance(saved_boundary, dict)
        or {
            key: value
            for key, value in saved_boundary.items()
            if key != "integration_line_um"
        }
        != expected_boundary
        or not isinstance(integration_line, list)
        or len(integration_line) != 2
        or any(
            not math.isclose(float(actual), wanted, abs_tol=1e-9)
            for actual_point, expected_point in zip(
                integration_line, native_port.integration_line_um, strict=True
            )
            for actual, wanted in zip(actual_point, expected_point, strict=True)
        )
    ):
        raise RuntimeError("completed receipt modal native evidence mismatch")


def validate_terminal_native_evidence(
    ports: Any, spec: HfssDrivenSpec, *, current: bool = False
) -> None:
    """Check each plain Terminal port; ``current`` selects the V19 evidence rules."""
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid terminal ports")
    for record, port in zip(ports, spec.ports, strict=True):
        if not isinstance(record, dict) or "unresolved_native" in record:
            raise RuntimeError(
                "completed receipt has unresolved terminal native evidence"
            )
        terminal_name = record.get("terminal_excitation")
        _check_terminal_identity(record, port, terminal_name)
        requested = record.get("requested")
        if requested != {
            "reference_objects": list(port.reference_objects),
            "renormalize": False,
            "deembed_um": port.deembed_um,
        }:
            raise RuntimeError("completed receipt terminal request does not match spec")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native terminal evidence")
        _check_terminal_native_identity(native, port, terminal_name, spec)
        _check_terminal_native_properties(native, port, terminal_name, current)


def _check_terminal_identity(
    record: dict[str, Any], port: Any, terminal_name: Any
) -> None:
    if (
        record.get("index") != port.index
        or record.get("boundary") != port.name
        or not isinstance(terminal_name, str)
        or not terminal_name
        or not isinstance(record.get("face_id"), int)
        or not isinstance(record.get("face_center_um"), list)
        or len(record["face_center_um"]) != 3
    ):
        raise RuntimeError("completed receipt terminal identity is invalid")


def _check_terminal_native_identity(
    native: dict[str, Any], port: Any, terminal_name: str, spec: HfssDrivenSpec
) -> None:
    if (
        native.get("excitation_names")
        != [item.name for item in spec.ports[: port.index]]
        or native.get("terminal_names") != [terminal_name]
        or native.get("reference_conductors") != list(port.reference_objects)
        or not isinstance(native.get("reference_conductor_ids"), list)
        or not native["reference_conductor_ids"]
        or not all(
            isinstance(value, int) for value in native["reference_conductor_ids"]
        )
    ):
        raise RuntimeError("completed receipt terminal native identity mismatch")


def _check_terminal_native_properties(
    native: dict[str, Any], port: Any, terminal_name: str, current: bool
) -> None:
    """Under V19 rules the de-embed flag and impedance are observed, not fixed."""
    boundary = native.get("boundary_properties")
    terminal = native.get("terminal_properties")
    expected_boundary = {
        "Deembed": boundary.get("Deembed") if current else True,
        "Deembed Dist": f"{port.deembed_um:g}um",
        "Name": port.name,
        "Num Terminals": "1",
        "Renorm All Terminals": False,
        "Type": "Wave Port",
        "Wave Port Type": "Terminal",
    }
    expected_terminal = {
        "Name": terminal_name,
        "Port Name": port.name,
        "Terminal Renormalizing Impedance": (
            terminal.get("Terminal Renormalizing Impedance") if current else "50ohm"
        ),
        "Type": "Terminal",
    }
    if (
        boundary != expected_boundary
        or terminal != expected_terminal
        or (
            current
            and (
                boundary.get("Deembed") not in (True, False, "true", "false")
                or native.get("inactive_renormalizing_impedance")
                != terminal.get("Terminal Renormalizing Impedance")
            )
        )
    ):
        raise RuntimeError("completed receipt terminal native property mismatch")
