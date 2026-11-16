"""Own the Q3D capacitance/conductance matrix export and its readback summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q3d import Q3dSpec


def export_q3d_matrices(
    app: Any, run_dir: Path, spec: Q3dSpec
) -> tuple[dict[str, str], dict[str, Any]]:
    """Export the capacitance/conductance matrix and retain its independent native readback summary."""
    control = _matrix_control(spec)
    output_dir = run_dir / "results" / "q3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "c_matrix.csv"
    export_matrix_file(app, path, spec, "C")
    rows, summary = parse_matrix_export(
        path,
        "Q3D",
        "C",
        control.frequency_ghz,
        {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
        **matrix_parse_options(spec, "C"),
    )
    relative = path.relative_to(run_dir).as_posix()
    return {relative: file_sha256(path)}, {
        "matrices": {
            "path": relative,
            "frequency_ghz": control.frequency_ghz,
            "native": summary,
            "primary_rows": len(rows),
        }
    }


def export_matrix_file(app: Any, path: Path, spec: Q3dSpec, problem: str) -> None:
    """One native LastAdaptive/Original export authority for full and stopped runs."""
    control = _matrix_control(spec)
    frequency_hz = control.frequency_ghz * 1e9
    app.odesign.ExportMatrixData(
        str(path),
        problem,
        "",
        f"{control.setup_name} : LastAdaptive",
        "Original",
        "ohm",
        "nH",
        "pF",
        "mho",
        frequency_hz,
        "Maxwell",
        0,
        False,
        15,
        20,
        1,
    )


class MatrixParseOptions(TypedDict, total=False):
    """Only capacitance parsing checks the declared signal-net label set."""

    expected_label_set: set[str]


def matrix_parse_options(spec: Q3dSpec, problem: str) -> MatrixParseOptions:
    """Select the label check for a matrix: a capacitance matrix must name exactly the Signal nets."""
    if problem == "C":
        return {
            "expected_label_set": {
                net.name for net in spec.nets if net.net_type == "Signal"
            }
        }
    return {}


def _matrix_control(spec: Q3dSpec) -> MatrixRunControl:
    if spec.run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    return spec.run_control
